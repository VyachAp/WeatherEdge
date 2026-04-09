"""ECMWF open-data ensemble forecast ingestion.

Downloads 51 ensemble members (1 control + 50 perturbed) from the ECMWF
open-data service, extracts point values using bilinear interpolation, and
computes ensemble statistics and exceedance probabilities.

Interface mirrors :mod:`src.ingestion.gfs` so both models can be used
interchangeably downstream.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from ecmwf.opendata import Client
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import settings
from src.db.engine import async_session
from src.db.models import Forecast

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_DIR = Path(settings.ECMWF_CACHE_DIR)
N_MEMBERS = 51  # 1 control + 50 perturbed
RUN_HOURS = [0, 12]
DATA_AVAILABILITY_DELAY_H = 8  # ECMWF open data can lag 6-8h
FORECAST_STEPS = [24, 48, 72, 96, 120, 144, 168, 192, 240]
ECMWF_PARAMS = ["2t", "tp", "10u", "10v"]


@dataclass(frozen=True)
class VariableDef:
    """Describes how to fetch a forecast variable from ECMWF GRIB files."""

    search: str | None  # ECMWF shortName (None for derived variables)
    unit: str
    components: list[str] = field(default_factory=list)


VARIABLE_MAP: dict[str, VariableDef] = {
    "temperature": VariableDef(search="2t", unit="K"),
    "precipitation": VariableDef(search="tp", unit="kg/m^2"),
    "wind_speed": VariableDef(
        search=None,
        unit="m/s",
        components=["10u", "10v"],
    ),
}

# Concurrency controls — lower than GFS to respect ECMWF rate limits
_executor = ThreadPoolExecutor(max_workers=4)
_download_semaphore = asyncio.Semaphore(4)

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_path(run_date: date, run_hour: int, step: int, fc_type: str) -> Path:
    """Deterministic local path for an ECMWF GRIB file.

    Args:
        fc_type: ``"cf"`` (control) or ``"pf"`` (perturbed).
    """
    return (
        CACHE_DIR
        / f"{run_date:%Y%m%d}"
        / f"{run_hour:02d}z"
        / f"{fc_type}_step{step:03d}.grib2"
    )


def _is_cache_valid(path: Path) -> bool:
    """Return True if *path* exists and is younger than the configured TTL."""
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < settings.ECMWF_CACHE_TTL_HOURS * 3600


def cleanup_stale_cache() -> int:
    """Delete GRIB files older than TTL.  Returns count of files removed."""
    if not CACHE_DIR.exists():
        return 0
    removed = 0
    cutoff = time.time() - settings.ECMWF_CACHE_TTL_HOURS * 3600
    for p in CACHE_DIR.rglob("*"):
        if p.is_file() and p.stat().st_mtime < cutoff:
            p.unlink()
            removed += 1
    # prune empty dirs
    for d in sorted(CACHE_DIR.rglob("*"), reverse=True):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    return removed


# ---------------------------------------------------------------------------
# Run resolution
# ---------------------------------------------------------------------------


def _resolve_run_for_valid_time(
    valid_time: datetime,
) -> tuple[date, int, int]:
    """Find the best (run_date, run_hour, step) for a given *valid_time*.

    Walks backward through run cycles, accounting for data-availability delay,
    and returns the first match whose forecast step is in FORECAST_STEPS.
    """
    now = datetime.utcnow()
    max_step = max(FORECAST_STEPS)

    for hours_back in range(0, max_step + 24, 12):
        candidate_run = valid_time - timedelta(hours=hours_back)
        # Snap to nearest run hour
        eligible = [h for h in RUN_HOURS if h <= candidate_run.hour]
        if not eligible:
            continue
        rh = max(eligible)
        run_dt = candidate_run.replace(hour=rh, minute=0, second=0, microsecond=0)

        # Check data availability
        if run_dt + timedelta(hours=DATA_AVAILABILITY_DELAY_H) > now:
            continue

        step_exact = (valid_time - run_dt).total_seconds() / 3600
        if step_exact < 0:
            continue

        # Snap to nearest supported step (tolerance ±12h for sparse ECMWF steps)
        step = min(FORECAST_STEPS, key=lambda s: abs(s - step_exact))
        if abs(step - step_exact) <= 12:
            return run_dt.date(), rh, step

    # Fallback: use the most recent available run with the closest step
    fallback_run = now - timedelta(hours=DATA_AVAILABILITY_DELAY_H)
    eligible = [h for h in RUN_HOURS if h <= fallback_run.hour]
    rh = max(eligible) if eligible else RUN_HOURS[-1]
    run_dt = fallback_run.replace(hour=rh, minute=0, second=0, microsecond=0)
    step_exact = (valid_time - run_dt).total_seconds() / 3600
    step = min(FORECAST_STEPS, key=lambda s: abs(s - step_exact))
    return run_dt.date(), rh, step


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type((OSError, ConnectionError, ValueError)),
    reraise=True,
)
def _download_step_sync(
    run_date: date,
    run_hour: int,
    step: int,
    fc_type: str,
) -> Path | None:
    """Download a single ECMWF GRIB file (synchronous).

    Args:
        fc_type: ``"cf"`` (control forecast) or ``"pf"`` (perturbed forecast).

    Returns path to the local file, or ``None`` on non-retryable failure.
    """
    cache = _cache_path(run_date, run_hour, step, fc_type)
    if _is_cache_valid(cache):
        return cache

    cache.parent.mkdir(parents=True, exist_ok=True)
    try:
        client = Client()
        client.download(
            date=int(f"{run_date:%Y%m%d}"),
            time=run_hour,
            step=step,
            type=fc_type,
            param=ECMWF_PARAMS,
            target=str(cache),
        )
        return cache
    except Exception:
        logger.warning(
            "ECMWF download failed: %s step=%d type=%s",
            run_date,
            step,
            fc_type,
            exc_info=True,
        )
        raise


async def _async_download_step(
    run_date: date,
    run_hour: int,
    step: int,
    fc_type: str,
) -> Path | None:
    """Async wrapper around :func:`_download_step_sync`."""
    async with _download_semaphore:
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(
                _executor,
                _download_step_sync,
                run_date,
                run_hour,
                step,
                fc_type,
            )
        except Exception:
            logger.warning(
                "Async ECMWF download failed: step=%d type=%s",
                step,
                fc_type,
                exc_info=True,
            )
            return None


# ---------------------------------------------------------------------------
# Public: fetch_ecmwf_run
# ---------------------------------------------------------------------------


async def fetch_ecmwf_run(
    run_date: date,
    run_hour: int = 0,
) -> dict[tuple[int, str], Path]:
    """Download ECMWF GRIB files for all steps (cf + pf).

    Returns ``{(step, fc_type): Path}`` for successful downloads.
    """
    if run_hour not in RUN_HOURS:
        raise ValueError(f"run_hour must be one of {RUN_HOURS}, got {run_hour}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[int, str, asyncio.Task]] = []
    for step in FORECAST_STEPS:
        for fc_type in ("cf", "pf"):
            coro = _async_download_step(run_date, run_hour, step, fc_type)
            tasks.append((step, fc_type, asyncio.ensure_future(coro)))

    results: dict[tuple[int, str], Path] = {}
    outcomes = await asyncio.gather(*(t[2] for t in tasks), return_exceptions=True)
    for (step, fc_type, _), outcome in zip(tasks, outcomes):
        if isinstance(outcome, Exception):
            logger.warning(
                "Step %d type %s failed: %s", step, fc_type, outcome
            )
        elif outcome is not None:
            results[(step, fc_type)] = outcome

    total = len(tasks)
    ok = len(results)
    logger.info(
        "fetch_ecmwf_run %s/%02dz: %d/%d downloads succeeded",
        run_date,
        run_hour,
        ok,
        total,
    )
    return results


# ---------------------------------------------------------------------------
# Point extraction (bilinear interpolation)
# ---------------------------------------------------------------------------


def _normalize_lon(lon: float) -> float:
    """Convert longitude to -180..180 range used by ECMWF grids."""
    return ((lon + 180) % 360) - 180


def _extract_members_at_point(
    grib_path: Path,
    lat: float,
    lon: float,
    shortname: str,
) -> list[float]:
    """Extract ensemble member values at (lat, lon) using bilinear interpolation.

    For ``pf`` files the ``"number"`` dimension holds all 50 perturbed members.
    For ``cf`` files a single value is returned as a one-element list.
    """
    if not grib_path.exists():
        return []
    try:
        ds = xr.open_dataset(
            str(grib_path),
            engine="cfgrib",
            backend_kwargs={"filter_by_keys": {"shortName": shortname}},
        )
        data_vars = list(ds.data_vars)
        if not data_vars:
            logger.warning("No data variables for shortName=%s in %s", shortname, grib_path)
            ds.close()
            return []
        var_name = data_vars[0]
        da = ds[var_name]

        lon_n = _normalize_lon(lon)
        lat_dim = "latitude" if "latitude" in da.dims else "lat"
        lon_dim = "longitude" if "longitude" in da.dims else "lon"

        # Bilinear interpolation for better accuracy at ECMWF resolution
        point = da.interp({lat_dim: lat, lon_dim: lon_n}, method="linear")

        if "number" in point.dims:
            # pf file: multiple ensemble members
            values = [float(v) for v in point.values if not np.isnan(v)]
        else:
            # cf file: single control member
            val = float(point.values)
            values = [val] if not np.isnan(val) else []

        ds.close()
        return values
    except Exception:
        logger.warning(
            "Failed to extract members from %s (shortName=%s)",
            grib_path,
            shortname,
            exc_info=True,
        )
        return []


def _extract_wind_speed_members(
    grib_path: Path,
    lat: float,
    lon: float,
) -> list[float]:
    """Compute wind speed magnitude from 10u and 10v ensemble members."""
    u_vals = _extract_members_at_point(grib_path, lat, lon, "10u")
    v_vals = _extract_members_at_point(grib_path, lat, lon, "10v")
    n = min(len(u_vals), len(v_vals))
    return [math.sqrt(u**2 + v**2) for u, v in zip(u_vals[:n], v_vals[:n])]


# ---------------------------------------------------------------------------
# Public: get_ensemble_stats
# ---------------------------------------------------------------------------


async def get_ensemble_stats(
    lat: float,
    lon: float,
    valid_date: datetime,
    variable: str,
    session: AsyncSession | None = None,
) -> dict[str, Any]:
    """Extract ensemble statistics for a single point / time / variable.

    Returns dict with keys: mean, std, min, max, q25, q75, iqr, members,
    n_members, run_time, valid_time.
    """
    if variable not in VARIABLE_MAP:
        raise ValueError(f"Unknown variable '{variable}'. Choose from {list(VARIABLE_MAP)}")

    vdef = VARIABLE_MAP[variable]
    run_date, run_hour, step = _resolve_run_for_valid_time(valid_date)
    run_time = datetime(run_date.year, run_date.month, run_date.day, run_hour)
    valid_time = run_time + timedelta(hours=step)

    loop = asyncio.get_running_loop()
    values: list[float] = []

    cf_path = _cache_path(run_date, run_hour, step, "cf")
    pf_path = _cache_path(run_date, run_hour, step, "pf")

    if vdef.search:
        # Direct variable (temperature, precipitation)
        cf_vals = await loop.run_in_executor(
            _executor, _extract_members_at_point, cf_path, lat, lon, vdef.search
        )
        pf_vals = await loop.run_in_executor(
            _executor, _extract_members_at_point, pf_path, lat, lon, vdef.search
        )
        values = cf_vals + pf_vals
    elif vdef.components:
        # Derived variable (wind_speed)
        cf_vals = await loop.run_in_executor(
            _executor, _extract_wind_speed_members, cf_path, lat, lon
        )
        pf_vals = await loop.run_in_executor(
            _executor, _extract_wind_speed_members, pf_path, lat, lon
        )
        values = cf_vals + pf_vals

    if len(values) < 10:
        logger.warning(
            "Only %d/%d ensemble members available for %s at (%.2f, %.2f) valid %s",
            len(values),
            N_MEMBERS,
            variable,
            lat,
            lon,
            valid_date,
        )

    if not values:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "q25": None,
            "q75": None,
            "iqr": None,
            "members": [],
            "n_members": 0,
            "run_time": run_time,
            "valid_time": valid_time,
        }

    arr = np.array(values)
    q25, q75 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    stats: dict[str, Any] = {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q25": q25,
        "q75": q75,
        "iqr": q75 - q25,
        "members": [float(v) for v in values],
        "n_members": len(values),
        "run_time": run_time,
        "valid_time": valid_time,
    }

    # Persist to DB
    own_session = session is None
    if own_session:
        session = async_session()
    try:
        await _persist_forecast(session, run_time, valid_time, lat, lon, variable, stats)
    except Exception:
        logger.warning("Failed to persist ECMWF forecast", exc_info=True)
    finally:
        if own_session:
            await session.close()

    return stats


# ---------------------------------------------------------------------------
# Public: get_probability
# ---------------------------------------------------------------------------


async def get_probability(
    lat: float,
    lon: float,
    valid_date: datetime,
    variable: str,
    threshold: float,
    operator: str = "above",
    session: AsyncSession | None = None,
) -> float:
    """Compute P(variable > threshold) or P(variable < threshold) from ensemble.

    Args:
        operator: ``"above"`` or ``"below"``.

    Returns probability in [0, 1], or 0.0 if no members are available.
    """
    if operator not in ("above", "below"):
        raise ValueError(f"operator must be 'above' or 'below', got '{operator}'")

    stats = await get_ensemble_stats(lat, lon, valid_date, variable, session=session)
    members = stats["members"]
    if not members:
        return 0.0

    if operator == "above":
        count = sum(1 for v in members if v > threshold)
    else:
        count = sum(1 for v in members if v < threshold)

    return count / len(members)


# ---------------------------------------------------------------------------
# DB persistence
# ---------------------------------------------------------------------------


async def _persist_forecast(
    session: AsyncSession,
    run_time: datetime,
    valid_time: datetime,
    lat: float,
    lon: float,
    variable: str,
    stats: dict[str, Any],
) -> None:
    """Insert a Forecast row with model_source='ecmwf'."""
    forecast = Forecast(
        model_source="ecmwf",
        run_time=run_time,
        valid_time=valid_time,
        location_lat=round(lat, 4),
        location_lon=round(lon, 4),
        variable=variable,
        ensemble_mean=stats["mean"],
        ensemble_std=stats["std"],
        ensemble_members=stats["members"],
        fetched_at=datetime.utcnow(),
    )
    session.add(forecast)
    await session.commit()


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


async def ingest_latest_ecmwf(
    locations: list[tuple[float, float]],
    variables: list[str] | None = None,
) -> int:
    """Fetch the latest ECMWF run and extract ensemble stats for locations.

    Returns count of forecast rows persisted, or 0 if ECMWF data is
    unavailable (allowing the pipeline to continue with GFS alone).
    """
    variables = variables or list(VARIABLE_MAP.keys())
    now = datetime.utcnow()
    avail = now - timedelta(hours=DATA_AVAILABILITY_DELAY_H)

    eligible = [h for h in RUN_HOURS if h <= avail.hour]
    if not eligible:
        # Too early in the day for any ECMWF run to be available
        logger.info("No ECMWF run available yet today, skipping")
        return 0
    run_hour = max(eligible)
    run_date = avail.date()

    try:
        await fetch_ecmwf_run(run_date, run_hour)
    except Exception:
        logger.error(
            "Failed to fetch ECMWF run %s/%02dz, skipping ECMWF ingestion",
            run_date,
            run_hour,
            exc_info=True,
        )
        return 0

    count = 0
    sess = async_session()
    try:
        run_time = datetime(run_date.year, run_date.month, run_date.day, run_hour)
        for step in FORECAST_STEPS:
            valid_time = run_time + timedelta(hours=step)
            for lat, lon in locations:
                for var in variables:
                    try:
                        await get_ensemble_stats(lat, lon, valid_time, var, session=sess)
                        count += 1
                    except Exception:
                        logger.warning(
                            "Failed to extract %s at (%.2f, %.2f) step=%d",
                            var,
                            lat,
                            lon,
                            step,
                            exc_info=True,
                        )
    except Exception:
        logger.error("ECMWF ingestion failed", exc_info=True)
    finally:
        await sess.close()

    logger.info("ingest_latest_ecmwf: persisted %d forecasts", count)
    return count
