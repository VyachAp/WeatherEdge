"""GEFS (GFS Ensemble) forecast ingestion via Herbie.

Downloads all 31 ensemble members (control + 30 perturbation) from NCEP GEFS,
extracts point values using nearest-neighbor interpolation, and computes
ensemble statistics and exceedance probabilities.
"""

from __future__ import annotations

import asyncio
import logging
import math
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from herbie import Herbie
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import settings
from src.db.engine import async_session
from src.db.models import Forecast

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_DIR = Path(settings.GEFS_CACHE_DIR)
N_MEMBERS = 31  # member 0 (control) + members 1-30
FORECAST_HOURS = [6, 12, 18, 24, 36, 48, 72, 96, 120, 144, 168]
RUN_HOURS = [0, 6, 12, 18]
DATA_AVAILABILITY_DELAY_H = 5  # GEFS data typically available ~5h after run


@dataclass(frozen=True)
class VariableDef:
    """Describes how to fetch a forecast variable from GRIB files."""

    search: str | None  # GRIB search string (None for derived variables)
    unit: str
    components: list[str] = field(default_factory=list)  # for derived vars like wind_speed


VARIABLE_MAP: dict[str, VariableDef] = {
    "temperature": VariableDef(search="TMP:2 m above ground", unit="K"),
    "precipitation": VariableDef(search="APCP:surface", unit="kg/m^2"),
    "wind_speed": VariableDef(
        search=None,
        unit="m/s",
        components=["UGRD:10 m above ground", "VGRD:10 m above ground"],
    ),
}

# Thread pool for sync Herbie calls
_executor = ThreadPoolExecutor(max_workers=4)
_download_semaphore = asyncio.Semaphore(8)

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_path(run_date: date, run_hour: int, fxx: int, member: int) -> Path:
    """Deterministic local path for a single GRIB file."""
    return (
        CACHE_DIR
        / f"{run_date:%Y%m%d}"
        / f"{run_hour:02d}z"
        / f"mem{member:02d}"
        / f"gefs.t{run_hour:02d}z.pgrb2a.0p50.f{fxx:03d}"
    )


def _is_cache_valid(path: Path) -> bool:
    """Return True if *path* exists and is younger than the configured TTL."""
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < settings.GEFS_CACHE_TTL_HOURS * 3600


def cleanup_stale_cache() -> int:
    """Delete GRIB files older than TTL.  Returns count of files removed."""
    if not CACHE_DIR.exists():
        return 0
    removed = 0
    cutoff = time.time() - settings.GEFS_CACHE_TTL_HOURS * 3600
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
    """Find the best (run_date, run_hour, fxx) for a given *valid_time*.

    Walks backward through run cycles, accounting for data-availability delay,
    and returns the first match whose forecast hour is in FORECAST_HOURS.
    """
    now = datetime.utcnow()
    max_fxx = max(FORECAST_HOURS)

    for hours_back in range(0, max_fxx + 24, 6):
        candidate_run = valid_time - timedelta(hours=hours_back)
        # Snap to nearest run hour
        rh = max(h for h in RUN_HOURS if h <= candidate_run.hour)
        run_dt = candidate_run.replace(hour=rh, minute=0, second=0, microsecond=0)

        # Check data availability
        if run_dt + timedelta(hours=DATA_AVAILABILITY_DELAY_H) > now:
            continue

        fxx_exact = (valid_time - run_dt).total_seconds() / 3600
        if fxx_exact < 0:
            continue

        # Snap to nearest supported forecast hour
        fxx = min(FORECAST_HOURS, key=lambda h: abs(h - fxx_exact))
        if abs(fxx - fxx_exact) <= 3:  # tolerance: ±3 h
            return run_dt.date(), rh, fxx

    # Fallback: use the most recent available run with the closest fxx
    fallback_run = now - timedelta(hours=DATA_AVAILABILITY_DELAY_H)
    rh = max(h for h in RUN_HOURS if h <= fallback_run.hour)
    run_dt = fallback_run.replace(hour=rh, minute=0, second=0, microsecond=0)
    fxx_exact = (valid_time - run_dt).total_seconds() / 3600
    fxx = min(FORECAST_HOURS, key=lambda h: abs(h - fxx_exact))
    return run_dt.date(), rh, fxx


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type((OSError, ConnectionError, ValueError)),
    reraise=True,
)
def _download_member_sync(
    run_date: date,
    run_hour: int,
    fxx: int,
    member: int,
    search: str,
) -> Path | None:
    """Download a single GRIB message via Herbie (synchronous).

    Returns path to the local file, or ``None`` on non-retryable failure.
    """
    cache = _cache_path(run_date, run_hour, fxx, member)
    if _is_cache_valid(cache):
        return cache

    run_dt = datetime(run_date.year, run_date.month, run_date.day, run_hour)
    try:
        H = Herbie(
            date=run_dt,
            model="gefs",
            product="atmos.5",
            member=member,
            fxx=fxx,
            save_dir=str(CACHE_DIR),
            verbose=False,
        )
        local = H.download(search)
        if local is not None:
            dest = cache
            dest.parent.mkdir(parents=True, exist_ok=True)
            src_path = Path(str(local))
            if src_path.exists() and src_path != dest:
                shutil.move(str(src_path), str(dest))
            return dest
    except Exception:
        logger.warning(
            "Download failed: member=%d fxx=%d search=%s",
            member,
            fxx,
            search,
            exc_info=True,
        )
        raise
    return None


async def _async_download_member(
    run_date: date,
    run_hour: int,
    fxx: int,
    member: int,
    search: str,
) -> Path | None:
    """Async wrapper around :func:`_download_member_sync`."""
    async with _download_semaphore:
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(
                _executor,
                _download_member_sync,
                run_date,
                run_hour,
                fxx,
                member,
                search,
            )
        except Exception:
            logger.warning(
                "Async download failed: member=%d fxx=%d", member, fxx, exc_info=True
            )
            return None


# ---------------------------------------------------------------------------
# Public: fetch_gefs_run
# ---------------------------------------------------------------------------


async def fetch_gefs_run(
    run_date: date,
    run_hour: int = 0,
) -> dict[tuple[int, int, str], Path]:
    """Download GEFS GRIB files for all members / fxx / variables.

    Returns ``{(fxx, member, search_string): Path}`` for successful downloads.
    """
    if run_hour not in RUN_HOURS:
        raise ValueError(f"run_hour must be one of {RUN_HOURS}, got {run_hour}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Build search strings for all variables
    all_searches: list[str] = []
    for vdef in VARIABLE_MAP.values():
        if vdef.search:
            all_searches.append(vdef.search)
        all_searches.extend(vdef.components)

    unique_searches = list(dict.fromkeys(all_searches))  # dedupe, preserve order

    tasks: list[tuple[int, int, str, asyncio.Task]] = []
    for fxx in FORECAST_HOURS:
        for member in range(N_MEMBERS):
            for search in unique_searches:
                coro = _async_download_member(run_date, run_hour, fxx, member, search)
                tasks.append((fxx, member, search, asyncio.ensure_future(coro)))

    results: dict[tuple[int, int, str], Path] = {}
    outcomes = await asyncio.gather(
        *(t[3] for t in tasks), return_exceptions=True
    )
    for (fxx, member, search, _), outcome in zip(tasks, outcomes):
        if isinstance(outcome, Exception):
            logger.warning(
                "Member %d fxx %d search '%s' failed: %s", member, fxx, search, outcome
            )
        elif outcome is not None:
            results[(fxx, member, search)] = outcome

    total = len(tasks)
    ok = len(results)
    logger.info(
        "fetch_gefs_run %s/%02dz: %d/%d downloads succeeded", run_date, run_hour, ok, total
    )
    return results


# ---------------------------------------------------------------------------
# Point extraction
# ---------------------------------------------------------------------------


def _normalize_lon(lon: float) -> float:
    """Convert longitude to 0-360 range used by GFS grids."""
    return lon % 360


def _extract_point_value(
    grib_path: Path,
    lat: float,
    lon: float,
) -> float | None:
    """Open a GRIB file and return the nearest-neighbor value at (lat, lon)."""
    try:
        ds = xr.open_dataset(str(grib_path), engine="cfgrib")
        # Identify the data variable (usually the only non-coord variable)
        data_vars = list(ds.data_vars)
        if not data_vars:
            logger.warning("No data variables in %s", grib_path)
            return None
        var_name = data_vars[0]
        da = ds[var_name]

        lon360 = _normalize_lon(lon)

        # Determine coordinate names (latitude/longitude or lat/lon)
        lat_dim = "latitude" if "latitude" in da.dims else "lat"
        lon_dim = "longitude" if "longitude" in da.dims else "lon"

        val = float(da.sel({lat_dim: lat, lon_dim: lon360}, method="nearest").values)
        ds.close()
        return val
    except Exception:
        logger.warning("Failed to extract point from %s", grib_path, exc_info=True)
        return None


def _extract_wind_speed(
    u_path: Path,
    v_path: Path,
    lat: float,
    lon: float,
) -> float | None:
    """Compute wind speed magnitude from U and V component GRIB files."""
    u = _extract_point_value(u_path, lat, lon)
    v = _extract_point_value(v_path, lat, lon)
    if u is not None and v is not None:
        return math.sqrt(u**2 + v**2)
    return None


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
    run_date, run_hour, fxx = _resolve_run_for_valid_time(valid_date)
    run_time = datetime(run_date.year, run_date.month, run_date.day, run_hour)
    valid_time = run_time + timedelta(hours=fxx)

    loop = asyncio.get_running_loop()
    values: list[float] = []

    for member in range(N_MEMBERS):
        if vdef.search:
            # Direct variable (temperature, precipitation)
            path = _cache_path(run_date, run_hour, fxx, member)
            val = await loop.run_in_executor(
                _executor, _extract_point_value, path, lat, lon
            )
            if val is not None:
                values.append(val)
        elif vdef.components:
            # Derived variable (wind_speed)
            u_path = _cache_path(run_date, run_hour, fxx, member)
            v_path = _cache_path(run_date, run_hour, fxx, member)
            val = await loop.run_in_executor(
                _executor, _extract_wind_speed, u_path, v_path, lat, lon
            )
            if val is not None:
                values.append(val)

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
        logger.warning("Failed to persist forecast", exc_info=True)
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
    """Insert a Forecast row."""
    forecast = Forecast(
        model_source="gefs",
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


async def ingest_latest_gefs(
    locations: list[tuple[float, float]],
    variables: list[str] | None = None,
) -> int:
    """Fetch the latest GEFS run and extract ensemble stats for locations.

    Returns count of forecast rows persisted.
    """
    variables = variables or list(VARIABLE_MAP.keys())
    now = datetime.utcnow()
    avail = now - timedelta(hours=DATA_AVAILABILITY_DELAY_H)
    run_hour = max(h for h in RUN_HOURS if h <= avail.hour)
    run_date = avail.date()

    await fetch_gefs_run(run_date, run_hour)

    count = 0
    sess = async_session()
    try:
        run_time = datetime(run_date.year, run_date.month, run_date.day, run_hour)
        for fxx in FORECAST_HOURS:
            valid_time = run_time + timedelta(hours=fxx)
            for lat, lon in locations:
                for var in variables:
                    try:
                        await get_ensemble_stats(lat, lon, valid_time, var, session=sess)
                        count += 1
                    except Exception:
                        logger.warning(
                            "Failed to extract %s at (%.2f, %.2f) fxx=%d",
                            var,
                            lat,
                            lon,
                            fxx,
                            exc_info=True,
                        )
    finally:
        await sess.close()

    logger.info("ingest_latest_gefs: persisted %d forecasts", count)
    return count
