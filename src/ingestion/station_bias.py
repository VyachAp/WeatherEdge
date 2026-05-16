"""Station bias tracking — per-station rolling record of observed vs forecast.

Records the difference between observed daily max temperature and model
forecast peak for each station, used to correct systematic model offset.
Also exposes per-station rolling forecast-error RMSE for the probability
engine's σ floor (the global ``ENSEMBLE_MIN_SIGMA_F=2.0`` simultaneously
over-pads low-variance stations like RJTT and under-pads high-variance
stations like KAUS — per-station RMSE matches the floor to actual error).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.config import settings
from src.db.models import StationBias

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Per-station forecast-error RMSE cache. Keyed by ICAO; each entry is
# ``(rmse_f, sample_days, computed_at)``. RMSE is derived from the
# existing ``station_biases`` rows (no extra schema needed) so it's
# cheap to refresh; the cache exists mostly to avoid one DB roundtrip
# per city per tick. TTL is intentionally short (1h) — daily_settlement
# at 22:00 UTC writes a new bias row that should be reflected
# immediately on the next tick after rollover.
_rmse_cache: dict[str, tuple[float | None, int, datetime]] = {}
_RMSE_TTL = timedelta(hours=1)


async def record_daily_outcome(
    session: AsyncSession,
    icao: str,
    date: datetime,
    observed_max_c: float,
    forecast_peak_c: float,
) -> None:
    """Record one day's bias observation. Upserts on (station_icao, date)."""
    bias = observed_max_c - forecast_peak_c

    stmt = pg_insert(StationBias).values(
        station_icao=icao,
        date=date,
        observed_max_c=observed_max_c,
        forecast_peak_c=forecast_peak_c,
        bias_c=bias,
        created_at=datetime.now(timezone.utc),
    ).on_conflict_do_update(
        constraint="uq_station_bias_day",
        set_={
            "observed_max_c": observed_max_c,
            "forecast_peak_c": forecast_peak_c,
            "bias_c": bias,
        },
    )
    await session.execute(stmt)
    logger.info("Recorded bias for %s on %s: %.2f°C", icao, date.date(), bias)


async def get_bias(session: AsyncSession, icao: str) -> float:
    """Return 30-day rolling mean bias (observed - forecast).

    Returns DEFAULT_STATION_BIAS_C (1.0) when no history exists.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=settings.STATION_BIAS_WINDOW_DAYS)
    stmt = select(func.avg(StationBias.bias_c)).where(
        StationBias.station_icao == icao,
        StationBias.date >= cutoff,
    )
    result = await session.execute(stmt)
    avg = result.scalar()

    if avg is None:
        return settings.DEFAULT_STATION_BIAS_C
    return float(avg)


async def get_station_rmse(
    session: AsyncSession, icao: str, *, days: int | None = None
) -> tuple[float | None, int]:
    """Return ``(rmse_f, sample_days)`` over the rolling window.

    RMSE is computed from existing ``station_biases.bias_c`` rows as
    ``sqrt(avg(bias_c²))`` over the last ``days`` days (default:
    ``settings.STATION_BIAS_WINDOW_DAYS``), converted from °C to °F.
    Result is cached in-process for ``_RMSE_TTL`` (1h) — the underlying
    bias rows are written by ``record_daily_outcome`` at most once per
    station per day, so a longer-than-necessary TTL would only delay
    pickup of new data without saving meaningful work.

    Returns ``(None, 0)`` when no bias rows exist for the station —
    caller should fall back to the global ``ENSEMBLE_MIN_SIGMA_F`` floor.
    """
    window = days if days is not None else settings.STATION_BIAS_WINDOW_DAYS

    cached = _rmse_cache.get(icao)
    now = datetime.now(timezone.utc)
    if cached is not None:
        rmse_f, sample_days, computed_at = cached
        if now - computed_at < _RMSE_TTL:
            return rmse_f, sample_days

    cutoff = now - timedelta(days=window)
    stmt = select(
        func.sqrt(func.avg(StationBias.bias_c * StationBias.bias_c)),
        func.count(StationBias.id),
    ).where(
        StationBias.station_icao == icao,
        StationBias.date >= cutoff,
    )
    result = await session.execute(stmt)
    rmse_c, sample_days = result.one()

    rmse_f: float | None
    if rmse_c is None:
        rmse_f = None
        sample_days = 0
    else:
        rmse_f = float(rmse_c) * 9.0 / 5.0
        sample_days = int(sample_days)

    _rmse_cache[icao] = (rmse_f, sample_days, now)
    return rmse_f, sample_days


def clear_station_rmse_cache() -> None:
    """Reset the in-process RMSE cache. Test-only — production code
    relies on the natural TTL expiry."""
    _rmse_cache.clear()


async def is_bias_runaway(session: AsyncSession, icao: str) -> bool:
    """Return True if absolute station bias exceeds the configured maximum.

    Stations with runaway bias should be excluded from trading until reviewed.
    """
    bias = await get_bias(session, icao)
    if abs(bias) > settings.STATION_BIAS_MAX_C:
        logger.warning("Bias runaway for %s: %.2f°C (max %.1f)", icao, bias, settings.STATION_BIAS_MAX_C)
        return True
    return False
