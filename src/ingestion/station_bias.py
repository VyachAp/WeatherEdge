"""Station bias tracking — per-station rolling record of observed vs forecast.

Records the difference between observed daily max temperature and model
forecast peak for each station, used to correct systematic model offset.
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


async def is_bias_runaway(session: AsyncSession, icao: str) -> bool:
    """Return True if absolute station bias exceeds the configured maximum.

    Stations with runaway bias should be excluded from trading until reviewed.
    """
    bias = await get_bias(session, icao)
    if abs(bias) > settings.STATION_BIAS_MAX_C:
        logger.warning("Bias runaway for %s: %.2f°C (max %.1f)", icao, bias, settings.STATION_BIAS_MAX_C)
        return True
    return False
