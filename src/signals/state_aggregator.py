"""State aggregator — gathers METAR, Open-Meteo, and bias data per city.

Produces a WeatherState snapshot that feeds into the probability engine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


def _c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


@dataclass(frozen=True)
class WeatherState:
    """Aggregated weather state per city for one pipeline tick."""

    station_icao: str
    current_max_f: float
    metar_trend_rate: float         # °F/hr from last 3-4 routine METARs
    dewpoint_trend_rate: float      # °F/hr dewpoint
    forecast_peak_f: float          # Open-Meteo peak + station bias
    hours_until_peak: float
    solar_declining: bool
    solar_decline_magnitude: float
    cloud_rising: bool
    cloud_rise_magnitude: float
    routine_count_today: int


async def aggregate_state(
    session: AsyncSession,
    icao: str,
    lat: float,
    lon: float,
) -> WeatherState | None:
    """Gather all data sources into a WeatherState for one station.

    Returns None if insufficient data (no routine METARs available).
    """
    # Lazy imports to avoid circular dependencies
    from src.ingestion.aviation import (
        get_routine_daily_max,
        get_temp_trend,
        fetch_metar_history,
        detect_metar_cycle,
    )
    from src.ingestion.openmeteo import (
        fetch_forecast,
        solar_declining as check_solar,
        cloud_rising as check_cloud,
    )
    from src.ingestion.station_bias import get_bias

    # 1. Routine daily max + count
    current_max_f, routine_count = await get_routine_daily_max(icao)
    if current_max_f is None:
        logger.debug("No routine METARs yet for %s, skipping", icao)
        return None

    # 2. Temperature & dewpoint trend (routine only)
    trend = await get_temp_trend(icao, hours=6, routine_only=True)
    metar_trend_rate = trend.get("rate_of_change_per_hour", 0.0)
    dewpoint_trend_rate = trend.get("dewpoint_rate", 0.0)

    # 3. Open-Meteo forecast
    forecast = await fetch_forecast(lat, lon)

    now_utc = datetime.now(timezone.utc)
    current_hour = now_utc.hour

    if forecast is not None:
        # Station bias correction
        bias_c = await get_bias(session, icao)
        adjusted_peak_c = forecast.peak_temp_c + bias_c
        forecast_peak_f = _c_to_f(adjusted_peak_c)
        hours_until_peak = max(0.0, forecast.peak_hour_utc - current_hour)

        is_solar_declining, solar_mag = check_solar(forecast, current_hour)
        is_cloud_rising, cloud_mag = check_cloud(forecast, current_hour)
    else:
        # Fallback: use current max as forecast peak, conservative
        forecast_peak_f = current_max_f
        hours_until_peak = 0.0
        is_solar_declining = False
        solar_mag = 0.0
        is_cloud_rising = False
        cloud_mag = 0.0

    # 4. METAR cycle detection (informational, for logging)
    history = await fetch_metar_history(icao, hours=6)
    cycle_minutes = detect_metar_cycle(history)
    if cycle_minutes:
        logger.debug("%s METAR cycle: %s", icao, cycle_minutes)

    return WeatherState(
        station_icao=icao,
        current_max_f=current_max_f,
        metar_trend_rate=metar_trend_rate,
        dewpoint_trend_rate=dewpoint_trend_rate,
        forecast_peak_f=forecast_peak_f,
        hours_until_peak=hours_until_peak,
        solar_declining=is_solar_declining,
        solar_decline_magnitude=solar_mag,
        cloud_rising=is_cloud_rising,
        cloud_rise_magnitude=cloud_mag,
        routine_count_today=routine_count,
    )
