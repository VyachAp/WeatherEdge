"""State aggregator — gathers METAR, Open-Meteo, and bias data per city.

Produces a WeatherState snapshot that feeds into the probability engine.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from typing import Any, TYPE_CHECKING

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


# ---------------------------------------------------------------------------
# Internal helpers that work on a pre-fetched history list (no extra HTTP)
# ---------------------------------------------------------------------------


def _routine_daily_max(
    history: list[dict[str, Any]],
) -> tuple[float | None, int]:
    """Compute (max_temp_f, routine_count) from routine METARs for the current UTC day."""
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    routine_temps: list[float] = []
    for m in history:
        if m.get("is_speci"):
            continue
        obs_at = m.get("observed_at")
        if obs_at is not None and isinstance(obs_at, datetime) and obs_at >= today_start:
            temp_f = m.get("temp_f")
            if temp_f is not None:
                routine_temps.append(temp_f)
    if not routine_temps:
        return None, 0
    return max(routine_temps), len(routine_temps)


def _compute_trend(
    history: list[dict[str, Any]],
    routine_only: bool = True,
) -> dict[str, float]:
    """Compute temp and dewpoint trend from a pre-fetched history list.

    Returns dict with rate_of_change_per_hour and dewpoint_rate.
    """
    filtered = [m for m in history if not m.get("is_speci")] if routine_only else history

    temps = [
        (m["observed_at"], m["temp_f"])
        for m in filtered
        if m.get("temp_f") is not None and m.get("observed_at") is not None
    ]

    if len(temps) < 2:
        return {"rate_of_change_per_hour": 0.0, "dewpoint_rate": 0.0}

    temps.sort(key=lambda x: x[0])

    def _slope(pairs: list[tuple[datetime, float]]) -> float:
        t0 = pairs[0][0]
        x = [(p[0] - t0).total_seconds() / 3600.0 for p in pairs]
        y = [p[1] for p in pairs]
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        denom = n * sum_x2 - sum_x * sum_x
        if denom == 0:
            return 0.0
        return (n * sum_xy - sum_x * sum_y) / denom

    temp_rate = round(_slope(temps), 2)

    dewpoints = [
        (m["observed_at"], m["dewpoint_f"])
        for m in filtered
        if m.get("dewpoint_f") is not None and m.get("observed_at") is not None
    ]
    dewpoint_rate = round(_slope(dewpoints), 2) if len(dewpoints) >= 2 else 0.0

    return {"rate_of_change_per_hour": temp_rate, "dewpoint_rate": dewpoint_rate}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def aggregate_state(
    session: AsyncSession,
    icao: str,
    lat: float,
    lon: float,
) -> WeatherState | None:
    """Gather all data sources into a WeatherState for one station.

    Returns None if insufficient data (no routine METARs available).

    Performance: fetches METAR history and Open-Meteo forecast concurrently.
    METAR history is fetched once (24h) and reused for daily max, trend, and
    cycle detection — no redundant AWC calls.
    """
    from src.ingestion.aviation import fetch_metar_history, detect_metar_cycle
    from src.ingestion.openmeteo import (
        fetch_forecast,
        solar_declining as check_solar,
        cloud_rising as check_cloud,
    )
    from src.ingestion.station_bias import get_bias

    # Fetch METAR history and Open-Meteo forecast concurrently
    metar_task = _safe_fetch_metar(icao)
    forecast_task = fetch_forecast(lat, lon)

    history, forecast = await asyncio.gather(metar_task, forecast_task)

    # 1. Routine daily max + count (from pre-fetched history)
    if history is None:
        return None

    current_max_f, routine_count = _routine_daily_max(history)
    if current_max_f is None:
        logger.debug("No routine METARs yet for %s, skipping", icao)
        return None

    # 2. Temperature & dewpoint trend (from pre-fetched history, no HTTP)
    # Use only the last 6 hours of history for trend
    now_utc = datetime.now(timezone.utc)
    cutoff_6h = now_utc.replace(microsecond=0) - timedelta(hours=6)
    recent_history = [
        m for m in history
        if m.get("observed_at") is not None
        and isinstance(m["observed_at"], datetime)
        and m["observed_at"] >= cutoff_6h
    ]
    trend = _compute_trend(recent_history, routine_only=True)
    metar_trend_rate = trend["rate_of_change_per_hour"]
    dewpoint_trend_rate = trend["dewpoint_rate"]

    # 3. Open-Meteo forecast (already fetched concurrently)
    current_hour = now_utc.hour

    if forecast is not None:
        try:
            bias_c = await get_bias(session, icao)
        except Exception:
            logger.warning("Bias fetch failed for %s, using default", icao, exc_info=True)
            bias_c = 1.0
        adjusted_peak_c = forecast.peak_temp_c + bias_c
        forecast_peak_f = _c_to_f(adjusted_peak_c)
        peak_dt = now_utc.replace(hour=forecast.peak_hour_utc, minute=0, second=0, microsecond=0)
        hours_until_peak = max(0.0, (peak_dt - now_utc).total_seconds() / 3600.0)

        is_solar_declining, solar_mag = check_solar(forecast, current_hour)
        is_cloud_rising, cloud_mag = check_cloud(forecast, current_hour)
    else:
        forecast_peak_f = current_max_f
        hours_until_peak = 0.0
        is_solar_declining = False
        solar_mag = 0.0
        is_cloud_rising = False
        cloud_mag = 0.0

    # 4. METAR cycle detection (from pre-fetched history, no HTTP)
    cycle_minutes = detect_metar_cycle(recent_history)
    if cycle_minutes:
        logger.debug("%s METAR cycle: %s", icao, cycle_minutes)

    # 5. Diagnostic: alert when latest routine obs exceeds same-hour forecast.
    try:
        from src.signals.forecast_exceedance import check_and_alert_exceedance
        await check_and_alert_exceedance(icao, history, forecast)
    except Exception:
        logger.warning("exceedance alert check failed for %s", icao, exc_info=True)

    state = WeatherState(
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
    logger.info(
        "[%s] state: max=%.0f°F, trend=%+.1f°F/hr, forecast_peak=%.0f°F in %.1fh, "
        "solar_declining=%s, cloud_rising=%s, routine_count=%d",
        icao, current_max_f, metar_trend_rate, forecast_peak_f, hours_until_peak,
        is_solar_declining, is_cloud_rising, routine_count,
    )
    return state


async def _safe_fetch_metar(icao: str) -> list[dict[str, Any]] | None:
    """Fetch 24h METAR history, returning None on failure."""
    from src.ingestion.aviation import fetch_metar_history
    try:
        return await fetch_metar_history(icao, hours=24)
    except Exception:
        logger.warning("METAR history fetch failed for %s", icao, exc_info=True)
        return None
