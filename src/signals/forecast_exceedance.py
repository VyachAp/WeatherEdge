"""Telegram alert when today's projected daily max is set to beat the forecast peak.

Fires at most once per new routine METAR per station, gated to the window where
the signal still has trading value. The DB row is written for every new routine
METAR that exceeds the same-hour forecast (calibration history), but the
Telegram push is suppressed once the day's peak has almost certainly passed or
there isn't enough data to trust the projection.

Peak-passed heuristic (any of):
- observed max is within 0.5°F of (bias-adjusted) forecast peak AND METAR trend ≤ 0
- Open-Meteo peak hour is past AND METAR trend ≤ 0
- solar_declining AND METAR trend ≤ 0

Projected daily max:
- observed max plus trend-based upward extrapolation for up to 3h before the
  forecast peak, damped by solar decline and cloud rise magnitudes, and nudged
  down slightly when dewpoint is rising fast.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from src.db.engine import async_session
from src.db.models import ForecastExceedanceAlert
from src.execution.alerter import _escape_md2, get_alerter
from src.ingestion.openmeteo import OpenMeteoForecast

if TYPE_CHECKING:
    from src.signals.state_aggregator import WeatherState

logger = logging.getLogger(__name__)

EXCEEDANCE_THRESHOLD_F = 0.5  # same-hour delta still gates DB recording
DELTA_THRESHOLD_F = 1.0       # projected_max − forecast_peak to trigger push
MIN_ROUTINE_COUNT_FOR_PUSH = 3
EXTRAPOLATION_HOURS_CAP = 3.0
PEAK_TOLERANCE_F = 0.5        # current_max within this of forecast_peak = "reached"


def _c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def _pick_latest_routine(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    routine = [
        m for m in history
        if not m.get("is_speci")
        and m.get("temp_f") is not None
        and isinstance(m.get("observed_at"), datetime)
    ]
    if not routine:
        return None
    return max(routine, key=lambda m: m["observed_at"])


def _closest_hour_index(observed_at: datetime, n_hours: int) -> int:
    idx = observed_at.hour + (1 if observed_at.minute >= 30 else 0)
    return max(0, min(idx, n_hours - 1))


def _peak_passed(state: WeatherState) -> bool:
    """Heuristic for 'today's peak has almost certainly passed'.

    Uses only WeatherState fields. Any one of the three branches triggers.
    """
    if state.metar_trend_rate <= 0 and (
        state.current_max_f >= state.forecast_peak_f - PEAK_TOLERANCE_F
    ):
        return True
    if state.metar_trend_rate <= 0 and state.hours_until_peak <= 0:
        return True
    if state.metar_trend_rate <= 0 and state.solar_declining:
        return True
    return False


def _project_daily_max(state: WeatherState) -> float:
    """Project today's final daily max using METAR trend + close-hours forecast.

    Linear extrapolation from the current observed max, capped at 3h, damped by
    close-hours solar decline and cloud rise, nudged down if dewpoint is
    climbing fast (moisture absorbing heat). Never returns below observed max.
    """
    projected = state.current_max_f
    if state.hours_until_peak > 0 and state.metar_trend_rate > 0:
        hours = min(state.hours_until_peak, EXTRAPOLATION_HOURS_CAP)
        if state.solar_declining:
            hours *= max(0.0, 1.0 - state.solar_decline_magnitude)
        if state.cloud_rising:
            hours *= max(0.0, 1.0 - state.cloud_rise_magnitude)
        projected += state.metar_trend_rate * hours
    if state.dewpoint_trend_rate > 1.0:
        projected -= 0.5
    return max(projected, state.current_max_f)


async def check_and_record_daily_max_alert(
    icao: str,
    state: WeatherState,
    history: list[dict[str, Any]],
    forecast: OpenMeteoForecast | None,
) -> None:
    """Record a forecast-exceedance row and, when appropriate, push a Telegram alert.

    No-op when there's no forecast or no routine METAR. Otherwise always records
    a DB row for calibration. Telegram push fires only when peak hasn't passed,
    we have enough routine METARs, and the projected daily max materially
    exceeds the bias-adjusted forecast peak.
    """
    if forecast is None or not forecast.hourly_temps_c:
        return

    latest = _pick_latest_routine(history)
    if latest is None:
        return

    observed_at: datetime = latest["observed_at"]
    obs_temp_f: float = float(latest["temp_f"])

    hour_idx = _closest_hour_index(observed_at, len(forecast.hourly_temps_c))
    forecast_temp_f = _c_to_f(forecast.hourly_temps_c[hour_idx])
    same_hour_delta_f = obs_temp_f - forecast_temp_f

    # Same-hour threshold still gates recording — below it, there's nothing
    # interesting to log. Keep the historical semantics of this table.
    if same_hour_delta_f <= EXCEEDANCE_THRESHOLD_F:
        return

    peak_passed = _peak_passed(state)
    projected_max_f = _project_daily_max(state)
    projection_delta_f = projected_max_f - state.forecast_peak_f

    push = (
        not peak_passed
        and state.routine_count_today >= MIN_ROUTINE_COUNT_FOR_PUSH
        and projection_delta_f > DELTA_THRESHOLD_F
    )

    async with async_session() as session:
        existing = await session.execute(
            select(ForecastExceedanceAlert.id).where(
                ForecastExceedanceAlert.station_icao == icao,
                ForecastExceedanceAlert.observed_at == observed_at,
            )
        )
        if existing.scalar() is not None:
            return

        session.add(ForecastExceedanceAlert(
            station_icao=icao,
            observed_at=observed_at,
            observed_temp_f=obs_temp_f,
            forecast_temp_f=forecast_temp_f,
            delta_f=same_hour_delta_f,
            forecast_hour_utc=hour_idx,
            current_max_f=state.current_max_f,
            forecast_peak_f=state.forecast_peak_f,
            projected_max_f=projected_max_f,
            metar_trend_rate=state.metar_trend_rate,
            peak_passed=peak_passed,
            alerted=push,
        ))
        try:
            await session.commit()
        except IntegrityError:
            await session.rollback()
            return

    logger.info(
        "[%s] exceedance row: obs=%.1f°F @ %s vs forecast@%02dZ=%.1f°F "
        "(same_hour_delta=+%.1f°F) | max=%.1f°F, forecast_peak=%.1f°F, "
        "projected=%.1f°F, trend=%+.1f°F/hr, peak_passed=%s, alerted=%s",
        icao, obs_temp_f, observed_at.isoformat(), hour_idx, forecast_temp_f,
        same_hour_delta_f, state.current_max_f, state.forecast_peak_f,
        projected_max_f, state.metar_trend_rate, peak_passed, push,
    )

    if not push:
        return

    e = _escape_md2
    text = (
        f"\U0001f321 *Daily max set to beat forecast* `{e(icao)}`\n"
        f"Obs max: {e(f'{state.current_max_f:.1f}')}°F "
        f"\\(trend {e(f'{state.metar_trend_rate:+.1f}')}°F/hr, "
        f"{e(state.routine_count_today)} routine METARs\\)\n"
        f"Forecast peak: {e(f'{state.forecast_peak_f:.1f}')}°F "
        f"in {e(f'{state.hours_until_peak:.1f}')}h\n"
        f"Projected: {e(f'{projected_max_f:.1f}')}°F "
        f"\\({e(f'{projection_delta_f:+.1f}')}°F vs forecast\\)"
    )
    try:
        await get_alerter()._enqueue(text)
    except Exception:
        logger.warning("daily-max alert enqueue failed for %s", icao, exc_info=True)
