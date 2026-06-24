"""Telegram alert when today's projected daily max is set to beat the forecast peak.

The DB row is written for every new routine METAR that exceeds the same-hour
forecast (calibration history). The Telegram push is suppressed when the day's
peak has almost certainly passed, when we don't have enough routine METARs yet,
or when this station already got a push within ``ALERT_COOLDOWN``.

Peak-passed heuristic (any of):
- observed max is within 0.5°F of (bias-adjusted) forecast peak AND short trend ≤ 0
- Open-Meteo peak hour is past AND short trend ≤ 0
- solar_declining AND short trend ≤ 0
- forecast peak hour ≥ 1h behind us AND short trend essentially flat (≤ 0.1°F/hr)

Here "short trend" is the 2.5h linear regression on routine METAR temperatures
(``WeatherState.metar_trend_rate_short``) — a 6h regression stays positive well
past the real peak on sharp-cooling afternoons, so it's the wrong signal here.

Projected daily max:
- Blend of trend-extrapolation and forecast peak with weight ``α = exp(-h/2)``.
  Near the peak (small h) we trust the observed slope; far out (large h) we
  trust the forecast, because diurnal warming is concave and a morning rate of
  +2°C/hr does not sustain for 3h.
- Extrapolation is damped by close-hours solar decline and cloud rise, nudged
  down when dewpoint climbs fast, and capped at ``forecast_peak + 5°F``
  (≈ 2.8°C) to keep implausible overshoots out of the push channel.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from src.config import settings
from src.db.engine import async_session
from src.db.models import ForecastExceedanceAlert
# Tests patch `get_alerter` / `_escape_md2` on this module to assert that
# Telegram push behavior is gated correctly; the actual dispatch was
# removed in commit 50a9a13. Keeping the imports so the patches resolve.
from src.execution.alerter import _escape_md2, get_alerter  # noqa: F401
from src.ingestion.openmeteo import OpenMeteoForecast
from src.signals.mapper import f_to_c, unit_for_station
from src.signals.projected_market_lookup import lookup_projected_binary
from src.signals.projection import (
    c_to_f as _c_to_f,
    closest_hour_index as _closest_hour_index,
    effective_trend as _effective_trend,
    legacy_project_daily_max as _legacy_project_daily_max,
    peak_passed as _peak_passed,
    pick_latest_routine as _pick_latest_routine,
    project_daily_max as _project_daily_max,
)

if TYPE_CHECKING:
    from src.signals.state_aggregator import WeatherState

logger = logging.getLogger(__name__)

# Module-level aliases bound from `settings` — overridable via .env, see
# `src.config.Settings` for defaults. Process restart required for .env
# edits to take effect (Pydantic loads once at boot).
EXCEEDANCE_THRESHOLD_F: float = settings.EXCEEDANCE_THRESHOLD_F
DELTA_THRESHOLD_F: float = settings.EXCEEDANCE_DELTA_THRESHOLD_F
MIN_ROUTINE_COUNT_FOR_PUSH: int = settings.EXCEEDANCE_MIN_ROUTINE_COUNT
# Strong-residual fast path: when the latest obs is already running ≥1°F
# (=2× EXCEEDANCE_THRESHOLD_F) over its same-hour forecast, the underlying
# signal is unambiguous. Allow the push at routine #2 to shave 30–60 min
# off the morning lag instead of waiting for routine #3.
STRONG_RESIDUAL_DELTA_F: float = settings.EXCEEDANCE_STRONG_RESIDUAL_DELTA_F
STRONG_RESIDUAL_MIN_ROUTINES: int = settings.EXCEEDANCE_STRONG_RESIDUAL_MIN_ROUTINES
EXTRAPOLATION_HOURS_CAP: float = settings.EXCEEDANCE_EXTRAPOLATION_HOURS_CAP
PEAK_TOLERANCE_F: float = settings.EXCEEDANCE_PEAK_TOLERANCE_F
EXTRAPOLATION_HALFLIFE_H: float = settings.EXCEEDANCE_EXTRAPOLATION_HALFLIFE_H
MAX_OVERSHOOT_F: float = settings.EXCEEDANCE_MAX_OVERSHOOT_F
DEWPOINT_NUDGE_F: float = settings.EXCEEDANCE_DEWPOINT_NUDGE_F
ALERT_COOLDOWN: timedelta = timedelta(minutes=settings.EXCEEDANCE_ALERT_COOLDOWN_MINUTES)
# Residual-carry tunables — how much of the observed-vs-forecast gap to carry
# forward to the forecast peak. Halflife controls how quickly the level
# residual fades toward zero with distance to peak; K is the fraction of a
# positive trend residual carried linearly up to EXTRAPOLATION_HOURS_CAP.
RESIDUAL_DECAY_HALFLIFE_H: float = settings.EXCEEDANCE_RESIDUAL_DECAY_HALFLIFE_H
RESIDUAL_TREND_CARRY_K: float = settings.EXCEEDANCE_RESIDUAL_TREND_CARRY_K
# Post-peak trend carry — when `hours_until_peak <= 0` but obs is still rising,
# Open-Meteo's nominal peak was simply too early (common in hot arid cities like
# OPKC/Phoenix/Delhi). Extrapolate a bounded, solar/cloud-damped amount beyond
# the observed max. Hours cap is shorter than pre-peak because diurnal concavity
# is sharper after nominal peak. Shared with probability_engine via Settings.
POST_PEAK_HOURS_CAP: float = settings.POST_PEAK_HOURS_CAP
POST_PEAK_TREND_CARRY_K: float = settings.POST_PEAK_TREND_CARRY_K
POST_PEAK_MIN_TREND_F_PER_HR: float = settings.POST_PEAK_MIN_TREND_F_PER_HR


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

    min_routines = (
        STRONG_RESIDUAL_MIN_ROUTINES
        if same_hour_delta_f > STRONG_RESIDUAL_DELTA_F
        else MIN_ROUTINE_COUNT_FOR_PUSH
    )
    push = (
        not peak_passed
        and state.routine_count_today >= min_routines
        and projection_delta_f > DELTA_THRESHOLD_F
    )

    # Cooldown: at most one Telegram push per station per ALERT_COOLDOWN
    # window. The DB row is still written (alerted=False) so calibration
    # history and this query continue to reflect reality on the next tick.
    if push:
        async with async_session() as cooldown_session:
            recent = await cooldown_session.execute(
                select(ForecastExceedanceAlert.id)
                .where(
                    ForecastExceedanceAlert.station_icao == icao,
                    ForecastExceedanceAlert.alerted.is_(True),
                    ForecastExceedanceAlert.alerted_at
                    >= datetime.now(timezone.utc) - ALERT_COOLDOWN,
                )
                .limit(1)
            )
            if recent.scalar() is not None:
                push = False

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

    slope_desc = (
        f"slope=+{state.forecast_residual_slope_f_per_hr:+.2f}°F/hr"
        f" n={state.forecast_residual_count}"
        if state.forecast_residual_slope_f_per_hr is not None
        else "slope=n/a"
    )
    logger.info(
        "[%s] exceedance row: obs=%.1f°F @ %s vs forecast@%02dZ=%.1f°F "
        "(same_hour_delta=+%.1f°F) | max=%.1f°F, forecast_peak=%.1f°F, "
        "projected=%.1f°F, trend=%+.1f°F/hr, %s, peak_passed=%s, alerted=%s",
        icao, obs_temp_f, observed_at.isoformat(), hour_idx, forecast_temp_f,
        same_hour_delta_f, state.current_max_f, state.forecast_peak_f,
        projected_max_f, state.metar_trend_rate, slope_desc,
        peak_passed, push,
    )
