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

from src.db.engine import async_session
from src.db.models import ForecastExceedanceAlert
from src.execution.alerter import _escape_md2, get_alerter
from src.ingestion.openmeteo import OpenMeteoForecast
from src.signals.mapper import f_to_c, unit_for_station
from src.signals.projected_market_lookup import lookup_projected_binary

if TYPE_CHECKING:
    from src.signals.state_aggregator import WeatherState

logger = logging.getLogger(__name__)

EXCEEDANCE_THRESHOLD_F = 0.5  # same-hour delta still gates DB recording
DELTA_THRESHOLD_F = 1.0       # projected_max − forecast_peak to trigger push
MIN_ROUTINE_COUNT_FOR_PUSH = 3
EXTRAPOLATION_HOURS_CAP = 3.0
PEAK_TOLERANCE_F = 0.5        # current_max within this of forecast_peak = "reached"
EXTRAPOLATION_HALFLIFE_H = 2.0  # α = exp(-h/halflife); trust extrapolation near peak
MAX_OVERSHOOT_F = 5.0           # ≈ 2.8°C plausibility ceiling vs forecast_peak
DEWPOINT_NUDGE_F = 0.5
ALERT_COOLDOWN = timedelta(minutes=30)  # one Telegram push per station per 30 min
# Residual-carry tunables — how much of the observed-vs-forecast gap to carry
# forward to the forecast peak. Halflife controls how quickly the level
# residual fades toward zero with distance to peak; K is the fraction of a
# positive trend residual carried linearly up to EXTRAPOLATION_HOURS_CAP.
RESIDUAL_DECAY_HALFLIFE_H = 2.0
RESIDUAL_TREND_CARRY_K = 0.5
# Post-peak trend carry — when `hours_until_peak <= 0` but obs is still rising,
# Open-Meteo's nominal peak was simply too early (common in hot arid cities like
# OPKC/Phoenix/Delhi). Extrapolate a bounded, solar/cloud-damped amount beyond
# the observed max. Hours cap is shorter than pre-peak because diurnal concavity
# is sharper after nominal peak.
POST_PEAK_HOURS_CAP = 1.5
# K=0.75 post-peak is more aggressive than the pre-peak residual carry (K=0.5)
# because post-peak the raw trend is used directly instead of the observed-minus-
# forecast residual — and a positive post-peak trend is itself evidence that
# Open-Meteo's nominal peak hour was wrong, so the signal deserves higher weight.
POST_PEAK_TREND_CARRY_K = 0.75
POST_PEAK_MIN_TREND_F_PER_HR = 0.5


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


def _effective_trend(state: WeatherState) -> float:
    """Prefer the short-window (2.5h) trend; fall back to the 6h regression."""
    return state.metar_trend_rate_short or state.metar_trend_rate


def _peak_passed(state: WeatherState) -> bool:
    """Heuristic for 'today's peak has almost certainly passed'.

    The 6h trend can stay positive well past the real peak on sharp-cooling
    afternoons, so the three cooling-branch checks use the short-window trend.
    The fourth branch catches the case where the forecast peak hour is at
    least an hour behind us and the short trend is essentially flat.
    """
    trend = _effective_trend(state)
    if trend <= 0 and state.current_max_f >= state.forecast_peak_f - PEAK_TOLERANCE_F:
        return True
    if trend <= 0 and state.hours_until_peak <= 0:
        return True
    if trend <= 0 and state.solar_declining:
        return True
    if state.hours_until_peak <= -1.0 and state.metar_trend_rate_short <= 0.1:
        return True
    return False


def _project_daily_max(state: WeatherState) -> float:
    """Project today's final daily max, anchored on the forecast trajectory.

    Residual carry: start from ``forecast_peak_f`` and add the level residual
    (``latest_obs_f − forecast_now_f``) with exponential decay toward peak
    time, plus a damped trend residual (observed slope − forecast-implied
    slope). When observations track the forecast exactly, projection equals
    the forecast peak. When obs run hot, projection sits above the peak and
    is clipped at ``forecast_peak + MAX_OVERSHOOT_F``; the observed max is
    always a floor.

    Falls back to the legacy linear-blend behaviour when the residual fields
    are unavailable (e.g. forecast missing hourly_temps_c or no routine METAR
    yet in the 6h window).
    """
    if state.forecast_temp_now_f is None or state.forecast_residual_f is None:
        return _legacy_project_daily_max(state)

    projected = state.forecast_peak_f

    if state.hours_until_peak > 0:
        alpha = math.exp(-state.hours_until_peak / RESIDUAL_DECAY_HALFLIFE_H)
        projected += alpha * state.forecast_residual_f

        observed_slope = _effective_trend(state)
        forecast_slope = state.forecast_slope_to_peak_f_per_hr or 0.0
        residual_trend = observed_slope - forecast_slope
        if residual_trend > 0:
            hours = min(state.hours_until_peak, EXTRAPOLATION_HOURS_CAP)
            if state.solar_declining:
                hours *= max(0.0, 1.0 - state.solar_decline_magnitude)
            if state.cloud_rising:
                hours *= max(0.0, 1.0 - state.cloud_rise_magnitude)
            projected += RESIDUAL_TREND_CARRY_K * residual_trend * hours
    else:
        observed_slope = _effective_trend(state)
        if observed_slope > POST_PEAK_MIN_TREND_F_PER_HR:
            hours = POST_PEAK_HOURS_CAP
            if state.solar_declining:
                hours *= max(0.0, 1.0 - state.solar_decline_magnitude)
            if state.cloud_rising:
                hours *= max(0.0, 1.0 - state.cloud_rise_magnitude)
            anchor = max(state.current_max_f, state.forecast_peak_f)
            projected = anchor + POST_PEAK_TREND_CARRY_K * observed_slope * hours

    if state.dewpoint_trend_rate > 1.0:
        projected -= DEWPOINT_NUDGE_F
    projected = min(projected, state.forecast_peak_f + MAX_OVERSHOOT_F)
    return max(projected, state.current_max_f)


def _legacy_project_daily_max(state: WeatherState) -> float:
    """Pre-residual linear-blend projector. Kept for the no-forecast-now path."""
    projected = state.current_max_f
    trend = _effective_trend(state)
    if state.hours_until_peak > 0 and trend > 0:
        hours = min(state.hours_until_peak, EXTRAPOLATION_HOURS_CAP)
        if state.solar_declining:
            hours *= max(0.0, 1.0 - state.solar_decline_magnitude)
        if state.cloud_rising:
            hours *= max(0.0, 1.0 - state.cloud_rise_magnitude)
        extrapolated = state.current_max_f + trend * hours
        alpha = math.exp(-state.hours_until_peak / EXTRAPOLATION_HALFLIFE_H)
        projected = alpha * extrapolated + (1.0 - alpha) * state.forecast_peak_f
    elif state.hours_until_peak <= 0 and trend > POST_PEAK_MIN_TREND_F_PER_HR:
        hours = POST_PEAK_HOURS_CAP
        if state.solar_declining:
            hours *= max(0.0, 1.0 - state.solar_decline_magnitude)
        if state.cloud_rising:
            hours *= max(0.0, 1.0 - state.cloud_rise_magnitude)
        anchor = max(state.current_max_f, state.forecast_peak_f)
        projected = anchor + POST_PEAK_TREND_CARRY_K * trend * hours
    if state.dewpoint_trend_rate > 1.0:
        projected -= DEWPOINT_NUDGE_F
    projected = min(projected, state.forecast_peak_f + MAX_OVERSHOOT_F)
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
    unit = unit_for_station(icao)
    # Display the same trend the projection actually used (short-window when
    # available, 6h regression as fallback) — see _effective_trend.
    effective_trend_f_per_hr = _effective_trend(state)
    if unit == "°C":
        obs_max = f_to_c(state.current_max_f)
        trend = effective_trend_f_per_hr * 5.0 / 9.0
        forecast_peak = f_to_c(state.forecast_peak_f)
        projected = f_to_c(projected_max_f)
        proj_delta = projection_delta_f * 5.0 / 9.0
    else:
        obs_max = state.current_max_f
        trend = effective_trend_f_per_hr
        forecast_peak = state.forecast_peak_f
        projected = projected_max_f
        proj_delta = projection_delta_f

    text = (
        f"\U0001f321 *Daily max set to beat forecast* `{e(icao)}`\n"
        f"Obs max: {e(f'{obs_max:.1f}')}{unit} "
        f"\\(trend {e(f'{trend:+.1f}')}{unit}/hr, "
        f"{e(state.routine_count_today)} routine METARs\\)\n"
        f"Forecast peak: {e(f'{forecast_peak:.1f}')}{unit} "
        f"in {e(f'{state.hours_until_peak:.1f}')}h\n"
        f"Projected: {e(f'{projected:.1f}')}{unit} "
        f"\\({e(f'{proj_delta:+.1f}')}{unit} vs forecast\\)"
    )

    pm_line = await _format_polymarket_line(icao, projected_max_f, unit)
    if pm_line:
        text += f"\n{pm_line}"

    try:
        await get_alerter()._enqueue(text)
    except Exception:
        logger.warning("daily-max alert enqueue failed for %s", icao, exc_info=True)


_OP_SYMBOL = {
    "above": "≥", "at_least": "≥", "exceed": "≥",
    "below": "≤", "at_most": "≤",
}


async def _format_polymarket_line(
    icao: str, projected_max_f: float, unit: str,
) -> str | None:
    """Return a MarkdownV2-escaped 'Polymarket: ...' line, or None."""
    result = await lookup_projected_binary(icao, projected_max_f)
    if result is None:
        return None
    _, threshold_f, operator, yes_price = result

    threshold_disp = f_to_c(threshold_f) if unit == "°C" else threshold_f
    op_symbol = _OP_SYMBOL.get(operator, operator)
    e = _escape_md2
    return (
        f"Polymarket: {e(op_symbol)}{e(f'{threshold_disp:.0f}')}{unit} today "
        f"@ {e(f'{yes_price:.2f}')} \\(closest to projected\\)"
    )
