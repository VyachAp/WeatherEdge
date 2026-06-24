"""Pure daily-max projection math.

Split out of ``signals.forecast_exceedance`` so backtest / replay code
can import the math without dragging in DB writes or Telegram dispatch.

All functions take a ``WeatherState`` (and at most a routine-METAR
history list for ``_pick_latest_routine``) and return floats / booleans.
No I/O, no module-global side effects.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from src.config import settings
from src.signals.state_aggregator import WeatherState


# ---------------------------------------------------------------------------
# Module-level aliases bound from `settings`. Override via .env; see
# `src.config.Settings`. Restart required for changes (Pydantic loads
# once at boot).
# ---------------------------------------------------------------------------
EXTRAPOLATION_HOURS_CAP: float = settings.EXCEEDANCE_EXTRAPOLATION_HOURS_CAP
PEAK_TOLERANCE_F: float = settings.EXCEEDANCE_PEAK_TOLERANCE_F
EXTRAPOLATION_HALFLIFE_H: float = settings.EXCEEDANCE_EXTRAPOLATION_HALFLIFE_H
MAX_OVERSHOOT_F: float = settings.EXCEEDANCE_MAX_OVERSHOOT_F
DEWPOINT_NUDGE_F: float = settings.EXCEEDANCE_DEWPOINT_NUDGE_F
RESIDUAL_DECAY_HALFLIFE_H: float = settings.EXCEEDANCE_RESIDUAL_DECAY_HALFLIFE_H
RESIDUAL_TREND_CARRY_K: float = settings.EXCEEDANCE_RESIDUAL_TREND_CARRY_K
POST_PEAK_HOURS_CAP: float = settings.POST_PEAK_HOURS_CAP
POST_PEAK_TREND_CARRY_K: float = settings.POST_PEAK_TREND_CARRY_K
POST_PEAK_MIN_TREND_F_PER_HR: float = settings.POST_PEAK_MIN_TREND_F_PER_HR


def c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def pick_latest_routine(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    routine = [
        m for m in history
        if not m.get("is_speci")
        and m.get("temp_f") is not None
        and isinstance(m.get("observed_at"), datetime)
    ]
    if not routine:
        return None
    return max(routine, key=lambda m: m["observed_at"])


def closest_hour_index(observed_at: datetime, n_hours: int) -> int:
    idx = observed_at.hour + (1 if observed_at.minute >= 30 else 0)
    return max(0, min(idx, n_hours - 1))


def effective_trend(state: WeatherState) -> float:
    """Prefer the short-window (2.5h) trend; fall back to the 6h regression."""
    return state.metar_trend_rate_short or state.metar_trend_rate


def peak_passed(state: WeatherState) -> bool:
    """Heuristic for 'today's peak has almost certainly passed'.

    The 6h trend can stay positive well past the real peak on sharp-cooling
    afternoons, so the three cooling-branch checks use the short-window trend.
    The fourth branch catches the case where the forecast peak hour is at
    least an hour behind us and the short trend is essentially flat.
    """
    trend = effective_trend(state)
    if trend <= 0 and state.current_max_f >= state.forecast_peak_f - PEAK_TOLERANCE_F:
        return True
    if trend <= 0 and state.hours_until_peak <= 0:
        return True
    if trend <= 0 and state.solar_declining:
        return True
    if state.hours_until_peak <= -1.0 and state.metar_trend_rate_short <= 0.1:
        return True
    return False


def project_daily_max(state: WeatherState) -> float:
    """Project today's final daily max, anchored on the forecast trajectory.

    Pre-peak: exponential decay of the level residual (halflife
    ``RESIDUAL_DECAY_HALFLIFE_H``) plus a damped carry of the
    obs-slope-vs-forecast-forward-slope trend residual. Post-peak: damped
    carry of the observed trend off the observed/forecast max anchor. Returns
    ``forecast_peak_f`` exactly when obs track the forecast. Always clipped at
    ``forecast_peak + MAX_OVERSHOOT_F`` and floored at observed max. Falls back
    to ``legacy_project_daily_max`` when there's no current forecast.

    (The residual-slope "v2" projector was removed 2026-06-24 — it overshot the
    warming concavity at high-volume stations; see docs/graveyard.md.)
    """
    if state.forecast_temp_now_f is None or state.forecast_residual_f is None:
        return legacy_project_daily_max(state)
    residual = state.forecast_residual_f
    projected = state.forecast_peak_f

    if state.hours_until_peak > 0:
        alpha = math.exp(-state.hours_until_peak / RESIDUAL_DECAY_HALFLIFE_H)
        projected += alpha * residual

        observed_slope = effective_trend(state)
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
        observed_slope = effective_trend(state)
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


def legacy_project_daily_max(state: WeatherState) -> float:
    """Pre-residual linear-blend projector. Kept for the no-forecast-now path."""
    projected = state.current_max_f
    trend = effective_trend(state)
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
