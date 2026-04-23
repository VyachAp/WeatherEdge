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
    metar_trend_rate: float         # °F/hr, 6h routine-METAR linear regression
    dewpoint_trend_rate: float      # °F/hr dewpoint
    forecast_peak_f: float          # Open-Meteo peak + station bias
    hours_until_peak: float         # may be negative once forecast peak hour has passed
    solar_declining: bool
    solar_decline_magnitude: float
    cloud_rising: bool
    cloud_rise_magnitude: float
    routine_count_today: int
    metar_trend_rate_short: float = 0.0  # °F/hr regression over last 2.5h
    has_forecast: bool = False      # True iff forecast data was fetched successfully
    # Sorted (observed_at_utc, temp_f) tuples for routine METARs in the last 24h.
    # Used by lock_rules.evaluate_lock to compute per-market daily max anchored
    # to the market's target day — which may differ from the station's "today"
    # local day (e.g. a market asking about Apr 21 Dallas may close at 07:00
    # local Apr 21, and our snapshot may be at 23:00 local Apr 20).
    routine_history: tuple[tuple[datetime, float], ...] = ()
    # Forecast-trajectory residual fields. All four are bias-adjusted to the
    # same reference frame as forecast_peak_f so that residual=0 and slope
    # match imply projected == forecast_peak_f. None when no forecast data
    # or no routine METAR available.
    latest_obs_temp_f: float | None = None
    forecast_temp_now_f: float | None = None
    forecast_slope_to_peak_f_per_hr: float | None = None
    forecast_residual_f: float | None = None
    # Ensemble-derived σ in °F (std at peak hour across NWP models, already
    # converted from °C via ×1.8). None when fewer than ENSEMBLE_MIN_MODELS
    # returned data → probability engine falls back to hours-based σ.
    forecast_sigma_f: float | None = None
    ensemble_model_count: int = 1


# ---------------------------------------------------------------------------
# Internal helpers that work on a pre-fetched history list (no extra HTTP)
# ---------------------------------------------------------------------------


def _closest_hour_index(ts: datetime, n_hours: int) -> int:
    # Duplicated from forecast_exceedance._closest_hour_index to avoid a
    # state_aggregator → forecast_exceedance import cycle.
    idx = ts.hour + (1 if ts.minute >= 30 else 0)
    return max(0, min(idx, n_hours - 1))


def _latest_routine_temp_f(history: list[dict[str, Any]]) -> float | None:
    """Return temp_f of the newest non-SPECI METAR in ``history``, or None."""
    routine = [
        m for m in history
        if not m.get("is_speci")
        and m.get("temp_f") is not None
        and isinstance(m.get("observed_at"), datetime)
    ]
    if not routine:
        return None
    return float(max(routine, key=lambda m: m["observed_at"])["temp_f"])


def _routine_daily_max(
    history: list[dict[str, Any]],
    icao: str,
    now_utc: datetime | None = None,
) -> tuple[float | None, int]:
    """Compute (max_temp_f, routine_count) for the LOCAL-city day at ``icao``.

    Polymarket "highest temp on [date] in [city]" markets resolve on Wunderground's
    local-day aggregation, not the UTC day. Using station-local boundaries keeps
    the computed daily max aligned with the resolution window, especially at UTC
    midnight when the UTC day rolls over but the local day hasn't.
    """
    from src.signals.mapper import icao_timezone

    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    tz = icao_timezone(icao)
    now_local = now_utc.astimezone(tz)
    local_day_start = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    local_day_end = local_day_start + timedelta(days=1)
    utc_start = local_day_start.astimezone(timezone.utc)
    utc_end = local_day_end.astimezone(timezone.utc)

    routine_temps: list[float] = []
    for m in history:
        if m.get("is_speci"):
            continue
        obs_at = m.get("observed_at")
        if (
            obs_at is not None
            and isinstance(obs_at, datetime)
            and utc_start <= obs_at < utc_end
        ):
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


def build_state_from_metars(
    icao: str,
    history: list[dict[str, Any]],
    forecast: Any | None,
    bias_c: float,
    now_utc: datetime,
) -> WeatherState | None:
    """Assemble a WeatherState from already-fetched METAR history + forecast.

    Pure function — no HTTP, no DB. Used by `aggregate_state` (live) and by the
    backtest harness (replay). Returns None if the station has no routine METARs
    for the UTC day of ``now_utc``.
    """
    from src.ingestion.aviation import detect_metar_cycle
    from src.ingestion.openmeteo import (
        solar_declining as check_solar,
        cloud_rising as check_cloud,
    )

    current_max_f, routine_count = _routine_daily_max(history, icao=icao, now_utc=now_utc)
    if current_max_f is None:
        return None

    cutoff_6h = now_utc.replace(microsecond=0) - timedelta(hours=6)
    recent_history = [
        m for m in history
        if m.get("observed_at") is not None
        and isinstance(m["observed_at"], datetime)
        and cutoff_6h <= m["observed_at"] <= now_utc
    ]
    trend = _compute_trend(recent_history, routine_only=True)
    metar_trend_rate = trend["rate_of_change_per_hour"]
    dewpoint_trend_rate = trend["dewpoint_rate"]

    cutoff_short = now_utc.replace(microsecond=0) - timedelta(hours=2, minutes=30)
    short_history = [
        m for m in recent_history
        if m.get("observed_at") is not None
        and isinstance(m["observed_at"], datetime)
        and m["observed_at"] >= cutoff_short
    ]
    metar_trend_rate_short = _compute_trend(short_history, routine_only=True)[
        "rate_of_change_per_hour"
    ]

    has_forecast = forecast is not None
    latest_obs_temp_f: float | None = None
    forecast_temp_now_f: float | None = None
    forecast_slope_to_peak_f_per_hr: float | None = None
    forecast_residual_f: float | None = None
    forecast_sigma_f: float | None = None
    ensemble_model_count: int = 1
    if has_forecast:
        adjusted_peak_c = forecast.peak_temp_c + bias_c
        forecast_peak_f = _c_to_f(adjusted_peak_c)
        peak_dt = now_utc.replace(hour=forecast.peak_hour_utc, minute=0, second=0, microsecond=0)
        hours_until_peak = (peak_dt - now_utc).total_seconds() / 3600.0
        is_solar_declining, solar_mag = check_solar(forecast, now_utc.hour)
        is_cloud_rising, cloud_mag = check_cloud(forecast, now_utc.hour)

        # Residual fields — all in the same bias-adjusted °F frame as
        # forecast_peak_f so that residual=0 + matching slope ⇒
        # projected == forecast_peak_f.
        hourly = forecast.hourly_temps_c
        if hourly:
            hour_idx = _closest_hour_index(now_utc, len(hourly))
            forecast_temp_now_f = _c_to_f(hourly[hour_idx] + bias_c)
            forecast_slope_to_peak_f_per_hr = (
                (forecast_peak_f - forecast_temp_now_f) / hours_until_peak
                if hours_until_peak > 0 else 0.0
            )
            latest_obs_temp_f = _latest_routine_temp_f(recent_history)
            if latest_obs_temp_f is not None:
                forecast_residual_f = latest_obs_temp_f - forecast_temp_now_f

        # Ensemble spread → σ (°C × 1.8 = °F; peak_temp_std_c is 0 when the
        # deterministic fallback ran).
        peak_std_c = getattr(forecast, "peak_temp_std_c", 0.0) or 0.0
        ensemble_model_count = getattr(forecast, "model_count", 1) or 1
        if peak_std_c > 0:
            forecast_sigma_f = peak_std_c * 9.0 / 5.0
    else:
        forecast_peak_f = current_max_f
        hours_until_peak = 0.0
        is_solar_declining = False
        solar_mag = 0.0
        is_cloud_rising = False
        cloud_mag = 0.0

    cycle_minutes = detect_metar_cycle(recent_history)
    if cycle_minutes:
        logger.debug("%s METAR cycle: %s", icao, cycle_minutes)

    # Build compact routine history for per-market daily-max computation
    # downstream. Sorted ascending by observed_at.
    routine_points: list[tuple[datetime, float]] = []
    for m in history:
        if m.get("is_speci"):
            continue
        obs = m.get("observed_at")
        temp = m.get("temp_f")
        if (
            obs is not None and isinstance(obs, datetime)
            and temp is not None
        ):
            routine_points.append((obs, float(temp)))
    routine_points.sort(key=lambda p: p[0])

    return WeatherState(
        station_icao=icao,
        current_max_f=current_max_f,
        metar_trend_rate=metar_trend_rate,
        metar_trend_rate_short=metar_trend_rate_short,
        dewpoint_trend_rate=dewpoint_trend_rate,
        forecast_peak_f=forecast_peak_f,
        hours_until_peak=hours_until_peak,
        solar_declining=is_solar_declining,
        solar_decline_magnitude=solar_mag,
        cloud_rising=is_cloud_rising,
        cloud_rise_magnitude=cloud_mag,
        routine_count_today=routine_count,
        has_forecast=has_forecast,
        routine_history=tuple(routine_points),
        latest_obs_temp_f=latest_obs_temp_f,
        forecast_temp_now_f=forecast_temp_now_f,
        forecast_slope_to_peak_f_per_hr=forecast_slope_to_peak_f_per_hr,
        forecast_residual_f=forecast_residual_f,
        forecast_sigma_f=forecast_sigma_f,
        ensemble_model_count=ensemble_model_count,
    )


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
    from src.ingestion.openmeteo import fetch_forecast
    from src.ingestion.station_bias import get_bias

    metar_task = _safe_fetch_metar(icao)
    forecast_task = fetch_forecast(lat, lon)
    history, forecast = await asyncio.gather(metar_task, forecast_task)

    if history is None:
        return None

    if forecast is not None:
        try:
            bias_c = await get_bias(session, icao)
        except Exception:
            logger.warning("Bias fetch failed for %s, using default", icao, exc_info=True)
            bias_c = 1.0
    else:
        bias_c = 0.0  # Not used when forecast is None

    now_utc = datetime.now(timezone.utc)
    state = build_state_from_metars(icao, history, forecast, bias_c, now_utc)
    if state is None:
        logger.debug("No routine METARs yet for %s, skipping", icao)
        return None

    sigma_desc = (
        f"σ={state.forecast_sigma_f:.2f}°F (n={state.ensemble_model_count})"
        if state.forecast_sigma_f is not None
        else "σ=hours-based"
    )
    logger.info(
        "[%s] state: max=%.0f°F, trend=%+.1f°F/hr, forecast_peak=%.0f°F in %.1fh, "
        "%s, solar_declining=%s, cloud_rising=%s, routine_count=%d",
        icao, state.current_max_f, state.metar_trend_rate, state.forecast_peak_f,
        state.hours_until_peak, sigma_desc, state.solar_declining, state.cloud_rising,
        state.routine_count_today,
    )

    # Diagnostic alert: record same-hour exceedances and push Telegram when
    # the projected daily max is set to beat the forecast peak.
    try:
        from src.signals.forecast_exceedance import check_and_record_daily_max_alert
        await check_and_record_daily_max_alert(icao, state, history, forecast)
    except Exception:
        logger.warning("daily-max alert check failed for %s", icao, exc_info=True)

    return state


async def _safe_fetch_metar(icao: str) -> list[dict[str, Any]] | None:
    """Fetch 24h METAR history, returning None on failure."""
    from src.ingestion.aviation import fetch_metar_history
    try:
        return await fetch_metar_history(icao, hours=24)
    except Exception:
        logger.warning("METAR history fetch failed for %s", icao, exc_info=True)
        return None
