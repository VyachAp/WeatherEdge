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
    # d(residual)/dt fit over routines in the last 6h. Captures "forecast
    # falling further behind hour-over-hour" — invisible to the point-residual
    # field above. Used by `_project_daily_max` (v2 path) when we have at
    # least RESIDUAL_SLOPE_MIN_POINTS confirming routines; falls back to the
    # halflife decay otherwise. None when forecast or routines insufficient.
    forecast_residual_slope_f_per_hr: float | None = None
    forecast_residual_count: int = 0
    # Ensemble-derived σ in °F (std at peak hour across NWP models, already
    # converted from °C via ×1.8). None when fewer than ENSEMBLE_MIN_MODELS
    # returned data → probability engine falls back to hours-based σ.
    forecast_sigma_f: float | None = None
    ensemble_model_count: int = 1
    # Climatological prior for the trading day's daily max — multi-year
    # mean and std for this (station, day-of-year) from
    # ``ingestion.station_normals``. None when no normal exists for the
    # station or when CLIMATE_PRIOR_ENABLED is False; the probability
    # engine then degrades to its pre-prior baseline.
    climate_prior_mean_f: float | None = None
    climate_prior_std_f: float | None = None


@dataclass(frozen=True)
class CachedAggregationInputs:
    """Snapshot of build_state_from_metars inputs from the last successful
    aggregate_state call for a station.

    The fast-poll loop reuses these to rebuild state with a fresh METAR
    appended, without re-paying the forecast / bias / climate-normal HTTP
    cost. Forecast and bias drift slowly within the cache window, so this
    is safe inside ~30 minutes; older entries are rejected by
    ``get_cached_aggregation_inputs``.
    """

    cached_at_utc: datetime
    history: list[dict[str, Any]]
    forecast: Any | None
    bias_c: float
    climate_prior_mean_f: float | None
    climate_prior_std_f: float | None


_state_cache: dict[str, CachedAggregationInputs] = {}
_STATE_CACHE_MAX_AGE = timedelta(minutes=30)


def get_cached_aggregation_inputs(icao: str) -> CachedAggregationInputs | None:
    """Return cached inputs from the last successful aggregate_state for
    ``icao``, or None when absent or older than ``_STATE_CACHE_MAX_AGE``."""
    cached = _state_cache.get(icao)
    if cached is None:
        return None
    if datetime.now(timezone.utc) - cached.cached_at_utc > _STATE_CACHE_MAX_AGE:
        return None
    return cached


def clear_state_cache() -> None:
    """Drop all cached inputs (called at daily settlement)."""
    _state_cache.clear()


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


def _compute_residual_slope(
    history: list[dict[str, Any]],
    hourly_temps_c: list[float],
    bias_c: float,
    now_utc: datetime,
) -> tuple[float | None, int]:
    """Fit linear slope of (obs − same-hour forecast) °F over routines in
    the last 6h (the same window as the trend regression).

    Returns ``(slope_f_per_hr, n_points)``. ``slope`` is None when fewer
    than 2 routine METARs fall in the window with valid temps. The slope
    captures "forecast falling further behind hour-over-hour" — invisible
    to a single-point residual snapshot.
    """
    if not hourly_temps_c:
        return None, 0

    cutoff = now_utc - timedelta(hours=6)
    pairs: list[tuple[float, float]] = []
    for m in history:
        if m.get("is_speci"):
            continue
        obs_at = m.get("observed_at")
        temp_f = m.get("temp_f")
        if not isinstance(obs_at, datetime) or temp_f is None:
            continue
        if obs_at < cutoff or obs_at > now_utc:
            continue
        hour_idx = _closest_hour_index(obs_at, len(hourly_temps_c))
        forecast_f = _c_to_f(hourly_temps_c[hour_idx] + bias_c)
        residual = float(temp_f) - forecast_f
        hours_from_cutoff = (obs_at - cutoff).total_seconds() / 3600.0
        pairs.append((hours_from_cutoff, residual))

    if len(pairs) < 2:
        return None, len(pairs)

    n = len(pairs)
    sum_x = sum(x for x, _ in pairs)
    sum_y = sum(y for _, y in pairs)
    sum_xy = sum(x * y for x, y in pairs)
    sum_x2 = sum(x * x for x, _ in pairs)
    denom = n * sum_x2 - sum_x * sum_x
    if denom == 0:
        return 0.0, n
    slope = (n * sum_xy - sum_x * sum_y) / denom
    return slope, n


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
    climate_prior_mean_f: float | None = None,
    climate_prior_std_f: float | None = None,
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
    forecast_residual_slope_f_per_hr: float | None = None
    forecast_residual_count: int = 0
    forecast_sigma_f: float | None = None
    ensemble_model_count: int = 1
    if has_forecast:
        adjusted_peak_c = forecast.peak_temp_c + bias_c
        forecast_peak_f = _c_to_f(adjusted_peak_c)
        peak_dt = now_utc.replace(hour=forecast.peak_hour_utc, minute=0, second=0, microsecond=0)
        hours_until_peak = (peak_dt - now_utc).total_seconds() / 3600.0
        # Open-Meteo's `forecast_days=1, timezone=UTC` returns 24 hourly
        # slots covering [today_utc 00:00, today_utc 23:00]. For stations
        # west of UTC (Americas, Pacific) today's local-day heating peak
        # occurs late in the UTC day — often at hour 23 — and yesterday's
        # heating tail bleeds into hour 0 of the same UTC day. The
        # `argmax` then reports hour 0 (yesterday's tail), which gives
        # `hours_until_peak ≈ -17h` mid-day local. The peak we actually
        # care about is today's, which is the *next* occurrence of the
        # same hour-of-day. Shift by 24h whenever the same-day peak lies
        # more than 12h in the past.
        if hours_until_peak < -12.0:
            peak_dt += timedelta(days=1)
            hours_until_peak += 24.0
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

            slope, count = _compute_residual_slope(
                recent_history, hourly, bias_c, now_utc,
            )
            forecast_residual_slope_f_per_hr = slope
            forecast_residual_count = count

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
        forecast_residual_slope_f_per_hr=forecast_residual_slope_f_per_hr,
        forecast_residual_count=forecast_residual_count,
        forecast_sigma_f=forecast_sigma_f,
        ensemble_model_count=ensemble_model_count,
        climate_prior_mean_f=climate_prior_mean_f,
        climate_prior_std_f=climate_prior_std_f,
    )


async def aggregate_state(
    session: AsyncSession,
    icao: str,
    lat: float,
    lon: float,
) -> WeatherState | None:
    """Gather all data sources into a WeatherState for one station.

    Returns None if insufficient data (no routine METARs available).

    Performance: fetches METAR history, deterministic Open-Meteo, and
    ensemble Open-Meteo concurrently. METAR history is fetched once (24h)
    and reused for daily max, trend, and cycle detection — no redundant
    AWC calls.

    Forecast composition: deterministic single-source provides the central
    forecast (peak, hourly arrays) — same reference frame as
    ``record_daily_outcome``, so the stored bias_c applies cleanly. The
    ensemble contributes the spread-at-peak-hour for σ. When the ensemble
    fails, σ falls back to the hours-based schedule in the probability
    engine; when deterministic fails, ensemble is used alone (lossy bias
    frame, but better than no forecast).
    """
    from src.config import settings
    from src.ingestion.openmeteo import (
        fetch_deterministic_forecast,
        fetch_ensemble_forecast,
    )
    from src.ingestion.station_bias import get_bias
    from src.ingestion.station_normals import get_normal
    from src.signals.mapper import icao_timezone, today_local

    metar_task = _safe_fetch_metar(icao)
    deterministic_task = fetch_deterministic_forecast(lat, lon)
    ensemble_task = fetch_ensemble_forecast(lat, lon)
    history, deterministic, ensemble = await asyncio.gather(
        metar_task, deterministic_task, ensemble_task,
    )

    if history is None:
        return None

    forecast = _blend_forecasts(deterministic, ensemble)

    if forecast is not None:
        try:
            bias_c = await get_bias(session, icao)
        except Exception:
            logger.warning("Bias fetch failed for %s, using default", icao, exc_info=True)
            bias_c = 1.0
    else:
        bias_c = 0.0  # Not used when forecast is None

    # Climate-normal prior — read once per station per tick. Uses today's
    # station-local DOY (off-by-one DOY between adjacent days changes the
    # normal by <0.1°F, negligible vs the per-market target-day distinction).
    climate_prior_mean_f: float | None = None
    climate_prior_std_f: float | None = None
    if settings.CLIMATE_PRIOR_ENABLED:
        try:
            tz = icao_timezone(icao)
            normal = await get_normal(session, icao, today_local(tz))
        except Exception:
            logger.warning("Climate normal fetch failed for %s", icao, exc_info=True)
            normal = None
        if normal is not None:
            std_f = normal.std_max_c * 9.0 / 5.0
            if (
                settings.CLIMATE_PRIOR_MIN_SIGMA_F * 0.5  # accept tighter, the engine clamps
                <= std_f
                <= settings.CLIMATE_PRIOR_MAX_SIGMA_F
            ):
                climate_prior_mean_f = normal.mean_max_c * 9.0 / 5.0 + 32.0
                climate_prior_std_f = std_f
            else:
                logger.info(
                    "[%s] climate normal σ=%.2f°F outside [%.1f, %.1f] — prior bypassed",
                    icao, std_f,
                    settings.CLIMATE_PRIOR_MIN_SIGMA_F * 0.5,
                    settings.CLIMATE_PRIOR_MAX_SIGMA_F,
                )

    now_utc = datetime.now(timezone.utc)
    state = build_state_from_metars(
        icao, history, forecast, bias_c, now_utc,
        climate_prior_mean_f=climate_prior_mean_f,
        climate_prior_std_f=climate_prior_std_f,
    )
    if state is None:
        logger.debug("No routine METARs yet for %s, skipping", icao)
        return None

    # Cache inputs so the fast-poll loop can rebuild state from a fresh
    # routine METAR without re-fetching forecast / bias / climate normals.
    _state_cache[icao] = CachedAggregationInputs(
        cached_at_utc=now_utc,
        history=history,
        forecast=forecast,
        bias_c=bias_c,
        climate_prior_mean_f=climate_prior_mean_f,
        climate_prior_std_f=climate_prior_std_f,
    )

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


def _blend_forecasts(deterministic: Any, ensemble: Any) -> Any:
    """Combine deterministic central forecast with ensemble spread.

    Returns an ``OpenMeteoForecast`` whose temperatures, hourly arrays, and
    peak hour come from the deterministic single-source response, while
    ``peak_temp_std_c`` / ``hourly_temps_std_c`` / ``model_count`` come from
    the ensemble. ``peak_temp_std_c`` is read from the ensemble's hourly
    spread *at the deterministic peak hour* — disagreement at the time we
    expect the peak, which is what the σ estimate should reflect.

    Falls back to whichever forecast is available when one fetch failed.
    """
    from src.ingestion.openmeteo import OpenMeteoForecast

    if deterministic is None:
        return ensemble  # No central — ensemble alone (bias frame mismatch noted in caller).
    if ensemble is None or not ensemble.hourly_temps_std_c:
        return deterministic  # No spread — sigma will fall back to hours-based.

    peak_hour = deterministic.peak_hour_utc
    if 0 <= peak_hour < len(ensemble.hourly_temps_std_c):
        peak_std = ensemble.hourly_temps_std_c[peak_hour]
    else:
        peak_std = ensemble.peak_temp_std_c

    return OpenMeteoForecast(
        peak_temp_c=deterministic.peak_temp_c,
        peak_hour_utc=deterministic.peak_hour_utc,
        hourly_temps_c=deterministic.hourly_temps_c,
        hourly_cloud_cover=deterministic.hourly_cloud_cover,
        hourly_solar_radiation=deterministic.hourly_solar_radiation,
        hourly_dewpoint_c=deterministic.hourly_dewpoint_c,
        hourly_wind_speed=deterministic.hourly_wind_speed,
        hourly_temps_std_c=ensemble.hourly_temps_std_c,
        peak_temp_std_c=peak_std,
        model_count=ensemble.model_count,
    )
