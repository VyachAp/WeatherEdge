"""Open-Meteo forecast client for hourly weather data.

Fetches hourly temperature, dewpoint, cloud cover, solar radiation, and wind
from the Open-Meteo API. No API key required.

Default mode is multi-model ensemble (ECMWF, GFS, ICON, GEM):
the median across models drives the central forecast, and the standard
deviation at peak hour (``peak_temp_std_c``) quantifies synoptic
uncertainty. Median (not mean) so a single regional model running outside
its valid domain — historically MeteoFrance ARPEGE returning ~-8°C of bias
in the Middle East / South Asia — can't drag the central forecast off.
Falls back to the deterministic single-source endpoint when fewer than
``ENSEMBLE_MIN_MODELS`` return usable peak-hour data.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field

import httpx

from src.config import settings

logger = logging.getLogger(__name__)

OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_VARS = [
    "temperature_2m",
    "dewpoint_2m",
    "cloudcover",
    "shortwave_radiation",
    "windspeed_10m",
]
HOURLY_PARAMS = ",".join(HOURLY_VARS)


@dataclass(frozen=True)
class OpenMeteoForecast:
    """Parsed hourly forecast for a single day.

    When ``model_count > 1`` the hourly arrays carry the cross-model median
    and ``peak_temp_std_c`` / ``hourly_temps_std_c`` expose the spread. When
    the deterministic fallback ran, ``model_count == 1`` and std fields are
    0 / [].
    """

    peak_temp_c: float
    peak_hour_utc: int
    hourly_temps_c: list[float]
    hourly_cloud_cover: list[int]
    hourly_solar_radiation: list[float]
    hourly_dewpoint_c: list[float]
    hourly_wind_speed: list[float]
    hourly_temps_std_c: list[float] = field(default_factory=list)
    peak_temp_std_c: float = 0.0
    model_count: int = 1


def _c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


async def fetch_forecast(lat: float, lon: float) -> OpenMeteoForecast | None:
    """Fetch today's hourly forecast from Open-Meteo.

    Tries the multi-model ensemble first; falls back to the deterministic
    endpoint when fewer than ``settings.ENSEMBLE_MIN_MODELS`` models returned
    data (or on ensemble-fetch failure). Returns None only if both paths fail.
    """
    ensemble = await fetch_ensemble_forecast(lat, lon)
    if ensemble is not None:
        return ensemble
    return await fetch_deterministic_forecast(lat, lon)


async def fetch_ensemble_forecast(lat: float, lon: float) -> OpenMeteoForecast | None:
    """Multi-model ensemble forecast (median across configured models).

    Returns None if fewer than ``ENSEMBLE_MIN_MODELS`` models came back with
    data, leaving the caller to decide on a fallback. The pipeline uses this
    purely as a spread source — ``peak_temp_std_c`` / ``hourly_temps_std_c``
    feed the probability engine's σ; central tendency comes from the
    deterministic forecast so the bias-correction frame stays stable.
    """
    models = [m.strip() for m in settings.ENSEMBLE_MODELS.split(",") if m.strip()]
    if not models:
        return None

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": HOURLY_PARAMS,
        "forecast_days": 1,
        "timezone": "UTC",
        "models": ",".join(models),
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(OPENMETEO_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        logger.warning(
            "Open-Meteo ensemble fetch failed for (%.2f, %.2f), falling back to deterministic",
            lat, lon, exc_info=True,
        )
        return None

    try:
        return _parse_ensemble_response(data, models)
    except Exception:
        logger.warning(
            "Open-Meteo ensemble parse failed for (%.2f, %.2f), falling back to deterministic",
            lat, lon, exc_info=True,
        )
        return None


def _parse_ensemble_response(
    data: dict,
    models: list[str],
) -> OpenMeteoForecast | None:
    """Parse a multi-model Open-Meteo response into an OpenMeteoForecast.

    Returns None if fewer than ``settings.ENSEMBLE_MIN_MODELS`` usable models
    appear in the response, so the caller can trigger the deterministic
    fallback.
    """
    hourly = data.get("hourly") or {}
    if not hourly:
        return None

    # Per-model temp arrays. Models may come back with nulls or missing
    # columns (not all models cover all lat/lon).
    per_model_temps: dict[str, list[float | None]] = {}
    for m in models:
        col = hourly.get(f"temperature_2m_{m}")
        if col is None:
            continue
        parsed = [float(v) if v is not None else None for v in col]
        if any(v is not None for v in parsed):
            per_model_temps[m] = parsed

    min_models = max(1, settings.ENSEMBLE_MIN_MODELS)
    if len(per_model_temps) < min_models:
        logger.info(
            "Open-Meteo ensemble returned %d usable models (<%d); falling back",
            len(per_model_temps), min_models,
        )
        return None

    # Determine hourly length from the first model; skip hours where no model
    # reports (shouldn't happen for 24h UTC day, but defensive).
    n_hours = max(len(v) for v in per_model_temps.values())
    median_temps: list[float] = []
    std_temps: list[float] = []
    for h in range(n_hours):
        vals = [
            temps[h]
            for temps in per_model_temps.values()
            if h < len(temps) and temps[h] is not None
        ]
        if not vals:
            # Hole in data; carry forward last median or use 0 — but flag.
            median_temps.append(median_temps[-1] if median_temps else 0.0)
            std_temps.append(0.0)
            continue
        median_temps.append(statistics.median(vals))
        std_temps.append(statistics.pstdev(vals) if len(vals) > 1 else 0.0)

    peak_temp = max(median_temps)
    peak_hour = median_temps.index(peak_temp)
    # Std at peak hour across models that have data at that exact hour.
    peak_hour_vals = [
        temps[peak_hour]
        for temps in per_model_temps.values()
        if peak_hour < len(temps) and temps[peak_hour] is not None
    ]
    peak_std = (
        statistics.pstdev(peak_hour_vals) if len(peak_hour_vals) > 1 else 0.0
    )

    # Hourly medians for auxiliary variables — one per model column, null-safe.
    def _median_across_models(var: str, cast: type) -> list:
        series: list[list[float | None]] = []
        for m in per_model_temps.keys():
            col = hourly.get(f"{var}_{m}")
            if col is None:
                continue
            series.append(
                [float(v) if v is not None else None for v in col]
            )
        if not series:
            return []
        out: list = []
        length = max(len(s) for s in series)
        for h in range(length):
            vals = [s[h] for s in series if h < len(s) and s[h] is not None]
            if not vals:
                out.append(cast(0))
                continue
            out.append(cast(statistics.median(vals)))
        return out

    dewpoints = _median_across_models("dewpoint_2m", float)
    clouds = _median_across_models("cloudcover", int)
    solar = _median_across_models("shortwave_radiation", float)
    winds = _median_across_models("windspeed_10m", float)

    logger.info(
        "Open-Meteo ensemble: %d models, median peak=%.1f°C ± %.2f°C at hour %d",
        len(per_model_temps), peak_temp, peak_std, peak_hour,
    )

    return OpenMeteoForecast(
        peak_temp_c=peak_temp,
        peak_hour_utc=peak_hour,
        hourly_temps_c=median_temps,
        hourly_cloud_cover=clouds,
        hourly_solar_radiation=solar,
        hourly_dewpoint_c=dewpoints,
        hourly_wind_speed=winds,
        hourly_temps_std_c=std_temps,  # std around median (peak hour included)
        peak_temp_std_c=peak_std,
        model_count=len(per_model_temps),
    )


async def fetch_deterministic_forecast(lat: float, lon: float) -> OpenMeteoForecast | None:
    """Single-source forecast (no ``models=`` param).

    Used both as the in-process fallback for ``fetch_forecast`` when the
    ensemble can't muster ``ENSEMBLE_MIN_MODELS`` and as the bias-recording
    reference at settlement time — pinning bias measurement to a stable
    forecast definition that doesn't shift when the ensemble model list or
    aggregation function changes.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": HOURLY_PARAMS,
        "forecast_days": 1,
        "timezone": "UTC",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(OPENMETEO_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        logger.warning("Open-Meteo fetch failed for (%.2f, %.2f)", lat, lon, exc_info=True)
        return None

    try:
        hourly = data["hourly"]
        temps = [float(t) for t in hourly["temperature_2m"]]
        clouds = [int(c) for c in hourly["cloudcover"]]
        solar = [float(s) for s in hourly["shortwave_radiation"]]
        dewpoints = [float(d) for d in hourly["dewpoint_2m"]]
        winds = [float(w) for w in hourly["windspeed_10m"]]

        peak_temp = max(temps)
        peak_hour = temps.index(peak_temp)

        return OpenMeteoForecast(
            peak_temp_c=peak_temp,
            peak_hour_utc=peak_hour,
            hourly_temps_c=temps,
            hourly_cloud_cover=clouds,
            hourly_solar_radiation=solar,
            hourly_dewpoint_c=dewpoints,
            hourly_wind_speed=winds,
            hourly_temps_std_c=[],
            peak_temp_std_c=0.0,
            model_count=1,
        )
    except (KeyError, ValueError, TypeError):
        logger.warning("Open-Meteo parse failed for (%.2f, %.2f)", lat, lon, exc_info=True)
        return None


def solar_declining(
    forecast: OpenMeteoForecast,
    current_hour_utc: int,
    window_hours: int = 2,
) -> tuple[bool, float]:
    """Check if solar radiation is declining in the next window.

    Returns (is_declining, magnitude) where magnitude is the fractional drop
    (0.0 = no drop, 1.0 = drops to zero).
    """
    idx = max(0, min(current_hour_utc, len(forecast.hourly_solar_radiation) - 1))
    current_solar = forecast.hourly_solar_radiation[idx]

    if current_solar <= 0:
        return False, 0.0

    end_idx = min(idx + window_hours, len(forecast.hourly_solar_radiation) - 1)
    future_solar = forecast.hourly_solar_radiation[end_idx]

    drop = (current_solar - future_solar) / current_solar
    return drop > 0.5, max(0.0, drop)


def cloud_rising(
    forecast: OpenMeteoForecast,
    current_hour_utc: int,
    window_hours: int = 2,
) -> tuple[bool, float]:
    """Check if cloud cover is rising in the next window.

    Returns (is_rising, magnitude) where magnitude is the cloud cover
    increase as a fraction of 100%.
    """
    idx = max(0, min(current_hour_utc, len(forecast.hourly_cloud_cover) - 1))
    current_cloud = forecast.hourly_cloud_cover[idx]

    end_idx = min(idx + window_hours, len(forecast.hourly_cloud_cover) - 1)
    future_cloud = forecast.hourly_cloud_cover[end_idx]

    rise = (future_cloud - current_cloud) / 100.0
    is_rising = future_cloud > 70 and rise > 0.1
    return is_rising, max(0.0, rise)


def dewpoint_trend(
    forecast: OpenMeteoForecast,
    current_hour_utc: int,
    window_hours: int = 3,
) -> float:
    """Return dewpoint trend in C/hr over the next window.

    Positive = rising dewpoint (moisture increasing, reduces upside).
    """
    idx = max(0, min(current_hour_utc, len(forecast.hourly_dewpoint_c) - 1))
    end_idx = min(idx + window_hours, len(forecast.hourly_dewpoint_c) - 1)

    if end_idx <= idx:
        return 0.0

    delta = forecast.hourly_dewpoint_c[end_idx] - forecast.hourly_dewpoint_c[idx]
    hours = end_idx - idx
    return delta / hours
