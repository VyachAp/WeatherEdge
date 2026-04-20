"""Open-Meteo forecast client for hourly weather data.

Fetches hourly temperature, dewpoint, cloud cover, solar radiation, and wind
from the Open-Meteo API. No API key required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_PARAMS = (
    "temperature_2m,dewpoint_2m,cloudcover,"
    "shortwave_radiation,windspeed_10m"
)


@dataclass(frozen=True)
class OpenMeteoForecast:
    """Parsed hourly forecast for a single day."""

    peak_temp_c: float
    peak_hour_utc: int
    hourly_temps_c: list[float]
    hourly_cloud_cover: list[int]
    hourly_solar_radiation: list[float]
    hourly_dewpoint_c: list[float]
    hourly_wind_speed: list[float]


def _c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


async def fetch_forecast(lat: float, lon: float) -> OpenMeteoForecast | None:
    """Fetch today's hourly forecast from Open-Meteo.

    Returns None on any fetch or parse error.
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
