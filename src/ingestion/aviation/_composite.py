"""Composite aviation functions built on the multi-provider aggregator.

These are the NEW public API functions that leverage multi-source data
for more accurate weather intelligence.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from src.ingestion.aviation._aggregator import get_aggregator
from src.ingestion.aviation._conversions import c_to_f, mm_to_inches
from src.ingestion.aviation._types import (
    MinuteObs,
    Observation,
    PrecipAccum,
    SynopObs,
    TempTrend,
    WeatherBriefing,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Real-time observations
# ---------------------------------------------------------------------------


async def get_latest_observation(station: str) -> Observation | None:
    """Get the latest METAR from the fastest available source."""
    agg = get_aggregator()
    return await agg.get_metar(station)


async def get_latest_observations_bulk(stations: list[str]) -> dict[str, Observation]:
    """Get latest METARs for multiple stations from the fastest available source."""
    agg = get_aggregator()
    return await agg.get_metar_bulk(stations)


# ---------------------------------------------------------------------------
# Historical observations
# ---------------------------------------------------------------------------


async def get_observation_history(station: str, hours: int = 24) -> list[Observation]:
    """Get METAR history from the best available archive source."""
    agg = get_aggregator()
    return await agg.get_metar_history(station, hours)


async def get_synop_history(wmo_id: str, hours: int = 24) -> list[SynopObs]:
    """Get SYNOP observations from OGIMET (only source)."""
    agg = get_aggregator()
    return await agg.get_synop(wmo_id, hours)


async def get_one_minute_data(station: str, hours: int = 6) -> list[MinuteObs]:
    """Get 1-minute ASOS data (US stations only, from IEM)."""
    agg = get_aggregator()
    return await agg.get_one_minute(station, hours)


# ---------------------------------------------------------------------------
# TAF
# ---------------------------------------------------------------------------


async def get_latest_taf(station: str) -> dict[str, Any] | None:
    """Get the latest TAF from the best available source."""
    agg = get_aggregator()
    return await agg.get_taf(station)


async def get_taf_amendment_count_multi(station: str, hours: int = 24) -> int:
    """Count TAF amendments using the existing DB-based approach.

    This delegates to the legacy function which queries the database.
    """
    # Import at call time to avoid circular dependency
    from src.ingestion.aviation import taf_amendment_count
    return await taf_amendment_count(station, hours)


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------


async def get_active_sigmets(region: str = "global") -> list[dict[str, Any]]:
    """Get active SIGMETs (from AWC — the only real-time source)."""
    agg = get_aggregator()
    all_alerts = await agg.get_sigmets()
    return [a for a in all_alerts if a.get("alert_type") == "SIGMET"]


async def get_pireps_near(lat: float, lon: float, radius_nm: int = 100) -> list[dict[str, Any]]:
    """Get PIREPs near a location (from AWC — the only source)."""
    agg = get_aggregator()
    return await agg.get_pireps(lat, lon, radius_nm)


# ---------------------------------------------------------------------------
# Trend analysis
# ---------------------------------------------------------------------------


async def compute_temperature_trend(station: str, hours: int = 6) -> TempTrend:
    """Compute temperature trend from historical observations.

    Uses the best available source for history. Returns trend analysis
    including current, min, max, rate of change, and direction.
    """
    agg = get_aggregator()

    # Try to get 1-minute data first (US only, extreme precision)
    one_min = await agg.get_one_minute(station, hours)
    if one_min and len(one_min) >= 2:
        return _compute_trend_from_minute_obs(one_min, hours)

    # Fall back to METAR history
    history = await agg.get_metar_history(station, hours)
    if not history:
        return TempTrend(source="none")

    return _compute_trend_from_observations(history, hours)


def _compute_trend_from_observations(
    observations: list[Observation], hours: float,
) -> TempTrend:
    """Compute temperature trend from Observation list."""
    temps = [
        (obs.observed_at, obs.temp_f)
        for obs in observations
        if obs.temp_f is not None
    ]

    if not temps:
        return TempTrend(source="metar")

    temps.sort(key=lambda x: x[0])
    temp_values = [t[1] for t in temps]
    current = temp_values[-1]

    if len(temps) < 2:
        return TempTrend(
            current_f=current,
            min_f=current,
            max_f=current,
            trend_direction="steady",
            rate_per_hour=0.0,
            observation_count=1,
            period_hours=hours,
            source="metar",
        )

    # Linear regression
    t0 = temps[0][0]
    x = [(t[0] - t0).total_seconds() / 3600.0 for t in temps]
    y = temp_values

    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_x2 = sum(xi * xi for xi in x)

    denom = n * sum_x2 - sum_x * sum_x
    slope = (n * sum_xy - sum_x * sum_y) / denom if denom != 0 else 0.0

    if slope > 0.5:
        direction = "rising"
    elif slope < -0.5:
        direction = "falling"
    else:
        direction = "steady"

    return TempTrend(
        current_f=current,
        min_f=min(temp_values),
        max_f=max(temp_values),
        trend_direction=direction,
        rate_per_hour=round(slope, 2),
        observation_count=len(temps),
        period_hours=hours,
        source="metar",
    )


def _compute_trend_from_minute_obs(
    observations: list[MinuteObs], hours: float,
) -> TempTrend:
    """Compute temperature trend from 1-minute observations."""
    temps = [
        (obs.observed_at, c_to_f(obs.temp_c))
        for obs in observations
        if obs.temp_c is not None
    ]
    temps = [(t, v) for t, v in temps if v is not None]

    if not temps:
        return TempTrend(source="one_minute")

    temps.sort(key=lambda x: x[0])
    temp_values = [t[1] for t in temps]
    current = temp_values[-1]

    if len(temps) < 2:
        return TempTrend(
            current_f=current, min_f=current, max_f=current,
            trend_direction="steady", rate_per_hour=0.0,
            observation_count=1, period_hours=hours, source="one_minute",
        )

    t0 = temps[0][0]
    x = [(t[0] - t0).total_seconds() / 3600.0 for t in temps]
    y = temp_values

    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_x2 = sum(xi * xi for xi in x)

    denom = n * sum_x2 - sum_x * sum_x
    slope = (n * sum_xy - sum_x * sum_y) / denom if denom != 0 else 0.0

    if slope > 0.5:
        direction = "rising"
    elif slope < -0.5:
        direction = "falling"
    else:
        direction = "steady"

    return TempTrend(
        current_f=current,
        min_f=min(temp_values),
        max_f=max(temp_values),
        trend_direction=direction,
        rate_per_hour=round(slope, 3),
        observation_count=len(temps),
        period_hours=hours,
        source="one_minute",
    )


# ---------------------------------------------------------------------------
# Precipitation
# ---------------------------------------------------------------------------


async def compute_precip_accumulation(station: str, hours: int = 24) -> PrecipAccum:
    """Combine METAR and SYNOP precipitation data for most accurate totals.

    SYNOP stations report precipitation totals more reliably than METAR.
    When both are available, prefer SYNOP for the total but use METAR
    to fill gaps.
    """
    agg = get_aggregator()

    # Get METAR history for precipitation
    metar_history = await agg.get_metar_history(station, hours)
    metar_precip_mm = _extract_metar_precip(metar_history)

    # If we have a WMO ID, try SYNOP too
    # (WMO ID lookup would require a station database; for now use METAR only)
    synop_precip_mm = None

    total_mm = metar_precip_mm
    if synop_precip_mm is not None:
        total_mm = synop_precip_mm  # SYNOP is more reliable for totals
    elif metar_precip_mm is not None:
        total_mm = metar_precip_mm

    return PrecipAccum(
        station=station,
        period_hours=float(hours),
        total_mm=total_mm,
        total_inches=mm_to_inches(total_mm),
        metar_precip_mm=metar_precip_mm,
        synop_precip_mm=synop_precip_mm,
        source="metar+synop" if synop_precip_mm is not None else "metar",
    )


def _extract_metar_precip(observations: list[Observation]) -> float | None:
    """Extract total precipitation from METAR observations.

    METARs typically include hourly precipitation in remarks (P0010 = 0.10 inches).
    This is a best-effort extraction from raw METAR text.
    """
    import re

    total_inches = 0.0
    found_any = False

    for obs in observations:
        if not obs.raw_metar:
            continue
        # Hourly precip: P0010 = 0.10 inches
        m = re.search(r"\bP(\d{4})\b", obs.raw_metar)
        if m:
            total_inches += int(m.group(1)) / 100.0
            found_any = True

    if not found_any:
        return None

    return total_inches * 25.4  # Convert to mm


# ---------------------------------------------------------------------------
# Composite briefing
# ---------------------------------------------------------------------------


async def get_full_weather_picture(station: str) -> WeatherBriefing:
    """Build a complete aviation weather briefing from all available sources.

    Combines latest observation, TAF, temperature trend, precipitation,
    SPECI events, PIREPs, and SIGMETs.
    """
    import asyncio

    agg = get_aggregator()

    # Gather all data concurrently
    obs_task = agg.get_metar(station)
    taf_task = agg.get_taf(station)
    trend_task = compute_temperature_trend(station, hours=6)
    precip_task = compute_precip_accumulation(station, hours=24)
    sigmets_task = agg.get_sigmets()

    obs, taf, trend, precip, sigmets = await asyncio.gather(
        obs_task, taf_task, trend_task, precip_task, sigmets_task,
        return_exceptions=True,
    )

    # Handle exceptions gracefully
    if isinstance(obs, BaseException):
        obs = None
    if isinstance(taf, BaseException):
        taf = None
    if isinstance(trend, BaseException):
        trend = None
    if isinstance(precip, BaseException):
        precip = None
    if isinstance(sigmets, BaseException):
        sigmets = []

    # Check for 1-minute data availability (US stations)
    one_min_available = False
    try:
        one_min = await agg.get_one_minute(station, hours=1)
        one_min_available = len(one_min) > 0
    except Exception:
        pass

    # Get SPECI events from history
    speci_events: list[dict[str, Any]] = []
    try:
        history = await agg.get_metar_history(station, hours=12)
        for h in history:
            if h.is_speci:
                speci_events.append({
                    "observed_at": h.observed_at.isoformat(),
                    "raw_metar": h.raw_metar,
                    "temp_f": h.temp_f,
                })
    except Exception:
        pass

    return WeatherBriefing(
        station=station,
        generated_at=datetime.now(timezone.utc),
        current_obs=obs,
        taf=taf,
        temp_trend=trend,
        precip_accum=precip,
        speci_events=speci_events,
        pireps=[],  # PIREPs require lat/lon; caller should add if available
        sigmets=sigmets if isinstance(sigmets, list) else [],
        one_minute_available=one_min_available,
    )
