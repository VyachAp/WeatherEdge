"""Data types for the multi-source aviation weather system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class Observation:
    """Unified METAR observation across all providers."""

    station_icao: str
    observed_at: datetime
    temp_c: float | None = None
    temp_f: float | None = None
    dewpoint_c: float | None = None
    dewpoint_f: float | None = None
    wind_speed_kts: float | None = None
    wind_gust_kts: float | None = None
    wind_dir: str | None = None
    visibility_m: float | None = None
    visibility_miles: float | None = None
    pressure_hpa: float | None = None
    sky_condition: tuple[dict, ...] = ()
    ceiling_ft: int | None = None
    flight_category: str | None = None
    is_speci: bool = False
    raw_metar: str | None = None
    source: str = "unknown"


@dataclass(frozen=True)
class SynopObs:
    """SYNOP station observation (WMO land stations)."""

    wmo_id: str
    observed_at: datetime
    temp_c: float | None = None
    dewpoint_c: float | None = None
    pressure_hpa: float | None = None
    wind_speed_kts: float | None = None
    wind_dir: int | None = None
    precip_mm: float | None = None
    precip_period_hours: int | None = None
    cloud_cover_oktas: int | None = None
    raw_synop: str | None = None


@dataclass(frozen=True)
class MinuteObs:
    """1-minute ASOS observation (US stations only)."""

    station_icao: str
    observed_at: datetime
    temp_c: float | None = None
    dewpoint_c: float | None = None
    wind_speed_kts: float | None = None
    wind_dir: int | None = None
    precip_mm: float | None = None
    pressure_hpa: float | None = None


@dataclass(frozen=True)
class TempTrend:
    """Temperature trend analysis result."""

    current_f: float | None = None
    min_f: float | None = None
    max_f: float | None = None
    trend_direction: str = "unknown"  # rising, falling, steady, unknown
    rate_per_hour: float = 0.0
    observation_count: int = 0
    period_hours: float = 0.0
    source: str = "metar"


@dataclass(frozen=True)
class PrecipAccum:
    """Combined METAR + SYNOP precipitation accumulation."""

    station: str
    period_hours: float = 0.0
    total_mm: float | None = None
    total_inches: float | None = None
    metar_precip_mm: float | None = None
    synop_precip_mm: float | None = None
    source: str = "metar"


@dataclass
class WeatherBriefing:
    """Complete pilot-style weather briefing for a station."""

    station: str
    generated_at: datetime | None = None
    current_obs: Observation | None = None
    taf: dict | None = None
    temp_trend: TempTrend | None = None
    precip_accum: PrecipAccum | None = None
    speci_events: list[dict] = field(default_factory=list)
    pireps: list[dict] = field(default_factory=list)
    sigmets: list[dict] = field(default_factory=list)
    one_minute_available: bool = False
