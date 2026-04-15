"""Raw METAR and SYNOP string parsers.

Uses python-metar for METAR strings and a lightweight custom parser for SYNOP.
These are needed for providers that return raw text (IEM, OGIMET, NOAA).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from src.ingestion.aviation._conversions import c_to_f, m_to_miles
from src.ingestion.aviation._types import Observation, SynopObs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Raw METAR parser (uses python-metar library)
# ---------------------------------------------------------------------------


def parse_raw_metar(
    raw_text: str,
    station: str = "",
    source: str = "unknown",
) -> Observation | None:
    """Parse a raw METAR string into an Observation.

    Uses the python-metar library for robust parsing with fallback
    to regex for fields that metar library doesn't handle.
    """
    try:
        from metar.Metar import Metar
    except ImportError:
        logger.warning("python-metar not installed; raw METAR parsing unavailable")
        return None

    raw_text = raw_text.strip()
    if not raw_text:
        return None

    # Detect SPECI
    is_speci = raw_text.startswith("SPECI")

    try:
        obs = Metar(raw_text, strict=False)
    except Exception:
        logger.debug("Failed to parse METAR: %s", raw_text[:80], exc_info=True)
        return None

    # Extract temperature
    temp_c = obs.temp.value() if obs.temp else None
    dewpoint_c = obs.dewpt.value() if obs.dewpt else None

    # Wind
    wind_speed_kts = obs.wind_speed.value("KT") if obs.wind_speed else None
    wind_gust_kts = obs.wind_gust.value("KT") if obs.wind_gust else None
    wind_dir = str(int(obs.wind_dir.value())) if obs.wind_dir else None

    # Visibility
    vis_m = obs.vis.value("M") if obs.vis else None
    vis_miles = m_to_miles(vis_m)

    # Pressure
    pressure_hpa = obs.press.value("HPA") if obs.press else None

    # Sky condition & ceiling
    sky_condition: list[dict[str, Any]] = []
    ceiling_ft: int | None = None
    for sky_entry in obs.sky:
        cover = sky_entry[0] if sky_entry[0] else None
        base_ft = int(sky_entry[1].value("FT")) if sky_entry[1] else None
        sky_condition.append({"cover": cover, "base_ft": base_ft})
        if cover in ("BKN", "OVC", "OVX") and ceiling_ft is None:
            ceiling_ft = base_ft

    # Flight category
    flight_category = _compute_flight_category(vis_miles, ceiling_ft)

    # Observation time
    if obs.time:
        observed_at = obs.time.replace(tzinfo=timezone.utc)
    else:
        observed_at = datetime.now(timezone.utc)

    station_id = obs.station_id or station

    return Observation(
        station_icao=station_id,
        observed_at=observed_at,
        temp_c=temp_c,
        temp_f=c_to_f(temp_c),
        dewpoint_c=dewpoint_c,
        dewpoint_f=c_to_f(dewpoint_c),
        wind_speed_kts=wind_speed_kts,
        wind_gust_kts=wind_gust_kts,
        wind_dir=wind_dir,
        visibility_m=vis_m,
        visibility_miles=vis_miles,
        pressure_hpa=pressure_hpa,
        sky_condition=tuple(sky_condition),
        ceiling_ft=ceiling_ft,
        flight_category=flight_category,
        is_speci=is_speci,
        raw_metar=raw_text,
        source=source,
    )


def _compute_flight_category(
    vis_miles: float | None,
    ceiling_ft: int | None,
) -> str:
    """Compute VFR/MVFR/IFR/LIFR flight category."""
    vis = vis_miles if vis_miles is not None else 99
    ceil = ceiling_ft if ceiling_ft is not None else 99999

    if vis < 1 or ceil < 500:
        return "LIFR"
    elif vis < 3 or ceil < 1000:
        return "IFR"
    elif vis <= 5 or ceil <= 3000:
        return "MVFR"
    else:
        return "VFR"


# ---------------------------------------------------------------------------
# Lightweight SYNOP parser
# ---------------------------------------------------------------------------

# SYNOP format reference: WMO Manual on Codes (FM 12)
# A SYNOP message looks like:
#   AAXX DDHHi  IIiii iRixhVV Nddff 1sTTT 2sTTT 3PPPP 4PPPP 5appp 6RRRt ...
#
# We extract the most relevant fields for weather trading:
#   - Temperature (group 1): 1sTTT
#   - Dewpoint (group 2): 2sTTT
#   - Pressure (groups 3/4): 3P0P0P0P0 4PPPP
#   - Wind: Nddff (dd=direction in tens of degrees, ff=speed in m/s or kts)
#   - Precipitation (group 6): 6RRRtR

_SYNOP_TEMP_RE = re.compile(r"\b1([01])(\d{3})\b")
_SYNOP_DEWPOINT_RE = re.compile(r"\b2([01])(\d{3})\b")
_SYNOP_PRESSURE_RE = re.compile(r"\b4(\d{4})\b")
_SYNOP_WIND_RE = re.compile(r"\b\d(\d{2})(\d{2,3})\b")  # Nddff
_SYNOP_PRECIP_RE = re.compile(r"\b6(\d{3})(\d)\b")

# Precipitation period codes (tR in group 6)
_PRECIP_PERIOD_MAP = {
    "1": 6,    # 6 hours
    "2": 12,   # 12 hours
    "3": 18,   # 18 hours
    "4": 24,   # 24 hours
    "5": 1,    # 1 hour
    "6": 2,    # 2 hours
    "7": 3,    # 3 hours
    "8": 9,    # 9 hours
    "9": 15,   # 15 hours
}


def parse_raw_synop(
    raw_text: str,
    wmo_id: str = "",
    observed_at: datetime | None = None,
) -> SynopObs | None:
    """Parse a raw SYNOP string into a SynopObs.

    Extracts temperature, dewpoint, pressure, wind, and precipitation
    from the standard FM 12 SYNOP format.
    """
    if not raw_text or len(raw_text) < 10:
        return None

    if observed_at is None:
        observed_at = datetime.now(timezone.utc)

    # Temperature: group 1sTTT (s=0 positive, s=1 negative, TTT in tenths °C)
    temp_c = None
    m = _SYNOP_TEMP_RE.search(raw_text)
    if m:
        sign = -1.0 if m.group(1) == "1" else 1.0
        temp_c = sign * int(m.group(2)) / 10.0

    # Dewpoint: group 2sTTT
    dewpoint_c = None
    m = _SYNOP_DEWPOINT_RE.search(raw_text)
    if m:
        sign = -1.0 if m.group(1) == "1" else 1.0
        dewpoint_c = sign * int(m.group(2)) / 10.0

    # Station-level pressure: group 4PPPP (in tenths of hPa)
    pressure_hpa = None
    m = _SYNOP_PRESSURE_RE.search(raw_text)
    if m:
        p_raw = int(m.group(1))
        # Convention: if first digit is 0-4, prepend 10; if 5-9, prepend 9
        if p_raw < 5000:
            pressure_hpa = (10000 + p_raw) / 10.0
        else:
            pressure_hpa = (9000 + p_raw) / 10.0

    # Wind: from Nddff group (dd in tens of degrees, ff in m/s)
    wind_speed_kts = None
    wind_dir = None
    # Find the wind group — it's typically the 5th group in the message
    parts = raw_text.split()
    if len(parts) >= 5:
        wind_part = parts[4] if len(parts) > 4 else ""
        if len(wind_part) == 5 and wind_part.isdigit():
            wind_dir_val = int(wind_part[1:3]) * 10
            wind_speed_ms = int(wind_part[3:5])
            wind_dir = wind_dir_val if wind_dir_val <= 360 else None
            wind_speed_kts = wind_speed_ms * 1.94384  # m/s to knots

    # Precipitation: group 6RRRtR
    # This group appears in the main body AFTER the pressure groups (3xxxx/4xxxx)
    # and BEFORE section 333.  The wind group Nddff can also start with 6 (6 oktas)
    # so we search only after a pressure group to avoid false matches.
    precip_mm = None
    precip_period_hours = None
    main_body = raw_text.split(" 333 ")[0] if " 333 " in raw_text else raw_text
    # Find precip group: look for 6RRRtR that appears after a 4xxxx or 5xxxx group
    precip_match = re.search(r"(?:4\d{4}|5\d{4})\s+6(\d{3})(\d)", main_body)
    if precip_match:
        rrr = int(precip_match.group(1))
        tr = precip_match.group(2)
        if rrr < 990:
            precip_mm = float(rrr)
        elif rrr == 990:
            precip_mm = 0.0  # trace
        precip_period_hours = int(_PRECIP_PERIOD_MAP.get(tr, 0)) or None

    # Cloud cover: from N in Nddff (first digit, oktas 0-8, 9=sky obscured)
    cloud_cover_oktas = None
    if len(parts) >= 5 and len(parts[4]) >= 1 and parts[4][0].isdigit():
        n = int(parts[4][0])
        if 0 <= n <= 9:
            cloud_cover_oktas = n

    return SynopObs(
        wmo_id=wmo_id,
        observed_at=observed_at,
        temp_c=temp_c,
        dewpoint_c=dewpoint_c,
        pressure_hpa=pressure_hpa,
        wind_speed_kts=wind_speed_kts,
        wind_dir=wind_dir,
        precip_mm=precip_mm,
        precip_period_hours=precip_period_hours,
        cloud_cover_oktas=cloud_cover_oktas,
        raw_synop=raw_text,
    )
