"""Multi-source global aviation weather data system.

Aggregates data from 6 providers with automatic failover:
  - AWC (aviationweather.gov) — METAR, TAF, PIREP, SIGMET
  - CheckWX — real-time parsed METAR/TAF (API key required)
  - IEM (Iowa Environmental Mesonet) — METAR history, 1-minute ASOS
  - OGIMET — global METAR/TAF/SYNOP archive
  - NOAA/NWS — raw METAR text
  - AVWX — backup parsed METAR/TAF (API key required)

All original function signatures are preserved for backward compatibility.
New multi-source functions are also exported from this module.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time as _time
from datetime import datetime, timedelta, timezone
from typing import Any, Hashable, TYPE_CHECKING

import httpx
from sqlalchemy.dialects.postgresql import insert as pg_insert
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings
from src.db.engine import async_session
from src.db.models import AviationAlert, MetarObservation, Pirep, TafForecast

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AWC_BASE = "https://aviationweather.gov/api/data"
METAR_URL = f"{AWC_BASE}/metar"
TAF_URL = f"{AWC_BASE}/taf"
PIREP_URL = f"{AWC_BASE}/pirep"
AIRSIGMET_URL = f"{AWC_BASE}/airsigmet"

# Max stations per bulk request to avoid URL length issues
_BULK_BATCH_SIZE = 20

# Precipitation weather codes in TAFs
_PRECIP_CODES = {"RA", "SN", "TS", "TSRA", "TSSN", "DZ", "FZRA", "FZDZ",
                 "SG", "GR", "GS", "PL", "IC", "SHRA", "SHSN", "SHGR"}

# ---------------------------------------------------------------------------
# In-memory TTL cache for aviation HTTP responses
# ---------------------------------------------------------------------------


class _TTLCache:
    """Simple in-memory TTL cache. Not thread-safe, but fine for asyncio."""

    def __init__(self, ttl_seconds: float = 300.0) -> None:
        self._ttl = ttl_seconds
        self._store: dict[Hashable, tuple[float, Any]] = {}

    def get(self, key: Hashable) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        ts, value = entry
        if _time.monotonic() - ts > self._ttl:
            del self._store[key]
            return None
        return value

    def set(self, key: Hashable, value: Any) -> None:
        self._store[key] = (_time.monotonic(), value)

    def clear(self) -> None:
        self._store.clear()


_cache = _TTLCache(ttl_seconds=300)  # 5-minute TTL


def clear_aviation_cache() -> None:
    """Clear the in-memory aviation data cache. Useful for testing."""
    _cache.clear()


# ---------------------------------------------------------------------------
# Rate limiting & HTTP helpers
# ---------------------------------------------------------------------------

_awc_semaphore = asyncio.Semaphore(max(1, int(settings.AWC_RATE_LIMIT_RPS)))
_awc_interval = 1.0 / settings.AWC_RATE_LIMIT_RPS


async def _throttle() -> None:
    """Acquire rate-limit slot and schedule release after interval."""
    await _awc_semaphore.acquire()

    async def _release() -> None:
        await asyncio.sleep(_awc_interval)
        _awc_semaphore.release()

    asyncio.create_task(_release())


def _make_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={
            "User-Agent": settings.AWC_USER_AGENT,
            "Accept-Encoding": "gzip",
        },
        timeout=15,
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
    reraise=True,
)
async def _awc_get_json(
    client: httpx.AsyncClient,
    url: str,
    params: dict[str, Any] | None = None,
) -> Any:
    """Fetch JSON from AWC API with rate limiting and retry."""
    await _throttle()
    resp = await client.get(url, params=params)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# METAR parsing & fetching
# ---------------------------------------------------------------------------


def _c_to_f(temp_c: float | None) -> float | None:
    if temp_c is None:
        return None
    return temp_c * 9.0 / 5.0 + 32.0


def _parse_awc_observed_at(raw: dict[str, Any]) -> datetime:
    """Extract observation timestamp from an AWC API object.

    AWC returns ``obsTime`` as a Unix epoch integer and ``reportTime`` as an
    ISO8601 string. Try ISO first, then epoch, then fall back to now().
    The historical implementation tried ``fromisoformat`` on the Unix int,
    silently failed, and stamped every observation with ``now()`` — which
    made temperature-trend regression collapse the time axis and blow up
    the slope.
    """
    report_time = raw.get("reportTime")
    if report_time:
        try:
            return datetime.fromisoformat(str(report_time).replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            pass
    obs_time = raw.get("obsTime")
    if obs_time is not None:
        try:
            return datetime.fromtimestamp(int(obs_time), tz=timezone.utc)
        except (ValueError, TypeError):
            try:
                return datetime.fromisoformat(str(obs_time).replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass
    return datetime.now(timezone.utc)


def _parse_metar_json(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert AWC METAR JSON object to DB-ready dict."""
    temp_c = raw.get("temp")
    dewpoint_c = raw.get("dwpt")

    clouds = raw.get("clouds", []) or []
    sky_condition = [
        {"cover": c.get("cover"), "base_ft": c.get("base")}
        for c in clouds
    ]

    ceiling_ft = None
    for c in clouds:
        cover = c.get("cover")
        if cover in ("BKN", "OVC", "OVX"):
            ceiling_ft = c.get("base")
            break

    vis_raw = raw.get("visib")
    try:
        vis_miles = float(str(vis_raw).rstrip("+").lstrip("P")) if vis_raw is not None else None
    except (ValueError, TypeError):
        vis_miles = None
    vis_m = vis_miles * 1609.34 if vis_miles is not None else None

    observed_at = _parse_awc_observed_at(raw)

    return {
        "station_icao": raw.get("icaoId"),
        "observed_at": observed_at,
        "temp_c": temp_c,
        "dewpoint_c": dewpoint_c,
        "temp_f": _c_to_f(temp_c),
        "dewpoint_f": _c_to_f(dewpoint_c),
        "wind_speed_kts": raw.get("wspd"),
        "wind_dir": str(raw.get("wdir")) if raw.get("wdir") is not None else None,
        "wind_gust_kts": raw.get("wgst"),
        "visibility_m": vis_m,
        "visibility_miles": vis_miles,
        "pressure_hpa": raw.get("altim"),
        "sky_condition": sky_condition,
        "ceiling_ft": ceiling_ft,
        "flight_category": raw.get("fltcat"),
        "is_speci": raw.get("metar_type") == "SPECI",
        "raw_metar": raw.get("rawOb"),
        "fetched_at": datetime.now(timezone.utc),
    }


async def fetch_latest_metars(
    station_list: list[str],
    session: AsyncSession | None = None,
) -> list[dict[str, Any]]:
    """Fetch latest METARs for a list of ICAO stations.

    Uses bulk requests with comma-separated ICAOs (up to 20 per request).
    Persists results to the database and returns parsed dicts.
    """
    if not station_list:
        return []

    own_session = session is None
    if own_session:
        session = async_session()

    all_parsed: list[dict[str, Any]] = []

    try:
        async with _make_client() as client:
            for i in range(0, len(station_list), _BULK_BATCH_SIZE):
                batch = station_list[i : i + _BULK_BATCH_SIZE]
                ids_str = ",".join(batch)
                try:
                    data = await _awc_get_json(
                        client, METAR_URL, {"ids": ids_str, "format": "json"}
                    )
                except Exception:
                    logger.warning(
                        "METAR fetch failed for batch %s", ids_str, exc_info=True
                    )
                    continue

                if not isinstance(data, list):
                    continue

                for entry in data:
                    try:
                        parsed = _parse_metar_json(entry)
                        all_parsed.append(parsed)
                        await session.execute(
                            pg_insert(MetarObservation)
                            .values(**parsed)
                            .on_conflict_do_nothing(constraint="uq_metar_station_obs")
                        )
                    except Exception:
                        logger.warning(
                            "METAR parse failed for %s",
                            entry.get("icaoId", "?"),
                            exc_info=True,
                        )

        await session.commit()
        logger.info("Persisted %d METAR observations", len(all_parsed))
    except Exception:
        await session.rollback()
        raise
    finally:
        if own_session:
            await session.close()

    return all_parsed


async def fetch_metar_history(
    station: str,
    hours: int = 24,
    session: AsyncSession | None = None,
) -> list[dict[str, Any]]:
    """Fetch historical METARs for a single station."""
    cache_key = ("metar_history", station, hours)
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    own_session = session is None
    if own_session:
        session = async_session()

    parsed: list[dict[str, Any]] = []
    try:
        async with _make_client() as client:
            data = await _awc_get_json(
                client,
                METAR_URL,
                {"ids": station, "format": "json", "hours": hours},
            )

            if isinstance(data, list):
                for entry in data:
                    try:
                        p = _parse_metar_json(entry)
                        parsed.append(p)
                        await session.execute(
                            pg_insert(MetarObservation)
                            .values(**p)
                            .on_conflict_do_nothing(constraint="uq_metar_station_obs")
                        )
                    except Exception:
                        logger.warning(
                            "METAR history parse failed", exc_info=True
                        )

        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        if own_session:
            await session.close()

    _cache.set(cache_key, parsed)
    return parsed


async def get_current_temp(station: str) -> float | None:
    """Return the latest temperature in Fahrenheit for a station."""
    metars = await fetch_latest_metars([station])
    if not metars:
        return None
    return metars[0].get("temp_f")


async def get_routine_daily_max(
    station: str,
    hours: int = 24,
    reference_utc: datetime | None = None,
) -> tuple[float | None, int]:
    """Return (max_temp_f, routine_count) for the station's LOCAL-city day.

    Filters out SPECI reports before computing daily max. The routine_count
    is used by the MIN_ROUTINE_COUNT circuit breaker.

    Uses local-city timezone so the daily max aligns with Wunderground's
    resolution-day convention — see `icao_timezone` in `signals.mapper`.

    ``reference_utc`` anchors the local day. Default is "now", which is
    correct for intra-day callers reading the current day's state. At
    settlement time, pass the market's resolution timestamp (e.g.
    ``market.end_date``) — otherwise stations far east of UTC (KST, JST,
    UTC+8) pick the *next* local day and record the pre-dawn window
    instead of the market's target day.

    The reference is nudged back 1µs before truncating to local midnight
    so that UTC+12 stations (NZ), whose 12:00 UTC market close lands
    exactly on their 00:00 local-next-day boundary, anchor to the day
    that ENDS at close rather than the next one.
    """
    from src.signals.mapper import icao_timezone

    tz = icao_timezone(station)
    ref_utc = reference_utc or datetime.now(timezone.utc)
    ref_local = (ref_utc - timedelta(microseconds=1)).astimezone(tz)
    local_day_start = ref_local.replace(hour=0, minute=0, second=0, microsecond=0)
    local_day_end = local_day_start + timedelta(days=1)
    utc_start = local_day_start.astimezone(timezone.utc)
    utc_end = local_day_end.astimezone(timezone.utc)

    # Fetch enough history to reach the start of the target local day from
    # now. For UTC+9 at settlement (22:00 UTC), the local-day start is 31h
    # ago; the default 24h window misses the first third of the day.
    now_utc = datetime.now(timezone.utc)
    needed_hours = int((now_utc - utc_start).total_seconds() // 3600) + 2
    fetch_hours = max(hours, needed_hours)
    history = await fetch_metar_history(station, hours=fetch_hours)

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


def detect_metar_cycle(observations: list[dict[str, Any]]) -> list[int]:
    """Auto-detect METAR publication minutes from recent routine observations.

    Examines the minute-of-hour across the provided routine METARs and returns
    the most common publication minutes (e.g., [20, 50] or [0, 30]).
    Expects at least 4 observations for meaningful detection.
    """
    minutes: list[int] = []
    for m in observations:
        if m.get("is_speci"):
            continue
        obs_at = m.get("observed_at")
        if obs_at is not None and isinstance(obs_at, datetime):
            minutes.append(obs_at.minute)

    if len(minutes) < 4:
        return []

    # Cluster minutes by rounding to nearest 5
    from collections import Counter
    rounded = [((m + 2) // 5) * 5 % 60 for m in minutes]
    counts = Counter(rounded)
    # Return minutes that appear at least twice, sorted
    return sorted(m for m, c in counts.items() if c >= 2)


async def get_temp_trend(
    station: str,
    hours: int = 6,
    *,
    routine_only: bool = False,
) -> dict[str, Any]:
    """Compute temperature trend from recent METAR observations.

    When routine_only=True, SPECI reports are excluded from the calculation.
    Returns dict with: current, min, max, trend_direction, rate_of_change_per_hour,
    dewpoint_rate (°F/hr).
    """
    history = await fetch_metar_history(station, hours=hours)

    if routine_only:
        history = [m for m in history if not m.get("is_speci")]

    temps = [
        (m["observed_at"], m["temp_f"])
        for m in history
        if m.get("temp_f") is not None
    ]

    if not temps:
        return {
            "current": None,
            "min": None,
            "max": None,
            "trend_direction": "unknown",
            "rate_of_change_per_hour": 0.0,
        }

    temps.sort(key=lambda x: x[0])
    temp_values = [t[1] for t in temps]
    current = temp_values[-1]

    if len(temps) < 2:
        return {
            "current": current,
            "min": current,
            "max": current,
            "trend_direction": "steady",
            "rate_of_change_per_hour": 0.0,
        }

    # Linear regression for rate of change
    t0 = temps[0][0]
    x = [(t[0] - t0).total_seconds() / 3600.0 for t in temps]
    y = temp_values

    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_x2 = sum(xi * xi for xi in x)

    denom = n * sum_x2 - sum_x * sum_x
    if denom == 0:
        slope = 0.0
    else:
        slope = (n * sum_xy - sum_x * sum_y) / denom

    if slope > 0.5:
        direction = "rising"
    elif slope < -0.5:
        direction = "falling"
    else:
        direction = "steady"

    # Dewpoint trend (same linear regression approach)
    if routine_only:
        filtered_history = [m for m in history if not m.get("is_speci")]
    else:
        filtered_history = history
    dewpoints = [
        (m["observed_at"], m["dewpoint_f"])
        for m in filtered_history
        if m.get("dewpoint_f") is not None and m.get("observed_at") is not None
    ]
    dewpoint_rate = 0.0
    if len(dewpoints) >= 2:
        dewpoints.sort(key=lambda dp: dp[0])
        dp_t0 = dewpoints[0][0]
        dp_x = [(dp[0] - dp_t0).total_seconds() / 3600.0 for dp in dewpoints]
        dp_y = [dp[1] for dp in dewpoints]
        dp_n = len(dp_x)
        dp_sum_x = sum(dp_x)
        dp_sum_y = sum(dp_y)
        dp_sum_xy = sum(xi * yi for xi, yi in zip(dp_x, dp_y))
        dp_sum_x2 = sum(xi * xi for xi in dp_x)
        dp_denom = dp_n * dp_sum_x2 - dp_sum_x * dp_sum_x
        if dp_denom != 0:
            dewpoint_rate = round((dp_n * dp_sum_xy - dp_sum_x * dp_sum_y) / dp_denom, 2)

    return {
        "current": current,
        "min": min(temp_values),
        "max": max(temp_values),
        "trend_direction": direction,
        "rate_of_change_per_hour": round(slope, 2),
        "dewpoint_rate": dewpoint_rate,
    }


async def detect_speci_events(
    station: str,
    hours: int = 12,
) -> list[dict[str, Any]]:
    """Return SPECI reports for a station within the given time window."""
    history = await fetch_metar_history(station, hours=hours)
    return [m for m in history if m.get("is_speci")]


# ---------------------------------------------------------------------------
# TAF parsing & fetching
# ---------------------------------------------------------------------------


def _parse_taf_json(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert AWC TAF JSON object to DB-ready dict."""
    periods: list[dict[str, Any]] = []
    for p in raw.get("fcsts", []) or []:
        periods.append({
            "type": p.get("fcstType"),
            "from": p.get("timeFrom"),
            "to": p.get("timeTo"),
            "wind_dir": p.get("wdir"),
            "wind_speed_kts": p.get("wspd"),
            "wind_gust_kts": p.get("wgst"),
            "visibility_miles": p.get("visib"),
            "sky_condition": [
                {"cover": c.get("cover"), "base_ft": c.get("base")}
                for c in (p.get("clouds") or [])
            ],
            "weather": p.get("wxString"),
            "prob": p.get("probability"),
        })

    raw_text = raw.get("rawTAF", "")
    amendment_number = 1 if (" AMD " in raw_text or " COR " in raw_text) else 0

    issue_str = raw.get("issueTime", "")
    from_str = raw.get("validTimeFrom", "")
    to_str = raw.get("validTimeTo", "")

    def _parse_dt(s: str) -> datetime:
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return datetime.now(timezone.utc)

    return {
        "station_icao": raw.get("icaoId"),
        "issued_at": _parse_dt(issue_str),
        "valid_from": _parse_dt(from_str),
        "valid_to": _parse_dt(to_str),
        "periods": periods,
        "amendment_number": amendment_number,
        "raw_taf": raw_text,
        "fetched_at": datetime.now(timezone.utc),
    }


async def fetch_latest_tafs(
    station_list: list[str],
    session: AsyncSession | None = None,
) -> list[dict[str, Any]]:
    """Fetch latest TAFs for a list of ICAO stations."""
    if not station_list:
        return []

    cache_key = ("tafs", tuple(sorted(station_list)))
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    own_session = session is None
    if own_session:
        session = async_session()

    all_parsed: list[dict[str, Any]] = []

    try:
        async with _make_client() as client:
            for i in range(0, len(station_list), _BULK_BATCH_SIZE):
                batch = station_list[i : i + _BULK_BATCH_SIZE]
                ids_str = ",".join(batch)
                try:
                    data = await _awc_get_json(
                        client, TAF_URL, {"ids": ids_str, "format": "json"}
                    )
                except Exception:
                    logger.warning(
                        "TAF fetch failed for batch %s", ids_str, exc_info=True
                    )
                    continue

                if not isinstance(data, list):
                    continue

                for entry in data:
                    try:
                        parsed = _parse_taf_json(entry)
                        all_parsed.append(parsed)
                        session.add(TafForecast(**parsed))
                    except Exception:
                        logger.warning(
                            "TAF parse failed for %s",
                            entry.get("icaoId", "?"),
                            exc_info=True,
                        )

        await session.commit()
        logger.info("Persisted %d TAF forecasts", len(all_parsed))
    except Exception:
        await session.rollback()
        raise
    finally:
        if own_session:
            await session.close()

    _cache.set(cache_key, all_parsed)
    return all_parsed


def _find_taf_period(
    periods: list[dict[str, Any]],
    target_time: datetime,
) -> dict[str, Any] | None:
    """Find the TAF period (FM type) that covers the target time.

    Also returns overlapping TEMPO/PROB30 groups via the 'overlays' key.
    """
    base_period = None
    overlays: list[dict[str, Any]] = []

    for p in periods:
        pfrom = p.get("from", "")
        pto = p.get("to", "")
        try:
            dt_from = datetime.fromisoformat(pfrom.replace("Z", "+00:00"))
            dt_to = datetime.fromisoformat(pto.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            continue

        if dt_from <= target_time <= dt_to:
            ptype = p.get("type", "")
            if ptype in ("FM", "BECMG", ""):
                base_period = p
            elif ptype in ("TEMPO", "PROB30", "PROB40"):
                overlays.append(p)

    if base_period is not None:
        base_period = dict(base_period)
        base_period["overlays"] = overlays

    return base_period


async def get_taf_temperature_forecast(
    station: str,
    target_time: datetime,
) -> dict[str, Any]:
    """Get TAF-based temperature forecast for a station at a target time.

    Parses TX (max) and TN (min) temperature groups from the raw TAF text.
    Returns the forecast high/low in Fahrenheit if available.
    """
    tafs = await fetch_latest_tafs([station])
    if not tafs:
        return {"temp_f": None, "max_f": None, "min_f": None, "confidence": 0.0, "source": "no_taf"}

    taf = tafs[0]
    raw_text = taf.get("raw_taf", "")

    # Parse TX/TN groups: TX25/1215Z TN12/0306Z
    # TX = max temp in Celsius, TN = min temp in Celsius
    # Negative temps: TXM02/1215Z (M = minus)
    max_c = _parse_taf_temp_group(raw_text, "TX")
    min_c = _parse_taf_temp_group(raw_text, "TN")

    max_f = _c_to_f(max_c)
    min_f = _c_to_f(min_c)

    period = _find_taf_period(taf.get("periods", []), target_time)
    confidence = 0.7 if period and not period.get("overlays") else 0.5
    if max_f is not None or min_f is not None:
        confidence = max(confidence, 0.75)

    return {
        "temp_f": max_f,  # backwards compat: use max as the point forecast
        "max_f": max_f,
        "min_f": min_f,
        "confidence": confidence,
        "source": "taf_txtn" if (max_f or min_f) else "taf",
    }


# Regex for TAF temperature groups: TX25/1215Z or TNM02/0306Z
_TAF_TEMP_RE = re.compile(
    r"\b(?P<type>TX|TN)(?P<sign>M?)(?P<temp>\d{1,2})/(?P<day>\d{2})(?P<hour>\d{2})(?:\d{2})?Z?\b"
)


def _parse_taf_temp_group(raw_taf: str, group_type: str) -> float | None:
    """Extract TX (max) or TN (min) temperature from raw TAF text.

    Returns temperature in Celsius, or None if not found.
    """
    for m in _TAF_TEMP_RE.finditer(raw_taf):
        if m.group("type") == group_type:
            temp = float(m.group("temp"))
            if m.group("sign") == "M":
                temp = -temp
            return temp
    return None


async def get_taf_precip_probability(
    station: str,
    target_time: datetime,
) -> float:
    """Estimate precipitation probability from TAF for a target time.

    Checks for precipitation weather codes in all TAF periods that cover
    the target time, weighting by PROB30/TEMPO probability groups.
    """
    tafs = await fetch_latest_tafs([station])
    if not tafs:
        return 0.0

    taf = tafs[0]
    periods = taf.get("periods", [])
    precip_prob = 0.0

    for p in periods:
        pfrom = p.get("from", "")
        pto = p.get("to", "")
        try:
            dt_from = datetime.fromisoformat(pfrom.replace("Z", "+00:00"))
            dt_to = datetime.fromisoformat(pto.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            continue

        if not (dt_from <= target_time <= dt_to):
            continue

        wx = p.get("weather") or ""
        wx_tokens = set(wx.upper().split())
        if not (wx_tokens & _PRECIP_CODES):
            continue

        ptype = p.get("type", "")
        prob_value = p.get("prob")

        if ptype in ("FM", "BECMG", ""):
            # Categorical forecast of precip
            precip_prob = max(precip_prob, 0.8)
        elif prob_value is not None:
            # PROB30 → 0.30, PROB40 → 0.40
            precip_prob = max(precip_prob, prob_value / 100.0)
        elif ptype == "TEMPO":
            # TEMPO without explicit probability → ~40% chance
            precip_prob = max(precip_prob, 0.4)

    return min(precip_prob, 1.0)


async def taf_amendment_count(
    station: str,
    hours: int = 24,
    session: AsyncSession | None = None,
) -> int:
    """Count distinct TAF issuances for a station in the time window.

    High amendment count indicates unstable forecast conditions.
    """
    from sqlalchemy import func, select

    own_session = session is None
    if own_session:
        session = async_session()

    try:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        stmt = (
            select(func.count(func.distinct(TafForecast.issued_at)))
            .where(TafForecast.station_icao == station)
            .where(TafForecast.issued_at >= cutoff)
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none() or 0
    finally:
        if own_session:
            await session.close()


# ---------------------------------------------------------------------------
# PIREP parsing & fetching
# ---------------------------------------------------------------------------


def _parse_pirep_json(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert AWC PIREP JSON object to DB-ready dict."""
    observed_at = _parse_awc_observed_at(raw)

    return {
        "report_id": raw.get("airepId") or raw.get("pirepId"),
        "observed_at": observed_at,
        "lat": raw.get("lat"),
        "lon": raw.get("lon"),
        "altitude_ft": raw.get("altFt") or raw.get("fltlvl"),
        "icing_type": raw.get("icgType1"),
        "icing_intensity": raw.get("icgInt1"),
        "turbulence_type": raw.get("tbType1"),
        "turbulence_intensity": raw.get("tbInt1"),
        "weather": raw.get("wxString"),
        "raw_text": raw.get("rawOb"),
        "fetched_at": datetime.now(timezone.utc),
    }


async def fetch_pireps_near(
    lat: float,
    lon: float,
    radius_nm: int = 100,
    session: AsyncSession | None = None,
) -> list[dict[str, Any]]:
    """Fetch PIREPs near a location within a radius (nautical miles)."""
    cache_key = ("pireps", round(lat, 2), round(lon, 2), radius_nm)
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    own_session = session is None
    if own_session:
        session = async_session()

    parsed: list[dict[str, Any]] = []

    try:
        async with _make_client() as client:
            data = await _awc_get_json(
                client,
                PIREP_URL,
                {
                    "format": "json",
                    "age": 6,
                    "dist": radius_nm,
                    "lat": lat,
                    "lon": lon,
                },
            )

            if isinstance(data, list):
                for entry in data:
                    try:
                        p = _parse_pirep_json(entry)
                        parsed.append(p)
                        session.add(Pirep(**p))
                    except Exception:
                        logger.warning("PIREP parse failed", exc_info=True)

        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        if own_session:
            await session.close()

    _cache.set(cache_key, parsed)
    return parsed


async def has_severe_weather_reports(
    lat: float,
    lon: float,
    hours: int = 6,
) -> bool:
    """Check if there are MOD/SEV/EXTM severity PIREPs near a location."""
    pireps = await fetch_pireps_near(lat, lon, radius_nm=100)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    severe_intensities = {"MOD", "SEV", "EXTM"}
    for p in pireps:
        if p.get("observed_at") and p["observed_at"] < cutoff:
            continue
        if (
            p.get("icing_intensity") in severe_intensities
            or p.get("turbulence_intensity") in severe_intensities
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# SIGMET / AIRMET parsing & fetching
# ---------------------------------------------------------------------------


def _parse_airsigmet_json(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert AWC SIGMET/AIRMET JSON to DB-ready dict."""
    coords = raw.get("coords") or []
    area = [{"lat": c.get("lat"), "lon": c.get("lon")} for c in coords]

    from_str = raw.get("validTimeFrom", "")
    to_str = raw.get("validTimeTo", "")

    def _dt(s: str) -> datetime | None:
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None

    return {
        "alert_id": raw.get("airSigmetId"),
        "alert_type": raw.get("airsigmetType"),
        "hazard": raw.get("hazard"),
        "severity": raw.get("severity"),
        "area": area,
        "valid_from": _dt(from_str),
        "valid_to": _dt(to_str),
        "raw_text": raw.get("rawAirSigmet"),
        "fetched_at": datetime.now(timezone.utc),
    }


async def _fetch_airsigmets(
    session: AsyncSession | None = None,
) -> list[dict[str, Any]]:
    """Fetch all active SIGMETs and AIRMETs."""
    cache_key = ("airsigmets",)
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    own_session = session is None
    if own_session:
        session = async_session()

    parsed: list[dict[str, Any]] = []

    try:
        async with _make_client() as client:
            data = await _awc_get_json(
                client, AIRSIGMET_URL, {"format": "json"}
            )

            if isinstance(data, list):
                for entry in data:
                    try:
                        p = _parse_airsigmet_json(entry)
                        parsed.append(p)
                        session.add(AviationAlert(**p))
                    except Exception:
                        logger.warning("AIRSIGMET parse failed", exc_info=True)

        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        if own_session:
            await session.close()

    _cache.set(cache_key, parsed)
    return parsed


async def fetch_active_sigmets(
    session: AsyncSession | None = None,
) -> list[dict[str, Any]]:
    """Fetch only active SIGMETs."""
    all_alerts = await _fetch_airsigmets(session)
    return [a for a in all_alerts if a.get("alert_type") == "SIGMET"]


async def fetch_active_airmets(
    session: AsyncSession | None = None,
) -> list[dict[str, Any]]:
    """Fetch only active AIRMETs."""
    all_alerts = await _fetch_airsigmets(session)
    return [a for a in all_alerts if a.get("alert_type") == "AIRMET"]


def _point_in_polygon(
    lat: float,
    lon: float,
    polygon: list[dict[str, float]],
) -> bool:
    """Ray-casting point-in-polygon test."""
    n = len(polygon)
    if n < 3:
        return False

    inside = False
    j = n - 1
    for i in range(n):
        yi = polygon[i].get("lat", 0.0)
        xi = polygon[i].get("lon", 0.0)
        yj = polygon[j].get("lat", 0.0)
        xj = polygon[j].get("lon", 0.0)

        if ((yi > lat) != (yj > lat)) and (
            lon < (xj - xi) * (lat - yi) / (yj - yi) + xi
        ):
            inside = not inside
        j = i

    return inside


async def alerts_affecting_location(
    lat: float,
    lon: float,
    session: AsyncSession | None = None,
) -> list[dict[str, Any]]:
    """Return active SIGMETs/AIRMETs whose area polygon contains the location."""
    all_alerts = await _fetch_airsigmets(session)
    now = datetime.now(timezone.utc)

    affecting: list[dict[str, Any]] = []
    for alert in all_alerts:
        valid_to = alert.get("valid_to")
        if valid_to and valid_to < now:
            continue
        area = alert.get("area", [])
        if _point_in_polygon(lat, lon, area):
            affecting.append(alert)

    return affecting


# ---------------------------------------------------------------------------
# Composite functions
# ---------------------------------------------------------------------------


async def get_aviation_weather_picture(
    station: str,
) -> dict[str, Any]:
    """Build a complete aviation weather picture for a station.

    Combines latest METAR, active TAF, temp trend, SPECI events,
    nearby PIREPs, and relevant SIGMETs.
    """
    from src.signals.mapper import geocode, icao_for_location

    # Fetch METAR + TAF concurrently
    metar_task = fetch_latest_metars([station])
    taf_task = fetch_latest_tafs([station])
    trend_task = get_temp_trend(station, hours=6)
    speci_task = detect_speci_events(station, hours=12)

    metars, tafs, trend, specis = await asyncio.gather(
        metar_task, taf_task, trend_task, speci_task
    )

    latest_metar = metars[0] if metars else {}
    latest_taf = tafs[0] if tafs else {}

    # Get PIREPs and alerts near station if we can geocode it
    pireps: list[dict[str, Any]] = []
    alerts: list[dict[str, Any]] = []
    coords = geocode(station) if len(station) == 4 else None

    # Try reverse lookup: use the city associated with this ICAO
    if coords is None:
        from src.signals.mapper import CITY_ICAO

        for city, icao in CITY_ICAO.items():
            if icao == station:
                coords = geocode(city)
                break

    if coords:
        lat, lon = coords
        pireps, alerts = await asyncio.gather(
            fetch_pireps_near(lat, lon, radius_nm=100),
            alerts_affecting_location(lat, lon),
        )

    return {
        "station": station,
        "current_temp_f": latest_metar.get("temp_f"),
        "current_conditions": latest_metar.get("raw_metar"),
        "ceiling_ft": latest_metar.get("ceiling_ft"),
        "visibility_miles": latest_metar.get("visibility_miles"),
        "flight_category": latest_metar.get("flight_category"),
        "wind": {
            "speed_kts": latest_metar.get("wind_speed_kts"),
            "dir": latest_metar.get("wind_dir"),
            "gust_kts": latest_metar.get("wind_gust_kts"),
        },
        "taf_periods": latest_taf.get("periods", []),
        "temp_trend": trend,
        "speci_events": specis,
        "nearby_pireps": pireps,
        "active_alerts": alerts,
    }





# ---------------------------------------------------------------------------
# New multi-source API (re-exports from submodules)
# ---------------------------------------------------------------------------

from src.ingestion.aviation._types import (  # noqa: E402
    MinuteObs,
    Observation,
    PrecipAccum,
    SynopObs,
    TempTrend,
    WeatherBriefing,
)
from src.ingestion.aviation._conversions import (  # noqa: E402
    c_to_f,
    f_to_c,
    kts_to_mph,
    mph_to_kts,
    kts_to_kmh,
    kts_to_ms,
    m_to_miles,
    miles_to_m,
    m_to_ft,
    hpa_to_inhg,
    inhg_to_hpa,
    hpa_to_mmhg,
    mm_to_inches,
    inches_to_mm,
    mm_to_cm,
    nm_to_km,
)
from src.ingestion.aviation._caching import (  # noqa: E402
    _TTLCache as _TTLCacheNew,
    clear_all_caches,
)
from src.ingestion.aviation._rate_limit import (  # noqa: E402
    ProviderRateLimiter,
    RateLimitExhausted,
)
from src.ingestion.aviation._base_provider import AviationProvider  # noqa: E402
from src.ingestion.aviation._aggregator import (  # noqa: E402
    ProviderAggregator,
    get_aggregator,
    reset_aggregator,
)
from src.ingestion.aviation._provider_awc import AWCProvider  # noqa: E402
from src.ingestion.aviation._parsers import (  # noqa: E402
    parse_raw_metar,
    parse_raw_synop,
)
from src.ingestion.aviation._composite import (  # noqa: E402
    get_latest_observation,
    get_latest_observations_bulk,
    get_observation_history,
    get_synop_history,
    get_one_minute_data,
    get_latest_taf,
    get_taf_amendment_count_multi,
    get_active_sigmets,
    get_pireps_near as get_pireps_near_multi,
    compute_temperature_trend,
    compute_precip_accumulation,
    get_full_weather_picture,
)
