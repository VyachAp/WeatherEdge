"""Weather Company v3 observation ingestion for airport ICAO stations.

Polls per-minute current observations, deduplicates by validTimeLocal,
tracks temperature trends, and detects peak/threshold events for
temporal edge trading.
"""

from __future__ import annotations

import asyncio
import logging
import time as _time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from sqlalchemy.exc import IntegrityError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lightweight TTL cache (avoids importing aviation package which pulls DB)
# ---------------------------------------------------------------------------


class _SimpleCache:
    def __init__(self, ttl_seconds: float) -> None:
        self._ttl = ttl_seconds
        self._store: dict = {}

    def get(self, key: Any) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        ts, value = entry
        if _time.monotonic() - ts > self._ttl:
            del self._store[key]
            return None
        return value

    def set(self, key: Any, value: Any) -> None:
        self._store[key] = (_time.monotonic(), value)

    def clear(self) -> None:
        self._store.clear()


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class _RateLimitExhausted(Exception):
    pass


class _WxRateLimiter:
    def __init__(self, max_per_second: float, daily_budget: int) -> None:
        self._interval = 1.0 / max_per_second
        self._semaphore = asyncio.Semaphore(max(1, int(max_per_second)))
        self._daily_budget = daily_budget
        self._daily_count = 0
        self._budget_reset_at = _time.monotonic() + 86400

    async def acquire(self) -> None:
        now = _time.monotonic()
        if now > self._budget_reset_at:
            self._daily_count = 0
            self._budget_reset_at = now + 86400
        if self._daily_count >= self._daily_budget:
            raise _RateLimitExhausted("WX daily budget exhausted")
        self._daily_count += 1
        await self._semaphore.acquire()

        async def _release() -> None:
            await asyncio.sleep(self._interval)
            self._semaphore.release()

        asyncio.create_task(_release())


_wx_limiter = _WxRateLimiter(
    max_per_second=settings.WX_RATE_LIMIT_RPS,
    daily_budget=settings.WX_DAILY_BUDGET,
)
_wx_cache = _SimpleCache(ttl_seconds=60)

_API_BASE = "https://api.weather.com/v3/wx/observations/current"

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


def _c_to_f(temp_c: float | None) -> float | None:
    if temp_c is None:
        return None
    return temp_c * 9.0 / 5.0 + 32.0


@dataclass(frozen=True)
class WxObservation:
    """Parsed Weather Company v3 observation."""

    station_icao: str
    valid_time_utc: datetime
    valid_time_local: str  # raw string, used as dedup key
    temp_c: float | None = None
    dewpoint_c: float | None = None
    humidity: float | None = None
    wind_speed_ms: float | None = None
    wind_gust_ms: float | None = None
    wind_dir: int | None = None
    pressure_hpa: float | None = None
    pressure_trend: str | None = None  # "Falling"/"Rising"/"Steady"
    precip_1h_mm: float | None = None
    precip_6h_mm: float | None = None
    precip_24h_mm: float | None = None
    snow_1h_mm: float | None = None
    snow_24h_mm: float | None = None
    temp_max_since_7am_c: float | None = None
    temp_max_24h_c: float | None = None
    temp_min_24h_c: float | None = None
    cloud_cover: int | None = None
    visibility_km: float | None = None
    uv_index: int | None = None

    @property
    def temp_f(self) -> float | None:
        return _c_to_f(self.temp_c)


@dataclass
class TrendAnalysis:
    """Temperature trend analysis from observation buffer."""

    current_temp_f: float
    observed_max_f: float  # Our max from stored readings (not API's)
    observed_min_f: float  # Our min from stored readings
    temp_rate_per_hour: float | None  # from linear regression
    is_rising: bool
    is_falling: bool
    peak_likely_passed: bool  # temp falling + supporting indicators
    minutes_since_last_rise: int  # how long since temp last increased
    wind_trend: str | None  # "increasing"/"decreasing"/"steady"
    humidity_trend: str | None
    pressure_trend: str | None


@dataclass
class ThresholdEvent:
    """A significant weather event detected by the WX monitor."""

    station_icao: str
    event_type: str  # "threshold_crossed", "peak_likely_done"
    temp_f: float
    threshold_f: float | None  # for threshold_crossed events
    confidence: float  # 0-1
    detail: str  # human-readable explanation


# ---------------------------------------------------------------------------
# In-memory rolling buffer
# ---------------------------------------------------------------------------

_observation_buffer: dict[str, deque[WxObservation]] = {}
_BUFFER_MAXLEN = 120  # ~10 hours of 5-min updates

# Track last valid_time_local per station for dedup
_last_valid_time: dict[str, str] = {}


def _buffer_append(obs: WxObservation) -> bool:
    """Append observation to buffer. Returns True if new (not a duplicate)."""
    icao = obs.station_icao
    last = _last_valid_time.get(icao)
    if last == obs.valid_time_local:
        return False  # Duplicate — same validTimeLocal

    _last_valid_time[icao] = obs.valid_time_local
    if icao not in _observation_buffer:
        _observation_buffer[icao] = deque(maxlen=_BUFFER_MAXLEN)
    _observation_buffer[icao].append(obs)
    return True


def get_buffer_history(icao: str, count: int | None = None) -> list[WxObservation]:
    """Return observations from buffer. None = all, or last N."""
    buf = _observation_buffer.get(icao)
    if not buf:
        return []
    if count is None:
        return list(buf)
    return list(buf)[-count:]


def clear_buffers() -> None:
    _observation_buffer.clear()
    _last_valid_time.clear()


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------


def _make_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={"User-Agent": "WeatherEdge/1.0", "Accept-Encoding": "gzip"},
        timeout=15,
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
    reraise=True,
)
async def _wx_get_json(client: httpx.AsyncClient, icao: str) -> dict[str, Any] | None:
    await _wx_limiter.acquire()
    resp = await client.get(
        _API_BASE,
        params={
            "apiKey": settings.WX_API_KEY,
            "icaoCode": icao,
            "units": "m",
            "format": "json",
            "language": "en-US",
        },
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_observation(icao: str, data: dict[str, Any]) -> WxObservation | None:
    """Parse a v3 observation JSON into WxObservation."""
    valid_utc_epoch = data.get("validTimeUtc")
    valid_local = data.get("validTimeLocal")
    if not valid_utc_epoch or not valid_local:
        return None

    try:
        valid_dt = datetime.fromtimestamp(valid_utc_epoch, tz=timezone.utc)
    except (ValueError, TypeError, OSError):
        return None

    return WxObservation(
        station_icao=icao,
        valid_time_utc=valid_dt,
        valid_time_local=valid_local,
        temp_c=data.get("temperature"),
        dewpoint_c=data.get("temperatureDewPoint"),
        humidity=data.get("relativeHumidity"),
        wind_speed_ms=data.get("windSpeed"),
        wind_gust_ms=data.get("windGust"),
        wind_dir=data.get("windDirection"),
        pressure_hpa=data.get("pressureMeanSeaLevel"),
        pressure_trend=data.get("pressureTendencyTrend"),
        precip_1h_mm=data.get("precip1Hour"),
        precip_6h_mm=data.get("precip6Hour"),
        precip_24h_mm=data.get("precip24Hour"),
        snow_1h_mm=data.get("snow1Hour"),
        snow_24h_mm=data.get("snow24Hour"),
        temp_max_since_7am_c=data.get("temperatureMaxSince7Am"),
        temp_max_24h_c=data.get("temperatureMax24Hour"),
        temp_min_24h_c=data.get("temperatureMin24Hour"),
        cloud_cover=data.get("cloudCover"),
        visibility_km=data.get("visibility"),
        uv_index=data.get("uvIndex"),
    )


# ---------------------------------------------------------------------------
# Core fetch
# ---------------------------------------------------------------------------


async def fetch_wx_current(icao: str) -> WxObservation | None:
    """Fetch current observation for an ICAO station.

    Returns None on failure. Cached for 60 seconds.
    """
    if not settings.WX_API_KEY:
        return None

    cached = _wx_cache.get(("wx", icao))
    if cached is not None:
        return cached

    try:
        async with _make_client() as client:
            data = await _wx_get_json(client, icao)
    except (httpx.HTTPError, _RateLimitExhausted) as exc:
        logger.warning("WX fetch failed for %s: %s", icao, exc)
        return None
    except Exception:
        logger.exception("Unexpected error fetching WX %s", icao)
        return None

    if not data:
        return None

    obs = _parse_observation(icao, data)
    if obs is None:
        return None

    _wx_cache.set(("wx", icao), obs)
    return obs


async def poll_and_store(icao: str, session: Any) -> WxObservation | None:
    """Fetch, dedup, persist to DB, and append to buffer.

    Returns the observation if it was new (not a duplicate), None otherwise.
    """
    obs = await fetch_wx_current(icao)
    if obs is None:
        return None

    is_new = _buffer_append(obs)
    if not is_new:
        return None  # Already have this validTimeLocal

    # Persist to DB
    from src.db.models import WxObservation as WxObservationModel

    row = WxObservationModel(
        station_icao=obs.station_icao,
        valid_time_utc=obs.valid_time_utc,
        valid_time_local=obs.valid_time_local,
        temp_c=obs.temp_c,
        dewpoint_c=obs.dewpoint_c,
        humidity=obs.humidity,
        wind_speed_ms=obs.wind_speed_ms,
        wind_gust_ms=obs.wind_gust_ms,
        wind_dir=obs.wind_dir,
        pressure_hpa=obs.pressure_hpa,
        pressure_trend=obs.pressure_trend,
        precip_1h_mm=obs.precip_1h_mm,
        precip_6h_mm=obs.precip_6h_mm,
        precip_24h_mm=obs.precip_24h_mm,
        snow_1h_mm=obs.snow_1h_mm,
        snow_24h_mm=obs.snow_24h_mm,
        temp_max_since_7am_c=obs.temp_max_since_7am_c,
        temp_max_24h_c=obs.temp_max_24h_c,
        temp_min_24h_c=obs.temp_min_24h_c,
        cloud_cover=obs.cloud_cover,
        visibility_km=obs.visibility_km,
        uv_index=obs.uv_index,
    )
    session.add(row)
    try:
        await session.flush()
    except IntegrityError:
        await session.rollback()
        logger.debug("WX dup skipped %s valid=%s", icao, obs.valid_time_local)
        return None

    logger.debug(
        "WX new obs %s: temp=%.1f°C valid=%s",
        icao,
        obs.temp_c if obs.temp_c is not None else float("nan"),
        obs.valid_time_local,
    )
    return obs


async def poll_stations(icaos: list[str], session: Any) -> dict[str, WxObservation]:
    """Poll multiple stations concurrently. Returns dict of icao→new observation."""
    if not icaos or not settings.WX_API_KEY:
        return {}

    results = await asyncio.gather(
        *(poll_and_store(icao, session) for icao in icaos),
        return_exceptions=True,
    )

    new_obs: dict[str, WxObservation] = {}
    for icao, result in zip(icaos, results):
        if isinstance(result, WxObservation):
            new_obs[icao] = result
    return new_obs


# ---------------------------------------------------------------------------
# Trend analysis
# ---------------------------------------------------------------------------


def _compute_rate(history: list[WxObservation]) -> float | None:
    """Linear regression on temperature over time → F/hour."""
    temps_f: list[float] = []
    times_min: list[float] = []

    if len(history) < 2:
        return None

    t0 = history[0].valid_time_utc
    for obs in history:
        if obs.temp_c is not None:
            temps_f.append(obs.temp_c * 9.0 / 5.0 + 32.0)
            times_min.append((obs.valid_time_utc - t0).total_seconds() / 60.0)

    if len(temps_f) < 2:
        return None

    n = len(temps_f)
    mean_t = sum(times_min) / n
    mean_temp = sum(temps_f) / n
    numerator = sum((t - mean_t) * (tmp - mean_temp) for t, tmp in zip(times_min, temps_f))
    denominator = sum((t - mean_t) ** 2 for t in times_min)
    if denominator == 0:
        return 0.0

    slope_per_min = numerator / denominator
    return slope_per_min * 60.0  # F/hour


def _simple_trend(values: list[float | None]) -> str | None:
    """Determine 'increasing'/'decreasing'/'steady' from recent values."""
    nums = [v for v in values if v is not None]
    if len(nums) < 2:
        return None
    first_half = sum(nums[: len(nums) // 2]) / max(1, len(nums) // 2)
    second_half = sum(nums[len(nums) // 2 :]) / max(1, len(nums) - len(nums) // 2)
    diff = second_half - first_half
    if diff > 0.5:
        return "increasing"
    elif diff < -0.5:
        return "decreasing"
    return "steady"


def analyze_trend(icao: str) -> TrendAnalysis | None:
    """Analyze temperature trend from observation buffer.

    Returns None if insufficient data.
    """
    history = get_buffer_history(icao)
    temps_f = [(obs, obs.temp_c * 9.0 / 5.0 + 32.0) for obs in history if obs.temp_c is not None]

    if len(temps_f) < 3:
        return None

    current_obs, current_f = temps_f[-1]
    observed_max_f = max(t for _, t in temps_f)
    observed_min_f = min(t for _, t in temps_f)

    # Rate from last 6 readings (~30 min of 5-min updates)
    recent = [obs for obs in history if obs.temp_c is not None][-6:]
    rate = _compute_rate(recent)

    is_rising = rate is not None and rate > 0.5
    is_falling = rate is not None and rate < -0.5

    # Minutes since last temperature rise
    minutes_since_rise = 0
    for i in range(len(temps_f) - 1, 0, -1):
        if temps_f[i][1] > temps_f[i - 1][1]:
            break
        dt = (temps_f[i][0].valid_time_utc - temps_f[i - 1][0].valid_time_utc).total_seconds()
        minutes_since_rise += dt / 60.0

    # Peak detection: temp falling + supporting indicators
    peak_confirm_min = settings.WX_PEAK_CONFIRM_MINUTES
    below_max = current_f < observed_max_f - 0.5  # At least 0.5F below peak

    supporting_indicators = 0
    # Wind increasing?
    wind_trend = _simple_trend([obs.wind_speed_ms for obs in recent])
    if wind_trend == "increasing":
        supporting_indicators += 1
    # Humidity rising?
    humidity_trend = _simple_trend([obs.humidity for obs in recent])
    if humidity_trend == "increasing":
        supporting_indicators += 1
    # Pressure falling?
    if current_obs.pressure_trend == "Falling":
        supporting_indicators += 1
    # Past 15:00 local? (check from valid_time_local string)
    try:
        local_hour = int(current_obs.valid_time_local.split(" ")[1].split(":")[0])
        if local_hour >= 15:
            supporting_indicators += 1
    except (IndexError, ValueError):
        pass

    peak_likely_passed = (
        is_falling
        and minutes_since_rise >= peak_confirm_min
        and below_max
        and supporting_indicators >= 1
    )

    return TrendAnalysis(
        current_temp_f=current_f,
        observed_max_f=observed_max_f,
        observed_min_f=observed_min_f,
        temp_rate_per_hour=rate,
        is_rising=is_rising,
        is_falling=is_falling,
        peak_likely_passed=peak_likely_passed,
        minutes_since_last_rise=int(minutes_since_rise),
        wind_trend=wind_trend,
        humidity_trend=humidity_trend,
        pressure_trend=current_obs.pressure_trend,
    )


# ---------------------------------------------------------------------------
# Threshold event detection
# ---------------------------------------------------------------------------


def detect_threshold_events(
    icao: str,
    thresholds: list[tuple[float, str]],
) -> list[ThresholdEvent]:
    """Detect if temperature has crossed thresholds or peak is established.

    *thresholds* is a list of (threshold_f, operator) pairs where operator
    is "above" or "below".

    Returns events for:
    - threshold_crossed: observed temp has crossed the threshold
    - peak_likely_done: peak analysis suggests the high/low is established
    """
    trend = analyze_trend(icao)
    if trend is None:
        return []

    events: list[ThresholdEvent] = []

    for threshold_f, operator in thresholds:
        if operator == "above":
            # Has temp crossed above threshold?
            if trend.observed_max_f >= threshold_f:
                confidence = 0.95 if trend.current_temp_f >= threshold_f else 0.85
                events.append(ThresholdEvent(
                    station_icao=icao,
                    event_type="threshold_crossed",
                    temp_f=trend.observed_max_f,
                    threshold_f=threshold_f,
                    confidence=confidence,
                    detail=(
                        f"Temp reached {trend.observed_max_f:.1f}F "
                        f"(threshold {threshold_f:.0f}F). "
                        f"Current: {trend.current_temp_f:.1f}F"
                    ),
                ))
            # Is peak done below threshold? → strong NO signal
            elif trend.peak_likely_passed and trend.observed_max_f < threshold_f:
                gap = threshold_f - trend.observed_max_f
                confidence = min(0.90, 0.70 + 0.05 * trend.minutes_since_last_rise / 15)
                events.append(ThresholdEvent(
                    station_icao=icao,
                    event_type="peak_likely_done",
                    temp_f=trend.observed_max_f,
                    threshold_f=threshold_f,
                    confidence=confidence,
                    detail=(
                        f"Peak likely {trend.observed_max_f:.1f}F, "
                        f"{gap:.1f}F below threshold {threshold_f:.0f}F. "
                        f"Falling for {trend.minutes_since_last_rise}min. "
                        f"Indicators: wind={trend.wind_trend}, "
                        f"humidity={trend.humidity_trend}, "
                        f"pressure={trend.pressure_trend}"
                    ),
                ))

        elif operator == "below":
            # Has temp dropped below threshold?
            if trend.observed_min_f <= threshold_f:
                confidence = 0.95 if trend.current_temp_f <= threshold_f else 0.85
                events.append(ThresholdEvent(
                    station_icao=icao,
                    event_type="threshold_crossed",
                    temp_f=trend.observed_min_f,
                    threshold_f=threshold_f,
                    confidence=confidence,
                    detail=(
                        f"Temp dropped to {trend.observed_min_f:.1f}F "
                        f"(threshold {threshold_f:.0f}F). "
                        f"Current: {trend.current_temp_f:.1f}F"
                    ),
                ))

    return events
