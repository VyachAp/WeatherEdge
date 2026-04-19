"""WU confirmation pipeline for temperature peak validation.

After the WX API detects a temperature peak (two consecutive readings
≥1°F below the observed max), this module scrapes Weather Underground
to obtain the "official" high as displayed in the UI — the data source
Polymarket uses for market resolution.

The pipeline tracks per-station state through:
  MONITORING → DECREASE_DETECTED → WU_CONFIRMED / WU_WAITING → ALERTED
"""

from __future__ import annotations

import enum
import logging
import re
import time as _time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from src.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class ConfirmationState(str, enum.Enum):
    MONITORING = "monitoring"
    DECREASE_DETECTED = "decrease_detected"
    WU_PENDING = "wu_pending"
    WU_WAITING = "wu_waiting"
    WU_CONFIRMED = "wu_confirmed"
    ALERTED = "alerted"


@dataclass
class StationConfirmation:
    """Tracks confirmation state for a single ICAO station on a given day."""

    icao: str
    state: ConfirmationState = ConfirmationState.MONITORING
    peak_temp_f: float = -999.0
    peak_time_utc: datetime | None = None
    peak_time_local: str | None = None  # e.g. "2026-04-19T15:23:00-0500"
    consecutive_drops: int = 0
    wu_high_f: float | None = None
    wu_latest_hour: str | None = None  # latest WU hourly time e.g. "1:53 PM"
    wu_scrape_count: int = 0
    last_wu_scrape_at: float | None = None  # monotonic timestamp
    db_vs_wu_delta: float | None = None
    created_date: date = field(default_factory=lambda: date.today())
    _wu_waiting_alerted: bool = field(default=False, repr=False)


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

_TIME_RE = re.compile(r"(\d{1,2}):(\d{2})\s*([APap][Mm])")


def _parse_wu_time(time_str: str) -> tuple[int, int] | None:
    """Parse a WU hourly time like '2:53 PM' → (14, 53)."""
    m = _TIME_RE.search(time_str)
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2))
    ampm = m.group(3).upper()
    if ampm == "AM" and hour == 12:
        hour = 0
    elif ampm == "PM" and hour != 12:
        hour += 12
    return (hour, minute)


def _extract_local_hour_minute(valid_time_local: str) -> tuple[int, int] | None:
    """Extract hour and minute from WX observation's valid_time_local.

    The format is ISO-like: '2026-04-19T15:23:00-0500' or similar.
    """
    try:
        # Try parsing the T-separated time portion
        time_part = valid_time_local.split("T")[1] if "T" in valid_time_local else valid_time_local.split(" ")[1]
        parts = time_part.split(":")
        return (int(parts[0]), int(parts[1]))
    except (IndexError, ValueError):
        return None


def _wu_covers_peak(
    hourly_readings: list,
    peak_local_hm: tuple[int, int],
) -> bool:
    """Check if the WU hourly data extends past the peak local time."""
    if not hourly_readings:
        return False

    latest_hm: tuple[int, int] | None = None
    for reading in hourly_readings:
        hm = _parse_wu_time(reading.time_local)
        if hm is not None:
            if latest_hm is None or hm > latest_hm:
                latest_hm = hm

    if latest_hm is None:
        return False

    return latest_hm >= peak_local_hm


# ---------------------------------------------------------------------------
# Confirmation tracker
# ---------------------------------------------------------------------------


class ConfirmationTracker:
    """In-memory state machine tracking temperature confirmation per station."""

    def __init__(self) -> None:
        self._trackers: dict[str, StationConfirmation] = {}

    def _get_or_create(self, icao: str) -> StationConfirmation:
        sc = self._trackers.get(icao)
        today = date.today()
        if sc is None or sc.created_date != today:
            sc = StationConfirmation(icao=icao, created_date=today)
            self._trackers[icao] = sc
        return sc

    def on_new_observation(self, icao: str, obs: object) -> ConfirmationState | None:
        """Process a new WxObservation. Returns new state on transition, else None."""
        temp_f = getattr(obs, "temp_f", None)
        if temp_f is None:
            return None

        sc = self._get_or_create(icao)

        # Already past monitoring — don't re-trigger
        if sc.state not in (ConfirmationState.MONITORING,):
            return None

        # New peak
        if temp_f > sc.peak_temp_f:
            sc.peak_temp_f = temp_f
            sc.peak_time_utc = getattr(obs, "valid_time_utc", None)
            sc.peak_time_local = getattr(obs, "valid_time_local", None)
            sc.consecutive_drops = 0
            return None

        # Check for drop ≥ configured threshold below peak
        drop = sc.peak_temp_f - temp_f
        if drop >= settings.WU_CONFIRM_DROP_F:
            sc.consecutive_drops += 1
        else:
            sc.consecutive_drops = 0

        if sc.consecutive_drops >= 2:
            sc.state = ConfirmationState.DECREASE_DETECTED
            logger.info(
                "Confirmation: %s decrease detected — peak %.1f°F, "
                "current %.1f°F, 2 consecutive drops",
                icao, sc.peak_temp_f, temp_f,
            )
            return ConfirmationState.DECREASE_DETECTED

        return None

    async def check_wu_confirmation(self, icao: str) -> ConfirmationState:
        """Scrape WU history and check if data covers the peak time."""
        from src.ingestion.wu_history import fetch_wu_history

        sc = self._get_or_create(icao)
        sc.wu_scrape_count += 1
        sc.last_wu_scrape_at = _time.monotonic()

        wu = await fetch_wu_history(icao, sc.created_date)
        if wu is None:
            sc.state = ConfirmationState.WU_WAITING
            logger.warning("Confirmation: WU scrape returned None for %s", icao)
            return sc.state

        # Find the WU max temperature from hourly readings
        wu_hourly_max: float | None = None
        if wu.hourly:
            temps = [h.temp_f for h in wu.hourly if h.temp_f is not None]
            if temps:
                wu_hourly_max = max(temps)
            sc.wu_latest_hour = wu.hourly[-1].time_local if wu.hourly else None

        # Use the summary high if available, fall back to hourly max
        sc.wu_high_f = wu.high_f if wu.high_f is not None else wu_hourly_max

        # Compute delta
        if sc.wu_high_f is not None:
            sc.db_vs_wu_delta = sc.peak_temp_f - sc.wu_high_f

        # Check if WU hourly data covers our peak time
        peak_hm = _extract_local_hour_minute(sc.peak_time_local) if sc.peak_time_local else None
        if peak_hm is None:
            # Can't determine local time — treat as covered if we have any data
            if sc.wu_high_f is not None:
                sc.state = ConfirmationState.WU_CONFIRMED
            else:
                sc.state = ConfirmationState.WU_WAITING
        elif wu.hourly and _wu_covers_peak(wu.hourly, peak_hm):
            sc.state = ConfirmationState.WU_CONFIRMED
            logger.info(
                "Confirmation: %s WU confirmed — WU high=%.1f°F, "
                "DB peak=%.1f°F, delta=%+.1f°F",
                icao,
                sc.wu_high_f or 0,
                sc.peak_temp_f,
                sc.db_vs_wu_delta or 0,
            )
        else:
            sc.state = ConfirmationState.WU_WAITING
            logger.info(
                "Confirmation: %s WU data not yet up to peak time (%s), "
                "latest WU hour: %s",
                icao, sc.peak_time_local, sc.wu_latest_hour,
            )

        return sc.state

    def get_stations_needing_wu_retry(self) -> list[str]:
        """Return station ICAOs in WU_WAITING that are due for a re-scrape."""
        now = _time.monotonic()
        retry_seconds = settings.WU_CONFIRM_RETRY_MINUTES * 60
        result: list[str] = []

        for icao, sc in self._trackers.items():
            if sc.state != ConfirmationState.WU_WAITING:
                continue
            if sc.wu_scrape_count >= settings.WU_CONFIRM_MAX_SCRAPES:
                continue
            if sc.last_wu_scrape_at is not None and (now - sc.last_wu_scrape_at) < retry_seconds:
                continue
            result.append(icao)

        return result

    async def match_markets(self, icao: str, session: object) -> list:
        """Find Polymarket markets matching the confirmed WU temperature."""
        from src.signals.reverse_lookup import find_markets_for_observation

        sc = self._trackers.get(icao)
        if sc is None or sc.wu_high_f is None:
            return []

        matches = await find_markets_for_observation(
            session, icao, sc.wu_high_f, hours_ahead=48.0,
        )
        return matches

    def get(self, icao: str) -> StationConfirmation | None:
        return self._trackers.get(icao)

    def all_stations(self) -> list[StationConfirmation]:
        return list(self._trackers.values())

    def reset(self) -> None:
        self._trackers.clear()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_tracker: ConfirmationTracker | None = None


def get_confirmation_tracker() -> ConfirmationTracker:
    global _tracker
    if _tracker is None:
        _tracker = ConfirmationTracker()
    return _tracker
