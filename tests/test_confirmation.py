"""Tests for the WU confirmation pipeline."""

from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ingestion.confirmation import (
    ConfirmationState,
    ConfirmationTracker,
    StationConfirmation,
    _extract_local_hour_minute,
    _parse_wu_time,
    _wu_covers_peak,
)


# ---------------------------------------------------------------------------
# Time parsing helpers
# ---------------------------------------------------------------------------


class TestParseWuTime:
    def test_pm(self):
        assert _parse_wu_time("2:53 PM") == (14, 53)

    def test_am(self):
        assert _parse_wu_time("9:15 AM") == (9, 15)

    def test_noon(self):
        assert _parse_wu_time("12:00 PM") == (12, 0)

    def test_midnight(self):
        assert _parse_wu_time("12:00 AM") == (0, 0)

    def test_lowercase(self):
        assert _parse_wu_time("3:30 pm") == (15, 30)

    def test_invalid(self):
        assert _parse_wu_time("invalid") is None

    def test_empty(self):
        assert _parse_wu_time("") is None


class TestExtractLocalHourMinute:
    def test_iso_format(self):
        assert _extract_local_hour_minute("2026-04-19T15:23:00-0500") == (15, 23)

    def test_space_separated(self):
        assert _extract_local_hour_minute("2026-04-19 14:05:00") == (14, 5)

    def test_invalid(self):
        assert _extract_local_hour_minute("not-a-date") is None


class TestWuCoversPeak:
    def _reading(self, time_str: str):
        r = MagicMock()
        r.time_local = time_str
        return r

    def test_covers(self):
        readings = [self._reading("1:00 PM"), self._reading("2:00 PM"), self._reading("3:30 PM")]
        assert _wu_covers_peak(readings, (15, 0)) is True

    def test_not_covers(self):
        readings = [self._reading("1:00 PM"), self._reading("2:00 PM")]
        assert _wu_covers_peak(readings, (15, 0)) is False

    def test_exact_match(self):
        readings = [self._reading("3:00 PM")]
        assert _wu_covers_peak(readings, (15, 0)) is True

    def test_empty(self):
        assert _wu_covers_peak([], (15, 0)) is False


# ---------------------------------------------------------------------------
# Mock observation helper
# ---------------------------------------------------------------------------


def _obs(temp_f: float, time_local: str = "2026-04-19T14:00:00-0500"):
    obs = MagicMock()
    obs.temp_f = temp_f
    obs.valid_time_utc = datetime(2026, 4, 19, 19, 0, tzinfo=timezone.utc)
    obs.valid_time_local = time_local
    return obs


# ---------------------------------------------------------------------------
# ConfirmationTracker.on_new_observation
# ---------------------------------------------------------------------------


class TestOnNewObservation:
    def setup_method(self):
        self.tracker = ConfirmationTracker()

    def test_tracks_peak(self):
        self.tracker.on_new_observation("KAUS", _obs(80.0))
        self.tracker.on_new_observation("KAUS", _obs(85.0))
        sc = self.tracker.get("KAUS")
        assert sc.peak_temp_f == 85.0

    def test_no_transition_on_first_obs(self):
        result = self.tracker.on_new_observation("KAUS", _obs(80.0))
        assert result is None

    def test_single_drop_no_transition(self):
        self.tracker.on_new_observation("KAUS", _obs(85.0))
        result = self.tracker.on_new_observation("KAUS", _obs(83.0))
        assert result is None
        sc = self.tracker.get("KAUS")
        assert sc.consecutive_drops == 1
        assert sc.state == ConfirmationState.MONITORING

    def test_two_consecutive_drops_triggers(self):
        self.tracker.on_new_observation("KAUS", _obs(85.0))
        self.tracker.on_new_observation("KAUS", _obs(83.5))  # drop 1
        result = self.tracker.on_new_observation("KAUS", _obs(83.0))  # drop 2
        assert result == ConfirmationState.DECREASE_DETECTED
        sc = self.tracker.get("KAUS")
        assert sc.state == ConfirmationState.DECREASE_DETECTED

    def test_drop_then_rise_resets(self):
        self.tracker.on_new_observation("KAUS", _obs(85.0))
        self.tracker.on_new_observation("KAUS", _obs(83.5))  # drop 1
        self.tracker.on_new_observation("KAUS", _obs(84.5))  # not a drop (within 1F)
        sc = self.tracker.get("KAUS")
        assert sc.consecutive_drops == 0

    def test_new_peak_resets_drops(self):
        self.tracker.on_new_observation("KAUS", _obs(85.0))
        self.tracker.on_new_observation("KAUS", _obs(83.5))  # drop 1
        self.tracker.on_new_observation("KAUS", _obs(86.0))  # new peak
        sc = self.tracker.get("KAUS")
        assert sc.consecutive_drops == 0
        assert sc.peak_temp_f == 86.0

    def test_drop_less_than_threshold_resets(self):
        """A reading that's below peak but less than 1F below doesn't count."""
        self.tracker.on_new_observation("KAUS", _obs(85.0))
        self.tracker.on_new_observation("KAUS", _obs(84.5))  # only 0.5F below — not enough
        sc = self.tracker.get("KAUS")
        assert sc.consecutive_drops == 0

    def test_exactly_1f_below_counts(self):
        self.tracker.on_new_observation("KAUS", _obs(85.0))
        self.tracker.on_new_observation("KAUS", _obs(84.0))  # exactly 1F below
        sc = self.tracker.get("KAUS")
        assert sc.consecutive_drops == 1

    def test_no_re_trigger_after_decrease_detected(self):
        self.tracker.on_new_observation("KAUS", _obs(85.0))
        self.tracker.on_new_observation("KAUS", _obs(83.5))
        self.tracker.on_new_observation("KAUS", _obs(83.0))  # triggers
        result = self.tracker.on_new_observation("KAUS", _obs(82.0))  # already detected
        assert result is None

    def test_none_temp_ignored(self):
        obs = MagicMock()
        obs.temp_f = None
        result = self.tracker.on_new_observation("KAUS", obs)
        assert result is None

    def test_daily_reset(self):
        self.tracker.on_new_observation("KAUS", _obs(85.0))
        sc = self.tracker.get("KAUS")
        # Simulate yesterday's date
        sc.created_date = date(2020, 1, 1)
        self.tracker.on_new_observation("KAUS", _obs(70.0))
        sc = self.tracker.get("KAUS")
        assert sc.peak_temp_f == 70.0  # fresh state


# ---------------------------------------------------------------------------
# WU confirmation
# ---------------------------------------------------------------------------


class TestCheckWuConfirmation:
    def setup_method(self):
        self.tracker = ConfirmationTracker()
        # Set up a station in DECREASE_DETECTED state
        self.tracker.on_new_observation("KAUS", _obs(85.0, "2026-04-19T14:00:00-0500"))
        self.tracker.on_new_observation("KAUS", _obs(83.5, "2026-04-19T14:05:00-0500"))
        self.tracker.on_new_observation("KAUS", _obs(83.0, "2026-04-19T14:10:00-0500"))

    @pytest.mark.asyncio
    async def test_wu_confirmed(self):
        wu = MagicMock()
        wu.high_f = 84.0
        reading = MagicMock()
        reading.time_local = "3:00 PM"
        reading.temp_f = 84.0
        wu.hourly = [reading]

        with patch("src.ingestion.wu_history.fetch_wu_history", new_callable=AsyncMock, return_value=wu):
            state = await self.tracker.check_wu_confirmation("KAUS")

        assert state == ConfirmationState.WU_CONFIRMED
        sc = self.tracker.get("KAUS")
        assert sc.wu_high_f == 84.0
        assert sc.db_vs_wu_delta == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_wu_waiting_data_lag(self):
        wu = MagicMock()
        wu.high_f = 80.0
        reading = MagicMock()
        reading.time_local = "12:00 PM"  # only up to noon, peak was at 14:00
        reading.temp_f = 80.0
        wu.hourly = [reading]

        with patch("src.ingestion.wu_history.fetch_wu_history", new_callable=AsyncMock, return_value=wu):
            state = await self.tracker.check_wu_confirmation("KAUS")

        assert state == ConfirmationState.WU_WAITING

    @pytest.mark.asyncio
    async def test_wu_scrape_failed(self):
        with patch("src.ingestion.wu_history.fetch_wu_history", new_callable=AsyncMock, return_value=None):
            state = await self.tracker.check_wu_confirmation("KAUS")

        assert state == ConfirmationState.WU_WAITING


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


class TestRetryLogic:
    def setup_method(self):
        self.tracker = ConfirmationTracker()

    def test_no_retry_if_not_waiting(self):
        self.tracker.on_new_observation("KAUS", _obs(85.0))
        assert self.tracker.get_stations_needing_wu_retry() == []

    def test_retry_after_interval(self):
        self.tracker.on_new_observation("KAUS", _obs(85.0))
        sc = self.tracker.get("KAUS")
        sc.state = ConfirmationState.WU_WAITING
        sc.wu_scrape_count = 1
        sc.last_wu_scrape_at = 0  # long ago
        assert "KAUS" in self.tracker.get_stations_needing_wu_retry()

    def test_no_retry_too_soon(self):
        import time

        self.tracker.on_new_observation("KAUS", _obs(85.0))
        sc = self.tracker.get("KAUS")
        sc.state = ConfirmationState.WU_WAITING
        sc.wu_scrape_count = 1
        sc.last_wu_scrape_at = time.monotonic()  # just scraped
        assert self.tracker.get_stations_needing_wu_retry() == []

    def test_no_retry_max_scrapes(self):
        self.tracker.on_new_observation("KAUS", _obs(85.0))
        sc = self.tracker.get("KAUS")
        sc.state = ConfirmationState.WU_WAITING
        sc.wu_scrape_count = 8  # max reached
        sc.last_wu_scrape_at = 0
        assert self.tracker.get_stations_needing_wu_retry() == []
