"""Unit tests for scheduler-level helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from src.scheduler import _minimal_state_for_easy_lock, _should_skip_future_day
from src.signals.lock_rules import evaluate_lock


def _market(end_date):
    return SimpleNamespace(end_date=end_date, id="m1")


def test_skip_future_day_resolves_tomorrow():
    now = datetime(2026, 4, 22, 15, 0, tzinfo=timezone.utc)
    market = _market(datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc))
    assert _should_skip_future_day(market, now) is True


def test_skip_future_day_same_day_evaluated():
    now = datetime(2026, 4, 22, 15, 0, tzinfo=timezone.utc)
    # Resolves later today UTC.
    market = _market(datetime(2026, 4, 22, 23, 30, tzinfo=timezone.utc))
    assert _should_skip_future_day(market, now) is False


def test_skip_future_day_past_day_evaluated():
    # A market whose end_date already passed is still "not future" by this
    # rule; other filters (close-buffer, near-resolved price) handle it.
    now = datetime(2026, 4, 22, 15, 0, tzinfo=timezone.utc)
    market = _market(datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc))
    assert _should_skip_future_day(market, now) is False


def test_skip_future_day_no_end_date_evaluated():
    now = datetime(2026, 4, 22, 15, 0, tzinfo=timezone.utc)
    market = _market(None)
    assert _should_skip_future_day(market, now) is False


def test_skip_future_day_uses_utc_calendar_date():
    # End_date is 23:00Z on Apr 22; "now" is 23:30Z on Apr 22. Same UTC day,
    # not skipped — even though local-time semantics elsewhere may vary.
    now = datetime(2026, 4, 22, 23, 30, tzinfo=timezone.utc)
    market = _market(datetime(2026, 4, 22, 23, 0, tzinfo=timezone.utc))
    assert _should_skip_future_day(market, now) is False


class TestMinimalStateForEasyLock:
    """Fast-poll path builds a trimmed WeatherState — only routine_history
    and station_icao are read by evaluate_lock's EASY branch."""

    def test_populates_max_and_count(self):
        # Use a city/ICAO where local-day anchoring is predictable.
        now = datetime(2026, 4, 23, 18, 0, tzinfo=timezone.utc)
        points = [
            (now - timedelta(hours=3), 62.0),
            (now - timedelta(hours=2), 68.0),
            (now - timedelta(hours=1), 71.0),
        ]
        state = _minimal_state_for_easy_lock("KJFK", points)
        assert state.station_icao == "KJFK"
        assert state.current_max_f == 71.0
        assert state.routine_count_today == 3
        assert state.has_forecast is False
        assert len(state.routine_history) == 3

    def test_sorts_routine_history_ascending(self):
        now = datetime(2026, 4, 23, 18, 0, tzinfo=timezone.utc)
        points = [
            (now, 70.0),
            (now - timedelta(hours=4), 55.0),
            (now - timedelta(hours=2), 65.0),
        ]
        state = _minimal_state_for_easy_lock("KJFK", points)
        times = [t for t, _ in state.routine_history]
        assert times == sorted(times)

    def test_triggers_easy_lock_when_threshold_cleared(self):
        # Obs max 72°F vs threshold 68°F + 2°F margin → YES is physically locked.
        now = datetime(2026, 4, 23, 18, 0, tzinfo=timezone.utc)
        points = [
            (now - timedelta(hours=3), 60.0),
            (now - timedelta(hours=2), 67.0),
            (now - timedelta(hours=1), 72.0),
        ]
        state = _minimal_state_for_easy_lock("KJFK", points)
        market = SimpleNamespace(
            id="m1",
            parsed_threshold=68,
            parsed_operator="above",
            end_date=now + timedelta(hours=4),
        )
        decision = evaluate_lock(state, market, now_utc=now)
        assert decision.side == "YES"
        assert decision.margin_f == 4.0

    def test_below_threshold_no_lock_fires_from_fast_path(self):
        # obs max 66 < threshold 70; HARD direction needs forecast/solar
        # context which the minimal state deliberately lacks, so lock is None.
        now = datetime(2026, 4, 23, 18, 0, tzinfo=timezone.utc)
        points = [
            (now - timedelta(hours=2), 60.0),
            (now - timedelta(hours=1), 66.0),
        ]
        state = _minimal_state_for_easy_lock("KJFK", points)
        market = SimpleNamespace(
            id="m1",
            parsed_threshold=70,
            parsed_operator="above",
            end_date=now + timedelta(hours=4),
        )
        decision = evaluate_lock(state, market, now_utc=now)
        assert decision.side is None
