"""Unit tests for scheduler-level helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from src.scheduler import _should_skip_future_day


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
