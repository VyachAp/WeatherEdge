"""Regression tests for market data-day resolution.

These cover the 2026-05-26 fix: ``resolve_target_local_day`` must return the
title day (= ``end_date``'s UTC date) for BOTH hemispheres. The prior
12h-backstep-then-localize returned title-day−1 for negative-UTC cities, which
made ``should_skip_future_day`` fail to skip the next-day Americas market — the
bot was betting tomorrow's São Paulo market a day early with today's state.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from src.execution.binary_market import should_skip_future_day
from src.signals.mapper import resolve_target_local_day

_NOON = lambda y, m, d: datetime(y, m, d, 12, 0, tzinfo=timezone.utc)  # noqa: E731


class TestResolveTargetLocalDay:
    def test_sao_paulo_noon_utc_is_title_day(self):
        # WAS date(2026, 5, 26) under the buggy backstep.
        assert resolve_target_local_day(
            _NOON(2026, 5, 27), ZoneInfo("America/Sao_Paulo")
        ) == date(2026, 5, 27)

    def test_buenos_aires_noon_utc_is_title_day(self):
        assert resolve_target_local_day(
            _NOON(2026, 5, 27), ZoneInfo("America/Argentina/Buenos_Aires")
        ) == date(2026, 5, 27)

    def test_atlanta_noon_utc_is_title_day(self):
        assert resolve_target_local_day(
            _NOON(2026, 5, 27), ZoneInfo("America/New_York")
        ) == date(2026, 5, 27)

    def test_tokyo_noon_utc_unchanged(self):
        # Eastern cities were correct before and stay correct (regression guard).
        assert resolve_target_local_day(
            _NOON(2026, 4, 26), ZoneInfo("Asia/Tokyo")
        ) == date(2026, 4, 26)

    def test_auckland_noon_utc_is_title_day(self):
        # Far-east (+12/+13) — end_date-localize would over-shoot to title+1;
        # the UTC-date rule keeps it on the title day.
        assert resolve_target_local_day(
            _NOON(2026, 5, 27), ZoneInfo("Pacific/Auckland")
        ) == date(2026, 5, 27)

    def test_none_end_date(self):
        assert resolve_target_local_day(None, ZoneInfo("UTC")) is None

    def test_naive_end_date_treated_as_utc(self):
        assert resolve_target_local_day(
            datetime(2026, 5, 27, 12, 0), ZoneInfo("America/Sao_Paulo")
        ) == date(2026, 5, 27)


class TestShouldSkipFutureDayStationAware:
    """The real (station-aware) path that the legacy tests in test_scheduler.py
    don't exercise. ``today_local`` reads the wall clock, so we pin it."""

    def _market(self, end_date):
        return SimpleNamespace(end_date=end_date, id="m1")

    def test_sao_paulo_tomorrow_market_is_skipped(self, monkeypatch):
        monkeypatch.setattr(
            "src.signals.mapper.today_local", lambda tz: date(2026, 5, 26)
        )
        now = datetime(2026, 5, 26, 7, 10, tzinfo=timezone.utc)  # 04:10 SP
        # "highest temp in Sao Paulo on May 27" — end_date May 27 12:00 UTC.
        market = self._market(_NOON(2026, 5, 27))
        assert should_skip_future_day(market, now, station_icao="SBGR") is True

    def test_sao_paulo_today_market_not_skipped(self, monkeypatch):
        monkeypatch.setattr(
            "src.signals.mapper.today_local", lambda tz: date(2026, 5, 26)
        )
        now = datetime(2026, 5, 26, 7, 10, tzinfo=timezone.utc)
        market = self._market(_NOON(2026, 5, 26))
        assert should_skip_future_day(market, now, station_icao="SBGR") is False

    def test_sao_paulo_yesterday_market_not_skipped(self, monkeypatch):
        monkeypatch.setattr(
            "src.signals.mapper.today_local", lambda tz: date(2026, 5, 26)
        )
        now = datetime(2026, 5, 26, 7, 10, tzinfo=timezone.utc)
        market = self._market(_NOON(2026, 5, 25))
        assert should_skip_future_day(market, now, station_icao="SBGR") is False
