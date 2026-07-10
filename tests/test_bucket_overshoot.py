"""Tests for the `bucket_overshoot` lock branch.

The rule: once the market-day running routine-METAR max has climbed
`BUCKET_OVERSHOOT_MARGIN_C` whole °C above a bucket's top, that bucket can never
be the day's max (the max is monotonic) → BUY NO.

Validated on prod 2026-07-10: 63 bets / 52 station-days, 2 losses,
EV +0.91/$1 [95% CI +0.53, +1.46]. See `src.config.Settings.BUCKET_OVERSHOOT_*`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from src.config import settings
from src.signals.lock_rules import evaluate_lock
from src.signals.state_aggregator import WeatherState


_ICAO = "KJFK"
_MARKET_END = datetime(2026, 6, 15, 23, 59, tzinfo=timezone.utc)
_NOW = _MARKET_END - timedelta(minutes=10)


@dataclass
class _Market:
    parsed_threshold: float | None
    parsed_operator: str | None
    end_date: datetime | None = _MARKET_END
    question: str = "Will the highest temperature in New York City be 20°C on June 15?"
    parsed_target_date: str | None = None


def _history(max_f: float, count: int = 4) -> tuple[tuple[datetime, float], ...]:
    """`count` hourly routines on the market's local day, peaking at `max_f`."""
    return tuple(
        (_NOW - timedelta(hours=count - 1 - i), max_f - 2.0 * (count - 1 - i))
        for i in range(count)
    )


def _state(max_f: float, count: int = 4, icao: str = _ICAO) -> WeatherState:
    hist = _history(max_f, count)
    return WeatherState(
        station_icao=icao,
        current_max_f=max_f,
        metar_trend_rate=0.0,
        dewpoint_trend_rate=0.0,
        forecast_peak_f=0.0,
        hours_until_peak=0.0,
        solar_declining=False,
        solar_decline_magnitude=0.0,
        cloud_rising=False,
        cloud_rise_magnitude=0.0,
        routine_count_today=count,
        has_forecast=False,
        routine_history=hist,
    )


@pytest.fixture(autouse=True)
def _enable(monkeypatch):
    monkeypatch.setattr(settings, "BUCKET_OVERSHOOT_LOCK_ENABLED", True)
    monkeypatch.setattr(settings, "BUCKET_OVERSHOOT_MARGIN_C", 1.0)
    monkeypatch.setattr(settings, "MIN_ROUTINE_COUNT", 3)


# 20°C == 68.0°F. The bucket dies once the max reaches 21°C == 69.8°F.
_C20 = _Market(parsed_threshold=68.0, parsed_operator="exactly")


def test_fires_no_when_max_is_one_full_c_above_bucket():
    d = evaluate_lock(_state(69.8), _C20, now_utc=_NOW)
    assert d.side == "NO"
    assert d.branch == "bucket_overshoot"
    assert d.observed_max_f == pytest.approx(69.8)
    assert d.margin_f == pytest.approx(0.0, abs=0.05)


def test_does_not_fire_at_the_bucket_itself():
    """Max == 20°C: the bucket is still live (it may well be the winner)."""
    assert evaluate_lock(_state(68.0), _C20, now_utc=_NOW).side is None


def test_does_not_fire_just_below_the_next_c_step():
    """69.0°F is above the bucket value but below 21°C — not yet certain."""
    assert evaluate_lock(_state(69.0), _C20, now_utc=_NOW).side is None


def test_fires_far_above():
    d = evaluate_lock(_state(80.6), _C20, now_utc=_NOW)
    assert d.side == "NO" and d.branch == "bucket_overshoot"
    assert d.margin_f == pytest.approx(80.6 - 69.8, abs=0.05)


def test_margin_of_two_c_requires_two_steps():
    """With MARGIN_C=2 the bucket only dies at 22°C (71.6°F)."""
    settings.BUCKET_OVERSHOOT_MARGIN_C = 2.0
    assert evaluate_lock(_state(69.8), _C20, now_utc=_NOW).side is None
    assert evaluate_lock(_state(71.6), _C20, now_utc=_NOW).branch == "bucket_overshoot"


def test_excluded_station_never_fires(monkeypatch):
    """High resolver-divergence stations are the only real loss mode."""
    monkeypatch.setattr(settings, "BUCKET_OVERSHOOT_EXCLUDED_STATIONS", "ZGSZ,KJFK")
    assert evaluate_lock(_state(80.6), _C20, now_utc=_NOW).side is None


def test_disabled_flag_is_a_no_op(monkeypatch):
    monkeypatch.setattr(settings, "BUCKET_OVERSHOOT_LOCK_ENABLED", False)
    assert evaluate_lock(_state(80.6), _C20, now_utc=_NOW).branch != "bucket_overshoot"


def test_requires_min_routine_count():
    """A single spurious hot METAR must not fire the branch."""
    assert evaluate_lock(_state(80.6, count=2), _C20, now_utc=_NOW).side is None
    assert evaluate_lock(_state(80.6, count=3), _C20, now_utc=_NOW).side == "NO"


def test_fahrenheit_market_steps_by_one_degree_f():
    m = _Market(
        parsed_threshold=88.0,
        parsed_operator="exactly",
        question="Will the highest temperature in New York City be 88°F on June 15?",
    )
    assert evaluate_lock(_state(88.0), m, now_utc=_NOW).side is None
    assert evaluate_lock(_state(89.0), m, now_utc=_NOW).branch == "bucket_overshoot"


def test_ignores_observations_outside_the_market_local_day():
    """The max must come from the market's own station-local day.

    Yesterday's heat cannot kill today's bucket. Regression guard for the
    pre-2026-05-26 wrong-day bug that produced the only real loss the old
    `range_overshoot` branch ever took.
    """
    stale = ((_NOW - timedelta(days=1), 95.0),)
    st = _state(68.0)
    st = WeatherState(**{**st.__dict__, "routine_history": stale + st.routine_history})
    d = evaluate_lock(st, _C20, now_utc=_NOW)
    assert d.side is None, "yesterday's 95°F must not kill today's 20°C bucket"


# ---------------------------------------------------------------------------
# Startup guard: a malformed signing key must never fail silently
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_startup_alerts_when_signing_key_is_unusable(monkeypatch):
    """Prod ran 2026-07-10 with a 39-byte key: every order path was dead while
    the rest of the bot kept writing telemetry and looked healthy."""
    from unittest.mock import AsyncMock, MagicMock

    from src.scheduler import jobs

    monkeypatch.setattr(jobs.settings, "AUTO_EXECUTE", True)
    monkeypatch.setattr(jobs.settings, "POLYMARKET_PRIVATE_KEY", "0x" + "ab" * 39)
    alerter = MagicMock()
    alerter._enqueue = AsyncMock()
    monkeypatch.setattr(jobs, "get_alerter", lambda: alerter)

    await jobs._assert_signing_key_usable()

    alerter._enqueue.assert_awaited_once()
    msg = alerter._enqueue.await_args.args[0]
    assert "NO ORDER CAN BE PLACED" in msg
    assert "ab" * 39 not in msg, "key material must never be logged or alerted"


@pytest.mark.asyncio
async def test_startup_guard_silent_on_valid_key(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    from src.scheduler import jobs

    monkeypatch.setattr(jobs.settings, "AUTO_EXECUTE", True)
    monkeypatch.setattr(jobs.settings, "POLYMARKET_PRIVATE_KEY", "0x" + "11" * 32)
    alerter = MagicMock(); alerter._enqueue = AsyncMock()
    monkeypatch.setattr(jobs, "get_alerter", lambda: alerter)
    await jobs._assert_signing_key_usable()
    alerter._enqueue.assert_not_awaited()


@pytest.mark.asyncio
async def test_startup_guard_noop_when_not_auto_executing(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    from src.scheduler import jobs

    monkeypatch.setattr(jobs.settings, "AUTO_EXECUTE", False)
    monkeypatch.setattr(jobs.settings, "POLYMARKET_PRIVATE_KEY", "garbage")
    alerter = MagicMock(); alerter._enqueue = AsyncMock()
    monkeypatch.setattr(jobs, "get_alerter", lambda: alerter)
    await jobs._assert_signing_key_usable()
    alerter._enqueue.assert_not_awaited()
