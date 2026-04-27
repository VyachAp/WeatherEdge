"""Tests for the deterministic lock-rule evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from src.signals.lock_rules import LockDecision, evaluate_lock
from src.signals.state_aggregator import WeatherState


# Use KJFK (America/New_York) as the test station — has a known timezone entry.
_TEST_ICAO = "KJFK"

# Reference market-end time: 2026-06-15 23:59 UTC. For America/New_York (EDT
# = UTC-4 in June), that's 19:59 local on 2026-06-15 — a sensible "end of
# local day" scenario for the tests. Observations leading up to this time
# anchor to the local calendar day 2026-06-15.
_MARKET_END = datetime(2026, 6, 15, 23, 59, tzinfo=timezone.utc)
_NOW_UTC = _MARKET_END - timedelta(minutes=10)  # Evaluate just before close


@dataclass
class _FakeMarket:
    parsed_threshold: float | None
    parsed_operator: str | None
    end_date: datetime | None = _MARKET_END
    question: str = ""
    parsed_target_date: str | None = None


def _build_history(current_max_f: float, count: int = 5) -> tuple[tuple[datetime, float], ...]:
    """Generate a `routine_history` ending at _NOW_UTC whose observed max
    equals ``current_max_f``. Each entry is hourly-spaced leading up to now.
    """
    if count <= 0:
        return ()
    obs: list[tuple[datetime, float]] = []
    # Peak is placed in the middle; surrounding values trail off by 2°F.
    peak_idx = count // 2
    for i in range(count):
        t = _NOW_UTC - timedelta(hours=count - 1 - i)
        # Temps rise to the peak then fall; ensures max == current_max_f.
        offset = abs(i - peak_idx) * 2.0
        obs.append((t, current_max_f - offset))
    return tuple(obs)


def _state(
    *,
    current_max_f: float = 70.0,
    forecast_peak_f: float = 75.0,
    metar_trend: float = 0.0,
    solar_declining: bool = False,
    routine_count: int = 5,
    has_forecast: bool = True,
) -> WeatherState:
    return WeatherState(
        station_icao=_TEST_ICAO,
        current_max_f=current_max_f,
        metar_trend_rate=metar_trend,
        dewpoint_trend_rate=0.0,
        forecast_peak_f=forecast_peak_f,
        hours_until_peak=0.0,
        solar_declining=solar_declining,
        solar_decline_magnitude=0.6 if solar_declining else 0.0,
        cloud_rising=False,
        cloud_rise_magnitude=0.0,
        routine_count_today=routine_count,
        has_forecast=has_forecast,
        routine_history=_build_history(current_max_f, count=routine_count),
    )


def _eval(state: WeatherState, market: _FakeMarket) -> LockDecision:
    """Wrapper that pins now_utc so test observations fall in window."""
    return evaluate_lock(state, market, now_utc=_NOW_UTC)


class TestEasyDirectionYES:
    """Observed max already clears an above/at_least threshold — LOCKED YES."""

    def test_above_locked_yes(self):
        mkt = _FakeMarket(parsed_threshold=80.0, parsed_operator="above")
        state = _state(current_max_f=83.0)
        decision = _eval(state, mkt)
        assert decision.side == "YES"
        assert decision.margin_f == 3.0

    def test_at_least_locked_yes(self):
        mkt = _FakeMarket(parsed_threshold=80.0, parsed_operator="at_least")
        state = _state(current_max_f=82.1)
        decision = _eval(state, mkt)
        assert decision.side == "YES"

    def test_just_inside_margin_not_locked(self):
        # Margin default is 2.0°F; 81 is only 1°F over threshold 80 → no lock.
        mkt = _FakeMarket(parsed_threshold=80.0, parsed_operator="above")
        state = _state(current_max_f=81.0)
        assert _eval(state, mkt).side is None


class TestEasyDirectionNO:
    """Observed max already exceeds a below/at_most threshold — LOCKED NO."""

    def test_below_locked_no(self):
        mkt = _FakeMarket(parsed_threshold=80.0, parsed_operator="below")
        state = _state(current_max_f=83.0)
        decision = _eval(state, mkt)
        assert decision.side == "NO"

    def test_at_most_locked_no(self):
        mkt = _FakeMarket(parsed_threshold=80.0, parsed_operator="at_most")
        state = _state(current_max_f=85.0)
        assert _eval(state, mkt).side == "NO"


class TestHardDirectionAbove:
    """above/at_least threshold not yet reached — requires forecast + past-peak."""

    def test_lock_no_with_solar_declining_and_low_forecast(self):
        # current max 70 < threshold 80 - margin 2 → 70 < 78 ✓
        # forecast peak 76 < 80 ✓, solar declining ✓ → LOCK NO
        mkt = _FakeMarket(parsed_threshold=80.0, parsed_operator="above")
        state = _state(current_max_f=70.0, forecast_peak_f=76.0, solar_declining=True)
        assert _eval(state, mkt).side == "NO"

    def test_lock_no_with_falling_trend(self):
        # Alternative past-peak signal: metar trend <= 0
        mkt = _FakeMarket(parsed_threshold=80.0, parsed_operator="above")
        state = _state(current_max_f=70.0, forecast_peak_f=76.0, metar_trend=-0.5)
        assert _eval(state, mkt).side == "NO"

    def test_no_lock_when_forecast_above_threshold(self):
        # Forecast peak still projects to exceed threshold → can't lock NO.
        mkt = _FakeMarket(parsed_threshold=80.0, parsed_operator="above")
        state = _state(current_max_f=70.0, forecast_peak_f=82.0, solar_declining=True)
        assert _eval(state, mkt).side is None

    def test_no_lock_when_neither_past_peak_signal(self):
        # Even with low forecast, if temp is rising and solar isn't declining, no lock.
        mkt = _FakeMarket(parsed_threshold=80.0, parsed_operator="above")
        state = _state(
            current_max_f=70.0,
            forecast_peak_f=76.0,
            solar_declining=False,
            metar_trend=+1.5,
        )
        assert _eval(state, mkt).side is None


class TestHardDirectionBelow:
    """below/at_most threshold not yet crossed — YES locks on forecast + past-peak."""

    def test_lock_yes_for_below_when_safely_under(self):
        mkt = _FakeMarket(parsed_threshold=80.0, parsed_operator="below")
        state = _state(current_max_f=70.0, forecast_peak_f=76.0, solar_declining=True)
        assert _eval(state, mkt).side == "YES"

    def test_lock_yes_for_at_most(self):
        mkt = _FakeMarket(parsed_threshold=80.0, parsed_operator="at_most")
        state = _state(current_max_f=70.0, forecast_peak_f=76.0, metar_trend=-0.3)
        assert _eval(state, mkt).side == "YES"


class TestRejections:
    def test_exactly_no_question_unparseable_no_lock(self):
        # 'exactly' with no question text and no threshold-shaped fallback
        # has no parseable range — lock returns None.
        mkt = _FakeMarket(parsed_threshold=None, parsed_operator="exactly")
        state = _state(current_max_f=85.0)
        assert _eval(state, mkt).side is None

    def test_missing_threshold(self):
        mkt = _FakeMarket(parsed_threshold=None, parsed_operator="above")
        state = _state(current_max_f=100.0)
        assert _eval(state, mkt).side is None

    def test_missing_operator(self):
        mkt = _FakeMarket(parsed_threshold=80.0, parsed_operator=None)
        state = _state(current_max_f=100.0)
        assert _eval(state, mkt).side is None

    def test_routine_count_guard_for_standard_margin(self):
        # Standard EASY margin (1°F over threshold + 2°F lock margin) is
        # rejected with only 2 routines — needs MIN_ROUTINE_COUNT=3.
        mkt = _FakeMarket(parsed_threshold=80.0, parsed_operator="above")
        # current_max=83 → overshoot=3°F → between LOCK_MARGIN_F (2°F)
        # and super_margin (4°F). Standard rules: needs 3 routines.
        state = _state(current_max_f=83.0, routine_count=2)
        assert _eval(state, mkt).side is None

    def test_routine_count_floor_blocks_single_metar(self):
        # Hard floor: even a 10°F super-margin overshoot is rejected at
        # routine_count=1 to prevent single-METAR fluke trades.
        mkt = _FakeMarket(parsed_threshold=80.0, parsed_operator="above")
        state = _state(current_max_f=90.0, routine_count=1)
        assert _eval(state, mkt).side is None

    def test_hard_direction_requires_forecast(self):
        # NO-side lock on an above/at_least market needs forecast evidence to
        # claim peak won't reach threshold. Without has_forecast, don't lock.
        mkt = _FakeMarket(parsed_threshold=80.0, parsed_operator="above")
        state = _state(
            current_max_f=70.0, forecast_peak_f=76.0,
            solar_declining=True, has_forecast=False,
        )
        assert _eval(state, mkt).side is None

    def test_easy_direction_does_not_require_forecast(self):
        # Easy YES direction (observed max already clears threshold) is
        # mathematically locked by monotonicity — no forecast needed.
        mkt = _FakeMarket(parsed_threshold=80.0, parsed_operator="above")
        state = _state(current_max_f=85.0, has_forecast=False)
        assert _eval(state, mkt).side == "YES"

    def test_requires_end_date(self):
        # Without end_date we can't anchor to a target day → no lock.
        mkt = _FakeMarket(parsed_threshold=80.0, parsed_operator="above", end_date=None)
        state = _state(current_max_f=90.0)
        assert _eval(state, mkt).side is None


class TestSuperMarginEarlyLock:
    """EASY direction with overshoot >= 2× LOCK_MARGIN_F locks at routine #2.

    Daily max is monotonic — two confirming obs already 4°F+ over threshold
    cannot be undone by a third observation. Cuts ~30-60 min of morning lag
    on hot days where the threshold is clearly exceeded by the second routine.
    """

    def test_super_margin_above_locks_at_two_routines(self):
        mkt = _FakeMarket(parsed_threshold=80.0, parsed_operator="above")
        # Overshoot 5°F (>= super margin = 2 × 2 = 4°F) at count=2 → YES.
        state = _state(current_max_f=85.0, routine_count=2)
        decision = _eval(state, mkt)
        assert decision.side == "YES"
        assert decision.margin_f == 5.0

    def test_super_margin_below_locks_at_two_routines(self):
        mkt = _FakeMarket(parsed_threshold=80.0, parsed_operator="below")
        state = _state(current_max_f=85.0, routine_count=2)
        decision = _eval(state, mkt)
        assert decision.side == "NO"
        assert decision.margin_f == 5.0

    def test_borderline_super_margin_locks_at_two_routines(self):
        # Exactly threshold + super_margin (= threshold + 4°F) is the boundary.
        mkt = _FakeMarket(parsed_threshold=80.0, parsed_operator="above")
        state = _state(current_max_f=84.0, routine_count=2)
        assert _eval(state, mkt).side == "YES"

    def test_just_under_super_margin_requires_three_routines(self):
        # Overshoot 3°F (between standard margin 2°F and super margin 4°F)
        # still needs 3 routines.
        mkt = _FakeMarket(parsed_threshold=80.0, parsed_operator="above")
        state_two = _state(current_max_f=83.0, routine_count=2)
        state_three = _state(current_max_f=83.0, routine_count=3)
        assert _eval(state_two, mkt).side is None
        assert _eval(state_three, mkt).side == "YES"


class TestPerMarketWindow:
    """Regression tests: the key bug was that lock rule used state.current_max_f
    (snapshot's local day) rather than the market's target-day window."""

    def test_stale_high_observation_does_not_trigger_easy_yes(self):
        # Regression for the Dallas April 21 07:00-local-close bug: the market
        # closes 2h after local midnight, so only early-morning observations
        # count. A 90°F observation from 2 days ago in routine_history must NOT
        # trigger an easy YES lock for a 80°F at_least market — because the
        # market's target-day window only spans cold early-morning hours.
        early_close = _MARKET_END
        history = (
            (early_close - timedelta(days=2, hours=4), 90.0),  # Stale peak
            (early_close - timedelta(hours=5), 55.0),          # Target-day obs
            (early_close - timedelta(hours=4), 56.0),
            (early_close - timedelta(hours=3), 57.0),
            (early_close - timedelta(hours=2), 58.0),
            (early_close - timedelta(hours=1), 59.0),
        )
        state = WeatherState(
            station_icao=_TEST_ICAO,
            current_max_f=90.0,  # stale carry-over from state; rule must ignore
            metar_trend_rate=+1.0,  # rising → hard-direction NO can't fire
            dewpoint_trend_rate=0.0,
            forecast_peak_f=85.0,  # forecast says peak > threshold → blocks hard NO
            hours_until_peak=0.0,
            solar_declining=False,  # not declining → blocks hard NO
            solar_decline_magnitude=0.0,
            cloud_rising=False,
            cloud_rise_magnitude=0.0,
            routine_count_today=6,
            has_forecast=True,
            routine_history=history,
        )
        mkt = _FakeMarket(parsed_threshold=80.0, parsed_operator="at_least")
        # Target-day max = 59°F; neither easy YES (59 < 80+2) nor hard NO
        # (forecast 85 > threshold 80, temp rising) can fire.
        assert _eval(state, mkt).side is None

    def test_lock_fires_only_on_target_day_observations(self):
        # Same shape as above but with today's observations exceeding threshold.
        history = tuple(
            (_MARKET_END - timedelta(hours=12 - i), 70.0 + i * 2.0)
            for i in range(6)
        )
        # Max in this history is 70 + 5*2 = 80°F
        state = WeatherState(
            station_icao=_TEST_ICAO,
            current_max_f=80.0,
            metar_trend_rate=0.0,
            dewpoint_trend_rate=0.0,
            forecast_peak_f=80.0,
            hours_until_peak=0.0,
            solar_declining=False,
            solar_decline_magnitude=0.0,
            cloud_rising=False,
            cloud_rise_magnitude=0.0,
            routine_count_today=6,
            has_forecast=True,
            routine_history=history,
        )
        mkt = _FakeMarket(parsed_threshold=74.0, parsed_operator="at_least")
        # 80°F - 74°F = 6°F margin, well above the 2°F lock margin → YES.
        decision = _eval(state, mkt)
        assert decision.side == "YES"
        assert decision.margin_f == 6.0


class TestDecisionShape:
    def test_direction_maps_to_trade_direction(self):
        from src.db.models import TradeDirection

        mkt = _FakeMarket(parsed_threshold=80.0, parsed_operator="above")
        yes = _eval(_state(current_max_f=85.0), mkt)
        assert yes.direction == TradeDirection.BUY_YES

        mkt_no = _FakeMarket(parsed_threshold=80.0, parsed_operator="below")
        no = _eval(_state(current_max_f=85.0), mkt_no)
        assert no.direction == TradeDirection.BUY_NO

        none = LockDecision(side=None)
        assert none.direction is None

    def test_reasons_populated(self):
        mkt = _FakeMarket(parsed_threshold=80.0, parsed_operator="above")
        state = _state(current_max_f=85.0)
        decision = _eval(state, mkt)
        assert decision.reasons
        assert any("85" in r for r in decision.reasons)


class TestRangeMarkets:
    """Lock-rule support for 'between X-Y°F' and 'X°C exactly' shapes."""

    def test_overshoot_locks_no(self):
        # 'between 80-81°F' market, observed max already 85°F → NO lock.
        mkt = _FakeMarket(
            parsed_threshold=None,
            parsed_operator="bracket",
            question="Will the highest temperature in NYC be between 80-81°F on June 15?",
        )
        state = _state(current_max_f=85.0)
        decision = _eval(state, mkt)
        assert decision.side == "NO"
        # Margin is observed - high (3°F over 81 + 2 margin = 4 → margin_f=4)
        assert decision.margin_f >= 4.0

    def test_undershoot_with_no_more_heating_locks_no(self):
        mkt = _FakeMarket(
            parsed_threshold=None,
            parsed_operator="bracket",
            question="Will the highest temperature in NYC be between 80-81°F on June 15?",
        )
        # Far below the range AND past peak with declining solar.
        state = _state(
            current_max_f=70.0, forecast_peak_f=75.0, solar_declining=True,
        )
        assert _eval(state, mkt).side == "NO"

    def test_in_range_post_peak_locks_yes(self):
        mkt = _FakeMarket(
            parsed_threshold=None,
            parsed_operator="bracket",
            question="Will the highest temperature in NYC be between 80-81°F on June 15?",
        )
        # Observed 80, past peak, solar declining, trend flat, forecast caps at 81.
        state = _state(
            current_max_f=80.0, forecast_peak_f=81.0,
            solar_declining=True, metar_trend=0.0,
        )
        decision = _eval(state, mkt)
        assert decision.side == "YES"

    def test_in_range_pre_peak_no_lock(self):
        # Same market, observed in-range but still pre-peak → don't lock yet.
        mkt = _FakeMarket(
            parsed_threshold=None,
            parsed_operator="bracket",
            question="Will the highest temperature in NYC be between 80-81°F on June 15?",
        )
        state = WeatherState(
            station_icao=_TEST_ICAO,
            current_max_f=80.0, metar_trend_rate=+1.0,
            dewpoint_trend_rate=0.0,
            forecast_peak_f=81.0, hours_until_peak=2.0,  # peak still ahead
            solar_declining=False, solar_decline_magnitude=0.0,
            cloud_rising=False, cloud_rise_magnitude=0.0,
            routine_count_today=5, has_forecast=True,
            routine_history=_build_history(80.0, count=5),
        )
        assert _eval(state, mkt).side is None

    def test_celsius_exactly_overshoot_locks_no(self):
        # '17°C exactly' resolves to °F bucket {62, 63}. Observed 70°F
        # (>> 63 + margin) → NO.
        from src.signals.mapper import f_to_c

        c17_in_f = 17.0 * 9.0 / 5.0 + 32.0  # = 62.6°F
        mkt = _FakeMarket(
            parsed_threshold=c17_in_f,
            parsed_operator="exactly",
            question="Will the highest temperature in Amsterdam be 17°C on June 15?",
        )
        state = _state(current_max_f=70.0)
        assert _eval(state, mkt).side == "NO"
        # Sanity: Celsius round-trip
        assert round(f_to_c(c17_in_f)) == 17

    def test_celsius_exactly_in_range_locks_yes(self):
        # 17°C exactly = bucket [62, 63]. Observed 63°F (= 17.2°C, rounds to 17),
        # past peak, declining, forecast caps at 63°F → YES.
        c17_in_f = 17.0 * 9.0 / 5.0 + 32.0
        mkt = _FakeMarket(
            parsed_threshold=c17_in_f,
            parsed_operator="exactly",
            question="Will the highest temperature in Amsterdam be 17°C on June 15?",
        )
        state = _state(
            current_max_f=63.0, forecast_peak_f=63.0,
            solar_declining=True, metar_trend=0.0,
        )
        decision = _eval(state, mkt)
        assert decision.side == "YES"
