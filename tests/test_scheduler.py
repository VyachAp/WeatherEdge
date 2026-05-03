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


class TestBinaryMarketEdgeSideSelection:
    """`_binary_market_edge` picks the BUY_YES or BUY_NO side based on
    whichever has positive edge. The side-effective frame guarantees
    that price/probability filters work symmetrically."""

    @staticmethod
    def _setup(*, our_prob_yes, yes_price, op="at_least", threshold=80,
               depth_yes=100.0, depth_no=100.0):
        from src.scheduler import _binary_market_edge
        from src.signals.probability_engine import BucketDistribution

        market = SimpleNamespace(
            id="m1", question="Will the highest temp be 80°F or higher",
            parsed_threshold=threshold, parsed_operator=op,
            current_yes_price=yes_price, end_date=None, outcomes=["Yes", "No"],
        )
        # Build a distribution that yields exactly `our_prob_yes` for `at_least`.
        dist = BucketDistribution(
            current_max_f=70,
            probabilities={threshold: our_prob_yes, threshold - 1: 1.0 - our_prob_yes},
            reasoning=["test"],
        )
        end_time = datetime.now(timezone.utc) + timedelta(hours=5)
        no_calls = []

        def _no_depth():
            no_calls.append(1)
            return depth_no

        edge = _binary_market_edge(
            dist, market, end_time, routine_count=5,
            depth_yes=depth_yes, depth_no_fn=_no_depth,
        )
        return edge, no_calls

    def test_picks_yes_when_prob_above_price(self):
        from src.db.models import TradeDirection

        edge, no_calls = self._setup(our_prob_yes=0.70, yes_price=0.50)
        assert edge.direction == TradeDirection.BUY_YES
        assert edge.our_probability == 0.70
        assert edge.market_price == 0.50
        assert edge.edge == 0.20
        assert edge.passes is True
        # NO depth never fetched on the YES branch.
        assert no_calls == []

    def test_picks_no_when_prob_below_price(self):
        from src.db.models import TradeDirection

        edge, no_calls = self._setup(our_prob_yes=0.30, yes_price=0.55)
        assert edge.direction == TradeDirection.BUY_NO
        # NO frame: prob = 1 - 0.30 = 0.70, price = 1 - 0.55 = 0.45, edge = 0.25
        assert edge.our_probability == 0.70
        assert edge.market_price == 0.45
        assert edge.edge == 0.25
        assert edge.passes is True
        # NO depth was lazily fetched.
        assert no_calls == [1]

    def test_no_side_uses_no_depth_for_filter(self):
        # NO branch picked, but NO depth too thin → rejected by depth filter
        # in NO frame, not silently let through using YES depth.
        edge, _ = self._setup(
            our_prob_yes=0.30, yes_price=0.55, depth_yes=500.0, depth_no=2.0,
        )
        assert edge.passes is False
        assert "depth" in (edge.reject_reason or "")

    def test_no_side_below_market(self):
        # 'below' op: P(YES) for "max < threshold". Build distribution
        # explicitly so that our_prob_yes for 'below' is 0.30 (model thinks
        # max-below-80 is unlikely; market overprices it at 0.55) → NO edge.
        from src.db.models import TradeDirection
        from src.scheduler import _binary_market_edge
        from src.signals.probability_engine import BucketDistribution

        market = SimpleNamespace(
            id="m1", question="Will the highest temp be below 80°F",
            parsed_threshold=80, parsed_operator="below",
            current_yes_price=0.55, end_date=None, outcomes=["Yes", "No"],
        )
        # 30% of mass below threshold (b<80), 70% at-or-above.
        dist = BucketDistribution(
            current_max_f=70,
            probabilities={79: 0.30, 80: 0.70},
            reasoning=["test"],
        )
        end_time = datetime.now(timezone.utc) + timedelta(hours=5)
        edge = _binary_market_edge(
            dist, market, end_time, routine_count=5,
            depth_yes=100.0, depth_no_fn=lambda: 100.0,
        )
        assert edge.direction == TradeDirection.BUY_NO
        # NO frame: prob = 1 - 0.30 = 0.70, price = 1 - 0.55 = 0.45, edge = 0.25
        assert edge.our_probability == 0.70
        assert edge.market_price == 0.45

    def test_zero_edge_either_side_rejected(self):
        edge, _ = self._setup(our_prob_yes=0.50, yes_price=0.50)
        # |edge| = 0 < MIN_EDGE; both sides reject. The returned candidate
        # carries the edge=0 from whichever side was tried.
        assert edge.passes is False
        assert "edge" in (edge.reject_reason or "")

    def test_no_side_passes_min_entry_price_in_no_frame(self):
        # YES price 0.55 → NO price 0.45, which clears the 0.40 floor.
        edge, _ = self._setup(our_prob_yes=0.30, yes_price=0.55)
        assert edge.market_price == 0.45
        assert edge.passes is True  # 0.45 >= MIN_ENTRY_PRICE (0.40)

    def test_no_side_fails_min_entry_price_when_yes_near_one(self):
        # YES at 0.99 → NO at 0.01, fails MIN_ENTRY_PRICE. Lock-rule path
        # (with its own LOCK_RULE_MIN_PRICE=0.05) is the right tool here,
        # not the probability path.
        edge, _ = self._setup(our_prob_yes=0.40, yes_price=0.99)
        # Edge would be (1-0.40) - (1-0.99) = 0.60 - 0.01 = 0.59 — huge,
        # but the price filter should still reject because no_price < 0.40.
        assert edge.passes is False
        assert "price" in (edge.reject_reason or "")

    def test_no_side_passes_min_probability_in_no_frame(self):
        # PR-3 floor is 0.50. NO trade with our_prob_yes=0.30 has effective
        # NO prob = 0.70, well above the 0.50 floor — passes even though
        # raw P(YES)=0.30 is below the floor. Proves PR-1's side-aware
        # gate works with PR-3's lower threshold.
        from src.db.models import TradeDirection

        edge, _ = self._setup(our_prob_yes=0.30, yes_price=0.55)
        assert edge.direction == TradeDirection.BUY_NO
        assert edge.our_probability == 0.70  # NO frame
        assert edge.passes is True


class TestBinaryMarketEdgeAsymmetricPricing:
    """When a real (yes_bid, yes_ask) quote is supplied, each side must
    be evaluated against its own BUY-side cost (yes_ask for YES,
    1-yes_bid for NO). Without this, a wide post-move spread (e.g. dust
    bid=0.20 + dust ask=0.55 on a market trading near YES≈0) makes the
    arithmetic mid look mid-priced and invents a phantom edge that
    wouldn't fill in live mode.

    Regression for the 2026-04-26 Taipei "exactly 28°C" incident: the
    bot logged P(NO)=0.807 vs mkt=0.625 / edge=+0.182 and queued a
    BUY_NO at limit 0.645, but real NO ask was ~0.999 (Gamma reported
    NO outcomePrice=0.9995, YES bestAsk=0.001).
    """

    @staticmethod
    def _eval(*, our_prob_yes, yes_bid, yes_ask, op="exactly", threshold=82,
              depth_yes=100.0, depth_no=100.0):
        from src.scheduler import _binary_market_edge
        from src.signals.probability_engine import BucketDistribution

        market = SimpleNamespace(
            id="m1",
            question="Will the highest temperature in Taipei be 28°C on April 26",
            parsed_threshold=threshold, parsed_operator=op,
            current_yes_price=(yes_bid + yes_ask) / 2,  # legacy mid path
            end_date=None, outcomes=["Yes", "No"],
        )
        # For the "exactly 82°F" market, range_f = (82, 82). Put 19.3% mass
        # inside the range and 80.7% outside, mirroring the Taipei log line.
        dist = BucketDistribution(
            current_max_f=78,
            probabilities={threshold: our_prob_yes, threshold - 1: 1.0 - our_prob_yes},
            reasoning=["test"],
        )
        end_time = datetime.now(timezone.utc) + timedelta(hours=5)
        return _binary_market_edge(
            dist, market, end_time, routine_count=5,
            depth_yes=depth_yes, depth_no_fn=lambda: depth_no,
            yes_bid=yes_bid, yes_ask=yes_ask,
        )

    def test_wide_spread_kills_phantom_no_edge(self):
        # Reproduces the Taipei incident: dust spread 0.20/0.55, model
        # P(NO)=0.807, mid-derived NO=0.625 would suggest +0.182 edge.
        # Real NO buy cost = 1 - 0.20 = 0.80 → edge = 0.807 - 0.80 ≈ 0.007
        # → fails MIN_EDGE=0.05 and the trade is correctly rejected.
        edge = self._eval(our_prob_yes=0.193, yes_bid=0.20, yes_ask=0.55)
        # Direction will still be NO (it's the higher-edge side), but the
        # tiny real edge must fail the gate.
        assert edge.market_price == 0.80, (
            f"NO buy price should be 1 - yes_bid = 0.80, got {edge.market_price}"
        )
        assert abs(edge.edge - 0.007) < 0.001
        assert edge.passes is False
        assert "edge" in (edge.reject_reason or "")

    def test_yes_side_uses_ask_not_mid(self):
        # YES bid=0.40, ask=0.55 (tighter spread). With model P(YES)=0.70:
        #   - Mid-based: edge = 0.70 - 0.475 = +0.225 (passes)
        #   - Asymmetric (real ask): edge = 0.70 - 0.55 = +0.15 (still passes,
        #     but smaller and accurate)
        edge = self._eval(
            our_prob_yes=0.70, yes_bid=0.40, yes_ask=0.55,
            op="at_least", threshold=82,
        )
        from src.db.models import TradeDirection
        assert edge.direction == TradeDirection.BUY_YES
        assert edge.market_price == 0.55
        assert abs(edge.edge - 0.15) < 0.001
        assert edge.passes is True

    def test_omitting_quote_falls_back_to_mid(self):
        # Backward-compat: when the caller doesn't pass yes_bid/yes_ask,
        # the function still works against current_yes_price (the mid
        # legacy callers were storing). Same scenario as the Taipei one
        # but without the quote — phantom edge IS reported. This test
        # documents the fallback so we know if it ever changes.
        from src.scheduler import _binary_market_edge
        from src.signals.probability_engine import BucketDistribution

        market = SimpleNamespace(
            id="m1", question="Will the highest temp be 82°F or higher",
            parsed_threshold=82, parsed_operator="at_least",
            current_yes_price=0.375, end_date=None, outcomes=["Yes", "No"],
        )
        dist = BucketDistribution(
            current_max_f=78,
            probabilities={82: 0.193, 81: 0.807},
            reasoning=["test"],
        )
        end_time = datetime.now(timezone.utc) + timedelta(hours=5)
        edge = _binary_market_edge(
            dist, market, end_time, routine_count=5,
            depth_yes=100.0, depth_no_fn=lambda: 100.0,
            # no yes_bid / yes_ask — legacy path
        )
        # NO frame: prob=0.807, price=1-0.375=0.625, edge=+0.182 (legacy
        # phantom). Documented behavior — callers that want correctness
        # MUST pass the quote.
        assert edge.market_price == 0.625
        assert abs(edge.edge - 0.182) < 0.001


# ---------------------------------------------------------------------------
# Per-station local-day cache rollover (replaces the legacy 22:00 UTC wipe)
# ---------------------------------------------------------------------------


class TestPerStationCacheRollover:
    """`_maybe_clear_per_station_caches` clears only stations whose local
    day rolled over since last call, not all stations globally."""

    def setup_method(self):
        # Reset module-level state between tests.
        from src import scheduler as sch
        sch._locked_markets_fired_today.clear()
        sch._unified_fired_today.clear()
        sch._last_routine_seen.clear()
        sch._market_to_icao.clear()
        sch._local_day_seen.clear()

    def test_first_call_seeds_state_no_clears(self):
        from src import scheduler as sch
        # Pre-populate dedup state for two stations.
        sch._locked_markets_fired_today.add("mkt_kjfk_1")
        sch._market_to_icao["mkt_kjfk_1"] = "KJFK"
        sch._last_routine_seen["KJFK"] = datetime.now(timezone.utc)

        sch._maybe_clear_per_station_caches()

        # First call seeds _local_day_seen but doesn't drop anything (no
        # prior cookie to compare against).
        assert "mkt_kjfk_1" in sch._locked_markets_fired_today
        assert "KJFK" in sch._last_routine_seen
        assert "KJFK" in sch._local_day_seen

    def test_only_rolled_over_stations_cleared(self):
        from src import scheduler as sch
        from datetime import date

        sch._locked_markets_fired_today.update({"mkt_kjfk_1", "mkt_egll_1"})
        sch._market_to_icao.update({"mkt_kjfk_1": "KJFK", "mkt_egll_1": "EGLL"})
        sch._last_routine_seen.update({
            "KJFK": datetime.now(timezone.utc),
            "EGLL": datetime.now(timezone.utc),
        })

        # Pretend we've already seen yesterday's local-date for KJFK only.
        # (EGLL hasn't rolled — its cookie matches today_local; KJFK's
        # cookie is yesterday so it should clear.)
        from src.signals.mapper import icao_timezone, today_local
        kjfk_today = today_local(icao_timezone("KJFK"))
        egll_today = today_local(icao_timezone("EGLL"))
        sch._local_day_seen["KJFK"] = kjfk_today - timedelta(days=1)
        sch._local_day_seen["EGLL"] = egll_today

        sch._maybe_clear_per_station_caches()

        # KJFK was rolled over → its dedup entries dropped.
        assert "mkt_kjfk_1" not in sch._locked_markets_fired_today
        assert "mkt_kjfk_1" not in sch._market_to_icao
        assert "KJFK" not in sch._last_routine_seen
        assert sch._local_day_seen["KJFK"] == kjfk_today

        # EGLL had no rollover → entries preserved.
        assert "mkt_egll_1" in sch._locked_markets_fired_today
        assert "EGLL" in sch._last_routine_seen

    def test_idempotent_when_no_rollover(self):
        from src import scheduler as sch
        from src.signals.mapper import icao_timezone, today_local

        sch._locked_markets_fired_today.add("mkt_kjfk_1")
        sch._market_to_icao["mkt_kjfk_1"] = "KJFK"
        sch._local_day_seen["KJFK"] = today_local(icao_timezone("KJFK"))

        # Two consecutive calls should be no-ops.
        sch._maybe_clear_per_station_caches()
        sch._maybe_clear_per_station_caches()

        assert "mkt_kjfk_1" in sch._locked_markets_fired_today
        assert "mkt_kjfk_1" in sch._market_to_icao
