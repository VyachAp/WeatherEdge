"""Unit tests for scheduler-level helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.scheduler import (
    _has_active_trade,
    _log_evaluation,
    _minimal_state_for_easy_lock,
    _should_skip_future_day,
    _upsert_signal,
)
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

        # Probabilities sit above MIN_PROBABILITY=0.85 so the side-selection
        # logic can be exercised without colliding with the prob filter.
        edge, no_calls = self._setup(our_prob_yes=0.90, yes_price=0.70)
        assert edge.direction == TradeDirection.BUY_YES
        assert edge.our_probability == 0.90
        assert edge.market_price == 0.70
        assert edge.edge == pytest.approx(0.20)
        assert edge.passes is True
        # NO depth never fetched on the YES branch.
        assert no_calls == []

    def test_picks_no_when_prob_below_price(self):
        from src.db.models import TradeDirection

        edge, no_calls = self._setup(our_prob_yes=0.10, yes_price=0.55)
        assert edge.direction == TradeDirection.BUY_NO
        # NO frame: prob = 1 - 0.10 = 0.90, price = 1 - 0.55 = 0.45, edge = 0.45
        assert edge.our_probability == 0.90
        assert edge.market_price == pytest.approx(0.45)
        assert edge.edge == pytest.approx(0.45)
        assert edge.passes is True
        # NO depth was lazily fetched.
        assert no_calls == [1]

    def test_no_side_uses_no_depth_for_filter(self):
        # NO branch picked, but NO depth too thin → rejected by depth filter
        # in NO frame, not silently let through using YES depth.
        edge, _ = self._setup(
            our_prob_yes=0.10, yes_price=0.55, depth_yes=500.0, depth_no=2.0,
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
        edge, _ = self._setup(our_prob_yes=0.10, yes_price=0.55)
        assert edge.market_price == pytest.approx(0.45)
        assert edge.passes is True  # 0.45 >= MIN_ENTRY_PRICE (0.40)

    def test_no_side_fails_min_entry_price_when_yes_near_one(self):
        # YES at 0.99 → NO at 0.01, fails MIN_ENTRY_PRICE. Lock-rule path
        # (with its own LOCK_RULE_MIN_PRICE=0.05) is the right tool here,
        # not the probability path. P(YES)=0.05 keeps NO_prob=0.95 well
        # above MIN_PROBABILITY=0.85 so the price filter is the one
        # that fires (not the probability filter).
        edge, _ = self._setup(our_prob_yes=0.05, yes_price=0.99)
        assert edge.passes is False
        assert "price" in (edge.reject_reason or "")

    def test_no_side_passes_min_probability_in_no_frame(self):
        # NO trade with our_prob_yes=0.10 has effective NO prob = 0.90,
        # comfortably above MIN_PROBABILITY=0.85 — passes even though
        # raw P(YES)=0.10 is far below the floor. Proves the side-aware
        # gate works in the NO frame.
        from src.db.models import TradeDirection

        edge, _ = self._setup(our_prob_yes=0.10, yes_price=0.55)
        assert edge.direction == TradeDirection.BUY_NO
        assert edge.our_probability == 0.90  # NO frame
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
        # YES bid=0.40, ask=0.55 (tighter spread). Model P(YES)=0.90 sits
        # comfortably above MIN_PROBABILITY=0.85:
        #   - Mid-based: edge = 0.90 - 0.475 = +0.425 (passes)
        #   - Asymmetric (real ask): edge = 0.90 - 0.55 = +0.35 (still
        #     passes, but smaller and accurate — the point of the test)
        edge = self._eval(
            our_prob_yes=0.90, yes_bid=0.40, yes_ask=0.55,
            op="at_least", threshold=82,
        )
        from src.db.models import TradeDirection
        assert edge.direction == TradeDirection.BUY_YES
        assert edge.market_price == 0.55
        assert abs(edge.edge - 0.35) < 0.001
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


class TestBinaryMarketEdgeLeadGate:
    """Bracket-like (exactly/range/bracket) markets must not be traded too
    far before close. Live data (2026-05-22): exactly-market probability
    trades lose -$126 in the 12-24h lead band but make +$57 in the 0-12h
    band — far from peak the Gaussian collapses P(a single bucket) → ~0,
    inventing NO edge. The gate (settings.EXACTLY_MAX_LEAD_HOURS=12)
    rejects edges evaluated earlier than the cutoff; threshold ops are exempt.
    """

    @pytest.fixture(autouse=True)
    def _disable_master_switch(self, monkeypatch):
        """Pin the bracket-like-NO master switch OFF — this class tests the
        lead gate in isolation, not the master switch that overrides it.
        Needed because the live `.env` ships with the switch flipped on.
        """
        from src.config import settings
        monkeypatch.setattr(settings, "BRACKET_LIKE_NO_DISABLED", False)

    @staticmethod
    def _eval(*, hours_to_close, op="exactly", our_prob_yes=0.10,
              yes_bid=0.45, yes_ask=0.55, threshold=82):
        from src.scheduler import _binary_market_edge
        from src.signals.probability_engine import BucketDistribution

        market = SimpleNamespace(
            id="m1",
            question="Will the highest temperature in Amsterdam be 28°C on April 26",
            parsed_threshold=threshold, parsed_operator=op,
            current_yes_price=(yes_bid + yes_ask) / 2,
            end_date=None, outcomes=["Yes", "No"],
        )
        # Strong NO edge by default: P(NO)=0.90, NO cost 1-0.45=0.55, edge 0.35
        # — clears every standard filter, so only the lead gate can reject it.
        dist = BucketDistribution(
            current_max_f=78,
            probabilities={threshold: our_prob_yes, threshold - 1: 1.0 - our_prob_yes},
            reasoning=["test"],
        )
        end_time = datetime.now(timezone.utc) + timedelta(hours=hours_to_close)
        return _binary_market_edge(
            dist, market, end_time, routine_count=5,
            depth_yes=100.0, depth_no_fn=lambda: 100.0,
            yes_bid=yes_bid, yes_ask=yes_ask,
        )

    def test_exactly_far_lead_rejected(self):
        # 20h before close — passes every standard filter but the lead gate.
        from src.db.models import TradeDirection

        edge = self._eval(hours_to_close=20)
        assert edge.direction == TradeDirection.BUY_NO
        assert edge.passes is False
        assert "lead" in (edge.reject_reason or "")

    def test_exactly_near_lead_passes(self):
        # Same edge 6h before close — within the 12h window → passes.
        edge = self._eval(hours_to_close=6)
        assert edge.passes is True

    def test_exactly_at_boundary_passes(self):
        # Exactly 12h out: gate is strict `>` and the function recomputes
        # `now` microseconds later, so minutes_to_close lands just under 720.
        edge = self._eval(hours_to_close=12)
        assert edge.passes is True

    def test_threshold_far_lead_not_gated(self):
        # at_least market 20h out with a passing YES edge — threshold ops are
        # exempt from the bracket-like lead gate.
        from src.db.models import TradeDirection

        edge = self._eval(hours_to_close=20, op="at_least", our_prob_yes=0.90)
        assert edge.direction == TradeDirection.BUY_YES
        assert edge.passes is True


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


class TestHasActiveTrade:
    """`_has_active_trade` short-circuits any second attempt at firing on
    a (market, direction) pair while a PENDING/OPEN trade is on file —
    the durable, mode-agnostic safety net behind the in-process dedup."""

    @pytest.mark.asyncio
    async def test_returns_true_when_pending_or_open_exists(self):
        from src.db.models import TradeDirection

        session = AsyncMock()
        result = AsyncMock()
        result.scalar_one_or_none = lambda: 42  # any truthy id
        session.execute.return_value = result

        assert await _has_active_trade(session, "mkt_x", TradeDirection.BUY_NO) is True

    @pytest.mark.asyncio
    async def test_returns_false_when_no_active_trade(self):
        from src.db.models import TradeDirection

        session = AsyncMock()
        result = AsyncMock()
        result.scalar_one_or_none = lambda: None
        session.execute.return_value = result

        assert await _has_active_trade(session, "mkt_y", TradeDirection.BUY_YES) is False


class TestUpsertSignal:
    """`_upsert_signal` issues a single ``INSERT ... ON CONFLICT DO
    UPDATE ... RETURNING`` against ``uq_signals_market_direction``. The
    insert-vs-update branching is enforced by Postgres, so these tests
    cover the *statement shape* (values + set_ + returning), and the
    pass-through of the RETURNING clause into the ORM object. Integration
    tests against a real DB cover the conflict semantics themselves."""

    @staticmethod
    def _capture_session(returned_signal):
        """Build an AsyncMock session whose ``scalars(stmt)`` records the
        statement and returns a result that yields ``returned_signal``."""
        captured: dict = {}
        session = AsyncMock()

        async def fake_scalars(stmt, execution_options=None):
            captured["stmt"] = stmt
            captured["execution_options"] = execution_options
            result = AsyncMock()
            result.one = lambda: returned_signal
            return result

        session.scalars = fake_scalars
        return session, captured

    @staticmethod
    def _on_conflict_clauses(stmt) -> tuple[dict, dict]:
        """Extract ``(values_dict, on_conflict_set_dict)`` from a pg insert."""
        # pg_insert(...).values(...) stores values on the compile-state;
        # the simplest robust read is via the parameters dict on the
        # compiled statement.
        compiled = stmt.compile()
        params = dict(compiled.params)
        # ON CONFLICT DO UPDATE set_ payload lives under `_post_values_clause`
        # in 2.0 — but we can inspect more reliably by looking at the dialect
        # element directly.
        on_conflict = stmt._post_values_clause  # type: ignore[attr-defined]
        # update_values_to_set yields (col_name_str, bound_value) tuples.
        set_dict = {name: val for name, val in on_conflict.update_values_to_set}
        return params, set_dict

    @pytest.mark.asyncio
    async def test_statement_carries_all_values_and_returning_signal(self):
        from src.db.models import Signal, TradeDirection

        returned = Signal(
            market_id="mkt_z",
            direction=TradeDirection.BUY_YES,
            model_prob=0.7,
            market_prob=0.4,
            edge=0.3,
            confidence=0.7,
        )
        session, captured = self._capture_session(returned)

        sig = await _upsert_signal(
            session,
            market_id="mkt_z",
            direction=TradeDirection.BUY_YES,
            model_prob=0.7,
            market_prob=0.4,
            edge=0.3,
            confidence=0.7,
        )

        # Helper passes the RETURNING row straight through.
        assert sig is returned
        # populate_existing forces the ORM to refresh from RETURNING even when
        # a stale identity-mapped row exists (matters for repeated ticks).
        assert captured["execution_options"] == {"populate_existing": True}

        values, _ = self._on_conflict_clauses(captured["stmt"])
        assert values["market_id"] == "mkt_z"
        assert values["model_prob"] == 0.7
        assert values["market_prob"] == 0.4
        assert values["edge"] == 0.3
        assert values["confidence"] == 0.7
        # Probability is the default path when kind is unspecified.
        assert values["signal_kind"] == "probability"
        # created_at is set by the helper, not the DB default — so refresh
        # semantics work on the UPDATE branch too.
        assert isinstance(values["created_at"], datetime)

    @pytest.mark.asyncio
    async def test_on_conflict_set_refreshes_all_mutable_fields(self):
        from src.db.models import Signal, TradeDirection

        returned = Signal(
            market_id="mkt_z",
            direction=TradeDirection.BUY_YES,
            model_prob=0.8,
            market_prob=0.45,
            edge=0.35,
            confidence=0.8,
        )
        session, captured = self._capture_session(returned)

        await _upsert_signal(
            session,
            market_id="mkt_z",
            direction=TradeDirection.BUY_YES,
            model_prob=0.8,
            market_prob=0.45,
            edge=0.35,
            confidence=0.8,
        )

        _, set_dict = self._on_conflict_clauses(captured["stmt"])
        # Every mutable field is refreshed on conflict so a repeat tick
        # overwrites the previous evaluation rather than silently retaining
        # stale values.
        for field in (
            "model_prob", "raw_model_prob", "calibrated",
            "market_prob", "edge", "confidence",
            "signal_kind", "lock_branch", "lock_routine_count",
            "lock_observed_max_f", "created_at",
        ):
            assert field in set_dict, f"{field} missing from ON CONFLICT set_"

    @pytest.mark.asyncio
    async def test_lock_fields_land_in_values_and_set(self):
        from src.db.models import Signal, TradeDirection

        returned = Signal(
            market_id="mkt_lock",
            direction=TradeDirection.BUY_NO,
            model_prob=1.0,
            market_prob=0.92,
            edge=0.08,
            confidence=4.5,
            signal_kind="lock",
            lock_branch="easy_super",
            lock_routine_count=2,
            lock_observed_max_f=85.0,
        )
        session, captured = self._capture_session(returned)

        sig = await _upsert_signal(
            session,
            market_id="mkt_lock",
            direction=TradeDirection.BUY_NO,
            model_prob=1.0,
            market_prob=0.92,
            edge=0.08,
            confidence=4.5,
            signal_kind="lock",
            lock_branch="easy_super",
            lock_routine_count=2,
            lock_observed_max_f=85.0,
        )

        # Returned row carries the lock context — same object semantics as
        # the probability path.
        assert sig.signal_kind == "lock"
        assert sig.lock_branch == "easy_super"
        assert sig.lock_routine_count == 2
        assert sig.lock_observed_max_f == 85.0
        # And the lock fields appear in BOTH the values AND the set_ so the
        # UPDATE branch also persists them (a market that flips
        # probability→lock between ticks must overwrite, not keep stale).
        values, set_dict = self._on_conflict_clauses(captured["stmt"])
        assert values["signal_kind"] == "lock"
        assert values["lock_branch"] == "easy_super"
        assert set_dict["lock_branch"] is not None
        assert set_dict["signal_kind"] is not None


class TestLogEvaluation:
    """`_log_evaluation` appends a calibration data point per evaluation
    tick. The append-only design (no UPSERT, no flush) is intentional —
    each tick is its own row."""

    @pytest.mark.asyncio
    async def test_emits_one_row_with_all_fields(self):
        from src.db.models import EvaluationLog, TradeDirection

        added: list = []
        session = AsyncMock()
        session.add = lambda obj: added.append(obj)

        await _log_evaluation(
            session,
            market_id="mkt_eval",
            direction=TradeDirection.BUY_YES,
            signal_kind="probability",
            model_prob=0.72,
            market_prob=0.55,
            edge=0.17,
            passes=True,
            reject_reason=None,
            depth_usd=120.0,
            minutes_to_close=240.0,
            routine_count=4,
        )

        assert len(added) == 1
        row = added[0]
        assert isinstance(row, EvaluationLog)
        assert row.market_id == "mkt_eval"
        assert row.signal_kind == "probability"
        assert row.passes is True
        assert row.reject_reason is None
        assert row.depth_usd == 120.0
        assert row.routine_count == 4
        # Helper does not flush — caller batches.
        session.flush.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejected_evaluation_carries_reason(self):
        from src.db.models import TradeDirection

        added: list = []
        session = AsyncMock()
        session.add = lambda obj: added.append(obj)

        await _log_evaluation(
            session,
            market_id="mkt_eval",
            direction=TradeDirection.BUY_NO,
            signal_kind="lock",
            model_prob=1.0,
            market_prob=0.04,
            edge=0.96,
            passes=False,
            reject_reason="price 0.04 outside [0.05, 0.95]",
            depth_usd=None,
            minutes_to_close=12.5,
            routine_count=3,
        )

        assert len(added) == 1
        row = added[0]
        assert row.passes is False
        assert "outside" in row.reject_reason
        assert row.signal_kind == "lock"


class TestStuckAlertCooldown:
    """``_load_stuck_alert_cooldown`` / ``_persist_stuck_alert_cooldown``
    persist the heartbeat-suppression timer in ``bot_state`` so a process
    restart doesn't reset the cooldown window. Mirror the same pattern
    as ``circuit_breakers._load_paused_until`` / ``_persist_paused_until``.
    """

    @pytest.mark.asyncio
    async def test_load_returns_none_when_row_missing(self):
        from src.scheduler import _load_stuck_alert_cooldown

        session = AsyncMock()
        session.get = AsyncMock(return_value=None)

        assert await _load_stuck_alert_cooldown(session) is None

    @pytest.mark.asyncio
    async def test_load_returns_none_when_value_empty_or_invalid(self):
        from src.scheduler import _load_stuck_alert_cooldown

        # value=None
        session = AsyncMock()
        session.get = AsyncMock(return_value=SimpleNamespace(value=None))
        assert await _load_stuck_alert_cooldown(session) is None

        # value missing at_iso key
        session.get = AsyncMock(return_value=SimpleNamespace(value={}))
        assert await _load_stuck_alert_cooldown(session) is None

        # value with malformed iso string — returns None and warns
        session.get = AsyncMock(
            return_value=SimpleNamespace(value={"at_iso": "not-a-date"})
        )
        assert await _load_stuck_alert_cooldown(session) is None

    @pytest.mark.asyncio
    async def test_load_parses_iso_timestamp(self):
        from src.scheduler import _load_stuck_alert_cooldown

        at = datetime(2026, 5, 18, 12, 30, tzinfo=timezone.utc)
        session = AsyncMock()
        session.get = AsyncMock(
            return_value=SimpleNamespace(value={"at_iso": at.isoformat()})
        )

        got = await _load_stuck_alert_cooldown(session)
        assert got == at

    @pytest.mark.asyncio
    async def test_load_fails_open_when_bot_state_table_missing(self):
        """Same graceful-degrade as circuit_breakers — a missing migration
        must not kill the reconcile job."""
        from sqlalchemy.exc import ProgrammingError

        from src.scheduler import _load_stuck_alert_cooldown

        session = AsyncMock()
        session.get = AsyncMock(
            side_effect=ProgrammingError("missing table", None, None)
        )

        assert await _load_stuck_alert_cooldown(session) is None
        session.rollback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_persist_emits_upsert_with_iso_value(self):
        from src.scheduler import _persist_stuck_alert_cooldown

        session = AsyncMock()
        at = datetime(2026, 5, 18, 12, 30, tzinfo=timezone.utc)

        await _persist_stuck_alert_cooldown(session, at)

        # Statement was executed once (the upsert) — exact bound values
        # are exercised by integration tests against a real DB; here we
        # assert the helper got to the execute() call without raising.
        session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_persist_fails_open_when_bot_state_table_missing(self):
        from sqlalchemy.exc import ProgrammingError

        from src.scheduler import _persist_stuck_alert_cooldown

        session = AsyncMock()
        session.execute = AsyncMock(
            side_effect=ProgrammingError("missing table", None, None)
        )

        # Must not raise — the reconcile loop must keep running even when
        # the bot_state migration hasn't been applied.
        await _persist_stuck_alert_cooldown(
            session, datetime(2026, 5, 18, 12, 30, tzinfo=timezone.utc)
        )
        session.rollback.assert_awaited_once()
