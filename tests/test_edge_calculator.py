"""Tests for the per-bucket edge calculator."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.signals.edge_calculator import BucketEdge, compute_edges
from src.signals.probability_engine import BucketDistribution


def _make_dist(probs: dict[int, float], current_max: int = 75) -> BucketDistribution:
    return BucketDistribution(
        current_max_f=current_max,
        probabilities=probs,
        reasoning=["test"],
    )


class TestEdgeComputation:
    def test_positive_edge(self):
        # Probabilities bumped above 0.85 to reflect MIN_PROBABILITY raised
        # 2026-05-08 to filter mid-confidence noise. Filter math unchanged.
        dist = _make_dist({80: 0.90, 81: 0.10})
        prices = {80: 0.70, 81: 0.20}
        depths = {80: 100.0, 81: 100.0}
        end = datetime.now(timezone.utc) + timedelta(hours=5)

        edges = compute_edges(dist, prices, routine_count=5, market_end_time=end, orderbook_depths=depths)
        assert len(edges) == 2
        assert edges[0].edge == pytest.approx(0.20)
        assert edges[0].passes is True

    def test_negative_edge_filtered(self):
        dist = _make_dist({80: 0.40})
        prices = {80: 0.50}
        depths = {80: 100.0}
        end = datetime.now(timezone.utc) + timedelta(hours=5)

        edges = compute_edges(dist, prices, routine_count=5, market_end_time=end, orderbook_depths=depths)
        assert len(edges) == 1
        assert edges[0].passes is False
        assert "edge" in edges[0].reject_reason


class TestMinProbabilityFilter:
    def test_low_probability_rejected(self):
        # Prob 0.80 fails the new 0.85 floor; edge=0.30 is fine.
        dist = _make_dist({80: 0.80})
        prices = {80: 0.50}
        depths = {80: 100.0}
        end = datetime.now(timezone.utc) + timedelta(hours=5)

        edges = compute_edges(dist, prices, routine_count=5, market_end_time=end, orderbook_depths=depths)
        assert edges[0].passes is False
        assert "probability" in edges[0].reject_reason

    def test_at_floor_passes(self):
        # 0.85 sits exactly on the floor and should pass.
        dist = _make_dist({80: 0.85})
        prices = {80: 0.70}  # Edge = 0.15 (passes edge filter)
        depths = {80: 100.0}
        end = datetime.now(timezone.utc) + timedelta(hours=5)

        edges = compute_edges(dist, prices, routine_count=5, market_end_time=end, orderbook_depths=depths)
        assert edges[0].passes is True

    def test_high_probability_passes(self):
        dist = _make_dist({80: 0.95})
        prices = {80: 0.70}
        depths = {80: 100.0}
        end = datetime.now(timezone.utc) + timedelta(hours=5)

        edges = compute_edges(dist, prices, routine_count=5, market_end_time=end, orderbook_depths=depths)
        assert edges[0].passes is True


class TestPriceFilter:
    def test_price_too_low(self):
        dist = _make_dist({80: 0.90})
        prices = {80: 0.30}  # Below 0.40
        depths = {80: 100.0}
        end = datetime.now(timezone.utc) + timedelta(hours=5)

        edges = compute_edges(dist, prices, routine_count=5, market_end_time=end, orderbook_depths=depths)
        assert edges[0].passes is False
        assert "price" in edges[0].reject_reason

    def test_price_too_high(self):
        dist = _make_dist({80: 0.99})
        prices = {80: 0.98}  # Above 0.97
        depths = {80: 100.0}
        end = datetime.now(timezone.utc) + timedelta(hours=5)

        edges = compute_edges(dist, prices, routine_count=5, market_end_time=end, orderbook_depths=depths)
        assert edges[0].passes is False


class TestRoutineCountFilter:
    def test_insufficient_metars(self):
        dist = _make_dist({80: 0.90})
        prices = {80: 0.70}
        depths = {80: 100.0}
        end = datetime.now(timezone.utc) + timedelta(hours=5)

        edges = compute_edges(dist, prices, routine_count=2, market_end_time=end, orderbook_depths=depths)
        assert edges[0].passes is False
        assert "routine" in edges[0].reject_reason


class TestMarketCloseFilter:
    def test_closing_too_soon(self):
        dist = _make_dist({80: 0.90})
        prices = {80: 0.70}
        depths = {80: 100.0}
        end = datetime.now(timezone.utc) + timedelta(minutes=15)

        edges = compute_edges(dist, prices, routine_count=5, market_end_time=end, orderbook_depths=depths)
        assert edges[0].passes is False
        assert "closing" in edges[0].reject_reason


class TestDepthFilter:
    def test_insufficient_depth(self):
        dist = _make_dist({80: 0.90})
        prices = {80: 0.70}
        depths = {80: 5.0}  # Below MIN_DEPTH_USD
        end = datetime.now(timezone.utc) + timedelta(hours=5)

        edges = compute_edges(dist, prices, routine_count=5, market_end_time=end, orderbook_depths=depths)
        assert edges[0].passes is False
        assert "depth" in edges[0].reject_reason

    def test_no_depth_data_uses_zero(self):
        dist = _make_dist({80: 0.90})
        prices = {80: 0.70}
        end = datetime.now(timezone.utc) + timedelta(hours=5)

        # No depths dict → defaults to empty
        edges = compute_edges(dist, prices, routine_count=5, market_end_time=end)
        assert edges[0].passes is False
        assert "depth" in edges[0].reject_reason


class TestMultipleBuckets:
    def test_some_pass_some_fail(self):
        dist = _make_dist({78: 0.05, 79: 0.15, 80: 0.90, 81: 0.10})
        prices = {78: 0.90, 79: 0.60, 80: 0.70, 81: 0.50}
        depths = {78: 100.0, 79: 100.0, 80: 100.0, 81: 100.0}
        end = datetime.now(timezone.utc) + timedelta(hours=5)

        edges = compute_edges(dist, prices, routine_count=5, market_end_time=end, orderbook_depths=depths)
        passing = [e for e in edges if e.passes]
        failing = [e for e in edges if not e.passes]
        # Bucket 80: edge = 0.20, prob = 0.90 → passes
        assert any(e.bucket_value == 80 for e in passing)
        # Bucket 78: prob = 0.05 → fails probability filter
        assert any(e.bucket_value == 78 for e in failing)


class TestCheckFiltersMinRoutineOverride:
    """`_check_filters` accepts a `min_routine_count` override so the
    lock-rule path can fire on 2 routines for super-margin EASY locks
    without the scheduler-side gate undoing the relaxation."""

    def test_default_uses_settings_min_routine_count(self):
        from src.signals.edge_calculator import _check_filters

        # routine_count=2 with default min (3) → rejected. Prob bumped
        # above MIN_PROBABILITY=0.85 so it doesn't short-circuit the
        # earlier probability filter.
        reason = _check_filters(
            edge=0.2, prob=0.9, price=0.7,
            routine_count=2, minutes_to_close=120, depth=100.0,
        )
        assert reason is not None
        assert "routine count" in reason

    def test_override_to_two_allows_two_routines(self):
        from src.signals.edge_calculator import _check_filters

        reason = _check_filters(
            edge=0.2, prob=0.9, price=0.7,
            routine_count=2, minutes_to_close=120, depth=100.0,
            min_routine_count=2,
        )
        assert reason is None

    def test_override_still_rejects_below_override(self):
        from src.signals.edge_calculator import _check_filters

        reason = _check_filters(
            edge=0.2, prob=0.9, price=0.7,
            routine_count=1, minutes_to_close=120, depth=100.0,
            min_routine_count=2,
        )
        assert reason is not None
        assert "routine count 1 < 2" in reason


class TestBucketEdgeDirection:
    """`BucketEdge.direction` is BUY_YES by default (back-compat) but is
    set explicitly by `_binary_market_edge` based on which side of the
    market the model disagrees with."""

    def test_default_direction_is_yes(self):
        from src.db.models import TradeDirection

        e = BucketEdge(
            bucket_value=80, our_probability=0.7, market_price=0.5,
            edge=0.2, passes=True, reject_reason=None,
        )
        assert e.direction == TradeDirection.BUY_YES

    def test_explicit_no_direction(self):
        from src.db.models import TradeDirection

        e = BucketEdge(
            bucket_value=80, our_probability=0.7, market_price=0.45,
            edge=0.25, passes=True, reject_reason=None,
            direction=TradeDirection.BUY_NO,
        )
        assert e.direction == TradeDirection.BUY_NO


def _eval_binary(
    *, question, threshold_f, op, our_prob_in_window,
    yes_bid, yes_ask, forecast_peak_f, current_max_f, hours_until_peak,
    end_hours=5, routine_count=5,
):
    """Drive `binary_market_edge` for a single market with the new
    forecast/observation context kwargs that feed the single-bucket NO
    guards. Mass is placed inside the bucket window (for bracket-like ops)
    or at/above threshold (for threshold ops) so `our_prob_yes` is known.
    """
    from src.execution.binary_market import market_range_f
    from src.signals.edge_calculator import binary_market_edge

    market = SimpleNamespace(
        id="m1", question=question,
        parsed_threshold=threshold_f, parsed_operator=op,
        current_yes_price=(yes_bid + yes_ask) / 2,
        end_date=None, outcomes=["Yes", "No"],
    )
    rng = market_range_f(market)
    probs: dict[int, float] = {}
    if rng is not None:
        low, high = rng
        window = list(range(low, high + 1))
        per = our_prob_in_window / len(window)
        for b in window:
            probs[b] = per
        probs[low - 12] = max(0.0, 1.0 - our_prob_in_window)
    else:
        thr = int(round(threshold_f))
        probs = {thr: our_prob_in_window, thr - 1: 1.0 - our_prob_in_window}

    dist = BucketDistribution(
        current_max_f=int(current_max_f), probabilities=probs, reasoning=["test"],
    )
    end_time = datetime.now(timezone.utc) + timedelta(hours=end_hours)
    return binary_market_edge(
        dist, market, end_time, routine_count=routine_count,
        depth_yes=100.0, depth_no_fn=lambda: 100.0,
        yes_bid=yes_bid, yes_ask=yes_ask,
        forecast_peak_f=forecast_peak_f, current_max_f=current_max_f,
        hours_until_peak=hours_until_peak,
    )


class TestSingleBucketNoGuards:
    """Three layered guards stop the incoherent "NO on every adjacent
    single-°C bucket" pattern (e.g. Amsterdam 2026-05-23: NO on 27/28/29°C
    while the blended forecast was pinned at 30°C). All guards are no-ops
    for threshold ops and when the forecast context isn't supplied.
    """

    Q28 = "Will the highest temperature in Amsterdam be 28°C on May 23"
    Q30 = "Will the highest temperature in Amsterdam be 30°C on May 23"

    def test_no_inside_landing_band_rejected(self):
        from src.db.models import TradeDirection

        # forecast 30°C (86.5°F), obs 23°C (73°F), 6.6h pre-peak. The 28°C
        # window (82-83°F) sits inside the landing band [72, 87.5]°F.
        edge = _eval_binary(
            question=self.Q28, threshold_f=82.4, op="exactly",
            our_prob_in_window=0.05, yes_bid=0.45, yes_ask=0.55,
            forecast_peak_f=86.5, current_max_f=73.0, hours_until_peak=6.6,
        )
        assert edge.direction == TradeDirection.BUY_NO
        assert edge.passes is False
        assert "landing band" in (edge.reject_reason or "")

    def test_no_above_band_passes(self):
        from src.db.models import TradeDirection

        # forecast 25.6°C (78°F), obs 23°C (73°F), 2h to peak. The 30°C
        # window (86°F) is clear of the band [72, 79]°F → a genuinely
        # out-of-reach NO bet still fires.
        edge = _eval_binary(
            question=self.Q30, threshold_f=86.0, op="exactly",
            our_prob_in_window=0.10, yes_bid=0.45, yes_ask=0.55,
            forecast_peak_f=78.0, current_max_f=73.0, hours_until_peak=2.0,
        )
        assert edge.direction == TradeDirection.BUY_NO
        assert edge.passes is True
        assert edge.reject_reason is None

    def test_overconfidence_cap_clamps_no_prob(self):
        # P(window)=0 would manufacture NO=1.0; the cap floors YES to
        # 1 - 0.92 = 0.08, so the NO side can't exceed 0.92.
        edge = _eval_binary(
            question=self.Q30, threshold_f=86.0, op="exactly",
            our_prob_in_window=0.0, yes_bid=0.45, yes_ask=0.55,
            forecast_peak_f=78.0, current_max_f=73.0, hours_until_peak=2.0,
        )
        assert edge.our_probability == pytest.approx(0.92)

    def test_peak_relative_lead_gate_uses_hours_until_peak(self):
        # Only 2h to CLOSE but 13h to PEAK → rejected. Bucket is clear of the
        # band, so it's unambiguously the peak-relative lead gate firing
        # (the close-vs-peak bug that let pre-dawn Amsterdam bets through).
        edge = _eval_binary(
            question=self.Q30, threshold_f=86.0, op="exactly",
            our_prob_in_window=0.10, yes_bid=0.45, yes_ask=0.55,
            forecast_peak_f=78.0, current_max_f=73.0, hours_until_peak=13.0,
            end_hours=2,
        )
        assert edge.passes is False
        assert "lead" in (edge.reject_reason or "")
        assert "peak" in (edge.reject_reason or "")

    def test_threshold_market_unaffected_by_guards(self):
        from src.db.models import TradeDirection

        # at_least market, 13h to peak, forecast context supplied: threshold
        # ops bypass the lead gate, the landing-band guard, and the cap.
        edge = _eval_binary(
            question="Will the highest temperature in Amsterdam be 82°F or higher",
            threshold_f=82, op="at_least",
            our_prob_in_window=0.90, yes_bid=0.45, yes_ask=0.55,
            forecast_peak_f=86.0, current_max_f=73.0, hours_until_peak=13.0,
        )
        assert edge.direction == TradeDirection.BUY_YES
        assert edge.passes is True
