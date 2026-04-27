"""Tests for the per-bucket edge calculator."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

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
        dist = _make_dist({80: 0.70, 81: 0.30})
        prices = {80: 0.50, 81: 0.20}
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
        # Prob 0.45 fails the new 0.50 floor; edge=0.15 is fine.
        dist = _make_dist({80: 0.45})
        prices = {80: 0.30}
        depths = {80: 100.0}
        end = datetime.now(timezone.utc) + timedelta(hours=5)

        edges = compute_edges(dist, prices, routine_count=5, market_end_time=end, orderbook_depths=depths)
        assert edges[0].passes is False
        assert "probability" in edges[0].reject_reason

    def test_at_floor_passes(self):
        # 0.55 used to be rejected (was below 0.60); now passes (>= 0.50).
        dist = _make_dist({80: 0.55})
        prices = {80: 0.45}  # Edge = 0.10 (passes edge filter)
        depths = {80: 100.0}
        end = datetime.now(timezone.utc) + timedelta(hours=5)

        edges = compute_edges(dist, prices, routine_count=5, market_end_time=end, orderbook_depths=depths)
        assert edges[0].passes is True

    def test_high_probability_passes(self):
        dist = _make_dist({80: 0.80})
        prices = {80: 0.50}
        depths = {80: 100.0}
        end = datetime.now(timezone.utc) + timedelta(hours=5)

        edges = compute_edges(dist, prices, routine_count=5, market_end_time=end, orderbook_depths=depths)
        assert edges[0].passes is True


class TestPriceFilter:
    def test_price_too_low(self):
        dist = _make_dist({80: 0.80})
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
        dist = _make_dist({80: 0.80})
        prices = {80: 0.50}
        depths = {80: 100.0}
        end = datetime.now(timezone.utc) + timedelta(hours=5)

        edges = compute_edges(dist, prices, routine_count=2, market_end_time=end, orderbook_depths=depths)
        assert edges[0].passes is False
        assert "routine" in edges[0].reject_reason


class TestMarketCloseFilter:
    def test_closing_too_soon(self):
        dist = _make_dist({80: 0.80})
        prices = {80: 0.50}
        depths = {80: 100.0}
        end = datetime.now(timezone.utc) + timedelta(minutes=15)

        edges = compute_edges(dist, prices, routine_count=5, market_end_time=end, orderbook_depths=depths)
        assert edges[0].passes is False
        assert "closing" in edges[0].reject_reason


class TestDepthFilter:
    def test_insufficient_depth(self):
        dist = _make_dist({80: 0.80})
        prices = {80: 0.50}
        depths = {80: 5.0}  # Below MIN_DEPTH_USD
        end = datetime.now(timezone.utc) + timedelta(hours=5)

        edges = compute_edges(dist, prices, routine_count=5, market_end_time=end, orderbook_depths=depths)
        assert edges[0].passes is False
        assert "depth" in edges[0].reject_reason

    def test_no_depth_data_uses_zero(self):
        dist = _make_dist({80: 0.80})
        prices = {80: 0.50}
        end = datetime.now(timezone.utc) + timedelta(hours=5)

        # No depths dict → defaults to empty
        edges = compute_edges(dist, prices, routine_count=5, market_end_time=end)
        assert edges[0].passes is False
        assert "depth" in edges[0].reject_reason


class TestMultipleBuckets:
    def test_some_pass_some_fail(self):
        dist = _make_dist({78: 0.05, 79: 0.15, 80: 0.70, 81: 0.10})
        prices = {78: 0.90, 79: 0.60, 80: 0.50, 81: 0.50}
        depths = {78: 100.0, 79: 100.0, 80: 100.0, 81: 100.0}
        end = datetime.now(timezone.utc) + timedelta(hours=5)

        edges = compute_edges(dist, prices, routine_count=5, market_end_time=end, orderbook_depths=depths)
        passing = [e for e in edges if e.passes]
        failing = [e for e in edges if not e.passes]
        # Bucket 80: edge = 0.20, prob = 0.70 → passes
        assert any(e.bucket_value == 80 for e in passing)
        # Bucket 78: prob = 0.05 → fails probability filter
        assert any(e.bucket_value == 78 for e in failing)


class TestCheckFiltersMinRoutineOverride:
    """`_check_filters` accepts a `min_routine_count` override so the
    lock-rule path can fire on 2 routines for super-margin EASY locks
    without the scheduler-side gate undoing the relaxation."""

    def test_default_uses_settings_min_routine_count(self):
        from src.signals.edge_calculator import _check_filters

        # routine_count=2 with default min (3) → rejected.
        reason = _check_filters(
            edge=0.2, prob=0.7, price=0.5,
            routine_count=2, minutes_to_close=120, depth=100.0,
        )
        assert reason is not None
        assert "routine count" in reason

    def test_override_to_two_allows_two_routines(self):
        from src.signals.edge_calculator import _check_filters

        reason = _check_filters(
            edge=0.2, prob=0.7, price=0.5,
            routine_count=2, minutes_to_close=120, depth=100.0,
            min_routine_count=2,
        )
        assert reason is None

    def test_override_still_rejects_below_override(self):
        from src.signals.edge_calculator import _check_filters

        reason = _check_filters(
            edge=0.2, prob=0.7, price=0.5,
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
