"""Unit tests for the first-class cost model."""

from __future__ import annotations

from src.signals.cost_model import (
    CostBreakdown,
    effective_cost,
    expected_slippage,
    net_edge,
)


def test_no_fee_no_slippage_cost_equals_buy_price():
    c = effective_cost(buy_price=0.55, depth_usd=10_000.0, stake_usd=10.0)
    assert abs(c.effective_cost - 0.55) < 1e-9


def test_fee_adds_proportional_cost():
    c = effective_cost(buy_price=0.50, depth_usd=10_000.0, stake_usd=10.0, fee_rate=0.02)
    assert abs(c.fee - 0.01) < 1e-9
    assert abs(c.effective_cost - 0.51) < 1e-9


def test_slippage_grows_with_over_depth_order():
    small = expected_slippage(100.0, 100.0)
    big = expected_slippage(300.0, 100.0)
    assert big > small >= 0.0


def test_slippage_blind_book_costs_max():
    assert expected_slippage(10.0, None) == expected_slippage(10.0, 0.0)


def test_effective_cost_clipped_at_one():
    c = effective_cost(buy_price=0.99, depth_usd=1.0, stake_usd=10_000.0, fee_rate=0.5)
    assert c.effective_cost <= 1.0


def test_net_edge_signs():
    cheap = effective_cost(buy_price=0.40, depth_usd=10_000.0, stake_usd=10.0)
    assert net_edge(0.70, cheap) > 0       # underpriced → +EV
    pricey = effective_cost(buy_price=0.90, depth_usd=10_000.0, stake_usd=10.0)
    assert net_edge(0.70, pricey) < 0      # overpriced → −EV


def test_half_spread_reported():
    c = effective_cost(buy_price=0.60, bid=0.58, ask=0.60, depth_usd=100.0)
    assert isinstance(c, CostBreakdown)
    assert abs(c.half_spread - 0.01) < 1e-9
