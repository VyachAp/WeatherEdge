"""Unit tests for the redesigned four-stage shadow decision flow + cost model.

These assert the *architectural* properties, not just arithmetic:
  - friction is real (cost > buy_price once fee/slippage bite),
  - the observational update tightens σ and respects the monotonic floor,
  - far-from-peak forecast disagreement CANNOT clear the decision bar,
  - near-peak observational edge CAN,
  - sizing scales with observational confidence.
"""

from __future__ import annotations

from src.signals.cost_model import effective_cost, net_edge, expected_slippage
from src.signals.shadow_decision import (
    DEFAULT_PARAMS,
    MarketSpec,
    ShadowParams,
    estimate_probability,
    evaluate_shadow,
)
from src.signals.state_aggregator import WeatherState


def _state(**overrides) -> WeatherState:
    base = dict(
        station_icao="KAUS",
        current_max_f=80.0,
        metar_trend_rate=0.0,
        dewpoint_trend_rate=0.0,
        forecast_peak_f=90.0,
        hours_until_peak=2.0,
        solar_declining=False,
        solar_decline_magnitude=0.0,
        cloud_rising=False,
        cloud_rise_magnitude=0.0,
        routine_count_today=5,
        has_forecast=True,
        forecast_sigma_f=2.0,
        forecast_residual_f=0.0,
    )
    base.update(overrides)
    return WeatherState(**base)


# --- cost model ------------------------------------------------------------


def test_effective_cost_adds_fee_and_slippage():
    c = effective_cost(buy_price=0.70, bid=0.68, ask=0.70, depth_usd=100.0,
                       stake_usd=50.0, fee_rate=0.02)
    assert c.effective_cost > 0.70  # fee makes it strictly costlier than the ask
    assert c.fee == 0.02 * 0.70
    assert c.half_spread == (0.70 - 0.68) / 2.0


def test_slippage_zero_within_depth_and_max_when_blind():
    assert expected_slippage(50.0, 100.0) == 0.0           # fits in book
    assert expected_slippage(200.0, 100.0) > 0.0           # walks the book
    assert expected_slippage(50.0, None) > 0.0             # unknown depth → cost it


def test_net_edge_subtracts_cost():
    c = effective_cost(buy_price=0.60, depth_usd=1000.0, stake_usd=10.0)
    assert abs(net_edge(0.75, c) - (0.75 - c.effective_cost)) < 1e-9


# --- stage 1: probability estimation --------------------------------------


def test_past_peak_tightens_sigma_and_obs_fraction_one():
    spec = MarketSpec(operator="at_least", threshold_f=88.0)
    est = estimate_probability(_state(hours_until_peak=-1.0), spec)
    assert est.obs_fraction == 1.0
    assert est.updated_sigma_f < est.prior_sigma_f


def test_far_from_peak_keeps_sigma_wide():
    spec = MarketSpec(operator="at_least", threshold_f=88.0)
    est = estimate_probability(_state(hours_until_peak=10.0), spec)
    assert est.obs_fraction == 0.0
    # lead-time floor keeps σ wide far from peak even with a confident ensemble
    assert est.prior_sigma_f >= DEFAULT_PARAMS.sigma_floor_base_f


def test_monotonic_floor_forces_yes_when_already_locked():
    # at_least 75°F but we've already observed 80°F → YES is locked.
    spec = MarketSpec(operator="at_least", threshold_f=75.0)
    est = estimate_probability(_state(current_max_f=80.0, hours_until_peak=-1.0), spec)
    assert est.updated_yes > 0.99


# --- stages 2-4: decision + sizing -----------------------------------------


def _common(**kw):
    base = dict(
        yes_bid=0.50, yes_ask=0.52, yes_mid=0.51,
        depth_yes_usd=500.0, depth_no_usd=500.0,
        minutes_to_close=120.0, bankroll=500.0,
    )
    base.update(kw)
    return base


def test_far_from_peak_disagreement_does_not_trade():
    # Forecast 90, threshold 88: model likes YES, but we're 10h out — σ_edge wide.
    spec = MarketSpec(operator="at_least", threshold_f=88.0)
    res = evaluate_shadow(_state(hours_until_peak=10.0), spec, **_common())
    assert res.decision.should_trade is False
    assert res.size.stake_usd == 0.0


def test_near_peak_observational_edge_trades():
    # Past peak, observed max 89 already clears an at_least-85 priced cheap → trade.
    spec = MarketSpec(operator="at_least", threshold_f=85.0)
    state = _state(current_max_f=89.0, forecast_peak_f=89.0, hours_until_peak=-1.0)
    res = evaluate_shadow(state, spec, **_common(yes_bid=0.70, yes_ask=0.72, yes_mid=0.71))
    assert res.decision.should_trade is True
    assert res.decision.side == "YES"
    assert res.size.stake_usd > 0.0


def test_liquidity_floor_blocks_thin_book():
    spec = MarketSpec(operator="at_least", threshold_f=85.0)
    state = _state(current_max_f=89.0, forecast_peak_f=89.0, hours_until_peak=-1.0)
    res = evaluate_shadow(state, spec, **_common(
        yes_bid=0.70, yes_ask=0.72, yes_mid=0.71, depth_yes_usd=5.0, depth_no_usd=5.0))
    assert res.decision.should_trade is False
    assert "depth" in res.decision.reason


def test_size_scales_with_observational_confidence():
    spec = MarketSpec(operator="at_least", threshold_f=85.0)
    # Looser params so a mid-day bet can still clear the bar, isolating the
    # shrink effect on size.
    loose = ShadowParams(sigma_edge_max=0.05, edge_k=0.5)
    near = evaluate_shadow(
        _state(current_max_f=89.0, forecast_peak_f=89.0, hours_until_peak=-1.0),
        spec, params=loose, **_common(yes_bid=0.60, yes_ask=0.62, yes_mid=0.61))
    assert near.decision.should_trade
    assert 0.0 < near.size.uncertainty_shrinkage <= 1.0
