"""Tests for the lead-time σ floor: force-override + shadow telemetry (Phase 2).

``_compute_sigma`` gained a ``force_lead_time`` override so
``shadow_sigma_fields`` can compute the counterfactual σ (arm forced on)
without flipping ``SIGMA_FLOOR_LEAD_TIME_ENABLED``. The shadow dict is the
measure-before-flip instrument — it captures the full lead range that live
evals span.
"""

from __future__ import annotations

from unittest.mock import patch

from src.signals import probability_engine as pe
from src.signals.state_aggregator import WeatherState


def _state(hours_until_peak: float, forecast_sigma_f=1.0) -> WeatherState:
    return WeatherState(
        station_icao="KAUS",
        current_max_f=80.0,
        metar_trend_rate=0.0,
        dewpoint_trend_rate=0.0,
        forecast_peak_f=90.0,
        hours_until_peak=hours_until_peak,
        solar_declining=False,
        solar_decline_magnitude=0.0,
        cloud_rising=False,
        cloud_rise_magnitude=0.0,
        routine_count_today=5,
        forecast_sigma_f=forecast_sigma_f,
        ensemble_model_count=4,
    )


# --- _compute_sigma force_lead_time override --------------------------------


def test_force_lead_time_none_matches_setting():
    """force_lead_time=None reproduces the live-setting behavior exactly."""
    st = _state(10.0)
    with patch.object(pe.settings, "SIGMA_FLOOR_LEAD_TIME_ENABLED", False):
        live = pe._compute_sigma(st, [])
        forced_off = pe._compute_sigma(st, [], force_lead_time=False)
        none_path = pe._compute_sigma(st, [], force_lead_time=None)
    assert none_path == live
    assert forced_off == live


def test_force_lead_time_true_widens_far_from_peak():
    """At 10h pre-peak, forcing the arm on lifts the floor above the global."""
    st = _state(10.0)
    with patch.object(pe.settings, "SIGMA_FLOOR_LEAD_TIME_ENABLED", False), \
         patch.object(pe.settings, "SIGMA_LEAD_TIME_SLOPE_F_PER_HR", 0.3):
        off = pe._compute_sigma(st, [], force_lead_time=False)
        on = pe._compute_sigma(st, [], force_lead_time=True)
    # 10h × 0.3 = 3.0°F lead floor > the off-path σ → arm widens it.
    assert on > off
    assert on >= 3.0 - 1e-9


def test_force_lead_time_noop_past_peak():
    """Past peak (h≤0) the arm contributes nothing regardless of force."""
    st = _state(-2.0)
    off = pe._compute_sigma(st, [], force_lead_time=False)
    on = pe._compute_sigma(st, [], force_lead_time=True)
    assert on == off


# --- shadow_sigma_fields ----------------------------------------------------


def test_shadow_disabled_returns_none():
    with patch.object(pe.settings, "SHADOW_SIGMA_LEADTIME_ENABLED", False):
        assert pe.shadow_sigma_fields(_state(10.0)) is None


def test_shadow_none_without_ensemble():
    # hours-based fallback → lead-time arm not in play → no telemetry.
    with patch.object(pe.settings, "SHADOW_SIGMA_LEADTIME_ENABLED", True):
        assert pe.shadow_sigma_fields(_state(10.0, forecast_sigma_f=None)) is None


def test_shadow_far_from_peak_positive_delta():
    with patch.object(pe.settings, "SHADOW_SIGMA_LEADTIME_ENABLED", True), \
         patch.object(pe.settings, "SIGMA_FLOOR_LEAD_TIME_ENABLED", False), \
         patch.object(pe.settings, "SIGMA_LEAD_TIME_SLOPE_F_PER_HR", 0.3):
        out = pe.shadow_sigma_fields(_state(10.0))
    assert out["hours_until_peak"] == 10.0
    assert out["with_arm_f"] > out["live_f"]
    assert out["delta"] > 0.0
    assert out["delta"] == round(out["with_arm_f"] - out["live_f"], 3)


def test_shadow_zero_delta_past_peak():
    with patch.object(pe.settings, "SHADOW_SIGMA_LEADTIME_ENABLED", True):
        out = pe.shadow_sigma_fields(_state(-3.0))
    assert out is not None
    assert out["delta"] == 0.0
