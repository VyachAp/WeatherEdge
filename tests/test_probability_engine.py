"""Tests for the signal-based probability engine."""

from __future__ import annotations

import pytest

from src.signals.probability_engine import BucketDistribution, compute_distribution
from src.signals.state_aggregator import WeatherState


def _make_state(**overrides) -> WeatherState:
    """Create a WeatherState with sensible defaults."""
    defaults = dict(
        station_icao="KJFK",
        current_max_f=75.0,
        metar_trend_rate=1.0,
        dewpoint_trend_rate=0.0,
        forecast_peak_f=80.0,
        hours_until_peak=3.0,
        solar_declining=False,
        solar_decline_magnitude=0.0,
        cloud_rising=False,
        cloud_rise_magnitude=0.0,
        routine_count_today=5,
    )
    defaults.update(overrides)
    return WeatherState(**defaults)


BUCKETS = list(range(70, 90))


class TestMonotonicity:
    """P(bucket < current_max) must be zero."""

    def test_no_mass_below_current_max(self):
        state = _make_state(current_max_f=78.0)
        dist = compute_distribution(state, BUCKETS)
        for b, p in dist.probabilities.items():
            if b < 78:
                assert p == 0.0, f"Bucket {b} has mass {p} below current_max 78"

    def test_all_mass_above_current_max(self):
        state = _make_state(current_max_f=70.0)
        dist = compute_distribution(state, BUCKETS)
        total_above = sum(p for b, p in dist.probabilities.items() if b >= 70)
        assert total_above == pytest.approx(1.0, abs=1e-6)

    def test_high_current_max_concentrates_mass(self):
        """When current_max is near top of buckets, mass concentrates at top."""
        state = _make_state(current_max_f=88.0)
        dist = compute_distribution(state, BUCKETS)
        for b, p in dist.probabilities.items():
            if b < 88:
                assert p == 0.0

    def test_current_max_above_all_buckets(self):
        """Edge case: current_max exceeds all buckets."""
        state = _make_state(current_max_f=95.0)
        dist = compute_distribution(state, list(range(70, 90)))
        total = sum(dist.probabilities.values())
        assert total == pytest.approx(1.0, abs=1e-6)


class TestNormalization:
    """Distribution must sum to 1.0."""

    def test_probabilities_sum_to_one(self):
        state = _make_state()
        dist = compute_distribution(state, BUCKETS)
        total = sum(dist.probabilities.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_empty_buckets(self):
        state = _make_state()
        dist = compute_distribution(state, [])
        assert dist.probabilities == {}

    def test_single_bucket(self):
        state = _make_state(current_max_f=75.0)
        dist = compute_distribution(state, [80])
        assert dist.probabilities[80] == pytest.approx(1.0)


class TestTimeDependency:
    """Distribution should be wider when peak is far away."""

    def test_past_peak_tight(self):
        state = _make_state(hours_until_peak=0.0, current_max_f=80.0, forecast_peak_f=80.0)
        dist = compute_distribution(state, BUCKETS)
        # Most mass should be concentrated near current_max
        top_bucket = max(dist.probabilities, key=dist.probabilities.get)
        assert abs(top_bucket - 80) <= 1

    def test_far_from_peak_wider(self):
        state_far = _make_state(hours_until_peak=6.0, current_max_f=72.0)
        state_near = _make_state(hours_until_peak=1.0, current_max_f=72.0)
        dist_far = compute_distribution(state_far, BUCKETS)
        dist_near = compute_distribution(state_near, BUCKETS)

        # Far distribution should have lower peak probability (wider spread)
        peak_far = max(dist_far.probabilities.values())
        peak_near = max(dist_near.probabilities.values())
        assert peak_far < peak_near


class TestSolarCloudCap:
    """Solar decline + cloud rise should cap upside."""

    def test_cap_applied(self):
        state = _make_state(
            solar_declining=True,
            solar_decline_magnitude=0.6,
            cloud_rising=True,
            cloud_rise_magnitude=0.3,
            current_max_f=78.0,
        )
        dist = compute_distribution(state, BUCKETS)
        # Buckets above current_max should have zero mass
        for b, p in dist.probabilities.items():
            if b > 78:
                assert p == 0.0, f"Bucket {b} should be capped"

    def test_no_cap_without_solar_decline(self):
        state = _make_state(
            solar_declining=False,
            cloud_rising=True,
            cloud_rise_magnitude=0.5,
            current_max_f=75.0,
        )
        dist = compute_distribution(state, BUCKETS)
        # Some buckets above current_max should have mass
        above_mass = sum(p for b, p in dist.probabilities.items() if b > 75)
        assert above_mass > 0


class TestMETARTrend:
    """Rising METAR trend before peak should shift distribution up."""

    def test_rising_trend_shifts_up(self):
        state_rising = _make_state(metar_trend_rate=3.0, hours_until_peak=4.0)
        state_flat = _make_state(metar_trend_rate=0.0, hours_until_peak=4.0)

        dist_rising = compute_distribution(state_rising, BUCKETS)
        dist_flat = compute_distribution(state_flat, BUCKETS)

        # Expected value should be higher with rising trend
        ev_rising = sum(b * p for b, p in dist_rising.probabilities.items())
        ev_flat = sum(b * p for b, p in dist_flat.probabilities.items())
        assert ev_rising > ev_flat


class TestDewpointEffect:
    """Rising dewpoint should reduce upside."""

    def test_rising_dewpoint_tightens(self):
        state_wet = _make_state(dewpoint_trend_rate=2.0)
        state_dry = _make_state(dewpoint_trend_rate=-2.0)

        dist_wet = compute_distribution(state_wet, BUCKETS)
        dist_dry = compute_distribution(state_dry, BUCKETS)

        # Wet distribution should have higher peak probability (tighter)
        peak_wet = max(dist_wet.probabilities.values())
        peak_dry = max(dist_dry.probabilities.values())
        assert peak_wet >= peak_dry


class TestReasoning:
    """Every distribution should have reasoning entries."""

    def test_reasoning_not_empty(self):
        state = _make_state()
        dist = compute_distribution(state, BUCKETS)
        assert len(dist.reasoning) > 0

    def test_baseline_always_present(self):
        state = _make_state()
        dist = compute_distribution(state, BUCKETS)
        assert any("baseline" in r for r in dist.reasoning)

    def test_cap_mentioned_when_active(self):
        state = _make_state(
            solar_declining=True,
            solar_decline_magnitude=0.6,
            cloud_rising=True,
            cloud_rise_magnitude=0.3,
        )
        dist = compute_distribution(state, BUCKETS)
        assert any("cap" in r.lower() for r in dist.reasoning)
