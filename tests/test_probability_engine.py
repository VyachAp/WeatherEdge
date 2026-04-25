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

    def test_matching_forecast_slope_no_shift(self):
        """Obs slope matches forecast-implied slope → no residual shift."""
        state_match = _make_state(
            metar_trend_rate=2.0, hours_until_peak=4.0,
            forecast_slope_to_peak_f_per_hr=2.0,
            current_max_f=72.0,  # well below center so floor doesn't kick in
        )
        state_none = _make_state(
            metar_trend_rate=0.0, hours_until_peak=4.0,
            forecast_slope_to_peak_f_per_hr=0.0,
            current_max_f=72.0,
        )

        dist_match = compute_distribution(state_match, BUCKETS)
        dist_none = compute_distribution(state_none, BUCKETS)

        ev_match = sum(b * p for b, p in dist_match.probabilities.items())
        ev_none = sum(b * p for b, p in dist_none.probabilities.items())
        # Centers are identical (both = forecast_peak_f) → EVs should match.
        assert ev_match == pytest.approx(ev_none, abs=0.1)
        assert not any("METAR trend" in r for r in dist_match.reasoning)

    def test_outpacing_forecast_shifts_up(self):
        """Obs slope > forecast slope → residual shift applied."""
        state_outpace = _make_state(
            metar_trend_rate=3.0, hours_until_peak=4.0,
            forecast_slope_to_peak_f_per_hr=1.0,
            current_max_f=72.0,
        )
        state_match = _make_state(
            metar_trend_rate=1.0, hours_until_peak=4.0,
            forecast_slope_to_peak_f_per_hr=1.0,
            current_max_f=72.0,
        )

        dist_outpace = compute_distribution(state_outpace, BUCKETS)
        dist_match = compute_distribution(state_match, BUCKETS)
        ev_outpace = sum(b * p for b, p in dist_outpace.probabilities.items())
        ev_match = sum(b * p for b, p in dist_match.probabilities.items())
        assert ev_outpace > ev_match
        assert any("residual" in r for r in dist_outpace.reasoning)

    def test_missing_forecast_slope_uses_legacy(self):
        """No forecast slope → fall back to legacy raw-rate shift."""
        state_legacy = _make_state(
            metar_trend_rate=3.0, hours_until_peak=4.0,
            forecast_slope_to_peak_f_per_hr=None,
            current_max_f=72.0,
        )
        dist = compute_distribution(state_legacy, BUCKETS)
        # Legacy reasoning string includes "rising ... before peak".
        assert any("rising" in r and "before peak" in r for r in dist.reasoning)


class TestPostPeakRisingTrend:
    """Post-peak but still rising: Open-Meteo's nominal peak was too early.

    The Gaussian center should shift above the observed max so the distribution
    has mass on higher buckets. Monotonicity and solar/cloud caps still apply.
    """

    def test_rising_trend_shifts_center_above_observed_max(self):
        # OPKC-like: current_max=91, forecast_peak=88, trend=+1.8°F/hr post-peak.
        # Solar still strong. Expect mass on buckets above current_max.
        state = _make_state(
            current_max_f=91.0,
            forecast_peak_f=88.0,
            metar_trend_rate=1.8,
            hours_until_peak=-0.1,
            solar_declining=False,
        )
        buckets = list(range(85, 98))
        dist = compute_distribution(state, buckets)
        # Some mass above observed max (the point of the fix).
        mass_above = sum(p for b, p in dist.probabilities.items() if b > 91)
        assert mass_above > 0.1
        # Distribution must not collapse entirely onto current_max.
        assert dist.probabilities.get(91, 0.0) < 0.95
        assert any("past forecast peak" in r for r in dist.reasoning)

    def test_solar_cloud_cap_overrides_rising_trend(self):
        # Solar declining AND clouds rising → hard cap at int(current_max_f).
        # Even with a rising trend, no mass should sit above current_max.
        state = _make_state(
            current_max_f=91.0,
            forecast_peak_f=88.0,
            metar_trend_rate=1.8,
            hours_until_peak=-0.1,
            solar_declining=True,
            solar_decline_magnitude=0.8,
            cloud_rising=True,
            cloud_rise_magnitude=0.8,
        )
        buckets = list(range(85, 98))
        dist = compute_distribution(state, buckets)
        mass_above = sum(p for b, p in dist.probabilities.items() if b > 91)
        assert mass_above == 0.0

    def test_monotonicity_still_applied(self):
        # Rising trend past peak must still zero all buckets below current_max.
        state = _make_state(
            current_max_f=91.0,
            forecast_peak_f=88.0,
            metar_trend_rate=2.0,
            hours_until_peak=-0.5,
        )
        buckets = list(range(85, 98))
        dist = compute_distribution(state, buckets)
        for b, p in dist.probabilities.items():
            if b < 91:
                assert p == 0.0, f"bucket {b} has mass {p} below observed max"

    def test_flat_trend_past_peak_no_shift(self):
        # Trend below POST_PEAK_MIN_TREND_F_PER_HR → no new shift; existing
        # behaviour (lock/floor to observed max) applies.
        state_flat = _make_state(
            current_max_f=91.0,
            forecast_peak_f=88.0,
            metar_trend_rate=0.3,
            hours_until_peak=-0.1,
        )
        dist = compute_distribution(state_flat, list(range(85, 98)))
        assert not any("past forecast peak" in r for r in dist.reasoning)

    def test_declining_past_peak_still_locks(self):
        # Cooling past peak: existing "locked to observed max" branch fires.
        state = _make_state(
            current_max_f=91.0,
            forecast_peak_f=88.0,
            metar_trend_rate=-1.0,
            hours_until_peak=-1.0,
        )
        dist = compute_distribution(state, list(range(85, 98)))
        assert any("locked to observed max" in r for r in dist.reasoning)


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


class TestEnsembleSigma:
    """σ driven by ensemble spread instead of the hardcoded hours-based schedule."""

    def test_sigma_uses_ensemble_spread(self):
        state = _make_state(
            forecast_sigma_f=2.0,
            ensemble_model_count=5,
            hours_until_peak=3.0,  # would give hours-based σ=2.5
        )
        dist = compute_distribution(state, BUCKETS)
        # Expected σ = 2.0 * 1.3 = 2.6 (above floor 2.5*0.5=1.25, below max 5.0).
        assert any("ensemble spread 2.00°F" in r for r in dist.reasoning)
        assert any("from 5 models" in r for r in dist.reasoning)

    def test_fallback_to_hours_based_when_no_ensemble(self):
        state = _make_state(
            forecast_sigma_f=None,
            hours_until_peak=1.0,
        )
        dist = compute_distribution(state, BUCKETS)
        assert any(
            "hours-based, no ensemble" in r and "1.50°F" in r
            for r in dist.reasoning
        )

    def test_sigma_clipped_to_max(self):
        """Extreme ensemble spread gets clipped at ENSEMBLE_MAX_SIGMA_F=5.0."""
        state = _make_state(
            forecast_sigma_f=10.0,      # × 1.3 = 13.0, should clip to 5.0
            ensemble_model_count=5,
            hours_until_peak=3.0,
            current_max_f=60.0,         # low so gaussian has room
        )
        dist = compute_distribution(state, BUCKETS)
        # σ=5.0 is wide → peak probability should be modest.
        peak_prob = max(dist.probabilities.values())
        assert peak_prob < 0.20, f"expected wide distribution, got peak {peak_prob}"

    def test_sigma_clipped_to_min(self):
        """Tight ensemble agreement gets floored at ENSEMBLE_MIN_SIGMA_F=1.0."""
        state = _make_state(
            forecast_sigma_f=0.1,       # × 1.3 = 0.13, should floor to 1.0
            ensemble_model_count=5,
            hours_until_peak=3.0,
            current_max_f=60.0,
        )
        dist = compute_distribution(state, BUCKETS)
        # σ floored at 1.0 → reasoning still mentions ensemble path.
        assert any("ensemble spread" in r for r in dist.reasoning)

    def test_tighter_ensemble_gives_narrower_distribution(self):
        """Smaller σ from ensemble → higher peak probability."""
        narrow = _make_state(
            forecast_sigma_f=1.0,       # × 1.3 = 1.3
            ensemble_model_count=5,
            hours_until_peak=3.0,
            current_max_f=60.0,
        )
        wide = _make_state(
            forecast_sigma_f=3.5,       # × 1.3 = 4.55
            ensemble_model_count=5,
            hours_until_peak=3.0,
            current_max_f=60.0,
        )
        dist_narrow = compute_distribution(narrow, BUCKETS)
        dist_wide = compute_distribution(wide, BUCKETS)
        peak_narrow = max(dist_narrow.probabilities.values())
        peak_wide = max(dist_wide.probabilities.values())
        assert peak_narrow > peak_wide


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


class TestClimatePrior:
    """Bayesian Gaussian-Gaussian blend of forecast (likelihood) with the
    climate normal (prior). Modifies both center and σ."""

    def test_disabled_no_change(self):
        """With both prior fields None, output is identical to baseline."""
        baseline = compute_distribution(_make_state(), BUCKETS)
        with_no_prior = compute_distribution(
            _make_state(climate_prior_mean_f=None, climate_prior_std_f=None),
            BUCKETS,
        )
        assert baseline.probabilities == with_no_prior.probabilities
        assert not any("prior:" in r for r in with_no_prior.reasoning)

    def test_pulls_outlier_forecast_toward_normal(self):
        """Forecast 100°F + tight prior 80°F → posterior between, much closer
        to the prior because its variance is lower."""
        # Wide bucket grid so the posterior fits.
        buckets = list(range(70, 110))
        state = _make_state(
            current_max_f=70.0,
            forecast_peak_f=100.0,
            forecast_sigma_f=5.0,            # likelihood σ
            ensemble_model_count=3,
            climate_prior_mean_f=80.0,
            climate_prior_std_f=3.0,         # prior σ — tighter
            metar_trend_rate=0.0,            # no trend shift
            hours_until_peak=0.5,
            solar_declining=False,
            cloud_rising=False,
        )
        dist = compute_distribution(state, buckets)
        # Posterior mean is between 80 and 100, weighted toward 80 because
        # 1/9 (prior precision) > 1/25 (likelihood precision).
        # Closed form: posterior_mean = (80/9 + 100/25) / (1/9 + 1/25) ≈ 84.7
        # The mode of the integer-bucket distribution should be at ~85.
        mode_bucket = max(dist.probabilities, key=lambda b: dist.probabilities[b])
        assert 82 <= mode_bucket <= 87

    def test_tightens_sigma(self):
        """Posterior σ is always less than both inputs (Bayesian invariant),
        floored by CLIMATE_PRIOR_MIN_SIGMA_F."""
        # The reasoning string captures the posterior σ; assert it's <
        # both inputs and >= the floor.
        state = _make_state(
            forecast_sigma_f=4.0,
            ensemble_model_count=3,
            climate_prior_mean_f=82.0,
            climate_prior_std_f=4.0,
        )
        dist = compute_distribution(state, BUCKETS)
        prior_lines = [r for r in dist.reasoning if "prior:" in r]
        assert prior_lines, "expected a prior reasoning line"
        # Pull out posterior σ from "→ posterior μ=X.X°F, σ=Y.YY°F".
        import re
        m = re.search(r"σ=(\d+\.\d+)°F\s*$", prior_lines[0])
        assert m is not None, prior_lines[0]
        post_sigma = float(m.group(1))
        # Bayesian Gaussian blend with equal σ inputs: posterior_var = σ²/2,
        # posterior_σ = σ/√2 ≈ 2.83. Floor is 2.0, so posterior σ ≈ 2.83.
        assert post_sigma < 4.0
        assert post_sigma >= 2.0  # floor

    def test_wide_prior_minimal_pull(self):
        """When prior σ is very wide, the posterior is nearly the
        likelihood (prior carries almost no information)."""
        buckets = list(range(70, 110))
        state = _make_state(
            current_max_f=70.0,
            forecast_peak_f=95.0,
            forecast_sigma_f=2.0,            # tight likelihood
            ensemble_model_count=3,
            climate_prior_mean_f=80.0,
            climate_prior_std_f=20.0,        # very wide prior
            metar_trend_rate=0.0,
            hours_until_peak=0.5,
            solar_declining=False,
            cloud_rising=False,
        )
        dist = compute_distribution(state, buckets)
        mode_bucket = max(dist.probabilities, key=lambda b: dist.probabilities[b])
        # Posterior mean ≈ (95/4 + 80/400) / (1/4 + 1/400) ≈ 94.6 — barely
        # moved from the forecast. Within 1°F.
        assert 94 <= mode_bucket <= 96

    def test_min_sigma_floor_clamps_overshrunk_posterior(self):
        """Two very-tight inputs would naturally combine to σ ≈ 0.7°F; the
        floor should catch that and hold the posterior at MIN_SIGMA."""
        from src.config import settings
        state = _make_state(
            forecast_sigma_f=1.0,            # very tight likelihood
            ensemble_model_count=4,
            climate_prior_mean_f=82.0,
            climate_prior_std_f=1.0,         # very tight prior
        )
        dist = compute_distribution(state, BUCKETS)
        prior_lines = [r for r in dist.reasoning if "prior:" in r]
        assert prior_lines, "expected a prior reasoning line"
        import re
        m = re.search(r"σ=(\d+\.\d+)°F\s*$", prior_lines[0])
        assert m is not None
        post_sigma = float(m.group(1))
        assert post_sigma == pytest.approx(settings.CLIMATE_PRIOR_MIN_SIGMA_F)

    def test_zero_prior_sigma_is_passthrough(self):
        """Degenerate prior σ=0 must not divide-by-zero — fall through cleanly."""
        baseline = compute_distribution(_make_state(), BUCKETS)
        bad_prior = compute_distribution(
            _make_state(climate_prior_mean_f=80.0, climate_prior_std_f=0.0),
            BUCKETS,
        )
        assert baseline.probabilities == bad_prior.probabilities
