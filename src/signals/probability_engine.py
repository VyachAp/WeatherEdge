"""Signal-based probability engine — produces full bucket distributions.

Pure functions with no I/O. Takes a WeatherState and a list of bucket values,
returns a BucketDistribution with probabilities and human-readable reasoning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.config import settings
from src.signals.state_aggregator import WeatherState


# Post-peak trend shift — when `hours_until_peak <= 0` but the METAR trend is
# still rising, Open-Meteo's nominal peak was too early (systematic in hot arid
# cities). Shift the Gaussian center upward by a bounded, solar/cloud-damped
# amount so the distribution has mass above the current observed max.
POST_PEAK_EXTRAPOLATION_HOURS_CAP = 1.5
POST_PEAK_MAX_SHIFT_F = 3.0
POST_PEAK_MIN_TREND_F_PER_HR = 0.5
# Fraction of (rate * hours) carried into the center shift. Matches the
# forecast_exceedance post-peak carry so the alert and the trading Gaussian move
# in lockstep when Open-Meteo's nominal peak was too early.
POST_PEAK_TREND_CARRY_K = 0.75


@dataclass
class BucketDistribution:
    """Full probability distribution over temperature buckets."""

    current_max_f: int
    probabilities: dict[int, float]
    reasoning: list[str] = field(default_factory=list)


def compute_distribution(
    state: WeatherState,
    buckets: list[int],
) -> BucketDistribution:
    """Compute probability distribution over temperature buckets.

    Signal combination:
      1. Baseline Gaussian centered on forecast_peak_f
      2. Time-until-peak: variance widens/narrows
      3. Solar/cloud cap: truncate upside when heating shuts off
      4. Dewpoint trend: adjust variance
      5. METAR trend: shift distribution based on recent observations

    Hard constraint: P(bucket < current_max_f) = 0.
    """
    if not buckets:
        return BucketDistribution(current_max_f=int(state.current_max_f), probabilities={})

    reasoning: list[str] = []
    buckets = sorted(buckets)

    # --- Signal 1: Baseline Gaussian centered on forecast peak ---
    center = state.forecast_peak_f
    reasoning.append(f"baseline: forecast peak {center:.1f}°F")

    # Base sigma: ensemble spread when available, else hours-based schedule.
    base_sigma = _compute_sigma(state, reasoning)

    # --- Signal 2: METAR trend shift ---
    center = _apply_metar_trend(center, state, reasoning)

    # --- Signal 3: Solar/cloud cap ---
    cap = _apply_solar_cloud_cap(state, reasoning)

    # --- Signal 4: Dewpoint adjustment ---
    base_sigma = _apply_dewpoint_adjustment(base_sigma, state, reasoning)

    # --- Build raw distribution ---
    raw: dict[int, float] = {}
    for b in buckets:
        if cap is not None and b > cap:
            raw[b] = 0.0
        else:
            raw[b] = _gaussian_pdf(b, center, base_sigma)

    # --- Hard constraint: zero out P(bucket < current_max) ---
    current_max_bucket = int(state.current_max_f)
    zeroed = 0
    for b in buckets:
        if b < current_max_bucket and raw.get(b, 0) > 0:
            raw[b] = 0.0
            zeroed += 1

    if zeroed > 0:
        reasoning.append(
            f"monotonicity: zeroed {zeroed} buckets below observed max {current_max_bucket}°F"
        )

    # --- Normalize ---
    total = sum(raw.values())
    if total > 0:
        probs = {b: v / total for b, v in raw.items()}
    else:
        # Degenerate case: all mass was zeroed. Put everything on current max bucket.
        probs = {b: 0.0 for b in buckets}
        closest = min(buckets, key=lambda b: abs(b - current_max_bucket))
        probs[closest] = 1.0
        reasoning.append(f"degenerate: all mass on {closest}°F (no upside)")

    return BucketDistribution(
        current_max_f=current_max_bucket,
        probabilities=probs,
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------


def _hours_based_sigma(hours_until_peak: float) -> float:
    """Legacy sigma schedule based on time until forecast peak.

    Used as the fallback when no ensemble spread is available, and as a soft
    floor so σ can't collapse below a sensible minimum when models agree too
    tightly on a trivial central-day forecast.
    """
    if hours_until_peak <= 0:
        return 1.0
    if hours_until_peak <= 2:
        return 1.5
    if hours_until_peak <= 4:
        return 2.5
    return 3.5


def _compute_sigma(state: WeatherState, reasoning: list[str]) -> float:
    """Base σ for the Gaussian: ensemble spread with floors, else hours-based.

    When `state.forecast_sigma_f` is populated (multi-model fetch succeeded),
    it is inflated by `ENSEMBLE_SPREAD_MULTIPLIER` to correct for documented
    NWP under-dispersion, then clipped to [ENSEMBLE_MIN_SIGMA_F,
    ENSEMBLE_MAX_SIGMA_F]. A soft floor at half the hours-based σ prevents
    runaway overconfidence on stable days.
    """
    hours_floor = _hours_based_sigma(state.hours_until_peak)

    if state.forecast_sigma_f is None:
        reasoning.append(
            f"sigma={hours_floor:.2f}°F (hours-based, no ensemble; "
            f"hours_until_peak={state.hours_until_peak:.1f})"
        )
        return hours_floor

    raw = state.forecast_sigma_f * settings.ENSEMBLE_SPREAD_MULTIPLIER
    clipped = max(
        settings.ENSEMBLE_MIN_SIGMA_F,
        min(settings.ENSEMBLE_MAX_SIGMA_F, raw),
    )
    sigma = max(clipped, hours_floor * 0.5)
    reasoning.append(
        f"sigma={sigma:.2f}°F (ensemble spread {state.forecast_sigma_f:.2f}°F "
        f"× {settings.ENSEMBLE_SPREAD_MULTIPLIER} from {state.ensemble_model_count} models, "
        f"hours_floor={hours_floor:.2f}°F)"
    )
    return sigma


def _apply_metar_trend(
    center: float,
    state: WeatherState,
    reasoning: list[str],
) -> float:
    """Shift distribution center based on METAR temperature trend.

    Pre-peak: compare the observed 6h regression to the forecast's implied
    slope to peak. Only the residual (observations outpacing the forecast's
    own rise) shifts the center — matching slopes leave the distribution
    anchored on the forecast peak, because the forecast already accounts for
    that rise. Falls back to the legacy raw-rate shift when the forecast
    slope isn't available.

    Past peak: flat/declining trend locks the center to the observed max.
    """
    rate = state.metar_trend_rate
    forecast_slope = state.forecast_slope_to_peak_f_per_hr

    if state.hours_until_peak > 0:
        if forecast_slope is not None:
            residual_rate = rate - forecast_slope
            if residual_rate > 0.5:
                shift = min(residual_rate * 0.5, 2.0)
                center += shift
                reasoning.append(
                    f"METAR trend residual: obs {rate:+.1f} vs forecast-implied "
                    f"{forecast_slope:+.1f} °F/hr → shift +{shift:.1f}°F"
                )
        elif rate > 0.5:
            shift = min(rate * 0.5, 2.0)
            center += shift
            reasoning.append(
                f"METAR trend: rising {rate:.1f}°F/hr before peak, shift +{shift:.1f}°F"
            )
    elif state.hours_until_peak <= 0 and rate <= 0:
        lock_diff = state.current_max_f - center
        if lock_diff > 0:
            center = state.current_max_f
            reasoning.append(
                f"METAR trend: declining past peak, locked to observed max {state.current_max_f:.0f}°F"
            )
    elif state.hours_until_peak <= 0 and rate > POST_PEAK_MIN_TREND_F_PER_HR:
        hours = POST_PEAK_EXTRAPOLATION_HOURS_CAP
        if state.solar_declining:
            hours *= max(0.0, 1.0 - state.solar_decline_magnitude)
        if state.cloud_rising:
            hours *= max(0.0, 1.0 - state.cloud_rise_magnitude)
        shift = min(rate * hours * POST_PEAK_TREND_CARRY_K, POST_PEAK_MAX_SHIFT_F)
        if shift > 0:
            anchor = max(center, state.current_max_f)
            center = anchor + shift
            reasoning.append(
                f"METAR trend: rising {rate:+.1f}°F/hr past forecast peak, "
                f"shift center to {center:.1f}°F (hours={hours:.2f})"
            )

    if state.current_max_f > center:
        center = state.current_max_f
        reasoning.append(f"observed max {state.current_max_f:.0f}°F exceeds forecast, center adjusted up")

    return center


def _apply_solar_cloud_cap(
    state: WeatherState,
    reasoning: list[str],
) -> int | None:
    """Cap upside when solar is dropping and clouds are rising.

    Returns a temperature cap (bucket value) or None if no cap applies.
    """
    if state.solar_declining and state.cloud_rising:
        cap = int(state.current_max_f)
        reasoning.append(
            f"solar/cloud cap: solar declining ({state.solar_decline_magnitude:.0%}), "
            f"clouds rising ({state.cloud_rise_magnitude:.0%}) → cap at {cap}°F"
        )
        return cap

    if state.solar_declining and state.solar_decline_magnitude > 0.7:
        # Strong solar decline alone is also meaningful
        cap = int(state.current_max_f) + 1
        reasoning.append(
            f"solar decline ({state.solar_decline_magnitude:.0%}) → soft cap at {cap}°F"
        )
        return cap

    return None


def _apply_dewpoint_adjustment(
    sigma: float,
    state: WeatherState,
    reasoning: list[str],
) -> float:
    """Adjust distribution width based on dewpoint trend.

    Rising Td = moisture absorbing energy, reduce upside (tighter sigma).
    Falling Td = dry air, preserve upside.
    """
    dp_rate = state.dewpoint_trend_rate

    if dp_rate > 1.0:
        reduction = min(dp_rate * 0.2, 0.5)
        sigma = max(0.5, sigma - reduction)
        reasoning.append(
            f"dewpoint: rising {dp_rate:.1f}°F/hr, tightened sigma by {reduction:.2f}"
        )
    elif dp_rate < -1.0:
        expansion = min(abs(dp_rate) * 0.1, 0.3)
        sigma += expansion
        reasoning.append(
            f"dewpoint: falling {dp_rate:.1f}°F/hr, expanded sigma by {expansion:.2f}"
        )

    return sigma


def _gaussian_pdf(x: float, mu: float, sigma: float) -> float:
    """Unnormalized Gaussian density (for relative weighting)."""
    if sigma <= 0:
        return 1.0 if x == mu else 0.0
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2)
