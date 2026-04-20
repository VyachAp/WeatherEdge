"""Signal-based probability engine — produces full bucket distributions.

Pure functions with no I/O. Takes a WeatherState and a list of bucket values,
returns a BucketDistribution with probabilities and human-readable reasoning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.signals.state_aggregator import WeatherState


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

    # Base sigma depends on time until peak
    base_sigma = _compute_sigma(state.hours_until_peak, reasoning)

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


def _compute_sigma(hours_until_peak: float, reasoning: list[str]) -> float:
    """Compute Gaussian sigma based on time until forecast peak.

    Wider when peak is hours away; tightens as peak passes.
    """
    if hours_until_peak <= 0:
        sigma = 1.0
        reasoning.append("time: past peak hour, tight distribution (sigma=1.0)")
    elif hours_until_peak <= 2:
        sigma = 1.5
        reasoning.append(f"time: {hours_until_peak:.1f}h to peak, moderate width (sigma=1.5)")
    elif hours_until_peak <= 4:
        sigma = 2.5
        reasoning.append(f"time: {hours_until_peak:.1f}h to peak, wider (sigma=2.5)")
    else:
        sigma = 3.5
        reasoning.append(f"time: {hours_until_peak:.1f}h to peak, wide distribution (sigma=3.5)")
    return sigma


def _apply_metar_trend(
    center: float,
    state: WeatherState,
    reasoning: list[str],
) -> float:
    """Shift distribution center based on METAR temperature trend.

    Rising trend before peak = shift up. Flat/declining past peak = lock in.
    """
    rate = state.metar_trend_rate

    if state.hours_until_peak > 0 and rate > 0.5:
        # Rising before peak: meaningful upside signal
        shift = min(rate * 0.5, 2.0)  # Cap shift at 2°F
        center += shift
        reasoning.append(
            f"METAR trend: rising {rate:.1f}°F/hr before peak, shift +{shift:.1f}°F"
        )
    elif state.hours_until_peak <= 0 and rate <= 0:
        # Past peak and declining: strong lock-in
        lock_diff = state.current_max_f - center
        if lock_diff > 0:
            center = state.current_max_f
            reasoning.append(
                f"METAR trend: declining past peak, locked to observed max {state.current_max_f:.0f}°F"
            )

    # If current observed max exceeds forecast center, pull center up
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
