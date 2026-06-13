"""Four-stage shadow decision module — the redesigned decision flow.

This is the architectural rethink of the decision pipeline (see the
decision-pipeline diagnosis). It runs in **shadow mode**: it produces a full,
inspectable decision for every evaluated market and writes it to the shadow
ledger, but it places **no orders**. The existing probability + lock paths keep
trading live alongside it until the shadow ledger proves this model beats the
market baseline out-of-sample.

The one edge thesis it is built around, stated falsifiably:

    Our edge is information latency, not better forecasting. We read routine
    METARs — the resolution source — before the market reprices, so near/after
    the daily peak we know the realized intraday path ahead of the market. Our
    edge is the conditional distribution of the daily max GIVEN observations the
    market has not yet priced. We have NO edge from disagreeing with public
    forecasts far from peak.

Four cleanly separated, independently testable stages (all pure):

  1. ``estimate_probability`` — a small, inspectable model that produces a
     *forecast prior* (what the market roughly knows) and an *observational
     update* (the part we may know first), plus the **information-advantage**
     scalar = how far observations moved us off the prior. The lock case is the
     degenerate limit where the update collapses to a point mass.
  2. ``net_edge`` (via ``cost_model``) — prob minus all-in fill cost.
  3. ``decide_bet`` — a SINGLE rule: trade iff ``net_edge > k · σ_edge``, where
     σ_edge is large far from peak (we're mostly guessing) and shrinks as
     observations accumulate. This replaces the ~14-veto wall: edge that comes
     from forecast disagreement rather than observation simply can't clear the
     bar, because σ_edge stays wide.
  4. ``size_bet`` — fractional Kelly on the cost-adjusted odds, shrunk by
     observational confidence, capped by concentration. Size expresses edge AND
     uncertainty — no veto filters needed.

Nothing here touches I/O, the DB, or the Polymarket integration layer. The
orchestrator ``evaluate_shadow`` takes prices/depth as plain arguments so it is
fully unit-testable and replayable over ``forecast_archive``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from src.signals.cost_model import CostBreakdown, effective_cost, net_edge
from src.signals.state_aggregator import WeatherState


# ---------------------------------------------------------------------------
# Tunable parameters (a single frozen bundle so tests can vary them and the
# live wiring can later feed Settings-derived values). Defaults are deliberate,
# documented, and conservative — in shadow they only need to be reasonable.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShadowParams:
    # --- prior σ (forecast-only width) ---
    ensemble_spread_mult: float = 1.3   # inflate ensemble σ (under-dispersive)
    sigma_floor_base_f: float = 2.0     # σ floor at peak (no lead-time padding)
    sigma_lead_slope_f_per_hr: float = 0.30  # lead-time arm: + slope × hrs_to_peak
    sigma_min_f: float = 1.5
    sigma_max_f: float = 8.0

    # --- observational update ---
    typical_lead_hours: float = 8.0     # morning→peak window; sets obs_fraction
    residual_carry_k: float = 0.5       # damping on today's forecast residual
    residual_max_shift_f: float = 4.0   # cap on residual-driven mean shift
    sigma_shrink_max: float = 0.70      # max fractional σ tightening at full obs

    # --- decision rule ---
    edge_k: float = 1.0                 # require net_edge > k · σ_edge
    sigma_edge_min: float = 0.02        # estimation noise floor (near peak)
    sigma_edge_max: float = 0.25        # estimation noise far from peak
    min_depth_usd: float = 10.0         # hard liquidity floor
    min_minutes_to_close: float = 30.0  # hard time floor

    # --- sizing ---
    kelly_fraction: float = 0.10        # fractional Kelly (conservative start)
    concentration_cap: float = 0.05     # max fraction of bankroll per bet


DEFAULT_PARAMS = ShadowParams()


# ---------------------------------------------------------------------------
# Market outcome spec — the minimal description of "what YES means" for a
# market, in °F. Built by the orchestrator from the Market row via existing
# binary_market helpers; kept separate so the math stays pure & I/O-free.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketSpec:
    operator: str                 # above|at_least|below|at_most|exactly|range|bracket
    threshold_f: float | None = None
    low_f: float | None = None    # bracket-like window lower bound (°F)
    high_f: float | None = None   # bracket-like window upper bound (°F)


# ---------------------------------------------------------------------------
# Stage 1 — probability estimation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbEstimate:
    prior_yes: float        # P(YES) from the forecast alone (≈ what market knows)
    updated_yes: float      # P(YES) after the observational update
    info_advantage: float   # signed updated_yes − prior_yes (our edge source)
    obs_fraction: float     # fraction of the day's heating already observed (0..1)
    mean_f: float           # updated distribution mean (°F)
    prior_sigma_f: float
    updated_sigma_f: float
    reasoning: list[str] = field(default_factory=list)


def _phi(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _normal_yes_prob(spec: MarketSpec, mean: float, sigma: float) -> float:
    """Collapse a Normal(mean, sigma) over the daily max onto P(YES) for a market."""
    sigma = max(sigma, 1e-6)
    op = spec.operator
    if op in ("above", "at_least"):
        if spec.threshold_f is None:
            return 0.0
        return 1.0 - _phi((spec.threshold_f - mean) / sigma)
    if op in ("below", "at_most"):
        if spec.threshold_f is None:
            return 0.0
        return _phi((spec.threshold_f - mean) / sigma)
    # bracket-like window [low, high]
    if spec.low_f is None or spec.high_f is None:
        return 0.0
    return max(0.0, _phi((spec.high_f - mean) / sigma) - _phi((spec.low_f - mean) / sigma))


def _obs_fraction(state: WeatherState, p: ShadowParams) -> float:
    """Fraction of the day's heating already locked in (0 = pre-dawn, 1 = past peak).

    This is the spine of the whole architecture: it measures how much of today's
    daily max is *observed* rather than *forecast*. Past the forecast peak the
    max is essentially set (→1.0); far before it we hold only the public forecast
    (→0.0). Lead-time proxy — physically grounded and inspectable.
    """
    if state.hours_until_peak <= 0:
        return 1.0
    frac = 1.0 - (state.hours_until_peak / max(p.typical_lead_hours, 1e-6))
    return max(0.0, min(1.0, frac))


def _prior_sigma(state: WeatherState, p: ShadowParams) -> float:
    """Forecast-only σ with a lead-time-aware floor.

    This is where the long-deferred lead-time σ floor finally lives: a confident
    ensemble must NOT collapse σ far from peak, because the forecast-error
    variance hours out is real regardless of model agreement. floor grows with
    hours-to-peak; ensemble spread is used when it exceeds the floor.
    """
    floor = p.sigma_floor_base_f + p.sigma_lead_slope_f_per_hr * max(0.0, state.hours_until_peak)
    if state.forecast_sigma_f is not None:
        sigma = max(state.forecast_sigma_f * p.ensemble_spread_mult, floor)
    else:
        sigma = floor
    return max(p.sigma_min_f, min(p.sigma_max_f, sigma))


def estimate_probability(
    state: WeatherState,
    spec: MarketSpec,
    params: ShadowParams = DEFAULT_PARAMS,
) -> ProbEstimate:
    """Stage 1: forecast prior + observational update + information advantage."""
    reasoning: list[str] = []
    p = params

    # --- Prior: forecast alone, no observations ---
    prior_mean = state.forecast_peak_f
    prior_sigma = _prior_sigma(state, p)
    prior_yes = _normal_yes_prob(spec, prior_mean, prior_sigma)
    reasoning.append(
        f"prior: N(μ={prior_mean:.1f}, σ={prior_sigma:.1f}) → P(YES)={prior_yes:.3f}"
    )

    # --- Observational update ---
    obs_frac = _obs_fraction(state, p)

    # Mean shift: today's forecast residual (obs − same-hour forecast) tells us
    # the forecast is running hot/cold; carry a damped, bounded fraction forward.
    mean = prior_mean
    if state.forecast_residual_f is not None:
        shift = p.residual_carry_k * state.forecast_residual_f
        shift = max(-p.residual_max_shift_f, min(p.residual_max_shift_f, shift))
        mean = prior_mean + shift
        if abs(shift) >= 0.05:
            reasoning.append(f"update: residual shift {shift:+.1f}°F → μ={mean:.1f}")

    # σ tightening: remaining uncertainty shrinks as the day's heating completes.
    updated_sigma = prior_sigma * (1.0 - p.sigma_shrink_max * obs_frac)
    updated_sigma = max(p.sigma_min_f, updated_sigma)
    reasoning.append(
        f"update: obs_fraction={obs_frac:.2f} → σ {prior_sigma:.1f}→{updated_sigma:.1f}"
    )

    updated_yes = _normal_yes_prob(spec, mean, updated_sigma)

    # Hard monotonic floor: the daily max cannot be below the observed max. This
    # is genuine information (the lock path is its certain limit). Truncate the
    # updated Normal at current_max_f and renormalise the YES mass.
    cmax = state.current_max_f
    tail = 1.0 - _phi((cmax - mean) / max(updated_sigma, 1e-6))
    if tail > 1e-6:
        yes_above = _yes_prob_truncated(spec, mean, updated_sigma, cmax)
        updated_yes = max(0.0, min(1.0, yes_above / tail))
        reasoning.append(
            f"update: monotonic floor at observed max {cmax:.1f}°F"
        )

    info_adv = updated_yes - prior_yes
    reasoning.append(
        f"info_advantage={info_adv:+.3f} (updated {updated_yes:.3f} − prior {prior_yes:.3f})"
    )

    return ProbEstimate(
        prior_yes=prior_yes,
        updated_yes=updated_yes,
        info_advantage=info_adv,
        obs_fraction=obs_frac,
        mean_f=mean,
        prior_sigma_f=prior_sigma,
        updated_sigma_f=updated_sigma,
        reasoning=reasoning,
    )


def _yes_prob_truncated(spec: MarketSpec, mean: float, sigma: float, floor: float) -> float:
    """Un-normalised YES mass of Normal(mean,sigma) restricted to ``max ≥ floor``."""
    sigma = max(sigma, 1e-6)
    op = spec.operator
    if op in ("above", "at_least"):
        if spec.threshold_f is None:
            return 0.0
        lo = max(spec.threshold_f, floor)
        return 1.0 - _phi((lo - mean) / sigma)
    if op in ("below", "at_most"):
        if spec.threshold_f is None:
            return 0.0
        # P(floor ≤ max ≤ threshold); empty when threshold below the floor.
        if spec.threshold_f <= floor:
            return 0.0
        return _phi((spec.threshold_f - mean) / sigma) - _phi((floor - mean) / sigma)
    if spec.low_f is None or spec.high_f is None:
        return 0.0
    lo = max(spec.low_f, floor)
    if spec.high_f <= lo:
        return 0.0
    return _phi((spec.high_f - mean) / sigma) - _phi((lo - mean) / sigma)


# ---------------------------------------------------------------------------
# Stage 3 — bet decision (Stage 2 net_edge comes from cost_model)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BetDecision:
    should_trade: bool
    side: str | None          # "YES" | "NO" | None
    prob: float               # our P(side wins)
    buy_price: float          # side taker cost
    cost: CostBreakdown
    net_edge: float
    sigma_edge: float         # estimation uncertainty the edge must clear
    reason: str               # why we did / didn't trade


def _sigma_edge(est: ProbEstimate, state: WeatherState, p: ShadowParams) -> float:
    """Estimation uncertainty of our probability — wide far from peak, tight near.

    This is what makes the single decision rule encode the edge thesis: a
    far-from-peak bet (low obs_fraction) faces a large σ_edge, so forecast
    disagreement alone can't clear ``net_edge > k·σ_edge``. Thin data (no
    forecast, few routines) inflates it further.
    """
    base = p.sigma_edge_max - (p.sigma_edge_max - p.sigma_edge_min) * est.obs_fraction
    if not state.has_forecast:
        base += 0.05
    if state.routine_count_today < 3:
        base += 0.05
    return base


def decide_bet(
    est: ProbEstimate,
    *,
    yes_bid: float | None,
    yes_ask: float | None,
    yes_mid: float,
    depth_yes_usd: float | None,
    depth_no_usd: float | None,
    minutes_to_close: float | None,
    stake_hint_usd: float = 0.0,
    params: ShadowParams = DEFAULT_PARAMS,
) -> BetDecision:
    """Stage 3: pick the side and decide, on the single net-edge-vs-σ_edge rule.

    ``stake_hint_usd`` lets the slippage term in the cost be size-aware; a 0
    hint (the common shadow case) costs only fee + spread-crossing. This does
    the side-selection + cost math only; the σ_edge gate (which needs the
    WeatherState) is applied by ``_apply_gate`` so each step stays testable.
    """
    yes_buy = yes_ask if yes_ask is not None else yes_mid
    no_buy = (1.0 - yes_bid) if yes_bid is not None else (1.0 - yes_mid)

    yes_cost = effective_cost(
        buy_price=yes_buy, bid=yes_bid, ask=yes_ask,
        depth_usd=depth_yes_usd, stake_usd=stake_hint_usd,
    )
    no_cost = effective_cost(
        buy_price=no_buy,
        bid=(1.0 - yes_ask) if yes_ask is not None else None,
        ask=(1.0 - yes_bid) if yes_bid is not None else None,
        depth_usd=depth_no_usd, stake_usd=stake_hint_usd,
    )
    yes_ne = net_edge(est.updated_yes, yes_cost)
    no_ne = net_edge(1.0 - est.updated_yes, no_cost)

    if no_ne > yes_ne:
        side, prob, buy_price, cost, ne = "NO", 1.0 - est.updated_yes, no_buy, no_cost, no_ne
    else:
        side, prob, buy_price, cost, ne = "YES", est.updated_yes, yes_buy, yes_cost, yes_ne

    return BetDecision(
        should_trade=False, side=side, prob=prob, buy_price=buy_price,
        cost=cost, net_edge=ne, sigma_edge=0.0,
        reason="ungated (call _apply_gate to finish)",
    )


def _apply_gate(
    base: BetDecision, est: ProbEstimate, state: WeatherState,
    *, depth_usd: float | None, minutes_to_close: float | None,
    params: ShadowParams = DEFAULT_PARAMS,
) -> BetDecision:
    p = params
    sigma_edge = _sigma_edge(est, state, p)

    # Hard liquidity / time floors (sanity, not strategy).
    if depth_usd is not None and depth_usd < p.min_depth_usd:
        return replace(base, should_trade=False, sigma_edge=sigma_edge,
                       reason=f"depth ${depth_usd:.0f} < ${p.min_depth_usd:.0f}")
    if minutes_to_close is not None and minutes_to_close < p.min_minutes_to_close:
        return replace(base, should_trade=False, sigma_edge=sigma_edge,
                       reason=f"{minutes_to_close:.0f}m to close < {p.min_minutes_to_close:.0f}m")

    # The single decision rule.
    if base.net_edge > p.edge_k * sigma_edge:
        return replace(base, should_trade=True, sigma_edge=sigma_edge,
                       reason=f"net_edge {base.net_edge:+.3f} > k·σ_edge {p.edge_k * sigma_edge:.3f}")
    return replace(base, should_trade=False, sigma_edge=sigma_edge,
                   reason=f"net_edge {base.net_edge:+.3f} ≤ k·σ_edge {p.edge_k * sigma_edge:.3f}")


# ---------------------------------------------------------------------------
# Stage 4 — sizing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BetSize:
    stake_usd: float
    kelly_fraction_used: float
    uncertainty_shrinkage: float
    reason: str


def size_bet(
    decision: BetDecision,
    est: ProbEstimate,
    bankroll: float,
    params: ShadowParams = DEFAULT_PARAMS,
) -> BetSize:
    """Stage 4: fractional Kelly on cost-adjusted odds × observational confidence.

    Replaces the veto-filter wall with one number that scales in edge AND
    confidence: full Kelly on the price odds, shrunk by ``kelly_fraction`` and
    again by ``obs_fraction`` (so a thin-information bet is sized toward zero
    rather than blocked), capped by ``concentration_cap``.
    """
    p = params
    if not decision.should_trade or bankroll <= 0:
        return BetSize(0.0, 0.0, 0.0, "no-trade")

    price = max(1e-6, min(1.0 - 1e-6, decision.cost.effective_cost))
    prob = max(0.0, min(1.0, decision.prob))
    b = (1.0 - price) / price
    f_star = (prob * b - (1.0 - prob)) / b
    if f_star <= 0:
        return BetSize(0.0, 0.0, est.obs_fraction, "f*≤0 after cost")

    shrink = est.obs_fraction  # observational confidence
    frac = min(f_star * p.kelly_fraction * shrink, p.concentration_cap)
    stake = max(0.0, frac * bankroll)
    return BetSize(
        stake_usd=round(stake, 2),
        kelly_fraction_used=f_star * p.kelly_fraction,
        uncertainty_shrinkage=shrink,
        reason=f"kelly f*={f_star:.3f} × frac={p.kelly_fraction} × shrink={shrink:.2f}",
    )


# ---------------------------------------------------------------------------
# Orchestrator — assembles the four stages. Pure: prices/depth in, decision out.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShadowResult:
    estimate: ProbEstimate
    decision: BetDecision
    size: BetSize


def evaluate_shadow(
    state: WeatherState,
    spec: MarketSpec,
    *,
    yes_bid: float | None,
    yes_ask: float | None,
    yes_mid: float,
    depth_yes_usd: float | None,
    depth_no_usd: float | None,
    minutes_to_close: float | None,
    bankroll: float,
    params: ShadowParams = DEFAULT_PARAMS,
) -> ShadowResult:
    """Run all four stages for one market. Places no orders; returns the decision."""
    est = estimate_probability(state, spec, params)
    base = decide_bet(
        est, yes_bid=yes_bid, yes_ask=yes_ask, yes_mid=yes_mid,
        depth_yes_usd=depth_yes_usd, depth_no_usd=depth_no_usd,
        minutes_to_close=minutes_to_close, params=params,
    )
    # depth on the chosen side for the liquidity floor.
    side_depth = depth_yes_usd if base.side == "YES" else depth_no_usd
    decision = _apply_gate(
        base, est, state, depth_usd=side_depth,
        minutes_to_close=minutes_to_close, params=params,
    )
    size = size_bet(decision, est, bankroll, params)
    return ShadowResult(estimate=est, decision=decision, size=size)
