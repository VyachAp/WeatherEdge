"""Per-bucket edge calculator with trade filters.

Computes edge for each bucket in a BucketDistribution against market prices,
applies the redesigned trade filters, and returns tradeable edges.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.config import settings
from src.db.models import TradeDirection
from src.signals.probability_engine import BucketDistribution

logger = logging.getLogger(__name__)


def _in_price_valley(effective_price: float) -> bool:
    """True when the side's BUY price lands in the mid "overconfidence valley".

    Per-trade EV is U-shaped in the effective price of the side bought
    (2026-06-06 audit): +EV at the extremes, -EV in
    ``[VALLEY_PRICE_LOW, VALLEY_PRICE_HIGH)``. Drives both the live price-band
    edge policy (P1/P2) in :func:`binary_market_edge` and the
    :func:`shadow_valley_fields` counterfactual.
    """
    return settings.VALLEY_PRICE_LOW <= effective_price < settings.VALLEY_PRICE_HIGH


@dataclass
class BucketEdge:
    """Edge analysis for a single bucket on one side of a binary market.

    Fields are interpreted *in the frame of the side we're proposing to
    trade*: ``our_probability`` is P(side=YES) when ``direction=BUY_YES``
    and P(side=NO) when ``direction=BUY_NO``; ``market_price`` is the
    cost of one share of that side; ``edge = our_probability - market_price``.
    The ``MIN_EDGE``, ``MIN_PROBABILITY``, ``MIN/MAX_ENTRY_PRICE`` filters
    in :func:`_check_filters` all evaluate against these side-effective
    values, so a high-confidence NO trade is gated correctly even when
    the underlying YES probability is low.
    """

    bucket_value: int
    our_probability: float
    market_price: float
    edge: float
    passes: bool
    reject_reason: str | None
    direction: TradeDirection = field(default=TradeDirection.BUY_YES)
    # Engine probability *before* calibration was applied. `our_probability`
    # is the post-calibration value used for sizing/edge gates; this field
    # is the calibration regression's ground-truth input. None when the
    # producer didn't go through `apply_calibration` (e.g. bracket path).
    raw_probability: float | None = None
    # Whether `apply_calibration` actually corrected `our_probability` this
    # evaluation. False when APPLY_CALIBRATION is off, cache is cold, or
    # MIN_CALIBRATION_SAMPLES hasn't been met.
    calibrated: bool = False


def binary_market_edge(
    dist: BucketDistribution,
    market,
    end_time: datetime,
    routine_count: int,
    depth_yes: float,
    depth_no_fn=None,
    yes_bid: float | None = None,
    yes_ask: float | None = None,
    forecast_peak_f: float | None = None,
    current_max_f: float | None = None,
    hours_until_peak: float | None = None,
    has_forecast: bool | None = None,
) -> BucketEdge | None:
    """Pick the best side (YES or NO) of a binary market and gate it.

    Computes ``our_prob_yes`` from the distribution under the operator,
    then evaluates *both* sides at their actual BUY-side cost:

      * YES: price = ``yes_ask`` (what a YES buyer pays)
      * NO:  price = ``1 - yes_bid`` (what a NO buyer pays; equivalent to
             the NO-token ask given the ``YES + NO = 1`` constraint)

    The (yes_bid, yes_ask) quote is optional. When omitted, both sides
    fall back to ``market.current_yes_price`` symmetrically — preserves
    legacy behavior for callers (and tests) that don't have a quote.

    Why asymmetric pricing matters: after a sharp move the orderbook can
    have stale dust on the dead side (e.g. bid=0.20, ask=0.55 on a
    market that's actually trading near YES=0). The arithmetic mid then
    invents a phantom "edge" that wouldn't fill. Charging each side its
    real ask cost makes both sides correctly fail the MIN_EDGE filter.

    For a binary market ``edge_NO == -edge_YES`` only when the spread is
    zero; with a real spread the two sides see independent edges.

    ``depth_no_fn`` is an optional zero-arg callable returning NO-side
    orderbook depth in USD; only invoked when the chosen direction is
    NO, so the additional CLOB call is skipped on the (more common) YES
    side.

    ``forecast_peak_f`` / ``current_max_f`` / ``hours_until_peak`` carry the
    ``WeatherState`` context used by three single-bucket (exactly/range/
    bracket) NO-side guards — defense against the -$164 / 179-trade
    ``model_prob ≥ 0.999`` NO loss class (e.g. betting NO on 27/28/29°C
    while the forecast is pinned at 30°C):
      * Layer 1 — lead gate measured to the forecast **peak** (not close).
      * Layer 2 — refuse NO on a window inside the plausible landing band
        ``[observed_max, forecast_peak]`` (± margin), collapsing toward the
        observed max once past peak.
      * Layer 3 — floor ``our_prob_yes`` so NO confidence ≤
        ``SINGLE_BUCKET_MAX_NO_PROB``.
    All three are no-ops for threshold ops (above/at_least/below/at_most).
    When the context kwargs are ``None`` (legacy callers / tests) Layer 2 is
    skipped and Layer 1 falls back to time-to-close.

    On top of those guards, ``BRACKET_LIKE_NO_DISABLED`` (default False) is a
    **master switch** that rejects any otherwise-passing bracket-like NO
    edge. Fires after Layers 1–3 under the ``reason is None`` guard, so an
    edge that already failed a more specific guard keeps its original
    reason in ``evaluation_logs``. Operational kill for the structurally
    -EV class while σ recalibration is pending.

    Returns the passing-side ``BucketEdge`` if one passes; otherwise the
    higher-edge candidate (with ``passes=False`` and a reject reason),
    so callers can still log what was attempted.
    """
    # Local imports avoid a startup-time cycle on `binary_market`
    # (which lives in src.execution and pulls in this module's parent
    # package indirectly via the polymarket question parser).
    from src.execution.binary_market import market_range_f, operator_class
    from src.signals.calibration import apply_calibration

    op = market.parsed_operator
    mid_price = market.current_yes_price or 0.0
    # Per-side BUY costs. Fall back to the symmetric mid when a real
    # quote isn't supplied (keeps legacy callers + tests working).
    yes_buy_price = yes_ask if (yes_ask is not None and yes_ask > 0) else mid_price
    no_buy_price = (1.0 - yes_bid) if (yes_bid is not None and yes_bid > 0) else (1.0 - mid_price)
    bucket_value: int
    # Bucket window (°F) for bracket-like ops — drives the Layer 2 landing-band
    # NO guard below. Stays None for threshold ops (no window).
    bucket_low: int | None = None
    bucket_high: int | None = None

    if op in ("above", "at_least"):
        threshold = int(market.parsed_threshold)
        bucket_value = threshold
        our_prob_yes = sum(p for b, p in dist.probabilities.items() if b >= threshold)
    elif op in ("below", "at_most"):
        threshold = int(market.parsed_threshold)
        bucket_value = threshold
        our_prob_yes = sum(p for b, p in dist.probabilities.items() if b < threshold)
    elif op in ("exactly", "range", "bracket"):
        rng = market_range_f(market)
        if rng is None:
            return None
        low, high = rng
        bucket_value = (low + high) // 2
        bucket_low, bucket_high = low, high
        our_prob_yes = sum(p for b, p in dist.probabilities.items() if low <= b <= high)
        # Layer 3 — single-bucket overconfidence cap. Floor YES so the NO
        # side can't exceed SINGLE_BUCKET_MAX_NO_PROB. Far from peak the
        # Gaussian collapses P(a single ~2°F window) → ~0 and manufactures
        # full-confidence NO; this caps that tail (the -$164 / 179-trade
        # `model_prob ≥ 0.999` band) independent of lead time.
        our_prob_yes = max(our_prob_yes, 1.0 - settings.SINGLE_BUCKET_MAX_NO_PROB)
    else:
        return None

    our_prob_yes = round(our_prob_yes, 4)
    yes_edge = round(our_prob_yes - yes_buy_price, 4)
    no_prob = round(1.0 - our_prob_yes, 4)
    no_price = round(no_buy_price, 4)
    no_edge = round(no_prob - no_price, 4)
    # Keep the variable name `yes_price` for the BucketEdge.market_price
    # field on the YES branch — reads as "what a YES buyer pays".
    yes_price = round(yes_buy_price, 4)

    now = datetime.now(timezone.utc)
    minutes_to_close = (end_time - now).total_seconds() / 60.0

    # Pick the side whose edge is positive. If both are non-positive,
    # the higher-edge side is still returned (with passes=False) so the
    # caller's log line shows what was considered.
    if no_edge > yes_edge:
        direction = TradeDirection.BUY_NO
        side_prob = no_prob
        side_price = no_price
        side_edge = no_edge
        side_depth = depth_no_fn() if depth_no_fn is not None else 0.0
    else:
        direction = TradeDirection.BUY_YES
        side_prob = our_prob_yes
        side_price = yes_price
        side_edge = yes_edge
        side_depth = depth_yes

    # Apply calibration when enabled — corrects the side-effective
    # probability based on resolved-signal history. No-op when
    # `APPLY_CALIBRATION=False`, when fewer than `MIN_CALIBRATION_SAMPLES`
    # resolved signals exist, or when the cache is stale (refresh happens
    # at the top of each tick).
    side_prob_raw = side_prob
    # Forward the canonical operator class so per-class calibration (when
    # ``PER_OPERATOR_CALIBRATION_ENABLED=True``) can pick the right fit.
    # When the flag is off the helper falls back to the pooled fit, so
    # this is a no-op for legacy behavior.
    side_prob, calibrated = apply_calibration(
        side_prob_raw, operator_class=operator_class(market),
    )
    if calibrated:
        side_edge = round(side_prob - side_price, 4)
        logger.debug(
            "calibrated %s: prob %.3f→%.3f, edge %.3f→%.3f",
            direction.value, side_prob_raw, side_prob,
            round(side_prob_raw - side_price, 4), side_edge,
        )

    # Threshold ops (above/at_least/below/at_most) were never the source of
    # the bracket-overconfidence bleed that forced the global MIN_PROBABILITY
    # / MIN_EDGE up, so they may run on the (optional) looser THRESHOLD_MIN_*
    # floors. Bracket-like ops keep the strict global floors (None override)
    # plus the three single-bucket NO guards below. Both overrides are None by
    # default → no behavior change until set via .env after telemetry validation.
    is_threshold = op in ("above", "at_least", "below", "at_most")
    reason = _check_filters(
        edge=side_edge, prob=side_prob, price=side_price,
        routine_count=routine_count,
        minutes_to_close=minutes_to_close,
        depth=side_depth,
        min_edge=settings.THRESHOLD_MIN_EDGE if is_threshold else None,
        min_probability=settings.THRESHOLD_MIN_PROBABILITY if is_threshold else None,
        min_entry_price=settings.PROBABILITY_MIN_ENTRY_PRICE,
    )

    # Layer 1 — bracket-like (exactly/range/bracket) max-lead gate, measured
    # against the forecast PEAK (not market close). Far from peak the Gaussian
    # collapses P(a single ~2°F bucket) → ~0, manufacturing NO edge that
    # empirically reverts. The lead is measured to peak because a market whose
    # close precedes its peak (e.g. Amsterdam close 12:00 UTC, peak 14:00 UTC)
    # would otherwise read as "near close" while the day's heating is still
    # entirely ahead. Falls back to time-to-close when peak is unknown.
    # Applied after `_check_filters` under the `reason is None` guard so an
    # edge that already fails (e.g. depth) keeps its original reason — the
    # lead reason only marks edges that would otherwise have passed. Threshold
    # ops are never in this tuple.
    lead_h = (
        hours_until_peak if hours_until_peak is not None
        else minutes_to_close / 60.0
    )
    if (
        reason is None
        and op in ("exactly", "range", "bracket")
        and lead_h > settings.EXACTLY_MAX_LEAD_HOURS
    ):
        reason = (
            f"bracket-like lead {lead_h:.1f}h "
            f"> {settings.EXACTLY_MAX_LEAD_HOURS:.0f}h to peak"
        )

    # Layer 2 — plausible-landing-band guard (NO side only). Never bet NO on a
    # single-bucket window the day could still plausibly land in: the band
    # spans the observed max up to the forecast peak (collapsing toward the
    # observed max once past peak, so genuinely out-of-reach NO bets still
    # fire). Primary fix for the adjacent-NO loss class — e.g. NO on
    # 27/28/29°C while the forecast was pinned at 30°C. Skipped when the
    # forecast/observation context isn't supplied (legacy callers / tests).
    if (
        reason is None
        and direction == TradeDirection.BUY_NO
        and op in ("exactly", "range", "bracket")
        and bucket_low is not None
        and bucket_high is not None
        and forecast_peak_f is not None
        and current_max_f is not None
    ):
        margin = settings.SINGLE_BUCKET_NO_BAND_MARGIN_F
        upper_anchor = (
            current_max_f
            if (hours_until_peak is not None and hours_until_peak <= 0)
            else max(forecast_peak_f, current_max_f)
        )
        band_lo = current_max_f - margin
        band_hi = upper_anchor + margin
        # Reject when [bucket_low, bucket_high] overlaps [band_lo, band_hi].
        if bucket_low <= band_hi and bucket_high >= band_lo:
            reason = (
                f"NO inside landing band [{band_lo:.0f},{band_hi:.0f}]°F "
                f"(bucket [{bucket_low},{bucket_high}]°F)"
            )

    # Master switch — bracket-like NO is structurally -EV until lead-time-aware
    # σ recalibration ships (see docs/improvements.md). Live data 2026-05-30:
    # all-time -$260 / -7.4% ROI, last 14d -$346 / -14.3%. Layers 1-3 block
    # specific failure modes but the surviving NO evals still score -0.023
    # EV/$1 in the live tuner. Applied after Layers 1-2 under the `reason is
    # None` guard so a more specific reject reason wins in `evaluation_logs`
    # — this label only marks the otherwise-passing residual.
    if (
        reason is None
        and settings.BRACKET_LIKE_NO_DISABLED
        and direction == TradeDirection.BUY_NO
        and op in ("exactly", "range", "bracket")
    ):
        reason = "bracket-like NO disabled (sigma recalibration pending)"

    # Forecast-required guard for threshold above/at_least BUY_NO — don't bet a
    # forecast-cool NO when there is no forecast. On an Open-Meteo failure the
    # WeatherState degenerates (has_forecast False → forecast_peak_f ==
    # current_max_f, σ NULL), the model prob collapses to ~1.0 and Kelly sizes
    # the max — the exact degenerate state holding 3 of the 4 big post-06-30
    # losses (Shanghai −$27.52). Mirrors the guard the HARD-lock path already
    # enforces (which the probability path lacked). Applied under the `reason is
    # None` guard so a more specific reject reason wins in `evaluation_logs`.
    # No-op when has_forecast is None (legacy callers / tests). See settings
    # entry + memory project_conviction_degenerate_state_bug_2026-06-21.
    if (
        reason is None
        and settings.PROBABILITY_THRESHOLD_NO_REQUIRE_FORECAST
        and direction == TradeDirection.BUY_NO
        and op in ("above", "at_least")
        and has_forecast is False
    ):
        reason = "threshold NO without forecast (degenerate state)"

    # Price-band edge policy — the bot's per-trade EV is U-shaped in the
    # effective price of the side bought: -EV in the mid "overconfidence
    # valley" [VALLEY_PRICE_LOW, VALLEY_PRICE_HIGH), +EV at the extremes
    # (2026-06-06 audit; 60d go-forward valley -0.054 EV/$1). Applied to the
    # chosen side after every other guard under the `reason is None` guard, so
    # a more specific reject reason still wins in `evaluation_logs`. Operator-
    # and side-agnostic (it bands on price, not class). Both layers no-op by
    # default; P1 (block) wins over P2 (edge floor) when both are set.
    if reason is None and _in_price_valley(side_price):
        if settings.VALLEY_BLOCK_ENABLED:
            reason = (
                f"price-valley blocked "
                f"[{settings.VALLEY_PRICE_LOW:.2f},{settings.VALLEY_PRICE_HIGH:.2f}) (P1)"
            )
        elif (
            settings.VALLEY_MIN_EDGE is not None
            and side_edge < settings.VALLEY_MIN_EDGE
        ):
            reason = (
                f"price-valley edge {side_edge:.3f} "
                f"< {settings.VALLEY_MIN_EDGE} floor (P2)"
            )

    return BucketEdge(
        bucket_value=bucket_value,
        our_probability=side_prob,
        market_price=side_price,
        edge=side_edge,
        passes=reason is None,
        reject_reason=reason,
        direction=direction,
        raw_probability=side_prob_raw,
        calibrated=calibrated,
    )


def shadow_valley_fields(effective_price: float, edge: float) -> dict | None:
    """Counterfactual for the price-band P2 refinement (measure-before-flip).

    Returns the dict stamped into ``evaluation_logs.shadow_json.valley`` so a
    future report can join valley evaluations to their resolved outcome (via
    ``market_id`` — no fired trade required, since blocked valley evals are
    still logged) and confirm the ``edge >= SHADOW_VALLEY_MIN_EDGE`` split is
    +EV on out-of-sample data before ``VALLEY_MIN_EDGE`` is set live.
    ``p2_would_block`` is what the P2 policy *would* do at the shadow
    threshold, independent of the live ``VALLEY_BLOCK_ENABLED`` / ``VALLEY_MIN_EDGE``
    flags. Pure; returns ``None`` when ``SHADOW_VALLEY_POLICY_ENABLED`` is off.
    Stamped at the probability-path ``log_evaluation`` call.
    """
    if not settings.SHADOW_VALLEY_POLICY_ENABLED:
        return None
    in_v = _in_price_valley(effective_price)
    return {
        "in_valley": in_v,
        "eff_price": round(effective_price, 4),
        "edge": round(edge, 4),
        "p2_would_block": bool(in_v and edge < settings.SHADOW_VALLEY_MIN_EDGE),
        "p2_min_edge": settings.SHADOW_VALLEY_MIN_EDGE,
    }


def _check_filters(
    edge: float,
    prob: float,
    price: float,
    routine_count: int,
    minutes_to_close: float,
    depth: float,
    min_routine_count: int | None = None,
    min_edge: float | None = None,
    min_probability: float | None = None,
    min_entry_price: float | None = None,
) -> str | None:
    """Return rejection reason or None if all filters pass.

    All inputs are interpreted in the **side-effective frame** — i.e.
    ``prob`` is the probability of the side we're betting on (YES or NO),
    ``price`` is the per-share cost of that side, ``edge = prob - price``,
    and ``depth`` is the depth on the buy book of that side's token. The
    same gate works symmetrically for YES and NO trades because the math
    is identical once expressed in the chosen side's units.

    ``min_routine_count`` overrides ``settings.MIN_ROUTINE_COUNT`` for
    callers that have their own routine-count gate (e.g. the lock-rule path
    can fire on 2 routines for super-margin EASY locks).

    ``min_edge`` / ``min_probability`` override ``settings.MIN_EDGE`` /
    ``settings.MIN_PROBABILITY`` for callers that gate a specific operator
    class (``binary_market_edge`` passes the threshold-only
    ``THRESHOLD_MIN_*`` overrides). ``None`` = use the global setting, so
    legacy callers (lock path) are unchanged.

    ``min_entry_price`` overrides ``settings.MIN_ENTRY_PRICE`` so the
    probability path can concentrate on the near-lock band
    (``PROBABILITY_MIN_ENTRY_PRICE``) without raising the floor on the lock
    path, which legitimately wants cheap-but-certain bets. ``None`` = global.
    """
    eff_min_edge = settings.MIN_EDGE if min_edge is None else min_edge
    eff_min_prob = (
        settings.MIN_PROBABILITY if min_probability is None else min_probability
    )
    eff_min_entry_price = (
        settings.MIN_ENTRY_PRICE if min_entry_price is None else min_entry_price
    )

    if edge < eff_min_edge:
        return f"edge {edge:.4f} < {eff_min_edge}"

    if prob < eff_min_prob:
        return f"probability {prob:.4f} < {eff_min_prob}"

    if price < eff_min_entry_price:
        return f"price {price:.2f} < {eff_min_entry_price}"

    if price > settings.MAX_ENTRY_PRICE:
        return f"price {price:.2f} > {settings.MAX_ENTRY_PRICE}"

    routine_min = (
        settings.MIN_ROUTINE_COUNT if min_routine_count is None
        else min_routine_count
    )
    if routine_count < routine_min:
        return f"routine count {routine_count} < {routine_min}"

    if minutes_to_close < settings.MARKET_CLOSE_BUFFER_MINUTES:
        return f"market closing in {minutes_to_close:.0f}m < {settings.MARKET_CLOSE_BUFFER_MINUTES}m"

    if depth < settings.MIN_DEPTH_USD:
        return f"depth ${depth:.0f} < ${settings.MIN_DEPTH_USD:.0f}"

    return None
