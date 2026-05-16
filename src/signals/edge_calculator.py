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

# Re-exported for backwards compatibility — `settings.MIN_EDGE` is the
# source of truth (default 0.05). Imports of `MIN_EDGE` from this module
# now resolve to the configured value rather than a hardcoded constant.
MIN_EDGE: float = settings.MIN_EDGE


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


def compute_edges(
    distribution: BucketDistribution,
    market_prices: dict[int, float],
    routine_count: int,
    market_end_time: datetime,
    orderbook_depths: dict[int, float] | None = None,
) -> list[BucketEdge]:
    """Compute edge for each bucket and apply all trade filters.

    Filters (from redesign doc):
      - edge >= 0.05
      - our_probability >= 0.60
      - 0.40 <= market_price <= 0.97
      - orderbook_depth >= $50
      - routine_count >= 3
      - market not closing within 30 minutes
    """
    if orderbook_depths is None:
        orderbook_depths = {}

    now = datetime.now(timezone.utc)
    minutes_to_close = (market_end_time - now).total_seconds() / 60.0

    edges: list[BucketEdge] = []

    for bucket, prob in distribution.probabilities.items():
        price = market_prices.get(bucket)
        if price is None:
            continue

        edge = prob - price

        # Apply filters
        reason = _check_filters(
            edge=edge,
            prob=prob,
            price=price,
            routine_count=routine_count,
            minutes_to_close=minutes_to_close,
            depth=orderbook_depths.get(bucket, 0.0),
        )

        edges.append(BucketEdge(
            bucket_value=bucket,
            our_probability=round(prob, 4),
            market_price=price,
            edge=round(edge, 4),
            passes=reason is None,
            reject_reason=reason,
        ))

    if edges:
        parts = []
        for e in edges:
            tag = "PASS" if e.passes else (e.reject_reason or "FAIL")
            parts.append(f"{e.bucket_value}→{e.edge:+.3f}(ours={e.our_probability:.2f},mkt={e.market_price:.2f},{tag})")
        logger.info("edges: %s", " | ".join(parts))

    return edges


def binary_market_edge(
    dist: BucketDistribution,
    market,
    end_time: datetime,
    routine_count: int,
    depth_yes: float,
    depth_no_fn=None,
    yes_bid: float | None = None,
    yes_ask: float | None = None,
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

    Returns the passing-side ``BucketEdge`` if one passes; otherwise the
    higher-edge candidate (with ``passes=False`` and a reject reason),
    so callers can still log what was attempted.
    """
    # Local imports avoid a startup-time cycle on `binary_market`
    # (which lives in src.execution and pulls in this module's parent
    # package indirectly via the polymarket question parser).
    from src.execution.binary_market import market_range_f
    from src.signals.calibration import apply_calibration

    op = market.parsed_operator
    mid_price = market.current_yes_price or 0.0
    # Per-side BUY costs. Fall back to the symmetric mid when a real
    # quote isn't supplied (keeps legacy callers + tests working).
    yes_buy_price = yes_ask if (yes_ask is not None and yes_ask > 0) else mid_price
    no_buy_price = (1.0 - yes_bid) if (yes_bid is not None and yes_bid > 0) else (1.0 - mid_price)
    bucket_value: int

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
        our_prob_yes = sum(p for b, p in dist.probabilities.items() if low <= b <= high)
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
    side_prob, calibrated = apply_calibration(side_prob_raw)
    if calibrated:
        side_edge = round(side_prob - side_price, 4)
        logger.debug(
            "calibrated %s: prob %.3f→%.3f, edge %.3f→%.3f",
            direction.value, side_prob_raw, side_prob,
            round(side_prob_raw - side_price, 4), side_edge,
        )

    reason = _check_filters(
        edge=side_edge, prob=side_prob, price=side_price,
        routine_count=routine_count,
        minutes_to_close=minutes_to_close,
        depth=side_depth,
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


def _check_filters(
    edge: float,
    prob: float,
    price: float,
    routine_count: int,
    minutes_to_close: float,
    depth: float,
    min_routine_count: int | None = None,
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
    """
    if edge < settings.MIN_EDGE:
        return f"edge {edge:.4f} < {settings.MIN_EDGE}"

    if prob < settings.MIN_PROBABILITY:
        return f"probability {prob:.4f} < {settings.MIN_PROBABILITY}"

    if price < settings.MIN_ENTRY_PRICE:
        return f"price {price:.2f} < {settings.MIN_ENTRY_PRICE}"

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
