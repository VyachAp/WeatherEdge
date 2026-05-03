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
