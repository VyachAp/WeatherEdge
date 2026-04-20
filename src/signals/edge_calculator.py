"""Per-bucket edge calculator with trade filters.

Computes edge for each bucket in a BucketDistribution against market prices,
applies the redesigned trade filters, and returns tradeable edges.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from src.config import settings
from src.signals.probability_engine import BucketDistribution

logger = logging.getLogger(__name__)

# Edge threshold for the unified pipeline (redesign doc: 0.05)
MIN_EDGE: float = 0.05


@dataclass
class BucketEdge:
    """Edge analysis for a single bucket."""

    bucket_value: int
    our_probability: float
    market_price: float
    edge: float
    passes: bool
    reject_reason: str | None


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

    return edges


def _check_filters(
    edge: float,
    prob: float,
    price: float,
    routine_count: int,
    minutes_to_close: float,
    depth: float,
) -> str | None:
    """Return rejection reason or None if all filters pass."""
    if edge < MIN_EDGE:
        return f"edge {edge:.4f} < {MIN_EDGE}"

    if prob < settings.MIN_PROBABILITY:
        return f"probability {prob:.4f} < {settings.MIN_PROBABILITY}"

    if price < settings.MIN_ENTRY_PRICE:
        return f"price {price:.2f} < {settings.MIN_ENTRY_PRICE}"

    if price > settings.MAX_ENTRY_PRICE:
        return f"price {price:.2f} > {settings.MAX_ENTRY_PRICE}"

    if routine_count < settings.MIN_ROUTINE_COUNT:
        return f"routine count {routine_count} < {settings.MIN_ROUTINE_COUNT}"

    if minutes_to_close < settings.MARKET_CLOSE_BUFFER_MINUTES:
        return f"market closing in {minutes_to_close:.0f}m < {settings.MARKET_CLOSE_BUFFER_MINUTES}m"

    if depth < settings.MIN_DEPTH_USD:
        return f"depth ${depth:.0f} < ${settings.MIN_DEPTH_USD:.0f}"

    return None
