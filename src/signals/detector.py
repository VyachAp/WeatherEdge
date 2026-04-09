"""Edge detection and signal filtering.

Computes the edge between model consensus and market price, applies quality
filters, persists actionable signals to the database, and returns them
ranked by expected value.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from sqlalchemy import select

from src.config import settings
from src.db.engine import async_session as session_factory
from src.db.models import Signal, TradeDirection
from src.signals.consensus import compute_calibrated_consensus
from src.signals.mapper import map_all_markets, map_short_range_markets

if TYPE_CHECKING:
    from src.db.models import Market

    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Filter thresholds
# ---------------------------------------------------------------------------

MIN_LIQUIDITY: float = 300.0
MIN_CONFIDENCE: float = 0.55
MIN_VOLUME: float = 100.0
MIN_DAYS: int = 1
MAX_DAYS: int = 7

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


SHORT_RANGE_MIN_EDGE_DISCOUNT: float = 0.75  # 25% reduction when aviation confirms


@dataclass
class ActionableSignal:
    """A signal that has passed all quality filters."""

    market_id: str
    question: str
    direction: TradeDirection
    edge: float
    confidence: float
    consensus_prob: float
    market_prob: float
    gfs_prob: float | None
    ecmwf_prob: float | None
    aviation_prob: float | None
    days_to_resolution: int
    hours_to_resolution: float
    ev_score: float


# ---------------------------------------------------------------------------
# Edge computation
# ---------------------------------------------------------------------------


def compute_edge(
    consensus_prob: float,
    market_prob: float,
) -> tuple[float, TradeDirection]:
    """Compute edge and optimal direction.

    Returns ``(edge, direction)`` where *edge* is always the absolute value
    of the larger side.
    """
    edge_yes = consensus_prob - market_prob
    edge_no = market_prob - consensus_prob  # equivalent: (1-consensus) - (1-market)

    if edge_yes >= edge_no:
        return (edge_yes, TradeDirection.BUY_YES)
    return (edge_no, TradeDirection.BUY_NO)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def passes_filters(
    edge: float,
    confidence: float,
    days_to_resolution: int,
    market: Market,
    aviation_prob: float | None = None,
    hours_to_resolution: float | None = None,
) -> bool:
    """Apply tiered signal quality filters.

    Three tiers based on aviation confirmation and forecast horizon:
    - Ultra-short (aviation + ≤12h): most relaxed thresholds
    - Short-range (aviation + ≤30h): moderate relaxation
    - Standard (no aviation): original thresholds
    """
    has_aviation = aviation_prob is not None and hours_to_resolution is not None and hours_to_resolution <= 30

    if has_aviation and hours_to_resolution <= 12:
        # Ultra-short: aviation data is near-observation quality
        min_edge = settings.MIN_EDGE * settings.SR_MIN_EDGE_DISCOUNT
        min_liq = settings.SR_MIN_LIQUIDITY
        min_vol = settings.SR_MIN_VOLUME
        min_days = 0
    elif has_aviation:
        # Short-range (12-30h): moderate relaxation
        min_edge = settings.MIN_EDGE * SHORT_RANGE_MIN_EDGE_DISCOUNT
        min_liq = 200.0
        min_vol = 75.0
        min_days = 0
    else:
        # Standard: original thresholds
        min_edge = settings.MIN_EDGE
        min_liq = MIN_LIQUIDITY
        min_vol = MIN_VOLUME
        min_days = MIN_DAYS

    if abs(edge) < min_edge:
        return False
    if (market.liquidity or 0) < min_liq:
        return False
    if confidence < MIN_CONFIDENCE:
        return False
    if not (min_days <= days_to_resolution <= MAX_DAYS):
        return False
    if (market.volume or 0) < min_vol:
        return False
    return True


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


async def persist_signal(
    signal: ActionableSignal,
    session: AsyncSession,
) -> Signal:
    """Write a Signal row to the database."""
    row = Signal(
        market_id=signal.market_id,
        model_prob=signal.consensus_prob,
        market_prob=signal.market_prob,
        edge=signal.edge,
        direction=signal.direction,
        confidence=signal.confidence,
        gfs_prob=signal.gfs_prob,
        ecmwf_prob=signal.ecmwf_prob,
        aviation_prob=signal.aviation_prob,
    )
    session.add(row)
    return row


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


async def detect_signals(
    session: AsyncSession | None = None,
) -> list[ActionableSignal]:
    """Run the full signal-detection pipeline.

    1. Map active markets to forecast probabilities.
    2. Compute consensus + calibration for each.
    3. Compute edge and apply filters.
    4. Sort by expected-value score and persist.
    """
    own_session = session is None
    if own_session:
        async with session_factory() as session:
            ms = await map_all_markets(session)
            return await _detect(session, ms)
    ms = await map_all_markets(session)  # type: ignore[arg-type]
    return await _detect(session, ms)  # type: ignore[arg-type]


async def detect_signals_short_range(
    session: AsyncSession | None = None,
) -> list[ActionableSignal]:
    """Run signal detection for short-range markets only (≤30h).

    Uses the same pipeline as ``detect_signals`` but only processes
    markets within 30 hours of resolution, with deduplication against
    recently created signals.
    """
    own_session = session is None
    if own_session:
        async with session_factory() as session:
            ms = await map_short_range_markets(session)
            return await _detect(session, ms, dedup_minutes=60)
    ms = await map_short_range_markets(session)  # type: ignore[arg-type]
    return await _detect(session, ms, dedup_minutes=60)  # type: ignore[arg-type]


async def _detect(
    session: AsyncSession,
    market_signals: list,
    dedup_minutes: int = 0,
) -> list[ActionableSignal]:
    logger.info("Processing %d mapped market signals", len(market_signals))

    # Build dedup set of recently signaled markets
    recent_market_ids: set[str] = set()
    if dedup_minutes > 0:
        cutoff = datetime.now(tz=timezone.utc) - timedelta(minutes=dedup_minutes)
        stmt = select(Signal.market_id).where(Signal.created_at >= cutoff)
        result = await session.execute(stmt)
        recent_market_ids = {row[0] for row in result.all()}
        if recent_market_ids:
            logger.info("Dedup: skipping %d recently signaled markets", len(recent_market_ids))

    actionable: list[ActionableSignal] = []

    for ms in market_signals:
        if ms.market_id in recent_market_ids:
            continue
        consensus = await compute_calibrated_consensus(
            ms.gfs_prob,
            ms.ecmwf_prob,
            session,
            aviation_prob=ms.aviation_prob,
            hours_to_resolution=ms.hours_to_resolution,
            aviation_context=ms.aviation_context,
        )

        edge, direction = compute_edge(consensus.consensus_prob, ms.market_prob)

        if not passes_filters(
            edge,
            consensus.confidence,
            ms.days_to_resolution,
            ms.market,
            aviation_prob=ms.aviation_prob,
            hours_to_resolution=ms.hours_to_resolution,
        ):
            continue

        ev_score = abs(edge) * consensus.confidence

        sig = ActionableSignal(
            market_id=ms.market_id,
            question=ms.question,
            direction=direction,
            edge=edge,
            confidence=consensus.confidence,
            consensus_prob=consensus.consensus_prob,
            market_prob=ms.market_prob,
            gfs_prob=ms.gfs_prob,
            ecmwf_prob=ms.ecmwf_prob,
            aviation_prob=ms.aviation_prob,
            days_to_resolution=ms.days_to_resolution,
            hours_to_resolution=ms.hours_to_resolution,
            ev_score=ev_score,
        )

        actionable.append(sig)

    # Sort by EV score descending
    actionable.sort(key=lambda s: s.ev_score, reverse=True)

    # Persist all signals
    for sig in actionable:
        await persist_signal(sig, session)
    await session.flush()

    logger.info("Detected %d actionable signals", len(actionable))
    return actionable
