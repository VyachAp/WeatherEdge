"""Aviation-based probability confidence scoring with optional recalibration.

Computes a confidence score for aviation-derived probabilities based on
forecast lead time and aviation intelligence context, and applies a simple
linear recalibration when enough historical data is available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from src.db.engine import async_session as session_factory
from src.db.models import Signal, Trade, TradeStatus

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_CALIBRATION_SAMPLES: int = 50


# ---------------------------------------------------------------------------
# Consensus dataclass
# ---------------------------------------------------------------------------


@dataclass
class ConsensusResult:
    """Result of aviation probability confidence computation."""

    consensus_prob: float
    confidence: float
    aviation_prob: float | None = None
    calibrated: bool = False


# ---------------------------------------------------------------------------
# Core consensus logic
# ---------------------------------------------------------------------------


def _compute_weights(
    hours_to_resolution: float,
    has_aviation: bool,
) -> dict[str, float]:
    """Compute aviation weight based on forecast horizon."""
    if not has_aviation or hours_to_resolution > 30:
        return {"aviation": 0.0}
    return {"aviation": 1.0}


def compute_consensus(
    aviation_prob: float | None = None,
    hours_to_resolution: float | None = None,
    aviation_context: object | None = None,
    *,
    wx_trend_rate: float | None = None,
    metar_trend_rate: float | None = None,
) -> ConsensusResult:
    """Compute aviation-based probability with lead-time confidence scoring.

    Raises ``ValueError`` if aviation_prob is ``None``.
    """
    if aviation_prob is None:
        raise ValueError("Aviation probability is required")

    hrs = hours_to_resolution if hours_to_resolution is not None else 999.0
    weights = _compute_weights(hrs, has_aviation=True)

    raw = weights["aviation"] * aviation_prob

    # Confidence scales with lead time
    if hrs <= 6:
        confidence = 0.80
    elif hrs <= 12:
        confidence = 0.70
    elif hrs <= 24:
        confidence = 0.55
    else:
        confidence = 0.40

    # Apply aviation intelligence adjustments
    if aviation_context is not None:
        amend = getattr(aviation_context, "taf_amendment_count", 0)
        if amend > 2:
            confidence -= 0.05 * min(amend - 2, 4)  # max -0.20
        if getattr(aviation_context, "speci_events_2h", 0) > 0:
            confidence += 0.05
        if getattr(aviation_context, "has_severe_pireps", False):
            confidence += 0.08
        if getattr(aviation_context, "active_sigmet_count", 0) > 0:
            confidence += 0.10

    # WX corroboration/contradiction adjustment
    if wx_trend_rate is not None and metar_trend_rate is not None:
        same_direction = (wx_trend_rate >= 0) == (metar_trend_rate >= 0)
        if same_direction:
            confidence += 0.05  # WX confirms METAR direction
        else:
            confidence -= 0.03  # WX contradicts METAR

    # No [0.01, 0.99] clamp: weights are always 0 or 1, so raw == aviation_prob.
    # Above/below/wind probabilities are already clamped in the aviation layer;
    # bracket probabilities intentionally pass through raw so that near-zero
    # longshots don't get floored into fake edge.
    consensus_prob = max(0.0, min(1.0, raw))
    confidence = max(0.0, min(1.0, confidence))

    return ConsensusResult(
        consensus_prob=consensus_prob,
        confidence=confidence,
        aviation_prob=aviation_prob,
    )


# ---------------------------------------------------------------------------
# Bayesian recalibration
# ---------------------------------------------------------------------------


async def get_calibration_coefficients(
    session: AsyncSession,
) -> tuple[float, float] | None:
    """Compute linear recalibration coefficients from resolved signals.

    Queries signals that have associated trades with status WON or LOST,
    fits ``actual = slope * predicted + intercept``, and returns
    ``(slope, intercept)``.  Returns ``None`` when fewer than
    :data:`MIN_CALIBRATION_SAMPLES` resolved signals exist.
    """
    stmt = (
        select(Signal)
        .options(joinedload(Signal.trades))
        .join(Trade, Trade.signal_id == Signal.id)
        .where(Trade.status.in_([TradeStatus.WON, TradeStatus.LOST]))
    )
    result = await session.execute(stmt)
    signals = result.unique().scalars().all()

    if len(signals) < MIN_CALIBRATION_SAMPLES:
        return None

    predicted = []
    actual = []
    for sig in signals:
        predicted.append(sig.model_prob)
        # Determine outcome from trade status
        won = any(t.status == TradeStatus.WON for t in sig.trades)
        actual.append(1.0 if won else 0.0)

    predicted_arr = np.array(predicted)
    actual_arr = np.array(actual)
    slope, intercept = np.polyfit(predicted_arr, actual_arr, 1)
    logger.info(
        "Calibration fitted on %d signals: slope=%.4f intercept=%.4f",
        len(signals),
        slope,
        intercept,
    )
    return (float(slope), float(intercept))


async def compute_calibrated_consensus(
    aviation_prob: float | None = None,
    session: AsyncSession | None = None,
    hours_to_resolution: float | None = None,
    aviation_context: object | None = None,
    *,
    wx_trend_rate: float | None = None,
    metar_trend_rate: float | None = None,
) -> ConsensusResult:
    """Compute consensus with optional Bayesian recalibration.

    If a session is provided and enough historical data exists, applies a
    linear recalibration to the raw consensus probability.
    """
    result = compute_consensus(
        aviation_prob, hours_to_resolution, aviation_context,
        wx_trend_rate=wx_trend_rate,
        metar_trend_rate=metar_trend_rate,
    )

    if session is not None:
        try:
            coeffs = await get_calibration_coefficients(session)
        except Exception:
            logger.exception("Calibration query failed; using raw consensus")
            coeffs = None

        if coeffs is not None:
            slope, intercept = coeffs
            calibrated = slope * result.consensus_prob + intercept
            result.consensus_prob = max(0.01, min(0.99, calibrated))
            result.calibrated = True

    return result
