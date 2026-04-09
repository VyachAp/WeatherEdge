"""Multi-model consensus probability with optional Bayesian recalibration.

Combines GFS and ECMWF exceedance probabilities into a single consensus
estimate, computes a confidence score based on model agreement, and applies
a simple linear recalibration when enough historical data is available.
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

ECMWF_WEIGHT: float = 0.6
GFS_WEIGHT: float = 0.4
MIN_CALIBRATION_SAMPLES: int = 50


# ---------------------------------------------------------------------------
# Consensus dataclass
# ---------------------------------------------------------------------------


@dataclass
class ConsensusResult:
    """Result of multi-model consensus computation."""

    consensus_prob: float
    confidence: float
    gfs_prob: float | None
    ecmwf_prob: float | None
    calibrated: bool = False


# ---------------------------------------------------------------------------
# Core consensus logic
# ---------------------------------------------------------------------------


def compute_consensus(
    gfs_prob: float | None,
    ecmwf_prob: float | None,
) -> ConsensusResult:
    """Weighted average of model probabilities with confidence scoring.

    Raises ``ValueError`` if both inputs are ``None``.
    """
    if gfs_prob is None and ecmwf_prob is None:
        raise ValueError("At least one model probability is required")

    if gfs_prob is not None and ecmwf_prob is not None:
        raw = ECMWF_WEIGHT * ecmwf_prob + GFS_WEIGHT * gfs_prob
        confidence = 1.0 - abs(gfs_prob - ecmwf_prob)
    elif ecmwf_prob is not None:
        raw = ecmwf_prob
        confidence = 0.5
    else:
        raw = gfs_prob  # type: ignore[assignment]
        confidence = 0.5

    consensus_prob = max(0.01, min(0.99, raw))
    confidence = max(0.0, min(1.0, confidence))

    return ConsensusResult(
        consensus_prob=consensus_prob,
        confidence=confidence,
        gfs_prob=gfs_prob,
        ecmwf_prob=ecmwf_prob,
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
    gfs_prob: float | None,
    ecmwf_prob: float | None,
    session: AsyncSession | None = None,
) -> ConsensusResult:
    """Compute consensus with optional Bayesian recalibration.

    If a session is provided and enough historical data exists, applies a
    linear recalibration to the raw consensus probability.
    """
    result = compute_consensus(gfs_prob, ecmwf_prob)

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
