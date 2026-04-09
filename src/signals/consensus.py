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
    aviation_prob: float | None = None
    calibrated: bool = False


# ---------------------------------------------------------------------------
# Core consensus logic
# ---------------------------------------------------------------------------

# Confidence boost when aviation agrees with NWP consensus within this spread.
_AVIATION_AGREEMENT_THRESHOLD: float = 0.15
_AVIATION_AGREEMENT_BOOST: float = 0.10


def _compute_weights(
    hours_to_resolution: float,
    has_aviation: bool,
    has_gfs: bool,
    has_ecmwf: bool,
) -> dict[str, float]:
    """Compute dynamic model weights based on forecast horizon."""
    if not has_aviation or hours_to_resolution > 30:
        aw = 0.0
    elif hours_to_resolution <= 6:
        aw = 0.50
    elif hours_to_resolution <= 12:
        aw = 0.30
    elif hours_to_resolution <= 24:
        aw = 0.15
    else:  # 24-30h
        aw = 0.08

    nwp_share = 1.0 - aw

    if has_gfs and has_ecmwf:
        gw = nwp_share * GFS_WEIGHT / (GFS_WEIGHT + ECMWF_WEIGHT)
        ew = nwp_share * ECMWF_WEIGHT / (GFS_WEIGHT + ECMWF_WEIGHT)
    elif has_ecmwf:
        gw = 0.0
        ew = nwp_share
    elif has_gfs:
        gw = nwp_share
        ew = 0.0
    else:
        # Only aviation available
        gw = 0.0
        ew = 0.0
        aw = 1.0

    return {"aviation": aw, "gfs": gw, "ecmwf": ew}


def compute_consensus(
    gfs_prob: float | None,
    ecmwf_prob: float | None,
    aviation_prob: float | None = None,
    hours_to_resolution: float | None = None,
    aviation_context: object | None = None,
) -> ConsensusResult:
    """Weighted average of model probabilities with dynamic horizon-based weighting.

    Raises ``ValueError`` if all inputs are ``None``.
    """
    if gfs_prob is None and ecmwf_prob is None and aviation_prob is None:
        raise ValueError("At least one model probability is required")

    hrs = hours_to_resolution if hours_to_resolution is not None else 999.0
    weights = _compute_weights(
        hrs,
        has_aviation=aviation_prob is not None,
        has_gfs=gfs_prob is not None,
        has_ecmwf=ecmwf_prob is not None,
    )

    raw = 0.0
    if gfs_prob is not None:
        raw += weights["gfs"] * gfs_prob
    if ecmwf_prob is not None:
        raw += weights["ecmwf"] * ecmwf_prob
    if aviation_prob is not None:
        raw += weights["aviation"] * aviation_prob

    # Confidence scoring based on model spread
    available = [p for p in (gfs_prob, ecmwf_prob, aviation_prob) if p is not None]
    if len(available) >= 2:
        confidence = 1.0 - (max(available) - min(available))
    else:
        confidence = 0.5

    # Boost confidence when aviation agrees with NWP consensus
    if aviation_prob is not None and (gfs_prob is not None or ecmwf_prob is not None):
        nwp_probs = [p for p in (gfs_prob, ecmwf_prob) if p is not None]
        nwp_avg = sum(nwp_probs) / len(nwp_probs)
        if abs(aviation_prob - nwp_avg) < _AVIATION_AGREEMENT_THRESHOLD:
            confidence = min(1.0, confidence + _AVIATION_AGREEMENT_BOOST)

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

    consensus_prob = max(0.01, min(0.99, raw))
    confidence = max(0.0, min(1.0, confidence))

    return ConsensusResult(
        consensus_prob=consensus_prob,
        confidence=confidence,
        gfs_prob=gfs_prob,
        ecmwf_prob=ecmwf_prob,
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
    gfs_prob: float | None,
    ecmwf_prob: float | None,
    session: AsyncSession | None = None,
    aviation_prob: float | None = None,
    hours_to_resolution: float | None = None,
    aviation_context: object | None = None,
) -> ConsensusResult:
    """Compute consensus with optional Bayesian recalibration.

    If a session is provided and enough historical data exists, applies a
    linear recalibration to the raw consensus probability.
    """
    result = compute_consensus(
        gfs_prob, ecmwf_prob, aviation_prob, hours_to_resolution, aviation_context,
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
