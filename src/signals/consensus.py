"""Signal calibration utilities.

Provides linear recalibration coefficients from resolved signal history,
plus an in-process cache + sync `apply_calibration` helper used by the
edge evaluator. Calibration is gated by ``settings.APPLY_CALIBRATION`` so
it can be A/B-toggled without code changes while we validate that the
fitted slope/intercept actually improve Brier on a held-out window.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from src.config import settings
from src.db.models import Signal, Trade, TradeStatus

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

MIN_CALIBRATION_SAMPLES: int = 50

# In-process cache for fitted (slope, intercept) — refreshed by
# ``refresh_calibration`` (called from the scheduler tick) and read
# synchronously by ``apply_calibration`` from the edge evaluator.
_CACHE_TTL_SEC: float = 1800.0
_cached_coeffs: tuple[float, float] | None = None
_cached_at: float | None = None


async def get_calibration_coefficients(
    session: AsyncSession,
) -> tuple[float, float] | None:
    """Compute linear recalibration coefficients from resolved signals.

    Queries signals that have associated trades with status WON or LOST,
    fits ``actual = slope * predicted + intercept``, and returns
    ``(slope, intercept)``.  Returns ``None`` when fewer than
    :data:`MIN_CALIBRATION_SAMPLES` resolved signals exist.

    Convention: ``Signal.model_prob`` is stored in the **side-effective
    frame** — i.e. it is the model's probability of the side the trade
    actually bet on (= P(YES) for BUY_YES trades, = 1-P(YES) for BUY_NO).
    That keeps this regression's input range consistent across both
    directions: ``predicted`` is "model's confidence in winning",
    ``actual`` is "did we win", and the slope/intercept describe how
    well-calibrated those confidences are.
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


async def refresh_calibration(session: AsyncSession) -> tuple[float, float] | None:
    """Refresh the in-process calibration cache.

    Called from the unified pipeline tick. Cheap when called more often
    than `_CACHE_TTL_SEC` (returns the cached value); fits a fresh
    regression otherwise.
    """
    global _cached_coeffs, _cached_at
    now = time.time()
    if _cached_at is not None and (now - _cached_at) < _CACHE_TTL_SEC:
        return _cached_coeffs

    coeffs = await get_calibration_coefficients(session)
    _cached_coeffs = coeffs
    _cached_at = now
    return coeffs


def get_cached_calibration() -> tuple[float, float] | None:
    """Sync read of the in-process calibration cache.

    Returns ``None`` when no fit has succeeded yet, when the cache has
    aged out, or when too few resolved signals exist (the underlying fit
    needs ``MIN_CALIBRATION_SAMPLES`` resolved trades).
    """
    if _cached_at is None:
        return None
    if (time.time() - _cached_at) > _CACHE_TTL_SEC:
        return None
    return _cached_coeffs


def apply_calibration(prob: float) -> tuple[float, bool]:
    """Apply the cached linear calibration to a side-effective probability.

    Returns ``(corrected_prob, applied)``. When the
    ``APPLY_CALIBRATION`` setting is False or no coefficients are
    cached, returns ``(prob, False)`` so the caller can log/branch.

    The regression is fit on ``Signal.model_prob`` (= side-effective
    probability), so this helper expects the same frame: the model's
    confidence on the side a trade actually bet on. Applying it to a
    raw P(YES) for a NO trade would mix two distributions; callers
    must apply it after side selection.
    """
    if not getattr(settings, "APPLY_CALIBRATION", False):
        return prob, False
    coeffs = get_cached_calibration()
    if coeffs is None:
        return prob, False
    slope, intercept = coeffs
    corrected = max(0.0, min(1.0, slope * prob + intercept))
    return corrected, True


def reset_calibration_cache() -> None:
    """Test helper — drop the in-process cache."""
    global _cached_coeffs, _cached_at
    _cached_coeffs = None
    _cached_at = None
