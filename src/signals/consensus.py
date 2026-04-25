"""Signal calibration utilities.

Provides linear recalibration coefficients from resolved signal history.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from src.db.models import Signal, Trade, TradeStatus

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

MIN_CALIBRATION_SAMPLES: int = 50


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
