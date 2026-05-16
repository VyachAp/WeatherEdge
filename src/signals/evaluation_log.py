"""Append-only telemetry writer for the ``evaluation_logs`` table.

Called by BOTH trading paths (probability + lock-rule) for EVERY edge
evaluation — passing AND rejected. ``signals`` is de-duplicated per
(market, direction) and only carries passing edges, so it's the wrong
source for filter-tuning backtests. This module is the right one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.db.models import EvaluationLog

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def log_evaluation(
    session: "AsyncSession",
    *,
    market_id: str,
    direction,
    signal_kind: str,
    model_prob: float,
    market_prob: float,
    edge: float,
    passes: bool,
    reject_reason: str | None,
    depth_usd: float | None,
    minutes_to_close: float | None,
    routine_count: int | None,
    raw_model_prob: float | None = None,
    calibrated: bool = False,
) -> None:
    """Append one ``EvaluationLog`` row capturing this edge evaluation.

    Without this row, MIN_EDGE / MIN_PROBABILITY / MIN_DEPTH_USD tuning
    is blind — ``signals`` only carries passing edges and is de-duped to
    one row per (market, side). Append-only, no UPSERT: each tick's
    evaluation is a separate calibration data point.

    Does not flush — the caller's next ``session.flush()`` (typically the
    ON CONFLICT upsert in ``persistence.dedup.upsert_signal``) batches
    these along with surrounding writes.
    """
    session.add(EvaluationLog(
        market_id=market_id,
        direction=direction,
        signal_kind=signal_kind,
        model_prob=model_prob,
        raw_model_prob=raw_model_prob,
        calibrated=calibrated,
        market_prob=market_prob,
        edge=edge,
        passes=passes,
        reject_reason=reject_reason,
        depth_usd=depth_usd,
        minutes_to_close=minutes_to_close,
        routine_count=routine_count,
    ))
