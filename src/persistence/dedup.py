"""Durable trade/signal dedup helpers.

These two helpers are the *DB-backed* safety net against duplicate
trades on the same (market, direction). In-process dedup sets
(``_unified_fired_today`` / ``_locked_markets_fired_today``) are
same-tick speed-ups but lose state on restart; this module's checks
survive restarts and races between the unified and fast-lock-poll jobs.

- ``has_active_trade(session, market_id, direction)`` — short-circuits
  callers before they place a second order on a market they already have
  a PENDING or OPEN position in.
- ``upsert_signal(session, ...)`` — single atomic ``INSERT ... ON
  CONFLICT DO UPDATE ... RETURNING`` against the schema-level
  ``uq_signals_market_direction`` constraint (migration ``i9j0k1l2m3n4``).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db.models import Signal, Trade, TradeStatus

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def has_active_trade(
    session: "AsyncSession", market_id: str, direction,
) -> bool:
    """True iff a PENDING or OPEN Trade row already exists for this pair.

    Hard guard against duplicate firing. In live mode this is the only
    thing that prevents the bot from re-betting the same market+side
    every tick until the Kelly exposure cap kicks in. In dry-run it
    stops the Signal/Trade table from accumulating one row per tick.

    Notes:
      - Filters on status only, no Market.end_date check needed: PENDING/
        OPEN trades on resolved markets are fixed up by ``resolve_trades``
        within minutes of expiry, so a stale row blocking an already-
        resolved market is a non-issue (we wouldn't trade into it anyway).
      - The migration ``i9j0k1l2m3n4`` collapsed pre-existing duplicate
        PENDING rows so this guard doesn't permanently lock out markets
        that already accumulated multiple dry-run attempts.
    """
    result = await session.execute(
        select(Trade.id).where(
            Trade.market_id == market_id,
            Trade.direction == direction,
            Trade.status.in_([TradeStatus.PENDING, TradeStatus.OPEN]),
        ).limit(1)
    )
    return result.scalar_one_or_none() is not None


async def upsert_signal(
    session: "AsyncSession",
    *,
    market_id: str,
    direction,
    model_prob: float,
    market_prob: float,
    edge: float,
    signal_kind: str = "probability",
    lock_branch: str | None = None,
    lock_routine_count: int | None = None,
    lock_observed_max_f: float | None = None,
    lock_margin_f: float | None = None,
    raw_model_prob: float | None = None,
    calibrated: bool = False,
) -> Signal:
    """Insert-or-refresh the unique ``(market_id, direction)`` Signal row.

    Schema-level ``uq_signals_market_direction`` (migration
    ``i9j0k1l2m3n4``) means we'd otherwise collide on every re-evaluation
    tick. Implemented as a single atomic ``INSERT ... ON CONFLICT DO
    UPDATE ... RETURNING`` so the path is race-safe at the DB level (no
    longer relies on APScheduler ``max_instances=1`` to serialize a
    SELECT/INSERT window). Refreshes ``created_at`` so callers can see
    "this signal was last evaluated at X" in the DB.

    ``signal_kind`` and the ``lock_*`` fields land on the row regardless
    of whether it's INSERT or UPDATE; a market that flips between the
    probability and lock paths between ticks (e.g. early-day probability
    edge → lock fires when the threshold gets crossed) will overwrite
    the kind/branch on the next tick. That's the intended semantic:
    Signal reflects the current trading rationale, not history.
    """
    now = datetime.now(timezone.utc)
    stmt = (
        pg_insert(Signal)
        .values(
            market_id=market_id,
            direction=direction,
            model_prob=model_prob,
            raw_model_prob=raw_model_prob,
            calibrated=calibrated,
            market_prob=market_prob,
            edge=edge,
            signal_kind=signal_kind,
            lock_branch=lock_branch,
            lock_routine_count=lock_routine_count,
            lock_observed_max_f=lock_observed_max_f,
            lock_margin_f=lock_margin_f,
            created_at=now,
        )
        .on_conflict_do_update(
            constraint="uq_signals_market_direction",
            set_={
                "model_prob": model_prob,
                "raw_model_prob": raw_model_prob,
                "calibrated": calibrated,
                "market_prob": market_prob,
                "edge": edge,
                "signal_kind": signal_kind,
                "lock_branch": lock_branch,
                "lock_routine_count": lock_routine_count,
                "lock_observed_max_f": lock_observed_max_f,
                "lock_margin_f": lock_margin_f,
                "created_at": now,
            },
        )
        .returning(Signal)
    )
    result = await session.scalars(
        stmt, execution_options={"populate_existing": True}
    )
    return result.one()
