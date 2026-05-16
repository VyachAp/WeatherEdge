"""Append-only telemetry writer for the ``decision_logs`` table.

Companion to :mod:`src.signals.evaluation_log`. ``evaluation_logs``
captures every per-side edge evaluation (passing AND rejected) at the
filter boundary; ``decision_logs`` captures what happened *after* a
passing evaluation reached the persistence / sizing / order-placement
stages. The gap between the two is what made the 2026-05-16 debug
session expensive: 762 passing evals → 0 Signal rows, with no
telemetry on why.

Call sites (instrumented for the first cut):
  * ``src/scheduler/__init__.py::job_unified_pipeline`` — probability
    path persistence + dedup + sizing + order placement.
  * ``src/execution/lock_rule_executor.py::try_lock_rule_trade`` — lock
    path same.

Outcome vocabulary intentionally enumerable so SQL aggregation works
without LIKE patterns. Keep new outcomes ALL-UPPERCASE_SNAKE? No — match
``signal_kind`` style: lowercase snake. Add new constants here so call
sites import from one place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.db.models import DecisionLog

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.models import TradeDirection


# Success branches — a Signal row was created and (in live mode) a Trade
# row at least reached PENDING. Distinguishing them lets dashboards
# track the funnel filled→pending→signal-only.
OUTCOME_SIGNAL_WRITTEN = "signal_written"      # Signal upserted, no Trade attempted (rare)
OUTCOME_TRADE_PENDING = "trade_pending"        # Trade row created, dry-run or delayed
OUTCOME_TRADE_FILLED = "trade_filled"          # FAK filled live, status=OPEN

# Skip / failure branches — sizing or persistence path declined the trade.
OUTCOME_DUP_INPROC = "dup_blocked_inproc"      # In-process dedup set already had it
OUTCOME_DUP_DB = "dup_blocked_db"              # DB-backed _has_active_trade check
OUTCOME_STAKE_BELOW_MIN = "stake_below_min"    # Kelly × dd_mult < MIN_STAKE_USD
OUTCOME_DRAWDOWN_PAUSED = "drawdown_paused"    # dd_mult == 0 (PAUSED state)
OUTCOME_CLUSTER_CAP = "cluster_cap_hit"        # CLUSTER_STAKE_CAP_USD would be exceeded
OUTCOME_CAP_EXCEEDED = "cap_exceeded"          # MAX_EXPOSURE_PCT / DAILY_SPEND_CAP_USD
OUTCOME_NO_TOKEN_IDS = "no_token_ids"          # Gamma had no clob token pair for market
OUTCOME_NO_CLIENT = "no_client"                # CLOB client unavailable / no_private_key
OUTCOME_NO_FILL = "no_fill"                    # FAK posted but engine returned 0 size
OUTCOME_ORDER_FAILED = "order_failed"          # place_order raised / returned False


async def log_decision(
    session: "AsyncSession",
    *,
    market_id: str,
    direction: "TradeDirection | str",
    signal_kind: str,
    outcome: str,
    requested_stake_usd: float | None = None,
    actual_stake_usd: float | None = None,
    dd_multiplier: float | None = None,
    dd_level: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Append one ``DecisionLog`` row capturing this post-filter decision.

    Does not flush — caller's surrounding ``session.flush()`` /
    ``session.commit()`` batches it. Cheap to call from every branch;
    aggregate volume is at most one row per market per tick.
    """
    session.add(DecisionLog(
        market_id=market_id,
        direction=direction,
        signal_kind=signal_kind,
        outcome=outcome,
        requested_stake_usd=requested_stake_usd,
        actual_stake_usd=actual_stake_usd,
        dd_multiplier=dd_multiplier,
        dd_level=dd_level,
        metadata_json=metadata,
    ))
