"""strategy telemetry: evaluation_logs + Trade exec context + Signal lock fields

Revision ID: j0k1l2m3n4o5
Revises: i9j0k1l2m3n4
Create Date: 2026-05-04 16:00:00.000000

Three additive changes that unblock strategy iteration once
``AUTO_EXECUTE=true`` starts producing live trades:

1. ``evaluation_logs`` — append-only row per pipeline edge evaluation
   (BOTH the chosen side AND any side that failed filters). Captures
   model_prob/market_prob/edge/passes/reject_reason/depth/min_to_close/
   routine_count/signal_kind so MIN_EDGE / MIN_PROBABILITY / MIN_DEPTH_USD
   tuning can be backtested against rejected candidates. Without this,
   filter-tightening decisions are blind: ``signals`` only carries
   passing edges and is now de-duplicated to one row per (market, side).

2. ``trades.submit_yes_bid / submit_yes_ask / submit_depth_usd /
   submit_at`` — snapshot of the live YES bid/ask and the buy-side
   orderbook depth at the moment ``place_order`` was called. Lets
   slippage post-mortems decompose ``fill_price - entry_price`` into
   "spread we accepted" vs "depth we exhausted" once the pipeline runs
   live FAK orders.

3. ``signals.signal_kind`` (``'probability' | 'lock'``) +
   ``lock_branch / lock_routine_count / lock_observed_max_f`` — per-row
   flag for which path produced the signal, plus the structured lock
   decision context (which branch fired, observed daily max at fire
   time, routine_count). Today only ``confidence=margin_f`` is
   persisted; everything else is in JSON logs and lost on rotation.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "j0k1l2m3n4o5"
down_revision: Union[str, None] = "i9j0k1l2m3n4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # -- 1. evaluation_logs (append-only edge-evaluation telemetry) --
    op.create_table(
        "evaluation_logs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("market_id", sa.String(), nullable=False),
        sa.Column(
            "direction",
            postgresql.ENUM("BUY_YES", "BUY_NO", name="tradedirection", create_type=False),
            nullable=False,
        ),
        sa.Column("signal_kind", sa.String(), nullable=False),  # 'probability' | 'lock'
        sa.Column("model_prob", sa.Float(), nullable=False),
        sa.Column("market_prob", sa.Float(), nullable=False),
        sa.Column("edge", sa.Float(), nullable=False),
        sa.Column("passes", sa.Boolean(), nullable=False),
        sa.Column("reject_reason", sa.String(), nullable=True),
        sa.Column("depth_usd", sa.Float(), nullable=True),
        sa.Column("minutes_to_close", sa.Float(), nullable=True),
        sa.Column("routine_count", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(["market_id"], ["markets.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_eval_logs_market_created",
        "evaluation_logs",
        ["market_id", "created_at"],
    )

    # -- 2. trades.submit_* exec context --
    op.add_column("trades", sa.Column("submit_yes_bid", sa.Float(), nullable=True))
    op.add_column("trades", sa.Column("submit_yes_ask", sa.Float(), nullable=True))
    op.add_column("trades", sa.Column("submit_depth_usd", sa.Float(), nullable=True))
    op.add_column(
        "trades",
        sa.Column("submit_at", sa.DateTime(timezone=True), nullable=True),
    )

    # -- 3. signals.signal_kind + lock decision context --
    # signal_kind defaults to 'probability' so the historical 168 rows are
    # readable; the lock-path UPSERT in scheduler.py overwrites it on the
    # next tick that touches a lock-fired market.
    op.add_column(
        "signals",
        sa.Column(
            "signal_kind",
            sa.String(),
            nullable=False,
            server_default="probability",
        ),
    )
    op.add_column("signals", sa.Column("lock_branch", sa.String(), nullable=True))
    op.add_column("signals", sa.Column("lock_routine_count", sa.Integer(), nullable=True))
    op.add_column("signals", sa.Column("lock_observed_max_f", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("signals", "lock_observed_max_f")
    op.drop_column("signals", "lock_routine_count")
    op.drop_column("signals", "lock_branch")
    op.drop_column("signals", "signal_kind")
    op.drop_column("trades", "submit_at")
    op.drop_column("trades", "submit_depth_usd")
    op.drop_column("trades", "submit_yes_ask")
    op.drop_column("trades", "submit_yes_bid")
    op.drop_index("ix_eval_logs_market_created", table_name="evaluation_logs")
    op.drop_table("evaluation_logs")
