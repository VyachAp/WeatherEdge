"""decision_logs: post-filter outcome telemetry

Revision ID: n4o5p6q7r8s9
Revises: m3n4o5p6q7r8
Create Date: 2026-05-16 19:30:00.000000

Adds the ``decision_logs`` append-only table that captures what happened
to each per-side edge evaluation *after* ``_check_filters`` returned
``passes=True``. ``evaluation_logs`` answers "did this evaluation clear
the filter gates?" but stops there — it can't distinguish "passed and
became a Trade" from "passed but got dedup'd / zero-staked / drawdown-
paused / order-failed." Today's debug session (2026-05-16) cost ~4
hours because 762 passing evals materialised into 0 Signal rows and we
had no telemetry on the gap.

Cardinality: one row per per-side decision (whether the gate passed
or not is in evaluation_logs; decisions are mostly the passing slice
of that, with a few extra entries for paths that bypass _check_filters
like the lock-rule early-return on already-fired markets). Expect
~50-500 rows/day at current volume.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "n4o5p6q7r8s9"
down_revision: Union[str, None] = "m3n4o5p6q7r8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "decision_logs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("market_id", sa.String(), nullable=False),
        sa.Column(
            "direction",
            postgresql.ENUM(
                "BUY_YES", "BUY_NO", name="tradedirection", create_type=False
            ),
            nullable=False,
        ),
        sa.Column("signal_kind", sa.String(), nullable=False),
        sa.Column("outcome", sa.String(), nullable=False),
        sa.Column("requested_stake_usd", sa.Float(), nullable=True),
        sa.Column("actual_stake_usd", sa.Float(), nullable=True),
        sa.Column("dd_multiplier", sa.Float(), nullable=True),
        sa.Column("dd_level", sa.String(), nullable=True),
        sa.Column(
            "metadata_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
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
        "ix_decision_logs_outcome_created",
        "decision_logs",
        ["outcome", "created_at"],
    )
    op.create_index(
        "ix_decision_logs_market_created",
        "decision_logs",
        ["market_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_decision_logs_market_created", table_name="decision_logs")
    op.drop_index("ix_decision_logs_outcome_created", table_name="decision_logs")
    op.drop_table("decision_logs")
