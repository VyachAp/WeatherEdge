"""shadow_ledger: unselected decision ledger for the four-stage shadow flow

Revision ID: u1v2w3x4y5z6
Revises: t0u1v2w3x4y5
Create Date: 2026-06-12 12:00:00.000000

The redesigned decision pipeline (information-latency thesis) runs in shadow:
it evaluates every market each tick, writes its full decision here, and places
NO orders. This table is the clean, *unselected* dataset the legacy
``evaluation_logs`` could never be — it scores every evaluated market against
``market_resolutions.yes_won`` (join on market_id), so calibration can be
measured per observation-fraction bucket and compared to the market baseline
out-of-sample before any capital is routed to the new model.

Additive (one new table) → instant metadata-only DDL on Postgres; running old
code ignores it. Apply BEFORE deploying the shadow runner.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


revision: str = "u1v2w3x4y5z6"
down_revision: Union[str, None] = "t0u1v2w3x4y5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "shadow_ledger",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "market_id", sa.String(), sa.ForeignKey("markets.id"), nullable=False
        ),
        sa.Column("station_icao", sa.String(), nullable=True),
        sa.Column("target_date_local", sa.Date(), nullable=True),
        # stage 1 — probability estimate
        sa.Column("prior_yes", sa.Float(), nullable=False),
        sa.Column("updated_yes", sa.Float(), nullable=False),
        sa.Column("info_advantage", sa.Float(), nullable=False),
        sa.Column("obs_fraction", sa.Float(), nullable=False),
        sa.Column("mean_f", sa.Float(), nullable=True),
        sa.Column("prior_sigma_f", sa.Float(), nullable=True),
        sa.Column("updated_sigma_f", sa.Float(), nullable=True),
        # market context
        sa.Column("yes_bid", sa.Float(), nullable=True),
        sa.Column("yes_ask", sa.Float(), nullable=True),
        sa.Column("market_mid", sa.Float(), nullable=True),
        sa.Column("depth_usd", sa.Float(), nullable=True),
        sa.Column("minutes_to_close", sa.Float(), nullable=True),
        # stages 2-4 — cost / edge / decision / size
        sa.Column("side", sa.String(), nullable=True),
        sa.Column("effective_cost", sa.Float(), nullable=True),
        sa.Column("net_edge", sa.Float(), nullable=True),
        sa.Column("sigma_edge", sa.Float(), nullable=True),
        sa.Column("would_trade", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("shadow_stake_usd", sa.Float(), nullable=True),
        sa.Column("decision_reason", sa.String(), nullable=True),
        sa.Column("feature_json", JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_shadow_ledger_market_created", "shadow_ledger", ["market_id", "created_at"]
    )
    op.create_index("ix_shadow_ledger_created", "shadow_ledger", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_shadow_ledger_created", table_name="shadow_ledger")
    op.drop_index("ix_shadow_ledger_market_created", table_name="shadow_ledger")
    op.drop_table("shadow_ledger")
