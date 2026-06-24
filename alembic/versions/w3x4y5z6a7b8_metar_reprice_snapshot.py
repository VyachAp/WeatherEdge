"""metar_reprice_snapshots: event-driven price snapshots keyed to METAR arrival

Revision ID: w3x4y5z6a7b8
Revises: v2w3x4y5z6a7
Create Date: 2026-06-24 12:30:00.000000

Phase 2 of the self-improvement data-collection roadmap. Makes the
information-latency thesis falsifiable: when a new routine METAR is detected,
snapshot the YES quote + depth for the station's active markets, then re-snapshot
on following ticks. Diffing yes_mid across created_at (grouped by market_id +
metar_observed_at) measures how fast the market reprices toward information we
already hold. Written best-effort by the scheduler, entirely behind
``REPRICE_SNAPSHOT_ENABLED`` (default off).

Additive (one new table) → instant metadata-only DDL on Postgres; running old
code ignores it. Apply BEFORE flipping ``REPRICE_SNAPSHOT_ENABLED``.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "w3x4y5z6a7b8"
down_revision: Union[str, None] = "v2w3x4y5z6a7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "metar_reprice_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "market_id", sa.String(), sa.ForeignKey("markets.id"), nullable=False
        ),
        sa.Column("station_icao", sa.String(), nullable=False),
        sa.Column(
            "metar_observed_at", sa.DateTime(timezone=True), nullable=False
        ),
        sa.Column("new_obs_temp_f", sa.Float(), nullable=True),
        sa.Column("new_observed_max_f", sa.Float(), nullable=True),
        sa.Column("obs_fraction", sa.Float(), nullable=True),
        sa.Column("yes_bid", sa.Float(), nullable=True),
        sa.Column("yes_ask", sa.Float(), nullable=True),
        sa.Column("yes_mid", sa.Float(), nullable=True),
        sa.Column("depth_yes_usd", sa.Float(), nullable=True),
        sa.Column("depth_no_usd", sa.Float(), nullable=True),
        sa.Column("minutes_to_close", sa.Float(), nullable=True),
        sa.Column("seconds_since_obs", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_metar_reprice_market_obs",
        "metar_reprice_snapshots",
        ["market_id", "metar_observed_at"],
    )
    op.create_index(
        "ix_metar_reprice_created",
        "metar_reprice_snapshots",
        ["created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_metar_reprice_created", table_name="metar_reprice_snapshots"
    )
    op.drop_index(
        "ix_metar_reprice_market_obs", table_name="metar_reprice_snapshots"
    )
    op.drop_table("metar_reprice_snapshots")
