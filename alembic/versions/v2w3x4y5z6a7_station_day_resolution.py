"""station_day_resolutions: continuous resolver ground-truth per station-day

Revision ID: v2w3x4y5z6a7
Revises: u1v2w3x4y5z6
Create Date: 2026-06-24 12:00:00.000000

Phase 1 of the self-improvement data-collection roadmap. ``market_resolutions``
(M3) only yields a measurable resolver divergence on ~33% of settled markets
(YES-pinned outcomes); the rest imply a one-sided/no bound. This table
intersects the bounds across a whole station-day's market ladder into a single
continuous resolved-max estimate, so "our observation vs how the market actually
closed" becomes a signed number on (ideally) every laddered station-day. Built
at daily settlement from already-labeled ``market_resolutions`` rows — no new
on-chain calls.

Additive (one new table) → instant metadata-only DDL on Postgres; running old
code ignores it. Apply BEFORE deploying the station-day settlement hook.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "v2w3x4y5z6a7"
down_revision: Union[str, None] = "u1v2w3x4y5z6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "station_day_resolutions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("station_icao", sa.String(), nullable=False),
        sa.Column("parsed_location", sa.String(), nullable=True),
        sa.Column("target_date_local", sa.Date(), nullable=False),
        sa.Column("unit", sa.String(), nullable=True),
        sa.Column("resolved_max_lower_f", sa.Float(), nullable=True),
        sa.Column("resolved_max_upper_f", sa.Float(), nullable=True),
        sa.Column("resolved_max_point_f", sa.Float(), nullable=True),
        sa.Column("resolved_source", sa.String(), nullable=True),
        sa.Column("n_buckets_resolved", sa.Integer(), nullable=True),
        sa.Column("routine_metar_max_f", sa.Float(), nullable=True),
        sa.Column("divergence_point_f", sa.Float(), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "station_icao", "target_date_local",
            name="uq_stationday_resolution",
        ),
    )
    op.create_index(
        "ix_stationday_resolution_date",
        "station_day_resolutions",
        ["target_date_local"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_stationday_resolution_date", table_name="station_day_resolutions"
    )
    op.drop_table("station_day_resolutions")
