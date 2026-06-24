"""forecast_error_daily: per-station, per-lead-time forecast-error dataset

Revision ID: x4y5z6a7b8c9
Revises: w3x4y5z6a7b8
Create Date: 2026-06-24 13:00:00.000000

Phase 3 of the self-improvement data-collection roadmap. Reads the accumulating
``forecast_archive`` corpus + ``station_day_resolutions`` (Phase 1) into the
forecast-error-by-lead-time dataset the deferred lead-time σ floor and the
climate prior need to be fit from data. Built best-effort at daily settlement.

Additive (one new table) → instant metadata-only DDL on Postgres; running old
code ignores it.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "x4y5z6a7b8c9"
down_revision: Union[str, None] = "w3x4y5z6a7b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "forecast_error_daily",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("station_icao", sa.String(), nullable=False),
        sa.Column("target_date_local", sa.Date(), nullable=False),
        sa.Column("lead_bucket_h", sa.Integer(), nullable=False),
        sa.Column("forecast_peak_f", sa.Float(), nullable=True),
        sa.Column("forecast_sigma_f", sa.Float(), nullable=True),
        sa.Column("realized_max_f", sa.Float(), nullable=True),
        sa.Column("resolved_max_f", sa.Float(), nullable=True),
        sa.Column("error_vs_metar_f", sa.Float(), nullable=True),
        sa.Column("error_vs_resolved_f", sa.Float(), nullable=True),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "station_icao", "target_date_local", "lead_bucket_h",
            name="uq_fc_error_station_day_lead",
        ),
    )
    op.create_index(
        "ix_fc_error_station_date",
        "forecast_error_daily",
        ["station_icao", "target_date_local"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_fc_error_station_date", table_name="forecast_error_daily"
    )
    op.drop_table("forecast_error_daily")
