"""add forecast_archive table

Revision ID: h8i9j0k1l2m3
Revises: g7h8i9j0k1l2
Create Date: 2026-05-03 12:00:00.000000

Captures every Open-Meteo blended forecast snapshot taken by
``aggregate_state``, anchored to the station-local target day. Multiple
rows per (station, target_date_local) are expected — each unified pipeline
tick writes one — so a backtest can replay how the forecast evolved
through the heating cycle. Backtests join archive rows to resolved
``MetarObservation.temp_f`` daily maxes for per-path Brier scoring.

Hourly arrays are stored as JSONB to keep the existing
``OpenMeteoForecast`` shape lossless without exploding into 24+ rows per
snapshot. Postgres-only — sqlite is not in scope.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "h8i9j0k1l2m3"
down_revision: Union[str, None] = "g7h8i9j0k1l2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "forecast_archive",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("station_icao", sa.String(), nullable=False),
        # Station-local calendar date the forecast targets. Stored as a
        # naive DATE because the local-day boundary is the comparison key
        # downstream (joins against MetarObservation aggregated by local
        # day). Avoids per-station TZ ambiguity at query time.
        sa.Column("target_date_local", sa.Date(), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("peak_temp_c", sa.Float(), nullable=False),
        sa.Column("peak_hour_utc", sa.Integer(), nullable=False),
        sa.Column("peak_temp_std_c", sa.Float(), nullable=False, server_default="0"),
        sa.Column("model_count", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("hourly_temps_c", postgresql.JSONB(), nullable=False),
        sa.Column("hourly_cloud_cover", postgresql.JSONB(), nullable=False),
        sa.Column("hourly_solar_radiation", postgresql.JSONB(), nullable=False),
        sa.Column("hourly_dewpoint_c", postgresql.JSONB(), nullable=False),
        sa.Column("hourly_wind_speed", postgresql.JSONB(), nullable=False),
        sa.Column("hourly_temps_std_c", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_forecast_archive_icao_target_fetched",
        "forecast_archive",
        ["station_icao", "target_date_local", "fetched_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_forecast_archive_icao_target_fetched",
        table_name="forecast_archive",
    )
    op.drop_table("forecast_archive")
