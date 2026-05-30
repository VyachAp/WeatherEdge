"""drop wx_observations table and the WX pipeline

Revision ID: s9t0u1v2w3x4
Revises: r8s9t0u1v2w3
Create Date: 2026-05-30 13:00:00.000000

Drops the ``wx_observations`` table along with the entire Weather Company
v3 observation pipeline (``src/ingestion/wx.py``, the ``WxObservation``
ORM, the cleanup job in ``job_daily_settlement``, the ``bet find --station``
CLI buffer hook, the ``WX_*`` Settings, and ``tests/test_wx.py``).

The pipeline was an early experiment in 5-min cadence auxiliary
observations sourced from Weather Company v3, intended to supplement the
once-an-hour routine METAR feed for last-mile peak detection. No
scheduler job ever (or ever again) called ``poll_stations`` in
production, so the table accumulated zero writes; the daily cleanup
deleted zero rows; the CLI hook silently no-op'd behind an ``if buf:``
guard against an always-empty in-process buffer. See
``docs/graveyard.md#2026-05-30-wx-weather-com-v3-pipeline``.

Defensive ``IF EXISTS``: the table was provisioned by
``5cd46cb0df3b_add_wx_observations_table_and_wx_prob_.py`` (and
subsequently extended with a ``units`` column by ``a1b2c3d4e5f6``); the
``IF EXISTS`` guard tolerates the case where ``Base.metadata.create_all``
was used instead of ``alembic upgrade head`` on some env (the same
class of drift documented in Modules 1-2 of this audit pass).

The downgrade re-creates the table at the post-``a1b2c3d4e5f6`` shape
so the chain re-runs cleanly on a roll-back. JSONB columns are not
involved here so this is a plain ``CREATE TABLE``.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "s9t0u1v2w3x4"
down_revision: Union[str, None] = "r8s9t0u1v2w3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Indexes / constraints are dropped automatically by DROP TABLE.
    op.execute("DROP TABLE IF EXISTS wx_observations")


def downgrade() -> None:
    op.create_table(
        "wx_observations",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("station_icao", sa.String(), nullable=False),
        sa.Column("units", sa.String(), nullable=False, server_default="m"),
        sa.Column("valid_time_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column("valid_time_local", sa.String(), nullable=False),
        sa.Column("temp_c", sa.Float(), nullable=True),
        sa.Column("dewpoint_c", sa.Float(), nullable=True),
        sa.Column("humidity", sa.Float(), nullable=True),
        sa.Column("wind_speed_ms", sa.Float(), nullable=True),
        sa.Column("wind_gust_ms", sa.Float(), nullable=True),
        sa.Column("wind_dir", sa.Integer(), nullable=True),
        sa.Column("pressure_hpa", sa.Float(), nullable=True),
        sa.Column("pressure_trend", sa.String(), nullable=True),
        sa.Column("precip_1h_mm", sa.Float(), nullable=True),
        sa.Column("precip_6h_mm", sa.Float(), nullable=True),
        sa.Column("precip_24h_mm", sa.Float(), nullable=True),
        sa.Column("snow_1h_mm", sa.Float(), nullable=True),
        sa.Column("snow_24h_mm", sa.Float(), nullable=True),
        sa.Column("temp_max_since_7am_c", sa.Float(), nullable=True),
        sa.Column("temp_max_24h_c", sa.Float(), nullable=True),
        sa.Column("temp_min_24h_c", sa.Float(), nullable=True),
        sa.Column("cloud_cover", sa.Integer(), nullable=True),
        sa.Column("visibility_km", sa.Float(), nullable=True),
        sa.Column("uv_index", sa.Integer(), nullable=True),
        sa.Column(
            "fetched_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "station_icao", "valid_time_local", name="uq_wx_station_time",
        ),
    )
    op.create_index(
        "ix_wx_station_valid",
        "wx_observations",
        ["station_icao", "valid_time_utc"],
    )
