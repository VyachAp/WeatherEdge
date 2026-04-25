"""add station_normals table

Revision ID: f6g7h8i9j0k1
Revises: e5f6g7h8i9j0
Create Date: 2026-04-26 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "f6g7h8i9j0k1"
down_revision: Union[str, None] = "e5f6g7h8i9j0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "station_normals",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("station_icao", sa.String(), nullable=False),
        sa.Column("day_of_year", sa.Integer(), nullable=False),
        sa.Column("mean_max_c", sa.Float(), nullable=False),
        sa.Column("std_max_c", sa.Float(), nullable=False),
        sa.Column("sample_years", sa.Integer(), nullable=False),
        sa.Column(
            "source", sa.String(), nullable=False,
            server_default="openmeteo_archive_era5",
        ),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "station_icao", "day_of_year", name="uq_station_normal_doy",
        ),
    )
    op.create_index(
        "ix_station_normal_icao_doy",
        "station_normals",
        ["station_icao", "day_of_year"],
    )


def downgrade() -> None:
    op.drop_index("ix_station_normal_icao_doy", table_name="station_normals")
    op.drop_table("station_normals")
