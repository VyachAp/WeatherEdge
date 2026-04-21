"""add forecast_exceedance_alerts table

Revision ID: c3d4e5f6g7h8
Revises: b2c3d4e5f6g7
Create Date: 2026-04-22 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'c3d4e5f6g7h8'
down_revision: Union[str, None] = 'b2c3d4e5f6g7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'forecast_exceedance_alerts',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('station_icao', sa.String(), nullable=False),
        sa.Column('observed_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('observed_temp_f', sa.Float(), nullable=False),
        sa.Column('forecast_temp_f', sa.Float(), nullable=False),
        sa.Column('delta_f', sa.Float(), nullable=False),
        sa.Column('forecast_hour_utc', sa.Integer(), nullable=False),
        sa.Column('alerted_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('station_icao', 'observed_at', name='uq_exceedance_station_obs'),
    )


def downgrade() -> None:
    op.drop_table('forecast_exceedance_alerts')
