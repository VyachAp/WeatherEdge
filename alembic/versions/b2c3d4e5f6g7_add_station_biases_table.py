"""add station_biases table

Revision ID: b2c3d4e5f6g7
Revises: a1b2c3d4e5f6
Create Date: 2026-04-20 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'b2c3d4e5f6g7'
down_revision: Union[str, None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'station_biases',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('station_icao', sa.String(), nullable=False),
        sa.Column('date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('observed_max_c', sa.Float(), nullable=False),
        sa.Column('forecast_peak_c', sa.Float(), nullable=False),
        sa.Column('bias_c', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('station_icao', 'date', name='uq_station_bias_day'),
    )
    op.create_index('ix_station_bias_icao_date', 'station_biases', ['station_icao', 'date'])


def downgrade() -> None:
    op.drop_index('ix_station_bias_icao_date', table_name='station_biases')
    op.drop_table('station_biases')
