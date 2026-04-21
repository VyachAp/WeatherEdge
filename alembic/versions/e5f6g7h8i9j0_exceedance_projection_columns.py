"""add projection columns to forecast_exceedance_alerts

Revision ID: e5f6g7h8i9j0
Revises: d4e5f6g7h8i9
Create Date: 2026-04-22 16:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'e5f6g7h8i9j0'
down_revision: Union[str, None] = 'd4e5f6g7h8i9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('forecast_exceedance_alerts', sa.Column('current_max_f', sa.Float(), nullable=True))
    op.add_column('forecast_exceedance_alerts', sa.Column('forecast_peak_f', sa.Float(), nullable=True))
    op.add_column('forecast_exceedance_alerts', sa.Column('projected_max_f', sa.Float(), nullable=True))
    op.add_column('forecast_exceedance_alerts', sa.Column('metar_trend_rate', sa.Float(), nullable=True))
    op.add_column('forecast_exceedance_alerts', sa.Column('peak_passed', sa.Boolean(), nullable=True))
    op.add_column('forecast_exceedance_alerts', sa.Column('alerted', sa.Boolean(), nullable=True))


def downgrade() -> None:
    op.drop_column('forecast_exceedance_alerts', 'alerted')
    op.drop_column('forecast_exceedance_alerts', 'peak_passed')
    op.drop_column('forecast_exceedance_alerts', 'metar_trend_rate')
    op.drop_column('forecast_exceedance_alerts', 'projected_max_f')
    op.drop_column('forecast_exceedance_alerts', 'forecast_peak_f')
    op.drop_column('forecast_exceedance_alerts', 'current_max_f')
