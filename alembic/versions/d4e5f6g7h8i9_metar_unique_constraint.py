"""add unique constraint on metar_observations(station_icao, observed_at)

Revision ID: d4e5f6g7h8i9
Revises: c3d4e5f6g7h8
Create Date: 2026-04-22 13:00:00.000000

"""
from typing import Sequence, Union

from alembic import op


revision: str = 'd4e5f6g7h8i9'
down_revision: Union[str, None] = 'c3d4e5f6g7h8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        DELETE FROM metar_observations a
        USING metar_observations b
        WHERE a.id > b.id
          AND a.station_icao = b.station_icao
          AND a.observed_at  = b.observed_at;
    """)

    op.drop_index('ix_metar_station_observed', table_name='metar_observations')

    op.create_unique_constraint(
        'uq_metar_station_obs',
        'metar_observations',
        ['station_icao', 'observed_at'],
    )


def downgrade() -> None:
    op.drop_constraint('uq_metar_station_obs', 'metar_observations', type_='unique')
    op.create_index(
        'ix_metar_station_observed',
        'metar_observations',
        ['station_icao', 'observed_at'],
    )
