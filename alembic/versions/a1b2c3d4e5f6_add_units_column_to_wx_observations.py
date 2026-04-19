"""add units column to wx_observations

Revision ID: a1b2c3d4e5f6
Revises: 5cd46cb0df3b
Create Date: 2026-04-19 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = '5cd46cb0df3b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('wx_observations', sa.Column('units', sa.String(), server_default='m', nullable=False))


def downgrade() -> None:
    op.drop_column('wx_observations', 'units')
