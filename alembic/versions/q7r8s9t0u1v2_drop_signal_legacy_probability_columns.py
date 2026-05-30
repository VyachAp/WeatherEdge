"""drop Signal legacy per-model probability columns

Revision ID: q7r8s9t0u1v2
Revises: p6q7r8s9t0u1
Create Date: 2026-05-30 12:00:00.000000

Drops the four always-NULL columns left over from the v1 multi-model
architecture (GFS / ECMWF / aviation METAR / Weather.com each contributed
its own probability column). The unified pipeline merges all signals into
one ``WeatherState`` before ``compute_distribution`` — no code in ``src/``
has written or read these columns since the unified refactor. See
``docs/graveyard.md#2026-05-30-signalgfs_prob--ecmwf_prob--aviation_prob--wx_prob-columns``
for context.

Defensive ``IF EXISTS`` / ``IF NOT EXISTS``: ``gfs_prob`` and ``ecmwf_prob``
were never added by an Alembic migration — they were declared in
``models.py`` and would only exist in databases where ``create_all()`` was
run with sufficient DDL privileges. Same failure class as the 2026-05-16
``bot_state`` table drift; we don't want this migration to crash on prod
DBs that happen to be missing one of the four columns.

EXPAND/CONTRACT: this is a CONTRACT step. Deploy the writer-side change
(model file edit) BEFORE running this migration so the running code has
already stopped issuing ``INSERT ... gfs_prob ...`` (it didn't anyway,
but the discipline matters for the next column drop where it might).
"""
from typing import Sequence, Union

from alembic import op


revision: str = "q7r8s9t0u1v2"
down_revision: Union[str, None] = "p6q7r8s9t0u1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_COLUMNS = ("gfs_prob", "ecmwf_prob", "aviation_prob", "wx_prob")


def upgrade() -> None:
    for col in _COLUMNS:
        op.execute(f"ALTER TABLE signals DROP COLUMN IF EXISTS {col}")


def downgrade() -> None:
    for col in _COLUMNS:
        op.execute(f"ALTER TABLE signals ADD COLUMN IF NOT EXISTS {col} DOUBLE PRECISION")
