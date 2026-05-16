"""raw_model_prob + calibrated columns on signals + evaluation_logs

Revision ID: l2m3n4o5p6q7
Revises: k1l2m3n4o5p6
Create Date: 2026-05-16 12:00:00.000000

The probability-path calibration loop in ``consensus.py`` was being fit
against the **calibrated** value stored in ``signals.model_prob`` —
because that column held the post-calibration probability, not the raw
engine output. Re-fitting a regression against its own corrected output
is a feedback loop that defeats calibration. Audit on 2026-05-16 found
the probability path Brier at 0.229 (vs 0.111 for the lock path) with
the ``model_prob ≈ 1.0`` bin only winning 78 % of the time.

This migration adds two nullable columns on both ``signals`` and
``evaluation_logs``:

- ``raw_model_prob`` — engine probability *before* calibration was
  applied. The calibration regression in
  ``consensus.get_calibration_coefficients`` re-fits from this column
  (with ``COALESCE(raw_model_prob, model_prob)`` for legacy rows that
  pre-date the column).
- ``calibrated`` — flag indicating whether the value in ``model_prob``
  was actually corrected this tick. Lets ``calibrated=True/False`` be
  SQL-checked directly instead of via log scraping. ``MIN_CALIBRATION_
  SAMPLES=50`` and a cold in-process cache after restart both produce
  ``calibrated=False`` rows.

Columns are nullable so the migration is non-blocking; old rows stay
readable and downstream code falls back to ``model_prob`` when raw is
NULL.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "l2m3n4o5p6q7"
down_revision: Union[str, None] = "k1l2m3n4o5p6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("signals", sa.Column("raw_model_prob", sa.Float(), nullable=True))
    op.add_column("signals", sa.Column("calibrated", sa.Boolean(), nullable=True))
    op.add_column(
        "evaluation_logs",
        sa.Column("raw_model_prob", sa.Float(), nullable=True),
    )
    op.add_column(
        "evaluation_logs",
        sa.Column("calibrated", sa.Boolean(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("evaluation_logs", "calibrated")
    op.drop_column("evaluation_logs", "raw_model_prob")
    op.drop_column("signals", "calibrated")
    op.drop_column("signals", "raw_model_prob")
