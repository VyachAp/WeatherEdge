"""replace overloaded Signal.confidence with Signal.lock_margin_f

Revision ID: r8s9t0u1v2w3
Revises: q7r8s9t0u1v2
Create Date: 2026-05-30 12:30:00.000000

``Signal.confidence`` was overloaded across the two trading paths:

  * probability path wrote ``confidence = our_probability`` — but that is
    already stored in ``model_prob`` on the same row, so the column was
    a pure duplicate
  * lock path wrote ``confidence = decision.margin_f`` — a °F margin, not
    a probability — so the same column carried values like ``4.5`` (°F)
    on lock rows next to ``0.97`` (probability) on prob rows

The alerter Detail view and ``bet info`` formatted both as
``{value:.2f}``, so a lock alert shipped ``Confidence: 5.00`` to the
operator (looks like 500% prob, not a 5°F margin). The Streamlit
dashboard treated the column as a probability throughout (``{:.1%}``
formatter, calibration scatter plot vs edge), which silently broke
for lock signals.

This migration drops the overloaded ``confidence`` column and adds
``lock_margin_f`` as a sibling of the existing
``lock_branch``/``lock_routine_count``/``lock_observed_max_f`` cluster.
The lock path's value is preserved by backfilling
``lock_margin_f = confidence WHERE signal_kind = 'lock'`` before the drop.
The probability path's value was a duplicate of ``model_prob`` and is
discarded (any analysis that needs it can read ``model_prob`` directly).

See ``docs/graveyard.md#2026-05-30-signalconfidence-column-overload``.

Defensive ``IF EXISTS`` / ``IF NOT EXISTS``: the column may already be
absent (Module 1 of this audit pass demonstrated that ``models.py``-only
column declarations can fail to propagate to prod via ``create_all``).
Backfill runs unconditionally — a missing source column yields zero rows
updated, which is the correct no-op.

EXPAND → BACKFILL → CONTRACT: this is one migration but conceptually
three steps in order: add the new column, copy the data, drop the old.
Deploy the new code AFTER this migration runs so ``upsert_signal`` is
not writing to a column it no longer believes exists; alternatively,
the new ``upsert_signal`` will simply not pass ``confidence`` and the
old column (if still around) will go NULL on UPDATE — harmless because
the column is about to be dropped anyway.
"""
from typing import Sequence, Union

from alembic import op


revision: str = "r8s9t0u1v2w3"
down_revision: Union[str, None] = "q7r8s9t0u1v2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # EXPAND — add the new lock-specific column
    op.execute("ALTER TABLE signals ADD COLUMN IF NOT EXISTS lock_margin_f DOUBLE PRECISION")
    # BACKFILL — preserve the lock-path values; probability-path values
    # are duplicates of model_prob and intentionally dropped
    op.execute(
        "UPDATE signals "
        "SET lock_margin_f = confidence "
        "WHERE signal_kind = 'lock' "
        "  AND lock_margin_f IS NULL "
        "  AND confidence IS NOT NULL"
    )
    # CONTRACT — drop the overloaded column
    op.execute("ALTER TABLE signals DROP COLUMN IF EXISTS confidence")


def downgrade() -> None:
    op.execute("ALTER TABLE signals ADD COLUMN IF NOT EXISTS confidence DOUBLE PRECISION")
    op.execute(
        "UPDATE signals "
        "SET confidence = lock_margin_f "
        "WHERE signal_kind = 'lock' "
        "  AND confidence IS NULL "
        "  AND lock_margin_f IS NOT NULL"
    )
    op.execute("ALTER TABLE signals DROP COLUMN IF EXISTS lock_margin_f")
