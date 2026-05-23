"""evaluation_logs forecast/observation context columns

Revision ID: p6q7r8s9t0u1
Revises: o5p6q7r8s9t0
Create Date: 2026-05-23 12:00:00.000000

The single-bucket NO guards shipped 2026-05-23 (landing-band + peak-relative
lead gate, in ``binary_market_edge``) decide off ``forecast_peak_f`` /
``current_max_f`` / ``hours_until_peak`` — none of which were captured in
``evaluation_logs``. ``evaluation_logs`` is the documented source of truth
for filter-tuning backtests, so without these we can't recompute the guards
from telemetry or re-fit their margins (``SINGLE_BUCKET_NO_BAND_MARGIN_F``,
``EXACTLY_MAX_LEAD_HOURS``) from live data — the exact blind spot that made
the 2026-05-23 impact analysis fall back to estimates for Layer 2.

Four nullable Float columns (no default → metadata-only ALTER, instant on
Postgres). ``forecast_sigma_f`` is the ensemble peak σ, captured to track
the σ-collapse root cause behind the deferred lead-time-aware σ-floor work.
NULL on lock-path rows without forecast context and on all legacy rows.

EXPAND step: apply this migration BEFORE deploying the writer change, so the
running scheduler (old code) ignores the new columns and the new code's
INSERTs land cleanly.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "p6q7r8s9t0u1"
down_revision: Union[str, None] = "o5p6q7r8s9t0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("evaluation_logs", sa.Column("forecast_peak_f", sa.Float(), nullable=True))
    op.add_column("evaluation_logs", sa.Column("current_max_f", sa.Float(), nullable=True))
    op.add_column("evaluation_logs", sa.Column("hours_until_peak", sa.Float(), nullable=True))
    op.add_column("evaluation_logs", sa.Column("forecast_sigma_f", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("evaluation_logs", "forecast_sigma_f")
    op.drop_column("evaluation_logs", "hours_until_peak")
    op.drop_column("evaluation_logs", "current_max_f")
    op.drop_column("evaluation_logs", "forecast_peak_f")
