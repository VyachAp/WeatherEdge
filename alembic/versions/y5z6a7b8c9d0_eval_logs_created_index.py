"""evaluation_logs: add bare created_at index for time-window reports

Revision ID: y5z6a7b8c9d0
Revises: x4y5z6a7b8c9
Create Date: 2026-07-09 00:00:00.000000

The only non-PK index on ``evaluation_logs`` (~2M rows / 868 MB) is the
composite ``ix_eval_logs_market_created (market_id, created_at)``. Because
``created_at`` is the *second* column, Postgres cannot range-scan it for the
time-window queries the reports run (``created_at >= ? ORDER BY created_at``):
perf-review / evals-report / decisions-report / no-trade-report. That forced a
full index scan across every ``market_id`` group + an external merge sort
(~18.6 s for a 7-day window, measured via EXPLAIN ANALYZE on prod).

Add a single-column btree on ``created_at`` so those scans become a true range
scan in ``created_at`` order (no sort node). Keeps the composite for the
per-market lookups (calibration/valley/resolution joins).

Built CONCURRENTLY: this runs against a live table the scheduler writes to every
tick, so a plain ``CREATE INDEX`` would take a write-blocking lock for the whole
build. CONCURRENTLY cannot run inside a transaction, so we drop out of alembic's
implicit per-migration transaction first. ``IF NOT EXISTS`` makes this a no-op if
the index was already built by hand on prod.
"""
from typing import Sequence, Union

from alembic import op


revision: str = "y5z6a7b8c9d0"
down_revision: Union[str, None] = "x4y5z6a7b8c9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Leave alembic's implicit transaction; CONCURRENTLY can't run in one.
    op.execute("COMMIT")
    op.execute(
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_eval_logs_created "
        "ON evaluation_logs (created_at)"
    )


def downgrade() -> None:
    op.execute("COMMIT")
    op.execute("DROP INDEX CONCURRENTLY IF EXISTS ix_eval_logs_created")
