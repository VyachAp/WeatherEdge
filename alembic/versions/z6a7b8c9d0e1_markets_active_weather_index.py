"""markets: index (parsed_variable, end_date) for the active-market lookup

Revision ID: z6a7b8c9d0e1
Revises: y5z6a7b8c9d0
Create Date: 2026-07-14 00:00:00.000000

``get_active_weather_markets`` runs

    SELECT * FROM markets
     WHERE parsed_variable = 'temperature' AND end_date > now()
     ORDER BY end_date

on **every** ``job_fast_lock_poll`` tick (10 s) and every unified tick (5 min).
``markets`` only carried a PK and ``ix_markets_condition_id``, so this was a
sequential scan of the whole table — 55,428 rows scanned to return 880, 3.2 s of
DB execution time measured via EXPLAIN ANALYZE on prod, ~8 s wall-clock in the
job.

That mattered because it is the *fixed* cost of the fast-poll tick: the tick's
total fixed cost was ~24 s against a 10 s interval, so APScheduler dropped 63 %
of ticks ("maximum number of running instances reached") and the effective poll
period was ~27 s. ``bucket_overshoot`` is a seconds-scale race whose EV decays to
zero by ~60 s of decision delay, so that self-inflicted lag was pure EV decay.

A composite btree with ``parsed_variable`` (equality) first and ``end_date``
(range + ORDER BY) second turns the seq scan into an index range scan that also
satisfies the sort.

Built CONCURRENTLY — ``job_scan_markets`` writes this table every 15 min and a
plain CREATE INDEX would hold a write-blocking lock for the whole build.
"""
from typing import Sequence, Union

from alembic import op


revision: str = "z6a7b8c9d0e1"
down_revision: Union[str, None] = "y5z6a7b8c9d0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Leave alembic's implicit transaction; CONCURRENTLY can't run in one.
    op.execute("COMMIT")
    op.execute(
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_markets_active_weather "
        "ON markets (parsed_variable, end_date)"
    )


def downgrade() -> None:
    op.execute("COMMIT")
    op.execute("DROP INDEX CONCURRENTLY IF EXISTS ix_markets_active_weather")
