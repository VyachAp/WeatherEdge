"""dedupe signals + collapse duplicate PENDING trades; add UNIQUE (market_id, direction)

Revision ID: i9j0k1l2m3n4
Revises: h8i9j0k1l2m3
Create Date: 2026-05-04 14:00:00.000000

The unified pipeline created a fresh ``signals`` row each tick, even when
the same ``(market_id, direction)`` was being re-evaluated unchanged. The
in-process dedup at ``scheduler.py`` only covered the dry-run probability
path and was reset on restart, so by 2026-05-04 the table held 444 rows
mapping to 166 distinct ``(market_id, direction)`` pairs.

Companion problem on ``trades``: failed-FAK retries and dry-run repeats
left multiple PENDING rows for the same pair on still-active markets.
The DB-backed safety guard introduced in this change skips evaluation
when any active Trade exists for the pair, so those duplicates would
permanently block their markets unless collapsed first.

This migration:
  1. Repoints every ``trades.signal_id`` referencing a non-latest
     duplicate signal at the latest signal in its
     ``(market_id, direction)`` group, then deletes the older signals.
  2. Adds ``UNIQUE (market_id, direction)`` on ``signals`` so future
     UPSERTs in scheduler.py refresh in place.
  3. Collapses duplicate PENDING trade rows for the same pair on
     unresolved markets — keeps the latest by id, deletes the rest.
"""
from typing import Sequence, Union

from alembic import op


revision: str = "i9j0k1l2m3n4"
down_revision: Union[str, None] = "h8i9j0k1l2m3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1a. Repoint trades at the latest signal in each (market_id, direction) group.
    op.execute("""
        WITH latest AS (
            SELECT market_id, direction, MAX(id) AS keep_id
            FROM signals
            GROUP BY market_id, direction
        )
        UPDATE trades t
        SET signal_id = latest.keep_id
        FROM signals s
        JOIN latest
          ON latest.market_id = s.market_id
         AND latest.direction = s.direction
        WHERE t.signal_id = s.id
          AND s.id < latest.keep_id;
    """)

    # 1b. Delete non-latest signals.
    op.execute("""
        DELETE FROM signals s
        USING (
            SELECT market_id, direction, MAX(id) AS keep_id
            FROM signals
            GROUP BY market_id, direction
        ) latest
        WHERE s.market_id = latest.market_id
          AND s.direction = latest.direction
          AND s.id < latest.keep_id;
    """)

    # 2. Now that the table is clean, add the unique constraint.
    op.create_unique_constraint(
        "uq_signals_market_direction",
        "signals",
        ["market_id", "direction"],
    )

    # 3. Collapse duplicate PENDING trades for the same (market_id, direction)
    #    on still-active markets. Without this step the new active-trade
    #    guard would permanently block any market that previously accumulated
    #    multiple PENDING dry-run rows.
    op.execute("""
        WITH latest_pending AS (
            SELECT t.market_id, t.direction, MAX(t.id) AS keep_id
            FROM trades t
            JOIN markets m ON m.id = t.market_id
            WHERE t.status = 'PENDING'
              AND m.end_date >= NOW()
            GROUP BY t.market_id, t.direction
            HAVING COUNT(*) > 1
        )
        DELETE FROM trades t
        USING latest_pending lp
        WHERE t.market_id = lp.market_id
          AND t.direction = lp.direction
          AND t.status = 'PENDING'
          AND t.id < lp.keep_id;
    """)


def downgrade() -> None:
    op.drop_constraint("uq_signals_market_direction", "signals", type_="unique")
    # The data deletions are not reversible — duplicate signal/trade rows
    # are gone after upgrade. Downgrade only releases the schema constraint.
