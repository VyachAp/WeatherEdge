"""add trade redeemed_at column

Revision ID: g7h8i9j0k1l2
Revises: f6g7h8i9j0k1
Create Date: 2026-05-02 12:00:00.000000

Backfill assumes pre-existing WON trades are already redeemed (sets
``redeemed_at = closed_at``). If you have unredeemed wins at deploy time,
NULL them out manually before this migration runs, or run ``bet redeem``
first.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "g7h8i9j0k1l2"
down_revision: Union[str, None] = "f6g7h8i9j0k1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "trades",
        sa.Column("redeemed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_trades_redeemed_at", "trades", ["redeemed_at"])
    # SQLAlchemy's ``Enum(TradeStatus)`` stores enum *names* (uppercase),
    # not ``.value`` strings — the Postgres ``tradestatus`` enum has
    # values 'PENDING'/'OPEN'/'WON'/'LOST'.
    op.execute(
        "UPDATE trades SET redeemed_at = closed_at "
        "WHERE status = 'WON' AND closed_at IS NOT NULL"
    )


def downgrade() -> None:
    op.drop_index("ix_trades_redeemed_at", table_name="trades")
    op.drop_column("trades", "redeemed_at")
