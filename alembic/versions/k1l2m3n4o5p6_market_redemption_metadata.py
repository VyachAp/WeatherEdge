"""market redemption metadata: condition_id, neg_risk, clob_token_ids

Revision ID: k1l2m3n4o5p6
Revises: j0k1l2m3n4o5
Create Date: 2026-05-15 12:00:00.000000

`bet redeem` constructs the on-chain `redeemPositions()` call from the
market's `conditionId`, `negRisk`, and `clobTokenIds`. Today these are
fetched live from Gamma's `GET /markets?id=<market_id>` at redeem time,
but Gamma drops resolved/closed markets from that index and returns
`[]` — stranding redemption for any trade whose market has already
resolved.

Caching this trio on the `markets` row at scan time (every 15 min while
the market is active) makes redemption Gamma-independent. All three
columns are nullable so the migration is non-blocking; ingestion fills
them on the next `job_scan_markets` tick.

Index on `condition_id` keeps the reverse lookup cheap should we ever
need to map an on-chain id back to a `Market` row.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "k1l2m3n4o5p6"
down_revision: Union[str, None] = "j0k1l2m3n4o5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("markets", sa.Column("condition_id", sa.String(), nullable=True))
    op.add_column("markets", sa.Column("neg_risk", sa.Boolean(), nullable=True))
    op.add_column(
        "markets",
        sa.Column("clob_token_ids", postgresql.JSONB(), nullable=True),
    )
    op.create_index("ix_markets_condition_id", "markets", ["condition_id"])


def downgrade() -> None:
    op.drop_index("ix_markets_condition_id", table_name="markets")
    op.drop_column("markets", "clob_token_ids")
    op.drop_column("markets", "neg_risk")
    op.drop_column("markets", "condition_id")
