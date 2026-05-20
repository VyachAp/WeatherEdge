"""trade exchange_error column for full submission failure messages

Revision ID: o5p6q7r8s9t0
Revises: n4o5p6q7r8s9
Create Date: 2026-05-20 17:00:00.000000

Pre-fix, ``polymarket_client.place_order`` truncated submission exceptions
to ``f"exception:{type(exc).__name__}"[:50]`` and only the class name landed
in ``Trade.exchange_status``. With 115 ``exception:PolyApiException`` rows
piling up in a single week (2026-05-13 → 05-20) and no diagnostic payload
attached, every postmortem required digging through DigitalOcean logs to
correlate timestamps — a workflow that breaks down once log retention
rotates.

``exchange_error`` (nullable Text) captures the truncated full ``str(exc)``
alongside the existing short status. Downstream queries that match on
``exchange_status IN ('matched','delayed','exception:*')`` stay valid;
``exchange_error`` is read by future log dumps / dashboards only.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "o5p6q7r8s9t0"
down_revision: Union[str, None] = "n4o5p6q7r8s9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "trades",
        sa.Column("exchange_error", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("trades", "exchange_error")
