"""bot_state key/value table

Revision ID: m3n4o5p6q7r8
Revises: l2m3n4o5p6q7
Create Date: 2026-05-16 18:00:00.000000

Adds a generic ``bot_state`` table for durable runtime values that must
survive a process restart. First consumer: the consecutive-loss circuit
breaker's ``_paused_until`` window, which was previously an in-process
variable — a process crash mid-pause silently dropped the protective
window and the bot resumed trading immediately. Keys are dotted strings
so future consumers can namespace their own values without a migration
per new key (value is JSONB).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "m3n4o5p6q7r8"
down_revision: Union[str, None] = "l2m3n4o5p6q7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "bot_state",
        sa.Column("key", sa.String(), nullable=False),
        sa.Column("value", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("key"),
    )


def downgrade() -> None:
    op.drop_table("bot_state")
