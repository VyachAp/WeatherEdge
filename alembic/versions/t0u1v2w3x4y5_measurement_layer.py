"""measurement layer: eval shadow_json + config_epochs + exposure_snapshots + market_resolutions

Revision ID: t0u1v2w3x4y5
Revises: s9t0u1v2w3x4
Create Date: 2026-05-30 18:00:00.000000

Phase 0 of the profitability roadmap — make every later feature flip
data-proven:

  * ``evaluation_logs.shadow_json`` (M1) — JSONB counterfactual column.
    Lets an experiment stash the would-be value of a not-yet-live feature
    (e.g. per-class vs pooled calibration) so a flip is validated *before*
    it's enabled. NULL until an experiment opts in.
  * ``config_epochs`` (M2) — one row per distinct trading-config snapshot,
    written at startup only when the tracked flags change. Telemetry rows
    are bucketed into an epoch by timestamp at read time, giving clean
    before/after-flip A/B.
  * ``exposure_snapshots`` (M4) — per-tick capital/headroom timeseries,
    the standing chart behind the Phase-0 capital gate.
  * ``market_resolutions`` (M3) — resolved outcome + implied daily-max
    bound + signed resolver divergence; the de-circularised ground truth
    Phase 3 reads.

All additive and nullable (one new column + three new tables) → instant
metadata-only ALTER on Postgres; running old code ignores them. Apply
BEFORE deploying the writer changes (EXPAND step).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


revision: str = "t0u1v2w3x4y5"
down_revision: Union[str, None] = "s9t0u1v2w3x4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # M1 — counterfactual column on the existing telemetry table.
    op.add_column(
        "evaluation_logs", sa.Column("shadow_json", JSONB(), nullable=True)
    )

    # M2 — config epochs.
    op.create_table(
        "config_epochs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("flags_json", JSONB(), nullable=False),
        sa.Column("note", sa.String(), nullable=True),
    )

    # M4 — per-tick exposure/headroom snapshots.
    op.create_table(
        "exposure_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("equity", sa.Float(), nullable=False),
        sa.Column("exposure", sa.Float(), nullable=False),
        sa.Column("effective_cap", sa.Float(), nullable=False),
        sa.Column("headroom", sa.Float(), nullable=False),
        sa.Column("n_open", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("n_open_by_class", JSONB(), nullable=True),
        sa.Column("dd_level", sa.String(), nullable=True),
        sa.Column("dd_mult", sa.Float(), nullable=True),
        sa.Column("n_candidates", sa.Integer(), nullable=True),
        sa.Column("n_sized_to_zero", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_exposure_snapshots_created", "exposure_snapshots", ["created_at"]
    )

    # M3 — resolved outcome + resolver-divergence ground truth.
    op.create_table(
        "market_resolutions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "market_id",
            sa.String(),
            sa.ForeignKey("markets.id"),
            nullable=False,
        ),
        sa.Column("station_icao", sa.String(), nullable=True),
        sa.Column("parsed_location", sa.String(), nullable=True),
        sa.Column("target_date_local", sa.Date(), nullable=True),
        sa.Column("unit", sa.String(), nullable=True),
        sa.Column("parsed_operator", sa.String(), nullable=True),
        sa.Column("parsed_threshold", sa.Float(), nullable=True),
        sa.Column("bucket_low_f", sa.Float(), nullable=True),
        sa.Column("bucket_high_f", sa.Float(), nullable=True),
        sa.Column("yes_won", sa.Boolean(), nullable=False),
        sa.Column("resolved_max_lower_f", sa.Float(), nullable=True),
        sa.Column("resolved_max_upper_f", sa.Float(), nullable=True),
        sa.Column("routine_metar_max_f", sa.Float(), nullable=True),
        sa.Column("divergence_f", sa.Float(), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("market_id", name="uq_market_resolution_market"),
    )
    op.create_index(
        "ix_market_resolution_icao_date",
        "market_resolutions",
        ["station_icao", "target_date_local"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_market_resolution_icao_date", table_name="market_resolutions"
    )
    op.drop_table("market_resolutions")
    op.drop_index(
        "ix_exposure_snapshots_created", table_name="exposure_snapshots"
    )
    op.drop_table("exposure_snapshots")
    op.drop_table("config_epochs")
    op.drop_column("evaluation_logs", "shadow_json")
