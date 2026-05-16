"""Tests for ``signals.decision_log.log_decision``.

The helper writes one ``DecisionLog`` row per per-side post-filter
decision. Contracts that matter:

* Calls ``session.add(DecisionLog(...))`` once per invocation — no
  flush, no commit (caller batches with surrounding writes).
* All thirteen outcome constants are importable and distinct strings
  (downstream SQL aggregation depends on stable enum values).
* Optional fields (stake / dd / metadata) round-trip onto the row.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.db.models import DecisionLog, TradeDirection
from src.signals import decision_log as dl


def test_outcome_constants_distinct_strings() -> None:
    constants = [
        dl.OUTCOME_SIGNAL_WRITTEN,
        dl.OUTCOME_TRADE_PENDING,
        dl.OUTCOME_TRADE_FILLED,
        dl.OUTCOME_DUP_INPROC,
        dl.OUTCOME_DUP_DB,
        dl.OUTCOME_STAKE_BELOW_MIN,
        dl.OUTCOME_DRAWDOWN_PAUSED,
        dl.OUTCOME_CLUSTER_CAP,
        dl.OUTCOME_CAP_EXCEEDED,
        dl.OUTCOME_NO_TOKEN_IDS,
        dl.OUTCOME_NO_CLIENT,
        dl.OUTCOME_NO_FILL,
        dl.OUTCOME_ORDER_FAILED,
    ]
    assert all(isinstance(c, str) and c for c in constants)
    assert len(set(constants)) == len(constants), "duplicate outcome string"


@pytest.mark.asyncio
async def test_log_decision_adds_row_with_minimal_fields() -> None:
    session = MagicMock()
    session.add = MagicMock()

    await dl.log_decision(
        session,
        market_id="m1",
        direction=TradeDirection.BUY_NO,
        signal_kind="lock",
        outcome=dl.OUTCOME_DUP_INPROC,
    )

    session.add.assert_called_once()
    row = session.add.call_args.args[0]
    assert isinstance(row, DecisionLog)
    assert row.market_id == "m1"
    assert row.direction == TradeDirection.BUY_NO
    assert row.signal_kind == "lock"
    assert row.outcome == dl.OUTCOME_DUP_INPROC
    # Optional fields default to None
    assert row.requested_stake_usd is None
    assert row.actual_stake_usd is None
    assert row.dd_multiplier is None
    assert row.dd_level is None
    assert row.metadata_json is None


@pytest.mark.asyncio
async def test_log_decision_passes_through_sizing_and_metadata() -> None:
    session = MagicMock()
    session.add = MagicMock()

    await dl.log_decision(
        session,
        market_id="m2",
        direction=TradeDirection.BUY_YES,
        signal_kind="probability",
        outcome=dl.OUTCOME_STAKE_BELOW_MIN,
        requested_stake_usd=12.5,
        actual_stake_usd=4.2,
        dd_multiplier=0.5,
        dd_level="caution",
        metadata={"bucket": 72, "edge": 0.08},
    )

    row = session.add.call_args.args[0]
    assert row.requested_stake_usd == 12.5
    assert row.actual_stake_usd == 4.2
    assert row.dd_multiplier == 0.5
    assert row.dd_level == "caution"
    assert row.metadata_json == {"bucket": 72, "edge": 0.08}


@pytest.mark.asyncio
async def test_log_decision_does_not_flush_or_commit() -> None:
    """Caller batches flush/commit with surrounding writes — log_decision
    must not force its own DB round-trip."""
    session = MagicMock()
    session.add = MagicMock()
    session.flush = MagicMock()
    session.commit = MagicMock()

    await dl.log_decision(
        session,
        market_id="m3",
        direction=TradeDirection.BUY_NO,
        signal_kind="lock",
        outcome=dl.OUTCOME_TRADE_FILLED,
    )

    session.flush.assert_not_called()
    session.commit.assert_not_called()
