"""Phase 0.2 regression tests: dry-run must not produce phantom fills.

These tests pin the contract that `place_order` provides to its callers in
dry-run mode, plus the scheduler's dry-run handling for the probability
path (the lock-rule path's equivalent guard at `scheduler.py:1014-1020` is
covered by the same contract).

Background: previously the unified pipeline marked probability-path trades
as `OPEN` with the requested stake whenever `place_order` returned True,
even in dry-run — `place_order` returns True without filling, so
`trade.stake_usd` retained its requested value and resolution attributed
fictitious P&L against `entry_price=edge.market_price`. The fix was to
read `trade.exchange_status == "dry_run"` after `place_order` returns and
explicitly zero `stake_usd` while keeping `status=PENDING`.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.db.models import Trade, TradeDirection, TradeStatus
from src.execution.polymarket_client import is_live, place_order


def _trade(stake_usd: float = 10.0, entry_price: float = 0.55) -> Trade:
    return Trade(
        market_id="m_test",
        direction=TradeDirection.BUY_YES,
        stake_usd=stake_usd,
        entry_price=entry_price,
        status=TradeStatus.PENDING,
    )


def test_is_live_false_when_no_private_key():
    """The dry-run sentinel — when this is False, `place_order` short-circuits."""
    with patch("src.execution.polymarket_client.settings") as mock_settings:
        mock_settings.POLYMARKET_PRIVATE_KEY = ""
        mock_settings.AUTO_EXECUTE = True
        assert is_live() is False


def test_is_live_false_when_auto_execute_off():
    with patch("src.execution.polymarket_client.settings") as mock_settings:
        mock_settings.POLYMARKET_PRIVATE_KEY = "0xdead"
        mock_settings.AUTO_EXECUTE = False
        assert is_live() is False


def test_place_order_dry_run_sets_exchange_status_and_returns_true():
    """Dry-run contract: returns True, stamps `exchange_status="dry_run"`,
    but does NOT touch `stake_usd` or `status`. Callers must read the
    sentinel and zero `stake_usd` themselves to avoid phantom fills."""
    trade = _trade(stake_usd=12.34, entry_price=0.40)
    session = SimpleNamespace()  # never touched in dry-run branch

    with patch("src.execution.polymarket_client.is_live", return_value=False):
        result = asyncio.run(place_order(trade, session))

    assert result is True
    assert trade.exchange_status == "dry_run"
    # Critical: place_order does NOT mutate stake_usd in dry-run. The bug
    # was assuming it did. If this assertion ever flips, callers' dry-run
    # guards become dead code; revisit them.
    assert trade.stake_usd == 12.34
    assert trade.status == TradeStatus.PENDING


def _apply_probability_post_place_outcome(
    trade: Trade,
    order_ok: bool,
    requested_stake: float,
) -> tuple[TradeStatus, float, str]:
    """Reproduces the dry-run/live branch logic at `scheduler.py:566-590`
    in isolation so we can exercise it without spinning up the full
    `job_unified_pipeline`. Keep this in sync with the scheduler if that
    block changes shape.

    Returns (final_status, final_stake_usd, label) where `label` is
    "dry_run", "open", "no_fill", or "place_failed".
    """
    is_dry_run = trade.exchange_status == "dry_run"
    if order_ok and is_dry_run:
        trade.stake_usd = 0.0
        trade.status = TradeStatus.PENDING
        return trade.status, trade.stake_usd, "dry_run"
    if order_ok and (trade.stake_usd or 0.0) > 0:
        trade.status = TradeStatus.OPEN
        return trade.status, trade.stake_usd, "open"
    if order_ok:
        trade.status = TradeStatus.PENDING
        return trade.status, trade.stake_usd, "no_fill"
    return trade.status, trade.stake_usd, "place_failed"


def test_dry_run_path_keeps_pending_and_zeros_stake():
    trade = _trade(stake_usd=15.0)
    trade.exchange_status = "dry_run"  # what place_order leaves behind

    status, stake, label = _apply_probability_post_place_outcome(
        trade, order_ok=True, requested_stake=15.0,
    )

    assert label == "dry_run"
    assert status == TradeStatus.PENDING
    assert stake == 0.0


def test_live_fill_marks_open_with_actual_stake():
    trade = _trade(stake_usd=15.0)
    trade.exchange_status = "live"
    # In a real fill, place_order would have updated stake_usd to the
    # actual matched amount via _update_fill_details. Simulate a $14
    # partial fill.
    trade.stake_usd = 14.0

    status, stake, label = _apply_probability_post_place_outcome(
        trade, order_ok=True, requested_stake=15.0,
    )

    assert label == "open"
    assert status == TradeStatus.OPEN
    assert stake == 14.0


def test_live_no_fill_keeps_pending():
    trade = _trade(stake_usd=15.0)
    trade.exchange_status = "matched"
    # FAK posted but didn't match — `_update_fill_details` zeroed stake_usd.
    trade.stake_usd = 0.0

    status, stake, label = _apply_probability_post_place_outcome(
        trade, order_ok=True, requested_stake=15.0,
    )

    assert label == "no_fill"
    assert status == TradeStatus.PENDING
    assert stake == 0.0


def test_place_failed_does_not_open():
    trade = _trade(stake_usd=15.0)
    trade.exchange_status = "failed: rpc"

    status, stake, label = _apply_probability_post_place_outcome(
        trade, order_ok=False, requested_stake=15.0,
    )

    assert label == "place_failed"
    # Still PENDING from initial creation; the failure path does not flip
    # status or stake either way.
    assert status == TradeStatus.PENDING


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
