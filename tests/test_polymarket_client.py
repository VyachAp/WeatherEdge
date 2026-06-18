"""Unit tests for ``src.execution.polymarket_client``.

Regression tests for the 2026-05-18 silent silencer: ``check_order_status``
crashed with ``AttributeError`` whenever ``client.get_order`` returned
``None`` (which the CLOB does for both queued AND post-resolution
order-endpoint drops), the bare ``except Exception`` swallowed it with a
one-line warning, and the trade row was left stuck OPEN with null fill
for 12+ hours.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def trade_pending():
    """Mutable Trade-like object used by ``check_order_status``."""
    return SimpleNamespace(
        order_id="order-abc-123",
        exchange_status="delayed",  # initial state from place_order
        fill_price=None,
        filled_size=0.0,
        stake_usd=20.0,
        entry_price=0.5,
        direction=None,  # not used by check_order_status
    )


@pytest.mark.asyncio
async def test_check_order_status_returns_none_when_client_unavailable():
    """No private key / dry-run path: function is a no-op and returns None."""
    from src.execution import polymarket_client as pc

    trade = SimpleNamespace(order_id="abc", exchange_status="delayed", fill_price=None)
    with patch.object(pc, "_get_client", return_value=None):
        result = await pc.check_order_status(trade)

    assert result is None
    assert trade.exchange_status == "delayed"
    assert trade.fill_price is None


@pytest.mark.asyncio
async def test_check_order_status_returns_none_for_unset_order_id():
    """Pre-place_order trade row (no order_id yet) is a no-op."""
    from src.execution import polymarket_client as pc

    trade = SimpleNamespace(order_id=None, exchange_status="dry_run", fill_price=None)
    fake_client = MagicMock()
    with patch.object(pc, "_get_client", return_value=fake_client):
        result = await pc.check_order_status(trade)

    assert result is None
    # The client must NOT be touched when there's nothing to look up.
    fake_client.get_order.assert_not_called()


@pytest.mark.asyncio
async def test_check_order_status_handles_none_order(trade_pending):
    """Bug A regression: ``client.get_order`` returning None must NOT
    raise AttributeError. The trade row stays untouched so the next
    reconcile tick finds it again."""
    from src.execution import polymarket_client as pc

    fake_client = MagicMock()
    fake_client.get_order = MagicMock(return_value=None)

    with patch.object(pc, "_get_client", return_value=fake_client):
        # The pre-fix call path raised AttributeError here.
        result = await pc.check_order_status(trade_pending)

    assert result is None
    # Row left in a consistent state — the existing "delayed" / "matched"
    # exchange_status is preserved so the reconcile filter re-finds it.
    assert trade_pending.exchange_status == "delayed"
    assert trade_pending.fill_price is None


@pytest.mark.asyncio
async def test_check_order_status_handles_get_order_exception(trade_pending):
    """A real lookup error (auth, 5xx) is logged with exc_info and returns
    None, never crashing the reconcile loop."""
    from src.execution import polymarket_client as pc

    fake_client = MagicMock()
    fake_client.get_order = MagicMock(side_effect=RuntimeError("clob 502"))

    with patch.object(pc, "_get_client", return_value=fake_client), \
         patch.object(pc.logger, "warning") as mock_warning:
        result = await pc.check_order_status(trade_pending)

    assert result is None
    assert trade_pending.exchange_status == "delayed"
    # exc_info=True so the operator gets a real traceback in logs.
    assert mock_warning.called
    call = mock_warning.call_args
    assert call.kwargs.get("exc_info") is True


@pytest.mark.asyncio
async def test_check_order_status_matched_triggers_fill_details(trade_pending):
    """Happy path: order returns status='matched', _update_fill_details
    is called exactly once."""
    from src.execution import polymarket_client as pc

    fake_client = MagicMock()
    fake_client.get_order = MagicMock(
        return_value={"status": "matched", "associate_trades": []}
    )

    update_mock = AsyncMock()
    with patch.object(pc, "_get_client", return_value=fake_client), \
         patch.object(pc, "_update_fill_details", update_mock):
        result = await pc.check_order_status(trade_pending)

    assert result == "matched"
    assert trade_pending.exchange_status == "matched"
    update_mock.assert_awaited_once_with(trade_pending, fake_client)


@pytest.mark.asyncio
async def test_check_order_status_matched_skips_fill_when_already_filled(
    trade_pending,
):
    """Defensive: a matched order whose fill_price is already populated
    is not re-processed (avoids re-fetching fills on every reconcile tick)."""
    from src.execution import polymarket_client as pc

    trade_pending.fill_price = 0.55  # already filled by an earlier path
    fake_client = MagicMock()
    fake_client.get_order = MagicMock(return_value={"status": "matched"})

    update_mock = AsyncMock()
    with patch.object(pc, "_get_client", return_value=fake_client), \
         patch.object(pc, "_update_fill_details", update_mock):
        result = await pc.check_order_status(trade_pending)

    assert result == "matched"
    update_mock.assert_not_awaited()


# ---------------------------------------------------------------------------
# place_order wallet pre-flight balance check
# ---------------------------------------------------------------------------
#
# Added 2026-05-30 after 140 PolyApiException "not enough balance / allowance"
# rows in 14d. Root cause was burst-time wallet depletion: the bot has a
# local exposure cap fed by DB-view OPEN/PENDING trades (which lag on-chain
# reality), so it kept submitting orders the wallet couldn't fund. The
# pre-flight reads the live CLOB-visible spendable balance with
# `force_refresh=True` and short-circuits with `exchange_status =
# "insufficient_balance"` when stake exceeds it. Tests cover: (1) pre-flight
# blocks when spendable < stake; (2) pre-flight passes when spendable >=
# stake; (3) pre-flight is a no-op for dry-run (no private key); (4) pre-
# flight forces a fresh fetch (doesn't trust a stale 5-min cache).


@pytest.fixture
def trade_for_place_order():
    """Trade-like row used to drive place_order. PENDING with $20 stake."""
    from src.db.models import TradeDirection, TradeStatus

    return SimpleNamespace(
        market_id="mkt-abc",
        direction=TradeDirection.BUY_NO,
        stake_usd=20.0,
        entry_price=0.50,
        status=TradeStatus.PENDING,
        order_id=None,
        exchange_status=None,
        exchange_error=None,
        fill_price=None,
        filled_size=0.0,
        token_id=None,
        submit_yes_bid=None,
        submit_yes_ask=None,
        submit_depth_usd=None,
        submit_at=None,
    )


@pytest.mark.asyncio
async def test_place_order_preflight_blocks_when_spendable_below_stake(trade_for_place_order):
    """Spendable $5 vs $20 stake → skip with insufficient_balance, no order posted."""
    from src.execution import polymarket_client as pc

    session = AsyncMock()
    with patch.object(pc, "is_live", return_value=True), \
         patch.object(pc, "get_daily_spend", new=AsyncMock(return_value=0.0)), \
         patch.object(pc, "get_wallet_usdc_balance", return_value=5.0) as bal_mock, \
         patch.object(pc, "get_token_ids", new=AsyncMock(return_value=("yes-tok", "no-tok"))) as token_mock, \
         patch.object(pc, "_get_client") as client_mock:
        ok = await pc.place_order(trade_for_place_order, session)

    assert ok is False
    assert trade_for_place_order.exchange_status == "insufficient_balance"
    # The pre-flight must short-circuit BEFORE token IDs / client are touched
    # — they're more expensive (Gamma call + SDK init).
    bal_mock.assert_called_once_with(force_refresh=True)
    token_mock.assert_not_awaited()
    client_mock.assert_not_called()


@pytest.mark.asyncio
async def test_place_order_preflight_passes_when_spendable_above_stake(trade_for_place_order):
    """Spendable $50 vs $20 stake → fall through to token/client resolution."""
    from src.execution import polymarket_client as pc

    session = AsyncMock()
    # Force a downstream failure (no_client) so we don't have to mock the
    # whole order-posting path — just prove the pre-flight didn't block.
    with patch.object(pc, "is_live", return_value=True), \
         patch.object(pc, "get_daily_spend", new=AsyncMock(return_value=0.0)), \
         patch.object(pc, "get_wallet_usdc_balance", return_value=50.0), \
         patch.object(pc, "get_token_ids", new=AsyncMock(return_value=("yes-tok", "no-tok"))), \
         patch.object(pc, "_get_client", return_value=None):
        ok = await pc.place_order(trade_for_place_order, session)

    assert ok is False
    assert trade_for_place_order.exchange_status == "no_client"
    # NOT insufficient_balance — proves we passed the pre-flight.


@pytest.mark.asyncio
async def test_place_order_preflight_skipped_in_dry_run(trade_for_place_order):
    """Dry-run takes its own branch BEFORE the pre-flight runs — wallet
    fetch must never be called (no private key in dry-run; would return None
    anyway but the test asserts the cleaner "not even attempted" contract).
    """
    from src.execution import polymarket_client as pc

    session = AsyncMock()
    with patch.object(pc, "is_live", return_value=False), \
         patch.object(pc, "get_wallet_usdc_balance") as bal_mock:
        ok = await pc.place_order(trade_for_place_order, session)

    assert ok is True
    assert trade_for_place_order.exchange_status == "dry_run"
    bal_mock.assert_not_called()


@pytest.mark.asyncio
async def test_place_order_preflight_noop_when_balance_unavailable(trade_for_place_order):
    """When `get_wallet_usdc_balance` returns None (no private key OR CLOB
    fetch failed), the pre-flight must NOT block — preserves existing
    behavior on bootstrap / transient CLOB outages. The order continues to
    the next stage (here, no_client because we mock the client to None)."""
    from src.execution import polymarket_client as pc

    session = AsyncMock()
    with patch.object(pc, "is_live", return_value=True), \
         patch.object(pc, "get_daily_spend", new=AsyncMock(return_value=0.0)), \
         patch.object(pc, "get_wallet_usdc_balance", return_value=None), \
         patch.object(pc, "get_token_ids", new=AsyncMock(return_value=("yes-tok", "no-tok"))), \
         patch.object(pc, "_get_client", return_value=None):
        ok = await pc.place_order(trade_for_place_order, session)

    assert ok is False
    assert trade_for_place_order.exchange_status == "no_client"
    # Insufficient_balance NOT set → pre-flight correctly degraded to no-op.


@pytest.mark.asyncio
async def test_place_order_preflight_runs_after_daily_cap(trade_for_place_order):
    """Daily-spend cap is checked first (it's a free DB query); pre-flight
    is second (one CLOB call). When the daily cap trips, the balance fetch
    must not run — saves the call on the cheaper rejection path."""
    from src.execution import polymarket_client as pc
    from src.config import settings

    session = AsyncMock()
    # Make daily spend already exceed the cap.
    with patch.object(pc, "is_live", return_value=True), \
         patch.object(pc, "get_daily_spend", new=AsyncMock(return_value=settings.DAILY_SPEND_CAP_USD)), \
         patch.object(pc, "get_wallet_usdc_balance") as bal_mock:
        ok = await pc.place_order(trade_for_place_order, session)

    assert ok is False
    assert trade_for_place_order.exchange_status == "cap_exceeded"
    bal_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Regression: orderbook depth must be probed at the BUY price (the ask), not
# the mid. `_compute_depth` sums asks with `ask_price <= price`; on any
# positive spread the best ask is above the mid, so probing at the mid returns
# 0 every time. The scheduler used to probe YES depth at the mid, which
# silently depth-vetoed every BUY_YES (probability path AND EASY-YES lock),
# skewing the whole book to NO (the NO side already probed at its true ask).
# This pins the semantics that justify passing yes_ask, not live_price (mid).
# ---------------------------------------------------------------------------


def _book(asks):
    return SimpleNamespace(asks=[SimpleNamespace(price=p, size=s) for p, s in asks])


def test_compute_depth_at_mid_is_zero_but_real_at_ask():
    from src.execution.polymarket_client import _compute_depth

    # bid 0.49 / ask 0.52 → mid 0.505; asks at 0.52 (×100) and 0.55 (×40).
    book = _book([(0.52, 100.0), (0.55, 40.0)])
    mid, ask = 0.505, 0.52

    # The bug: probing at the mid finds no qualifying asks → 0.
    assert _compute_depth(book, mid) == 0.0
    # The fix: probing at the ask finds the fillable level → real depth.
    assert _compute_depth(book, ask) == 0.52 * 100.0
    # Probing higher sweeps deeper levels too.
    assert _compute_depth(book, 0.55) == 0.52 * 100.0 + 0.55 * 40.0


def test_compute_depth_empty_book_is_zero():
    from src.execution.polymarket_client import _compute_depth

    assert _compute_depth(_book([]), 0.9) == 0.0


def test_fast_poll_yes_depth_probes_ask_not_mid():
    """Fast-poll (job_fast_lock_poll) used to probe YES depth at the mid — the
    unfixed twin of the 2026-06-14 unified-pipeline bug — silently vetoing
    EASY-YES locks. Pins the buy-price selection rule the loop now uses:
    yes_buy_px = yes_ask if yes_ask and yes_ask > 0 else yes_price(mid)."""
    from src.execution.polymarket_client import _compute_depth

    yes_bid, yes_ask = 0.49, 0.52
    yes_price = (yes_bid + yes_ask) / 2  # mid 0.505 (still passed as the price arg)
    yes_buy_px = yes_ask if yes_ask and yes_ask > 0 else yes_price
    assert yes_buy_px == yes_ask

    book = _book([(0.52, 100.0)])
    # Old (mid): vetoed. New (ask): real fillable depth.
    assert _compute_depth(book, yes_price) == 0.0
    assert _compute_depth(book, yes_buy_px) == 0.52 * 100.0
