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
