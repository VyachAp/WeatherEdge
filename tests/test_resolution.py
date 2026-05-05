"""Tests for resolution.get_current_bankroll, get_unredeemed_won_payout,
and resolve_trades price refresh."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.db.models import Market, Trade, TradeDirection, TradeStatus
from src.resolution import (
    _refresh_market_price,
    get_current_bankroll,
    get_open_trade_value,
    get_unredeemed_won_payout,
    resolve_trades,
)


@pytest.fixture
def mock_session():
    s = AsyncMock()
    return s


class TestGetUnredeemedWonPayout:
    @pytest.mark.asyncio
    async def test_returns_zero_with_no_won_trades(self, mock_session):
        result = MagicMock()
        result.scalar_one.return_value = 0.0
        mock_session.execute.return_value = result

        total = await get_unredeemed_won_payout(mock_session)
        assert total == 0.0

    @pytest.mark.asyncio
    async def test_returns_sum_of_stake_plus_pnl(self, mock_session):
        result = MagicMock()
        # 100 stake @ 0.50 entry → pnl = 100, stake+pnl = 200
        result.scalar_one.return_value = 200.0
        mock_session.execute.return_value = result

        total = await get_unredeemed_won_payout(mock_session)
        assert total == 200.0


def _bankroll_results(
    *,
    unredeemed: float = 0.0,
    open_rows: list[tuple] | None = None,
    log_balance: float | None = None,
):
    """Build the MagicMock result objects for a get_current_bankroll call.

    Order of execute() calls inside get_current_bankroll:
      1. get_unredeemed_won_payout — scalar_one
      2. get_open_trade_value — result.all() returns rows
      3. (no-wallet branch only) BankrollLog query — scalar_one_or_none

    Always returns 3 results; wallet-path tests slice ``[:2]`` to drop the
    unused BankrollLog result.
    """
    unredeemed_result = MagicMock()
    unredeemed_result.scalar_one.return_value = unredeemed

    open_result = MagicMock()
    open_result.all.return_value = open_rows or []

    log_result = MagicMock()
    log_result.scalar_one_or_none.return_value = log_balance

    return [unredeemed_result, open_result, log_result]


class TestGetCurrentBankroll:
    @pytest.mark.asyncio
    async def test_wallet_plus_unredeemed_when_wallet_present(self, mock_session):
        # Wallet path: only the unredeemed and open-value queries fire.
        mock_session.execute.side_effect = _bankroll_results(
            unredeemed=200.0,
            open_rows=[],
        )[:2]

        with patch(
            "src.execution.polymarket_client.get_wallet_usdc_balance",
            return_value=1000.0,
        ):
            bankroll = await get_current_bankroll(mock_session)

        assert bankroll == 1200.0

    @pytest.mark.asyncio
    async def test_excludes_redeemed_trades_via_query(self, mock_session):
        """When all WONs are redeemed and no OPEN trades, bankroll == wallet."""
        mock_session.execute.side_effect = _bankroll_results(
            unredeemed=0.0,
            open_rows=[],
        )[:2]

        with patch(
            "src.execution.polymarket_client.get_wallet_usdc_balance",
            return_value=1200.0,
        ):
            bankroll = await get_current_bankroll(mock_session)

        # Exactly the wallet — no double-count after redeem.
        assert bankroll == 1200.0

    @pytest.mark.asyncio
    async def test_open_trade_value_added_to_wallet(self, mock_session):
        """Regression: BUY_YES @ entry 0.50 with 100 stake and current 0.60
        is worth 100/0.50 × 0.60 = 120 — must be added to wallet equity."""
        mock_session.execute.side_effect = _bankroll_results(
            unredeemed=0.0,
            open_rows=[(TradeDirection.BUY_YES, 100.0, 0.50, 0.60)],
        )[:2]

        with patch(
            "src.execution.polymarket_client.get_wallet_usdc_balance",
            return_value=900.0,
        ):
            bankroll = await get_current_bankroll(mock_session)

        assert bankroll == pytest.approx(1020.0)

    @pytest.mark.asyncio
    async def test_no_wallet_falls_back_to_bankroll_log(self, mock_session):
        mock_session.execute.side_effect = _bankroll_results(
            unredeemed=50.0,
            open_rows=[],
            log_balance=950.0,
        )

        with patch(
            "src.execution.polymarket_client.get_wallet_usdc_balance",
            return_value=None,
        ):
            bankroll = await get_current_bankroll(mock_session)

        assert bankroll == 1000.0

    @pytest.mark.asyncio
    async def test_no_wallet_no_log_uses_initial(self, mock_session):
        mock_session.execute.side_effect = _bankroll_results(
            unredeemed=0.0,
            open_rows=[],
            log_balance=None,
        )

        with patch(
            "src.execution.polymarket_client.get_wallet_usdc_balance",
            return_value=None,
        ), patch("src.resolution.settings") as mock_settings:
            mock_settings.INITIAL_BANKROLL = 500.0
            bankroll = await get_current_bankroll(mock_session)

        assert bankroll == 500.0


class TestGetOpenTradeValue:
    @pytest.mark.asyncio
    async def test_no_open_trades_returns_zero(self, mock_session):
        result = MagicMock()
        result.all.return_value = []
        mock_session.execute.return_value = result

        assert await get_open_trade_value(mock_session) == 0.0

    @pytest.mark.asyncio
    async def test_buy_yes_marked_at_current_price(self, mock_session):
        # 100 stake, 0.50 entry → 200 shares. Current YES 0.60 → 200 × 0.60 = 120.
        result = MagicMock()
        result.all.return_value = [
            (TradeDirection.BUY_YES, 100.0, 0.50, 0.60),
        ]
        mock_session.execute.return_value = result

        assert await get_open_trade_value(mock_session) == pytest.approx(120.0)

    @pytest.mark.asyncio
    async def test_buy_no_uses_one_minus_yes_price(self, mock_session):
        # 80 stake, 0.40 entry → 200 shares. Current YES 0.30 → NO worth 0.70 → 140.
        result = MagicMock()
        result.all.return_value = [
            (TradeDirection.BUY_NO, 80.0, 0.40, 0.30),
        ]
        mock_session.execute.return_value = result

        assert await get_open_trade_value(mock_session) == pytest.approx(140.0)

    @pytest.mark.asyncio
    async def test_missing_yes_price_falls_back_to_cost_basis(self, mock_session):
        # Stale market with no cached price → count the dollar at par.
        result = MagicMock()
        result.all.return_value = [
            (TradeDirection.BUY_YES, 50.0, 0.45, None),
        ]
        mock_session.execute.return_value = result

        assert await get_open_trade_value(mock_session) == 50.0

    @pytest.mark.asyncio
    async def test_missing_entry_price_falls_back_to_cost_basis(self, mock_session):
        result = MagicMock()
        result.all.return_value = [
            (TradeDirection.BUY_YES, 25.0, None, 0.60),
        ]
        mock_session.execute.return_value = result

        assert await get_open_trade_value(mock_session) == 25.0

    @pytest.mark.asyncio
    async def test_zero_or_missing_stake_skipped(self, mock_session):
        result = MagicMock()
        result.all.return_value = [
            (TradeDirection.BUY_YES, 0.0, 0.50, 0.60),
            (TradeDirection.BUY_YES, None, 0.50, 0.60),
            (TradeDirection.BUY_YES, 10.0, 0.50, 0.40),  # 10/0.5 × 0.4 = 8
        ]
        mock_session.execute.return_value = result

        assert await get_open_trade_value(mock_session) == pytest.approx(8.0)

    @pytest.mark.asyncio
    async def test_clipped_when_cached_price_outside_unit_interval(self, mock_session):
        # Defensive: a stale 1.05 cached YES price shouldn't manufacture > stake equity.
        result = MagicMock()
        result.all.return_value = [
            (TradeDirection.BUY_YES, 100.0, 0.50, 1.05),
        ]
        mock_session.execute.return_value = result

        # 100/0.50 × clip(1.05) = 200 × 1.0 = 200
        assert await get_open_trade_value(mock_session) == pytest.approx(200.0)


class TestRefreshMarketPrice:
    @pytest.mark.asyncio
    async def test_writes_live_mid_to_market(self):
        market = MagicMock()
        market.id = "0xabc"
        market.current_yes_price = 0.50

        with patch(
            "src.execution.polymarket_client.get_token_ids",
            new=AsyncMock(return_value=("yes_token", "no_token")),
        ), patch(
            "src.execution.polymarket_client.get_best_bid_ask",
            return_value=(0.97, 0.98),
        ):
            mid = await _refresh_market_price(market)

        assert mid == pytest.approx(0.975)
        # Function returns the refreshed mid but does not mutate the ORM row.
        assert market.current_yes_price == 0.50

    @pytest.mark.asyncio
    async def test_falls_back_to_stored_price_on_failure(self):
        market = MagicMock()
        market.id = "0xabc"
        market.current_yes_price = 0.42

        with patch(
            "src.execution.polymarket_client.get_token_ids",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ):
            mid = await _refresh_market_price(market)

        assert mid == 0.42
        assert market.current_yes_price == 0.42

    @pytest.mark.asyncio
    async def test_no_token_ids_returns_stored(self):
        market = MagicMock()
        market.id = "0xabc"
        market.current_yes_price = 0.30

        with patch(
            "src.execution.polymarket_client.get_token_ids",
            new=AsyncMock(return_value=None),
        ):
            mid = await _refresh_market_price(market)

        assert mid == 0.30


class TestResolveTradesPriceRefresh:
    @pytest.mark.asyncio
    async def test_stale_price_refreshed_to_resolution_threshold(self, mock_session):
        """Stored price is 0.50 (stale), live mid is 0.975 — trade resolves WON."""
        market = MagicMock()
        market.id = "cond_1"
        market.current_yes_price = 0.50
        market.end_date = datetime.utcnow() - timedelta(hours=1)

        trade = MagicMock()
        trade.market = market
        trade.market_id = market.id
        trade.direction = TradeDirection.BUY_YES
        trade.stake_usd = 100.0
        trade.entry_price = 0.50
        trade.status = TradeStatus.OPEN

        scalars = MagicMock()
        scalars.unique.return_value = [trade]
        result = MagicMock()
        result.scalars.return_value = scalars
        mock_session.execute.return_value = result

        with patch(
            "src.execution.polymarket_client.get_token_ids",
            new=AsyncMock(return_value=("yes", "no")),
        ), patch(
            "src.execution.polymarket_client.get_best_bid_ask",
            return_value=(0.97, 0.98),
        ):
            resolved = await resolve_trades(mock_session)

        assert len(resolved) == 1
        assert trade.status == TradeStatus.WON
        assert trade.exit_price == 1.0
        assert trade.pnl == pytest.approx(100.0)  # 100 * (1/0.5 - 1)
