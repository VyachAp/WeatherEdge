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


class TestGetCurrentBankroll:
    @pytest.mark.asyncio
    async def test_wallet_plus_unredeemed_when_wallet_present(self, mock_session):
        unredeemed_result = MagicMock()
        unredeemed_result.scalar_one.return_value = 200.0
        mock_session.execute.return_value = unredeemed_result

        with patch(
            "src.execution.polymarket_client.get_wallet_usdc_balance",
            return_value=1000.0,
        ):
            bankroll = await get_current_bankroll(mock_session)

        assert bankroll == 1200.0

    @pytest.mark.asyncio
    async def test_excludes_redeemed_trades_via_query(self, mock_session):
        """When all WONs are redeemed, the SUM is 0, so bankroll == wallet."""
        unredeemed_result = MagicMock()
        unredeemed_result.scalar_one.return_value = 0.0
        mock_session.execute.return_value = unredeemed_result

        with patch(
            "src.execution.polymarket_client.get_wallet_usdc_balance",
            return_value=1200.0,
        ):
            bankroll = await get_current_bankroll(mock_session)

        # Exactly the wallet — no double-count after redeem.
        assert bankroll == 1200.0

    @pytest.mark.asyncio
    async def test_no_wallet_falls_back_to_bankroll_log(self, mock_session):
        unredeemed_result = MagicMock()
        unredeemed_result.scalar_one.return_value = 50.0

        log_result = MagicMock()
        log_result.scalar_one_or_none.return_value = 950.0

        # Two execute calls: unredeemed sum, then bankroll log.
        mock_session.execute.side_effect = [unredeemed_result, log_result]

        with patch(
            "src.execution.polymarket_client.get_wallet_usdc_balance",
            return_value=None,
        ):
            bankroll = await get_current_bankroll(mock_session)

        assert bankroll == 1000.0

    @pytest.mark.asyncio
    async def test_no_wallet_no_log_uses_initial(self, mock_session):
        unredeemed_result = MagicMock()
        unredeemed_result.scalar_one.return_value = 0.0

        log_result = MagicMock()
        log_result.scalar_one_or_none.return_value = None

        mock_session.execute.side_effect = [unredeemed_result, log_result]

        with patch(
            "src.execution.polymarket_client.get_wallet_usdc_balance",
            return_value=None,
        ), patch("src.resolution.settings") as mock_settings:
            mock_settings.INITIAL_BANKROLL = 500.0
            bankroll = await get_current_bankroll(mock_session)

        assert bankroll == 500.0


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
