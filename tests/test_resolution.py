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
    """Legacy CLOB-mid fallback path (``market.condition_id IS NULL``).

    With on-chain ``payoutDenominator`` as the primary resolution
    signal (shipped 2026-05-19), the CLOB-mid 0.95/0.05 heuristic is
    only used for pre-caching legacy rows where the bot never
    persisted a ``condition_id``. These tests pin that fallback.
    """

    @pytest.mark.asyncio
    async def test_stale_price_refreshed_to_resolution_threshold(self, mock_session):
        """Stored price is 0.50 (stale), live mid is 0.975 — trade resolves WON."""
        market = MagicMock()
        market.id = "cond_1"
        market.condition_id = None  # force legacy fallback path
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


class TestResolveTradesCatchTwoFallback:
    """Tests for the ``_refresh_market_price returns None`` fallback in
    ``resolve_trades``. Polymarket drops resolved markets from the CLOB
    listing, so ``_refresh_market_price`` returns None and pre-2026-05-17
    the loop's ``continue`` left trades OPEN forever, pinning the
    exposure cap. Fallback marks LOST when the trade never filled
    (``fill_price IS NULL``) after the grace period; leaves OPEN with a
    warning otherwise (operator runs ``admin reconcile-stuck``)."""

    def _stuck_trade(self, *, fill_price, hours_past_end):
        market = MagicMock()
        market.id = "cond_stuck"
        # ``condition_id = None`` forces the legacy CLOB-mid fallback
        # path; the new chain-primary path is covered separately in
        # ``TestResolveTradesOnChain``.
        market.condition_id = None
        # Both the live CLOB lookup AND the stored snapshot return None
        # — this is the catch-22 state pre-2026-05-17 (Polymarket drops
        # the market from CLOB; the local snapshot was never populated
        # because resolution happened before the next scan_markets tick).
        market.current_yes_price = None
        market.end_date = datetime.utcnow() - timedelta(hours=hours_past_end)

        trade = MagicMock()
        trade.id = 1234
        trade.market = market
        trade.market_id = market.id
        trade.direction = TradeDirection.BUY_NO
        trade.stake_usd = 10.0
        trade.entry_price = 0.80
        trade.fill_price = fill_price
        trade.filled_size = None  # exercises _backfill_null_fill
        trade.token_id = "12345"
        trade.status = TradeStatus.OPEN
        return trade

    @pytest.mark.asyncio
    async def test_null_fill_past_grace_balance_zero_marks_lost(
        self, mock_session,
    ):
        """Delayed-never-filled trade with **on-chain balance 0** is marked
        LOST with ``pnl=0`` after the grace window so exposure releases
        without inventing a phantom loss (2026-05-20 fix).

        Pre-fix this path assigned ``pnl=-stake`` on the assumption
        ``fill_price IS NULL`` ⇒ "order never landed", but that conflated
        truly-never-landed orders (bal==0) with reconcile-backfill-gap
        orders (bal>0). Now the balanceOf reading is authoritative.
        """
        trade = self._stuck_trade(fill_price=None, hours_past_end=24)
        scalars = MagicMock()
        scalars.unique.return_value = [trade]
        result = MagicMock()
        result.scalars.return_value = scalars
        mock_session.execute.return_value = result

        # CLOB returns no token IDs → _refresh_market_price returns
        # None via the early-return path. Force the on-chain balance
        # check to return 0 so we're testing the "truly never landed"
        # exit (the legacy fallback path; condition_id is None so
        # branch 4 runs).
        with patch(
            "src.execution.polymarket_client.get_token_ids",
            new=AsyncMock(return_value=None),
        ), patch(
            "src.resolution._build_ctf_readonly",
            new=AsyncMock(return_value=(MagicMock(), "0xfunder")),
        ), patch(
            "src.resolution._query_token_balance",
            new=AsyncMock(return_value=0),
        ):
            resolved = await resolve_trades(mock_session)

        assert len(resolved) == 1
        assert trade.status == TradeStatus.LOST
        assert trade.pnl == pytest.approx(0.0)  # no money moved
        assert trade.exit_price == 0.0

    @pytest.mark.asyncio
    async def test_null_fill_past_grace_balance_positive_preserves_open(
        self, mock_session,
    ):
        """On-chain balance > 0 ⇒ the order DID match on-chain (just
        missing fill data in our DB). Don't mark LOST — preserve OPEN
        status, backfill ``fill_price`` from ``entry_price``, and wait
        for UMA to report. This is the post-2026-05-20 phantom-loss
        guard."""
        trade = self._stuck_trade(fill_price=None, hours_past_end=24)
        scalars = MagicMock()
        scalars.unique.return_value = [trade]
        result = MagicMock()
        result.scalars.return_value = scalars
        mock_session.execute.return_value = result

        with patch(
            "src.execution.polymarket_client.get_token_ids",
            new=AsyncMock(return_value=None),
        ), patch(
            "src.resolution._build_ctf_readonly",
            new=AsyncMock(return_value=(MagicMock(), "0xfunder")),
        ), patch(
            "src.resolution._query_token_balance",
            new=AsyncMock(return_value=10_000_000),  # 10 shares atomic
        ):
            resolved = await resolve_trades(mock_session)

        # Trade stays OPEN, no resolution row appended.
        assert resolved == []
        assert trade.status == TradeStatus.OPEN
        # fill_price was backfilled from entry_price.
        assert trade.fill_price == pytest.approx(0.80)
        # filled_size = stake_usd / entry_price
        assert trade.filled_size == pytest.approx(10.0 / 0.80)

    @pytest.mark.asyncio
    async def test_null_fill_past_grace_balance_unknown_no_chain(
        self, mock_session,
    ):
        """When the chain is unreachable (no CTF, no funder), we cannot
        distinguish "never landed" from "matched but reconcile gap".
        Conservative posture: preserve pre-fix ``pnl=-stake`` so an RPC
        outage doesn't silently mask losses (operator still has the
        warning + `admin reconcile-stuck` to investigate)."""
        trade = self._stuck_trade(fill_price=None, hours_past_end=24)
        scalars = MagicMock()
        scalars.unique.return_value = [trade]
        result = MagicMock()
        result.scalars.return_value = scalars
        mock_session.execute.return_value = result

        with patch(
            "src.execution.polymarket_client.get_token_ids",
            new=AsyncMock(return_value=None),
        ), patch(
            "src.resolution._build_ctf_readonly",
            new=AsyncMock(return_value=(None, None)),
        ):
            resolved = await resolve_trades(mock_session)

        assert len(resolved) == 1
        assert trade.status == TradeStatus.LOST
        assert trade.pnl == pytest.approx(-10.0)
        assert trade.exit_price == 0.0

    @pytest.mark.asyncio
    async def test_null_fill_within_grace_left_open(self, mock_session):
        """Grace window guards against transient CLOB blips — recent
        end_dates leave the trade OPEN so retry on next tick."""
        trade = self._stuck_trade(fill_price=None, hours_past_end=1)
        scalars = MagicMock()
        scalars.unique.return_value = [trade]
        result = MagicMock()
        result.scalars.return_value = scalars
        mock_session.execute.return_value = result

        with patch(
            "src.execution.polymarket_client.get_token_ids",
            new=AsyncMock(return_value=None),
        ):
            resolved = await resolve_trades(mock_session)

        assert resolved == []
        assert trade.status == TradeStatus.OPEN

    @pytest.mark.asyncio
    async def test_populated_fill_past_grace_left_open_with_warning(
        self, mock_session, caplog,
    ):
        """fill_price IS NOT NULL → real on-chain position. Don't silently
        LOST; emit a warning telling the operator to run
        ``admin reconcile-stuck`` for on-chain payout settlement."""
        import logging as _logging
        trade = self._stuck_trade(fill_price=0.80, hours_past_end=48)
        scalars = MagicMock()
        scalars.unique.return_value = [trade]
        result = MagicMock()
        result.scalars.return_value = scalars
        mock_session.execute.return_value = result

        with patch(
            "src.execution.polymarket_client.get_token_ids",
            new=AsyncMock(return_value=None),
        ), caplog.at_level(_logging.WARNING, logger="src.resolution"):
            resolved = await resolve_trades(mock_session)

        assert resolved == []
        assert trade.status == TradeStatus.OPEN
        assert any(
            "reconcile-stuck" in r.message and r.levelname == "WARNING"
            for r in caplog.records
        )


class TestResolveTradesOnChain:
    """Chain-primary resolution path (added 2026-05-19).

    ``resolve_trades`` now calls ``payoutDenominator(conditionId)``
    on the ConditionalTokens contract before falling back to
    CLOB-mid. The chain is authoritative — CLOB-mid had been
    flipping trades WON hours-to-days before UMA actually
    reported on-chain, producing false ``💸 redeem`` nudges that
    reverted with ``result for condition not received yet``.
    """

    def _trade(self, *, direction, fill_price=0.5):
        market = MagicMock()
        market.id = "mk_chain"
        market.condition_id = "0xabc123"
        market.end_date = datetime.now(timezone.utc) - timedelta(hours=2)

        trade = MagicMock()
        trade.id = 7777
        trade.market = market
        trade.market_id = market.id
        trade.direction = direction
        trade.stake_usd = 100.0
        trade.entry_price = 0.40
        trade.fill_price = fill_price
        trade.status = TradeStatus.OPEN
        return trade

    @pytest.fixture
    def mock_ctf(self):
        return MagicMock(name="ctf")

    @pytest.mark.asyncio
    async def test_chain_yes_buy_yes_marks_won(self, mock_session, mock_ctf):
        """On-chain says YES won + we hold YES → WON, full payout."""
        trade = self._trade(direction=TradeDirection.BUY_YES)
        scalars = MagicMock()
        scalars.unique.return_value = [trade]
        result = MagicMock()
        result.scalars.return_value = scalars
        mock_session.execute.return_value = result

        with patch(
            "src.resolution._build_ctf_readonly",
            new=AsyncMock(return_value=(mock_ctf, "0xfunder")),
        ), patch(
            "src.resolution._query_payout_outcome",
            new=AsyncMock(return_value=True),  # YES won
        ):
            resolved = await resolve_trades(mock_session)

        assert len(resolved) == 1
        assert trade.status == TradeStatus.WON
        assert trade.exit_price == 1.0
        assert trade.pnl == pytest.approx(100.0 * (1 / 0.40 - 1))

    @pytest.mark.asyncio
    async def test_chain_yes_buy_no_marks_lost(self, mock_session, mock_ctf):
        """On-chain says YES won + we hold NO → LOST."""
        trade = self._trade(direction=TradeDirection.BUY_NO)
        scalars = MagicMock()
        scalars.unique.return_value = [trade]
        result = MagicMock()
        result.scalars.return_value = scalars
        mock_session.execute.return_value = result

        with patch(
            "src.resolution._build_ctf_readonly",
            new=AsyncMock(return_value=(mock_ctf, "0xfunder")),
        ), patch(
            "src.resolution._query_payout_outcome",
            new=AsyncMock(return_value=True),  # YES won
        ):
            resolved = await resolve_trades(mock_session)

        assert len(resolved) == 1
        assert trade.status == TradeStatus.LOST
        assert trade.pnl == pytest.approx(-100.0)

    @pytest.mark.asyncio
    async def test_chain_no_buy_no_marks_won(self, mock_session, mock_ctf):
        """On-chain says NO won + we hold NO → WON."""
        trade = self._trade(direction=TradeDirection.BUY_NO)
        scalars = MagicMock()
        scalars.unique.return_value = [trade]
        result = MagicMock()
        result.scalars.return_value = scalars
        mock_session.execute.return_value = result

        with patch(
            "src.resolution._build_ctf_readonly",
            new=AsyncMock(return_value=(mock_ctf, "0xfunder")),
        ), patch(
            "src.resolution._query_payout_outcome",
            new=AsyncMock(return_value=False),  # NO won
        ):
            resolved = await resolve_trades(mock_session)

        assert len(resolved) == 1
        assert trade.status == TradeStatus.WON
        assert trade.exit_price == 1.0

    @pytest.mark.asyncio
    async def test_chain_unresolved_within_grace_left_open(
        self, mock_session, mock_ctf,
    ):
        """``payoutDenominator == 0`` + within grace → wait, leave OPEN.

        Prevents the false-WON failure mode where CLOB-mid would
        trigger a redeem nudge that reverts on-chain.
        """
        trade = self._trade(direction=TradeDirection.BUY_NO)
        # 2h past end_date < RESOLVE_NO_PRICE_GRACE_HOURS (4h default)
        scalars = MagicMock()
        scalars.unique.return_value = [trade]
        result = MagicMock()
        result.scalars.return_value = scalars
        mock_session.execute.return_value = result

        with patch(
            "src.resolution._build_ctf_readonly",
            new=AsyncMock(return_value=(mock_ctf, "0xfunder")),
        ), patch(
            "src.resolution._query_payout_outcome",
            new=AsyncMock(return_value=None),  # UMA hasn't reported
        ):
            resolved = await resolve_trades(mock_session)

        assert resolved == []
        assert trade.status == TradeStatus.OPEN

    @pytest.mark.asyncio
    async def test_chain_unresolved_past_grace_null_fill_balance_zero(
        self, mock_session, mock_ctf,
    ):
        """``payoutDenominator == 0`` + past grace + null fill + bal==0
        → LOST with ``pnl=0`` (2026-05-20 fix).

        Order truly never landed on-chain (balanceOf confirms zero
        position); release the exposure but don't invent a loss. Pre-fix
        this path assigned ``pnl=-stake``, contributing to the phantom
        $228 of "losses" on 2026-05-20.
        """
        trade = self._trade(direction=TradeDirection.BUY_NO, fill_price=None)
        trade.token_id = "12345"
        trade.market.end_date = datetime.now(timezone.utc) - timedelta(hours=24)
        scalars = MagicMock()
        scalars.unique.return_value = [trade]
        result = MagicMock()
        result.scalars.return_value = scalars
        mock_session.execute.return_value = result

        with patch(
            "src.resolution._build_ctf_readonly",
            new=AsyncMock(return_value=(mock_ctf, "0xfunder")),
        ), patch(
            "src.resolution._query_payout_outcome",
            new=AsyncMock(return_value=None),
        ), patch(
            "src.resolution._query_token_balance",
            new=AsyncMock(return_value=0),
        ):
            resolved = await resolve_trades(mock_session)

        assert len(resolved) == 1
        assert trade.status == TradeStatus.LOST
        assert trade.pnl == pytest.approx(0.0)  # no money moved

    @pytest.mark.asyncio
    async def test_chain_unresolved_past_grace_null_fill_balance_positive(
        self, mock_session, mock_ctf,
    ):
        """``payoutDenominator == 0`` + past grace + null fill + bal>0
        → preserve OPEN, backfill fill_price (2026-05-20 phantom-loss
        guard).

        The order DID match on-chain (balanceOf > 0) — ``fill_price=None``
        is just a ``job_reconcile_orders`` backfill gap, not proof of
        non-existence. Pre-fix the bot wrote off these healthy positions
        as full-stake losses, producing the $228 phantom-loss event on
        2026-05-20.
        """
        trade = self._trade(direction=TradeDirection.BUY_NO, fill_price=None)
        trade.filled_size = None
        trade.token_id = "12345"
        trade.market.end_date = datetime.now(timezone.utc) - timedelta(hours=24)
        scalars = MagicMock()
        scalars.unique.return_value = [trade]
        result = MagicMock()
        result.scalars.return_value = scalars
        mock_session.execute.return_value = result

        with patch(
            "src.resolution._build_ctf_readonly",
            new=AsyncMock(return_value=(mock_ctf, "0xfunder")),
        ), patch(
            "src.resolution._query_payout_outcome",
            new=AsyncMock(return_value=None),
        ), patch(
            "src.resolution._query_token_balance",
            new=AsyncMock(return_value=10_000_000),  # 10 shares atomic
        ):
            resolved = await resolve_trades(mock_session)

        # Trade stays OPEN, no resolution row appended.
        assert resolved == []
        assert trade.status == TradeStatus.OPEN
        # fill_price was backfilled from entry_price (set in _trade()).
        assert trade.fill_price is not None
        assert trade.filled_size is not None
        assert trade.filled_size > 0

    @pytest.mark.asyncio
    async def test_chain_unresolved_past_grace_filled_warns(
        self, mock_session, mock_ctf, caplog,
    ):
        """``payoutDenominator == 0`` + past grace + filled position →
        WARN + leave OPEN. Real on-chain position needs operator
        intervention (``admin reconcile-stuck``), not silent LOST."""
        import logging as _logging
        trade = self._trade(direction=TradeDirection.BUY_NO, fill_price=0.20)
        trade.market.end_date = datetime.now(timezone.utc) - timedelta(hours=48)
        scalars = MagicMock()
        scalars.unique.return_value = [trade]
        result = MagicMock()
        result.scalars.return_value = scalars
        mock_session.execute.return_value = result

        with patch(
            "src.resolution._build_ctf_readonly",
            new=AsyncMock(return_value=(mock_ctf, "0xfunder")),
        ), patch(
            "src.resolution._query_payout_outcome",
            new=AsyncMock(return_value=None),
        ), caplog.at_level(_logging.WARNING, logger="src.resolution"):
            resolved = await resolve_trades(mock_session)

        assert resolved == []
        assert trade.status == TradeStatus.OPEN
        assert any(
            "reconcile-stuck" in r.message and r.levelname == "WARNING"
            for r in caplog.records
        )
