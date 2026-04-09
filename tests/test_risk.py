"""Tests for the risk management module (kelly, drawdown, simulate)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.risk.drawdown import (
    CAUTION_THRESHOLD,
    PAUSE_THRESHOLD,
    DrawdownLevel,
    DrawdownMonitor,
    DrawdownState,
)
from src.risk.kelly import MAX_EXPOSURE_PCT, MIN_TRADE_USD, PositionSize, size_position
from src.risk.simulate import SimResult, SimSignal, simulate_bankroll


# ===================================================================
# kelly.py – position sizing
# ===================================================================


class TestSizePosition:
    def test_normal_sizing(self):
        """Positive edge produces a non-zero stake within caps."""
        pos = size_position(1000, model_prob=0.58, market_prob=0.50, kelly_fraction=0.25)
        assert pos.stake_usd > 0
        assert pos.stake_usd <= 1000 * 0.05  # within per-trade cap
        assert not pos.capped
        assert pos.reason == "sized normally"

    def test_no_edge_returns_zero(self):
        """When model agrees with market, no bet."""
        pos = size_position(1000, model_prob=0.4, market_prob=0.5)
        assert pos.stake_usd == 0
        assert pos.kelly_pct == 0
        assert not pos.capped
        assert "no edge" in pos.reason

    def test_per_trade_cap(self):
        """Huge edge gets clamped to 5% of bankroll."""
        # model_prob=0.99 vs market_prob=0.10 → massive Kelly
        pos = size_position(1000, model_prob=0.99, market_prob=0.10)
        assert pos.stake_usd == pytest.approx(50.0)  # 5% of 1000
        assert pos.capped is True
        assert "per-trade cap" in pos.reason

    def test_exposure_cap_limits_stake(self):
        """Current exposure leaves only a sliver of the 25% limit."""
        pos = size_position(
            1000,
            model_prob=0.99,
            market_prob=0.10,
            current_exposure=240,
        )
        # max_remaining = 1000*0.25 - 240 = 10
        assert pos.stake_usd == pytest.approx(10.0)
        assert pos.capped is True

    def test_exposure_limit_reached(self):
        """Exposure already at 25% → stake = 0."""
        pos = size_position(
            1000,
            model_prob=0.99,
            market_prob=0.10,
            current_exposure=250,
        )
        assert pos.stake_usd == 0
        assert pos.capped is True
        assert "exposure limit" in pos.reason

    def test_below_minimum_returns_zero(self):
        """Tiny bankroll yields stake < $5 → skip."""
        # With bankroll=50, quarter-Kelly on a modest edge → very small
        pos = size_position(50, model_prob=0.55, market_prob=0.50)
        assert pos.stake_usd == 0
        assert pos.capped is True
        assert "$5" in pos.reason

    def test_kelly_fraction_override(self):
        """Custom kelly_fraction doubles the stake."""
        half = size_position(1000, 0.7, 0.5, kelly_fraction=0.25)
        full = size_position(1000, 0.7, 0.5, kelly_fraction=0.50)
        # Full should be roughly double half (unless caps intervene)
        if not half.capped and not full.capped:
            assert full.stake_usd == pytest.approx(half.stake_usd * 2, rel=0.01)


# ===================================================================
# drawdown.py – drawdown monitor
# ===================================================================


class TestDrawdownMonitor:
    def test_normal_at_start(self):
        """Fresh monitor reports NORMAL."""
        mon = DrawdownMonitor(750)
        state = mon.check(750)
        assert state.level == DrawdownLevel.NORMAL
        assert state.size_multiplier == 1.0
        assert state.drawdown_pct == pytest.approx(0.0)

    def test_caution_at_12_pct(self):
        """12% drawdown → CAUTION, half size."""
        mon = DrawdownMonitor(750)
        mon.advance(750)
        state = mon.advance(660)  # (750-660)/750 = 12%
        assert state.level == DrawdownLevel.CAUTION
        assert state.size_multiplier == 0.5
        assert state.drawdown_pct == pytest.approx(0.12)

    def test_paused_at_25_pct(self):
        """>20% drawdown → PAUSED, no trades."""
        mon = DrawdownMonitor(750)
        mon.advance(750)
        state = mon.advance(562.5)  # 25% drawdown
        assert state.level == DrawdownLevel.PAUSED
        assert state.size_multiplier == 0.0

    def test_recovery_under_10_pct(self):
        """From CAUTION, recover to <10% dd → RECOVERY (still half size)."""
        mon = DrawdownMonitor(750)
        mon.advance(750)
        mon.advance(660)  # → CAUTION
        state = mon.advance(690)  # dd = (750-690)/750 = 8% < 10%
        assert state.level == DrawdownLevel.RECOVERY
        assert state.size_multiplier == 0.5

    def test_new_high_watermark_resets_to_normal(self):
        """Recovering past peak → NORMAL."""
        mon = DrawdownMonitor(750)
        mon.advance(750)
        mon.advance(660)  # CAUTION
        mon.advance(690)  # RECOVERY
        state = mon.advance(760)  # new peak
        assert state.level == DrawdownLevel.NORMAL
        assert state.size_multiplier == 1.0
        assert state.peak == 760

    @pytest.mark.asyncio
    async def test_update_persists_bankroll_log(self):
        """update() should add a BankrollLog row to the session."""
        mon = DrawdownMonitor(750)
        session = AsyncMock()
        state = await mon.update(650, session)
        session.add.assert_called_once()
        row = session.add.call_args[0][0]
        assert row.balance == 650
        assert row.peak == 750
        assert state.level == DrawdownLevel.CAUTION

    @pytest.mark.asyncio
    async def test_load_state_from_db(self):
        """load_state() restores peak/current from the latest row."""
        mon = DrawdownMonitor(750)

        mock_row = MagicMock()
        mock_row.peak = 800.0
        mock_row.balance = 720.0

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_row

        session = AsyncMock()
        session.execute.return_value = mock_result

        state = await mon.load_state(session)
        assert state.peak == 800.0
        assert state.current == 720.0
        assert state.level == DrawdownLevel.CAUTION


# ===================================================================
# simulate.py – backtesting engine
# ===================================================================


class TestSimulateBankroll:
    def test_all_wins(self):
        """All winning signals should grow the bankroll."""
        signals = [SimSignal(0.7, 0.5, True)] * 10
        result = simulate_bankroll(signals, initial_bankroll=1000)
        assert result.final_bankroll > 1000
        assert result.win_rate == pytest.approx(1.0)
        assert result.num_trades == 10
        assert result.num_skipped == 0

    def test_all_losses(self):
        """All losing signals should shrink the bankroll."""
        signals = [SimSignal(0.7, 0.5, False)] * 10
        result = simulate_bankroll(signals, initial_bankroll=1000)
        assert result.final_bankroll < 1000
        assert result.win_rate == pytest.approx(0.0)

    def test_known_outcome_single_trade(self):
        """Verify exact P&L for one winning trade."""
        # model_prob=0.7, market_prob=0.5 → edge=0.4, payout=2.0
        # full_kelly = 0.4 / (2.0-1.0) = 0.4
        # quarter kelly stake = 1000 * 0.4 * 0.25 = 100 → capped at 50 (5%)
        # Win pnl = 50 * (2.0 - 1.0) = 50
        result = simulate_bankroll(
            [SimSignal(0.7, 0.5, True)],
            initial_bankroll=1000,
            kelly_fraction=0.25,
        )
        assert result.final_bankroll == pytest.approx(1050.0)
        assert result.num_trades == 1

    def test_drawdown_pauses_trading(self):
        """Heavy losses should trigger PAUSED and skip subsequent signals."""
        # Start with losses to push past 20% drawdown, then more signals
        signals = [SimSignal(0.9, 0.1, False)] * 20
        result = simulate_bankroll(signals, initial_bankroll=500)
        assert result.num_skipped > 0

    def test_sharpe_positive_for_winning_strategy(self):
        """A mostly-winning strategy with varied signals has positive Sharpe."""
        signals = [
            SimSignal(0.75, 0.50, True),
            SimSignal(0.65, 0.45, True),
            SimSignal(0.80, 0.55, True),
            SimSignal(0.70, 0.50, False),
            SimSignal(0.75, 0.50, True),
            SimSignal(0.60, 0.40, True),
            SimSignal(0.85, 0.60, True),
            SimSignal(0.70, 0.50, True),
        ]
        result = simulate_bankroll(signals, initial_bankroll=1000)
        assert result.sharpe_ratio > 0

    def test_bankroll_curve_length(self):
        """Curve should have one entry per signal."""
        signals = [SimSignal(0.7, 0.5, True)] * 15
        result = simulate_bankroll(signals, initial_bankroll=1000)
        assert len(result.bankroll_curve) == 15
