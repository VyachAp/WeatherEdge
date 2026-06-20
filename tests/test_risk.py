"""Tests for the risk management module (kelly, drawdown, simulate)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import settings
from src.execution.binary_market import (
    near_lock_conviction_eligible,
    near_peak_floor_eligible,
)
from src.risk.drawdown import (
    CAUTION_THRESHOLD,
    PAUSE_THRESHOLD,
    DrawdownLevel,
    DrawdownMonitor,
    DrawdownState,
)
from src.risk.kelly import (
    MAX_EXPOSURE_PCT,
    MIN_TRADE_USD,
    PositionSize,
    size_locked_position,
    size_position,
)
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

    def test_near_lock_blocked_by_default_prob_cap(self, monkeypatch):
        """A certain bet bought at 0.95 sizes to $0 under the global 0.90 cap —
        the near-lock band the 0.80 floor admits but Kelly forbids."""
        monkeypatch.setattr(settings, "KELLY_PROB_CAP", 0.90)
        pos = size_position(1000, model_prob=1.0, market_prob=0.95)
        assert pos.stake_usd == 0
        assert "no edge" in pos.reason

    def test_prob_cap_override_unlocks_near_lock_bet(self, monkeypatch):
        """Passing a higher prob_cap (conviction sizing) turns the same bet into
        a positive stake — the buy at 0.95 now has positive Kelly edge."""
        monkeypatch.setattr(settings, "KELLY_PROB_CAP", 0.90)
        pos = size_position(1000, model_prob=1.0, market_prob=0.95, prob_cap=0.99)
        assert pos.stake_usd > 0
        # Still bounded by the per-trade cap (5% of bankroll under the test
        # default), proving the cascade still applies.
        assert pos.stake_usd <= 1000 * settings.MAX_POSITION_PCT + 0.01

    def test_prob_cap_none_reproduces_global(self, monkeypatch):
        """prob_cap=None == omitted == settings.KELLY_PROB_CAP."""
        monkeypatch.setattr(settings, "KELLY_PROB_CAP", 0.90)
        a = size_position(1000, model_prob=1.0, market_prob=0.95)
        b = size_position(1000, model_prob=1.0, market_prob=0.95, prob_cap=None)
        assert a.stake_usd == b.stake_usd == 0

    def test_per_trade_cap(self):
        """Huge edge gets clamped to 5% of bankroll."""
        # model_prob=0.99 vs market_prob=0.10 → massive Kelly
        pos = size_position(1000, model_prob=0.99, market_prob=0.10)
        assert pos.stake_usd == pytest.approx(50.0)  # 5% of 1000
        assert pos.capped is True
        assert "per-trade cap" in pos.reason

    def test_exposure_cap_limits_stake(self):
        """Current exposure leaves only a sliver of the 25% limit.

        Uses bankroll=2000 so the pct cap (500) is above the
        ``MAX_EXPOSURE_USD_FLOOR=300`` and stays the binding constraint.
        """
        pos = size_position(
            2000,
            model_prob=0.99,
            market_prob=0.10,
            current_exposure=490,
        )
        # effective_cap = max(2000*0.25=500, 300) = 500
        # max_remaining = 500 - 490 = 10
        assert pos.stake_usd == pytest.approx(10.0)
        assert pos.capped is True

    def test_exposure_limit_reached(self):
        """Exposure already at 25% → stake = 0 (bankroll above floor zone)."""
        pos = size_position(
            2000,
            model_prob=0.99,
            market_prob=0.10,
            current_exposure=500,
        )
        assert pos.stake_usd == 0
        assert pos.capped is True
        assert "exposure limit" in pos.reason

    def test_usd_floor_binds_at_small_bankroll(self):
        """At small bankroll, the absolute USD floor binds instead of the
        percent cap so the bot keeps trading. With bankroll=$441 the
        pct cap is $110 but the floor lifts effective cap to $300 —
        $134.77 of stuck exposure no longer pins the cap immediately.
        Regression for the 2026-05-17 silencing incident."""
        # Bankroll * 0.25 = 110.25; floor = 300; effective = 300.
        # With $134 exposure (where the pre-floor cap would have already
        # been blown by 22%), we still have $166 budget.
        pos = size_position(
            441,
            model_prob=0.99,
            market_prob=0.50,
            current_exposure=134,
        )
        assert pos.stake_usd > 0, "Floor should leave headroom for a new trade"
        # Per-trade cap = 441 * 0.05 = $22.05, so the sized stake should be
        # capped there (not at exposure-cap remaining of ~$166).
        assert pos.stake_usd == pytest.approx(22.05, abs=0.10)

    def test_usd_floor_inactive_at_large_bankroll(self):
        """Above the crossover (~$1200 at defaults), the pct cap binds
        and the floor has no effect."""
        pos = size_position(
            5000,
            model_prob=0.99,
            market_prob=0.10,
            current_exposure=1240,
        )
        # effective_cap = max(5000*0.25=1250, 300) = 1250
        # max_remaining = 1250 - 1240 = 10
        assert pos.stake_usd == pytest.approx(10.0)
        assert pos.capped is True

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

    def test_max_position_usd_cap(self):
        """Explicit USD cap limits the position."""
        pos = size_position(
            10_000,
            model_prob=0.99,
            market_prob=0.10,
            max_position_usd=100.0,
        )
        assert pos.stake_usd <= 100.0
        assert pos.capped is True
        assert "max position" in pos.reason

    def test_orderbook_depth_cap(self):
        """Position capped at 20% of visible orderbook depth."""
        pos = size_position(
            10_000,
            model_prob=0.99,
            market_prob=0.10,
            orderbook_depth=200.0,
        )
        # 20% of 200 = 40
        assert pos.stake_usd <= 40.0
        assert pos.capped is True
        assert "depth cap" in pos.reason

    def test_no_depth_cap_when_none(self):
        """No depth cap applied when orderbook_depth is None."""
        pos = size_position(
            1000,
            model_prob=0.7,
            market_prob=0.5,
            orderbook_depth=None,
        )
        # Should behave same as before — no depth cap in reason
        assert "depth" not in pos.reason

    def test_depth_cap_not_applied_when_depth_is_large(self):
        """Large depth doesn't constrain the position."""
        pos = size_position(
            1000,
            model_prob=0.7,
            market_prob=0.5,
            orderbook_depth=100_000.0,
        )
        assert "depth" not in pos.reason

    # -- near-peak floor-up (floor_to_usd) -----------------------------------

    def test_floor_to_usd_lifts_sub_min(self):
        """Raw Kelly below $5 is floored up to floor_to_usd when caps allow."""
        # bankroll 316, tiny edge → raw Kelly ≈ $3.16 (< $5), but per-trade cap
        # = $15.80 and exposure room = $300 both leave headroom for a $5 floor.
        dropped = size_position(316, model_prob=0.52, market_prob=0.50)
        assert dropped.stake_usd == 0  # sanity: drops without the floor
        floored = size_position(
            316, model_prob=0.52, market_prob=0.50, floor_to_usd=5.0
        )
        assert floored.stake_usd == pytest.approx(5.0)
        assert floored.reason.startswith("floored")

    def test_floor_to_usd_none_preserves_drop(self):
        """floor_to_usd=None keeps the original sub-min drop (regression)."""
        pos = size_position(
            316, model_prob=0.52, market_prob=0.50, floor_to_usd=None
        )
        assert pos.stake_usd == 0
        assert "$5" in pos.reason

    def test_floor_clamped_by_depth(self):
        """A thin book clamps the floor below $5 → trade still drops."""
        # depth $20 → depth cap = $4 < $5, so an $8 floor can't be honored.
        pos = size_position(
            316,
            model_prob=0.52,
            market_prob=0.50,
            orderbook_depth=20.0,
            floor_to_usd=8.0,
        )
        assert pos.stake_usd == 0
        assert "$5" in pos.reason

    def test_floor_clamped_by_exposure(self):
        """Exposure-remaining below $5 clamps the floor → trade drops."""
        # exposure cap = $300 (USD floor binds at $316), 297 used → $3 room.
        pos = size_position(
            316,
            model_prob=0.60,
            market_prob=0.50,
            current_exposure=297.0,
            floor_to_usd=5.0,
        )
        assert pos.stake_usd == 0
        assert "$5" in pos.reason

    def test_floor_ignored_when_kelly_above_min(self):
        """floor_to_usd is a no-op when Kelly already sizes ≥ $5."""
        pos = size_position(
            1000, model_prob=0.58, market_prob=0.50, floor_to_usd=5.0
        )
        assert pos.stake_usd > 5.0
        assert not pos.reason.startswith("floored")


class TestSizeLockedPositionFloor:
    def test_locked_floor_lifts_sub_min(self):
        """Fixed 2% below $5 at small bankroll is floored up when caps allow."""
        # bankroll 150 → 2% = $3 (< $5); exposure/usd/depth caps leave room.
        dropped = size_locked_position(150, price=0.80)
        assert dropped.stake_usd == 0
        floored = size_locked_position(150, price=0.80, floor_to_usd=5.0)
        assert floored.stake_usd == pytest.approx(5.0)
        assert floored.reason.startswith("floored")

    def test_locked_floor_none_preserves_drop(self):
        """floor_to_usd=None keeps the original lock-path drop."""
        pos = size_locked_position(150, price=0.80, floor_to_usd=None)
        assert pos.stake_usd == 0
        assert "$5" in pos.reason


class TestSizeLockedPositionConviction:
    """Conviction-weighted lock sizing (``win_prob`` set) — Kelly-from-price
    capped by the concentration backstop, vs the flat 2% path."""

    def test_win_prob_none_reproduces_flat_path(self):
        """Regression: the default (win_prob=None) is bit-for-bit the legacy
        flat 2% result — 2% of $400 = $8, kelly_pct 0."""
        pos = size_locked_position(400, price=0.85)
        assert pos.stake_usd == pytest.approx(8.0)
        assert pos.kelly_pct == 0.0
        assert "fixed 2%" in pos.reason

    def test_super_margin_concentration_cap_binds(self):
        """At high win_prob the f*×fraction overshoots the cap, so the
        max_position_pct backstop governs: 15% of $400 = $60."""
        pos = size_locked_position(
            400, price=0.85,
            win_prob=0.99, kelly_fraction=0.25, max_position_pct=0.15,
        )
        # f* = 0.99 - 0.01/(0.15/0.85) ≈ 0.933; ×0.25 ≈ 0.233 → capped to 0.15.
        assert pos.kelly_pct == pytest.approx(0.15)
        assert pos.stake_usd == pytest.approx(60.0)
        assert "conviction kelly" in pos.reason

    def test_cheaper_lock_sizes_bigger_than_dearer(self):
        """When f* (not the cap) binds, a lower price → more EV → larger
        stake. Small bankroll keeps the $100 USD cap out of the way."""
        cheap = size_locked_position(
            100, price=0.60,
            win_prob=0.90, kelly_fraction=1.0, max_position_pct=1.0,
        )
        dear = size_locked_position(
            100, price=0.80,
            win_prob=0.90, kelly_fraction=1.0, max_position_pct=1.0,
        )
        # f*(0.60)=0.90-0.10/(0.4/0.6)=0.75 ; f*(0.80)=0.90-0.10/(0.2/0.8)=0.50
        assert cheap.kelly_pct == pytest.approx(0.75)
        assert dear.kelly_pct == pytest.approx(0.50)
        assert cheap.stake_usd > dear.stake_usd

    def test_relaxed_depth_cap_pct(self):
        """depth_cap_pct widens the depth ceiling for conviction locks: a $100
        book caps at 50% ($50) instead of the flat path's 15% ($15)."""
        big = size_locked_position(
            400, price=0.85, orderbook_depth=100.0,
            win_prob=0.99, kelly_fraction=0.25, max_position_pct=0.15,
            depth_cap_pct=0.50,
        )
        # raw would be $60 (15% cap), depth-capped to 50% of $100 = $50.
        assert big.stake_usd == pytest.approx(50.0)
        assert "50% of $100" in big.reason

    def test_exposure_cap_still_applies(self):
        """The standard exposure cap clamps conviction stakes too."""
        pos = size_locked_position(
            400, price=0.85, current_exposure=280.0,
            win_prob=0.99, kelly_fraction=0.25, max_position_pct=0.15,
        )
        # raw $60, but exposure floor cap = max(100, 300)=300; remaining=20.
        assert pos.stake_usd == pytest.approx(20.0)
        assert "exposure cap" in pos.reason

    def test_no_kelly_edge_at_high_price_returns_zero(self):
        """When price ≥ win_prob there's no Kelly edge → f*=0 → stake 0."""
        pos = size_locked_position(
            400, price=0.95,
            win_prob=0.90, kelly_fraction=0.25, max_position_pct=0.15,
        )
        assert pos.stake_usd == 0.0
        assert pos.kelly_pct == 0.0


# ===================================================================
# binary_market.py – near-peak floor-up gate
# ===================================================================


class TestNearPeakFloorEligible:
    @staticmethod
    def _market(op: str):
        m = MagicMock()
        m.parsed_operator = op
        return m

    def test_disabled_by_default(self):
        """Master switch off (default) → never eligible."""
        assert not near_peak_floor_eligible(
            self._market("at_least"), our_probability=1.0, hours_until_peak=0.0
        )

    def test_excludes_bracket_like(self, monkeypatch):
        """`exactly` stays validate-first — never floored even near peak."""
        monkeypatch.setattr(settings, "NEAR_PEAK_FLOOR_UP_ENABLED", True)
        assert not near_peak_floor_eligible(
            self._market("exactly"), our_probability=1.0, hours_until_peak=0.0
        )

    def test_confidence_arm_admits_far_from_peak(self, monkeypatch):
        """Karachi archetype: at_least, prob 1.0, ~5.5h pre-peak → eligible."""
        monkeypatch.setattr(settings, "NEAR_PEAK_FLOOR_UP_ENABLED", True)
        assert near_peak_floor_eligible(
            self._market("at_least"), our_probability=1.0, hours_until_peak=5.5
        )

    def test_near_peak_arm_admits_lower_confidence(self, monkeypatch):
        """Threshold op near peak with sub-min-prob still eligible via window."""
        monkeypatch.setattr(settings, "NEAR_PEAK_FLOOR_UP_ENABLED", True)
        assert near_peak_floor_eligible(
            self._market("above"), our_probability=0.88, hours_until_peak=1.0
        )

    def test_neither_arm_rejected(self, monkeypatch):
        """Low confidence AND far from peak → not eligible."""
        monkeypatch.setattr(settings, "NEAR_PEAK_FLOOR_UP_ENABLED", True)
        assert not near_peak_floor_eligible(
            self._market("at_least"), our_probability=0.88, hours_until_peak=6.0
        )


class TestNearLockConvictionEligible:
    @staticmethod
    def _market(op: str):
        m = MagicMock()
        m.parsed_operator = op
        return m

    def test_disabled_by_default(self):
        """Master switch off (default) → never eligible."""
        assert not near_lock_conviction_eligible(
            self._market("at_least"), our_probability=1.0, hours_until_peak=0.0
        )

    def test_eligible_at_peak_certain_threshold(self, monkeypatch):
        """The blocked archetype: at_least, prob 1.0, at peak → eligible."""
        monkeypatch.setattr(settings, "NEAR_LOCK_CONVICTION_SIZING_ENABLED", True)
        assert near_lock_conviction_eligible(
            self._market("at_least"), our_probability=1.0, hours_until_peak=0.0
        )

    def test_excludes_bracket_like(self, monkeypatch):
        """exactly/range/bracket must never bypass the prob cap."""
        monkeypatch.setattr(settings, "NEAR_LOCK_CONVICTION_SIZING_ENABLED", True)
        assert not near_lock_conviction_eligible(
            self._market("exactly"), our_probability=1.0, hours_until_peak=0.0
        )

    def test_rejects_pre_peak(self, monkeypatch):
        """Before peak the max is forecast not observed → not eligible (the
        safe-subset guard against the prob≈1.0/78%-win overconfidence)."""
        monkeypatch.setattr(settings, "NEAR_LOCK_CONVICTION_SIZING_ENABLED", True)
        assert not near_lock_conviction_eligible(
            self._market("at_least"), our_probability=1.0, hours_until_peak=2.0
        )

    def test_rejects_sub_certain_prob(self, monkeypatch):
        """Not physically locked (prob < MIN_PROB) → not eligible."""
        monkeypatch.setattr(settings, "NEAR_LOCK_CONVICTION_SIZING_ENABLED", True)
        assert not near_lock_conviction_eligible(
            self._market("at_least"), our_probability=0.95, hours_until_peak=0.0
        )

    def test_past_peak_uses_abs_window(self, monkeypatch):
        """Negative hours_until_peak (past peak) counts via abs()."""
        monkeypatch.setattr(settings, "NEAR_PEAK_FLOOR_UP_ENABLED", True)
        assert near_peak_floor_eligible(
            self._market("below"), our_probability=0.50, hours_until_peak=-1.5
        )


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

    def test_pause_threshold_uses_settings_override(self, monkeypatch):
        """Widening DRAWDOWN_PAUSE_THRESHOLD demotes a 24% dd from PAUSED to CAUTION."""
        mon = DrawdownMonitor(417)
        mon.advance(417)
        # 316/417 → 24.2% drawdown. Default 0.20 pauses; 0.30 only cautions.
        assert mon.check(316).level == DrawdownLevel.PAUSED
        monkeypatch.setattr(settings, "DRAWDOWN_PAUSE_THRESHOLD", 0.30)
        assert mon.check(316).level == DrawdownLevel.CAUTION

    def test_reload_to_lower_peak_cannot_underprotect(self):
        """A peak reloaded down to current equity still pauses on a real drop.

        Models `admin reset-drawdown-peak` lowering the peak to $316: equity at
        $316 is NORMAL, but a genuine fall to $250 (21% off the reset baseline)
        still PAUSES because check() re-maxes against live equity.
        """
        mon = DrawdownMonitor(316)  # peak reloaded to current equity
        assert mon.check(316).level == DrawdownLevel.NORMAL
        assert mon.check(250).level == DrawdownLevel.PAUSED

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
