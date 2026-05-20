"""Tests for circuit breakers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.risk.circuit_breakers import (
    CircuitBreakerState,
    _paused_until,
    check_circuit_breakers,
)
from src.db.models import TradeStatus
import src.risk.circuit_breakers as cb_module


@pytest.fixture
def mock_session():
    """Session that returns None from ``get`` (no persisted bot_state row)
    by default. Tests that set ``_paused_until`` in-process should NOT
    have it overwritten on hydration."""
    session = AsyncMock()
    session.get = AsyncMock(return_value=None)
    return session


@pytest.fixture(autouse=True)
def reset_pause_state():
    """Clear in-process pause + hydration flag before each test so cross-
    test state from a previous test's BotState hydration doesn't leak."""
    cb_module._paused_until = None
    cb_module._hydrated_from_db = False
    yield
    cb_module._paused_until = None
    cb_module._hydrated_from_db = False


class TestDailyLossStop:
    @pytest.mark.asyncio
    async def test_allows_trading_when_profitable(self, mock_session):
        # 1: daily PnL = +50, 2: consecutive losses = [],
        # 3: submit failures = 0 (added 2026-05-20).
        mock_session.execute = AsyncMock(side_effect=[
            MagicMock(scalar=MagicMock(return_value=50.0)),
            MagicMock(all=MagicMock(return_value=[])),
            MagicMock(scalar=MagicMock(return_value=0)),
        ])

        state = await check_circuit_breakers(mock_session)
        assert state.can_trade is True
        assert state.reason is None

    @pytest.mark.asyncio
    async def test_halts_on_daily_loss(self, mock_session):
        # Daily PnL = -250
        mock_session.execute = AsyncMock(side_effect=[
            MagicMock(scalar=MagicMock(return_value=-250.0)),
        ])

        state = await check_circuit_breakers(mock_session)
        assert state.can_trade is False
        assert "daily loss" in state.reason


class TestConsecutiveLossStop:
    @pytest.mark.asyncio
    async def test_pauses_after_3_losses(self, mock_session):
        # Daily PnL = -50 (not tripped)
        # Consecutive losses = [LOST, LOST, LOST]
        # Third execute is the BotState upsert that persists the new pause.
        mock_session.execute = AsyncMock(side_effect=[
            MagicMock(scalar=MagicMock(return_value=-50.0)),
            MagicMock(all=MagicMock(return_value=[
                (TradeStatus.LOST,),
                (TradeStatus.LOST,),
                (TradeStatus.LOST,),
            ])),
            MagicMock(),  # _persist_paused_until upsert
        ])

        state = await check_circuit_breakers(mock_session)
        assert state.can_trade is False
        assert "consecutive" in state.reason
        assert state.paused_until is not None

    @pytest.mark.asyncio
    async def test_allows_after_win_breaks_streak(self, mock_session):
        # Daily PnL = -50
        # Trades: LOST, WON → streak = 1 (not enough)
        # Submit failures = 0
        mock_session.execute = AsyncMock(side_effect=[
            MagicMock(scalar=MagicMock(return_value=-50.0)),
            MagicMock(all=MagicMock(return_value=[
                (TradeStatus.LOST,),
                (TradeStatus.WON,),
            ])),
            MagicMock(scalar=MagicMock(return_value=0)),
        ])

        state = await check_circuit_breakers(mock_session)
        assert state.can_trade is True


class TestPauseDuration:
    @pytest.mark.asyncio
    async def test_respects_pause_window(self, mock_session):
        # Set pause to 1 hour from now. Mark already-hydrated so the helper
        # doesn't overwrite the in-process value from a (mocked) DB read.
        cb_module._paused_until = datetime.now(timezone.utc) + timedelta(hours=1)
        cb_module._hydrated_from_db = True

        state = await check_circuit_breakers(mock_session)
        assert state.can_trade is False
        assert "remaining" in state.reason

    @pytest.mark.asyncio
    async def test_pause_expires(self, mock_session):
        # Set pause to 1 hour ago (expired)
        cb_module._paused_until = datetime.now(timezone.utc) - timedelta(hours=1)
        cb_module._hydrated_from_db = True

        # The expired-pause clear runs _persist_paused_until(None) which
        # calls session.get + (delete if exists) — get returns None per
        # the fixture, so no execute call. Then daily PnL + consecutive
        # losses + submit failures = 3 execute calls.
        mock_session.execute = AsyncMock(side_effect=[
            MagicMock(scalar=MagicMock(return_value=0.0)),
            MagicMock(all=MagicMock(return_value=[])),
            MagicMock(scalar=MagicMock(return_value=0)),
        ])

        state = await check_circuit_breakers(mock_session)
        assert state.can_trade is True

class TestSubmitFailurePause:
    """Submission-failure circuit breaker (added 2026-05-20).

    Counts trades with ``exchange_status LIKE 'exception:%'`` opened in
    the recent window. Once ``SUBMIT_FAIL_PAUSE_COUNT`` is hit, trading
    pauses for ``SUBMIT_FAIL_PAUSE_MINUTES`` and a Telegram alert
    fires. Catches the May 2026 failure mode where 115
    PolyApiException rows piled up in one week without being noticed.
    """

    @pytest.mark.asyncio
    async def test_pauses_after_burst_of_submit_failures(self, mock_session):
        from unittest.mock import patch

        # 1: daily PnL = 0, 2: no consecutive losses,
        # 3: submit failures = 5 (≥ SUBMIT_FAIL_PAUSE_COUNT default),
        # 4: the BotState persist upsert.
        mock_session.execute = AsyncMock(side_effect=[
            MagicMock(scalar=MagicMock(return_value=0.0)),
            MagicMock(all=MagicMock(return_value=[])),
            MagicMock(scalar=MagicMock(return_value=5)),
            MagicMock(),
        ])

        # Skip the Telegram fan-out — _maybe_alert_submit_failures
        # imports the alerter at call time, and we don't want a real
        # one in tests. Patching the module-level helper is enough.
        with patch(
            "src.risk.circuit_breakers._maybe_alert_submit_failures",
            new=AsyncMock(),
        ):
            state = await check_circuit_breakers(mock_session)

        assert state.can_trade is False
        assert "submission failures" in (state.reason or "")
        assert state.paused_until is not None

    @pytest.mark.asyncio
    async def test_allows_below_failure_threshold(self, mock_session):
        # 4 failures < threshold of 5 → still trade.
        mock_session.execute = AsyncMock(side_effect=[
            MagicMock(scalar=MagicMock(return_value=0.0)),
            MagicMock(all=MagicMock(return_value=[])),
            MagicMock(scalar=MagicMock(return_value=4)),
        ])

        state = await check_circuit_breakers(mock_session)
        assert state.can_trade is True


class TestBotStateHydration:
    @pytest.mark.asyncio
    async def test_hydrates_from_bot_state_on_first_call(self, mock_session):
        """A pause that was active at process shutdown is restored from
        ``bot_state`` so the protective window survives a restart."""
        future = datetime.now(timezone.utc) + timedelta(hours=1, minutes=15)
        persisted_row = MagicMock()
        persisted_row.value = {"until_iso": future.isoformat()}
        mock_session.get = AsyncMock(return_value=persisted_row)

        # _hydrated_from_db is False (fresh boot), so first call should
        # hit ``session.get`` and re-engage the pause.
        state = await check_circuit_breakers(mock_session)
        assert state.can_trade is False
        assert "remaining" in state.reason
        mock_session.get.assert_awaited_once()
