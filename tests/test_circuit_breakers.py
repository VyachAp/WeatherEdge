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
        # First call returns daily PnL = +50
        # Second call returns consecutive losses = 0
        mock_session.execute = AsyncMock(side_effect=[
            MagicMock(scalar=MagicMock(return_value=50.0)),
            MagicMock(all=MagicMock(return_value=[])),
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
        mock_session.execute = AsyncMock(side_effect=[
            MagicMock(scalar=MagicMock(return_value=-50.0)),
            MagicMock(all=MagicMock(return_value=[
                (TradeStatus.LOST,),
                (TradeStatus.WON,),
            ])),
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

        # Normal PnL and no losses
        mock_session.execute = AsyncMock(side_effect=[
            MagicMock(scalar=MagicMock(return_value=0.0)),
            MagicMock(all=MagicMock(return_value=[])),
        ])

        state = await check_circuit_breakers(mock_session)
        assert state.can_trade is True

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
