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
    return AsyncMock()


@pytest.fixture(autouse=True)
def reset_pause_state():
    """Clear global pause state before each test."""
    cb_module._paused_until = None
    yield
    cb_module._paused_until = None


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
        mock_session.execute = AsyncMock(side_effect=[
            MagicMock(scalar=MagicMock(return_value=-50.0)),
            MagicMock(all=MagicMock(return_value=[
                (TradeStatus.LOST,),
                (TradeStatus.LOST,),
                (TradeStatus.LOST,),
            ])),
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
        # Set pause to 1 hour from now
        cb_module._paused_until = datetime.now(timezone.utc) + timedelta(hours=1)

        state = await check_circuit_breakers(mock_session)
        assert state.can_trade is False
        assert "remaining" in state.reason

    @pytest.mark.asyncio
    async def test_pause_expires(self, mock_session):
        # Set pause to 1 hour ago (expired)
        cb_module._paused_until = datetime.now(timezone.utc) - timedelta(hours=1)

        # Normal PnL and no losses
        mock_session.execute = AsyncMock(side_effect=[
            MagicMock(scalar=MagicMock(return_value=0.0)),
            MagicMock(all=MagicMock(return_value=[])),
        ])

        state = await check_circuit_breakers(mock_session)
        assert state.can_trade is True
