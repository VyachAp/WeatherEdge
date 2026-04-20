"""Tests for station bias tracking."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ingestion.station_bias import get_bias, is_bias_runaway, record_daily_outcome


@pytest.fixture
def mock_session():
    session = AsyncMock()
    return session


class TestGetBias:
    @pytest.mark.asyncio
    async def test_returns_default_when_no_history(self, mock_session):
        """With no data, returns the default 1.0°C bias."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = None
        mock_session.execute.return_value = mock_result

        bias = await get_bias(mock_session, "KJFK")
        assert bias == 1.0

    @pytest.mark.asyncio
    async def test_returns_average_bias(self, mock_session):
        """Returns the average bias from the database."""
        mock_result = MagicMock()
        mock_result.scalar.return_value = 1.5
        mock_session.execute.return_value = mock_result

        bias = await get_bias(mock_session, "KJFK")
        assert bias == 1.5


class TestIsBiasRunaway:
    @pytest.mark.asyncio
    async def test_normal_bias_is_not_runaway(self, mock_session):
        mock_result = MagicMock()
        mock_result.scalar.return_value = 1.0
        mock_session.execute.return_value = mock_result

        assert await is_bias_runaway(mock_session, "KJFK") is False

    @pytest.mark.asyncio
    async def test_large_positive_bias_is_runaway(self, mock_session):
        mock_result = MagicMock()
        mock_result.scalar.return_value = 4.0
        mock_session.execute.return_value = mock_result

        assert await is_bias_runaway(mock_session, "KJFK") is True

    @pytest.mark.asyncio
    async def test_large_negative_bias_is_runaway(self, mock_session):
        mock_result = MagicMock()
        mock_result.scalar.return_value = -3.5
        mock_session.execute.return_value = mock_result

        assert await is_bias_runaway(mock_session, "KJFK") is True


class TestRecordDailyOutcome:
    @pytest.mark.asyncio
    async def test_executes_upsert(self, mock_session):
        """record_daily_outcome should execute an INSERT...ON CONFLICT statement."""
        now = datetime.now(timezone.utc)
        await record_daily_outcome(mock_session, "KJFK", now, 30.0, 28.5)
        mock_session.execute.assert_called_once()
