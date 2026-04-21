"""Tests for forecast exceedance Telegram alerts."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ingestion.openmeteo import OpenMeteoForecast
from src.signals.forecast_exceedance import (
    _closest_hour_index,
    _pick_latest_routine,
    check_and_alert_exceedance,
)


def _make_forecast(hourly_temps_c: list[float] | None = None) -> OpenMeteoForecast:
    temps = hourly_temps_c if hourly_temps_c is not None else [24.0] * 24
    return OpenMeteoForecast(
        peak_temp_c=max(temps),
        peak_hour_utc=temps.index(max(temps)),
        hourly_temps_c=temps,
        hourly_cloud_cover=[20] * 24,
        hourly_solar_radiation=[500.0] * 24,
        hourly_dewpoint_c=[15.0] * 24,
        hourly_wind_speed=[5.0] * 24,
    )


def _metar(observed_at: datetime, temp_f: float, is_speci: bool = False) -> dict:
    return {
        "observed_at": observed_at,
        "temp_f": temp_f,
        "is_speci": is_speci,
    }


class TestPickLatestRoutine:
    def test_returns_newest_routine(self):
        older = _metar(datetime(2026, 4, 22, 13, 53, tzinfo=timezone.utc), 75.0)
        newer = _metar(datetime(2026, 4, 22, 14, 53, tzinfo=timezone.utc), 78.0)
        assert _pick_latest_routine([older, newer])["temp_f"] == 78.0

    def test_skips_speci(self):
        routine = _metar(datetime(2026, 4, 22, 14, 53, tzinfo=timezone.utc), 75.0)
        speci = _metar(datetime(2026, 4, 22, 15, 10, tzinfo=timezone.utc), 80.0, is_speci=True)
        assert _pick_latest_routine([routine, speci])["temp_f"] == 75.0

    def test_returns_none_when_empty(self):
        assert _pick_latest_routine([]) is None

    def test_skips_missing_temp(self):
        m = _metar(datetime(2026, 4, 22, 14, 53, tzinfo=timezone.utc), 75.0)
        no_temp = {"observed_at": datetime(2026, 4, 22, 15, 0, tzinfo=timezone.utc), "temp_f": None, "is_speci": False}
        assert _pick_latest_routine([m, no_temp])["temp_f"] == 75.0


class TestClosestHourIndex:
    def test_rounds_down_before_half_hour(self):
        ts = datetime(2026, 4, 22, 14, 15, tzinfo=timezone.utc)
        assert _closest_hour_index(ts, 24) == 14

    def test_rounds_up_at_half_hour(self):
        ts = datetime(2026, 4, 22, 14, 30, tzinfo=timezone.utc)
        assert _closest_hour_index(ts, 24) == 15

    def test_clamps_to_last_hour(self):
        ts = datetime(2026, 4, 22, 23, 55, tzinfo=timezone.utc)
        assert _closest_hour_index(ts, 24) == 23


class _FakeResult:
    def __init__(self, value):
        self._value = value

    def scalar(self):
        return self._value


class _FakeSession:
    """Minimal async-session double for tests.

    Tracks session.add() calls and returns a configurable scalar() for the
    existence check.
    """

    def __init__(self, existing_id=None, commit_raises=None):
        self.added: list = []
        self._existing_id = existing_id
        self._commit_raises = commit_raises
        self.committed = False
        self.rolled_back = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def execute(self, _stmt):
        return _FakeResult(self._existing_id)

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        if self._commit_raises is not None:
            raise self._commit_raises
        self.committed = True

    async def rollback(self):
        self.rolled_back = True


class TestCheckAndAlertExceedance:
    @pytest.mark.asyncio
    async def test_fires_alert_when_delta_exceeds_threshold(self):
        observed_at = datetime(2026, 4, 22, 14, 53, tzinfo=timezone.utc)
        history = [_metar(observed_at, 78.5)]
        # Forecast at hour 15 (14:53 rounds up): 24°C ≈ 75.2°F, delta +3.3°F
        forecast = _make_forecast([24.0] * 24)

        fake_session = _FakeSession(existing_id=None)
        alerter = MagicMock()
        alerter._enqueue = AsyncMock()

        with patch("src.signals.forecast_exceedance.async_session", return_value=fake_session), \
             patch("src.signals.forecast_exceedance.get_alerter", return_value=alerter):
            await check_and_alert_exceedance("KLAX", history, forecast)

        assert len(fake_session.added) == 1
        assert fake_session.committed is True
        alerter._enqueue.assert_awaited_once()
        msg = alerter._enqueue.await_args.args[0]
        assert "KLAX" in msg
        assert "78.5" in msg

    @pytest.mark.asyncio
    async def test_no_alert_when_delta_below_threshold(self):
        observed_at = datetime(2026, 4, 22, 14, 53, tzinfo=timezone.utc)
        # Forecast 24°C ≈ 75.2°F; obs 75.5°F → delta 0.3°F (< 0.5)
        history = [_metar(observed_at, 75.5)]
        forecast = _make_forecast([24.0] * 24)

        fake_session = _FakeSession()
        alerter = MagicMock()
        alerter._enqueue = AsyncMock()

        with patch("src.signals.forecast_exceedance.async_session", return_value=fake_session), \
             patch("src.signals.forecast_exceedance.get_alerter", return_value=alerter):
            await check_and_alert_exceedance("KLAX", history, forecast)

        assert fake_session.added == []
        alerter._enqueue.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dedup_skips_when_row_exists(self):
        observed_at = datetime(2026, 4, 22, 14, 53, tzinfo=timezone.utc)
        history = [_metar(observed_at, 80.0)]
        forecast = _make_forecast([24.0] * 24)

        fake_session = _FakeSession(existing_id=42)  # row already present
        alerter = MagicMock()
        alerter._enqueue = AsyncMock()

        with patch("src.signals.forecast_exceedance.async_session", return_value=fake_session), \
             patch("src.signals.forecast_exceedance.get_alerter", return_value=alerter):
            await check_and_alert_exceedance("KLAX", history, forecast)

        assert fake_session.added == []
        assert fake_session.committed is False
        alerter._enqueue.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_speci_obs(self):
        observed_at = datetime(2026, 4, 22, 14, 53, tzinfo=timezone.utc)
        # Only observation is a SPECI — should be ignored even if hot
        history = [_metar(observed_at, 90.0, is_speci=True)]
        forecast = _make_forecast([24.0] * 24)

        fake_session = _FakeSession()
        alerter = MagicMock()
        alerter._enqueue = AsyncMock()

        with patch("src.signals.forecast_exceedance.async_session", return_value=fake_session), \
             patch("src.signals.forecast_exceedance.get_alerter", return_value=alerter):
            await check_and_alert_exceedance("KLAX", history, forecast)

        assert fake_session.added == []
        alerter._enqueue.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_alert_when_forecast_none(self):
        observed_at = datetime(2026, 4, 22, 14, 53, tzinfo=timezone.utc)
        history = [_metar(observed_at, 90.0)]

        fake_session = _FakeSession()
        alerter = MagicMock()
        alerter._enqueue = AsyncMock()

        with patch("src.signals.forecast_exceedance.async_session", return_value=fake_session), \
             patch("src.signals.forecast_exceedance.get_alerter", return_value=alerter):
            await check_and_alert_exceedance("KLAX", history, None)

        assert fake_session.added == []
        alerter._enqueue.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_alert_when_history_empty(self):
        forecast = _make_forecast([24.0] * 24)
        fake_session = _FakeSession()
        alerter = MagicMock()
        alerter._enqueue = AsyncMock()

        with patch("src.signals.forecast_exceedance.async_session", return_value=fake_session), \
             patch("src.signals.forecast_exceedance.get_alerter", return_value=alerter):
            await check_and_alert_exceedance("KLAX", [], forecast)

        assert fake_session.added == []
        alerter._enqueue.assert_not_awaited()
