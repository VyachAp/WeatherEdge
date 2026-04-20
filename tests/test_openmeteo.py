"""Tests for the Open-Meteo forecast client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ingestion.openmeteo import (
    OpenMeteoForecast,
    cloud_rising,
    dewpoint_trend,
    fetch_forecast,
    solar_declining,
)


def _make_forecast(**overrides) -> OpenMeteoForecast:
    defaults = dict(
        peak_temp_c=30.0,
        peak_hour_utc=14,
        hourly_temps_c=[20.0 + i * 0.8 for i in range(24)],
        hourly_cloud_cover=[10] * 12 + [60] * 12,
        hourly_solar_radiation=[0] * 6 + [200, 400, 600, 800, 700, 500, 300, 100] + [0] * 10,
        hourly_dewpoint_c=[15.0] * 24,
        hourly_wind_speed=[3.0] * 24,
    )
    defaults.update(overrides)
    return OpenMeteoForecast(**defaults)


class TestSolarDeclining:
    def test_detects_decline(self):
        forecast = _make_forecast(
            hourly_solar_radiation=[0] * 10 + [800, 300] + [0] * 12,
        )
        is_declining, mag = solar_declining(forecast, current_hour_utc=10, window_hours=1)
        assert is_declining is True
        assert mag > 0.5

    def test_no_decline_morning(self):
        forecast = _make_forecast(
            hourly_solar_radiation=[0, 100, 300, 500, 700, 800, 800, 700, 500, 300, 100, 0] + [0] * 12,
        )
        is_declining, mag = solar_declining(forecast, current_hour_utc=2, window_hours=2)
        assert is_declining is False

    def test_zero_solar_no_decline(self):
        forecast = _make_forecast(hourly_solar_radiation=[0] * 24)
        is_declining, mag = solar_declining(forecast, current_hour_utc=12)
        assert is_declining is False
        assert mag == 0.0


class TestCloudRising:
    def test_detects_rising(self):
        forecast = _make_forecast(
            hourly_cloud_cover=[10] * 10 + [30, 80] + [90] * 12,
        )
        is_rising, mag = cloud_rising(forecast, current_hour_utc=10, window_hours=1)
        assert is_rising is True

    def test_no_rise_clear_day(self):
        forecast = _make_forecast(hourly_cloud_cover=[10] * 24)
        is_rising, mag = cloud_rising(forecast, current_hour_utc=12)
        assert is_rising is False


class TestDewpointTrend:
    def test_rising_dewpoint(self):
        forecast = _make_forecast(
            hourly_dewpoint_c=[10.0, 11.0, 12.0, 13.0, 14.0, 15.0] + [15.0] * 18,
        )
        rate = dewpoint_trend(forecast, current_hour_utc=0, window_hours=3)
        assert rate == pytest.approx(1.0)

    def test_falling_dewpoint(self):
        forecast = _make_forecast(
            hourly_dewpoint_c=[15.0, 14.0, 13.0, 12.0] + [12.0] * 20,
        )
        rate = dewpoint_trend(forecast, current_hour_utc=0, window_hours=3)
        assert rate == pytest.approx(-1.0)

    def test_stable_dewpoint(self):
        forecast = _make_forecast(hourly_dewpoint_c=[15.0] * 24)
        rate = dewpoint_trend(forecast, current_hour_utc=12)
        assert rate == pytest.approx(0.0)


class TestFetchForecast:
    @pytest.mark.asyncio
    async def test_successful_fetch(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "hourly": {
                "temperature_2m": [20.0 + i for i in range(24)],
                "cloudcover": [50] * 24,
                "shortwave_radiation": [100] * 24,
                "dewpoint_2m": [10.0] * 24,
                "windspeed_10m": [5.0] * 24,
            }
        }

        with patch("src.ingestion.openmeteo.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await fetch_forecast(40.0, -74.0)

        assert result is not None
        assert result.peak_temp_c == 43.0  # 20 + 23
        assert result.peak_hour_utc == 23
        assert len(result.hourly_temps_c) == 24

    @pytest.mark.asyncio
    async def test_fetch_failure_returns_none(self):
        with patch("src.ingestion.openmeteo.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.side_effect = Exception("Network error")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await fetch_forecast(40.0, -74.0)

        assert result is None
