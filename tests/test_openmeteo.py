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
        hourly_temps_std_c=[0.5] * 24,
        peak_temp_std_c=0.8,
        model_count=5,
    )
    defaults.update(overrides)
    return OpenMeteoForecast(**defaults)


def _build_ensemble_response(
    model_temps: dict[str, list[float | None]],
    hours: int = 24,
) -> dict:
    """Shape an Open-Meteo multi-model response for mocking."""
    hourly: dict = {"time": [f"2026-04-23T{h:02d}:00" for h in range(hours)]}
    for m, temps in model_temps.items():
        hourly[f"temperature_2m_{m}"] = temps
        hourly[f"dewpoint_2m_{m}"] = [10.0] * hours
        hourly[f"cloudcover_{m}"] = [50] * hours
        hourly[f"shortwave_radiation_{m}"] = [100.0] * hours
        hourly[f"windspeed_10m_{m}"] = [5.0] * hours
    return {"hourly": hourly}


def _mock_async_client(responses: list):
    """Return an AsyncMock httpx client factory that cycles through responses."""
    mock_client = AsyncMock()
    mock_responses = []
    for body in responses:
        mr = MagicMock()
        mr.status_code = 200
        mr.raise_for_status = MagicMock()
        mr.json.return_value = body
        mock_responses.append(mr)
    mock_client.get.side_effect = mock_responses
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


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


class TestFetchForecastEnsemble:
    @pytest.mark.asyncio
    async def test_parses_multi_model(self):
        """4 models with spread → peak_temp_std_c > 0, model_count == 4."""
        base = [20.0 + i for i in range(24)]
        body = _build_ensemble_response({
            "ecmwf_ifs025":  [t + 0.0 for t in base],
            "gfs_seamless":  [t + 0.5 for t in base],
            "icon_seamless": [t - 0.5 for t in base],
            "gem_seamless":  [t + 1.0 for t in base],
        })

        with patch("src.ingestion.openmeteo.httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value = _mock_async_client([body])
            result = await fetch_forecast(40.0, -74.0)

        assert result is not None
        assert result.model_count == 4
        assert result.peak_temp_std_c > 0.3       # non-trivial spread
        assert result.peak_temp_std_c < 1.0       # and not absurd
        # Median of offsets [-0.5, 0.0, 0.5, 1.0] = 0.25 (mean of two middle vals).
        assert result.peak_temp_c == pytest.approx(base[-1] + 0.25, abs=0.01)
        assert len(result.hourly_temps_std_c) == 24

    @pytest.mark.asyncio
    async def test_falls_back_when_insufficient_models(self):
        """Only 2 models in response → deterministic fallback kicks in."""
        ensemble_body = _build_ensemble_response({
            "ecmwf_ifs025": [20.0 + i for i in range(24)],
            "gfs_seamless": [20.0 + i for i in range(24)],
        })
        deterministic_body = {
            "hourly": {
                "temperature_2m": [25.0 + i for i in range(24)],
                "cloudcover": [50] * 24,
                "shortwave_radiation": [100] * 24,
                "dewpoint_2m": [10.0] * 24,
                "windspeed_10m": [5.0] * 24,
            }
        }

        with patch("src.ingestion.openmeteo.httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value = _mock_async_client([
                ensemble_body, deterministic_body,
            ])
            result = await fetch_forecast(40.0, -74.0)

        assert result is not None
        assert result.model_count == 1
        assert result.peak_temp_std_c == 0.0
        assert result.hourly_temps_std_c == []
        # Picked up the deterministic body (peak=48 at hour 23), not ensemble
        assert result.peak_temp_c == 48.0

    @pytest.mark.asyncio
    async def test_spread_reflects_disagreement(self):
        """Wider inter-model spread → larger peak_temp_std_c."""
        base = [20.0 + i for i in range(24)]
        tight = _build_ensemble_response({
            "ecmwf_ifs025":  base,
            "gfs_seamless":  base,
            "icon_seamless": [t + 0.1 for t in base],
            "gem_seamless":  [t - 0.1 for t in base],
        })
        wide = _build_ensemble_response({
            "ecmwf_ifs025":  base,
            "gfs_seamless":  [t + 3.0 for t in base],
            "icon_seamless": [t - 2.5 for t in base],
            "gem_seamless":  [t + 1.5 for t in base],
        })

        with patch("src.ingestion.openmeteo.httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value = _mock_async_client([tight])
            tight_result = await fetch_forecast(40.0, -74.0)
        with patch("src.ingestion.openmeteo.httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value = _mock_async_client([wide])
            wide_result = await fetch_forecast(40.0, -74.0)

        assert tight_result is not None and wide_result is not None
        assert wide_result.peak_temp_std_c > tight_result.peak_temp_std_c * 3

    @pytest.mark.asyncio
    async def test_handles_null_values(self):
        """Null entries for some hours should be dropped from median/std."""
        base = [20.0 + i for i in range(24)]
        body = _build_ensemble_response({
            "ecmwf_ifs025":  base,
            "gfs_seamless":  [None] * 24,       # offline
            "icon_seamless": [t - 0.5 for t in base],
            "gem_seamless":  [None if i < 6 else t + 0.5
                              for i, t in enumerate(base)],
        })

        with patch("src.ingestion.openmeteo.httpx.AsyncClient") as mock_client_cls:
            mock_client_cls.return_value = _mock_async_client([body])
            result = await fetch_forecast(40.0, -74.0)

        # gfs is null throughout so it's dropped entirely; remaining 3 models.
        assert result is not None
        assert result.model_count == 3
        assert result.peak_temp_std_c > 0
