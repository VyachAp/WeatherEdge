"""Tests for state_aggregator.build_state_from_metars (pure-function path)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from src.signals.state_aggregator import build_state_from_metars


@dataclass
class _FakeForecast:
    """Minimal stand-in for ingestion.openmeteo.OpenMeteoForecast."""
    peak_temp_c: float
    peak_hour_utc: int
    hourly_temps_c: list[float]
    hourly_cloud_cover: list[int]
    hourly_solar_radiation: list[float]
    hourly_dewpoint_c: list[float]
    hourly_wind_speed: list[float]
    peak_temp_std_c: float = 0.0
    model_count: int = 1


def _hist_today(now_utc: datetime, max_f: float = 50.0):
    """Construct a routine METAR history with the desired max in window."""
    start = now_utc - timedelta(hours=4)
    return [
        {"observed_at": start + timedelta(hours=i), "temp_f": max_f - 4 + i,
         "is_speci": False, "dewpoint_f": max_f - 14 + i}
        for i in range(5)
    ]


class TestHoursUntilPeakMidnightWraparound:
    """When Open-Meteo's `forecast_days=1, timezone=UTC` returns
    peak_hour_utc=0 because today's local-day heating bleeds into hour 0
    of the next UTC day, the bot must interpret that as 'peak is later
    today local' rather than 'peak was -17h ago'.
    """

    def _make_forecast(self, peak_hour_utc: int) -> _FakeForecast:
        return _FakeForecast(
            peak_temp_c=21.0,  # ~70°F
            peak_hour_utc=peak_hour_utc,
            hourly_temps_c=[15.0] * 24,
            hourly_cloud_cover=[20] * 24,
            hourly_solar_radiation=[600.0] * 24,
            hourly_dewpoint_c=[5.0] * 24,
            hourly_wind_speed=[5.0] * 24,
            model_count=4,
        )

    def test_denver_morning_peak_in_future_not_past(self):
        """Reproduction of the user-reported bug: Denver 11:40 AM MDT,
        Open-Meteo returns peak_hour_utc=0 (= Apr 25 18:00 MDT, yesterday's
        evening). Bot should recognize today's peak is ~6h ahead, not
        yesterday's was 17h ago."""
        # Apr 26 17:40 UTC = 11:40 MDT.
        now = datetime(2026, 4, 26, 17, 40, tzinfo=timezone.utc)
        forecast = self._make_forecast(peak_hour_utc=0)
        history = _hist_today(now, max_f=38.0)
        state = build_state_from_metars("KDEN", history, forecast,
                                        bias_c=0.0, now_utc=now)
        assert state is not None
        # Today's Denver peak ~18:00 MDT = 00:00 UTC Apr 27 → ~6.3 h from now.
        assert state.hours_until_peak == pytest.approx(6.33, abs=0.1)

    def test_atlanta_evening_past_peak_unchanged(self):
        """Atlanta UTC-4, after sunset: peak_hour_utc=20 (= 16:00 EDT
        today) is in the past by ~3h. Should remain negative — do NOT
        shift to tomorrow."""
        # Apr 26 23:00 UTC = 19:00 EDT.
        now = datetime(2026, 4, 26, 23, 0, tzinfo=timezone.utc)
        forecast = self._make_forecast(peak_hour_utc=20)
        history = _hist_today(now, max_f=78.0)
        state = build_state_from_metars("KATL", history, forecast,
                                        bias_c=0.0, now_utc=now)
        assert state is not None
        # Peak was at 20:00 UTC = 16:00 EDT, ~3h before now. Stay negative.
        assert state.hours_until_peak == pytest.approx(-3.0, abs=0.1)

    def test_london_midday_peak_ahead(self):
        """London UTC+0/UTC+1: peak_hour_utc=14 mid-afternoon. At 09:00
        local, peak is ~5h ahead."""
        # Apr 26 09:00 UTC = 10:00 BST.
        now = datetime(2026, 4, 26, 9, 0, tzinfo=timezone.utc)
        forecast = self._make_forecast(peak_hour_utc=14)
        history = _hist_today(now, max_f=55.0)
        state = build_state_from_metars("EGLL", history, forecast,
                                        bias_c=0.0, now_utc=now)
        assert state is not None
        assert state.hours_until_peak == pytest.approx(5.0, abs=0.1)
