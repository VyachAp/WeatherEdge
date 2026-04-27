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


class TestResidualSlope:
    """build_state_from_metars populates the residual-slope fields used by
    the v2 projection path. Slope is the linear regression of
    (obs - same-hour forecast) over the last 6h of routine METARs."""

    def _forecast(self, hourly_temps_c: list[float]) -> _FakeForecast:
        return _FakeForecast(
            peak_temp_c=max(hourly_temps_c),
            peak_hour_utc=hourly_temps_c.index(max(hourly_temps_c)),
            hourly_temps_c=hourly_temps_c,
            hourly_cloud_cover=[20] * 24,
            hourly_solar_radiation=[500.0] * 24,
            hourly_dewpoint_c=[15.0] * 24,
            hourly_wind_speed=[5.0] * 24,
        )

    def test_growing_residual_yields_positive_slope(self):
        # Forecast is flat at 24°C (~75.2°F) all day. Obs starts at 76°F
        # (residual +0.8°F) and rises 1°F/hr → residual slope ≈ +1°F/hr.
        now = datetime(2026, 4, 22, 18, 0, tzinfo=timezone.utc)
        forecast = self._forecast([24.0] * 24)
        history = [
            {"observed_at": now - timedelta(hours=h),
             "temp_f": 76.0 + (5 - h),  # h=5→76, h=4→77, ..., h=0→81
             "is_speci": False}
            for h in range(5, -1, -1)
        ]
        state = build_state_from_metars(
            "KLAX", history, forecast, bias_c=0.0, now_utc=now,
        )
        assert state is not None
        assert state.forecast_residual_count == 6
        assert state.forecast_residual_slope_f_per_hr == pytest.approx(1.0, abs=0.05)

    def test_flat_residual_yields_zero_slope(self):
        # Obs and forecast move in lockstep → constant residual → slope 0.
        now = datetime(2026, 4, 22, 18, 0, tzinfo=timezone.utc)
        # Forecast climbs 1°C per hour (1.8°F).
        hourly = [10.0 + h * 1.0 for h in range(24)]
        forecast = self._forecast(hourly)
        # Obs always +1.8°F over the same-hour forecast cell.
        history = []
        for h in range(5, -1, -1):
            obs_at = now - timedelta(hours=h)
            forecast_f = (hourly[obs_at.hour] * 9.0 / 5.0) + 32.0
            history.append({
                "observed_at": obs_at,
                "temp_f": forecast_f + 1.8,
                "is_speci": False,
            })
        state = build_state_from_metars(
            "KLAX", history, forecast, bias_c=0.0, now_utc=now,
        )
        assert state is not None
        assert state.forecast_residual_slope_f_per_hr == pytest.approx(0.0, abs=0.05)

    def test_single_routine_yields_none_slope(self):
        now = datetime(2026, 4, 22, 18, 0, tzinfo=timezone.utc)
        forecast = self._forecast([24.0] * 24)
        history = [
            {"observed_at": now - timedelta(hours=1), "temp_f": 80.0, "is_speci": False},
        ]
        state = build_state_from_metars(
            "KLAX", history, forecast, bias_c=0.0, now_utc=now,
        )
        assert state is not None
        assert state.forecast_residual_slope_f_per_hr is None
        assert state.forecast_residual_count == 1

    def test_no_forecast_yields_none_slope(self):
        now = datetime(2026, 4, 22, 18, 0, tzinfo=timezone.utc)
        history = [
            {"observed_at": now - timedelta(hours=h), "temp_f": 80.0 - h, "is_speci": False}
            for h in range(5, -1, -1)
        ]
        state = build_state_from_metars(
            "KLAX", history, None, bias_c=0.0, now_utc=now,
        )
        assert state is not None
        assert state.forecast_residual_slope_f_per_hr is None
        assert state.forecast_residual_count == 0

    def test_speci_excluded_from_slope_fit(self):
        # Three routines with constant +1.8°F residual; one SPECI in the middle
        # with a wildly different value. Slope should still be ≈0.
        now = datetime(2026, 4, 22, 18, 0, tzinfo=timezone.utc)
        forecast = self._forecast([24.0] * 24)  # forecast = 75.2°F
        history = [
            {"observed_at": now - timedelta(hours=4), "temp_f": 77.0, "is_speci": False},
            {"observed_at": now - timedelta(hours=3), "temp_f": 95.0, "is_speci": True},
            {"observed_at": now - timedelta(hours=2), "temp_f": 77.0, "is_speci": False},
            {"observed_at": now - timedelta(hours=1), "temp_f": 77.0, "is_speci": False},
            {"observed_at": now, "temp_f": 77.0, "is_speci": False},
        ]
        state = build_state_from_metars(
            "KLAX", history, forecast, bias_c=0.0, now_utc=now,
        )
        assert state is not None
        assert state.forecast_residual_count == 4  # SPECI excluded
        assert state.forecast_residual_slope_f_per_hr == pytest.approx(0.0, abs=0.05)


class TestAggregationCache:
    """Module-level cache fed by aggregate_state, consumed by the fast-poll
    projection check. Lets fast-poll rebuild state from a fresh METAR
    without re-fetching forecast / bias / climate normals.
    """

    def test_get_returns_none_when_absent(self):
        from src.signals.state_aggregator import (
            clear_state_cache,
            get_cached_aggregation_inputs,
        )

        clear_state_cache()
        assert get_cached_aggregation_inputs("KJFK") is None

    def test_round_trip_within_window(self):
        from src.signals.state_aggregator import (
            CachedAggregationInputs,
            _state_cache,
            clear_state_cache,
            get_cached_aggregation_inputs,
        )

        clear_state_cache()
        now = datetime.now(timezone.utc)
        cached = CachedAggregationInputs(
            cached_at_utc=now, history=[], forecast=None, bias_c=1.0,
            climate_prior_mean_f=None, climate_prior_std_f=None,
        )
        _state_cache["KSFO"] = cached
        result = get_cached_aggregation_inputs("KSFO")
        assert result is cached

    def test_stale_entry_rejected(self):
        from src.signals.state_aggregator import (
            CachedAggregationInputs,
            _STATE_CACHE_MAX_AGE,
            _state_cache,
            clear_state_cache,
            get_cached_aggregation_inputs,
        )

        clear_state_cache()
        # 1 second past the max age → rejected.
        stale_at = datetime.now(timezone.utc) - _STATE_CACHE_MAX_AGE - timedelta(seconds=1)
        _state_cache["KSEA"] = CachedAggregationInputs(
            cached_at_utc=stale_at, history=[], forecast=None, bias_c=0.0,
            climate_prior_mean_f=None, climate_prior_std_f=None,
        )
        assert get_cached_aggregation_inputs("KSEA") is None

    def test_clear_state_cache_drops_all(self):
        from src.signals.state_aggregator import (
            CachedAggregationInputs,
            _state_cache,
            clear_state_cache,
            get_cached_aggregation_inputs,
        )

        now = datetime.now(timezone.utc)
        _state_cache["KORD"] = CachedAggregationInputs(
            cached_at_utc=now, history=[], forecast=None, bias_c=0.5,
            climate_prior_mean_f=None, climate_prior_std_f=None,
        )
        clear_state_cache()
        assert get_cached_aggregation_inputs("KORD") is None
        assert _state_cache == {}
