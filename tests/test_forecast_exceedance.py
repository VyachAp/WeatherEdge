"""Tests for projected-daily-max alerts / forecast-exceedance recording."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ingestion.openmeteo import OpenMeteoForecast
from src.signals.forecast_exceedance import (
    _closest_hour_index,
    _peak_passed,
    _pick_latest_routine,
    _project_daily_max,
    check_and_record_daily_max_alert,
)
from src.signals.state_aggregator import WeatherState


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


def _state(
    *,
    current_max_f: float = 78.0,
    metar_trend_rate: float = 0.0,
    dewpoint_trend_rate: float = 0.0,
    forecast_peak_f: float = 78.0,
    hours_until_peak: float = 0.0,
    solar_declining: bool = False,
    solar_decline_magnitude: float = 0.0,
    cloud_rising: bool = False,
    cloud_rise_magnitude: float = 0.0,
    routine_count_today: int = 4,
    icao: str = "KLAX",
) -> WeatherState:
    return WeatherState(
        station_icao=icao,
        current_max_f=current_max_f,
        metar_trend_rate=metar_trend_rate,
        dewpoint_trend_rate=dewpoint_trend_rate,
        forecast_peak_f=forecast_peak_f,
        hours_until_peak=hours_until_peak,
        solar_declining=solar_declining,
        solar_decline_magnitude=solar_decline_magnitude,
        cloud_rising=cloud_rising,
        cloud_rise_magnitude=cloud_rise_magnitude,
        routine_count_today=routine_count_today,
    )


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


class TestPeakPassed:
    def test_observed_at_forecast_and_cooling(self):
        # current_max within tolerance of forecast peak, trend flat/falling.
        assert _peak_passed(_state(current_max_f=77.6, forecast_peak_f=78.0, metar_trend_rate=-0.1))

    def test_past_peak_hour_and_cooling(self):
        assert _peak_passed(_state(hours_until_peak=0.0, metar_trend_rate=0.0))

    def test_solar_declining_and_cooling(self):
        assert _peak_passed(_state(
            solar_declining=True, metar_trend_rate=-0.2,
            # Force other branches false: keep obs well below forecast peak, keep hours>0.
            current_max_f=70.0, forecast_peak_f=80.0, hours_until_peak=2.0,
        ))

    def test_rising_trend_never_passed(self):
        # Strong upward trend blocks all three branches.
        assert not _peak_passed(_state(
            current_max_f=78.0, forecast_peak_f=78.0,
            metar_trend_rate=0.5, hours_until_peak=0.0,
            solar_declining=True,
        ))

    def test_cool_morning_not_passed(self):
        # Morning: obs below forecast, peak still ahead, sun climbing.
        assert not _peak_passed(_state(
            current_max_f=65.0, forecast_peak_f=80.0,
            metar_trend_rate=1.5, hours_until_peak=4.0,
        ))


class TestProjectDailyMax:
    def test_monotonic_floor(self):
        # Falling trend never drops projection below observed max.
        s = _state(current_max_f=78.0, metar_trend_rate=-1.0, hours_until_peak=2.0)
        assert _project_daily_max(s) == 78.0

    def test_rising_trend_extrapolates(self):
        # +1°F/hr for 2h → +2°F.
        s = _state(current_max_f=78.0, metar_trend_rate=1.0, hours_until_peak=2.0)
        assert _project_daily_max(s) == pytest.approx(80.0)

    def test_extrapolation_capped_at_three_hours(self):
        # hours_until_peak=6 should cap at 3h → +3°F.
        s = _state(current_max_f=78.0, metar_trend_rate=1.0, hours_until_peak=6.0)
        assert _project_daily_max(s) == pytest.approx(81.0)

    def test_solar_decline_damps_extrapolation(self):
        # 50% solar decline → 1h of 2h extrapolation kept → +1°F.
        s = _state(
            current_max_f=78.0, metar_trend_rate=1.0, hours_until_peak=2.0,
            solar_declining=True, solar_decline_magnitude=0.5,
        )
        assert _project_daily_max(s) == pytest.approx(79.0)

    def test_rising_dewpoint_nudges_down(self):
        # Base extrapolation +2°F, dewpoint rising >1°F/hr → -0.5°F nudge.
        s = _state(
            current_max_f=78.0, metar_trend_rate=1.0, hours_until_peak=2.0,
            dewpoint_trend_rate=1.5,
        )
        assert _project_daily_max(s) == pytest.approx(79.5)

    def test_past_peak_no_extrapolation(self):
        s = _state(current_max_f=78.0, metar_trend_rate=1.0, hours_until_peak=0.0)
        assert _project_daily_max(s) == 78.0


class _FakeResult:
    def __init__(self, value):
        self._value = value

    def scalar(self):
        return self._value


class _FakeSession:
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


class TestCheckAndRecordDailyMaxAlert:
    @pytest.mark.asyncio
    async def test_pushes_when_projection_beats_forecast(self):
        observed_at = datetime(2026, 4, 22, 18, 53, tzinfo=timezone.utc)
        # Morning/early afternoon: obs 78.5°F at hour 19, forecast 24°C (~75.2°F).
        # Same-hour delta +3.3°F passes recording gate.
        history = [_metar(observed_at, 78.5)]
        forecast = _make_forecast([24.0] * 24)
        # State: trend +1.5°F/hr, peak 1.5h out → projection 80.75, delta +1.75 > 1°F.
        state = _state(
            current_max_f=78.5, metar_trend_rate=1.5,
            forecast_peak_f=79.0, hours_until_peak=1.5,
            routine_count_today=4,
        )

        fake_session = _FakeSession(existing_id=None)
        alerter = MagicMock()
        alerter._enqueue = AsyncMock()

        with patch("src.signals.forecast_exceedance.async_session", return_value=fake_session), \
             patch("src.signals.forecast_exceedance.get_alerter", return_value=alerter):
            await check_and_record_daily_max_alert("KLAX", state, history, forecast)

        assert len(fake_session.added) == 1
        row = fake_session.added[0]
        assert row.station_icao == "KLAX"
        assert row.peak_passed is False
        assert row.alerted is True
        assert row.current_max_f == pytest.approx(78.5)
        assert row.forecast_peak_f == pytest.approx(79.0)
        assert row.projected_max_f > 79.0
        assert fake_session.committed is True
        alerter._enqueue.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_peak_passed_records_but_no_push(self):
        observed_at = datetime(2026, 4, 22, 20, 53, tzinfo=timezone.utc)
        history = [_metar(observed_at, 79.0)]
        forecast = _make_forecast([24.0] * 24)  # same-hour ~75.2°F, delta +3.8°F
        # Observed matches forecast peak, trend zero → peak_passed branch 1 trips.
        state = _state(
            current_max_f=79.0, forecast_peak_f=79.0,
            metar_trend_rate=0.0, hours_until_peak=0.0,
            routine_count_today=6,
        )

        fake_session = _FakeSession(existing_id=None)
        alerter = MagicMock()
        alerter._enqueue = AsyncMock()

        with patch("src.signals.forecast_exceedance.async_session", return_value=fake_session), \
             patch("src.signals.forecast_exceedance.get_alerter", return_value=alerter):
            await check_and_record_daily_max_alert("KLAX", state, history, forecast)

        assert len(fake_session.added) == 1
        row = fake_session.added[0]
        assert row.peak_passed is True
        assert row.alerted is False
        assert fake_session.committed is True
        alerter._enqueue.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_routine_count_below_minimum_no_push(self):
        observed_at = datetime(2026, 4, 22, 18, 53, tzinfo=timezone.utc)
        history = [_metar(observed_at, 78.5)]
        forecast = _make_forecast([24.0] * 24)
        state = _state(
            current_max_f=78.5, metar_trend_rate=1.0,
            forecast_peak_f=79.0, hours_until_peak=1.5,
            routine_count_today=2,  # below MIN_ROUTINE_COUNT_FOR_PUSH (3)
        )

        fake_session = _FakeSession(existing_id=None)
        alerter = MagicMock()
        alerter._enqueue = AsyncMock()

        with patch("src.signals.forecast_exceedance.async_session", return_value=fake_session), \
             patch("src.signals.forecast_exceedance.get_alerter", return_value=alerter):
            await check_and_record_daily_max_alert("KLAX", state, history, forecast)

        assert len(fake_session.added) == 1
        assert fake_session.added[0].alerted is False
        alerter._enqueue.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_projection_below_threshold_no_push(self):
        observed_at = datetime(2026, 4, 22, 18, 53, tzinfo=timezone.utc)
        history = [_metar(observed_at, 78.5)]
        forecast = _make_forecast([24.0] * 24)
        # Projection equals forecast_peak → delta 0, below 1°F threshold.
        state = _state(
            current_max_f=78.5, forecast_peak_f=78.5,
            metar_trend_rate=0.0, hours_until_peak=2.0,
            routine_count_today=4,
        )

        fake_session = _FakeSession(existing_id=None)
        alerter = MagicMock()
        alerter._enqueue = AsyncMock()

        with patch("src.signals.forecast_exceedance.async_session", return_value=fake_session), \
             patch("src.signals.forecast_exceedance.get_alerter", return_value=alerter):
            await check_and_record_daily_max_alert("KLAX", state, history, forecast)

        assert len(fake_session.added) == 1
        assert fake_session.added[0].alerted is False
        alerter._enqueue.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_same_hour_delta_below_threshold_skips_entirely(self):
        observed_at = datetime(2026, 4, 22, 14, 53, tzinfo=timezone.utc)
        # Forecast ~75.2°F, obs 75.5°F → delta 0.3°F, below recording gate.
        history = [_metar(observed_at, 75.5)]
        forecast = _make_forecast([24.0] * 24)
        state = _state(current_max_f=75.5, forecast_peak_f=80.0, metar_trend_rate=1.0,
                       hours_until_peak=3.0, routine_count_today=4)

        fake_session = _FakeSession()
        alerter = MagicMock()
        alerter._enqueue = AsyncMock()

        with patch("src.signals.forecast_exceedance.async_session", return_value=fake_session), \
             patch("src.signals.forecast_exceedance.get_alerter", return_value=alerter):
            await check_and_record_daily_max_alert("KLAX", state, history, forecast)

        assert fake_session.added == []
        alerter._enqueue.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dedup_skips_when_row_exists(self):
        observed_at = datetime(2026, 4, 22, 14, 53, tzinfo=timezone.utc)
        history = [_metar(observed_at, 80.0)]
        forecast = _make_forecast([24.0] * 24)
        state = _state(current_max_f=80.0, forecast_peak_f=78.0,
                       metar_trend_rate=1.0, hours_until_peak=2.0,
                       routine_count_today=4)

        fake_session = _FakeSession(existing_id=42)
        alerter = MagicMock()
        alerter._enqueue = AsyncMock()

        with patch("src.signals.forecast_exceedance.async_session", return_value=fake_session), \
             patch("src.signals.forecast_exceedance.get_alerter", return_value=alerter):
            await check_and_record_daily_max_alert("KLAX", state, history, forecast)

        assert fake_session.added == []
        assert fake_session.committed is False
        alerter._enqueue.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_speci_only_history_no_op(self):
        observed_at = datetime(2026, 4, 22, 14, 53, tzinfo=timezone.utc)
        history = [_metar(observed_at, 90.0, is_speci=True)]
        forecast = _make_forecast([24.0] * 24)
        state = _state()

        fake_session = _FakeSession()
        alerter = MagicMock()
        alerter._enqueue = AsyncMock()

        with patch("src.signals.forecast_exceedance.async_session", return_value=fake_session), \
             patch("src.signals.forecast_exceedance.get_alerter", return_value=alerter):
            await check_and_record_daily_max_alert("KLAX", state, history, forecast)

        assert fake_session.added == []
        alerter._enqueue.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_forecast_none_no_op(self):
        observed_at = datetime(2026, 4, 22, 14, 53, tzinfo=timezone.utc)
        history = [_metar(observed_at, 90.0)]
        state = _state()

        fake_session = _FakeSession()
        alerter = MagicMock()
        alerter._enqueue = AsyncMock()

        with patch("src.signals.forecast_exceedance.async_session", return_value=fake_session), \
             patch("src.signals.forecast_exceedance.get_alerter", return_value=alerter):
            await check_and_record_daily_max_alert("KLAX", state, history, None)

        assert fake_session.added == []
        alerter._enqueue.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_history_no_op(self):
        forecast = _make_forecast([24.0] * 24)
        state = _state()
        fake_session = _FakeSession()
        alerter = MagicMock()
        alerter._enqueue = AsyncMock()

        with patch("src.signals.forecast_exceedance.async_session", return_value=fake_session), \
             patch("src.signals.forecast_exceedance.get_alerter", return_value=alerter):
            await check_and_record_daily_max_alert("KLAX", state, [], forecast)

        assert fake_session.added == []
        alerter._enqueue.assert_not_awaited()
