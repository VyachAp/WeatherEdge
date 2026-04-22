"""Tests for projected-daily-max alerts / forecast-exceedance recording."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ingestion.openmeteo import OpenMeteoForecast
from src.signals.forecast_exceedance import (
    MAX_OVERSHOOT_F,
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
    metar_trend_rate_short: float = 0.0,
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
        metar_trend_rate_short=metar_trend_rate_short,
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

    def test_short_trend_drives_peak_passed(self):
        # 6h regression still positive because of the morning warm-up, but the
        # short-window trend has already turned negative — peak really has
        # passed, and the heuristic should recognise it via the short trend.
        assert _peak_passed(_state(
            current_max_f=70.0, forecast_peak_f=80.0,
            metar_trend_rate=0.5, metar_trend_rate_short=-0.2,
            solar_declining=True, hours_until_peak=2.0,
        ))

    def test_past_peak_hour_no_recent_rise_triggers_peak_passed(self):
        # Forecast peak hour already an hour and a half behind us and the short
        # trend is essentially flat → treat as peak passed regardless of the
        # 6h regression sign.
        assert _peak_passed(_state(
            current_max_f=72.0, forecast_peak_f=80.0,
            metar_trend_rate=0.3, metar_trend_rate_short=0.05,
            hours_until_peak=-1.5,
        ))


class TestProjectDailyMax:
    def test_monotonic_floor(self):
        # Falling trend never drops projection below observed max.
        s = _state(current_max_f=78.0, metar_trend_rate=-1.0, hours_until_peak=2.0)
        assert _project_daily_max(s) == 78.0

    def test_rising_trend_extrapolates(self):
        # trend=+1°F/hr, h=2 → extrapolated=78 → blended with forecast_peak=80
        # at α=exp(-1)=0.368: 0.368·78 + 0.632·80 ≈ 79.26.
        s = _state(
            current_max_f=76.0, metar_trend_rate=1.0,
            forecast_peak_f=80.0, hours_until_peak=2.0,
        )
        assert _project_daily_max(s) == pytest.approx(79.26, abs=0.05)

    def test_far_from_peak_blends_toward_forecast(self):
        # hours_until_peak=6 → extrapolation cap at 3h, α=exp(-3)=0.05 → projection
        # overwhelmingly weighted toward forecast_peak (80°F) regardless of trend.
        s = _state(
            current_max_f=76.0, metar_trend_rate=2.0,
            forecast_peak_f=80.0, hours_until_peak=6.0,
        )
        result = _project_daily_max(s)
        assert abs(result - 80.0) < 0.5

    def test_solar_decline_damps_extrapolation(self):
        # 50% solar decline halves extrapolation hours (2h → 1h effective):
        # extrapolated = 76 + 1·1 = 77; projection = 0.368·77 + 0.632·80 ≈ 78.90.
        # Compare with the no-damping case (79.26) to verify the damping path.
        damped = _state(
            current_max_f=76.0, metar_trend_rate=1.0,
            forecast_peak_f=80.0, hours_until_peak=2.0,
            solar_declining=True, solar_decline_magnitude=0.5,
        )
        undamped = _state(
            current_max_f=76.0, metar_trend_rate=1.0,
            forecast_peak_f=80.0, hours_until_peak=2.0,
        )
        assert _project_daily_max(damped) == pytest.approx(78.90, abs=0.05)
        assert _project_daily_max(damped) < _project_daily_max(undamped)

    def test_rising_dewpoint_nudges_down(self):
        # Blend projection 79.26 minus 0.5°F dewpoint nudge → 78.76.
        s = _state(
            current_max_f=76.0, metar_trend_rate=1.0,
            forecast_peak_f=80.0, hours_until_peak=2.0,
            dewpoint_trend_rate=1.5,
        )
        assert _project_daily_max(s) == pytest.approx(78.76, abs=0.05)

    def test_past_peak_no_extrapolation(self):
        s = _state(current_max_f=78.0, metar_trend_rate=1.0, hours_until_peak=0.0)
        assert _project_daily_max(s) == 78.0

    def test_short_trend_preferred_when_nonzero(self):
        # 6h trend +2°F/hr but short-window trend only +0.2°F/hr (curve bending
        # near peak): the projection should use the short trend, not the 6h one.
        s = _state(
            current_max_f=76.0, metar_trend_rate=2.0, metar_trend_rate_short=0.2,
            forecast_peak_f=80.0, hours_until_peak=2.0,
        )
        # extrapolated = 76 + 0.2·2 = 76.4; α=0.368 → 0.368·76.4 + 0.632·80 ≈ 78.67.
        assert _project_daily_max(s) == pytest.approx(78.67, abs=0.05)

    def test_plausibility_cap_clips_overshoot(self):
        # Absurdly high trend blown up by extrapolation must be clipped at
        # forecast_peak + MAX_OVERSHOOT_F (5°F) so we never project the Sun.
        s = _state(
            current_max_f=78.0, metar_trend_rate=20.0,
            forecast_peak_f=80.0, hours_until_peak=3.0,
        )
        assert _project_daily_max(s) == pytest.approx(80.0 + MAX_OVERSHOOT_F)

    def test_blend_tapers_with_distance_to_peak(self):
        # Same obs/trend/forecast, different hours_until_peak. Far-out projection
        # should sit closer to forecast_peak than the near-peak one.
        near = _state(
            current_max_f=76.0, metar_trend_rate=2.0,
            forecast_peak_f=80.0, hours_until_peak=1.0,
        )
        far = _state(
            current_max_f=76.0, metar_trend_rate=2.0,
            forecast_peak_f=80.0, hours_until_peak=5.0,
        )
        near_proj = _project_daily_max(near)
        far_proj = _project_daily_max(far)
        assert abs(far_proj - 80.0) < abs(near_proj - 80.0)


class _FakeResult:
    def __init__(self, value):
        self._value = value

    def scalar(self):
        return self._value


class _FakeSession:
    def __init__(self, existing_id=None, commit_raises=None, cooldown_id=None):
        self.added: list = []
        self._existing_id = existing_id
        self._cooldown_id = cooldown_id
        self._commit_raises = commit_raises
        self.committed = False
        self.rolled_back = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def execute(self, stmt):
        # Cooldown query filters on alerted_at; existence check does not.
        if "alerted_at" in str(stmt):
            return _FakeResult(self._cooldown_id)
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
        # trend +3°F/hr, peak 1.5h out: extrapolated 83, α≈0.47 → blend ~80.9,
        # delta ~1.9°F > 1°F threshold.
        state = _state(
            current_max_f=78.5, metar_trend_rate=3.0,
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
        # US station → message text uses °F.
        sent = alerter._enqueue.await_args.args[0]
        assert "°F" in sent
        assert "°C" not in sent
        assert "KLAX" in sent

    @pytest.mark.asyncio
    async def test_celsius_station_message_uses_celsius(self):
        # RKPK (Busan) is non-K → Celsius. Same scenario shape as the °F push test,
        # but with state values aligned to a Celsius-market city to verify the
        # alert text reports °C, °C/hr, etc.
        observed_at = datetime(2026, 4, 22, 18, 53, tzinfo=timezone.utc)
        history = [_metar(observed_at, 66.2)]  # 19°C — beats forecast at this hour
        forecast = _make_forecast([16.0] * 24)  # 16°C ≈ 60.8°F same-hour
        state = _state(
            current_max_f=66.2,             # 19.0°C
            metar_trend_rate=1.8,           # 1.0°C/hr
            forecast_peak_f=64.4,           # 18.0°C
            hours_until_peak=0.8,
            routine_count_today=5,
            icao="RKPK",
        )

        fake_session = _FakeSession(existing_id=None)
        alerter = MagicMock()
        alerter._enqueue = AsyncMock()

        with patch("src.signals.forecast_exceedance.async_session", return_value=fake_session), \
             patch("src.signals.forecast_exceedance.get_alerter", return_value=alerter):
            await check_and_record_daily_max_alert("RKPK", state, history, forecast)

        # Recording row stays in canonical °F (DB schema).
        row = fake_session.added[0]
        assert row.current_max_f == pytest.approx(66.2)
        assert row.forecast_peak_f == pytest.approx(64.4)

        alerter._enqueue.assert_awaited_once()
        sent = alerter._enqueue.await_args.args[0]
        # Strip MarkdownV2 backslash escapes for value assertions.
        unescaped = sent.replace("\\", "")
        assert "RKPK" in unescaped
        assert "°C" in unescaped
        assert "°F" not in unescaped
        # Spot-check converted values: 66.2°F → 19.0°C, 64.4°F → 18.0°C.
        assert "Obs max: 19.0°C" in unescaped
        assert "Forecast peak: 18.0°C" in unescaped
        # Trend 1.8°F/hr → 1.0°C/hr (rendered "+1.0°C/hr").
        assert "+1.0°C/hr" in unescaped

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

    @pytest.mark.asyncio
    async def test_cooldown_suppresses_second_push_within_60min(self):
        # Same push-happy scenario as test_pushes_when_projection_beats_forecast,
        # but the cooldown query returns a recent alerted row. Expectation: DB
        # row still written (alerted=False), but no Telegram enqueue.
        observed_at = datetime(2026, 4, 22, 18, 53, tzinfo=timezone.utc)
        history = [_metar(observed_at, 78.5)]
        forecast = _make_forecast([24.0] * 24)
        state = _state(
            current_max_f=78.5, metar_trend_rate=3.0,
            forecast_peak_f=79.0, hours_until_peak=1.5,
            routine_count_today=4,
        )

        fake_session = _FakeSession(existing_id=None, cooldown_id=123)
        alerter = MagicMock()
        alerter._enqueue = AsyncMock()

        with patch("src.signals.forecast_exceedance.async_session", return_value=fake_session), \
             patch("src.signals.forecast_exceedance.get_alerter", return_value=alerter):
            await check_and_record_daily_max_alert("KLAX", state, history, forecast)

        assert len(fake_session.added) == 1
        assert fake_session.added[0].alerted is False
        assert fake_session.committed is True
        alerter._enqueue.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cooldown_allows_push_when_no_recent_alert(self):
        # Push-happy scenario, cooldown query returns no recent alerted row.
        observed_at = datetime(2026, 4, 22, 18, 53, tzinfo=timezone.utc)
        history = [_metar(observed_at, 78.5)]
        forecast = _make_forecast([24.0] * 24)
        state = _state(
            current_max_f=78.5, metar_trend_rate=3.0,
            forecast_peak_f=79.0, hours_until_peak=1.5,
            routine_count_today=4,
        )

        fake_session = _FakeSession(existing_id=None, cooldown_id=None)
        alerter = MagicMock()
        alerter._enqueue = AsyncMock()

        with patch("src.signals.forecast_exceedance.async_session", return_value=fake_session), \
             patch("src.signals.forecast_exceedance.get_alerter", return_value=alerter):
            await check_and_record_daily_max_alert("KLAX", state, history, forecast)

        assert len(fake_session.added) == 1
        assert fake_session.added[0].alerted is True
        alerter._enqueue.assert_awaited_once()
