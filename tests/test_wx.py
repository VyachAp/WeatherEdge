"""Tests for Weather Company v3 observation ingestion module."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from src.ingestion.wx import (
    ThresholdEvent,
    TrendAnalysis,
    WxObservation,
    _buffer_append,
    _c_to_f,
    _compute_rate,
    _f_to_c,
    _is_us_station,
    _observation_buffer,
    _parse_observation,
    analyze_trend,
    clear_buffers,
    detect_threshold_events,
    fetch_wx_current,
    get_buffer_history,
)


# ---------------------------------------------------------------------------
# Sample API response (Munich EDDM)
# ---------------------------------------------------------------------------

SAMPLE_WX_RESPONSE = {
    "cloudCeiling": None,
    "cloudCover": 30,
    "cloudCoverPhrase": "Partly Cloudy",
    "dayOfWeek": "Friday",
    "dayOrNight": "D",
    "expirationTimeUtc": 1776438714,
    "iconCode": 34,
    "iconCodeExtend": 3400,
    "obsQualifierCode": None,
    "obsQualifierSeverity": None,
    "precip1Hour": 0.0,
    "precip6Hour": 0.0,
    "precip24Hour": 0.0,
    "pressureAltimeter": 1019.98,
    "pressureChange": -1.02,
    "pressureMeanSeaLevel": 1020.1,
    "pressureTendencyCode": 2,
    "pressureTendencyTrend": "Falling",
    "relativeHumidity": 38,
    "snow1Hour": 0.0,
    "snow6Hour": 0.0,
    "snow24Hour": 0.0,
    "sunriseTimeLocal": "2026-04-17T06:19:03+0200",
    "sunriseTimeUtc": 1776399543,
    "sunsetTimeLocal": "2026-04-17T20:06:26+0200",
    "sunsetTimeUtc": 1776449186,
    "temperature": 20,
    "temperatureChange24Hour": 2,
    "temperatureDewPoint": 5,
    "temperatureFeelsLike": 20,
    "temperatureHeatIndex": 20,
    "temperatureMax24Hour": 21,
    "temperatureMaxSince7Am": 21,
    "temperatureMin24Hour": 3,
    "temperatureWetBulbGlobe": 65,
    "temperatureWindChill": 20,
    "uvDescription": "Low",
    "uvIndex": 1,
    "validTimeLocal": "2026-04-17 17:01:54",
    "validTimeUtc": 1776438114,
    "visibility": 9.66,
    "windDirection": 100,
    "windDirectionCardinal": "E",
    "windGust": None,
    "windSpeed": 10,
    "wxPhraseLong": "Fair",
    "wxPhraseMedium": "Fair",
    "wxPhraseShort": "Fair",
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_obs(
    icao: str = "EDDM",
    temp_c: float = 20.0,
    minutes_ago: int = 0,
    valid_time_local: str | None = None,
    units: str = "m",
    **kwargs,
) -> WxObservation:
    """Helper to create a WxObservation with defaults."""
    vt = _now() - timedelta(minutes=minutes_ago)
    vtl = valid_time_local or vt.strftime("2026-04-17 %H:%M:%S")
    return WxObservation(
        station_icao=icao,
        valid_time_utc=vt,
        valid_time_local=vtl,
        units=units,
        temp_c=temp_c,
        dewpoint_c=kwargs.get("dewpoint_c", 5.0),
        humidity=kwargs.get("humidity", 38),
        wind_speed_ms=kwargs.get("wind_speed_ms", 10.0),
        wind_gust_ms=kwargs.get("wind_gust_ms"),
        wind_dir=kwargs.get("wind_dir", 100),
        pressure_hpa=kwargs.get("pressure_hpa", 1020.1),
        pressure_trend=kwargs.get("pressure_trend", "Falling"),
        precip_1h_mm=kwargs.get("precip_1h_mm", 0.0),
        precip_6h_mm=kwargs.get("precip_6h_mm", 0.0),
        precip_24h_mm=kwargs.get("precip_24h_mm", 0.0),
        snow_1h_mm=kwargs.get("snow_1h_mm", 0.0),
        snow_24h_mm=kwargs.get("snow_24h_mm", 0.0),
        temp_max_since_7am_c=kwargs.get("temp_max_since_7am_c", 21.0),
        temp_max_24h_c=kwargs.get("temp_max_24h_c", 21.0),
        temp_min_24h_c=kwargs.get("temp_min_24h_c", 3.0),
        cloud_cover=kwargs.get("cloud_cover", 30),
        visibility_km=kwargs.get("visibility_km", 9.66),
        uv_index=kwargs.get("uv_index", 1),
    )


@pytest.fixture(autouse=True)
def _clear_state():
    clear_buffers()
    yield
    clear_buffers()


# ===================================================================
# Parsing
# ===================================================================


class TestParsing:
    def test_parse_v3_response(self):
        obs = _parse_observation("EDDM", SAMPLE_WX_RESPONSE)
        assert obs is not None
        assert obs.station_icao == "EDDM"
        assert obs.units == "m"
        assert obs.temp_c == 20
        assert obs.humidity == 38
        assert obs.wind_speed_ms == 10
        assert obs.pressure_hpa == 1020.1
        assert obs.pressure_trend == "Falling"
        assert obs.precip_1h_mm == 0.0
        assert obs.temp_max_since_7am_c == 21
        assert obs.temp_min_24h_c == 3
        assert obs.cloud_cover == 30
        assert obs.valid_time_local == "2026-04-17 17:01:54"

    def test_parse_missing_valid_time_returns_none(self):
        data = {**SAMPLE_WX_RESPONSE}
        del data["validTimeUtc"]
        assert _parse_observation("EDDM", data) is None

    def test_parse_missing_valid_local_returns_none(self):
        data = {**SAMPLE_WX_RESPONSE}
        del data["validTimeLocal"]
        assert _parse_observation("EDDM", data) is None

    def test_temp_f_property(self):
        obs = _make_obs(temp_c=20.0)
        assert obs.temp_f == pytest.approx(68.0)

    def test_temp_f_none(self):
        obs = _make_obs(temp_c=None)
        assert obs.temp_f is None


# ===================================================================
# Buffer & dedup
# ===================================================================


class TestBuffer:
    def test_append_new_observation(self):
        obs = _make_obs(valid_time_local="2026-04-17 17:00:00")
        assert _buffer_append(obs) is True
        assert len(get_buffer_history("EDDM")) == 1

    def test_dedup_same_valid_time(self):
        obs1 = _make_obs(valid_time_local="2026-04-17 17:00:00")
        obs2 = _make_obs(valid_time_local="2026-04-17 17:00:00", temp_c=21.0)
        assert _buffer_append(obs1) is True
        assert _buffer_append(obs2) is False  # Duplicate
        assert len(get_buffer_history("EDDM")) == 1

    def test_different_valid_times_accepted(self):
        obs1 = _make_obs(valid_time_local="2026-04-17 17:00:00")
        obs2 = _make_obs(valid_time_local="2026-04-17 17:05:00")
        assert _buffer_append(obs1) is True
        assert _buffer_append(obs2) is True
        assert len(get_buffer_history("EDDM")) == 2

    def test_different_stations_independent(self):
        obs1 = _make_obs(icao="EDDM", valid_time_local="2026-04-17 17:00:00")
        obs2 = _make_obs(icao="KAUS", valid_time_local="2026-04-17 17:00:00")
        _buffer_append(obs1)
        _buffer_append(obs2)
        assert len(get_buffer_history("EDDM")) == 1
        assert len(get_buffer_history("KAUS")) == 1

    def test_empty_buffer(self):
        assert get_buffer_history("NONEXISTENT") == []

    def test_buffer_count_parameter(self):
        for i in range(10):
            _buffer_append(_make_obs(valid_time_local=f"2026-04-17 17:{i:02d}:00"))
        assert len(get_buffer_history("EDDM", 3)) == 3
        assert len(get_buffer_history("EDDM")) == 10


# ===================================================================
# Rate computation
# ===================================================================


class TestRate:
    def test_rising_rate(self):
        history = [
            _make_obs(temp_c=20.0, minutes_ago=20),
            _make_obs(temp_c=21.0, minutes_ago=15),
            _make_obs(temp_c=22.0, minutes_ago=10),
            _make_obs(temp_c=23.0, minutes_ago=5),
            _make_obs(temp_c=24.0, minutes_ago=0),
        ]
        rate = _compute_rate(history)
        assert rate is not None
        assert rate > 0  # Rising

    def test_steady_rate(self):
        history = [
            _make_obs(temp_c=20.0, minutes_ago=20),
            _make_obs(temp_c=20.0, minutes_ago=10),
            _make_obs(temp_c=20.0, minutes_ago=0),
        ]
        rate = _compute_rate(history)
        assert rate is not None
        assert abs(rate) < 0.1

    def test_insufficient_data(self):
        assert _compute_rate([_make_obs()]) is None
        assert _compute_rate([]) is None


# ===================================================================
# Trend analysis
# ===================================================================


class TestTrendAnalysis:
    def test_rising_trend(self):
        for i in range(6):
            _buffer_append(_make_obs(
                temp_c=18.0 + i * 0.5,
                valid_time_local=f"2026-04-17 12:{i * 5:02d}:00",
                minutes_ago=25 - i * 5,
            ))

        trend = analyze_trend("EDDM")
        assert trend is not None
        assert trend.is_rising is True
        assert trend.is_falling is False
        assert trend.observed_max_f > trend.observed_min_f

    def test_falling_trend_peak_detection(self):
        # Build buffer: temp rose then fell for 20 minutes
        readings = [
            (18.0, 50, "11:10:00"),
            (19.0, 45, "11:15:00"),
            (20.0, 40, "11:20:00"),  # peak
            (20.0, 35, "11:25:00"),
            (19.5, 30, "11:30:00"),
            (19.0, 25, "11:35:00"),
            (18.5, 20, "11:40:00"),
            (18.0, 15, "11:45:00"),
        ]
        for temp, mins_ago, time_str in readings:
            _buffer_append(_make_obs(
                temp_c=temp,
                minutes_ago=mins_ago,
                valid_time_local=f"2026-04-17 {time_str}",
                pressure_trend="Falling",
                humidity=45 + (50 - mins_ago),  # humidity rising
                wind_speed_ms=5.0 + (50 - mins_ago) * 0.1,  # wind increasing
            ))

        trend = analyze_trend("EDDM")
        assert trend is not None
        assert trend.is_falling is True
        assert trend.observed_max_f == pytest.approx(_c_to_f(20.0))
        assert trend.minutes_since_last_rise > 10

    def test_insufficient_data_returns_none(self):
        _buffer_append(_make_obs(valid_time_local="2026-04-17 17:00:00"))
        assert analyze_trend("EDDM") is None

    def test_empty_station_returns_none(self):
        assert analyze_trend("NONEXISTENT") is None


# ===================================================================
# Threshold events
# ===================================================================


class TestThresholdEvents:
    def test_threshold_crossed_above(self):
        # Temp clearly above 68F (20C)
        for i in range(5):
            _buffer_append(_make_obs(
                temp_c=21.0 + i * 0.2,
                valid_time_local=f"2026-04-17 14:{i * 5:02d}:00",
                minutes_ago=20 - i * 5,
            ))

        events = detect_threshold_events("EDDM", [(68.0, "above")])
        crossed = [e for e in events if e.event_type == "threshold_crossed"]
        assert len(crossed) == 1
        assert crossed[0].confidence >= 0.85

    def test_no_event_when_below_threshold(self):
        for i in range(5):
            _buffer_append(_make_obs(
                temp_c=15.0,
                valid_time_local=f"2026-04-17 14:{i * 5:02d}:00",
                minutes_ago=20 - i * 5,
            ))

        events = detect_threshold_events("EDDM", [(68.0, "above")])
        assert len(events) == 0

    def test_threshold_crossed_below(self):
        for i in range(5):
            _buffer_append(_make_obs(
                temp_c=0.0 - i * 0.5,
                valid_time_local=f"2026-04-17 04:{i * 5:02d}:00",
                minutes_ago=20 - i * 5,
            ))

        events = detect_threshold_events("EDDM", [(32.0, "below")])
        crossed = [e for e in events if e.event_type == "threshold_crossed"]
        assert len(crossed) == 1

    def test_no_events_insufficient_data(self):
        events = detect_threshold_events("NONEXISTENT", [(68.0, "above")])
        assert events == []


# ===================================================================
# Fetch (mocked HTTP)
# ===================================================================


class TestFetch:
    @pytest.mark.asyncio
    @patch("src.ingestion.wx.settings")
    @patch("src.ingestion.wx._wx_get_json", new_callable=AsyncMock)
    async def test_fetch_success(self, mock_get, mock_settings):
        mock_settings.WX_API_KEY = "test-key"
        mock_get.return_value = SAMPLE_WX_RESPONSE

        from src.ingestion.wx import _wx_cache
        _wx_cache.clear()

        obs = await fetch_wx_current("EDDM")
        assert obs is not None
        assert obs.station_icao == "EDDM"
        assert obs.temp_c == 20

    @pytest.mark.asyncio
    @patch("src.ingestion.wx.settings")
    async def test_fetch_disabled_without_key(self, mock_settings):
        mock_settings.WX_API_KEY = ""
        obs = await fetch_wx_current("EDDM")
        assert obs is None

    @pytest.mark.asyncio
    @patch("src.ingestion.wx.settings")
    @patch("src.ingestion.wx._wx_get_json", new_callable=AsyncMock)
    async def test_fetch_network_error(self, mock_get, mock_settings):
        import httpx
        mock_settings.WX_API_KEY = "test-key"
        mock_get.side_effect = httpx.TransportError("timeout")

        from src.ingestion.wx import _wx_cache
        _wx_cache.clear()

        obs = await fetch_wx_current("EDDM")
        assert obs is None


# ===================================================================
# Unit conversion
# ===================================================================


class TestConversion:
    def test_c_to_f(self):
        assert _c_to_f(0) == 32.0
        assert _c_to_f(100) == 212.0
        assert _c_to_f(20) == pytest.approx(68.0)
        assert _c_to_f(None) is None

    def test_f_to_c(self):
        assert _f_to_c(32.0) == pytest.approx(0.0)
        assert _f_to_c(212.0) == pytest.approx(100.0)
        assert _f_to_c(68.0) == pytest.approx(20.0)
        assert _f_to_c(None) is None


# ===================================================================
# US station detection
# ===================================================================


class TestIsUsStation:
    def test_k_prefix_conus(self):
        assert _is_us_station("KJFK") is True
        assert _is_us_station("KORD") is True
        assert _is_us_station("KLAX") is True

    def test_p_prefix_pacific(self):
        assert _is_us_station("PHNL") is True
        assert _is_us_station("PANC") is True

    def test_international_stations(self):
        assert _is_us_station("EDDM") is False
        assert _is_us_station("EGLL") is False
        assert _is_us_station("LFPG") is False
        assert _is_us_station("RKSI") is False

    def test_short_codes(self):
        assert _is_us_station("KJF") is False
        assert _is_us_station("") is False


# ===================================================================
# Imperial units handling
# ===================================================================


class TestImperialUnits:
    def test_temp_f_metric_converts(self):
        obs = _make_obs(temp_c=20.0, units="m")
        assert obs.temp_f == pytest.approx(68.0)

    def test_temp_f_imperial_no_conversion(self):
        obs = _make_obs(temp_c=68.0, units="e")  # temp_c holds °F
        assert obs.temp_f == 68.0  # No conversion applied

    def test_temp_f_none_both_units(self):
        assert _make_obs(temp_c=None, units="m").temp_f is None
        assert _make_obs(temp_c=None, units="e").temp_f is None

    def test_parse_with_imperial_units(self):
        obs = _parse_observation("KJFK", SAMPLE_WX_RESPONSE, units="e")
        assert obs is not None
        assert obs.units == "e"
        assert obs.temp_c == 20  # Raw API value stored as-is

    def test_parse_defaults_to_metric(self):
        obs = _parse_observation("EDDM", SAMPLE_WX_RESPONSE)
        assert obs is not None
        assert obs.units == "m"

    def test_trend_analysis_imperial(self):
        """Trend analysis with imperial observations should not double-convert."""
        for i in range(6):
            _buffer_append(_make_obs(
                icao="KJFK",
                temp_c=65.0 + i * 1.0,  # Already in °F
                valid_time_local=f"2026-04-17 12:{i * 5:02d}:00",
                minutes_ago=25 - i * 5,
                units="e",
            ))

        trend = analyze_trend("KJFK")
        assert trend is not None
        assert trend.is_rising is True
        # Values should be in °F range, not double-converted
        assert 60.0 < trend.current_temp_f < 80.0
        assert 60.0 < trend.observed_max_f < 80.0

    def test_compute_rate_imperial(self):
        """Rate computation with imperial data should produce sensible F/hour."""
        history = [
            _make_obs(icao="KJFK", temp_c=65.0, minutes_ago=20, units="e"),
            _make_obs(icao="KJFK", temp_c=66.0, minutes_ago=15, units="e"),
            _make_obs(icao="KJFK", temp_c=67.0, minutes_ago=10, units="e"),
            _make_obs(icao="KJFK", temp_c=68.0, minutes_ago=5, units="e"),
            _make_obs(icao="KJFK", temp_c=69.0, minutes_ago=0, units="e"),
        ]
        rate = _compute_rate(history)
        assert rate is not None
        assert rate > 0
        # 4°F over 20 min = 12°F/hour
        assert rate == pytest.approx(12.0, abs=1.0)
