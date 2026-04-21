"""Tests for aviation weather data fetcher."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ingestion.aviation import (
    _c_to_f,
    _find_taf_period,
    _parse_airsigmet_json,
    _parse_metar_json,
    _parse_pirep_json,
    _parse_taf_json,
    _point_in_polygon,
    clear_aviation_cache,
    detect_speci_events,
    fetch_active_airmets,
    fetch_active_sigmets,
    fetch_latest_metars,
    fetch_latest_tafs,
    fetch_metar_history,
    fetch_pireps_near,
    get_aviation_weather_picture,
    get_current_temp,
    get_taf_precip_probability,
    get_taf_temperature_forecast,
    get_temp_trend,
    has_severe_weather_reports,
    taf_amendment_count,
)

@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear the aviation TTL cache before each test."""
    clear_aviation_cache()
    yield
    clear_aviation_cache()


# ---------------------------------------------------------------------------
# Sample AWC API response fixtures
# ---------------------------------------------------------------------------

SAMPLE_METAR = {
    "icaoId": "KPHX",
    "obsTime": "2026-04-09T18:53:00Z",
    "temp": 32.2,
    "dwpt": 8.3,
    "wdir": 220,
    "wspd": 12,
    "wgst": 18,
    "visib": 10,
    "altim": 1013.2,
    "fltcat": "VFR",
    "metar_type": "METAR",
    "clouds": [
        {"cover": "FEW", "base": 12000},
        {"cover": "BKN", "base": 25000},
    ],
    "rawOb": "KPHX 091853Z 22012G18KT 10SM FEW120 BKN250 32/08 A2993",
}

SAMPLE_METAR_SPECI = {
    "icaoId": "KORD",
    "obsTime": "2026-04-09T15:30:00Z",
    "temp": 18.0,
    "dwpt": 12.0,
    "wdir": 180,
    "wspd": 25,
    "wgst": 35,
    "visib": 3,
    "altim": 1005.0,
    "fltcat": "IFR",
    "metar_type": "SPECI",
    "clouds": [{"cover": "OVC", "base": 800}],
    "rawOb": "SPECI KORD 091530Z 18025G35KT 3SM OVC008 18/12 A2967 RMK TSRA",
}

SAMPLE_METAR_MISSING = {
    "icaoId": "KZZZ",
    "obsTime": "2026-04-09T12:00:00Z",
    "temp": None,
    "dwpt": None,
    "wdir": None,
    "wspd": None,
    "visib": None,
    "altim": None,
    "fltcat": None,
    "metar_type": "METAR",
    "clouds": [],
    "rawOb": "KZZZ 091200Z AUTO /////KT ////",
}

SAMPLE_METAR_NEGATIVE_TEMP = {
    "icaoId": "PANC",
    "obsTime": "2026-01-15T06:00:00Z",
    "temp": -15.0,
    "dwpt": -20.0,
    "wdir": 10,
    "wspd": 8,
    "visib": 10,
    "altim": 1025.0,
    "fltcat": "VFR",
    "metar_type": "METAR",
    "clouds": [{"cover": "SCT", "base": 8000}],
    "rawOb": "PANC 150600Z 01008KT 10SM SCT080 M15/M20 A3025",
}

SAMPLE_TAF = {
    "icaoId": "KPHX",
    "issueTime": "2026-04-09T12:00:00Z",
    "validTimeFrom": "2026-04-09T12:00:00Z",
    "validTimeTo": "2026-04-10T18:00:00Z",
    "fcsts": [
        {
            "fcstType": "FM",
            "timeFrom": "2026-04-09T12:00:00Z",
            "timeTo": "2026-04-09T18:00:00Z",
            "wdir": 200,
            "wspd": 8,
            "visib": 10,
            "clouds": [{"cover": "FEW", "base": 12000}],
            "wxString": None,
        },
        {
            "fcstType": "TEMPO",
            "timeFrom": "2026-04-09T18:00:00Z",
            "timeTo": "2026-04-09T22:00:00Z",
            "wdir": 240,
            "wspd": 15,
            "wgst": 25,
            "visib": 5,
            "clouds": [{"cover": "BKN", "base": 5000}],
            "wxString": "TSRA",
            "probability": 30,
        },
        {
            "fcstType": "FM",
            "timeFrom": "2026-04-09T22:00:00Z",
            "timeTo": "2026-04-10T06:00:00Z",
            "wdir": 180,
            "wspd": 5,
            "visib": 10,
            "clouds": [{"cover": "SCT", "base": 20000}],
            "wxString": None,
        },
        {
            "fcstType": "BECMG",
            "timeFrom": "2026-04-10T06:00:00Z",
            "timeTo": "2026-04-10T10:00:00Z",
            "wdir": 160,
            "wspd": 10,
            "visib": 10,
            "clouds": [{"cover": "FEW", "base": 15000}],
            "wxString": None,
        },
        {
            "fcstType": "PROB30",
            "timeFrom": "2026-04-10T10:00:00Z",
            "timeTo": "2026-04-10T14:00:00Z",
            "wdir": 200,
            "wspd": 20,
            "wgst": 30,
            "visib": 3,
            "clouds": [{"cover": "OVC", "base": 2000}],
            "wxString": "RA",
            "probability": 30,
        },
    ],
    "rawTAF": (
        "TAF KPHX 091200Z 0912/1018 20008KT P6SM FEW120 "
        "TEMPO 0918/0922 24015G25KT 5SM TSRA BKN050 "
        "FM092200 18005KT P6SM SCT200 "
        "BECMG 1006/1010 16010KT P6SM FEW150 "
        "PROB30 1010/1014 20020G30KT 3SM RA OVC020"
    ),
}

SAMPLE_TAF_AMENDED = {
    "icaoId": "KJFK",
    "issueTime": "2026-04-09T14:00:00Z",
    "validTimeFrom": "2026-04-09T14:00:00Z",
    "validTimeTo": "2026-04-10T18:00:00Z",
    "fcsts": [
        {
            "fcstType": "FM",
            "timeFrom": "2026-04-09T14:00:00Z",
            "timeTo": "2026-04-10T18:00:00Z",
            "wdir": 270,
            "wspd": 15,
            "visib": 10,
            "clouds": [{"cover": "SCT", "base": 10000}],
        },
    ],
    "rawTAF": "TAF AMD KJFK 091400Z 0914/1018 27015KT P6SM SCT100",
}

SAMPLE_PIREP = {
    "airepId": "PIREP-001",
    "obsTime": "2026-04-09T17:30:00Z",
    "lat": 33.5,
    "lon": -112.0,
    "altFt": 18000,
    "icgType1": "RIME",
    "icgInt1": "LGT",
    "tbType1": "CAT",
    "tbInt1": "MOD",
    "wxString": None,
    "rawOb": "UA /OV PHX090020/TM 1730/FL180/TP B737/IC LGT RIME/TB MOD CAT",
}

SAMPLE_PIREP_SEVERE = {
    "airepId": "PIREP-002",
    "obsTime": "2026-04-09T18:00:00Z",
    "lat": 33.6,
    "lon": -112.1,
    "altFt": 25000,
    "icgType1": None,
    "icgInt1": None,
    "tbType1": "CONV",
    "tbInt1": "SEV",
    "wxString": "TS",
    "rawOb": "UA /OV PHX180010/TM 1800/FL250/TP A320/TB SEV CONV",
}

SAMPLE_SIGMET = {
    "airSigmetId": "SIGMET-001",
    "hazard": "TURB",
    "severity": "SEV",
    "airsigmetType": "SIGMET",
    "validTimeFrom": "2026-04-09T15:00:00Z",
    "validTimeTo": "2026-04-09T21:00:00Z",
    "coords": [
        {"lat": 34.0, "lon": -113.0},
        {"lat": 34.0, "lon": -111.0},
        {"lat": 33.0, "lon": -111.0},
        {"lat": 33.0, "lon": -113.0},
    ],
    "rawAirSigmet": "SIGMET NOVEMBER 3 VALID UNTIL...",
}

SAMPLE_AIRMET = {
    "airSigmetId": "AIRMET-001",
    "hazard": "IFR",
    "severity": "MOD",
    "airsigmetType": "AIRMET",
    "validTimeFrom": "2026-04-09T12:00:00Z",
    "validTimeTo": "2026-04-09T18:00:00Z",
    "coords": [
        {"lat": 42.0, "lon": -88.0},
        {"lat": 42.0, "lon": -86.0},
        {"lat": 40.0, "lon": -86.0},
        {"lat": 40.0, "lon": -88.0},
    ],
    "rawAirSigmet": "AIRMET SIERRA FOR IFR...",
}


# ---------------------------------------------------------------------------
# METAR parsing tests
# ---------------------------------------------------------------------------


class TestMetarParsing:
    def test_parse_basic_metar(self):
        result = _parse_metar_json(SAMPLE_METAR)
        assert result["station_icao"] == "KPHX"
        assert result["temp_c"] == 32.2
        assert result["wind_speed_kts"] == 12
        assert result["wind_dir"] == "220"
        assert result["wind_gust_kts"] == 18
        assert result["visibility_miles"] == 10
        assert result["pressure_hpa"] == 1013.2
        assert result["flight_category"] == "VFR"
        assert result["is_speci"] is False
        assert result["raw_metar"] is not None

    def test_temp_conversion_c_to_f(self):
        result = _parse_metar_json(SAMPLE_METAR)
        expected_f = 32.2 * 9.0 / 5.0 + 32.0
        assert result["temp_f"] == pytest.approx(expected_f, abs=0.01)

    def test_dewpoint_conversion(self):
        result = _parse_metar_json(SAMPLE_METAR)
        expected_f = 8.3 * 9.0 / 5.0 + 32.0
        assert result["dewpoint_f"] == pytest.approx(expected_f, abs=0.01)

    def test_negative_temperature(self):
        result = _parse_metar_json(SAMPLE_METAR_NEGATIVE_TEMP)
        assert result["temp_c"] == -15.0
        expected_f = -15.0 * 9.0 / 5.0 + 32.0
        assert result["temp_f"] == pytest.approx(expected_f, abs=0.01)
        assert result["temp_f"] < 32.0  # Should be below freezing

    def test_missing_fields_handled(self):
        result = _parse_metar_json(SAMPLE_METAR_MISSING)
        assert result["station_icao"] == "KZZZ"
        assert result["temp_c"] is None
        assert result["temp_f"] is None
        assert result["dewpoint_c"] is None
        assert result["dewpoint_f"] is None
        assert result["wind_speed_kts"] is None
        assert result["visibility_miles"] is None
        assert result["visibility_m"] is None
        assert result["pressure_hpa"] is None
        assert result["flight_category"] is None

    def test_speci_flag_detected(self):
        result = _parse_metar_json(SAMPLE_METAR_SPECI)
        assert result["is_speci"] is True
        assert result["station_icao"] == "KORD"

    def test_ceiling_calculation_bkn(self):
        result = _parse_metar_json(SAMPLE_METAR)
        # FEW at 12000, BKN at 25000 → ceiling = 25000 (first BKN/OVC)
        assert result["ceiling_ft"] == 25000

    def test_ceiling_calculation_ovc(self):
        result = _parse_metar_json(SAMPLE_METAR_SPECI)
        # OVC at 800
        assert result["ceiling_ft"] == 800

    def test_ceiling_none_for_few_sct_only(self):
        metar = dict(SAMPLE_METAR_NEGATIVE_TEMP)
        metar["clouds"] = [{"cover": "SCT", "base": 8000}]
        result = _parse_metar_json(metar)
        assert result["ceiling_ft"] is None

    def test_visibility_conversion(self):
        result = _parse_metar_json(SAMPLE_METAR)
        assert result["visibility_miles"] == 10
        assert result["visibility_m"] == pytest.approx(16093.4, abs=1.0)

    def test_observed_at_parsing(self):
        result = _parse_metar_json(SAMPLE_METAR)
        assert result["observed_at"].year == 2026
        assert result["observed_at"].month == 4
        assert result["observed_at"].day == 9
        assert result["observed_at"].hour == 18
        assert result["observed_at"].minute == 53

    def test_sky_condition_structure(self):
        result = _parse_metar_json(SAMPLE_METAR)
        assert len(result["sky_condition"]) == 2
        assert result["sky_condition"][0]["cover"] == "FEW"
        assert result["sky_condition"][0]["base_ft"] == 12000
        assert result["sky_condition"][1]["cover"] == "BKN"

    def test_empty_clouds(self):
        metar = dict(SAMPLE_METAR_MISSING)
        metar["clouds"] = None
        result = _parse_metar_json(metar)
        assert result["sky_condition"] == []
        assert result["ceiling_ft"] is None


class TestCToF:
    def test_freezing_point(self):
        assert _c_to_f(0.0) == pytest.approx(32.0)

    def test_boiling_point(self):
        assert _c_to_f(100.0) == pytest.approx(212.0)

    def test_none_input(self):
        assert _c_to_f(None) is None

    def test_negative(self):
        assert _c_to_f(-40.0) == pytest.approx(-40.0)


# ---------------------------------------------------------------------------
# TAF parsing tests
# ---------------------------------------------------------------------------


class TestTafParsing:
    def test_parse_basic_taf(self):
        result = _parse_taf_json(SAMPLE_TAF)
        assert result["station_icao"] == "KPHX"
        assert result["issued_at"].hour == 12
        assert result["valid_from"].hour == 12
        assert result["valid_to"].day == 10
        assert result["amendment_number"] == 0

    def test_taf_periods_extracted(self):
        result = _parse_taf_json(SAMPLE_TAF)
        assert len(result["periods"]) == 5

    def test_fm_period(self):
        result = _parse_taf_json(SAMPLE_TAF)
        fm = result["periods"][0]
        assert fm["type"] == "FM"
        assert fm["wind_dir"] == 200
        assert fm["wind_speed_kts"] == 8
        assert fm["visibility_miles"] == 10

    def test_tempo_group_parsed(self):
        result = _parse_taf_json(SAMPLE_TAF)
        tempo = result["periods"][1]
        assert tempo["type"] == "TEMPO"
        assert tempo["wind_gust_kts"] == 25
        assert tempo["weather"] == "TSRA"
        assert tempo["prob"] == 30

    def test_becmg_group_parsed(self):
        result = _parse_taf_json(SAMPLE_TAF)
        becmg = result["periods"][3]
        assert becmg["type"] == "BECMG"
        assert becmg["wind_dir"] == 160

    def test_prob30_group_parsed(self):
        result = _parse_taf_json(SAMPLE_TAF)
        prob30 = result["periods"][4]
        assert prob30["type"] == "PROB30"
        assert prob30["weather"] == "RA"
        assert prob30["prob"] == 30
        assert prob30["wind_gust_kts"] == 30

    def test_amendment_detected(self):
        result = _parse_taf_json(SAMPLE_TAF_AMENDED)
        assert result["amendment_number"] == 1

    def test_no_amendment(self):
        result = _parse_taf_json(SAMPLE_TAF)
        assert result["amendment_number"] == 0

    def test_sky_condition_in_periods(self):
        result = _parse_taf_json(SAMPLE_TAF)
        fm = result["periods"][0]
        assert len(fm["sky_condition"]) == 1
        assert fm["sky_condition"][0]["cover"] == "FEW"
        assert fm["sky_condition"][0]["base_ft"] == 12000


class TestFindTafPeriod:
    def test_find_fm_period(self):
        taf = _parse_taf_json(SAMPLE_TAF)
        target = datetime(2026, 4, 9, 14, 0, tzinfo=timezone.utc)
        period = _find_taf_period(taf["periods"], target)
        assert period is not None
        assert period["type"] == "FM"
        assert period["wind_dir"] == 200

    def test_find_period_with_overlays(self):
        taf = _parse_taf_json(SAMPLE_TAF)
        # 23:00 is in the FM 22-06 period AND no overlays
        target = datetime(2026, 4, 9, 23, 0, tzinfo=timezone.utc)
        period = _find_taf_period(taf["periods"], target)
        assert period is not None
        assert period["type"] == "FM"
        assert period["wind_dir"] == 180

    def test_no_matching_period(self):
        taf = _parse_taf_json(SAMPLE_TAF)
        # Way outside valid range
        target = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)
        period = _find_taf_period(taf["periods"], target)
        assert period is None


# ---------------------------------------------------------------------------
# PIREP parsing tests
# ---------------------------------------------------------------------------


class TestPirepParsing:
    def test_parse_basic_pirep(self):
        result = _parse_pirep_json(SAMPLE_PIREP)
        assert result["report_id"] == "PIREP-001"
        assert result["lat"] == 33.5
        assert result["lon"] == -112.0
        assert result["altitude_ft"] == 18000
        assert result["icing_type"] == "RIME"
        assert result["icing_intensity"] == "LGT"
        assert result["turbulence_type"] == "CAT"
        assert result["turbulence_intensity"] == "MOD"

    def test_parse_severe_pirep(self):
        result = _parse_pirep_json(SAMPLE_PIREP_SEVERE)
        assert result["turbulence_intensity"] == "SEV"
        assert result["turbulence_type"] == "CONV"


# ---------------------------------------------------------------------------
# SIGMET/AIRMET parsing tests
# ---------------------------------------------------------------------------


class TestAirsigmetParsing:
    def test_parse_sigmet(self):
        result = _parse_airsigmet_json(SAMPLE_SIGMET)
        assert result["alert_id"] == "SIGMET-001"
        assert result["alert_type"] == "SIGMET"
        assert result["hazard"] == "TURB"
        assert result["severity"] == "SEV"
        assert len(result["area"]) == 4

    def test_parse_airmet(self):
        result = _parse_airsigmet_json(SAMPLE_AIRMET)
        assert result["alert_id"] == "AIRMET-001"
        assert result["alert_type"] == "AIRMET"
        assert result["hazard"] == "IFR"


class TestPointInPolygon:
    def test_inside_polygon(self):
        polygon = [
            {"lat": 34.0, "lon": -113.0},
            {"lat": 34.0, "lon": -111.0},
            {"lat": 33.0, "lon": -111.0},
            {"lat": 33.0, "lon": -113.0},
        ]
        # Phoenix area (33.45, -112.07) should be inside
        assert _point_in_polygon(33.45, -112.07, polygon) is True

    def test_outside_polygon(self):
        polygon = [
            {"lat": 34.0, "lon": -113.0},
            {"lat": 34.0, "lon": -111.0},
            {"lat": 33.0, "lon": -111.0},
            {"lat": 33.0, "lon": -113.0},
        ]
        # Los Angeles (34.05, -118.24) should be outside
        assert _point_in_polygon(34.05, -118.24, polygon) is False

    def test_empty_polygon(self):
        assert _point_in_polygon(33.0, -112.0, []) is False

    def test_two_point_polygon(self):
        polygon = [{"lat": 33.0, "lon": -112.0}, {"lat": 34.0, "lon": -112.0}]
        assert _point_in_polygon(33.5, -112.0, polygon) is False


# ---------------------------------------------------------------------------
# Fetch function tests (mocked HTTP)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFetchMetars:
    @patch("src.ingestion.aviation._awc_get_json")
    @patch("src.ingestion.aviation.async_session")
    async def test_fetch_single_station(self, mock_session_factory, mock_get):
        mock_session = AsyncMock()
        mock_session_factory.return_value = mock_session

        mock_get.return_value = [SAMPLE_METAR]

        result = await fetch_latest_metars(["KPHX"])
        assert len(result) == 1
        assert result[0]["station_icao"] == "KPHX"
        assert mock_session.add.called
        assert mock_session.commit.called

    @patch("src.ingestion.aviation._awc_get_json")
    @patch("src.ingestion.aviation.async_session")
    async def test_fetch_multiple_stations(self, mock_session_factory, mock_get):
        mock_session = AsyncMock()
        mock_session_factory.return_value = mock_session

        metar2 = dict(SAMPLE_METAR)
        metar2["icaoId"] = "KJFK"
        mock_get.return_value = [SAMPLE_METAR, metar2]

        result = await fetch_latest_metars(["KPHX", "KJFK"])
        assert len(result) == 2

    @patch("src.ingestion.aviation._awc_get_json")
    @patch("src.ingestion.aviation.async_session")
    async def test_station_failure_doesnt_fail_batch(
        self, mock_session_factory, mock_get
    ):
        mock_session = AsyncMock()
        mock_session_factory.return_value = mock_session

        mock_get.side_effect = Exception("API error")

        result = await fetch_latest_metars(["KPHX"])
        # Should return empty list, not raise
        assert result == []
        assert mock_session.commit.called

    @patch("src.ingestion.aviation._awc_get_json")
    @patch("src.ingestion.aviation.async_session")
    async def test_empty_response(self, mock_session_factory, mock_get):
        mock_session = AsyncMock()
        mock_session_factory.return_value = mock_session

        mock_get.return_value = []

        result = await fetch_latest_metars(["KPHX"])
        assert result == []

    async def test_empty_station_list(self):
        result = await fetch_latest_metars([])
        assert result == []


@pytest.mark.asyncio
class TestFetchTafs:
    @patch("src.ingestion.aviation._awc_get_json")
    @patch("src.ingestion.aviation.async_session")
    async def test_fetch_single_taf(self, mock_session_factory, mock_get):
        mock_session = AsyncMock()
        mock_session_factory.return_value = mock_session

        mock_get.return_value = [SAMPLE_TAF]

        result = await fetch_latest_tafs(["KPHX"])
        assert len(result) == 1
        assert result[0]["station_icao"] == "KPHX"
        assert len(result[0]["periods"]) == 5


@pytest.mark.asyncio
class TestFetchPireps:
    @patch("src.ingestion.aviation._awc_get_json")
    @patch("src.ingestion.aviation.async_session")
    async def test_fetch_pireps_near(self, mock_session_factory, mock_get):
        mock_session = AsyncMock()
        mock_session_factory.return_value = mock_session

        mock_get.return_value = [SAMPLE_PIREP, SAMPLE_PIREP_SEVERE]

        result = await fetch_pireps_near(33.45, -112.07, radius_nm=100)
        assert len(result) == 2

    @patch("src.ingestion.aviation.fetch_pireps_near")
    async def test_has_severe_weather_true(self, mock_fetch):
        # Override observed_at so the 6h cutoff in has_severe_weather_reports
        # doesn't filter out the fixture (which is frozen in April 2026).
        now = datetime.now(timezone.utc)
        severe = _parse_pirep_json(SAMPLE_PIREP_SEVERE)
        severe["observed_at"] = now
        normal = _parse_pirep_json(SAMPLE_PIREP)
        normal["observed_at"] = now
        mock_fetch.return_value = [normal, severe]
        assert await has_severe_weather_reports(33.45, -112.07) is True

    @patch("src.ingestion.aviation.fetch_pireps_near")
    async def test_has_severe_weather_false(self, mock_fetch):
        # Use a PIREP with only LGT intensity (not MOD/SEV/EXTM)
        light_pirep = dict(SAMPLE_PIREP)
        light_pirep["tbInt1"] = "LGT"
        parsed = _parse_pirep_json(light_pirep)
        parsed["observed_at"] = datetime.now(timezone.utc)
        mock_fetch.return_value = [parsed]
        assert await has_severe_weather_reports(33.45, -112.07) is False


@pytest.mark.asyncio
class TestFetchAlerts:
    @patch("src.ingestion.aviation._awc_get_json")
    @patch("src.ingestion.aviation.async_session")
    async def test_fetch_active_sigmets(self, mock_session_factory, mock_get):
        mock_session = AsyncMock()
        mock_session_factory.return_value = mock_session

        mock_get.return_value = [SAMPLE_SIGMET, SAMPLE_AIRMET]

        result = await fetch_active_sigmets()
        assert len(result) == 1
        assert result[0]["alert_type"] == "SIGMET"

    @patch("src.ingestion.aviation._awc_get_json")
    @patch("src.ingestion.aviation.async_session")
    async def test_fetch_active_airmets(self, mock_session_factory, mock_get):
        mock_session = AsyncMock()
        mock_session_factory.return_value = mock_session

        mock_get.return_value = [SAMPLE_SIGMET, SAMPLE_AIRMET]

        result = await fetch_active_airmets()
        assert len(result) == 1
        assert result[0]["alert_type"] == "AIRMET"


# ---------------------------------------------------------------------------
# Temperature trend tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestTempTrend:
    @patch("src.ingestion.aviation.fetch_metar_history")
    async def test_rising_trend(self, mock_history):
        now = datetime.now(timezone.utc)
        mock_history.return_value = [
            {"observed_at": now - timedelta(hours=3), "temp_f": 80.0},
            {"observed_at": now - timedelta(hours=2), "temp_f": 83.0},
            {"observed_at": now - timedelta(hours=1), "temp_f": 86.0},
            {"observed_at": now, "temp_f": 89.0},
        ]

        result = await get_temp_trend("KPHX", hours=6)
        assert result["current"] == 89.0
        assert result["min"] == 80.0
        assert result["max"] == 89.0
        assert result["trend_direction"] == "rising"
        assert result["rate_of_change_per_hour"] > 0

    @patch("src.ingestion.aviation.fetch_metar_history")
    async def test_falling_trend(self, mock_history):
        now = datetime.now(timezone.utc)
        mock_history.return_value = [
            {"observed_at": now - timedelta(hours=2), "temp_f": 90.0},
            {"observed_at": now - timedelta(hours=1), "temp_f": 87.0},
            {"observed_at": now, "temp_f": 84.0},
        ]

        result = await get_temp_trend("KPHX", hours=6)
        assert result["trend_direction"] == "falling"
        assert result["rate_of_change_per_hour"] < 0

    @patch("src.ingestion.aviation.fetch_metar_history")
    async def test_steady_trend(self, mock_history):
        now = datetime.now(timezone.utc)
        mock_history.return_value = [
            {"observed_at": now - timedelta(hours=2), "temp_f": 75.0},
            {"observed_at": now - timedelta(hours=1), "temp_f": 75.2},
            {"observed_at": now, "temp_f": 75.1},
        ]

        result = await get_temp_trend("KPHX", hours=6)
        assert result["trend_direction"] == "steady"

    @patch("src.ingestion.aviation.fetch_metar_history")
    async def test_no_data(self, mock_history):
        mock_history.return_value = []

        result = await get_temp_trend("KZZZ", hours=6)
        assert result["current"] is None
        assert result["trend_direction"] == "unknown"

    @patch("src.ingestion.aviation.fetch_metar_history")
    async def test_single_observation(self, mock_history):
        now = datetime.now(timezone.utc)
        mock_history.return_value = [
            {"observed_at": now, "temp_f": 80.0},
        ]

        result = await get_temp_trend("KPHX", hours=6)
        assert result["current"] == 80.0
        assert result["trend_direction"] == "steady"
        assert result["rate_of_change_per_hour"] == 0.0


# ---------------------------------------------------------------------------
# SPECI detection tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSpeciDetection:
    @patch("src.ingestion.aviation.fetch_metar_history")
    async def test_detect_speci(self, mock_history):
        mock_history.return_value = [
            _parse_metar_json(SAMPLE_METAR),
            _parse_metar_json(SAMPLE_METAR_SPECI),
        ]

        specis = await detect_speci_events("KORD", hours=12)
        assert len(specis) == 1
        assert specis[0]["is_speci"] is True

    @patch("src.ingestion.aviation.fetch_metar_history")
    async def test_no_specis(self, mock_history):
        mock_history.return_value = [
            _parse_metar_json(SAMPLE_METAR),
        ]

        specis = await detect_speci_events("KPHX", hours=12)
        assert len(specis) == 0


# ---------------------------------------------------------------------------
# Composite probability tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestTafPrecipProbability:
    @patch("src.ingestion.aviation.fetch_latest_tafs")
    async def test_precip_from_prob30(self, mock_fetch):
        mock_fetch.return_value = [_parse_taf_json(SAMPLE_TAF)]

        # Target time in the PROB30 RA period
        target = datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc)
        prob = await get_taf_precip_probability("KPHX", target)
        assert prob == pytest.approx(0.3, abs=0.01)

    @patch("src.ingestion.aviation.fetch_latest_tafs")
    async def test_no_precip(self, mock_fetch):
        mock_fetch.return_value = [_parse_taf_json(SAMPLE_TAF)]

        # Target time in the clear FM period
        target = datetime(2026, 4, 9, 14, 0, tzinfo=timezone.utc)
        prob = await get_taf_precip_probability("KPHX", target)
        assert prob == 0.0

    @patch("src.ingestion.aviation.fetch_latest_tafs")
    async def test_no_taf(self, mock_fetch):
        mock_fetch.return_value = []

        target = datetime(2026, 4, 9, 14, 0, tzinfo=timezone.utc)
        prob = await get_taf_precip_probability("KPHX", target)
        assert prob == 0.0


@pytest.mark.asyncio
class TestTafAmendmentCount:
    @patch("src.ingestion.aviation.async_session")
    async def test_amendment_count(self, mock_session_factory):
        mock_session = AsyncMock()
        mock_session_factory.return_value = mock_session

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = 5
        mock_session.execute.return_value = mock_result

        count = await taf_amendment_count("KPHX", hours=24)
        assert count == 5

    @patch("src.ingestion.aviation.async_session")
    async def test_no_amendments(self, mock_session_factory):
        mock_session = AsyncMock()
        mock_session_factory.return_value = mock_session

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = 0
        mock_session.execute.return_value = mock_result

        count = await taf_amendment_count("KZZZ", hours=24)
        assert count == 0


# ---------------------------------------------------------------------------
# Rate limiting tests (moved up after removing legacy probability tests)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Rate limiting tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRateLimiting:
    @patch("src.ingestion.aviation._awc_semaphore")
    async def test_throttle_acquires_semaphore(self, mock_sem):
        from src.ingestion.aviation import _throttle

        mock_sem.acquire = AsyncMock()
        mock_sem.release = MagicMock()

        await _throttle()
        assert mock_sem.acquire.called


# ---------------------------------------------------------------------------
# ICAO mapping tests (in mapper.py)
# ---------------------------------------------------------------------------


class TestIcaoMapping:
    def test_exact_match(self):
        from src.signals.mapper import icao_for_location

        assert icao_for_location("Phoenix") == "KPHX"

    def test_case_insensitive(self):
        from src.signals.mapper import icao_for_location

        assert icao_for_location("CHICAGO") == "KORD"

    def test_substring_match(self):
        from src.signals.mapper import icao_for_location

        result = icao_for_location("new york city")
        assert result == "KJFK"

    def test_unknown_returns_none(self):
        from src.signals.mapper import icao_for_location

        assert icao_for_location("Timbuktu") is None

    def test_state_mapping(self):
        from src.signals.mapper import icao_for_location

        assert icao_for_location("Florida") == "KMIA"
        assert icao_for_location("Texas") == "KDFW"
