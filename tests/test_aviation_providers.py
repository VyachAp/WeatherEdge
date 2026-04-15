"""Tests for individual aviation weather providers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ingestion.aviation._caching import clear_all_caches
from src.ingestion.aviation._provider_awc import (
    AWCProvider,
    _parse_awc_observed_at,
    parse_metar_json,
    parse_taf_json,
    parse_pirep_json,
    parse_airsigmet_json,
    _metar_dict_to_observation,
)
from src.ingestion.aviation._provider_checkwx import CheckWXProvider
from src.ingestion.aviation._provider_iem import (
    IEMProvider,
    _parse_iem_csv,
    _parse_iem_1min_csv,
)
from src.ingestion.aviation._provider_noaa import NOAAProvider, _parse_noaa_metar_text
from src.ingestion.aviation._provider_ogimet import OGIMETProvider
from src.ingestion.aviation._provider_avwx import AVWXProvider
from src.ingestion.aviation._types import Observation


@pytest.fixture(autouse=True)
def _clear_caches():
    clear_all_caches()
    yield
    clear_all_caches()


# ---------------------------------------------------------------------------
# AWC Provider
# ---------------------------------------------------------------------------

SAMPLE_AWC_METAR = {
    "icaoId": "KPHX",
    "obsTime": "2026-04-09T18:53:00Z",
    "temp": 32.2,
    "dwpt": 8.3,
    "wdir": 220,
    "wspd": 12,
    "wgst": 18,
    "visib": 10,
    "altim": 1013.2,
    "clouds": [
        {"cover": "FEW", "base": 25000},
    ],
    "fltcat": "VFR",
    "rawOb": "KPHX 091853Z 22012G18KT 10SM FEW250 32/08 A2992",
    "metar_type": "METAR",
}


class TestAWCObservedAt:
    def test_unix_epoch_obstime(self):
        # Real AWC shape: obsTime is a Unix epoch int
        dt = _parse_awc_observed_at({"obsTime": 1776264780, "reportTime": None})
        assert dt == datetime(2026, 4, 15, 14, 53, tzinfo=timezone.utc)

    def test_prefer_report_time_iso(self):
        # When both are present, ISO reportTime is used first
        dt = _parse_awc_observed_at({
            "obsTime": 1776264780,
            "reportTime": "2026-04-15T15:00:00.000Z",
        })
        assert dt == datetime(2026, 4, 15, 15, 0, tzinfo=timezone.utc)

    def test_fallback_when_neither_present(self):
        before = datetime.now(timezone.utc)
        dt = _parse_awc_observed_at({})
        after = datetime.now(timezone.utc)
        assert before <= dt <= after

    def test_malformed_obstime_falls_back_to_report_time(self):
        dt = _parse_awc_observed_at({
            "obsTime": "not-a-time",
            "reportTime": "2026-04-15T15:00:00Z",
        })
        assert dt == datetime(2026, 4, 15, 15, 0, tzinfo=timezone.utc)


class TestAWCProvider:
    def test_parse_metar_json_basic(self):
        result = parse_metar_json(SAMPLE_AWC_METAR)
        assert result["station_icao"] == "KPHX"
        assert result["temp_c"] == 32.2
        assert result["temp_f"] == pytest.approx(90.0, abs=0.1)
        assert result["wind_speed_kts"] == 12
        assert result["wind_gust_kts"] == 18
        assert result["visibility_miles"] == 10
        assert result["flight_category"] == "VFR"
        assert result["is_speci"] is False

    def test_parse_metar_json_speci(self):
        raw = dict(SAMPLE_AWC_METAR)
        raw["metar_type"] = "SPECI"
        result = parse_metar_json(raw)
        assert result["is_speci"] is True

    def test_parse_metar_json_missing_fields(self):
        raw = {"icaoId": "KPHX"}
        result = parse_metar_json(raw)
        assert result["station_icao"] == "KPHX"
        assert result["temp_c"] is None
        assert result["temp_f"] is None

    def test_metar_dict_to_observation(self):
        parsed = parse_metar_json(SAMPLE_AWC_METAR)
        obs = _metar_dict_to_observation(parsed)
        assert isinstance(obs, Observation)
        assert obs.station_icao == "KPHX"
        assert obs.source == "awc"
        assert obs.temp_f == pytest.approx(90.0, abs=0.1)

    def test_parse_taf_json(self):
        raw = {
            "icaoId": "KPHX",
            "issueTime": "2026-04-09T18:00:00Z",
            "validTimeFrom": "2026-04-09T18:00:00Z",
            "validTimeTo": "2026-04-10T18:00:00Z",
            "rawTAF": "TAF KPHX 091800Z 0918/1018 22012KT P6SM FEW250",
            "fcsts": [
                {
                    "fcstType": "FM",
                    "timeFrom": "2026-04-09T18:00:00Z",
                    "timeTo": "2026-04-10T06:00:00Z",
                    "wdir": 220,
                    "wspd": 12,
                    "visib": 6,
                    "clouds": [{"cover": "FEW", "base": 25000}],
                }
            ],
        }
        result = parse_taf_json(raw)
        assert result["station_icao"] == "KPHX"
        assert len(result["periods"]) == 1
        assert result["periods"][0]["wind_speed_kts"] == 12
        assert result["amendment_number"] == 0

    def test_parse_taf_json_amendment(self):
        raw = {
            "icaoId": "KORD",
            "issueTime": "2026-04-09T18:00:00Z",
            "validTimeFrom": "2026-04-09T18:00:00Z",
            "validTimeTo": "2026-04-10T18:00:00Z",
            "rawTAF": "TAF AMD KORD 091800Z ...",
            "fcsts": [],
        }
        result = parse_taf_json(raw)
        assert result["amendment_number"] == 1

    def test_parse_pirep_json(self):
        raw = {
            "pirepId": "PIREP-001",
            "obsTime": "2026-04-09T17:00:00Z",
            "lat": 33.45,
            "lon": -112.07,
            "altFt": 25000,
            "tbInt1": "MOD",
            "tbType1": "CAT",
            "rawOb": "UA /OV PHX/TM 1700/FL250/TP B737/TB MOD CAT",
        }
        result = parse_pirep_json(raw)
        assert result["report_id"] == "PIREP-001"
        assert result["turbulence_intensity"] == "MOD"
        assert result["altitude_ft"] == 25000

    def test_parse_airsigmet_json(self):
        raw = {
            "airSigmetId": "SIGMET-001",
            "airsigmetType": "SIGMET",
            "hazard": "CONVECTION",
            "severity": "SEV",
            "validTimeFrom": "2026-04-09T16:00:00Z",
            "validTimeTo": "2026-04-09T22:00:00Z",
            "coords": [
                {"lat": 33.0, "lon": -113.0},
                {"lat": 34.0, "lon": -113.0},
                {"lat": 34.0, "lon": -111.0},
                {"lat": 33.0, "lon": -111.0},
            ],
            "rawAirSigmet": "SIGMET...",
        }
        result = parse_airsigmet_json(raw)
        assert result["alert_type"] == "SIGMET"
        assert result["hazard"] == "CONVECTION"
        assert len(result["area"]) == 4


# ---------------------------------------------------------------------------
# IEM Provider
# ---------------------------------------------------------------------------


SAMPLE_IEM_CSV = """#DEBUG: 1 stations found
station,valid,tmpf,dwpf,drct,sknt,gust,vsby,skyc1,skyc2,skyc3,skyl1,skyl2,skyl3,p01i,alti,metar,mslp,feel,lat,lon
KPHX,2026-04-09 17:53,90.0,30.0,220,12,18,10,FEW,M,M,25000,M,M,0.00,29.92,KPHX 091753Z 22012G18KT 10SM FEW250 32/08 A2992,M,M,33.43,-112.02
KPHX,2026-04-09 16:53,88.0,29.0,210,10,M,10,CLR,M,M,M,M,M,0.00,29.93,KPHX 091653Z 21010KT 10SM CLR 31/08 A2993,M,M,33.43,-112.02
"""


class TestIEMProvider:
    def test_parse_iem_csv_basic(self):
        observations = _parse_iem_csv(SAMPLE_IEM_CSV, "KPHX")
        assert len(observations) == 2
        # Most recent first (sorted)
        assert observations[0].temp_f == 90.0
        assert observations[0].wind_speed_kts == 12
        assert observations[0].wind_gust_kts == 18
        assert observations[0].source == "iem"
        assert observations[1].temp_f == 88.0

    def test_parse_iem_csv_missing_values(self):
        csv_text = """#DEBUG
station,valid,tmpf,dwpf,drct,sknt,gust,vsby,skyc1,skyc2,skyc3,skyl1,skyl2,skyl3,p01i,alti,metar,mslp,feel,lat,lon
KPHX,2026-04-09 17:53,M,M,M,M,M,M,M,M,M,M,M,M,M,M,,M,M,33.43,-112.02
"""
        observations = _parse_iem_csv(csv_text, "KPHX")
        assert len(observations) == 1
        assert observations[0].temp_f is None
        assert observations[0].wind_speed_kts is None

    def test_parse_iem_csv_empty(self):
        observations = _parse_iem_csv("", "KPHX")
        assert observations == []

    def test_parse_iem_1min_csv(self):
        csv_text = """#DEBUG
station,valid,tmpf,dwpf,drct,sknt,pres,precip
KPHX,2026-04-09 17:53,90.0,30.0,220,12,29.92,0.00
KPHX,2026-04-09 17:52,89.9,30.0,220,11,29.92,0.00
"""
        results = _parse_iem_1min_csv(csv_text, "KPHX")
        assert len(results) == 2
        # Most recent first
        assert results[0].temp_c is not None
        assert results[0].wind_speed_kts == 12


# ---------------------------------------------------------------------------
# NOAA Provider
# ---------------------------------------------------------------------------


class TestNOAAProvider:
    def test_parse_noaa_metar_text(self):
        text = """2026/04/09 18:53
KPHX 091853Z 22012G18KT 10SM FEW250 32/08 A2992 RMK AO2"""
        obs = _parse_noaa_metar_text(text, "KPHX")
        assert obs is not None
        assert obs.station_icao == "KPHX"
        assert obs.source == "noaa"
        assert obs.wind_speed_kts is not None

    def test_parse_noaa_metar_empty(self):
        obs = _parse_noaa_metar_text("", "KPHX")
        assert obs is None


# ---------------------------------------------------------------------------
# CheckWX Provider
# ---------------------------------------------------------------------------


class TestCheckWXProvider:
    def test_parse_decoded_metar(self):
        provider = CheckWXProvider(api_key="test_key")
        raw = {
            "icao": "KPHX",
            "observed": "2026-04-09T18:53:00Z",
            "temperature": {"celsius": 32.2},
            "dewpoint": {"celsius": 8.3},
            "wind": {"speed_kts": 12, "gust_kts": 18, "degrees": 220},
            "visibility": {"meters": 16093, "miles": 10},
            "barometer": {"hpa": 1013.2},
            "clouds": [{"code": "FEW", "base_feet_agl": 25000}],
            "flight_category": "VFR",
            "raw_text": "KPHX 091853Z 22012G18KT 10SM FEW250 32/08 A2992",
        }
        obs = provider._parse_decoded_metar(raw)
        assert obs.station_icao == "KPHX"
        assert obs.temp_c == 32.2
        assert obs.source == "checkwx"
        assert obs.wind_speed_kts == 12
        assert obs.flight_category == "VFR"


# ---------------------------------------------------------------------------
# AVWX Provider
# ---------------------------------------------------------------------------


class TestAVWXProvider:
    def test_parse_metar(self):
        provider = AVWXProvider(api_key="test_key")
        raw = {
            "station": "KPHX",
            "time": {"dt": "2026-04-09T18:53:00Z"},
            "temperature": {"value": 32.2},
            "dewpoint": {"value": 8.3},
            "wind_direction": {"value": 220},
            "wind_speed": {"value": 12},
            "wind_gust": {"value": 18},
            "visibility": {"value": 16093},
            "altimeter": {"value": 1013.2},
            "clouds": [{"type": "FEW", "altitude": 250}],
            "flight_rules": "VFR",
            "raw": "KPHX 091853Z 22012G18KT 10SM FEW250 32/08 A2992",
        }
        obs = provider._parse_metar(raw)
        assert obs.station_icao == "KPHX"
        assert obs.source == "avwx"
        assert obs.temp_c == 32.2
        assert obs.ceiling_ft is None  # FEW doesn't set ceiling
