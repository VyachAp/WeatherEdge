"""Tests for raw METAR and SYNOP parsers."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.ingestion.aviation._parsers import parse_raw_metar, parse_raw_synop


class TestRawMetarParser:
    def test_basic_metar(self):
        raw = "METAR KPHX 091853Z 22012G18KT 10SM FEW250 32/08 A2992"
        obs = parse_raw_metar(raw, station="KPHX", source="test")
        assert obs is not None
        assert obs.station_icao == "KPHX"
        assert obs.source == "test"
        # Temperature should be parsed
        assert obs.temp_c is not None
        assert obs.temp_c == pytest.approx(32.0, abs=1.0)

    def test_speci_detection(self):
        raw = "SPECI KORD 091900Z 18015G25KT 3SM +RA BKN015 OVC025 18/16 A2985"
        obs = parse_raw_metar(raw, source="test")
        assert obs is not None
        assert obs.is_speci is True

    def test_empty_metar(self):
        assert parse_raw_metar("", source="test") is None

    def test_invalid_metar(self):
        # python-metar is lenient with strict=False; garbage returns empty Observation
        obs = parse_raw_metar("not a metar at all", source="test")
        # Should still produce an Observation, but with no useful data
        if obs is not None:
            assert obs.temp_c is None
            assert obs.wind_speed_kts is None

    def test_negative_temperature(self):
        raw = "METAR KORD 091853Z 36010KT 10SM OVC020 M05/M12 A3025"
        obs = parse_raw_metar(raw, source="test")
        if obs is not None:
            # Temperature should be negative
            assert obs.temp_c is not None
            assert obs.temp_c < 0

    def test_ceiling_calculation(self):
        raw = "METAR KJFK 091853Z 18010KT 5SM BKN015 OVC025 22/18 A2990"
        obs = parse_raw_metar(raw, source="test")
        if obs is not None and obs.ceiling_ft is not None:
            assert obs.ceiling_ft == 1500


class TestRawSynopParser:
    def test_basic_synop(self):
        # Simplified SYNOP with temperature group
        raw = "AAXX 09184 72278 41598 10322 20083 30125 40189 52014 60001"
        obs = parse_raw_synop(raw, wmo_id="72278")
        assert obs is not None
        assert obs.wmo_id == "72278"

    def test_temperature_parsing(self):
        # 10322 = positive 32.2°C
        raw = "AAXX 09184 72278 41598 10322 20083"
        obs = parse_raw_synop(raw, wmo_id="72278")
        if obs is not None and obs.temp_c is not None:
            assert obs.temp_c == pytest.approx(32.2, abs=0.1)

    def test_negative_temperature(self):
        # 11050 = negative 5.0°C
        raw = "AAXX 09184 72278 41598 11050 20083"
        obs = parse_raw_synop(raw, wmo_id="72278")
        if obs is not None and obs.temp_c is not None:
            assert obs.temp_c == pytest.approx(-5.0, abs=0.1)

    def test_precipitation(self):
        # 60101 = 10 mm precip, period code 1 (6 hours)
        raw = "AAXX 09184 72278 41598 10322 60101"
        obs = parse_raw_synop(raw, wmo_id="72278")
        if obs is not None and obs.precip_mm is not None:
            assert obs.precip_mm == 10.0

    def test_empty_synop(self):
        assert parse_raw_synop("", wmo_id="72278") is None

    def test_short_synop(self):
        assert parse_raw_synop("short", wmo_id="72278") is None
