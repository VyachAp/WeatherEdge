"""Tests for mapper utilities (geocoding, operator normalisation, date parsing)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.signals.mapper import (
    convert_threshold,
    geocode,
    normalize_operator,
    parse_target_date,
)


# ===================================================================
# mapper.py – geocoding
# ===================================================================


class TestGeocode:
    def test_known_city(self):
        coords = geocode("Phoenix")
        assert coords is not None
        assert pytest.approx(coords[0], abs=0.01) == 33.45

    def test_case_insensitive(self):
        assert geocode("phoenix") == geocode("Phoenix")

    def test_unknown_location_returns_none(self):
        assert geocode("Atlantis") is None

    def test_state_name(self):
        coords = geocode("Florida")
        assert coords is not None
        # Should resolve to state capital (Tallahassee)
        assert pytest.approx(coords[0], abs=0.1) == 30.4

    def test_substring_match(self):
        # "New York City" should match "new york city"
        coords = geocode("New York City")
        assert coords is not None


# ===================================================================
# mapper.py – operator mapping
# ===================================================================


class TestNormalizeOperator:
    def test_above(self):
        assert normalize_operator("above") == "above"

    def test_below(self):
        assert normalize_operator("below") == "below"

    def test_at_least_maps_to_above(self):
        assert normalize_operator("at_least") == "above"

    def test_at_most_maps_to_below(self):
        assert normalize_operator("at_most") == "below"

    def test_unknown_operator_returns_none(self):
        assert normalize_operator("record_breaking") is None


# ===================================================================
# mapper.py – date parsing
# ===================================================================


class TestParseDateTarget:
    def test_full_date(self):
        dt = parse_target_date("July 15, 2026")
        assert dt is not None
        assert dt.month == 7
        assert dt.day == 15
        assert dt.year == 2026
        assert dt.hour == 23
        assert dt.minute == 59

    def test_month_year_defaults_to_15th(self):
        dt = parse_target_date("January 2026")
        assert dt is not None
        assert dt.day == 15
        assert dt.month == 1
        assert dt.year == 2026
        assert dt.hour == 23
        assert dt.minute == 59

    def test_invalid_returns_none(self):
        assert parse_target_date("someday maybe") is None

    def test_returns_utc(self):
        dt = parse_target_date("July 15, 2026")
        assert dt is not None
        assert dt.tzinfo is not None

    def test_end_of_day(self):
        dt = parse_target_date("April 12")
        assert dt is not None
        assert dt.hour == 23
        assert dt.minute == 59
        assert dt.second == 59


# ===================================================================
# mapper.py – unit conversion
# ===================================================================


class TestConvertThreshold:
    def test_temperature_f_to_k(self):
        # 32°F = 273.15K
        assert pytest.approx(convert_threshold(32.0, "temperature"), abs=0.01) == 273.15

    def test_temperature_100f(self):
        # 100°F = 310.928K
        assert pytest.approx(convert_threshold(100.0, "temperature"), abs=0.1) == 310.93



# ===================================================================
# mapper.py – map_market
# ===================================================================


def _future_date_str(days: int = 3) -> str:
    """Return a date string N days from now, parseable by dateutil."""
    dt = datetime.now(tz=timezone.utc) + timedelta(days=days)
    return dt.strftime("%B %d, %Y")


def _short_range_date_str() -> str:
    """Return today's date string — parsed as end-of-day, within 30h window."""
    dt = datetime.now(tz=timezone.utc)
    return dt.strftime("%B %d, %Y")


def _make_market(**overrides):
    """Create a mock Market ORM object with sensible defaults."""
    m = MagicMock()
    m.id = overrides.get("id", "mkt_001")
    m.question = overrides.get("question", "Will it exceed 100°F in Phoenix?")
    m.parsed_location = overrides.get("parsed_location", "Phoenix")
    m.parsed_variable = overrides.get("parsed_variable", "temperature")
    m.parsed_threshold = overrides.get("parsed_threshold", 100.0)
    m.parsed_operator = overrides.get("parsed_operator", "above")
    m.parsed_target_date = overrides.get("parsed_target_date", _future_date_str(3))
    m.current_yes_price = overrides.get("current_yes_price", 0.45)
    m.volume = overrides.get("volume", 5000.0)
    m.liquidity = overrides.get("liquidity", 1000.0)
    m.end_date = overrides.get("end_date", None)
    return m


