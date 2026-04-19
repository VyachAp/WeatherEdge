"""Tests for Weather Underground history scraper parsing logic."""

from __future__ import annotations

from datetime import date

from src.ingestion.wu_history import (
    WuHourlyReading,
    _parse_hourly_row,
    _parse_num,
    _parse_summary_table,
    compare_wu_vs_wx,
    WuDailySummary,
)


# ---------------------------------------------------------------------------
# _parse_num
# ---------------------------------------------------------------------------


class TestParseNum:
    def test_integer(self):
        assert _parse_num("73") == 73.0

    def test_float(self):
        assert _parse_num("29.28") == 29.28

    def test_with_unit_f(self):
        assert _parse_num("73 °F") == 73.0

    def test_with_unit_in(self):
        assert _parse_num("29.28 in") == 29.28

    def test_with_percent(self):
        assert _parse_num("79 %") == 79.0

    def test_with_mph(self):
        assert _parse_num("15 mph") == 15.0

    def test_dash(self):
        assert _parse_num("-") is None

    def test_double_dash(self):
        assert _parse_num("--") is None

    def test_empty(self):
        assert _parse_num("") is None

    def test_zero(self):
        assert _parse_num("0.0 in") == 0.0

    def test_negative(self):
        assert _parse_num("-5") == -5.0


# ---------------------------------------------------------------------------
# _parse_summary_table
# ---------------------------------------------------------------------------


class TestParseSummaryTable:
    SAMPLE_ROWS = [
        ["Temperature (°F)", "Actual", "Historic Avg.", "Record"],
        ["", "", "", ""],
        ["High Temp", "73", "80.6", "99"],
        ["Low Temp", "51", "56.6", "35"],
        ["Day Average Temp", "63.35", "68.6", "-"],
        ["Precipitation (in)", "Actual", "Historic Avg.", "Record"],
        ["", "", "", ""],
        ["Precipitation (past 24 hours from 05:53:00)", "0.00", "4.50", "-"],
        ["Dew Point (°F)", "Actual", "Historic Avg.", "Record"],
        ["", "", "", ""],
        ["Dew Point", "56.76", "-", "-"],
        ["High", "69", "-", "-"],
        ["Low", "42", "-", "-"],
        ["Average", "56.76", "-", "-"],
        ["Wind (mph)", "Actual", "Historic Avg.", "Record"],
        ["", "", "", ""],
        ["Max Wind Speed", "24", "-", "-"],
        ["Visibility", "10", "-", "-"],
        ["Sea Level Pressure (in)", "Actual", "Historic Avg.", "Record"],
        ["", "", "", ""],
        ["Sea Level Pressure", "29.67", "-", "-"],
    ]

    def test_high_temp(self):
        result = _parse_summary_table(self.SAMPLE_ROWS)
        assert result["high_f"] == 73.0

    def test_low_temp(self):
        result = _parse_summary_table(self.SAMPLE_ROWS)
        assert result["low_f"] == 51.0

    def test_avg_temp(self):
        result = _parse_summary_table(self.SAMPLE_ROWS)
        assert result["avg_f"] == 63.35

    def test_precip(self):
        result = _parse_summary_table(self.SAMPLE_ROWS)
        assert result["precip_in"] == 0.0

    def test_dew_point_high(self):
        result = _parse_summary_table(self.SAMPLE_ROWS)
        assert result["dew_point_high_f"] == 69.0

    def test_dew_point_low(self):
        result = _parse_summary_table(self.SAMPLE_ROWS)
        assert result["dew_point_low_f"] == 42.0

    def test_wind_max(self):
        result = _parse_summary_table(self.SAMPLE_ROWS)
        assert result["wind_speed_max_mph"] == 24.0

    def test_pressure(self):
        result = _parse_summary_table(self.SAMPLE_ROWS)
        assert result["pressure_in"] == 29.67

    def test_visibility(self):
        result = _parse_summary_table(self.SAMPLE_ROWS)
        assert result["visibility_mi"] == 10.0


# ---------------------------------------------------------------------------
# _parse_hourly_row
# ---------------------------------------------------------------------------


class TestParseHourlyRow:
    def test_normal_row(self):
        cells = [
            "12:53 AM", "73 °F", "66 °F", "79 %",
            "S", "15 mph", "22 mph", "29.28 in", "0.0 in", "Cloudy",
        ]
        reading = _parse_hourly_row(cells)
        assert reading is not None
        assert reading.time_local == "12:53 AM"
        assert reading.temp_f == 73.0
        assert reading.dew_point_f == 66.0
        assert reading.humidity_pct == 79.0
        assert reading.wind_dir == "S"
        assert reading.wind_mph == 15.0
        assert reading.wind_gust_mph == 22.0
        assert reading.pressure_in == 29.28
        assert reading.precip_in == 0.0
        assert reading.condition == "Cloudy"

    def test_calm_wind(self):
        cells = [
            "7:26 AM", "69 °F", "68 °F", "96 %",
            "CALM", "0 mph", "0 mph", "29.26 in", "0.0 in", "Cloudy",
        ]
        reading = _parse_hourly_row(cells)
        assert reading is not None
        assert reading.wind_dir == "CALM"
        assert reading.wind_mph == 0.0

    def test_header_row_skipped(self):
        cells = ["Time", "Temperature", "Dew Point", "Humidity",
                 "Wind", "Wind Speed", "Wind Gust", "Pressure", "Precip.", "Condition"]
        assert _parse_hourly_row(cells) is None

    def test_too_few_cells(self):
        assert _parse_hourly_row(["12:53 AM", "73 °F"]) is None

    def test_condition_with_slash(self):
        cells = [
            "10:31 AM", "66 °F", "61 °F", "84 %",
            "N", "22 mph", "32 mph", "29.40 in", "0.0 in", "Light Rain / Windy",
        ]
        reading = _parse_hourly_row(cells)
        assert reading is not None
        assert reading.condition == "Light Rain / Windy"


# ---------------------------------------------------------------------------
# compare_wu_vs_wx
# ---------------------------------------------------------------------------


class TestCompareWuVsWx:
    def test_both_available(self):
        wu = WuDailySummary(station_icao="KAUS", date=date(2026, 4, 18),
                            high_f=73.0, low_f=51.0)
        comp = compare_wu_vs_wx(wu, wx_high_f=75.0, wx_low_f=50.0)
        assert comp["high_f_wu"] == 73.0
        assert comp["high_f_api"] == 75.0
        assert comp["high_f_delta"] == -2.0
        assert comp["low_f_delta"] == 1.0

    def test_api_missing(self):
        wu = WuDailySummary(station_icao="KAUS", date=date(2026, 4, 18),
                            high_f=73.0, low_f=51.0)
        comp = compare_wu_vs_wx(wu, wx_high_f=None, wx_low_f=None)
        assert comp["high_f_delta"] is None
        assert comp["low_f_delta"] is None

    def test_wu_missing(self):
        wu = WuDailySummary(station_icao="KAUS", date=date(2026, 4, 18))
        comp = compare_wu_vs_wx(wu, wx_high_f=75.0, wx_low_f=50.0)
        assert comp["high_f_delta"] is None
