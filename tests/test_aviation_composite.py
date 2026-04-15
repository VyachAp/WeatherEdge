"""Tests for composite aviation functions and new types."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from src.ingestion.aviation._caching import clear_all_caches
from src.ingestion.aviation._composite import (
    _compute_trend_from_observations,
    _extract_metar_precip,
)
from src.ingestion.aviation._conversions import (
    c_to_f,
    f_to_c,
    kts_to_mph,
    mph_to_kts,
    m_to_miles,
    miles_to_m,
    hpa_to_inhg,
    mm_to_inches,
    inches_to_mm,
)
from src.ingestion.aviation._types import (
    MinuteObs,
    Observation,
    PrecipAccum,
    SynopObs,
    TempTrend,
    WeatherBriefing,
)


@pytest.fixture(autouse=True)
def _clear():
    clear_all_caches()
    yield
    clear_all_caches()


# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------


class TestConversions:
    def test_c_to_f_basic(self):
        assert c_to_f(0.0) == pytest.approx(32.0)
        assert c_to_f(100.0) == pytest.approx(212.0)
        assert c_to_f(-40.0) == pytest.approx(-40.0)

    def test_c_to_f_none(self):
        assert c_to_f(None) is None

    def test_f_to_c_basic(self):
        assert f_to_c(32.0) == pytest.approx(0.0)
        assert f_to_c(212.0) == pytest.approx(100.0)

    def test_f_to_c_none(self):
        assert f_to_c(None) is None

    def test_roundtrip_temp(self):
        for t in [-40, 0, 20, 37, 100]:
            assert f_to_c(c_to_f(float(t))) == pytest.approx(float(t), abs=0.01)

    def test_kts_to_mph(self):
        assert kts_to_mph(100.0) == pytest.approx(115.078, abs=0.1)
        assert kts_to_mph(None) is None

    def test_mph_to_kts(self):
        assert mph_to_kts(115.078) == pytest.approx(100.0, abs=0.1)

    def test_m_to_miles(self):
        assert m_to_miles(1609.34) == pytest.approx(1.0, abs=0.01)
        assert m_to_miles(None) is None

    def test_miles_to_m(self):
        assert miles_to_m(1.0) == pytest.approx(1609.34, abs=1)

    def test_hpa_to_inhg(self):
        assert hpa_to_inhg(1013.25) == pytest.approx(29.92, abs=0.01)

    def test_mm_to_inches(self):
        assert mm_to_inches(25.4) == pytest.approx(1.0, abs=0.01)

    def test_inches_to_mm(self):
        assert inches_to_mm(1.0) == pytest.approx(25.4, abs=0.01)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class TestDataTypes:
    def test_observation_frozen(self):
        obs = Observation(
            station_icao="KPHX",
            observed_at=datetime.now(timezone.utc),
            source="test",
        )
        with pytest.raises(AttributeError):
            obs.station_icao = "KORD"  # type: ignore

    def test_synop_obs_frozen(self):
        obs = SynopObs(
            wmo_id="72278",
            observed_at=datetime.now(timezone.utc),
        )
        assert obs.wmo_id == "72278"

    def test_temp_trend_defaults(self):
        trend = TempTrend()
        assert trend.current_f is None
        assert trend.trend_direction == "unknown"
        assert trend.rate_per_hour == 0.0
        assert trend.source == "metar"

    def test_precip_accum_defaults(self):
        accum = PrecipAccum(station="KPHX")
        assert accum.total_mm is None
        assert accum.source == "metar"

    def test_weather_briefing_mutable(self):
        briefing = WeatherBriefing(station="KPHX")
        briefing.speci_events.append({"test": True})
        assert len(briefing.speci_events) == 1


# ---------------------------------------------------------------------------
# Trend computation
# ---------------------------------------------------------------------------


class TestTrendComputation:
    def _make_obs(self, temp_f: float, minutes_ago: int) -> Observation:
        from datetime import timedelta
        return Observation(
            station_icao="KPHX",
            observed_at=datetime.now(timezone.utc) - timedelta(minutes=minutes_ago),
            temp_f=temp_f,
            source="test",
        )

    def test_rising_trend(self):
        obs = [
            self._make_obs(80.0, 180),
            self._make_obs(83.0, 120),
            self._make_obs(86.0, 60),
            self._make_obs(89.0, 0),
        ]
        trend = _compute_trend_from_observations(obs, 3.0)
        assert trend.trend_direction == "rising"
        assert trend.rate_per_hour > 0
        assert trend.current_f == 89.0
        assert trend.min_f == 80.0
        assert trend.max_f == 89.0

    def test_falling_trend(self):
        obs = [
            self._make_obs(89.0, 180),
            self._make_obs(86.0, 120),
            self._make_obs(83.0, 60),
            self._make_obs(80.0, 0),
        ]
        trend = _compute_trend_from_observations(obs, 3.0)
        assert trend.trend_direction == "falling"
        assert trend.rate_per_hour < 0

    def test_steady_trend(self):
        obs = [
            self._make_obs(85.0, 180),
            self._make_obs(85.1, 120),
            self._make_obs(84.9, 60),
            self._make_obs(85.0, 0),
        ]
        trend = _compute_trend_from_observations(obs, 3.0)
        assert trend.trend_direction == "steady"

    def test_single_observation(self):
        obs = [self._make_obs(85.0, 0)]
        trend = _compute_trend_from_observations(obs, 1.0)
        assert trend.current_f == 85.0
        assert trend.trend_direction == "steady"
        assert trend.observation_count == 1

    def test_empty_observations(self):
        trend = _compute_trend_from_observations([], 1.0)
        assert trend.current_f is None
        assert trend.source == "metar"


# ---------------------------------------------------------------------------
# Precipitation extraction
# ---------------------------------------------------------------------------


class TestPrecipExtraction:
    def test_extract_metar_precip(self):
        obs = [
            Observation(
                station_icao="KPHX",
                observed_at=datetime.now(timezone.utc),
                raw_metar="KPHX 091853Z ... RMK AO2 P0010",
                source="test",
            ),
            Observation(
                station_icao="KPHX",
                observed_at=datetime.now(timezone.utc),
                raw_metar="KPHX 091753Z ... RMK AO2 P0025",
                source="test",
            ),
        ]
        total_mm = _extract_metar_precip(obs)
        assert total_mm is not None
        # 0.10 + 0.25 = 0.35 inches = ~8.89 mm
        assert total_mm == pytest.approx(8.89, abs=0.1)

    def test_extract_metar_precip_none(self):
        obs = [
            Observation(
                station_icao="KPHX",
                observed_at=datetime.now(timezone.utc),
                raw_metar="KPHX 091853Z 22012KT 10SM CLR",
                source="test",
            ),
        ]
        assert _extract_metar_precip(obs) is None

    def test_extract_metar_precip_no_raw(self):
        obs = [
            Observation(
                station_icao="KPHX",
                observed_at=datetime.now(timezone.utc),
                source="test",
            ),
        ]
        assert _extract_metar_precip(obs) is None
