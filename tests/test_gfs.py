"""Tests for the GEFS ensemble ingestion module."""

from __future__ import annotations

import math
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.ingestion.gfs import (
    FORECAST_HOURS,
    N_MEMBERS,
    RUN_HOURS,
    VARIABLE_MAP,
    VariableDef,
    _cache_path,
    _extract_point_value,
    _extract_wind_speed,
    _is_cache_valid,
    _normalize_lon,
    _persist_forecast,
    _resolve_run_for_valid_time,
    cleanup_stale_cache,
    fetch_gefs_run,
    get_ensemble_stats,
    get_probability,
)


# ---------------------------------------------------------------------------
# Variable mapping
# ---------------------------------------------------------------------------


class TestVariableMap:
    def test_has_temperature(self):
        assert "temperature" in VARIABLE_MAP
        assert VARIABLE_MAP["temperature"].search == "TMP:2 m above ground"

    def test_has_precipitation(self):
        assert "precipitation" in VARIABLE_MAP
        assert VARIABLE_MAP["precipitation"].search == "APCP:surface"

    def test_has_wind_speed(self):
        assert "wind_speed" in VARIABLE_MAP
        vdef = VARIABLE_MAP["wind_speed"]
        assert vdef.search is None
        assert len(vdef.components) == 2
        assert "UGRD:10 m above ground" in vdef.components
        assert "VGRD:10 m above ground" in vdef.components

    def test_variable_def_frozen(self):
        vdef = VariableDef(search="X", unit="Y")
        with pytest.raises(AttributeError):
            vdef.search = "Z"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


class TestCachePath:
    def test_format(self):
        p = _cache_path(date(2026, 4, 7), 0, 24, 5)
        assert "20260407" in str(p)
        assert "00z" in str(p)
        assert "mem05" in str(p)
        assert "f024" in str(p)

    def test_different_members_different_paths(self):
        p1 = _cache_path(date(2026, 1, 1), 12, 6, 0)
        p2 = _cache_path(date(2026, 1, 1), 12, 6, 1)
        assert p1 != p2


class TestIsCacheValid:
    def test_missing_file(self, tmp_path):
        assert _is_cache_valid(tmp_path / "nonexistent.grib2") is False

    def test_stale_file(self, tmp_path):
        f = tmp_path / "old.grib2"
        f.write_bytes(b"data")
        # Set mtime to 48 hours ago
        old_time = time.time() - 48 * 3600
        import os

        os.utime(f, (old_time, old_time))
        assert _is_cache_valid(f) is False

    def test_fresh_file(self, tmp_path):
        f = tmp_path / "fresh.grib2"
        f.write_bytes(b"data")
        assert _is_cache_valid(f) is True


class TestCleanupStaleCache:
    def test_removes_old_files(self, tmp_path):
        old = tmp_path / "sub" / "old.grib2"
        old.parent.mkdir(parents=True)
        old.write_bytes(b"data")
        old_time = time.time() - 48 * 3600
        import os

        os.utime(old, (old_time, old_time))

        fresh = tmp_path / "sub" / "fresh.grib2"
        fresh.write_bytes(b"data")

        with patch("src.ingestion.gfs.CACHE_DIR", tmp_path):
            removed = cleanup_stale_cache()

        assert removed == 1
        assert not old.exists()
        assert fresh.exists()


# ---------------------------------------------------------------------------
# Longitude normalization
# ---------------------------------------------------------------------------


class TestNormalizeLon:
    def test_positive(self):
        assert _normalize_lon(90.0) == 90.0

    def test_negative(self):
        assert _normalize_lon(-90.0) == 270.0

    def test_zero(self):
        assert _normalize_lon(0.0) == 0.0

    def test_wrap(self):
        assert _normalize_lon(360.0) == 0.0


# ---------------------------------------------------------------------------
# Run resolution
# ---------------------------------------------------------------------------


class TestResolveRun:
    def test_same_day(self):
        # Valid time 2026-04-07 18:00 with "now" well past availability
        valid = datetime(2026, 4, 7, 18, 0)
        with patch("src.ingestion.gfs.datetime") as mock_dt:
            mock_dt.utcnow.return_value = datetime(2026, 4, 8, 12, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            rd, rh, fxx = _resolve_run_for_valid_time(valid)

        assert rd == date(2026, 4, 7)
        assert rh in RUN_HOURS
        assert fxx in FORECAST_HOURS

    def test_fxx_in_forecast_hours(self):
        valid = datetime(2026, 4, 8, 6, 0)
        with patch("src.ingestion.gfs.datetime") as mock_dt:
            mock_dt.utcnow.return_value = datetime(2026, 4, 8, 12, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            _, _, fxx = _resolve_run_for_valid_time(valid)

        assert fxx in FORECAST_HOURS

    def test_falls_back_when_recent(self):
        # Valid time is in the future, now is close to valid_time
        valid = datetime(2026, 4, 8, 12, 0)
        with patch("src.ingestion.gfs.datetime") as mock_dt:
            mock_dt.utcnow.return_value = datetime(2026, 4, 8, 10, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            rd, rh, fxx = _resolve_run_for_valid_time(valid)

        # Should find a run from before the availability delay
        run_dt = datetime(rd.year, rd.month, rd.day, rh)
        assert run_dt + timedelta(hours=5) <= datetime(2026, 4, 8, 10, 0)


# ---------------------------------------------------------------------------
# Point extraction (mocked xarray)
# ---------------------------------------------------------------------------


class TestExtractPointValue:
    def test_extracts_value(self, tmp_path):
        fake_grib = tmp_path / "test.grib2"
        fake_grib.write_bytes(b"dummy")

        fake_da = MagicMock()
        fake_da.sel.return_value.values = np.float64(295.5)
        fake_da.dims = ("latitude", "longitude")

        fake_ds = MagicMock()
        fake_ds.data_vars = ["t2m"]
        fake_ds.__getitem__ = MagicMock(return_value=fake_da)

        with patch("src.ingestion.gfs.xr.open_dataset", return_value=fake_ds):
            val = _extract_point_value(fake_grib, 40.0, -105.0)

        assert val == pytest.approx(295.5)
        fake_da.sel.assert_called_once()

    def test_returns_none_on_missing_file(self, tmp_path):
        val = _extract_point_value(tmp_path / "missing.grib2", 40.0, -105.0)
        assert val is None

    def test_returns_none_on_empty_dataset(self, tmp_path):
        fake_grib = tmp_path / "empty.grib2"
        fake_grib.write_bytes(b"dummy")

        fake_ds = MagicMock()
        fake_ds.data_vars = []

        with patch("src.ingestion.gfs.xr.open_dataset", return_value=fake_ds):
            val = _extract_point_value(fake_grib, 40.0, -105.0)

        assert val is None


# ---------------------------------------------------------------------------
# Wind speed
# ---------------------------------------------------------------------------


class TestExtractWindSpeed:
    def test_uv_to_magnitude(self, tmp_path):
        u_path = tmp_path / "u.grib2"
        v_path = tmp_path / "v.grib2"
        u_path.write_bytes(b"u")
        v_path.write_bytes(b"v")

        with patch(
            "src.ingestion.gfs._extract_point_value",
            side_effect=[3.0, 4.0],
        ):
            val = _extract_wind_speed(u_path, v_path, 40.0, -105.0)

        assert val == pytest.approx(5.0)

    def test_missing_component(self, tmp_path):
        u_path = tmp_path / "u.grib2"
        v_path = tmp_path / "v.grib2"
        u_path.write_bytes(b"u")
        v_path.write_bytes(b"v")

        with patch(
            "src.ingestion.gfs._extract_point_value",
            side_effect=[3.0, None],
        ):
            val = _extract_wind_speed(u_path, v_path, 40.0, -105.0)

        assert val is None


# ---------------------------------------------------------------------------
# Ensemble stats (mocked extraction)
# ---------------------------------------------------------------------------

SAMPLE_MEMBERS = [float(x) for x in range(290, 321)]  # 31 values: 290..320


class TestGetEnsembleStats:
    @pytest.mark.asyncio
    async def test_basic_stats(self):
        with (
            patch("src.ingestion.gfs._resolve_run_for_valid_time", return_value=(date(2026, 4, 7), 0, 24)),
            patch("src.ingestion.gfs._extract_point_value", side_effect=SAMPLE_MEMBERS),
            patch("src.ingestion.gfs._persist_forecast", new_callable=AsyncMock),
        ):
            stats = await get_ensemble_stats(
                40.0,
                -105.0,
                datetime(2026, 4, 8, 0, 0),
                "temperature",
                session=AsyncMock(),
            )

        assert stats["n_members"] == 31
        assert stats["mean"] == pytest.approx(np.mean(SAMPLE_MEMBERS))
        assert stats["std"] == pytest.approx(np.std(SAMPLE_MEMBERS, ddof=1))
        assert stats["min"] == pytest.approx(290.0)
        assert stats["max"] == pytest.approx(320.0)
        assert stats["iqr"] == pytest.approx(stats["q75"] - stats["q25"])
        assert len(stats["members"]) == 31

    @pytest.mark.asyncio
    async def test_missing_members_warns(self, caplog):
        # Only 5 members return values, rest return None
        side_effects = [float(x) for x in range(300, 305)] + [None] * 26

        with (
            patch("src.ingestion.gfs._resolve_run_for_valid_time", return_value=(date(2026, 4, 7), 0, 24)),
            patch("src.ingestion.gfs._extract_point_value", side_effect=side_effects),
            patch("src.ingestion.gfs._persist_forecast", new_callable=AsyncMock),
        ):
            import logging

            with caplog.at_level(logging.WARNING, logger="src.ingestion.gfs"):
                stats = await get_ensemble_stats(
                    40.0,
                    -105.0,
                    datetime(2026, 4, 8, 0, 0),
                    "temperature",
                    session=AsyncMock(),
                )

        assert stats["n_members"] == 5
        assert "Only 5" in caplog.text

    @pytest.mark.asyncio
    async def test_no_members(self):
        with (
            patch("src.ingestion.gfs._resolve_run_for_valid_time", return_value=(date(2026, 4, 7), 0, 24)),
            patch("src.ingestion.gfs._extract_point_value", return_value=None),
            patch("src.ingestion.gfs._persist_forecast", new_callable=AsyncMock),
        ):
            stats = await get_ensemble_stats(
                40.0,
                -105.0,
                datetime(2026, 4, 8, 0, 0),
                "temperature",
                session=AsyncMock(),
            )

        assert stats["n_members"] == 0
        assert stats["mean"] is None
        assert stats["members"] == []

    @pytest.mark.asyncio
    async def test_unknown_variable_raises(self):
        with pytest.raises(ValueError, match="Unknown variable"):
            await get_ensemble_stats(40.0, -105.0, datetime(2026, 4, 8), "humidity")


# ---------------------------------------------------------------------------
# Probability
# ---------------------------------------------------------------------------


class TestGetProbability:
    @pytest.mark.asyncio
    async def test_above(self):
        members = list(range(1, 32))  # 1..31
        stats = {
            "members": members,
            "mean": 16.0,
            "std": 9.0,
            "min": 1.0,
            "max": 31.0,
            "q25": 8.5,
            "q75": 23.5,
            "iqr": 15.0,
            "n_members": 31,
            "run_time": datetime(2026, 4, 7),
            "valid_time": datetime(2026, 4, 8),
        }
        with patch("src.ingestion.gfs.get_ensemble_stats", new_callable=AsyncMock, return_value=stats):
            prob = await get_probability(40.0, -105.0, datetime(2026, 4, 8), "temperature", 20.0, "above")

        # Members above 20: 21, 22, ..., 31 = 11 members
        assert prob == pytest.approx(11 / 31)

    @pytest.mark.asyncio
    async def test_below(self):
        members = list(range(1, 32))
        stats = {
            "members": members,
            "mean": 16.0,
            "std": 9.0,
            "min": 1.0,
            "max": 31.0,
            "q25": 8.5,
            "q75": 23.5,
            "iqr": 15.0,
            "n_members": 31,
            "run_time": datetime(2026, 4, 7),
            "valid_time": datetime(2026, 4, 8),
        }
        with patch("src.ingestion.gfs.get_ensemble_stats", new_callable=AsyncMock, return_value=stats):
            prob = await get_probability(40.0, -105.0, datetime(2026, 4, 8), "temperature", 10.0, "below")

        # Members below 10: 1..9 = 9 members
        assert prob == pytest.approx(9 / 31)

    @pytest.mark.asyncio
    async def test_all_above(self):
        stats = {
            "members": [100.0] * 31,
            "mean": 100.0,
            "std": 0.0,
            "min": 100.0,
            "max": 100.0,
            "q25": 100.0,
            "q75": 100.0,
            "iqr": 0.0,
            "n_members": 31,
            "run_time": datetime(2026, 4, 7),
            "valid_time": datetime(2026, 4, 8),
        }
        with patch("src.ingestion.gfs.get_ensemble_stats", new_callable=AsyncMock, return_value=stats):
            prob = await get_probability(40.0, -105.0, datetime(2026, 4, 8), "temperature", 50.0, "above")

        assert prob == 1.0

    @pytest.mark.asyncio
    async def test_none_above(self):
        stats = {
            "members": [10.0] * 31,
            "mean": 10.0,
            "std": 0.0,
            "min": 10.0,
            "max": 10.0,
            "q25": 10.0,
            "q75": 10.0,
            "iqr": 0.0,
            "n_members": 31,
            "run_time": datetime(2026, 4, 7),
            "valid_time": datetime(2026, 4, 8),
        }
        with patch("src.ingestion.gfs.get_ensemble_stats", new_callable=AsyncMock, return_value=stats):
            prob = await get_probability(40.0, -105.0, datetime(2026, 4, 8), "temperature", 50.0, "above")

        assert prob == 0.0

    @pytest.mark.asyncio
    async def test_no_members_returns_zero(self):
        stats = {
            "members": [],
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "q25": None,
            "q75": None,
            "iqr": None,
            "n_members": 0,
            "run_time": datetime(2026, 4, 7),
            "valid_time": datetime(2026, 4, 8),
        }
        with patch("src.ingestion.gfs.get_ensemble_stats", new_callable=AsyncMock, return_value=stats):
            prob = await get_probability(40.0, -105.0, datetime(2026, 4, 8), "temperature", 50.0, "above")

        assert prob == 0.0

    @pytest.mark.asyncio
    async def test_invalid_operator_raises(self):
        with pytest.raises(ValueError, match="operator"):
            await get_probability(40.0, -105.0, datetime(2026, 4, 8), "temperature", 50.0, "equals")


# ---------------------------------------------------------------------------
# DB persistence (mocked session)
# ---------------------------------------------------------------------------


class TestPersistForecast:
    @pytest.mark.asyncio
    async def test_creates_row(self):
        session = AsyncMock()
        stats = {
            "mean": 300.0,
            "std": 5.0,
            "members": [295.0, 300.0, 305.0],
        }
        await _persist_forecast(
            session,
            run_time=datetime(2026, 4, 7, 0),
            valid_time=datetime(2026, 4, 8, 0),
            lat=40.0,
            lon=-105.0,
            variable="temperature",
            stats=stats,
        )

        session.add.assert_called_once()
        forecast = session.add.call_args[0][0]
        assert forecast.model_source == "gefs"
        assert forecast.variable == "temperature"
        assert forecast.ensemble_mean == 300.0
        assert forecast.ensemble_std == 5.0
        assert forecast.ensemble_members == [295.0, 300.0, 305.0]
        assert forecast.location_lat == 40.0
        assert forecast.location_lon == -105.0
        session.commit.assert_awaited_once()


# ---------------------------------------------------------------------------
# fetch_gefs_run (mocked downloads)
# ---------------------------------------------------------------------------


class TestFetchGefsRun:
    @pytest.mark.asyncio
    async def test_invalid_run_hour_raises(self):
        with pytest.raises(ValueError, match="run_hour"):
            await fetch_gefs_run(date(2026, 4, 7), run_hour=3)

    @pytest.mark.asyncio
    async def test_handles_partial_failure(self):
        call_count = 0

        async def mock_download(rd, rh, fxx, member, search):
            nonlocal call_count
            call_count += 1
            if member == 0:
                return Path(f"/tmp/fake_{fxx}_{member}.grib2")
            return None

        with patch("src.ingestion.gfs._async_download_member", side_effect=mock_download):
            results = await fetch_gefs_run(date(2026, 4, 7), 0)

        # Only member 0 succeeded for each (fxx, search) combo
        assert len(results) > 0
        assert all(m == 0 for (_, m, _) in results.keys())

    @pytest.mark.asyncio
    async def test_returns_all_on_success(self):
        async def mock_download(rd, rh, fxx, member, search):
            return Path(f"/tmp/fake_{fxx}_{member}_{search}.grib2")

        with patch("src.ingestion.gfs._async_download_member", side_effect=mock_download):
            results = await fetch_gefs_run(date(2026, 4, 7), 0)

        # Should have entries for all fxx x members x unique searches
        unique_searches = set()
        for vdef in VARIABLE_MAP.values():
            if vdef.search:
                unique_searches.add(vdef.search)
            unique_searches.update(vdef.components)

        expected = len(FORECAST_HOURS) * N_MEMBERS * len(unique_searches)
        assert len(results) == expected
