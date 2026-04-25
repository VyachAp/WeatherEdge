"""Tests for the climate-normal DAO + DOY canonicalization."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.ingestion.station_normals import (
    NormalForDay,
    _doy_for_lookup,
    clear_cache,
    get_normal,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_cache()
    yield
    clear_cache()


class TestDoyCanonicalization:
    """Feb 29 collapses to Feb 28 (DOY 59) and post-Feb-29 days in leap
    years shift back by one so DOYs are consistent across leap and
    non-leap years."""

    def test_non_leap_year_passthrough(self):
        # 2025 is not a leap year. DOYs are unchanged.
        assert _doy_for_lookup(date(2025, 1, 1)) == 1
        assert _doy_for_lookup(date(2025, 2, 28)) == 59
        assert _doy_for_lookup(date(2025, 3, 1)) == 60
        assert _doy_for_lookup(date(2025, 12, 31)) == 365

    def test_leap_year_pre_feb_29_passthrough(self):
        assert _doy_for_lookup(date(2024, 1, 1)) == 1
        assert _doy_for_lookup(date(2024, 2, 28)) == 59

    def test_leap_year_feb_29_collapses_to_feb_28(self):
        assert _doy_for_lookup(date(2024, 2, 29)) == 59

    def test_leap_year_post_feb_29_shifts_back(self):
        # March 1 in a leap year is DOY 61; we want it to map to 60
        # (= March 1 in non-leap year).
        assert _doy_for_lookup(date(2024, 3, 1)) == 60
        assert _doy_for_lookup(date(2024, 12, 31)) == 365


class TestGetNormal:
    @pytest.mark.asyncio
    async def test_returns_none_when_no_row(self):
        session = MagicMock()
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=result)
        normal = await get_normal(session, "KPHX", date(2025, 5, 15))
        assert normal is None

    @pytest.mark.asyncio
    async def test_returns_normal_for_day_when_row_exists(self):
        session = MagicMock()
        row = MagicMock()
        row.mean_max_c = 35.5
        row.std_max_c = 2.8
        row.sample_years = 10
        row.source = "openmeteo_archive_era5"
        result = MagicMock()
        result.scalar_one_or_none.return_value = row
        session.execute = AsyncMock(return_value=result)
        normal = await get_normal(session, "kphx", date(2025, 5, 15))
        assert normal is not None
        assert isinstance(normal, NormalForDay)
        assert normal.icao == "KPHX"
        assert normal.day_of_year == _doy_for_lookup(date(2025, 5, 15))
        assert normal.mean_max_c == 35.5
        assert normal.std_max_c == 2.8
        assert normal.sample_years == 10

    @pytest.mark.asyncio
    async def test_cache_avoids_second_db_hit(self):
        session = MagicMock()
        row = MagicMock()
        row.mean_max_c = 30.0
        row.std_max_c = 3.0
        row.sample_years = 8
        row.source = "openmeteo_archive_era5"
        result = MagicMock()
        result.scalar_one_or_none.return_value = row
        session.execute = AsyncMock(return_value=result)
        # First call hits DB.
        await get_normal(session, "KPHX", date(2025, 5, 15))
        # Second call should hit the cache, not the DB.
        await get_normal(session, "KPHX", date(2025, 5, 15))
        assert session.execute.await_count == 1

    @pytest.mark.asyncio
    async def test_feb_29_resolves_via_feb_28_doy(self):
        session = MagicMock()
        row = MagicMock()
        row.mean_max_c = 10.0
        row.std_max_c = 2.5
        row.sample_years = 9
        row.source = "openmeteo_archive_era5"
        result = MagicMock()
        result.scalar_one_or_none.return_value = row
        session.execute = AsyncMock(return_value=result)
        normal = await get_normal(session, "EGLL", date(2024, 2, 29))
        # Feb 29 collapses to DOY 59 (Feb 28 of the same year).
        assert normal is not None
        assert normal.day_of_year == 59
