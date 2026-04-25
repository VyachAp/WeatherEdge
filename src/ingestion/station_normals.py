"""Station climatological normals — multi-year mean and std of daily-max
temperature per (station, day-of-year). Read by the probability engine as
a Bayesian prior; written by ``scripts/backfill_station_normals.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db.models import StationNormal

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NormalForDay:
    icao: str
    day_of_year: int
    mean_max_c: float
    std_max_c: float
    sample_years: int
    source: str


# In-process cache. Keyed on (icao, doy). Climatological values are
# stable across the trading day — DOY rolls over at midnight UTC, which
# is also when daily settlement runs and the bot reloads state. Cache
# entries are immutable for the day.
_cache: dict[tuple[str, int], NormalForDay | None] = {}


def _doy_for_lookup(d: date) -> int:
    """Map a calendar date to the canonical DOY used for normals lookups.

    Feb 29 collapses to Feb 28's DOY (60 in non-leap arithmetic) so the
    backfill can ignore leap days entirely without producing missing-data
    holes on Feb 29 lookups.
    """
    doy = d.timetuple().tm_yday
    # On a leap year, March 1 is DOY 61 (it's DOY 60 in non-leap years).
    # We want non-leap-year-equivalent DOYs in storage so the backfill
    # output is consistent across years. So: in a leap year, anything
    # past Feb 29 shifts back by one to drop the extra day.
    if (d.year % 4 == 0 and (d.year % 100 != 0 or d.year % 400 == 0)):
        if doy == 60:  # Feb 29 itself
            return 59  # collapse to Feb 28
        if doy > 60:  # March 1 onward
            return doy - 1
    return doy


async def get_normal(
    session: AsyncSession,
    icao: str,
    target_date: date,
) -> NormalForDay | None:
    """Return the cached climate normal for one (station, calendar date).

    Returns ``None`` when the station has no row in ``station_normals``
    (e.g. the backfill hasn't run for it). The probability engine treats
    that as "no prior available" and degrades to the pre-prior baseline.
    """
    doy = _doy_for_lookup(target_date)
    key = (icao.upper(), doy)
    if key in _cache:
        return _cache[key]

    stmt = select(StationNormal).where(
        StationNormal.station_icao == icao.upper(),
        StationNormal.day_of_year == doy,
    )
    result = await session.execute(stmt)
    row = result.scalar_one_or_none()

    if row is None:
        _cache[key] = None
        return None

    normal = NormalForDay(
        icao=icao.upper(),
        day_of_year=doy,
        mean_max_c=float(row.mean_max_c),
        std_max_c=float(row.std_max_c),
        sample_years=int(row.sample_years),
        source=str(row.source),
    )
    _cache[key] = normal
    return normal


async def upsert_normal(
    session: AsyncSession,
    icao: str,
    day_of_year: int,
    mean_max_c: float,
    std_max_c: float,
    sample_years: int,
    source: str = "openmeteo_archive_era5",
) -> None:
    """Idempotent upsert of one normal row. Used by the backfill script."""
    stmt = pg_insert(StationNormal).values(
        station_icao=icao.upper(),
        day_of_year=day_of_year,
        mean_max_c=mean_max_c,
        std_max_c=std_max_c,
        sample_years=sample_years,
        source=source,
        computed_at=datetime.now(timezone.utc),
    ).on_conflict_do_update(
        constraint="uq_station_normal_doy",
        set_={
            "mean_max_c": mean_max_c,
            "std_max_c": std_max_c,
            "sample_years": sample_years,
            "source": source,
            "computed_at": datetime.now(timezone.utc),
        },
    )
    await session.execute(stmt)
    # Invalidate cache so a freshly-rewritten value is picked up on the
    # next read in the same process.
    _cache.pop((icao.upper(), day_of_year), None)


def clear_cache() -> None:
    """Drop the in-process cache. Used by tests."""
    _cache.clear()
