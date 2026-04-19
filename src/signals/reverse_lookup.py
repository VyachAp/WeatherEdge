"""Backward resolution: find Polymarket markets from weather data.

Given a city, ICAO station, or WX observation, query the local markets
table (populated by scan_and_ingest every 15 min) to find matching
weather markets.  This avoids the slow Gamma API pagination that
``bet search`` relies on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Market

logger = logging.getLogger(__name__)


@dataclass
class MarketMatch:
    """A market matched by reverse lookup, enriched with observation context."""

    market: Market
    observed_value: float | None = None  # current reading (F for temp)
    distance_to_threshold: float | None = None
    direction: str = ""  # "approaching", "crossed", "far", ""


# ---------------------------------------------------------------------------
# Core queries
# ---------------------------------------------------------------------------


async def find_markets_for_city(
    session: AsyncSession,
    city: str,
    *,
    variable: str | None = None,
    date_str: str | None = None,
    min_liquidity: float = 0.0,
) -> list[Market]:
    """Find active weather markets for a given city name.

    Matches against ``Market.parsed_location`` (case-insensitive).
    """
    now = datetime.now(timezone.utc)
    filters = [
        Market.parsed_location.isnot(None),
        Market.end_date > now,
        Market.parsed_location.ilike(f"%{city}%"),
    ]

    if variable:
        filters.append(Market.parsed_variable == variable)

    if date_str:
        filters.append(Market.parsed_target_date.ilike(f"%{date_str}%"))

    if min_liquidity > 0:
        filters.append(Market.liquidity >= min_liquidity)

    stmt = select(Market).where(*filters).order_by(Market.end_date.asc())
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def find_markets_for_station(
    session: AsyncSession,
    icao: str,
    *,
    variable: str | None = None,
    hours_ahead: float = 24.0,
) -> list[Market]:
    """Find active markets whose location maps to the given ICAO station."""
    from src.signals.mapper import cities_for_icao

    city_names = cities_for_icao(icao)
    if not city_names:
        logger.debug("No city names found for ICAO %s", icao)
        return []

    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=hours_ahead)

    location_filters = [Market.parsed_location.ilike(f"%{c}%") for c in city_names]

    filters = [
        Market.parsed_location.isnot(None),
        Market.end_date > now,
        Market.end_date <= cutoff,
        or_(*location_filters),
    ]

    if variable:
        filters.append(Market.parsed_variable == variable)

    stmt = select(Market).where(*filters).order_by(Market.end_date.asc())
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def find_markets_for_observation(
    session: AsyncSession,
    station_icao: str,
    observed_temp_f: float,
    *,
    hours_ahead: float = 24.0,
) -> list[MarketMatch]:
    """Find markets for a station and compute distance to threshold.

    Returns matches sorted by proximity to threshold (closest first).
    """
    markets = await find_markets_for_station(
        session, station_icao, hours_ahead=hours_ahead,
    )

    matches: list[MarketMatch] = []
    for m in markets:
        if m.parsed_threshold is None:
            matches.append(MarketMatch(market=m, observed_value=observed_temp_f))
            continue

        distance = observed_temp_f - m.parsed_threshold
        op = (m.parsed_operator or "").lower()

        if op in ("above", "at_least", "exceed"):
            if distance >= 0:
                direction = "crossed"
            elif abs(distance) <= 5:
                direction = "approaching"
            else:
                direction = "far"
        elif op in ("below", "at_most"):
            if distance <= 0:
                direction = "crossed"
            elif abs(distance) <= 5:
                direction = "approaching"
            else:
                direction = "far"
        else:
            direction = "approaching" if abs(distance) <= 5 else "far"

        matches.append(MarketMatch(
            market=m,
            observed_value=observed_temp_f,
            distance_to_threshold=distance,
            direction=direction,
        ))

    matches.sort(key=lambda mm: abs(mm.distance_to_threshold or 999))
    return matches


async def find_markets_for_event(
    session: AsyncSession,
    station_icao: str,
    threshold_f: float,
    *,
    hours_ahead: float = 48.0,
) -> list[Market]:
    """Find markets relevant to a WX threshold event.

    Looks for markets at the same station whose threshold is within 10F
    of the event threshold, extending the search window beyond the
    normal 12h ultra-short range.
    """
    markets = await find_markets_for_station(
        session, station_icao, variable="temperature", hours_ahead=hours_ahead,
    )

    relevant = []
    for m in markets:
        if m.parsed_threshold is None:
            relevant.append(m)
            continue
        if abs(m.parsed_threshold - threshold_f) <= 10:
            relevant.append(m)

    return relevant
