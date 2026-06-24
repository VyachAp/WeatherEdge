"""Continuous resolver ground-truth per station-day (Phase 1, 2026-06-24).

``market_resolution`` (M3) records, per *market*, the daily-max **bound** one
YES/NO outcome implies. Most outcomes (bracket/range NO, an un-sandwiched
threshold NO) imply only a one-sided or no bound, so ``divergence_f`` is
unmeasurable on ~67% of settled rows. This module closes the gap by
**intersecting the stored bounds across a whole station-day's market ladder**
into one continuous resolved-max estimate, then comparing it to our
routine-METAR daily max — the literal "our observation vs how the market
actually closed" signal the self-improvement loop wants.

Pure logic (``back_solve_resolved_max``) is separated from the best-effort DB
upsert so it can be unit-tested without a database. The settlement-path wiring
is the caller's responsibility. Reads only already-labeled
``market_resolutions`` rows — **no new on-chain calls**.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db.models import MarketResolution, StationDayResolution

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_RESOLVED_SOURCE = "bound_intersection"

# A sandwiched interval wider than this (°F) is too loose for a trustworthy
# point estimate — keep the bounds but emit no point (and no divergence).
_MAX_INTERVAL_WIDTH_F = 12.0


def back_solve_resolved_max(
    bounds: list[tuple[float | None, float | None]],
) -> tuple[float | None, float | None, float | None, int]:
    """Intersect per-market ``(lower, upper)`` bounds into one resolved-max window.

    Each settled market on a station-day implies a bound on the day's max
    (``market_resolution.implied_max_bounds``, already stored per row). The
    resolved max must satisfy *all* of them simultaneously, so the tightest
    consistent window is ``[max(lowers), min(uppers)]``. Returns
    ``(lower_f, upper_f, point_f, n_buckets)`` where:

    * ``lower_f`` / ``upper_f`` — the intersected window edges (either may be
      ``None`` when no market bounds that side).
    * ``point_f`` — the window midpoint, a continuous estimate, only when BOTH
      edges are finite, the window is non-inverted, and its width is within
      ``_MAX_INTERVAL_WIDTH_F``; else ``None``.
    * ``n_buckets`` — how many markets contributed at least one finite edge
      (a confidence proxy).

    Inverted windows (``upper < lower``, i.e. contradictory labels) keep their
    raw edges but yield ``point_f = None``.
    """
    lowers = [lo for lo, _hi in bounds if lo is not None]
    uppers = [hi for _lo, hi in bounds if hi is not None]
    n_buckets = sum(1 for lo, hi in bounds if lo is not None or hi is not None)

    lower_f = max(lowers) if lowers else None
    upper_f = min(uppers) if uppers else None

    point_f: float | None = None
    if lower_f is not None and upper_f is not None:
        width = upper_f - lower_f
        if 0.0 <= width <= _MAX_INTERVAL_WIDTH_F:
            point_f = round((lower_f + upper_f) / 2.0, 2)

    return lower_f, upper_f, point_f, n_buckets


def _divergence_point(
    routine_max_f: float | None, point_f: float | None
) -> float | None:
    """Signed °F gap: our routine max − the resolved point (positive = hotter)."""
    if routine_max_f is None or point_f is None:
        return None
    return round(routine_max_f - point_f, 2)


async def record_station_day_resolution(
    session: "AsyncSession",
    *,
    station_icao: str,
    parsed_location: str | None,
    target_date_local: "date",
    unit: str | None,
    bounds: list[tuple[float | None, float | None]],
    routine_metar_max_f: float | None = None,
) -> None:
    """Upsert one ``StationDayResolution`` row (keyed on station-day).

    ``bounds`` is the list of ``(resolved_max_lower_f, resolved_max_upper_f)``
    pairs from that station-day's ``market_resolutions`` rows. Best-effort:
    never raises. Does not commit — the caller batches it.
    """
    try:
        lower_f, upper_f, point_f, n_buckets = back_solve_resolved_max(bounds)
        values = dict(
            station_icao=station_icao,
            parsed_location=parsed_location,
            target_date_local=target_date_local,
            unit=unit,
            resolved_max_lower_f=lower_f,
            resolved_max_upper_f=upper_f,
            resolved_max_point_f=point_f,
            resolved_source=_RESOLVED_SOURCE,
            n_buckets_resolved=n_buckets,
            routine_metar_max_f=routine_metar_max_f,
            divergence_point_f=_divergence_point(routine_metar_max_f, point_f),
        )
        stmt = (
            pg_insert(StationDayResolution)
            .values(**values)
            .on_conflict_do_update(
                constraint="uq_stationday_resolution",
                set_={
                    k: v for k, v in values.items()
                    if k not in ("station_icao", "target_date_local")
                },
            )
        )
        await session.execute(stmt)
    except Exception:
        return


async def resolve_station_days(
    session: "AsyncSession",
    *,
    lookback_days: int = 45,
) -> int:
    """Back-solve a ``StationDayResolution`` for every recently-resolved station-day.

    Enumerates distinct ``(station_icao, target_date_local)`` from
    ``market_resolutions`` within ``lookback_days``, gathers each group's stored
    bounds + (shared) routine max, and upserts the intersected resolved-max
    window. Runs at daily settlement **after** the M3 label pass + straggler
    sweep so the routine max is already filled. Idempotent (upsert). Best-effort:
    never raises, never blocks settlement. Returns the number of station-days
    written. Does not commit — the caller batches it.
    """
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        rows = (
            await session.execute(
                select(
                    MarketResolution.station_icao,
                    MarketResolution.target_date_local,
                    MarketResolution.parsed_location,
                    MarketResolution.unit,
                    MarketResolution.resolved_max_lower_f,
                    MarketResolution.resolved_max_upper_f,
                    MarketResolution.routine_metar_max_f,
                ).where(
                    MarketResolution.station_icao.isnot(None),
                    MarketResolution.target_date_local.isnot(None),
                    MarketResolution.resolved_at >= cutoff,
                )
            )
        ).all()

        # Group by station-day.
        groups: dict[tuple[str, "date"], dict] = {}
        for icao, target, loc, unit, lo, hi, rmax in rows:
            key = (icao, target)
            g = groups.setdefault(
                key,
                {"parsed_location": loc, "unit": unit, "bounds": [],
                 "routine_max": None},
            )
            g["bounds"].append((lo, hi))
            # The routine max is shared across the station-day; take any
            # non-null (rows agree once the settlement backfill has run).
            if g["routine_max"] is None and rmax is not None:
                g["routine_max"] = rmax

        written = 0
        for (icao, target), g in groups.items():
            await record_station_day_resolution(
                session,
                station_icao=icao,
                parsed_location=g["parsed_location"],
                target_date_local=target,
                unit=g["unit"],
                bounds=g["bounds"],
                routine_metar_max_f=g["routine_max"],
            )
            written += 1
        return written
    except Exception:
        return 0
