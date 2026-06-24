"""Market-resolution ground-truth telemetry (M3) — Phase 0.

De-circularises filter tuning. ``evals-report`` today scores candidates
against our own routine-METAR daily max — the same source that feeds our
conviction, which is circular for °C cities (we read systematically
hotter than Polymarket's resolver; that's what killed the removed
``range_overshoot`` lock). This module persists, per settled
market, the resolved YES/NO outcome and the daily-max **bound** it
implies, so Phase 3 can measure the signed per-station divergence between
our observation and the actual resolver.

Pure logic (``implied_max_bounds`` / ``divergence_f``) is separated from
the best-effort DB upsert so it can be unit-tested without a database.
The settlement-path wiring is the caller's responsibility.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db.models import MarketResolution, MetarObservation
from src.execution.binary_market import market_range_f, market_unit
from src.signals.mapper import icao_timezone

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


def implied_max_bounds(
    market, yes_won: bool
) -> tuple[float | None, float | None]:
    """Lower/upper °F bound on the resolved daily max implied by an outcome.

    * bracket / range / exactly (window ``[lo, hi]``): YES → max ∈ [lo, hi]
      (a pinned interval); NO → no tight bound (max is simply outside it).
    * ``above`` / ``at_least`` X: YES → max ≥ X (lower bound); NO → max < X
      (upper bound).
    * ``below`` / ``at_most`` X: YES → max ≤ X (upper bound); NO → max > X
      (lower bound).

    Returns ``(lower_f, upper_f)``; either element may be ``None``.
    """
    rng = market_range_f(market)
    if rng is not None:
        lo, hi = float(rng[0]), float(rng[1])
        return (lo, hi) if yes_won else (None, None)

    op = market.parsed_operator
    thr = market.parsed_threshold
    if thr is None or op is None:
        return None, None
    thr = float(thr)
    if op in ("above", "at_least"):
        return (thr, None) if yes_won else (None, thr)
    if op in ("below", "at_most"):
        return (None, thr) if yes_won else (thr, None)
    return None, None


def divergence_f(
    routine_max_f: float | None,
    lower: float | None,
    upper: float | None,
) -> float | None:
    """Signed °F gap when our routine max violates the implied bound.

    Positive → we read hotter than the resolver allows (max above the
    upper bound); negative → we read colder (below the lower bound); 0.0
    when our observation is consistent with the outcome; ``None`` when we
    have no observation to compare.
    """
    if routine_max_f is None:
        return None
    if upper is not None and routine_max_f > upper:
        return round(routine_max_f - upper, 2)
    if lower is not None and routine_max_f < lower:
        return round(routine_max_f - lower, 2)
    return 0.0


async def record_market_resolution(
    session: "AsyncSession",
    market,
    *,
    yes_won: bool,
    station_icao: str | None = None,
    target_date_local: "date | None" = None,
    routine_metar_max_f: float | None = None,
) -> None:
    """Upsert one ``MarketResolution`` row (keyed on ``market_id``).

    Best-effort: never raises. Does not commit — the caller batches it.
    ``routine_metar_max_f`` may be filled later (it's known at daily
    settlement, when the station's routine daily max is computed); the
    divergence is recomputed from whatever is supplied.
    """
    try:
        rng = market_range_f(market)
        lower, upper = implied_max_bounds(market, yes_won)
        unit = "C" if market_unit(market) == "°C" else "F"
        values = dict(
            market_id=market.id,
            station_icao=station_icao,
            parsed_location=market.parsed_location,
            target_date_local=target_date_local,
            unit=unit,
            parsed_operator=market.parsed_operator,
            parsed_threshold=market.parsed_threshold,
            bucket_low_f=float(rng[0]) if rng else None,
            bucket_high_f=float(rng[1]) if rng else None,
            yes_won=yes_won,
            resolved_max_lower_f=lower,
            resolved_max_upper_f=upper,
            routine_metar_max_f=routine_metar_max_f,
            divergence_f=divergence_f(routine_metar_max_f, lower, upper),
        )
        stmt = (
            pg_insert(MarketResolution)
            .values(**values)
            .on_conflict_do_update(
                constraint="uq_market_resolution_market",
                set_={k: v for k, v in values.items() if k != "market_id"},
            )
        )
        await session.execute(stmt)
    except Exception:
        return


async def backfill_routine_max(
    session: "AsyncSession",
    *,
    station_icao: str,
    target_date_local: "date",
    routine_metar_max_f: float,
) -> int:
    """Fill ``routine_metar_max_f`` + recomputed divergence for a station-day.

    The routine daily max is computed once per station at daily settlement,
    but a station-day can carry several ``market_resolution`` rows (different
    thresholds / buckets), each with its own implied bound — so divergence is
    recomputed per row against the shared observed max. Rows whose
    ``routine_metar_max_f`` is already set are left untouched (the resolve-time
    insert seeds it ``None``; settlement fills it once).

    Returns the number of rows updated. Best-effort: never raises.
    """
    try:
        rows = (
            await session.execute(
                select(MarketResolution).where(
                    MarketResolution.station_icao == station_icao,
                    MarketResolution.target_date_local == target_date_local,
                    MarketResolution.routine_metar_max_f.is_(None),
                )
            )
        ).scalars().all()
        for row in rows:
            row.routine_metar_max_f = routine_metar_max_f
            row.divergence_f = divergence_f(
                routine_metar_max_f,
                row.resolved_max_lower_f,
                row.resolved_max_upper_f,
            )
        return len(rows)
    except Exception:
        return 0


async def stored_routine_daily_max_f(
    session: "AsyncSession",
    station_icao: str,
    target_date_local: "date",
) -> tuple[float | None, int]:
    """Max + count of routine (non-SPECI) METAR ``temp_f`` for a station-**local**
    day, read from the stored ``MetarObservation`` table.

    Unlike the live ``get_routine_daily_max`` (good only for recent days), this
    reads observations already persisted in the DB, so it stays valid for days
    the live feed no longer serves. Returns ``(None, 0)`` when no stored routine
    obs cover the day.
    """
    tz = icao_timezone(station_icao)
    local_start = datetime.combine(target_date_local, time(0, 0), tzinfo=tz)
    utc_start = local_start.astimezone(timezone.utc)
    utc_end = (local_start + timedelta(days=1)).astimezone(timezone.utc)
    temps = (
        await session.execute(
            select(MetarObservation.temp_f).where(
                MetarObservation.station_icao == station_icao,
                MetarObservation.observed_at >= utc_start,
                MetarObservation.observed_at < utc_end,
                MetarObservation.is_speci == False,  # noqa: E712
                MetarObservation.temp_f.isnot(None),
            )
        )
    ).scalars().all()
    vals = [float(t) for t in temps if t is not None]
    return (max(vals), len(vals)) if vals else (None, 0)


async def sweep_stranded_routine_max(
    session: "AsyncSession",
    *,
    lookback_days: int = 90,
    min_routines: int = 3,
) -> int:
    """Backfill ``routine_metar_max_f`` for still-null ``MarketResolution`` rows
    from **stored** METARs — the durable complement to the settlement loop's
    ~36h live-fetch window.

    A row is stranded when its station-day missed that one window (settlement
    outage, a data-starved day, or live history that has since aged out); this
    sweep recomputes the daily max from persisted observations so any straggler
    self-heals on the next run. Genuinely starved days (``count < min_routines``)
    are left NULL — an honest "unknown" beats a max read off one observation.

    Best-effort (never raises); bounded by ``lookback_days``. Returns the number
    of rows filled. Does not commit — the caller batches it.
    """
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        pairs = (
            await session.execute(
                select(
                    MarketResolution.station_icao,
                    MarketResolution.target_date_local,
                )
                .where(
                    MarketResolution.routine_metar_max_f.is_(None),
                    MarketResolution.station_icao.isnot(None),
                    MarketResolution.target_date_local.isnot(None),
                    MarketResolution.resolved_at >= cutoff,
                )
                .distinct()
            )
        ).all()

        filled = 0
        for station_icao, target_local in pairs:
            max_f, count = await stored_routine_daily_max_f(
                session, station_icao, target_local
            )
            if max_f is None or count < min_routines:
                continue
            filled += await backfill_routine_max(
                session,
                station_icao=station_icao,
                target_date_local=target_local,
                routine_metar_max_f=max_f,
            )
        return filled
    except Exception:
        return 0
