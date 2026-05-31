"""Market-resolution ground-truth telemetry (M3) — Phase 0.

De-circularises filter tuning. ``evals-report`` today scores candidates
against our own routine-METAR daily max — the same source that feeds our
conviction, which is circular for °C cities (we read systematically
hotter than Polymarket's resolver; that's what disabled
``RANGE_OVERSHOOT_LOCK_ENABLED``). This module persists, per settled
market, the resolved YES/NO outcome and the daily-max **bound** it
implies, so Phase 3 can measure the signed per-station divergence between
our observation and the actual resolver.

Pure logic (``implied_max_bounds`` / ``divergence_f``) is separated from
the best-effort DB upsert so it can be unit-tested without a database.
The settlement-path wiring is the caller's responsibility.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db.models import MarketResolution
from src.execution.binary_market import market_range_f, market_unit

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
