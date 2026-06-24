"""Forecast-error evolution from the archive corpus (Phase 3, 2026-06-24).

The ``forecast_archive`` corpus accumulates ~17k rows/day with no live reader.
This module turns it into the per-station, per-lead-time forecast-error dataset
the deferred lead-time σ floor (``SIGMA_LEAD_TIME_SLOPE_F_PER_HR``) and the
climate prior need: for each settled station-day, the forecast peak at a series
of lead buckets before peak, joined to the realized routine-METAR max and the
Phase-1 resolved max.

Builds on the existing replay reader (``forecast_archive_replay``) for the
correct °F / σ-delta conversions and point-in-time honesty. The pure
``compute_forecast_errors`` is separated from the DB upsert so the lead-bucketing
math is unit-tested without a database.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db.models import (
    ForecastArchive,
    ForecastErrorDaily,
    StationDayResolution,
)
from src.signals.forecast_archive_replay import (
    _aware,
    _c_to_f_abs,
    _c_to_f_delta,
)

if TYPE_CHECKING:
    from datetime import date as _date

    from sqlalchemy.ext.asyncio import AsyncSession

# Lead-to-peak buckets (hours before peak). 0 = the final forecast at peak;
# larger = earlier, less-informed forecasts. The σ-vs-lead curve is read off the
# RMSE of error_vs_* across these.
DEFAULT_LEAD_BUCKETS: tuple[int, ...] = (0, 6, 12, 18, 24, 36, 48)


def compute_forecast_errors(
    archive_rows: list,
    *,
    realized_max_f: float | None,
    resolved_max_f: float | None = None,
    lead_buckets: tuple[int, ...] = DEFAULT_LEAD_BUCKETS,
) -> list[dict]:
    """Forecast error at each lead bucket for one station-day (pure).

    ``archive_rows`` are that station-day's ``ForecastArchive`` snapshots (any
    order; each needs ``peak_temp_c``, ``peak_temp_std_c``, ``peak_hour_utc``,
    ``target_date_local``, ``fetched_at``). The canonical peak instant is taken
    from the final (latest-fetched) snapshot. For each lead bucket ``L`` we pick
    the most recent snapshot fetched at or before ``peak − L hours`` (point-in-
    time honest — the forecast we'd actually have held ``L`` hours out) and emit
    its peak vs the realized + resolved truth. Buckets with no snapshot yet
    available are skipped. Returns ``[]`` when there are no rows or no realized
    truth to score against.
    """
    if not archive_rows or realized_max_f is None:
        return []

    rows = sorted(archive_rows, key=lambda r: _aware(r.fetched_at))
    final = rows[-1]
    ph = int(final.peak_hour_utc) % 24
    peak_instant = datetime.combine(
        final.target_date_local, time(hour=ph), tzinfo=timezone.utc
    )

    out: list[dict] = []
    for lead in lead_buckets:
        as_of = peak_instant - timedelta(hours=lead)
        candidates = [r for r in rows if _aware(r.fetched_at) <= as_of]
        if not candidates:
            continue
        snap = candidates[-1]  # rows sorted asc → last is the most recent ≤ as_of
        peak_f = _c_to_f_abs(float(snap.peak_temp_c))
        std_c = float(snap.peak_temp_std_c or 0.0)
        sigma_f = _c_to_f_delta(std_c) if std_c > 0 else None
        err_metar = round(peak_f - realized_max_f, 2)
        err_resolved = (
            round(peak_f - resolved_max_f, 2)
            if resolved_max_f is not None else None
        )
        out.append({
            "lead_bucket_h": lead,
            "forecast_peak_f": round(peak_f, 2),
            "forecast_sigma_f": round(sigma_f, 2) if sigma_f is not None else None,
            "realized_max_f": realized_max_f,
            "resolved_max_f": resolved_max_f,
            "error_vs_metar_f": err_metar,
            "error_vs_resolved_f": err_resolved,
        })
    return out


async def record_forecast_error_daily(
    session: "AsyncSession",
    *,
    station_icao: str,
    target_date_local: "_date",
    realized_max_f: float | None,
    resolved_max_f: float | None = None,
    lead_buckets: tuple[int, ...] = DEFAULT_LEAD_BUCKETS,
) -> int:
    """Upsert one ``ForecastErrorDaily`` row per lead bucket for a station-day.

    Loads the station-day's ``ForecastArchive`` rows, runs the pure
    ``compute_forecast_errors``, and upserts on
    ``(station_icao, target_date_local, lead_bucket_h)``. Returns the number of
    rows written. Best-effort: never raises. Does not commit — the caller batches.
    """
    try:
        archive_rows = (
            await session.execute(
                select(ForecastArchive).where(
                    ForecastArchive.station_icao == station_icao,
                    ForecastArchive.target_date_local == target_date_local,
                )
            )
        ).scalars().all()
        if not archive_rows:
            return 0

        errors = compute_forecast_errors(
            list(archive_rows),
            realized_max_f=realized_max_f,
            resolved_max_f=resolved_max_f,
            lead_buckets=lead_buckets,
        )
        now = datetime.now(timezone.utc)
        written = 0
        for e in errors:
            values = dict(
                station_icao=station_icao,
                target_date_local=target_date_local,
                computed_at=now,
                **e,
            )
            stmt = (
                pg_insert(ForecastErrorDaily)
                .values(**values)
                .on_conflict_do_update(
                    constraint="uq_fc_error_station_day_lead",
                    set_={
                        k: v for k, v in values.items()
                        if k not in (
                            "station_icao", "target_date_local", "lead_bucket_h"
                        )
                    },
                )
            )
            await session.execute(stmt)
            written += 1
        return written
    except Exception:
        return 0


async def compute_recent_forecast_errors(
    session: "AsyncSession",
    *,
    lookback_days: int = 45,
) -> int:
    """Build ``forecast_error_daily`` rows for every recently-resolved station-day.

    Reads ``station_day_resolutions`` (Phase 1) within ``lookback_days`` — which
    carries both our realized routine max and the resolved point — and computes
    the forecast-error-by-lead dataset for each. Runs at daily settlement AFTER
    the Phase-1 station-day resolution so both truth columns are available.
    Idempotent (per-bucket upsert). Best-effort: never raises, never blocks
    settlement. Returns the number of station-days processed. Does not commit.
    """
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        sdr_rows = (
            await session.execute(
                select(
                    StationDayResolution.station_icao,
                    StationDayResolution.target_date_local,
                    StationDayResolution.routine_metar_max_f,
                    StationDayResolution.resolved_max_point_f,
                ).where(
                    StationDayResolution.resolved_at >= cutoff,
                    StationDayResolution.routine_metar_max_f.isnot(None),
                )
            )
        ).all()

        processed = 0
        for icao, target, realized, resolved in sdr_rows:
            n = await record_forecast_error_daily(
                session,
                station_icao=icao,
                target_date_local=target,
                realized_max_f=realized,
                resolved_max_f=resolved,
            )
            if n:
                processed += 1
        return processed
    except Exception:
        return 0
