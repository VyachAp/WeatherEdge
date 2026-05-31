"""Replay archived Open-Meteo forecasts into the backtest pipeline (Phase 5).

``backtest-v2`` previously built a placeholder ``WeatherState``
(``forecast_peak_f = mid_max + 2``, ``forecast_sigma_f = None``,
``hours_until_peak = 3.0``), so ``compute_distribution`` always took the
hours-based σ fallback and **never** exercised the ensemble-σ branch that
~62% of live evaluations use. That made the lead-time σ floor — and any
future distribution change touching the ensemble branch — un-validatable
by the backtest.

This module replays the real ``ForecastArchive`` corpus (≈150k rows,
≈149k carrying an ensemble spread) so the backtest runs through the actual
ensemble branch and can be scored against realized METAR ground truth.

The pure transform (:func:`archive_to_forecast_fields`) is separated from
the DB query (:func:`latest_archive_as_of`) so the unit conversions are
tested without a database. Point-in-time honest: the query only returns
snapshots fetched at or before the as-of instant (no lookahead).
"""

from __future__ import annotations

from datetime import datetime, time, timezone
from typing import TYPE_CHECKING

from sqlalchemy import select

from src.db.models import ForecastArchive

if TYPE_CHECKING:
    from datetime import date as _date

    from sqlalchemy.ext.asyncio import AsyncSession


def _c_to_f_abs(c: float) -> float:
    """Absolute °C → °F (temperature point)."""
    return c * 9.0 / 5.0 + 32.0


def _c_to_f_delta(c: float) -> float:
    """°C *delta / std* → °F — scale only, NO +32 offset.

    A standard deviation is an interval, not a point, so it converts with
    the 9/5 ratio alone. Getting this wrong would inflate every σ by 32°F.
    """
    return c * 9.0 / 5.0


def _aware(dt: datetime) -> datetime:
    """Normalize to aware UTC (test mocks often pass naive datetimes)."""
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def archive_to_forecast_fields(row: ForecastArchive, as_of_utc: datetime) -> dict:
    """Forecast ``WeatherState`` kwargs from a ``ForecastArchive`` row, as of T.

    Returns ``forecast_peak_f`` / ``forecast_sigma_f`` /
    ``hours_until_peak`` / ``ensemble_model_count`` / ``has_forecast`` —
    exactly the fields that drive ``_compute_sigma``'s ensemble branch.

    ``forecast_sigma_f`` is ``None`` when the archived ensemble spread is 0
    (single-model snapshot), so the replay mirrors the live
    ``forecast_sigma_f is None`` fallback rather than fabricating a 0°F σ.
    ``hours_until_peak`` is ``peak_instant − as_of`` where ``peak_instant``
    is ``target_date_local`` at ``peak_hour_utc`` (UTC); it can be negative
    (past peak), matching the live signal so the lead-time arm replays
    faithfully.
    """
    peak_f = _c_to_f_abs(float(row.peak_temp_c))
    std_c = float(row.peak_temp_std_c or 0.0)
    sigma_f = _c_to_f_delta(std_c) if std_c > 0 else None

    ph = int(row.peak_hour_utc) % 24
    peak_dt = datetime.combine(
        row.target_date_local, time(hour=ph), tzinfo=timezone.utc
    )
    hours_until_peak = (peak_dt - _aware(as_of_utc)).total_seconds() / 3600.0

    return {
        "forecast_peak_f": peak_f,
        "forecast_sigma_f": sigma_f,
        "hours_until_peak": hours_until_peak,
        "ensemble_model_count": int(row.model_count or 0),
        "has_forecast": True,
    }


async def latest_archive_as_of(
    session: "AsyncSession",
    icao: str,
    target_date_local: "_date",
    as_of_utc: datetime,
) -> ForecastArchive | None:
    """Most recent ``ForecastArchive`` for (icao, day) fetched at/before T.

    Point-in-time honest — never returns a snapshot fetched after the
    as-of instant. ``None`` when the corpus has no snapshot for that
    station-day yet (history before the writer existed), so the caller
    falls back to its placeholder.
    """
    stmt = (
        select(ForecastArchive)
        .where(
            ForecastArchive.station_icao == icao,
            ForecastArchive.target_date_local == target_date_local,
            ForecastArchive.fetched_at <= _aware(as_of_utc),
        )
        .order_by(ForecastArchive.fetched_at.desc())
        .limit(1)
    )
    return (await session.execute(stmt)).scalars().first()
