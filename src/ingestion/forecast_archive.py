"""Persistence layer for the per-tick Open-Meteo forecast snapshots.

Each successful ``aggregate_state`` invocation writes one row capturing
the blended forecast it just used. Replays in
``simulate_distribution_pipeline`` then read these back instead of
synthesising a flat ``forecast_peak_f = mid_max + 2.0`` placeholder.

Cheap and best-effort: the unified pipeline runs every 5 min and the
table is partitioned by (station, target-local-day, fetched_at), so even
60 active stations only generate ~17k rows/day — well within the budget
of an unindexed JSONB-heavy table for the kind of horizons we replay
over (≤90 days). Retention can be added later if rows accumulate
faster than backtests consume them.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import ForecastArchive
from src.signals.mapper import icao_timezone, today_local

logger = logging.getLogger(__name__)


async def archive_forecast_snapshot(
    session: AsyncSession,
    icao: str,
    forecast,
) -> None:
    """Persist one ``OpenMeteoForecast`` snapshot for later replay.

    ``forecast`` is duck-typed against ``src.ingestion.openmeteo.OpenMeteoForecast``
    so this module doesn't take a hard dependency on the ingestion type.
    Caller is responsible for catching/logging exceptions; this function
    will let SQLAlchemy errors propagate (the caller in ``aggregate_state``
    wraps the call in a ``try`` so archive failures never abort a tick).
    """
    if forecast is None:
        return

    target_date_local = today_local(icao_timezone(icao))

    row = ForecastArchive(
        station_icao=icao,
        target_date_local=target_date_local,
        fetched_at=datetime.now(timezone.utc),
        peak_temp_c=float(forecast.peak_temp_c),
        peak_hour_utc=int(forecast.peak_hour_utc),
        peak_temp_std_c=float(getattr(forecast, "peak_temp_std_c", 0.0) or 0.0),
        model_count=int(getattr(forecast, "model_count", 1) or 1),
        hourly_temps_c=list(forecast.hourly_temps_c or []),
        hourly_cloud_cover=list(forecast.hourly_cloud_cover or []),
        hourly_solar_radiation=list(forecast.hourly_solar_radiation or []),
        hourly_dewpoint_c=list(forecast.hourly_dewpoint_c or []),
        hourly_wind_speed=list(forecast.hourly_wind_speed or []),
        hourly_temps_std_c=(
            list(forecast.hourly_temps_std_c)
            if getattr(forecast, "hourly_temps_std_c", None)
            else None
        ),
    )
    session.add(row)
    # No flush/commit — let the caller's outer transaction batch this with
    # the rest of the tick's writes.
