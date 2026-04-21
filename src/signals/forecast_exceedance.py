"""Telegram alert when a routine METAR beats the same-hour Open-Meteo forecast.

Diagnostic signal — runs alongside the probability/edge pipeline. Fires whenever
the latest routine observation for a station exceeds the Open-Meteo forecast for
the matching UTC hour by more than EXCEEDANCE_THRESHOLD_F. Dedup is enforced at
the DB level via a unique index on (station_icao, observed_at).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from src.db.engine import async_session
from src.db.models import ForecastExceedanceAlert
from src.execution.alerter import get_alerter
from src.ingestion.openmeteo import OpenMeteoForecast

logger = logging.getLogger(__name__)

EXCEEDANCE_THRESHOLD_F = 0.5


def _c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def _pick_latest_routine(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    routine = [
        m for m in history
        if not m.get("is_speci")
        and m.get("temp_f") is not None
        and isinstance(m.get("observed_at"), datetime)
    ]
    if not routine:
        return None
    return max(routine, key=lambda m: m["observed_at"])


def _closest_hour_index(observed_at: datetime, n_hours: int) -> int:
    # Round to nearest hour, clamp to forecast array bounds.
    idx = observed_at.hour + (1 if observed_at.minute >= 30 else 0)
    return max(0, min(idx, n_hours - 1))


async def check_and_alert_exceedance(
    icao: str,
    history: list[dict[str, Any]],
    forecast: OpenMeteoForecast | None,
) -> None:
    """Fire a Telegram alert if the newest routine METAR exceeds the same-hour forecast.

    No-op when forecast is None, no routine METAR is available, the delta is at
    or below the threshold, or an alert for this (icao, observed_at) already
    exists in the DB. Opens its own session to stay independent of the pipeline's
    shared session (which is accessed concurrently in Phase 1).
    """
    if forecast is None or not forecast.hourly_temps_c:
        return

    latest = _pick_latest_routine(history)
    if latest is None:
        return

    observed_at: datetime = latest["observed_at"]
    obs_temp_f: float = float(latest["temp_f"])

    hour_idx = _closest_hour_index(observed_at, len(forecast.hourly_temps_c))
    forecast_temp_f = _c_to_f(forecast.hourly_temps_c[hour_idx])
    delta_f = obs_temp_f - forecast_temp_f

    if delta_f <= EXCEEDANCE_THRESHOLD_F:
        return

    async with async_session() as session:
        existing = await session.execute(
            select(ForecastExceedanceAlert.id).where(
                ForecastExceedanceAlert.station_icao == icao,
                ForecastExceedanceAlert.observed_at == observed_at,
            )
        )
        if existing.scalar() is not None:
            return

        session.add(ForecastExceedanceAlert(
            station_icao=icao,
            observed_at=observed_at,
            observed_temp_f=obs_temp_f,
            forecast_temp_f=forecast_temp_f,
            delta_f=delta_f,
            forecast_hour_utc=hour_idx,
        ))
        try:
            await session.commit()
        except IntegrityError:
            # Raced with another tick for the same (icao, observed_at).
            await session.rollback()
            return

    text = (
        f"\U0001f321 *Obs exceeds forecast* [{icao}]\n"
        f"Obs {observed_at.strftime('%H:%MZ')}: {obs_temp_f:.1f}°F (routine)\n"
        f"Forecast @{hour_idx:02d}Z: {forecast_temp_f:.1f}°F\n"
        f"Delta: +{delta_f:.1f}°F"
    )
    try:
        await get_alerter()._enqueue(text)
    except Exception:
        logger.warning("exceedance alert enqueue failed for %s", icao, exc_info=True)

    logger.info(
        "[%s] exceedance alert: obs=%.1f°F @ %s vs forecast=%.1f°F @ %02dZ (delta=+%.1f°F)",
        icao, obs_temp_f, observed_at.isoformat(), forecast_temp_f, hour_idx, delta_f,
    )
