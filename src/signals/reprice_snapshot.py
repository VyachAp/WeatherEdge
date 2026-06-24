"""Event-driven market-reprice snapshot telemetry (Phase 2, 2026-06-24).

The information-latency thesis — we hold the resolution-source observation before
the market reprices — was unfalsifiable because ``market_snapshots`` samples on a
fixed 15-min grid, decoupled from METAR arrival. This module captures, keyed to
the triggering METAR's ``observed_at``, the YES quote + depth for a market: once
the instant the new routine is detected (fast-poll T0) and again on the following
unified ticks. Diffing ``yes_mid`` across the rows of one
``(market_id, metar_observed_at)`` group measures how fast/far the market chases
the information.

Pure pass-through writer (mirrors ``exposure_snapshot.record_exposure_snapshot``)
plus a best-effort retention sweep. No external API calls — the caller already
holds the quote. Entirely behind ``settings.REPRICE_SNAPSHOT_ENABLED``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from sqlalchemy import delete

from src.db.models import MetarRepriceSnapshot

if TYPE_CHECKING:
    from datetime import datetime as _dt

    from sqlalchemy.ext.asyncio import AsyncSession


async def record_reprice_snapshot(
    session: "AsyncSession",
    *,
    market_id: str,
    station_icao: str,
    metar_observed_at: "_dt",
    new_obs_temp_f: float | None = None,
    new_observed_max_f: float | None = None,
    obs_fraction: float | None = None,
    yes_bid: float | None = None,
    yes_ask: float | None = None,
    yes_mid: float | None = None,
    depth_yes_usd: float | None = None,
    depth_no_usd: float | None = None,
    minutes_to_close: float | None = None,
    seconds_since_obs: float | None = None,
) -> None:
    """Append one ``MetarRepriceSnapshot`` row. Best-effort: never raises.

    Does not commit — the caller batches it. No-op unless the caller has already
    checked ``settings.REPRICE_SNAPSHOT_ENABLED`` (kept out of here so the writer
    stays a pure pass-through and tests don't need to toggle settings).
    """
    try:
        session.add(
            MetarRepriceSnapshot(
                market_id=market_id,
                station_icao=station_icao,
                metar_observed_at=metar_observed_at,
                new_obs_temp_f=new_obs_temp_f,
                new_observed_max_f=new_observed_max_f,
                obs_fraction=obs_fraction,
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                yes_mid=yes_mid,
                depth_yes_usd=depth_yes_usd,
                depth_no_usd=depth_no_usd,
                minutes_to_close=minutes_to_close,
                seconds_since_obs=seconds_since_obs,
            )
        )
    except Exception:
        return


async def sweep_reprice_retention(
    session: "AsyncSession", *, retention_days: int
) -> int:
    """Delete reprice snapshots older than ``retention_days``.

    The fast-poll + unified hooks can write hundreds–thousands of rows/day, so the
    table is capped at daily settlement (mirrors the WX-retention cleanup). Returns
    the number of rows deleted. Best-effort: never raises. Does not commit.
    """
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
        result = await session.execute(
            delete(MetarRepriceSnapshot).where(
                MetarRepriceSnapshot.created_at < cutoff
            )
        )
        return int(getattr(result, "rowcount", 0) or 0)
    except Exception:
        return 0
