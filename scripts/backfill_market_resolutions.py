"""One-shot historical backfill for ``market_resolutions`` (M3 / Phase 3).

``resolve_trades`` only writes a ``MarketResolution`` row going forward, so
the table held just a handful of rows while ~537 markets had already settled
(469 of them °C — exactly the cohort the °C resolver-divergence audit needs).
This script reconstructs the missing rows from settled ``Trade``s:

  * ``yes_won`` is recovered per market from any settled trade —
    ``status == WON`` iff ``direction == BUY_YES`` (and the mirror), which
    is just the inverse of ``_apply_settlement``'s ``trade_won`` rule.
  * the routine-METAR daily max is read from the **stored**
    ``MetarObservation`` table (NOT ``get_routine_daily_max``, which fetches
    live history only good for recent days), bucketed by the station-local
    target day — the same ground-truth query ``cli._daily_max_by_station_day``
    uses.
  * the implied-bound + divergence math is the shared pure M3 logic via
    ``record_market_resolution``.

Idempotent: ``record_market_resolution`` upserts on ``market_id``, so
re-running refreshes rather than duplicates. Read-mostly + best-effort per
market; one ``commit`` at the end.

Run from the repo root:  ``python -m scripts.backfill_market_resolutions``
Add ``--commit`` to persist (default is a dry-run that only prints counts).
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, time, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import joinedload

from src.db.engine import async_session
from src.db.models import (
    MetarObservation,
    Trade,
    TradeDirection,
    TradeStatus,
)
from src.signals.mapper import icao_for_location, icao_timezone, resolve_target_local_day
from src.signals.market_resolution import record_market_resolution


def _yes_won(trade: Trade) -> bool:
    """Invert ``_apply_settlement``: recover the market's YES outcome.

    ``trade_won`` = (BUY_YES and yes_won) or (BUY_NO and not yes_won).
    A WON BUY_YES ⇒ yes_won; a WON BUY_NO ⇒ NOT yes_won; symmetric for LOST.
    """
    won = trade.status == TradeStatus.WON
    is_yes = trade.direction == TradeDirection.BUY_YES
    # yes_won is True when (won and is_yes) or (lost and not is_yes).
    return (won and is_yes) or ((not won) and (not is_yes))


async def _stored_daily_max_f(session, icao, utc_start, utc_end) -> float | None:
    """Max routine-METAR temp_f in [utc_start, utc_end) from stored obs."""
    rows = (
        await session.execute(
            select(MetarObservation.temp_f).where(
                MetarObservation.station_icao == icao,
                MetarObservation.observed_at >= utc_start,
                MetarObservation.observed_at < utc_end,
                MetarObservation.is_speci == False,  # noqa: E712
                MetarObservation.temp_f.isnot(None),
            )
        )
    ).scalars().all()
    temps = [float(t) for t in rows if t is not None]
    return max(temps) if temps else None


async def run(commit: bool) -> None:
    written = 0
    no_max = 0
    no_station = 0
    seen_markets: set[str] = set()

    async with async_session() as session:
        # All settled trades, eager-load the market (needs parsed_* + question).
        trades = (
            await session.execute(
                select(Trade)
                .options(joinedload(Trade.market))
                .where(Trade.status.in_([TradeStatus.WON, TradeStatus.LOST]))
            )
        ).unique().scalars().all()

        # One row per market — a market can have several settled trades, but
        # they agree on yes_won, so take the first.
        by_market: dict[str, Trade] = {}
        for t in trades:
            if t.market is not None and t.market_id not in by_market:
                by_market[t.market_id] = t

        for market_id, trade in by_market.items():
            market = trade.market
            seen_markets.add(market_id)
            icao = (
                icao_for_location(market.parsed_location)
                if market.parsed_location else None
            )
            if not icao or not market.end_date:
                no_station += 1
                continue

            tz = icao_timezone(icao)
            target = resolve_target_local_day(market.end_date, tz)
            routine_max_f = None
            if target is not None:
                local_start = datetime.combine(target, time(0, 0), tzinfo=tz)
                utc_start = local_start.astimezone(timezone.utc)
                utc_end = (local_start + timedelta(days=1)).astimezone(timezone.utc)
                routine_max_f = await _stored_daily_max_f(
                    session, icao, utc_start, utc_end
                )
            if routine_max_f is None:
                no_max += 1

            await record_market_resolution(
                session, market,
                yes_won=_yes_won(trade),
                station_icao=icao,
                target_date_local=target,
                routine_metar_max_f=routine_max_f,
            )
            written += 1

        if commit:
            await session.commit()

    mode = "COMMITTED" if commit else "DRY-RUN (no commit)"
    print(f"[{mode}] settled markets seen: {len(seen_markets)}")
    print(f"  resolution rows written:   {written}")
    print(f"  skipped (no station/end):  {no_station}")
    print(f"  written w/o routine max:   {no_max} (divergence NULL — no stored METARs for that day)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--commit", action="store_true",
        help="Persist the rows (default: dry-run, prints counts only).",
    )
    args = ap.parse_args()
    asyncio.run(run(args.commit))


if __name__ == "__main__":
    main()
