"""One-shot historical backfill for ``station_day_resolutions`` (Phase 1).

The daily-settlement hook (``resolve_station_days``) only back-solves station-days
going forward. This drains history: it reuses the exact same pure logic over the
already-labeled ``market_resolutions`` corpus, so every laddered station-day in
the look-back gets its intersected resolved-max window + continuous divergence.

Depends on ``market_resolutions`` already being populated — run
``scripts.backfill_market_resolutions --commit`` first if needed.

Idempotent: ``resolve_station_days`` upserts on ``(station_icao,
target_date_local)``, so re-running refreshes rather than duplicates.

Run from the repo root:  ``python -m scripts.backfill_station_day_resolutions``
Add ``--commit`` to persist (default is a dry-run that only prints the count).
"""

from __future__ import annotations

import argparse
import asyncio

from src.db.engine import async_session
from src.signals.station_day_resolution import resolve_station_days


async def run(commit: bool, lookback_days: int) -> None:
    async with async_session() as session:
        written = await resolve_station_days(session, lookback_days=lookback_days)
        if commit:
            await session.commit()

    mode = "COMMITTED" if commit else "DRY-RUN (no commit)"
    print(f"[{mode}] station-days resolved (last {lookback_days}d): {written}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--commit", action="store_true",
        help="Persist the rows (default: dry-run, prints count only).",
    )
    ap.add_argument(
        "--days", type=int, default=120,
        help="Look-back window in days (default: 120).",
    )
    args = ap.parse_args()
    asyncio.run(run(args.commit, args.days))


if __name__ == "__main__":
    main()
