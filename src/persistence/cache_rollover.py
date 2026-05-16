"""Per-station local-day cache rollover for in-process dedup state.

All trading-path dedup state lives in-process (lost on restart) but must
be cleared at each station's *local* midnight — not at a global UTC
instant — because lock-rule / METAR / cluster-cap semantics are anchored
to the station's local day (see ``signals.lock_rules._market_daily_max``).

State dictionaries:

* ``locked_markets_fired_today: set[str]`` — market_ids that already
  fired a lock-path order this local day. Same-tick speed-up; durable
  dedup lives in ``persistence.dedup.has_active_trade``.
* ``unified_fired_today: set[tuple[str, str, int]]`` — dry-run-only
  sibling for the probability path keyed by ``(market_id, direction,
  bucket)``. In live mode the OPEN-trade exposure is source of truth.
* ``last_routine_seen: dict[str, datetime]`` — fast-poll cursor for new
  routine METARs (per station).
* ``market_to_icao: dict[str, str]`` — reverse map populated at
  lock-fire time so rollover can find which dedup entries belong to
  which station.
* ``local_day_seen: dict[str, date]`` — last-observed local date per
  station; the rollover detection cookie.

The public surface is two callables:

* ``maybe_clear_per_station_caches()`` — called every unified-pipeline
  tick; cheap (most calls are no-ops since nothing rolled over in 5 min).
* ``record_lock_fire(market_id, icao)`` — called by the lock-rule
  executor at fire-or-reject time so the rollover knows about this
  market on the next local-day boundary.
"""

from __future__ import annotations

from datetime import date, datetime


# ---------------------------------------------------------------------------
# In-process state — module-level so importers all read the same dicts.
# ---------------------------------------------------------------------------

locked_markets_fired_today: set[str] = set()
unified_fired_today: set[tuple[str, str, int]] = set()
last_routine_seen: dict[str, datetime] = {}
market_to_icao: dict[str, str] = {}
local_day_seen: dict[str, date] = {}


def record_lock_fire(market_id: str, icao: str) -> None:
    """Register that ``market_id`` had a lock-path attempt (fire OR
    reject) for ``icao`` so the per-station rollover can find it later.

    The fast-poll lock-rule path adds market ids on filter rejects too —
    that's intentional aggressive dedup so a rejected lock doesn't get
    re-evaluated 60× per 30 minutes. Cleared on the station's local-day
    rollover.
    """
    locked_markets_fired_today.add(market_id)
    market_to_icao[market_id] = icao


def maybe_clear_per_station_caches() -> None:
    """Drop per-station dedup / cache entries when a station's local day
    has rolled over since we last looked.

    Replaces the legacy global wipe at 22:00 UTC: each station now resets
    on its own local midnight, which is the actual data-day boundary the
    routine-METAR / lock-rule logic uses (see
    ``lock_rules._market_daily_max``).

    Cheap to call every unified-pipeline tick — most calls are no-ops
    because nothing rolled over in the last 5 minutes.
    """
    from src.signals.mapper import ICAO_TIMEZONE, icao_timezone, today_local
    from src.signals.state_aggregator import clear_state_cache_for_icao

    for icao in ICAO_TIMEZONE.keys():
        try:
            tz = icao_timezone(icao)
            today = today_local(tz)
        except Exception:
            continue
        prev = local_day_seen.get(icao)
        if prev is None:
            # First time we're seeing this station — record today's
            # local date but don't wipe state. Avoids clobbering dedup
            # entries written by fast-poll before the first unified tick.
            local_day_seen[icao] = today
            continue
        if prev == today:
            continue
        # Local day rolled over — drop dedup market_ids that belong to
        # this station, the per-station routine cursor, and the cached
        # forecast/bias inputs so the next tick re-fetches.
        mids_for_icao = [m for m, ic in market_to_icao.items() if ic == icao]
        for mid in mids_for_icao:
            locked_markets_fired_today.discard(mid)
            market_to_icao.pop(mid, None)
        if mids_for_icao:
            mids_set = set(mids_for_icao)
            unified_fired_today.difference_update(
                {k for k in unified_fired_today if k[0] in mids_set}
            )
        last_routine_seen.pop(icao, None)
        clear_state_cache_for_icao(icao)
        local_day_seen[icao] = today
