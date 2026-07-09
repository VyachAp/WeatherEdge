"""Counterfactual knowledge base (Phase 3 b — self-improvement persistence).

The DB-touching glue around the pure ``analysis.counterfactual`` miner:

* ``fetch_counterfactual_rows`` — the SQL join that both the
  ``counterfactual-report`` CLI command and the nightly snapshot share: the
  latest filter-REJECTED eval and post-filter-THROTTLED decision per
  (market, side), enriched with the market operator + the de-circularised
  ``market_resolutions`` outcome/station.
* ``record_knowledge_snapshot`` — runs the miner over the rolling window and
  persists the result (a ``bot_state`` row + a gitignored runtime artifact),
  so the Layer-2 analyst reads an accumulating, refining view of "which filters
  / stations / price bands are leaving money" without re-querying. Best-effort
  — mirrors ``signals.market_resolution`` / the perf-review artifact discipline
  and never raises into the nightly settlement caller.

Propose-only: this records knowledge for a human/analyst to act on; nothing here
mutates a filter or a config.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

BOT_STATE_KEY = "knowledge.counterfactual.latest"


async def fetch_counterfactual_rows(session, cutoff: datetime) -> tuple[list[dict], list[dict]]:
    """Return ``(rejected_rows, throttled_rows)`` for ``mine_counterfactuals``.

    ``rejected_rows``: the latest ``passes=False`` eval per (market, side) since
    ``cutoff`` — direction, reject-reason prefix, side price, hours-to-peak.
    ``throttled_rows``: the latest throttle-outcome decision per (market, side).
    Both enriched with the market operator (``op``) and the resolved outcome
    (``yes_won``) + ``station_icao`` from ``market_resolutions`` (None when the
    market hasn't resolved / has no M3 row).
    """
    from sqlalchemy import select

    from src.analysis.counterfactual import THROTTLE_OUTCOMES
    from src.db.models import DecisionLog, EvaluationLog, Market, MarketResolution

    # DISTINCT ON (market_id, direction) keeping the latest row per side does the
    # dedup in Postgres, so we transfer ~one row per market-side instead of every
    # passes=False eval since cutoff (~30k/day on a ~2M-row table). The read-path
    # was the "reports hang at prod scale" bottleneck — it was O(raw rows) over the
    # wire, not a bad plan. ORDER BY must lead with the DISTINCT ON columns.
    rej = (await session.execute(
        select(
            EvaluationLog.market_id, EvaluationLog.direction,
            EvaluationLog.reject_reason, EvaluationLog.market_prob,
            EvaluationLog.hours_until_peak,
        ).where(
            EvaluationLog.passes.is_(False),
            EvaluationLog.created_at >= cutoff,
        ).distinct(EvaluationLog.market_id, EvaluationLog.direction)
        .order_by(
            EvaluationLog.market_id, EvaluationLog.direction,
            EvaluationLog.created_at.desc(),
        )
    )).all()
    latest_rej: dict[tuple, dict] = {}
    for mid, direction, reason, price, htp in rej:
        dirv = getattr(direction, "value", str(direction))
        latest_rej[(mid, dirv)] = {
            "market_id": mid, "direction": dirv,
            "reject_reason": ((reason or "").split() or ["?"])[0],
            "side_price": float(price) if price is not None else None,
            "hours_until_peak": float(htp) if htp is not None else None,
        }

    thr = (await session.execute(
        select(
            DecisionLog.market_id, DecisionLog.direction, DecisionLog.outcome,
        ).where(
            DecisionLog.outcome.in_(THROTTLE_OUTCOMES),
            DecisionLog.created_at >= cutoff,
        ).distinct(DecisionLog.market_id, DecisionLog.direction)
        .order_by(
            DecisionLog.market_id, DecisionLog.direction,
            DecisionLog.created_at.desc(),
        )
    )).all()
    latest_thr: dict[tuple, dict] = {}
    for mid, direction, outcome in thr:
        dirv = getattr(direction, "value", str(direction))
        latest_thr[(mid, dirv)] = {
            "market_id": mid, "direction": dirv,
            "outcome": outcome, "side_price": None,
        }

    mids = {k[0] for k in latest_rej} | {k[0] for k in latest_thr}
    ops = (
        dict((await session.execute(
            select(Market.id, Market.parsed_operator).where(Market.id.in_(mids))
        )).all()) if mids else {}
    )
    res = {
        r[0]: (r[1], r[2]) for r in (await session.execute(
            select(
                MarketResolution.market_id, MarketResolution.yes_won,
                MarketResolution.station_icao,
            ).where(
                MarketResolution.market_id.in_(mids),
                MarketResolution.yes_won.isnot(None),
            )
        )).all()
    } if mids else {}

    def _enrich(d: dict) -> dict:
        yw, icao = res.get(d["market_id"], (None, None))
        d["op"] = ops.get(d["market_id"])
        d["station_icao"] = icao
        d["yes_won"] = bool(yw) if yw is not None else None
        return d

    rejected_rows = [_enrich(d) for d in latest_rej.values()]
    throttled_rows = [_enrich(d) for d in latest_thr.values()]
    return rejected_rows, throttled_rows


async def record_knowledge_snapshot(session, *, days: int = 30) -> dict | None:
    """Mine the counterfactual knowledge over the last ``days`` and persist it.

    Writes a ``bot_state`` row (``knowledge.counterfactual.latest``, DB-readable)
    and a gitignored ``runtime/knowledge_counterfactual.json`` for the Layer-2
    analyst. Best-effort: returns the snapshot, or ``None`` on any failure
    (never raises into ``job_daily_settlement``).
    """
    import json
    from pathlib import Path

    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from sqlalchemy.exc import ProgrammingError, SQLAlchemyError

    from src.analysis.counterfactual import mine_counterfactuals
    from src.db.models import BotState

    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        rejected_rows, throttled_rows = await fetch_counterfactual_rows(session, cutoff)
        mined = mine_counterfactuals(rejected_rows, throttled_rows)
        snapshot = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "window_days": days,
            "rejected_total": mined["rejected"]["total"],
            "throttled_total": mined["throttled"]["total"],
            # Headline = the actionable "we left money" cohorts (propose-only).
            "headline": mined["headline"],
            # Full per-cohort pivots for the analyst to drill into.
            "rejected": mined["rejected"],
            "throttled": mined["throttled"],
        }
    except (SQLAlchemyError, KeyError, ValueError):
        return None

    now = datetime.now(timezone.utc)
    try:
        await session.execute(
            pg_insert(BotState)
            .values(key=BOT_STATE_KEY, value=snapshot, updated_at=now)
            .on_conflict_do_update(
                index_elements=["key"],
                set_={"value": snapshot, "updated_at": now},
            )
        )
        await session.commit()
    except ProgrammingError:
        await session.rollback()
    except SQLAlchemyError:
        await session.rollback()
    try:
        out = Path("runtime")
        out.mkdir(exist_ok=True)
        (out / "knowledge_counterfactual.json").write_text(
            json.dumps(snapshot, indent=2, default=str)
        )
    except OSError:
        pass
    return snapshot
