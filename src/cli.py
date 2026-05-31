"""CLI entry point for WeatherEdge."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone

import click


@click.group()
def main() -> None:
    """WeatherEdge — weather-driven Polymarket edge detection."""


@main.command()
def run() -> None:
    """Start the scheduler daemon."""
    from src.scheduler import run_scheduler

    asyncio.run(run_scheduler())


@main.command()
def scan() -> None:
    """One-shot: fetch markets, forecasts, and generate signals."""

    async def _scan() -> None:
        from src.ingestion.polymarket import scan_and_ingest
        from src.scheduler import configure_logging

        configure_logging()
        count = await scan_and_ingest()
        click.echo(f"Scanned {count} weather markets")

    asyncio.run(_scan())


@main.command()
@click.option("--days", default=30, show_default=True, help="Days to backfill.")
def backfill(days: int) -> None:
    """Backfill historical market snapshots."""

    async def _backfill() -> None:
        from src.scheduler import backfill_markets, configure_logging

        configure_logging()
        count = await backfill_markets(days)
        click.echo(f"Backfilled {count} markets over {days} days")

    asyncio.run(_backfill())


@main.command()
def status() -> None:
    """Show current bankroll, open positions, and recent signals."""

    async def _status() -> None:
        from sqlalchemy import select

        from src.config import settings
        from src.db.engine import async_session
        from src.db.models import (
            BankrollLog,
            Signal,
            Trade,
            TradeStatus,
        )

        async with async_session() as session:
            # Bankroll
            row = (
                await session.execute(
                    select(BankrollLog)
                    .order_by(BankrollLog.timestamp.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            bankroll = row.balance if row else settings.INITIAL_BANKROLL
            peak = row.peak if row else bankroll
            dd = row.drawdown_pct if row else 0.0

            click.echo("=== Bankroll ===")
            click.echo(f"  Balance:  ${bankroll:,.2f}")
            click.echo(f"  Peak:     ${peak:,.2f}")
            click.echo(f"  Drawdown: {dd * 100:.1f}%")
            click.echo()

            # Open positions
            open_trades = (
                await session.execute(
                    select(Trade).where(Trade.status == TradeStatus.OPEN)
                )
            ).scalars().all()
            click.echo(f"=== Open Positions ({len(open_trades)}) ===")
            total_exposure = 0.0
            for t in open_trades:
                click.echo(
                    f"  {t.market_id[:12]}…  {t.direction.value:<8} "
                    f"${t.stake_usd:>7.2f}  entry={t.entry_price:.3f}"
                )
                total_exposure += t.stake_usd
            if open_trades:
                click.echo(f"  Total exposure: ${total_exposure:,.2f}")
            click.echo()

            # Recent signals
            signals = (
                await session.execute(
                    select(Signal).order_by(Signal.created_at.desc()).limit(10)
                )
            ).scalars().all()
            click.echo(f"=== Recent Signals ({len(signals)}) ===")
            for s in signals:
                age = datetime.now(timezone.utc) - (
                    s.created_at.replace(tzinfo=timezone.utc)
                    if s.created_at.tzinfo is None
                    else s.created_at
                )
                ago = f"{age.total_seconds() / 3600:.0f}h ago"
                # Path-aware tail: prob signals show model_prob; lock
                # signals show the °F margin from threshold instead.
                if s.signal_kind == "lock" and s.lock_margin_f is not None:
                    tail = f"margin={s.lock_margin_f:.1f}°F"
                else:
                    tail = f"prob={s.model_prob:.2f}"
                click.echo(
                    f"  {s.market_id[:12]}…  edge={s.edge:+.3f}  "
                    f"{s.direction.value:<8}  {tail}  {ago}"
                )

    asyncio.run(_status())


@main.command("paper-trade")
@click.option("--days", default=30, show_default=True, help="Simulation period in days.")
def paper_trade(days: int) -> None:
    """Simulate trading over historical signals."""

    async def _paper_trade() -> None:
        from sqlalchemy import select

        from src.db.engine import async_session
        from src.db.models import Signal, Trade, TradeStatus
        from src.risk.simulate import SimSignal, simulate_bankroll

        async with async_session() as session:
            cutoff = datetime.utcnow() - timedelta(days=days)
            signals = (
                await session.execute(
                    select(Signal)
                    .where(Signal.created_at >= cutoff)
                    .order_by(Signal.created_at)
                )
            ).scalars().all()

            if not signals:
                click.echo("No signals found in the given period.")
                return

            sim_signals: list[SimSignal] = []
            for sig in signals:
                trades = (
                    await session.execute(
                        select(Trade).where(Trade.signal_id == sig.id)
                    )
                ).scalars().all()
                outcome = any(t.status == TradeStatus.WON for t in trades)
                sim_signals.append(
                    SimSignal(
                        model_prob=sig.model_prob,
                        market_prob=sig.market_prob,
                        outcome=outcome,
                    )
                )

        result = simulate_bankroll(sim_signals)

        click.echo("=== Paper Trade Results ===")
        click.echo(f"  Period:       {days} days")
        click.echo(f"  Signals:      {len(sim_signals)}")
        click.echo(f"  Trades taken: {result.num_trades}")
        click.echo(f"  Skipped:      {result.num_skipped}")
        click.echo(f"  Final bankroll: ${result.final_bankroll:,.2f}")
        click.echo(f"  Max drawdown:   {result.max_drawdown:.1%}")
        click.echo(f"  Sharpe ratio:   {result.sharpe_ratio:.2f}")
        click.echo(f"  Win rate:       {result.win_rate:.1%}")

    asyncio.run(_paper_trade())


# --- Telemetry-report shared helpers -------------------------------------
# Operator classification splits the two telemetry tables the same way the
# trading paths treat them: the 2026-05-22/23 quality cuts (lead gate,
# landing-band, NO-prob cap, range_overshoot) all target *bracket-like* ops,
# while *threshold* ops are the clean class. The reports pivot on this split
# so "expected bracket-guard cuts" can be told apart from "threshold volume
# suppressed by the bracket-era global MIN_PROBABILITY / MIN_EDGE".
_OP_THRESHOLD = ("above", "at_least", "below", "at_most")
_OP_BRACKET = ("exactly", "range", "bracket")


def _op_class(op: str | None) -> str:
    if op in _OP_THRESHOLD:
        return "threshold"
    if op in _OP_BRACKET:
        return "bracket-like"
    return "other"


async def _load_market_map(session, market_ids):
    """Batch-load ``market_id -> Market`` for a set of telemetry rows.

    ``evaluation_logs`` / ``decision_logs`` carry only ``market_id``; the
    operator lives on ``markets.parsed_operator``. One round-trip keeps the
    remote DigitalOcean DB happy (mirrors how ``evals_report`` already
    batches ``Trade`` rows).
    """
    from sqlalchemy import select
    from src.db.models import Market

    ids = list({mid for mid in market_ids if mid})
    if not ids:
        return {}
    rows = (
        await session.execute(select(Market).where(Market.id.in_(ids)))
    ).scalars().all()
    return {m.id: m for m in rows}


async def _daily_max_by_station_day(session, icaos, utc_start, utc_end):
    """Build ``{(icao, local_date): max_temp_f}`` from routine METARs.

    Single query over the whole window for all stations; bucketed in Python
    by each observation's *station-local* date (the day Polymarket's
    "highest temperature" question is anchored to). Mirrors the ground-truth
    query shape in ``simulate_distribution_pipeline`` (routine-only,
    non-null temp). NOTE: routine-METAR max can diverge from Polymarket's
    resolver for °C cities — see the caveat printed by the recoverable-band
    section.
    """
    from sqlalchemy import select
    from src.db.models import MetarObservation
    from src.signals.mapper import icao_timezone

    icao_list = list({i for i in icaos if i})
    if not icao_list:
        return {}
    rows = (
        await session.execute(
            select(
                MetarObservation.station_icao,
                MetarObservation.observed_at,
                MetarObservation.temp_f,
            ).where(
                MetarObservation.station_icao.in_(icao_list),
                MetarObservation.observed_at >= utc_start,
                MetarObservation.observed_at < utc_end,
                MetarObservation.is_speci == False,  # noqa: E712
                MetarObservation.temp_f.isnot(None),
            )
        )
    ).all()

    tz_cache: dict[str, object] = {}
    out: dict[tuple[str, object], float] = {}
    for icao, observed_at, temp_f in rows:
        tz = tz_cache.get(icao)
        if tz is None:
            tz = icao_timezone(icao)
            tz_cache[icao] = tz
        local_date = observed_at.astimezone(tz).date()
        key = (icao, local_date)
        if key not in out or temp_f > out[key]:
            out[key] = temp_f
    return out


def _threshold_yes_won(op: str, threshold: float, actual_max_f: float) -> bool:
    """Did the YES side win under real resolution semantics.

    Uses the true inequality the market resolves on (not the engine's integer
    bucket approximation): ``above`` is strict ``>``, ``at_least`` is ``>=``,
    ``below`` strict ``<``, ``at_most`` is ``<=``. ``actual_max_f`` is the
    routine-METAR daily max (our proxy for Polymarket's resolver max — they
    diverge for °C cities, hence the advisory caveat). Threshold is in °F to
    match ``actual_max_f`` (binary_market_edge treats parsed_threshold as °F).
    """
    if op == "above":
        return actual_max_f > threshold
    if op == "at_least":
        return actual_max_f >= threshold
    if op == "below":
        return actual_max_f < threshold
    # at_most
    return actual_max_f <= threshold


# --- Measurement-layer report helpers (Phase 0 readers) ------------------
# These are PURE functions over plain dicts (no DB, no click) so the
# number-crunching behind exposure-report / resolution-report /
# shadow-report / epochs is unit-testable without mocking the async
# session — mirroring the pure M3 logic in signals.market_resolution.


def _quantiles(values: list[float]) -> tuple[float, float, float] | None:
    """Nearest-rank p25/p50/p75 (module-level; shared by the reports).

    Nearest-rank for tiny samples — interpolation isn't worth a numpy
    dependency here. Returns None for an empty input.
    """
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    p25 = s[max(0, n // 4 - 1)]
    p50 = s[n // 2]
    p75 = s[min(n - 1, (3 * n) // 4)]
    return p25, p50, p75


def _summarize_exposure(rows: list[dict], min_stake: float = 5.0) -> dict:
    """Summarise a window of ``exposure_snapshots`` rows (as dicts).

    Reports the Phase-0 capital gate: per-tick headroom quantiles + the
    fraction of ticks where headroom fell below one minimum stake (i.e.
    a new bet could not be funded). ``by_class`` sums the per-tick
    ``n_open_by_class`` so the bracket-vs-threshold exposure split is
    visible. Empty input → ``{"count": 0}``.
    """
    if not rows:
        return {"count": 0}
    headroom = [float(r["headroom"]) for r in rows]
    by_class: dict[str, int] = {}
    for r in rows:
        for k, v in (r.get("n_open_by_class") or {}).items():
            by_class[k] = by_class.get(k, 0) + int(v)
    n_low = sum(1 for h in headroom if h < min_stake)
    n_open_q = _quantiles([float(r.get("n_open") or 0) for r in rows])
    return {
        "count": len(rows),
        "headroom": _quantiles(headroom),
        "exposure": _quantiles([float(r["exposure"]) for r in rows]),
        "equity": _quantiles([float(r["equity"]) for r in rows]),
        "effective_cap": _quantiles([float(r["effective_cap"]) for r in rows]),
        "n_open_median": n_open_q[1] if n_open_q else 0.0,
        "by_class": by_class,
        "low_headroom_frac": n_low / len(rows),
    }


def _aggregate_divergence(rows: list[dict]) -> list[dict]:
    """Per-(station, unit) resolver-divergence stats from M3 rows (dicts).

    Only rows that have both ``divergence_f`` and ``routine_metar_max_f``
    (i.e. the daily-settlement backfill has run) contribute. Returns a
    list of ``{station_icao, unit, n, mean, std, min, max}`` sorted by
    ``|mean|`` desc — the worst-diverging stations first. This is the
    Phase-3 audit surface.
    """
    from collections import defaultdict

    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("divergence_f") is None or r.get("routine_metar_max_f") is None:
            continue
        groups[(r.get("station_icao"), r.get("unit"))].append(float(r["divergence_f"]))
    out: list[dict] = []
    for (icao, unit), vals in groups.items():
        n = len(vals)
        mean = sum(vals) / n
        std = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5 if n else 0.0
        out.append({
            "station_icao": icao, "unit": unit, "n": n,
            "mean": mean, "std": std, "min": min(vals), "max": max(vals),
        })
    out.sort(key=lambda d: abs(d["mean"]), reverse=True)
    return out


def _flatten_shadow(obj, prefix: str = "") -> dict:
    """Flatten a nested ``shadow_json`` dict to dotted leaf keys."""
    out: dict = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten_shadow(v, key))
    else:
        out[prefix] = obj
    return out


def _summarize_shadow(shadow_dicts: list[dict]) -> dict:
    """Per-leaf occupancy + numeric quantiles across shadow_json blobs.

    ``{leaf_key: {"count": int, "quantiles": (p25,p50,p75)|None}}``.
    Booleans are excluded from the numeric quantiles (they're flags, not
    measurements). Empty / all-None input → ``{}``.
    """
    from collections import defaultdict

    leaves: dict[str, list] = defaultdict(list)
    for sd in shadow_dicts:
        if not sd:
            continue
        for k, v in _flatten_shadow(sd).items():
            if v is not None:
                leaves[k].append(v)
    out: dict = {}
    for k, vals in leaves.items():
        nums = [
            float(v) for v in vals
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        ]
        out[k] = {"count": len(vals), "quantiles": _quantiles(nums) if nums else None}
    return out


def _flags_diff(prev: dict | None, cur: dict | None) -> dict:
    """``{key: (old, new)}`` for keys whose value changed between epochs."""
    prev = prev or {}
    cur = cur or {}
    diff: dict = {}
    for k in set(prev) | set(cur):
        a, b = prev.get(k), cur.get(k)
        if a != b:
            diff[k] = (a, b)
    return diff


async def _epoch_start(session, epoch_id: int | None):
    """``config_epochs.started_at`` for an epoch id (one PK lookup), or None."""
    if epoch_id is None:
        return None
    from sqlalchemy import select
    from src.db.models import ConfigEpoch

    return (
        await session.execute(
            select(ConfigEpoch.started_at).where(ConfigEpoch.id == epoch_id)
        )
    ).scalar_one_or_none()


async def _recoverable_band_section(session, rows, op_of) -> None:
    """Print the recoverable-band table for THRESHOLD-op probability evals.

    For each candidate ``(min_probability, min_edge)`` pair we re-run the real
    :func:`_check_filters` against each currently-rejected threshold eval's
    recorded fields. The newly-passing set is then scored against the actual
    routine-METAR daily max for the market's target local day:

      * **won%** — would-have-won rate (YES/NO per the eval's direction).
      * **break-even** — mean side price (``market_prob``); won% must clear
        this for the band to be +EV.
      * **EV/$1** — mean realised return per $1 staked (``1/price`` on a win,
        else 0, minus 1).

    A baseline row reports the quality of the threshold evals that ALREADY
    pass, so the recovered band can be compared like-for-like. Ground truth
    is routine-METAR-derived and diverges from Polymarket's resolver for °C
    cities — see the printed caveat.
    """
    from src.config import settings
    from src.signals.edge_calculator import _check_filters
    from src.signals.mapper import (
        icao_for_location,
        icao_timezone,
        resolve_target_local_day,
    )

    click.echo("## Recoverable rejected band (threshold ops)")
    click.echo()

    # Threshold-op probability evals with the fields needed to re-run filters.
    th_rows = [
        r for r in rows
        if _op_class(op_of[r.market_id]) == "threshold"
        and r.signal_kind == "probability"
        and r.depth_usd is not None
        and r.minutes_to_close is not None
        and r.routine_count is not None
    ]
    if not th_rows:
        click.echo("_No threshold probability evals with complete filter fields._")
        click.echo()
        return

    # Resolve ground-truth daily max per (icao, target local day). One METAR
    # query for the whole window; the market map is needed for icao + end_date.
    market_map = await _load_market_map(session, (r.market_id for r in th_rows))
    created = [r.created_at for r in th_rows]
    win_start = min(created) - timedelta(days=2)
    win_end = max(created) + timedelta(days=2)
    icaos = {
        icao_for_location(m.parsed_location)
        for m in market_map.values() if m.parsed_location
    }
    daily_max = await _daily_max_by_station_day(session, icaos, win_start, win_end)

    def _ground_truth_won(row) -> bool | None:
        m = market_map.get(row.market_id)
        if m is None or m.parsed_location is None or m.parsed_threshold is None:
            return None
        if m.end_date is None or m.parsed_operator not in _OP_THRESHOLD:
            return None
        icao = icao_for_location(m.parsed_location)
        if icao is None:
            return None
        tz = icao_timezone(icao)
        target_day = resolve_target_local_day(m.end_date, tz)
        if target_day is None:
            return None
        amax = daily_max.get((icao, target_day))
        if amax is None:
            return None
        yes_won = _threshold_yes_won(m.parsed_operator, m.parsed_threshold, amax)
        return yes_won if row.direction.value == "BUY_YES" else (not yes_won)

    def _score(rows_subset) -> tuple[int, int, int, float, float]:
        """(count, n_resolved, won, mean_break_even, mean_ev_per_$1)."""
        won = resolved = 0
        prices: list[float] = []
        evs: list[float] = []
        for r in rows_subset:
            gt = _ground_truth_won(r)
            if gt is None:
                continue
            resolved += 1
            prices.append(r.market_prob)
            if gt:
                won += 1
                evs.append((1.0 / r.market_prob - 1.0) if r.market_prob > 0 else 0.0)
            else:
                evs.append(-1.0)
        be = (sum(prices) / len(prices)) if prices else 0.0
        ev = (sum(evs) / len(evs)) if evs else 0.0
        return len(rows_subset), resolved, won, be, ev

    cur_edge = settings.MIN_EDGE
    cur_prob = settings.MIN_PROBABILITY
    rejected_th = [r for r in th_rows if not r.passes]
    passing_th = [r for r in th_rows if r.passes]

    click.echo(
        f"_Current threshold floors: MIN_PROBABILITY={cur_prob}, "
        f"MIN_EDGE={cur_edge}. Ground truth = routine-METAR daily max "
        f"(advisory for °C cities; diverges from Polymarket's resolver)._"
    )
    click.echo()
    click.echo("| Candidate (min_prob, min_edge) | Newly-pass | Resolved | Won% | Break-even | EV/$1 |")
    click.echo("|---|---:|---:|---:|---:|---:|")

    # Baseline: quality of threshold evals that ALREADY pass.
    n, res, won, be, ev = _score(passing_th)
    won_pct = f"{won/res*100:5.1f}%" if res else "  n/a"
    click.echo(
        f"| baseline (already passing) | {n:,} | {res:,} | {won_pct} "
        f"| {be:.3f} | {ev:+.3f} |"
    )

    # Candidate grid: loosen prob (edge held current), loosen edge (prob held
    # current), then a couple of combined loosenings. Skip candidates that
    # aren't actually looser than the current floor on at least one axis.
    candidates: list[tuple[float, float]] = [
        (0.80, cur_edge), (0.78, cur_edge), (0.75, cur_edge), (0.70, cur_edge),
        (cur_prob, 0.07), (cur_prob, 0.05),
        (0.78, 0.05), (0.75, 0.05),
    ]
    seen: set[tuple[float, float]] = set()
    for cand_prob, cand_edge in candidates:
        if (cand_prob, cand_edge) in seen:
            continue
        seen.add((cand_prob, cand_edge))
        if cand_prob >= cur_prob and cand_edge >= cur_edge:
            continue  # not a loosening
        newly = [
            r for r in rejected_th
            if _check_filters(
                edge=r.edge, prob=r.model_prob, price=r.market_prob,
                routine_count=r.routine_count,
                minutes_to_close=r.minutes_to_close,
                depth=r.depth_usd,
                min_edge=cand_edge, min_probability=cand_prob,
            ) is None
        ]
        n, res, won, be, ev = _score(newly)
        won_pct = f"{won/res*100:5.1f}%" if res else "  n/a"
        click.echo(
            f"| prob≥{cand_prob}, edge≥{cand_edge} | {n:,} | {res:,} | "
            f"{won_pct} | {be:.3f} | {ev:+.3f} |"
        )
    click.echo()


async def _single_bucket_no_band_section(session, rows, op_of) -> None:
    """Tune the single-bucket NO guards against actual outcomes.

    The ``exactly`` (single-°C bucket) ``BUY_NO`` class is the bot's #1 loss
    source. The 2026-05-23 guards live in :func:`binary_market_edge` as the
    landing-band margin (``SINGLE_BUCKET_NO_BAND_MARGIN_F``) and the
    NO-confidence cap (``SINGLE_BUCKET_MAX_NO_PROB``). ``backtest-v2`` does NOT
    exercise ``binary_market_edge``, so telemetry is the only offline way to
    pick magnitudes.

    For each candidate ``(margin, cap)`` we recompute both guards over the
    bracket-like NO evals that CURRENTLY pass, using the forecast/observation
    context recorded on each row (``forecast_peak_f / current_max_f /
    hours_until_peak``, migration ``p6q7r8s9t0u1``):

      * Layer 2 — bucket window overlaps ``[current_max−margin,
        max(forecast_peak,current_max)+margin]`` (collapsing toward
        current_max past peak).
      * Layer 3 — ``no_prob`` clamped to ``min(no_prob, cap)``; the NO bet is
        then blocked if its recomputed edge / prob fall under the bracket-like
        global floors.

    The blocked set is scored against the actual routine-METAR daily max
    (NO wins iff the max lands OUTSIDE the bucket window). A good ``(margin,
    cap)`` blocks a set with a HIGH lost-rate (we kill losers) while leaving
    survivors whose won% clears break-even (mean NO price). Ground truth is
    routine-METAR-derived and diverges from Polymarket's resolver for °C
    cities — advisory, same caveat as the threshold band.
    """
    from src.config import settings
    from src.execution.binary_market import market_range_f
    from src.signals.mapper import (
        icao_for_location,
        icao_timezone,
        resolve_target_local_day,
    )

    click.echo("## Single-bucket NO guard tuning (bracket-like ops)")
    click.echo()

    no_rows = [
        r for r in rows
        if _op_class(op_of[r.market_id]) == "bracket-like"
        and r.signal_kind == "probability"
        and r.direction.value == "BUY_NO"
        and r.passes
        and r.forecast_peak_f is not None
        and r.current_max_f is not None
        and r.model_prob is not None
        and r.market_prob is not None
    ]
    if not no_rows:
        click.echo(
            "_No currently-passing bracket-like NO evals with forecast context "
            "(pre-2026-05-23 rows lack it)._"
        )
        click.echo()
        return

    market_map = await _load_market_map(session, (r.market_id for r in no_rows))
    created = [r.created_at for r in no_rows]
    win_start = min(created) - timedelta(days=2)
    win_end = max(created) + timedelta(days=2)
    icaos = {
        icao_for_location(m.parsed_location)
        for m in market_map.values() if m.parsed_location
    }
    daily_max = await _daily_max_by_station_day(session, icaos, win_start, win_end)

    def _bucket_window(row) -> tuple[int, int] | None:
        m = market_map.get(row.market_id)
        return market_range_f(m) if m is not None else None

    def _no_won(row) -> bool | None:
        """True if the NO side won (daily max landed OUTSIDE the bucket)."""
        m = market_map.get(row.market_id)
        rng = _bucket_window(row)
        if m is None or rng is None or m.parsed_location is None or m.end_date is None:
            return None
        icao = icao_for_location(m.parsed_location)
        if icao is None:
            return None
        target_day = resolve_target_local_day(m.end_date, icao_timezone(icao))
        if target_day is None:
            return None
        amax = daily_max.get((icao, target_day))
        if amax is None:
            return None
        low_f, high_f = rng
        return not (low_f <= round(amax) <= high_f)

    def _blocked(row, margin: float, cap: float) -> bool:
        """Would (margin, cap) block this currently-passing NO bet?"""
        # Layer 2 — landing band.
        rng = _bucket_window(row)
        if rng is not None:
            low_f, high_f = rng
            past_peak = (
                row.hours_until_peak is not None and row.hours_until_peak <= 0
            )
            upper_anchor = (
                row.current_max_f if past_peak
                else max(row.forecast_peak_f, row.current_max_f)
            )
            band_lo = row.current_max_f - margin
            band_hi = upper_anchor + margin
            if low_f <= band_hi and high_f >= band_lo:
                return True
        # Layer 3 — clamp no_prob to the cap, re-check edge/prob floors.
        capped_no = min(row.model_prob, cap)
        no_edge = round(capped_no - row.market_prob, 4)
        if no_edge < settings.MIN_EDGE or capped_no < settings.MIN_PROBABILITY:
            return True
        return False

    def _score(subset) -> tuple[int, int, int, float, float]:
        """(count, resolved, no_won, mean_no_price, mean_ev_per_$1)."""
        won = resolved = 0
        prices: list[float] = []
        evs: list[float] = []
        for r in subset:
            gt = _no_won(r)
            if gt is None:
                continue
            resolved += 1
            prices.append(r.market_prob)
            if gt:
                won += 1
                evs.append((1.0 / r.market_prob - 1.0) if r.market_prob > 0 else 0.0)
            else:
                evs.append(-1.0)
        be = (sum(prices) / len(prices)) if prices else 0.0
        ev = (sum(evs) / len(evs)) if evs else 0.0
        return len(subset), resolved, won, be, ev

    cur_margin = settings.SINGLE_BUCKET_NO_BAND_MARGIN_F
    cur_cap = settings.SINGLE_BUCKET_MAX_NO_PROB
    click.echo(
        f"_Currently-passing bracket-like NO evals: {len(no_rows):,}. "
        f"Current guards: margin={cur_margin}°F, cap={cur_cap}. "
        f"Ground truth = routine-METAR daily max (advisory for °C cities)._"
    )
    click.echo()

    # Baseline — the status quo passing NO set (these fired and bled).
    n, res, won, be, ev = _score(no_rows)
    won_pct = f"{won/res*100:5.1f}%" if res else "  n/a"
    click.echo(
        "| (margin, cap) | Blocked | Blk resolved | Blk LOST% (saved) | "
        "Survivors | Surv won% | Surv EV/$1 |"
    )
    click.echo("|---|---:|---:|---:|---:|---:|---:|")
    click.echo(
        f"| baseline {cur_margin}/{cur_cap} (all passing) | 0 | — | — | "
        f"{n:,} | {won_pct} | {ev:+.3f} |"
    )

    candidates: list[tuple[float, float]] = [
        (1.5, cur_cap), (2.0, cur_cap), (2.5, cur_cap), (3.0, cur_cap),
        (cur_margin, 0.90), (cur_margin, 0.88), (cur_margin, 0.85), (cur_margin, 0.80),
        (2.0, 0.88), (2.0, 0.85), (2.5, 0.85),
    ]
    seen: set[tuple[float, float]] = set()
    for margin, cap in candidates:
        if (margin, cap) in seen or (margin <= cur_margin and cap >= cur_cap):
            continue
        seen.add((margin, cap))
        blocked = [r for r in no_rows if _blocked(r, margin, cap)]
        survivors = [r for r in no_rows if r not in blocked]
        bn, bres, bwon, _, _ = _score(blocked)
        # Of the resolved blocked, the fraction the NO side would have LOST.
        blk_lost_pct = f"{(bres - bwon)/bres*100:5.1f}%" if bres else "  n/a"
        sn, sres, swon, _, sev = _score(survivors)
        surv_won_pct = f"{swon/sres*100:5.1f}%" if sres else "  n/a"
        click.echo(
            f"| margin {margin}, cap {cap} | {bn:,} | {bres:,} | {blk_lost_pct} | "
            f"{sn:,} | {surv_won_pct} | {sev:+.3f} |"
        )
    click.echo()


@main.command("evals-report")
@click.option("--days", default=30, show_default=True, help="Look-back window in days.")
@click.option(
    "--signal-kind", default=None,
    type=click.Choice(["probability", "lock"], case_sensitive=False),
    help="Restrict to one path. Default: both.",
)
@click.option(
    "--operator", "operator_class", default=None,
    type=click.Choice(["threshold", "bracket-like"], case_sensitive=False),
    help="Restrict to one operator class. Default: both.",
)
@click.option(
    "--since-epoch", "since_epoch", type=int, default=None,
    help="Only rows at/after this config_epochs.id's start (see `epochs`).",
)
def evals_report(
    days: int, signal_kind: str | None, operator_class: str | None,
    since_epoch: int | None = None,
) -> None:
    """Markdown filter-tuning report from the ``evaluation_logs`` table.

    For every per-side edge evaluation in the look-back window:
      * Pass-rate by ``reject_reason`` — which filter rejects most candidates?
      * Reject reasons split by operator class (threshold vs bracket-like) —
        separates the *expected* 2026-05-22/23 bracket-guard cuts from
        *threshold* volume suppressed by the bracket-era global floors.
      * Recoverable rejected band (threshold ops) — for candidate
        THRESHOLD_MIN_PROBABILITY / THRESHOLD_MIN_EDGE values, how many
        rejected threshold evals would newly pass and would they have WON
        against the actual routine-METAR daily max (with break-even / EV)?
      * Edge / probability / price distribution split by PASSING vs REJECTED.
      * Slippage post-mortem joining passing rows to the resulting Trade
        (``fill_price`` vs the snapshotted ``submit_yes_*`` quote).

    ``evaluation_logs`` is the source of truth for filter tuning — ``signals``
    only carries passing edges and is de-duplicated per (market, side), so
    this is the only place to count the rejected mass. Use ``--operator
    threshold`` to focus the recoverable-band analysis on the clean class.
    """
    from statistics import mean, median

    from sqlalchemy import func, select

    from src.db.engine import async_session
    from src.db.models import EvaluationLog, Trade

    def _pct(num: int, denom: int) -> str:
        return f"{(num / denom * 100):5.1f}%" if denom else "  n/a"

    async def _run() -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        async with async_session() as session:
            ep = await _epoch_start(session, since_epoch)
            if ep is not None and ep > cutoff:
                cutoff = ep
            stmt = (
                select(EvaluationLog)
                .where(EvaluationLog.created_at >= cutoff)
                .order_by(EvaluationLog.created_at)
            )
            if signal_kind:
                stmt = stmt.where(EvaluationLog.signal_kind == signal_kind.lower())
            rows = (await session.execute(stmt)).scalars().all()

            # Operator class lives on the Market row, not the eval row — batch
            # load once and tag each eval so every section can pivot on it.
            market_map = await _load_market_map(session, (r.market_id for r in rows))
            op_of = {
                r.market_id: (
                    market_map[r.market_id].parsed_operator
                    if r.market_id in market_map else None
                )
                for r in rows
            }
            if operator_class:
                rows = [r for r in rows if _op_class(op_of[r.market_id]) == operator_class]

            total = len(rows)
            if total == 0:
                click.echo(
                    f"No evaluation_logs rows in the last {days}d "
                    f"(signal_kind={signal_kind or 'any'}, "
                    f"operator={operator_class or 'any'}). "
                    "Has the scheduler been running?"
                )
                return

            passing = [r for r in rows if r.passes]
            rejected = [r for r in rows if not r.passes]

            kind_label = signal_kind.lower() if signal_kind else "all"
            op_label = operator_class or "all ops"
            click.echo(f"# Evaluation report — {kind_label} path, {op_label}, last {days}d")
            click.echo()
            click.echo(f"- **Total evaluations:** {total:,}")
            click.echo(f"- **Passing:** {len(passing):,}  ({_pct(len(passing), total).strip()})")
            click.echo(f"- **Rejected:** {len(rejected):,}  ({_pct(len(rejected), total).strip()})")
            click.echo()

            # --- 1. Breakdown by reject_reason. The string is the gate name
            #     emitted by `_check_filters`; bucketing by its prefix gives a
            #     useful "which filter kills the most candidates" summary.
            click.echo("## Reject reasons")
            click.echo()
            click.echo("| Reason prefix | Count | % of evaluations |")
            click.echo("|---|---:|---:|")
            buckets: dict[str, int] = {}
            for r in rejected:
                reason = r.reject_reason or "(unknown)"
                # Take the part before the first " <" or numeric token so
                # "edge 0.04 < 0.05" and "edge 0.02 < 0.05" both roll up
                # to "edge".
                prefix = reason.split(" ")[0]
                buckets[prefix] = buckets.get(prefix, 0) + 1
            for prefix, count in sorted(buckets.items(), key=lambda kv: -kv[1]):
                click.echo(f"| {prefix} | {count:,} | {_pct(count, total).strip()} |")
            click.echo()

            # --- 1b. Reject reasons split by operator class. THIS is the
            #     headline that separates the *expected* bracket-guard cuts
            #     (lead/landing-band/NO-cap — keep) from *threshold* volume
            #     suppressed by the bracket-era global MIN_PROBABILITY /
            #     MIN_EDGE (recoverable).
            click.echo("## Reject reasons by operator class")
            click.echo()
            click.echo("| Op class | Reason prefix | Count | % of class evals |")
            click.echo("|---|---|---:|---:|")
            class_totals: dict[str, int] = {}
            for r in rows:
                cls = _op_class(op_of[r.market_id])
                class_totals[cls] = class_totals.get(cls, 0) + 1
            class_reject: dict[tuple[str, str], int] = {}
            for r in rejected:
                cls = _op_class(op_of[r.market_id])
                prefix = (r.reject_reason or "(unknown)").split(" ")[0]
                class_reject[(cls, prefix)] = class_reject.get((cls, prefix), 0) + 1
            for (cls, prefix), count in sorted(
                class_reject.items(), key=lambda kv: (kv[0][0], -kv[1])
            ):
                click.echo(
                    f"| {cls} | {prefix} | {count:,} | "
                    f"{_pct(count, class_totals.get(cls, 0)).strip()} |"
                )
            click.echo()

            # --- 1c. Recoverable rejected band (THRESHOLD ops only). For a
            #     sweep of candidate THRESHOLD_MIN_PROBABILITY / THRESHOLD_MIN_EDGE
            #     values, count currently-rejected threshold evals that WOULD
            #     pass under the looser floors, and check whether they'd have
            #     WON against the actual routine-METAR daily max. A band whose
            #     would-have-won rate clears its break-even (mean price) is +EV
            #     and safe to recover; a coin-flip / -EV band is correctly cut.
            await _recoverable_band_section(session, rows, op_of)

            # --- 1d. Single-bucket NO guard tuning (BRACKET-LIKE ops). The
            #     mirror of the threshold band for the `exactly` NO loss class:
            #     recompute the landing-band + NO-cap guards over the passing
            #     bracket-like NO evals for a grid of (margin, cap) and score
            #     the blocked set against the actual daily max. Pick the
            #     (margin, cap) that kills the most losers while leaving
            #     survivors above break-even.
            await _single_bucket_no_band_section(session, rows, op_of)

            # --- 2. Split by signal_kind so the user can spot if e.g. the
            #     probability path is dominated by rejects while lock fires
            #     mostly pass (or vice versa).
            kinds: dict[str, tuple[int, int]] = {}
            for r in rows:
                pass_count, total_count = kinds.get(r.signal_kind, (0, 0))
                kinds[r.signal_kind] = (
                    pass_count + (1 if r.passes else 0),
                    total_count + 1,
                )
            if len(kinds) > 1:
                click.echo("## By signal_kind")
                click.echo()
                click.echo("| Kind | Total | Passing | Pass-rate |")
                click.echo("|---|---:|---:|---:|")
                for kind, (p, t) in sorted(kinds.items()):
                    click.echo(f"| {kind} | {t:,} | {p:,} | {_pct(p, t).strip()} |")
                click.echo()

            # --- 3. Per-field distributions, split PASS vs REJECT. Anything
            #     that diverges sharply between the two columns is a filter
            #     candidate (or a sign the existing gate is mis-tuned).
            click.echo("## Distributions (P25 / P50 / P75)")
            click.echo()
            click.echo("| Field | Group | P25 | P50 | P75 |")
            click.echo("|---|---|---:|---:|---:|")
            for field in ("edge", "model_prob", "market_prob", "depth_usd"):
                for group_label, group_rows in (
                    ("PASS", passing), ("REJECT", rejected),
                ):
                    values = [
                        getattr(r, field) for r in group_rows
                        if getattr(r, field) is not None
                    ]
                    q = _quantiles(values)
                    if q is None:
                        continue
                    click.echo(
                        f"| {field} | {group_label} "
                        f"| {q[0]:7.3f} | {q[1]:7.3f} | {q[2]:7.3f} |"
                    )
            click.echo()

            # --- 4. Slippage post-mortem. For passing rows that ended up
            #     producing a filled Trade, compare submitted quote against
            #     realised fill_price. Matching is best-effort: nearest Trade
            #     by (market_id, direction, ±60s of submit_at).
            click.echo("## Slippage (passing edges with realised fills)")
            click.echo()
            trade_rows = (
                await session.execute(
                    select(Trade)
                    .where(Trade.fill_price.is_not(None))
                    .where(Trade.opened_at >= cutoff)
                )
            ).scalars().all()
            # Index by (market_id, direction) → list of Trades for fast lookup.
            trade_index: dict[tuple[str, str], list[Trade]] = {}
            for t in trade_rows:
                key = (t.market_id, t.direction.value)
                trade_index.setdefault(key, []).append(t)

            slippages: list[float] = []
            spread_walked: list[float] = []
            matched = 0
            for r in passing:
                trades = trade_index.get((r.market_id, r.direction.value), [])
                if not trades:
                    continue
                t = trades[0]  # first one; multiple rare and won't change the picture
                if t.submit_yes_ask is None or t.submit_yes_bid is None:
                    continue
                mid = (t.submit_yes_bid + t.submit_yes_ask) / 2.0
                # For BUY_YES the effective cost is the ask; slippage =
                # fill_price - mid (positive = we paid above mid).
                # For BUY_NO, fill_price is the NO-token fill and mid is
                # the YES mid — translate to NO mid (1 - mid).
                ref_mid = mid if r.direction.value == "BUY_YES" else (1.0 - mid)
                slippages.append(t.fill_price - ref_mid)
                spread_walked.append(t.submit_yes_ask - t.submit_yes_bid)
                matched += 1

            if not slippages:
                click.echo("_No passing evals with realised fills in this window._")
            else:
                click.echo(f"- **Matched fills:** {matched}")
                click.echo(
                    f"- **Mean slippage vs mid:** {mean(slippages):+.4f} "
                    f"(median {median(slippages):+.4f})"
                )
                click.echo(
                    f"- **Mean submit-time spread:** {mean(spread_walked):.4f} "
                    f"(median {median(spread_walked):.4f})"
                )
                click.echo()
                click.echo(
                    "_A positive slippage with a non-trivial spread means we're "
                    "paying away spread on entry; investigate whether MIN_DEPTH_USD "
                    "or the close-buffer gate needs tightening._"
                )

    asyncio.run(_run())


@main.command("decisions-report")
@click.option("--days", default=7, show_default=True, help="Look-back window in days.")
@click.option(
    "--signal-kind", default=None,
    type=click.Choice(["probability", "lock"], case_sensitive=False),
    help="Restrict to one path. Default: both.",
)
@click.option(
    "--operator", "operator_class", default=None,
    type=click.Choice(["threshold", "bracket-like"], case_sensitive=False),
    help="Restrict to one operator class. Default: both.",
)
@click.option(
    "--since-epoch", "since_epoch", type=int, default=None,
    help="Only rows at/after this config_epochs.id's start (see `epochs`).",
)
def decisions_report(
    days: int, signal_kind: str | None, operator_class: str | None,
    since_epoch: int | None = None,
) -> None:
    """Markdown post-filter funnel report from the ``decision_logs`` table.

    ``evaluation_logs`` answers "did the edge clear ``_check_filters``?";
    ``decision_logs`` answers "what happened next." Use this to confirm that
    a filter loosening actually produces trades rather than dying downstream
    at ``stake_below_min`` / ``cap_exceeded`` / ``drawdown_paused`` — the
    failure mode that would make a recovered threshold band invisible.

    Sections:
      * Outcomes overall + split by operator class.
      * For ``stake_below_min`` / ``drawdown_paused``: the binding constraint
        (``size_reason``), raw Kelly %, and depth from ``metadata_json``.
    """
    from sqlalchemy import select

    from src.db.engine import async_session
    from src.db.models import DecisionLog

    def _pct(num: int, denom: int) -> str:
        return f"{(num / denom * 100):5.1f}%" if denom else "  n/a"

    async def _run() -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        async with async_session() as session:
            ep = await _epoch_start(session, since_epoch)
            if ep is not None and ep > cutoff:
                cutoff = ep
            stmt = (
                select(DecisionLog)
                .where(DecisionLog.created_at >= cutoff)
                .order_by(DecisionLog.created_at)
            )
            if signal_kind:
                stmt = stmt.where(DecisionLog.signal_kind == signal_kind.lower())
            rows = (await session.execute(stmt)).scalars().all()

            market_map = await _load_market_map(session, (r.market_id for r in rows))
            op_of = {
                r.market_id: (
                    market_map[r.market_id].parsed_operator
                    if r.market_id in market_map else None
                )
                for r in rows
            }
            if operator_class:
                rows = [r for r in rows if _op_class(op_of[r.market_id]) == operator_class]

            total = len(rows)
            if total == 0:
                click.echo(
                    f"No decision_logs rows in the last {days}d "
                    f"(signal_kind={signal_kind or 'any'}, "
                    f"operator={operator_class or 'any'})."
                )
                return

            kind_label = signal_kind.lower() if signal_kind else "all"
            op_label = operator_class or "all ops"
            click.echo(f"# Decision funnel — {kind_label} path, {op_label}, last {days}d")
            click.echo()
            click.echo(f"- **Total decisions:** {total:,}")
            click.echo()

            # --- 1. Outcome counts. The post-filter funnel: success branches
            #     (signal_written/trade_pending/trade_filled) vs skip/fail.
            click.echo("## Outcomes")
            click.echo()
            click.echo("| Outcome | Count | % |")
            click.echo("|---|---:|---:|")
            outcomes: dict[str, int] = {}
            for r in rows:
                outcomes[r.outcome] = outcomes.get(r.outcome, 0) + 1
            for outcome, count in sorted(outcomes.items(), key=lambda kv: -kv[1]):
                click.echo(f"| {outcome} | {count:,} | {_pct(count, total).strip()} |")
            click.echo()

            # --- 2. Outcomes by operator class — does the threshold class
            #     reach trade_* or stall at a sizing/exposure gate?
            click.echo("## Outcomes by operator class")
            click.echo()
            click.echo("| Op class | Outcome | Count |")
            click.echo("|---|---|---:|")
            class_outcome: dict[tuple[str, str], int] = {}
            for r in rows:
                key = (_op_class(op_of[r.market_id]), r.outcome)
                class_outcome[key] = class_outcome.get(key, 0) + 1
            for (cls, outcome), count in sorted(
                class_outcome.items(), key=lambda kv: (kv[0][0], -kv[1])
            ):
                click.echo(f"| {cls} | {outcome} | {count:,} |")
            click.echo()

            # --- 3. Binding-constraint detail for the sizing-stall outcomes.
            #     metadata_json carries size_reason / kelly_pct / depth_usd
            #     (populated 2026-05-17) — tells us WHICH cap zeroed the stake.
            stall = [
                r for r in rows
                if r.outcome in ("stake_below_min", "drawdown_paused")
                and r.metadata_json
            ]
            if stall:
                click.echo("## Sizing-stall binding constraints")
                click.echo()
                click.echo("| Outcome | size_reason | Count |")
                click.echo("|---|---|---:|")
                reason_counts: dict[tuple[str, str], int] = {}
                for r in stall:
                    sr = str(r.metadata_json.get("size_reason", "(none)"))
                    reason_counts[(r.outcome, sr)] = reason_counts.get((r.outcome, sr), 0) + 1
                for (outcome, sr), count in sorted(
                    reason_counts.items(), key=lambda kv: -kv[1]
                ):
                    click.echo(f"| {outcome} | {sr} | {count:,} |")
                click.echo()

    asyncio.run(_run())


@main.command("backtest-v2")
@click.option("--days", default=30, show_default=True, help="Days of history to backtest.")
@click.option("--stations", default="", help="Comma-separated ICAO codes (default: all stations with data).")
def backtest_v2(days: int, stations: str) -> None:
    """Backtest the distribution probability engine against historical outcomes."""

    async def _backtest() -> None:
        from src.risk.simulate import simulate_distribution_pipeline

        if stations:
            station_list = [s.strip().upper() for s in stations.split(",")]
        else:
            # Discover stations from DB
            from sqlalchemy import select, distinct
            from src.db.engine import async_session
            from src.db.models import MetarObservation

            async with async_session() as session:
                result = await session.execute(
                    select(distinct(MetarObservation.station_icao))
                )
                station_list = [row[0] for row in result.all() if row[0]]

        if not station_list:
            click.echo("No stations found. Provide --stations or ensure METAR data exists.")
            return

        click.echo(f"Backtesting {len(station_list)} stations over {days} days...")
        result = await simulate_distribution_pipeline(station_list, days_back=days)

        click.echo(f"\n=== Distribution Backtest Results ===")
        click.echo(f"  Days evaluated:     {result.num_days}")
        click.echo(f"  Calibration error:  {result.calibration_error:.4f}")
        click.echo(f"  Brier score:        {result.brier_score:.6f}")

        if result.per_bucket:
            click.echo(f"\n  Per-bucket calibration (top 10 by count):")
            top = sorted(result.per_bucket, key=lambda b: b.count, reverse=True)[:10]
            for b in top:
                click.echo(
                    f"    {b.bucket_value:3d}°F: predicted={b.predicted_avg:.3f} "
                    f"actual={b.actual_rate:.3f} (n={b.count})"
                )

        threshold = 0.03
        if result.calibration_error <= threshold:
            click.echo(f"\n  PASS: calibration error {result.calibration_error:.4f} <= {threshold}")
        else:
            click.echo(f"\n  FAIL: calibration error {result.calibration_error:.4f} > {threshold}")

    asyncio.run(_backtest())


@main.command()
def migrate() -> None:
    """Run pending database migrations (adds missing columns)."""

    async def _migrate() -> None:
        from sqlalchemy import text

        from src.db.engine import engine

        columns = [
            ("trades", "order_id", "VARCHAR"),
            ("trades", "token_id", "VARCHAR"),
            ("trades", "fill_price", "FLOAT"),
            ("trades", "filled_size", "FLOAT"),
            ("trades", "exchange_status", "VARCHAR"),
        ]

        async with engine.begin() as conn:
            for table, col, dtype in columns:
                stmt = f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {dtype}"
                await conn.execute(text(stmt))
                click.echo(f"  OK: {table}.{col}")

        click.echo("Migration complete.")

    asyncio.run(_migrate())


@main.command()
def approve() -> None:
    """Approve USDC + Conditional Token contracts for Polymarket trading.

    Sends 6 on-chain transactions (2 tokens x 3 spender contracts).
    Requires POLYMARKET_PRIVATE_KEY in .env and POL for gas.
    """
    from src.config import settings

    if not settings.POLYMARKET_PRIVATE_KEY:
        click.echo("Error: POLYMARKET_PRIVATE_KEY not set in .env")
        raise SystemExit(1)

    from eth_account import Account
    from web3 import Web3

    RPC_URLS = [
        "https://polygon-bor-rpc.publicnode.com",
        "https://rpc.ankr.com/polygon",
        "https://polygon.drpc.org",
        "https://polygon-rpc.com",
    ]
    CHAIN_ID = 137

    # V2 SDK provides addresses for both V1 (legacy) and V2 (current) exchanges
    # in one struct. Approve the V2 exchanges + the neg-risk adapter, since
    # post-2026-04-28 only V2 routes orders.
    from py_clob_client_v2.config import get_contract_config
    cfg = get_contract_config(CHAIN_ID)
    USDC = cfg.collateral  # pUSD post-migration; V2 SDK keeps this in `collateral`
    CTF = cfg.conditional_tokens

    SPENDERS = {
        "CTF Exchange V2":       cfg.exchange_v2,
        "Neg Risk Exchange V2":  cfg.neg_risk_exchange_v2,
        "Neg Risk Adapter":      cfg.neg_risk_adapter,
    }
    click.echo(f"Collateral: {USDC} (pUSD)")
    click.echo(f"Spenders:   V2 exchanges + neg-risk adapter")

    ERC20_ABI = [
        {
            "inputs": [
                {"name": "spender", "type": "address"},
                {"name": "amount", "type": "uint256"},
            ],
            "name": "approve",
            "outputs": [{"name": "", "type": "bool"}],
            "stateMutability": "nonpayable",
            "type": "function",
        },
        {
            "inputs": [
                {"name": "owner", "type": "address"},
                {"name": "spender", "type": "address"},
            ],
            "name": "allowance",
            "outputs": [{"name": "", "type": "uint256"}],
            "stateMutability": "view",
            "type": "function",
        },
    ]

    ERC1155_ABI = [
        {
            "inputs": [
                {"name": "operator", "type": "address"},
                {"name": "approved", "type": "bool"},
            ],
            "name": "setApprovalForAll",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function",
        },
        {
            "inputs": [
                {"name": "account", "type": "address"},
                {"name": "operator", "type": "address"},
            ],
            "name": "isApprovedForAll",
            "outputs": [{"name": "", "type": "bool"}],
            "stateMutability": "view",
            "type": "function",
        },
    ]

    account = Account.from_key(settings.POLYMARKET_PRIVATE_KEY)
    address = account.address
    max_uint = 2**256 - 1

    click.echo(f"Wallet: {address}")

    # Try RPC endpoints until one works
    from web3.middleware import ExtraDataToPOAMiddleware

    w3 = None
    for rpc_url in RPC_URLS:
        try:
            _w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 30}))
            # Polygon (Bor) is PoA; without this, eth_getTransactionReceipt
            # blows up with "extraData is N bytes, but should be 32".
            _w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
            _w3.eth.get_balance(address)  # test connection
            w3 = _w3
            click.echo(f"Connected to {rpc_url}")
            break
        except Exception:
            click.echo(f"  RPC {rpc_url} failed, trying next...")

    if w3 is None:
        click.echo("Error: all Polygon RPC endpoints failed")
        raise SystemExit(1)

    bal = w3.eth.get_balance(address)
    click.echo(f"POL balance: {w3.from_wei(bal, 'ether'):.4f} (for gas)")
    if bal == 0:
        click.echo("Error: wallet has no POL for gas fees")
        raise SystemExit(1)

    usdc = w3.eth.contract(address=Web3.to_checksum_address(USDC), abi=ERC20_ABI)
    ctf = w3.eth.contract(address=Web3.to_checksum_address(CTF), abi=ERC1155_ABI)

    gas_price = w3.eth.gas_price  # eth_gasPrice RPC — no extraData issue
    nonce = w3.eth.get_transaction_count(address)

    def send_tx(tx_data):
        nonlocal nonce
        tx_data["nonce"] = nonce
        tx_data["chainId"] = CHAIN_ID
        tx_data["from"] = address
        tx_data["gasPrice"] = gas_price
        if "gas" not in tx_data:
            tx_data["gas"] = w3.eth.estimate_gas(tx_data)
        signed = w3.eth.account.sign_transaction(tx_data, private_key=settings.POLYMARKET_PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        nonce += 1
        return receipt

    # --- Check existing approvals first ---
    click.echo("\n=== Checking existing approvals ===")
    needs_usdc = []
    needs_ctf = []
    for name, spender in SPENDERS.items():
        spender_cs = Web3.to_checksum_address(spender)
        usdc_ok = usdc.functions.allowance(address, spender_cs).call() > 0
        ctf_ok = ctf.functions.isApprovedForAll(address, spender_cs).call()
        status_u = "OK" if usdc_ok else "NEEDED"
        status_c = "OK" if ctf_ok else "NEEDED"
        click.echo(f"  {name}: USDC={status_u}, CTF={status_c}")
        if not usdc_ok:
            needs_usdc.append((name, spender_cs))
        if not ctf_ok:
            needs_ctf.append((name, spender_cs))

    if not needs_usdc and not needs_ctf:
        click.echo("\nAll approvals already in place!")
    else:
        total_txs = len(needs_usdc) + len(needs_ctf)
        click.echo(f"\nNeed {total_txs} approval transaction(s). Proceeding...")

        # --- USDC approvals ---
        for name, spender in needs_usdc:
            click.echo(f"  Approving USDC for {name}...", nl=False)
            tx = usdc.functions.approve(spender, max_uint).build_transaction({"from": address, "gasPrice": gas_price})
            receipt = send_tx(tx)
            ok = "OK" if receipt["status"] == 1 else "FAILED"
            click.echo(f" {ok} (tx: {receipt['transactionHash'].hex()[:16]}...)")

        # --- CTF approvals ---
        for name, spender in needs_ctf:
            click.echo(f"  Approving CTF for {name}...", nl=False)
            tx = ctf.functions.setApprovalForAll(spender, True).build_transaction({"from": address, "gasPrice": gas_price})
            receipt = send_tx(tx)
            ok = "OK" if receipt["status"] == 1 else "FAILED"
            click.echo(f" {ok} (tx: {receipt['transactionHash'].hex()[:16]}...)")

    # --- Notify CLOB server ---
    click.echo("\nNotifying Polymarket CLOB server...")
    from py_clob_client_v2.clob_types import AssetType, BalanceAllowanceParams

    from src.execution.polymarket_client import build_clob_client

    client = build_clob_client()
    if client is None:
        click.echo("Error: POLYMARKET_PRIVATE_KEY not set in .env")
        raise SystemExit(1)
    client.update_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
    click.echo("Done! You can now trade on Polymarket.")


@main.command("test-trade")
@click.option("--amount", default=1.0, show_default=True, help="USDC amount to spend.")
def test_trade(amount: float) -> None:
    """Place a tiny test trade on the most liquid weather market.

    Uses a FOK market order. Verifies the full execution pipeline works.
    """

    async def _test_trade() -> None:
        from src.config import settings

        if not settings.POLYMARKET_PRIVATE_KEY:
            click.echo("Error: POLYMARKET_PRIVATE_KEY not set in .env")
            raise SystemExit(1)

        from eth_account import Account
        from py_clob_client_v2.clob_types import MarketOrderArgsV2 as MarketOrderArgs, OrderType

        from src.execution.polymarket_client import build_clob_client

        account = Account.from_key(settings.POLYMARKET_PRIVATE_KEY)
        funder_address = account.address

        client = build_clob_client()
        if client is None:
            click.echo("Error: POLYMARKET_PRIVATE_KEY not set in .env")
            raise SystemExit(1)

        import httpx
        from web3 import Web3

        # --- Diagnostic: check on-chain USDC.e balance ---
        # For UI-onboarded wallets, funds live on the configured funder
        # (proxy/safe), not on the EOA. Check the configured funder.
        configured_funder = settings.POLYMARKET_FUNDER_ADDRESS or funder_address
        USDC_E = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
        USDC_NATIVE = "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359"
        BAL_ABI = [{"inputs": [{"name": "account", "type": "address"}], "name": "balanceOf", "outputs": [{"name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"}]

        w3 = Web3(Web3.HTTPProvider("https://polygon-bor-rpc.publicnode.com", request_kwargs={"timeout": 10}))
        addr = Web3.to_checksum_address(configured_funder)
        usdc_e_bal = w3.eth.contract(address=Web3.to_checksum_address(USDC_E), abi=BAL_ABI).functions.balanceOf(addr).call()
        usdc_n_bal = w3.eth.contract(address=Web3.to_checksum_address(USDC_NATIVE), abi=BAL_ABI).functions.balanceOf(addr).call()

        click.echo(f"EOA:    {funder_address}")
        click.echo(f"Funder: {configured_funder}  (sig_type={settings.POLYMARKET_SIGNATURE_TYPE})")
        click.echo(f"USDC.e balance:      ${usdc_e_bal / 1e6:.2f}  (required by Polymarket)")
        click.echo(f"Native USDC balance: ${usdc_n_bal / 1e6:.2f}")

        if usdc_e_bal == 0 and usdc_n_bal > 0:
            click.echo("\nError: Your USDC is native USDC, but Polymarket requires USDC.e.")
            click.echo("Swap native USDC -> USDC.e on a DEX (e.g. Uniswap on Polygon).")
            raise SystemExit(1)
        if usdc_e_bal == 0:
            click.echo("\nError: No USDC.e on the configured funder. Deposit USDC.e to trade.")
            raise SystemExit(1)

        click.echo("Finding a tradeable market...")

        # Search for markets with valid CLOB token IDs
        found_market = None
        found_token = None
        found_tick_size = None
        found_neg_risk = None

        for search_tag in ["weather", "climate", None]:
            params = {"limit": 50, "active": "true", "order": "liquidity", "ascending": "false"}
            if search_tag:
                params["tag"] = search_tag

            async with httpx.AsyncClient() as http:
                resp = await http.get(
                    "https://gamma-api.polymarket.com/markets",
                    params=params,
                    timeout=15,
                )
                resp.raise_for_status()
                markets = resp.json()

            for m in markets:
                token_ids = m.get("clobTokenIds") or []
                if isinstance(token_ids, str):
                    import json
                    token_ids = json.loads(token_ids)
                if len(token_ids) < 2 or not token_ids[0]:
                    continue
                # Validate the token is actually tradeable on the CLOB
                try:
                    ts = client.get_tick_size(token_ids[0])
                    nr = client.get_neg_risk(token_ids[0])
                    found_market = m
                    found_token = token_ids[0]
                    found_tick_size = ts
                    found_neg_risk = nr
                    break
                except Exception as exc:
                    click.echo(f"  Skipping {m.get('question', '?')[:40]}... ({exc})")
                    continue

            if found_market:
                if search_tag:
                    click.echo(f"Found tradeable market via tag '{search_tag}'")
                break

        if not found_market or not found_token:
            click.echo("No tradeable market found with valid CLOB token IDs.")
            return

        click.echo(f"Market: {found_market.get('question', 'unknown')[:80]}")
        click.echo(f"YES token: {found_token[:20]}...")
        click.echo(f"Tick size: {found_tick_size}, Neg risk: {found_neg_risk}")
        click.echo(f"Amount: ${amount:.2f}")

        from py_clob_client_v2.order_builder.constants import BUY

        # Place a small FOK market buy on YES
        click.echo("Placing FOK market order...")
        market_order = MarketOrderArgs(token_id=found_token, amount=amount, side=BUY)
        signed = client.create_market_order(market_order)
        resp = client.post_order(signed, OrderType.FOK)

        click.echo(f"Response: {resp}")

        if resp.get("orderID"):
            click.echo(f"\nOrder ID: {resp['orderID']}")
            click.echo(f"Status: {resp.get('status', 'unknown')}")
            click.echo("\nTest trade successful! The execution pipeline works.")
        else:
            click.echo(f"\nOrder failed: {resp.get('errorMsg', 'unknown error')}")
            click.echo("This may be normal for FOK on a thin book. The API connection works.")

    asyncio.run(_test_trade())


@main.group()
def bet() -> None:
    """Place and manage manual bets on Polymarket."""


@bet.command("place")
@click.argument("market")
@click.option("--side", required=True, type=click.Choice(["yes", "no"], case_sensitive=False), help="Buy YES or NO.")
@click.option("--amount", required=True, type=float, help="USDC amount to spend.")
@click.option("--yes", "-y", "skip_confirm", is_flag=True, help="Skip confirmation prompt.")
@click.option("--ignore-cap", is_flag=True, help="Bypass daily spend cap check.")
def bet_place(market: str, side: str, amount: float, skip_confirm: bool, ignore_cap: bool) -> None:
    """Place a FOK market order on any Polymarket market.

    MARKET can be a Polymarket URL, slug, or condition ID.
    """

    async def _place() -> None:
        from src.config import settings
        from src.bet_helpers import (
            extract_token_ids,
            format_market_info,
            get_clob_client,
            get_usdc_balance,
            resolve_market,
        )

        if not settings.POLYMARKET_PRIVATE_KEY:
            click.echo("Error: POLYMARKET_PRIVATE_KEY not set in .env")
            raise SystemExit(1)

        # --- Resolve market ---
        click.echo(f"Resolving market: {market}")
        mkt = await resolve_market(market)
        if mkt is None:
            click.echo("Error: market not found.")
            raise SystemExit(1)

        # --- Extract token IDs ---
        token_pair = extract_token_ids(mkt)
        if token_pair is None:
            click.echo("Error: market has no tradeable token IDs.")
            raise SystemExit(1)

        yes_token, no_token = token_pair
        token_id = yes_token if side.lower() == "yes" else no_token

        # --- Check collateral balance ---
        # Polymarket V2 (post-2026-04-28) uses pUSD as collateral instead of
        # USDC.e. We still print all three so a user with legacy USDC.e
        # sitting around can see they need to deposit it via the UI.
        click.echo("Checking wallet balance...")
        pusd, usdc_e, usdc_native = await get_usdc_balance(settings.POLYMARKET_PRIVATE_KEY)
        click.echo(f"  pUSD:        ${pusd:.2f}  (V2 collateral)")
        click.echo(f"  USDC.e:      ${usdc_e:.2f}  (legacy V1 collateral)")
        click.echo(f"  Native USDC: ${usdc_native:.2f}")

        if pusd < amount:
            click.echo(f"\nError: insufficient pUSD (${pusd:.2f} < ${amount:.2f}).")
            if usdc_e >= amount or usdc_native >= amount:
                click.echo("Deposit via polymarket.com UI to convert USDC.e/native USDC into pUSD.")
            raise SystemExit(1)

        # --- Daily spend cap ---
        if not ignore_cap:
            from src.db.engine import async_session
            from src.execution.polymarket_client import get_daily_spend

            async with async_session() as session:
                daily_spend = await get_daily_spend(session)

            remaining = settings.DAILY_SPEND_CAP_USD - daily_spend
            click.echo(f"  24h spend:   ${daily_spend:.2f} / ${settings.DAILY_SPEND_CAP_USD:.2f}")
            if daily_spend + amount > settings.DAILY_SPEND_CAP_USD:
                click.echo(
                    f"\nError: daily spend cap would be exceeded "
                    f"(${daily_spend:.2f} + ${amount:.2f} > ${settings.DAILY_SPEND_CAP_USD:.2f}). "
                    f"Use --ignore-cap to override."
                )
                raise SystemExit(1)

        # --- Display market info ---
        click.echo(f"\n=== Market ===")
        click.echo(format_market_info(mkt))
        click.echo(f"\n=== Order ===")
        click.echo(f"  Side:   BUY {side.upper()}")
        click.echo(f"  Amount: ${amount:.2f}")
        click.echo(f"  Type:   FOK (Fill-or-Kill)")

        # --- Confirmation ---
        if not skip_confirm:
            click.echo()
            if not click.confirm("Place this order?"):
                click.echo("Cancelled.")
                return

        # --- Place order ---
        click.echo("\nInitialising CLOB client...")
        client = get_clob_client()

        # Validate token is tradeable
        try:
            tick_size = client.get_tick_size(token_id)
            neg_risk = client.get_neg_risk(token_id)
        except Exception as exc:
            click.echo(f"Error: token not tradeable on CLOB ({exc})")
            raise SystemExit(1)

        click.echo(f"  Tick size: {tick_size}, Neg risk: {neg_risk}")

        from py_clob_client_v2.clob_types import MarketOrderArgsV2 as MarketOrderArgs, OrderType
        from py_clob_client_v2.order_builder.constants import BUY

        click.echo("Placing FOK market order...")
        market_order = MarketOrderArgs(token_id=token_id, amount=amount, side=BUY)
        signed = client.create_market_order(market_order)
        resp = client.post_order(signed, OrderType.FOK)

        # --- Display result ---
        click.echo(f"\n=== Result ===")
        order_id = resp.get("orderID")
        status = (resp.get("status") or "unknown").lower()

        if order_id:
            click.echo(f"  Order ID: {order_id}")
            click.echo(f"  Status:   {status}")

            if status == "matched":
                try:
                    order = client.get_order(order_id)
                    trades = order.get("associate_trades", [])
                    if trades:
                        fill_price = float(trades[0].get("price", 0))
                        click.echo(f"  Fill price: {fill_price:.4f}")
                    size_matched = order.get("size_matched")
                    if size_matched:
                        click.echo(f"  Shares:     {float(size_matched):.2f}")
                except Exception:
                    click.echo("  (could not fetch fill details)")

                click.echo("\nOrder filled successfully.")

            elif status == "delayed":
                import time

                click.echo("\nOrder accepted but not yet filled. Polling for fill...")
                filled = False
                for attempt in range(3):
                    time.sleep(2)
                    try:
                        order = client.get_order(order_id)
                        current_status = order.get("status", "unknown")
                        click.echo(f"  [{attempt + 1}/3] Status: {current_status}")
                        if current_status.lower() == "matched":
                            try:
                                trades = order.get("associate_trades", [])
                                if trades:
                                    fill_price = float(trades[0].get("price", 0))
                                    click.echo(f"  Fill price: {fill_price:.4f}")
                                size_matched = order.get("size_matched")
                                if size_matched:
                                    click.echo(f"  Shares:     {float(size_matched):.2f}")
                            except Exception:
                                click.echo("  (could not fetch fill details)")
                            click.echo("\nOrder filled successfully.")
                            filled = True
                            break
                    except Exception:
                        click.echo(f"  [{attempt + 1}/3] (could not fetch status)")

                if not filled:
                    click.echo(f"\nOrder is still delayed (not filled).")
                    click.echo(f"  To cancel: python -m src.cli bet cancel {order_id}")
                    if click.confirm("Cancel this order now?"):
                        try:
                            client.cancel(order_id)
                            click.echo("Order cancelled.")
                        except Exception as exc:
                            click.echo(f"Failed to cancel: {exc}")
                    else:
                        click.echo("Order left open. Check later with: python -m src.cli bet orders")
            else:
                click.echo(f"\nOrder placed (status: {status}).")
        else:
            error_msg = resp.get("errorMsg", "unknown error")
            click.echo(f"  Status: FAILED")
            click.echo(f"  Error:  {error_msg}")
            click.echo("\nThe order was not filled. This may be normal for FOK on a thin order book.")

    asyncio.run(_place())


@bet.command("diagnose")
@click.option("--post-test", is_flag=True, help="Actually POST tiny ($1) FOK orders against a neg-risk and a non-neg-risk market to capture the API response. Costs up to $2 if filled.")
@click.option("--rotate-api-key", is_flag=True, help="Force-create fresh API credentials (instead of deriving existing ones) before signing.")
def bet_diagnose(post_test: bool, rotate_api_key: bool) -> None:
    """Diagnose the wallet/signature setup against Polymarket.

    Prints the EOA, configured funder, signature_type, USDC.e balance at
    each, and the proxy address Polymarket has registered for the EOA (if
    any). Useful for triaging ``order_version_mismatch`` errors.
    """
    import httpx
    from eth_account import Account
    from web3 import Web3

    from src.config import settings

    if not settings.POLYMARKET_PRIVATE_KEY:
        click.echo("Error: POLYMARKET_PRIVATE_KEY not set in .env")
        raise SystemExit(1)

    eoa = Account.from_key(settings.POLYMARKET_PRIVATE_KEY).address
    configured_funder = settings.POLYMARKET_FUNDER_ADDRESS or eoa
    sig_type = settings.POLYMARKET_SIGNATURE_TYPE

    click.echo("=== Configured ===")
    click.echo(f"  EOA:           {eoa}")
    click.echo(f"  Funder:        {configured_funder}{' (= EOA)' if configured_funder.lower() == eoa.lower() else ''}")
    click.echo(f"  sig_type:      {sig_type}  ({'EOA' if sig_type == 0 else 'POLY_PROXY' if sig_type == 1 else 'POLY_GNOSIS_SAFE' if sig_type == 2 else 'unknown'})")

    USDC_E_ADDR = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"  # legacy V1 collateral
    from py_clob_client_v2.config import get_contract_config
    pusd_addr = get_contract_config(settings.POLYMARKET_CHAIN_ID).collateral
    BAL_ABI = [{"inputs": [{"name": "account", "type": "address"}], "name": "balanceOf", "outputs": [{"name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"}]

    w3 = Web3(Web3.HTTPProvider("https://polygon-bor-rpc.publicnode.com", request_kwargs={"timeout": 10}))
    usdc_e = w3.eth.contract(address=Web3.to_checksum_address(USDC_E_ADDR), abi=BAL_ABI)
    pusd = w3.eth.contract(address=Web3.to_checksum_address(pusd_addr), abi=BAL_ABI)

    click.echo("\n=== On-chain collateral balances ===")
    for label, addr in [("EOA", eoa), ("Funder", configured_funder)]:
        if addr.lower() == eoa.lower() and label == "Funder":
            continue
        cs = Web3.to_checksum_address(addr)
        try:
            usdc_bal = usdc_e.functions.balanceOf(cs).call() / 1e6
            pusd_bal = pusd.functions.balanceOf(cs).call() / 1e6
            click.echo(f"  {label} ({addr}):")
            click.echo(f"    USDC.e (old collateral):  ${usdc_bal:.2f}")
            click.echo(f"    pUSD   (new collateral):  ${pusd_bal:.2f}")
        except Exception as exc:
            click.echo(f"  {label} ({addr}): lookup failed ({exc})")

    click.echo("\n=== Polymarket proxy lookup ===")
    proxy_endpoints = [
        ("CLOB proxy-wallet-address", f"https://clob.polymarket.com/proxy-wallet-address?address={eoa}"),
        ("Gamma get-account", f"https://gamma-api.polymarket.com/account?address={eoa}"),
    ]
    discovered_proxies: list[tuple[str, str]] = []
    with httpx.Client(timeout=10) as http:
        for label, url in proxy_endpoints:
            try:
                r = http.get(url)
                click.echo(f"  {label}: HTTP {r.status_code}")
                if r.status_code == 200:
                    body = r.text.strip()
                    click.echo(f"    body: {body[:300]}")
                    try:
                        data = r.json()
                        candidates: list[str] = []
                        if isinstance(data, dict):
                            for key in ("proxyAddress", "proxy_address", "proxy", "address", "smartWalletAddress"):
                                v = data.get(key)
                                if isinstance(v, str) and v.startswith("0x") and len(v) == 42:
                                    candidates.append(v)
                        elif isinstance(data, str) and data.startswith("0x") and len(data) == 42:
                            candidates.append(data)
                        for c in candidates:
                            if c.lower() != eoa.lower() and c != "0x0000000000000000000000000000000000000000":
                                discovered_proxies.append((label, c))
                    except Exception:
                        pass
            except Exception as exc:
                click.echo(f"  {label}: request failed ({exc})")

    if discovered_proxies:
        click.echo("\n=== Discovered proxies (different from EOA) ===")
        for label, addr in discovered_proxies:
            try:
                bal = usdc_e.functions.balanceOf(Web3.to_checksum_address(addr)).call() / 1e6
                click.echo(f"  {label}: {addr}  USDC.e=${bal:.2f}")
            except Exception:
                click.echo(f"  {label}: {addr}  (balance lookup failed)")
        click.echo("\nIf one of these holds your real Polymarket balance, set:")
        addr = discovered_proxies[0][1]
        click.echo(f"  POLYMARKET_FUNDER_ADDRESS={addr}")
        click.echo("  POLYMARKET_SIGNATURE_TYPE=2   # try 1 if 2 still fails")

    # ----- Build & inspect a sample signed order against BOTH neg-risk and regular -----
    click.echo("\n=== Sample order signing (inspect what SDK actually sends) ===")

    from py_clob_client_v2.client import ClobClient
    from py_clob_client_v2.clob_types import MarketOrderArgsV2, OrderType
    from py_clob_client_v2.config import get_contract_config as _v2_get_cfg
    from py_clob_client_v2.order_builder.constants import BUY

    if not settings.POLYMARKET_PRIVATE_KEY:
        click.echo("  (no client; POLYMARKET_PRIVATE_KEY missing)")
        return

    # V2 ClobClient: chain_id is positional arg #2.
    temp_client = ClobClient(
        settings.POLYMARKET_HOST,
        settings.POLYMARKET_CHAIN_ID,
        key=settings.POLYMARKET_PRIVATE_KEY,
    )
    if rotate_api_key:
        click.echo("  Forcing fresh API key creation...")
        try:
            creds = temp_client.create_api_key()
            click.echo(f"    new api_key: {str(creds.api_key)[:8]}...")
        except Exception as exc:  # noqa: BLE001
            click.echo(f"    create_api_key failed ({exc}); falling back to derive")
            creds = temp_client.create_or_derive_api_key()
    else:
        creds = temp_client.create_or_derive_api_key()
        click.echo(f"  Derived api_key: {str(creds.api_key)[:8]}...")

    funder = settings.POLYMARKET_FUNDER_ADDRESS or eoa
    client = ClobClient(
        settings.POLYMARKET_HOST,
        settings.POLYMARKET_CHAIN_ID,
        key=settings.POLYMARKET_PRIVATE_KEY,
        creds=creds,
        signature_type=settings.POLYMARKET_SIGNATURE_TYPE,
        funder=funder,
    )

    v2_cfg = _v2_get_cfg(settings.POLYMARKET_CHAIN_ID)
    click.echo(f"  V2 regular exchange:  {v2_cfg.exchange_v2}")
    click.echo(f"  V2 neg-risk exchange: {v2_cfg.neg_risk_exchange_v2}")
    click.echo(f"  V2 collateral (pUSD): {v2_cfg.collateral}")

    # Polymarket-side state for this wallet
    click.echo("\n=== Polymarket-side state for this wallet ===")
    try:
        addr_seen_by_clob = client.get_address()
        click.echo(f"  client.get_address():     {addr_seen_by_clob}")
    except Exception as exc:  # noqa: BLE001
        click.echo(f"  get_address(): {type(exc).__name__}: {exc}")
    try:
        keys = client.get_api_keys()
        click.echo(f"  get_api_keys():           {keys}")
    except Exception as exc:  # noqa: BLE001
        click.echo(f"  get_api_keys(): {type(exc).__name__}: {exc}")
    try:
        closed_only = client.get_closed_only_mode()
        click.echo(f"  get_closed_only_mode():   {closed_only}")
    except Exception as exc:  # noqa: BLE001
        click.echo(f"  get_closed_only_mode(): {type(exc).__name__}: {exc}")
    # Polymarket also exposes /balance-allowance which tells us what they think
    # we have available — useful contrast with on-chain balance.
    try:
        from py_clob_client_v2.clob_types import AssetType, BalanceAllowanceParams
        ba = client.get_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
        click.echo(f"  get_balance_allowance():  {ba}")
    except Exception as exc:  # noqa: BLE001
        click.echo(f"  get_balance_allowance(): {type(exc).__name__}: {exc}")

    # Find one neg-risk and one non-neg-risk market to test side-by-side
    with httpx.Client(timeout=15) as http:
        r = http.get(
            "https://gamma-api.polymarket.com/markets",
            params={"limit": 100, "active": "true", "order": "liquidity", "ascending": "false"},
        )
        candidates = r.json() if r.status_code == 200 else []

    found: dict[bool, dict] = {}  # neg_risk_flag -> {token, question, slug}
    for m in candidates:
        if len(found) == 2:
            break
        token_ids = m.get("clobTokenIds") or []
        if isinstance(token_ids, str):
            import json as _json
            token_ids = _json.loads(token_ids)
        if len(token_ids) < 2 or not token_ids[0]:
            continue
        try:
            nr = client.get_neg_risk(token_ids[0])
        except Exception:  # noqa: BLE001
            continue
        if nr in found:
            continue
        found[nr] = {
            "token": token_ids[0],
            "question": m.get("question", "?"),
            "slug": m.get("slug", ""),
        }

    for nr_flag in (False, True):
        if nr_flag not in found:
            click.echo(f"\n  -- {'NEG-RISK' if nr_flag else 'REGULAR'}: no tradeable market found --")
            continue
        info = found[nr_flag]
        click.echo(f"\n  -- {'NEG-RISK' if nr_flag else 'REGULAR'} test market --")
        click.echo(f"    Question:      {info['question'][:70]}")
        click.echo(f"    Token id:      {info['token'][:32]}...")
        try:
            args = MarketOrderArgsV2(token_id=info["token"], amount=1.0, side=BUY)
            signed = client.create_market_order(args)
        except Exception as exc:  # noqa: BLE001
            click.echo(f"    Signing failed: {type(exc).__name__}: {exc}")
            continue
        # SignedOrderV2 is a plain dataclass with fields:
        # salt, maker, signer, tokenId, makerAmount, takerAmount, side,
        # signatureType, timestamp, metadata, builder, expiration, signature
        import dataclasses as _dc
        body = _dc.asdict(signed)
        for k in ("signer", "maker", "signatureType", "salt", "timestamp", "metadata", "builder", "expiration"):
            if k in body:
                click.echo(f"    {k}: {body[k]}")
        sig = body.get("signature", "")
        click.echo(f"    signature: {str(sig)[:24]}...")

        if not post_test:
            continue

        # Actually post the order and capture the API response
        click.echo(f"    Posting $1 FOK to API...")
        try:
            resp = client.post_order(signed, OrderType.FOK)
            click.echo(f"    POST response: {resp}")
        except Exception as exc:  # noqa: BLE001
            click.echo(f"    POST failed: {type(exc).__name__}: {exc}")

    if not post_test:
        click.echo("\n  (Use --post-test to actually POST $1 orders and capture API responses)")
    if not rotate_api_key:
        click.echo("  (Use --rotate-api-key to force-create fresh API credentials before signing)")


@bet.command("info")
@click.argument("market")
def bet_info(market: str) -> None:
    """Display details about a Polymarket market.

    MARKET can be a Polymarket URL, slug, or condition ID.
    """

    async def _info() -> None:
        from src.bet_helpers import (
            extract_token_ids,
            format_market_info,
            resolve_market,
        )

        click.echo(f"Resolving market: {market}")
        mkt = await resolve_market(market)
        if mkt is None:
            click.echo("Error: market not found.")
            raise SystemExit(1)

        click.echo(f"\n=== Market Info ===")
        click.echo(format_market_info(mkt))

        token_pair = extract_token_ids(mkt)
        if token_pair:
            yes_token, no_token = token_pair
            click.echo(f"\n=== Token IDs ===")
            click.echo(f"  YES: {yes_token}")
            click.echo(f"  NO:  {no_token}")
        else:
            click.echo("\n  No tradeable token IDs found.")

    asyncio.run(_info())


@bet.command("search")
@click.argument("query")
@click.option("--limit", default=10, show_default=True, help="Max results to display.")
def bet_search(query: str, limit: int) -> None:
    """Search active Polymarket markets by keyword."""

    async def _search() -> None:
        from src.bet_helpers import search_markets

        click.echo(f"Searching for: {query}")
        results = await search_markets(query, limit=limit)

        if not results:
            click.echo("No markets found.")
            return

        click.echo(f"\n{'#':<4} {'Question':<60} {'YES':>6} {'Volume':>12} {'ID'}")
        click.echo("-" * 110)

        for i, m in enumerate(results, 1):
            question = (m.get("question") or "")[:58]
            outcome_prices = m.get("outcomePrices", "[]")
            if isinstance(outcome_prices, str):
                import json
                try:
                    outcome_prices = json.loads(outcome_prices)
                except (json.JSONDecodeError, TypeError):
                    outcome_prices = []

            yes_str = f"{float(outcome_prices[0]):.0%}" if outcome_prices else "?"
            vol = m.get("volume") or m.get("volumeNum") or 0
            vol_str = f"${float(vol):,.0f}"
            mid = m.get("id") or m.get("conditionId") or ""

            click.echo(f"{i:<4} {question:<60} {yes_str:>6} {vol_str:>12} {mid}")

    asyncio.run(_search())


@bet.command("find")
@click.option("--city", default=None, help="City name (e.g. Phoenix, Austin).")
@click.option("--station", default=None, help="ICAO station code (e.g. KPHX, KAUS).")
@click.option("--date", "date_str", default=None, help="Date filter (e.g. 'April 19').")
@click.option("--variable", default=None, help="Weather variable (temperature, precipitation, wind_speed).")
@click.option("--hours", default=72.0, show_default=True, help="Look-ahead window in hours.")
def bet_find(city: str | None, station: str | None, date_str: str | None, variable: str | None, hours: float) -> None:
    """Find Polymarket markets matching weather location/station (DB lookup).

    Much faster than 'bet search' — queries the local markets database
    populated by the 15-minute scanner instead of paginating the Gamma API.

    Examples:
      bet find --city Phoenix
      bet find --city Austin --date "April 19"
      bet find --station KPHX
      bet find --city Denver --variable temperature
    """

    async def _find() -> None:
        from src.db.engine import async_session
        from src.signals.reverse_lookup import (
            find_markets_for_city,
            find_markets_for_observation,
            find_markets_for_station,
        )

        if not city and not station:
            click.echo("Error: provide --city or --station")
            raise SystemExit(1)

        async with async_session() as session:
            if station:
                from src.signals.mapper import cities_for_icao

                city_names = cities_for_icao(station)
                click.echo(f"Station {station} -> cities: {', '.join(c.title() for c in city_names) or 'none'}")

                # Plain station lookup. (The observation-enriched path
                # was removed with the WX pipeline 2026-05-30 — see
                # docs/graveyard.md. `find_markets_for_observation` is
                # still available for callers that have an out-of-band
                # observation source.)
                markets = await find_markets_for_station(
                    session, station, variable=variable, hours_ahead=hours,
                )
            else:
                click.echo(f"Looking up markets for: {city}")
                markets = await find_markets_for_city(
                    session, city, variable=variable, date_str=date_str,
                )

            if not markets:
                click.echo("No markets found.")
                return

            click.echo(f"\n{'#':<4} {'Question':<52} {'YES':>5} {'Var':>14} {'Thresh':>7} {'Op':>8} {'Date':>12} {'Liq':>10}")
            click.echo("-" * 116)

            for i, m in enumerate(markets, 1):
                q = (m.question or "")[:50]
                yes_str = f"{m.current_yes_price:.0%}" if m.current_yes_price else "?"
                var = (m.parsed_variable or "?")[:12]
                thresh = f"{m.parsed_threshold:.0f}F" if m.parsed_threshold is not None else "?"
                op = (m.parsed_operator or "?")[:6]
                date = (m.parsed_target_date or "?")[:10]
                liq = f"${m.liquidity:,.0f}" if m.liquidity else "?"
                click.echo(f"{i:<4} {q:<52} {yes_str:>5} {var:>14} {thresh:>7} {op:>8} {date:>12} {liq:>10}")

    asyncio.run(_find())


@bet.command("cancel")
@click.argument("order_id")
def bet_cancel(order_id: str) -> None:
    """Cancel an open order on Polymarket."""

    async def _cancel() -> None:
        from src.config import settings
        from src.bet_helpers import get_clob_client

        if not settings.POLYMARKET_PRIVATE_KEY:
            click.echo("Error: POLYMARKET_PRIVATE_KEY not set in .env")
            raise SystemExit(1)

        click.echo(f"Cancelling order: {order_id}")
        client = get_clob_client()

        try:
            client.cancel(order_id)
            click.echo("Order cancelled successfully.")
        except Exception as exc:
            click.echo(f"Error: failed to cancel order ({exc})")
            raise SystemExit(1)

    asyncio.run(_cancel())


@bet.command("orders")
@click.option("--limit", "max_orders", default=20, show_default=True, help="Max orders to display.")
def bet_orders(max_orders: int) -> None:
    """List recent orders with their statuses (matched, delayed, cancelled, etc.)."""

    async def _orders() -> None:
        from src.config import settings
        from src.bet_helpers import get_clob_client

        if not settings.POLYMARKET_PRIVATE_KEY:
            click.echo("Error: POLYMARKET_PRIVATE_KEY not set in .env")
            raise SystemExit(1)

        click.echo("Fetching orders...")
        client = get_clob_client()

        from py_clob_client_v2.clob_types import OpenOrderParams
        orders = client.get_open_orders(OpenOrderParams())

        if not orders:
            click.echo("No orders found.")
            return

        # Sort by timestamp descending if available
        orders = orders[:max_orders]

        click.echo(f"\n=== Orders ({len(orders)}) ===")
        click.echo(f"{'ID':<14} {'Status':<12} {'Side':<6} {'Size':>8} {'Price':>8} {'Matched':>8}  Token")
        click.echo("-" * 80)
        for o in orders:
            oid = (o.get("id") or "?")[:12]
            status = o.get("status", "?")
            side = o.get("side", "?")
            size = o.get("original_size", o.get("size", "?"))
            price = o.get("price", "?")
            matched = o.get("size_matched", "0")
            asset = (o.get("asset_id") or "")[:16]
            click.echo(f"  {oid}..  {status:<12} {side:<6} {size:>8} {price:>8} {matched:>8}  {asset}...")

        # Show cancel hint for non-terminal orders
        active = [o for o in orders if o.get("status") in ("live", "delayed")]
        if active:
            click.echo(f"\n{len(active)} active order(s). Cancel with:")
            for o in active:
                click.echo(f"  python -m src.cli bet cancel {o.get('id', '?')}")

    asyncio.run(_orders())


async def _load_active_positions_from_db(
    *,
    include_settled: bool = False,
) -> tuple[list[tuple[str, dict]], dict[str, dict]]:
    """DB-driven counterpart to ``_load_active_positions``.

    Queries the local ``trades`` table for OPEN + WON-unredeemed rows
    (and additionally WON-redeemed + LOST when ``include_settled=True``),
    eager-loads ``Market``, and synthesises position dicts in the same
    shape the portfolio renderer expects from the CLOB-history helper.

    Fast path — no CLOB trade-history pagination, no per-condition
    ``/markets/{cid}`` round-trip. Trade rows with a NULL ``token_id``
    are skipped since they can't be matched to a CLOB asset; reach for
    ``--full-scan`` if you need to surface those.
    """
    from sqlalchemy import or_, select
    from sqlalchemy.orm import selectinload

    from src.db.engine import async_session
    from src.db.models import (
        Trade as TradeModel,
        TradeDirection,
        TradeStatus as TradeStatusEnum,
    )

    async with async_session() as session:
        active_clause = TradeModel.status == TradeStatusEnum.OPEN
        won_unredeemed_clause = (
            (TradeModel.status == TradeStatusEnum.WON)
            & TradeModel.redeemed_at.is_(None)
        )
        clauses = [active_clause, won_unredeemed_clause]
        if include_settled:
            clauses.append(
                (TradeModel.status == TradeStatusEnum.WON)
                & TradeModel.redeemed_at.is_not(None)
            )
            clauses.append(TradeModel.status == TradeStatusEnum.LOST)
        stmt = (
            select(TradeModel)
            .where(or_(*clauses))
            .options(selectinload(TradeModel.market))
        )
        rows = (await session.execute(stmt)).scalars().all()

    positions: dict[str, dict] = {}
    token_to_market: dict[str, dict] = {}

    for t in rows:
        token_id = t.token_id
        if not token_id:
            continue

        fill_price_eff = t.fill_price if t.fill_price else t.entry_price
        if not fill_price_eff or fill_price_eff <= 0:
            continue

        size = t.filled_size or 0.0
        if size <= 0 and t.stake_usd:
            size = t.stake_usd / fill_price_eff
        if size <= 0:
            continue
        cost = size * fill_price_eff

        pos = positions.setdefault(
            token_id,
            {
                "asset_id": token_id,
                "size": 0.0,
                "cost": 0.0,
                "side": "LONG",
                "market": "",
                "_statuses": set(),
            },
        )
        pos["size"] += size
        pos["cost"] += cost
        pos["_statuses"].add(
            t.status.value if hasattr(t.status, "value") else str(t.status)
        )

        if token_id not in token_to_market and t.market is not None:
            mkt = t.market
            outcome = (
                "Yes" if t.direction == TradeDirection.BUY_YES else "No"
            )
            token_to_market[token_id] = {
                "question": mkt.question or "",
                "slug": mkt.slug or "",
                "tokens": [{"token_id": token_id, "outcome": outcome}],
                "current_yes_price": mkt.current_yes_price,
                "_db_direction": (
                    "BUY_YES"
                    if t.direction == TradeDirection.BUY_YES
                    else "BUY_NO"
                ),
            }

    for pos in positions.values():
        pos["avg_price"] = (
            pos["cost"] / pos["size"] if pos["size"] else 0.0
        )

    ordered = sorted(positions.items(), key=lambda kv: kv[0])
    return ordered, token_to_market


async def _load_active_positions(
    *,
    show_all: bool = False,
) -> tuple[list[tuple[str, dict]], dict[str, dict], object]:
    """Shared position loader for ``bet portfolio`` and ``bet sell``.

    Returns ``(ordered_positions, token_to_market, clob_client)`` where
    ``ordered_positions`` is a stable, asset_id-sorted list of
    ``(asset_id, position_dict)`` tuples — same ordering used to assign
    the [#N] index in ``bet portfolio``, so a numeric identifier passed
    to ``bet sell`` resolves to the same position the user just saw.
    """
    from src.bet_helpers import (
        compute_positions,
        get_clob_client,
        get_ctf_readonly,
        get_trades_history,
    )

    client = get_clob_client()
    trades = get_trades_history(client)
    if not trades:
        return [], {}, client

    positions = compute_positions(trades)
    if not positions:
        return [], {}, client

    import httpx

    token_to_market: dict[str, dict] = {}
    async with httpx.AsyncClient(timeout=15) as http:
        seen_conds: dict[str, dict | None] = {}
        for asset_id, pos in positions.items():
            cond = pos.get("market", "")
            if not cond:
                continue
            if cond not in seen_conds:
                try:
                    resp = await http.get(
                        f"https://clob.polymarket.com/markets/{cond}",
                    )
                    resp.raise_for_status()
                    seen_conds[cond] = resp.json()
                except Exception:
                    seen_conds[cond] = None
            if seen_conds[cond]:
                token_to_market[asset_id] = seen_conds[cond]

    if not show_all:
        resolved_assets = [
            asset_id for asset_id, mkt in token_to_market.items()
            if mkt.get("closed") is True
            or str(mkt.get("closed", "")).lower() == "true"
        ]
        if resolved_assets:
            try:
                _w3, ctf, wallet_addr, _rpc = get_ctf_readonly()
            except Exception:
                pass
            else:
                for asset_id in resolved_assets:
                    try:
                        bal = ctf.functions.balanceOf(
                            wallet_addr, int(asset_id)
                        ).call()
                    except Exception:
                        continue
                    if bal == 0:
                        positions.pop(asset_id, None)
                        token_to_market.pop(asset_id, None)

    ordered = sorted(positions.items(), key=lambda kv: kv[0])
    return ordered, token_to_market, client


def _resolve_position_target(
    selector: str,
    ordered: list[tuple[str, dict]],
) -> tuple[str, dict] | None:
    """Resolve a CLI selector to a (asset_id, position) entry.

    Accepts either a 1-based numeric index from the portfolio listing or
    an asset_id prefix (≥4 chars). Returns None on no match.
    """
    sel = selector.strip()
    if sel.isdigit():
        idx = int(sel)
        if 1 <= idx <= len(ordered):
            return ordered[idx - 1]
        return None
    if len(sel) < 4:
        return None
    matches = [(aid, p) for aid, p in ordered if aid.startswith(sel)]
    if len(matches) == 1:
        return matches[0]
    return None


@bet.command("portfolio")
@click.option("--all", "show_all", is_flag=True, help="Include settled positions (WON-redeemed + LOST) alongside active ones.")
@click.option("--history", is_flag=True, help="Show full trade history instead of positions (uses CLOB).")
@click.option(
    "--full-scan",
    "full_scan",
    is_flag=True,
    help="Discover positions by paginating the entire CLOB trade history instead of querying the local Trade table. Slower; use as a safety net when local DB may have drifted from on-chain state.",
)
def bet_portfolio(show_all: bool, history: bool, full_scan: bool) -> None:
    """Show open positions and P&L from your Polymarket trades.

    Default path is DB-driven: queries the local ``trades`` table for
    OPEN + WON-unredeemed rows and fetches live prices in parallel from
    the CLOB orderbook. Pass ``--full-scan`` to fall back to the legacy
    CLOB-trade-history walk (slower; surfaces positions taken outside
    the bot).

    Each active position is printed with an ``[#N]`` index and an 8-char
    asset_id prefix; either form can be passed to ``bet sell``.
    """

    async def _portfolio() -> None:
        from src.config import settings
        from src.bet_helpers import (
            get_clob_client,
            get_open_orders,
            get_trades_history,
            get_usdc_balance,
        )
        from src.execution.polymarket_client import get_best_bid_ask

        if not settings.POLYMARKET_PRIVATE_KEY:
            click.echo("Error: POLYMARKET_PRIVATE_KEY not set in .env")
            raise SystemExit(1)

        click.echo("Connecting to Polymarket CLOB...")
        client = get_clob_client()

        # --- Wallet balance ---
        click.echo("Fetching wallet balance...")
        pusd, usdc_e, usdc_native = await get_usdc_balance(settings.POLYMARKET_PRIVATE_KEY)
        click.echo(f"\n=== Wallet ===")
        click.echo(f"  pUSD:        ${pusd:.2f}")
        click.echo(f"  USDC.e:      ${usdc_e:.2f}")
        if usdc_native > 0:
            click.echo(f"  Native USDC: ${usdc_native:.2f}")

        # --- Open orders ---
        click.echo("\nFetching open orders...")
        open_orders = get_open_orders(client)
        click.echo(f"\n=== Open Orders ({len(open_orders)}) ===")
        if open_orders:
            for o in open_orders:
                oid = o.get("id", "?")[:12]
                side = o.get("side", "?")
                size = o.get("original_size", o.get("size", "?"))
                price = o.get("price", "?")
                status = o.get("status", "?")
                asset = o.get("asset_id", "")[:16]
                click.echo(f"  {oid}...  {side:<5} size={size} price={price} status={status}  token={asset}...")
        else:
            click.echo("  No open orders.")

        if history:
            # History view still walks the CLOB trade-match feed — DB
            # only records bot-initiated trades, so a full audit needs
            # the on-chain stream.
            click.echo("\nFetching trade history (CLOB)...")
            trades = get_trades_history(client)
            if not trades:
                click.echo("\n  No trades found.")
                return
            # Resolve market questions for all unique asset_ids
            import httpx
            asset_ids = {t.get("asset_id", "") for t in trades}
            asset_ids.discard("")
            token_questions: dict[str, str] = {}
            async with httpx.AsyncClient(timeout=15) as http:
                for aid in asset_ids:
                    try:
                        resp = await http.get(
                            "https://gamma-api.polymarket.com/markets",
                            params={"clob_token_ids": aid},
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        if data:
                            mkt = data[0] if isinstance(data, list) else data
                            token_questions[aid] = (mkt.get("question") or "?")[:50]
                    except Exception:
                        pass

            click.echo(f"\n=== Trade History ({len(trades)}) ===")
            for t in trades:
                ts = t.get("match_time") or t.get("created_at") or t.get("timestamp", "?")
                if isinstance(ts, str) and len(ts) > 19:
                    ts = ts[:19]
                side = t.get("side", "?")
                size = float(t.get("size", 0))
                price = float(t.get("price", 0))
                cost = size * price
                aid = t.get("asset_id", "")
                question = token_questions.get(aid, aid[:20] + "...")
                click.echo(f"  {ts}  {side:<4} {size:>7.2f} @ ${price:.4f}  (${cost:.2f})  {question}")
            return

        # --- Positions ---
        if full_scan:
            click.echo("\nLoading positions (CLOB full scan)...")
            ordered, token_to_market, _ = await _load_active_positions(
                show_all=show_all
            )
            source_label = "CLOB"
        else:
            click.echo("\nLoading positions (DB)...")
            ordered, token_to_market = await _load_active_positions_from_db(
                include_settled=show_all
            )
            source_label = "DB"

        if not ordered:
            click.echo("\n=== Positions (0) ===")
            hint = ""
            if not full_scan:
                hint += " Try --full-scan to walk CLOB trade history."
            if not show_all:
                hint += " Use --all to include settled positions."
            click.echo("  No active or unredeemed positions." + hint)
            return

        mode_label = "all-time" if show_all else "active+unredeemed"
        click.echo(
            f"\n=== Positions ({len(ordered)}) [{mode_label}] [{source_label}] ==="
        )
        click.echo("  Reference each position by [#N] or its 8-char ID prefix in `bet sell`.")

        # Fetch live best-bid/ask for every position in parallel — turns
        # the prior per-position sequential round-trips into a single
        # gather. Each call still hits the 30 s orderbook cache in
        # polymarket_client, so back-to-back invocations are warm.
        async def _quote(aid: str) -> tuple[str, tuple[float, float] | None]:
            try:
                return aid, await asyncio.to_thread(get_best_bid_ask, aid)
            except Exception:
                return aid, None

        quote_results = await asyncio.gather(
            *(_quote(aid) for aid, _ in ordered)
        )
        quotes: dict[str, tuple[float, float] | None] = dict(quote_results)

        total_cost = 0.0
        total_value = 0.0

        for idx, (asset_id, pos) in enumerate(ordered, start=1):
            size = pos["size"]
            avg_price = pos["avg_price"]
            cost = abs(pos["cost"])
            total_cost += cost

            mkt = token_to_market.get(asset_id)
            question = ""
            token_side = ""
            if mkt:
                question = mkt.get("question", "")
                tokens = mkt.get("tokens", [])
                for tok in tokens:
                    if tok.get("token_id") == asset_id:
                        token_side = (tok.get("outcome") or "").upper()
                        break

            current_price: float | None = None
            quote = quotes.get(asset_id)
            if quote is not None:
                bid, ask = quote
                current_price = (bid + ask) / 2
            elif mkt and mkt.get("current_yes_price") is not None:
                # DB fallback: Market.current_yes_price is the YES mid
                # at last ingest — invert for BUY_NO holders.
                yes_p = float(mkt["current_yes_price"])
                current_price = (
                    1.0 - yes_p
                    if mkt.get("_db_direction") == "BUY_NO"
                    else yes_p
                )

            # Resolved markets have no live orderbook; fall back to the
            # known payout when we know the trade settled (DB path only).
            statuses = pos.get("_statuses") if isinstance(pos, dict) else None
            if current_price is None and statuses:
                if "won" in statuses:
                    current_price = 1.0
                elif "lost" in statuses:
                    current_price = 0.0

            current_value = (
                abs(size) * current_price if current_price is not None else None
            )
            if current_value is not None:
                total_value += current_value
                pnl = current_value - cost
                pnl_pct = (pnl / cost * 100) if cost else 0
                pnl_str = f"{'+'if pnl>=0 else ''}{pnl:.2f} ({pnl_pct:+.1f}%)"
            else:
                pnl_str = "?"

            header_id = f"[#{idx}] {asset_id[:8]}"
            if question:
                click.echo(f"\n  {header_id}  {question}")
            else:
                click.echo(f"\n  {header_id}  Token: {asset_id[:20]}...")
            side_label = f"LONG {token_side}" if token_side else pos["side"]
            status_tag = ""
            if statuses:
                if "won" in statuses and "open" not in statuses:
                    status_tag = "  [WON-unredeemed]"
                elif "lost" in statuses and "open" not in statuses:
                    status_tag = "  [LOST]"
            click.echo(f"    Side:      {side_label}{status_tag}")
            click.echo(f"    Size:      {abs(size):.2f} shares")
            click.echo(f"    Avg entry: ${avg_price:.4f}")
            click.echo(f"    Cost:      ${cost:.2f}")
            if current_price is not None:
                click.echo(f"    Current:   ${current_price:.4f}")
                click.echo(f"    Value:     ${current_value:.2f}")
            click.echo(f"    P&L:       {pnl_str}")
            if mkt:
                slug = mkt.get("slug", "")
                if slug:
                    click.echo(f"    URL:       https://polymarket.com/event/{slug}")

        # --- Summary ---
        click.echo(f"\n=== Summary ===")
        click.echo(f"  Total cost:     ${total_cost:.2f}")
        if total_value > 0:
            click.echo(f"  Current value:  ${total_value:.2f}")
            total_pnl = total_value - total_cost
            click.echo(f"  Total P&L:      {'+'if total_pnl>=0 else ''}{total_pnl:.2f}")
        click.echo(f"  USDC.e balance: ${usdc_e:.2f}")

    asyncio.run(_portfolio())


@bet.command("sell")
@click.argument("position", required=False)
@click.option("--all", "sell_all", is_flag=True, help="Sell every active position.")
@click.option("--slippage", "slippage_cents", default=2.0, show_default=True,
              type=float, help="Cents below best bid for the FAK floor.")
@click.option("--yes", "-y", "skip_confirm", is_flag=True, help="Skip confirmation prompts.")
def bet_sell(
    position: str | None,
    sell_all: bool,
    slippage_cents: float,
    skip_confirm: bool,
) -> None:
    """Sell an active position at the live best bid (FAK).

    POSITION is the [#N] index or asset_id prefix (≥4 chars) shown in
    ``bet portfolio``. Pass ``--all`` to close every active position.
    Each sell is a Fill-And-Kill MarketOrderArgsV2 with side=SELL at a
    floor of ``best_bid - --slippage`` cents; partial fills are kept,
    unfilled remainders are cancelled by the matching engine.
    """
    if not position and not sell_all:
        click.echo("Usage: bet sell <#N|asset_prefix>  OR  bet sell --all")
        raise SystemExit(1)
    if position and sell_all:
        click.echo("Pass either POSITION or --all, not both.")
        raise SystemExit(1)

    async def _sell() -> None:
        from src.config import settings
        from src.execution.polymarket_client import is_live, sell_position

        if not settings.POLYMARKET_PRIVATE_KEY:
            click.echo("Error: POLYMARKET_PRIVATE_KEY not set in .env")
            raise SystemExit(1)

        click.echo("Connecting to Polymarket CLOB...")
        ordered, token_to_market, _client = await _load_active_positions(show_all=False)
        if not ordered:
            click.echo("No active positions to sell.")
            return

        if sell_all:
            targets = list(ordered)
        else:
            assert position is not None
            match = _resolve_position_target(position, ordered)
            if match is None:
                click.echo(
                    f"No position matched '{position}'. "
                    f"Run `bet portfolio` to see [#N] / 8-char prefixes."
                )
                raise SystemExit(1)
            targets = [match]

        live = is_live()
        if not live:
            click.echo("(AUTO_EXECUTE=false → simulating sells, no orders posted)")

        click.echo(f"\nWill sell {len(targets)} position(s):")
        for asset_id, pos in targets:
            mkt = token_to_market.get(asset_id) or {}
            question = (mkt.get("question") or asset_id[:24])[:60]
            tokens = mkt.get("tokens", [])
            token_side = ""
            for tok in tokens:
                if tok.get("token_id") == asset_id:
                    token_side = (tok.get("outcome") or "").upper()
                    break
            click.echo(
                f"  {asset_id[:8]}  {abs(pos['size']):.2f} {token_side or 'shares'}  "
                f"avg=${pos['avg_price']:.4f}  ({question})"
            )

        if not skip_confirm:
            if not click.confirm(f"\nSubmit {len(targets)} sell order(s)?", default=False):
                click.echo("Aborted.")
                return

        total_cost = 0.0
        total_proceeds = 0.0
        ok_count = 0

        for asset_id, pos in targets:
            size = abs(pos["size"])
            cost = abs(pos["cost"])
            mkt = token_to_market.get(asset_id) or {}
            question = (mkt.get("question") or asset_id[:24])[:60]
            click.echo(f"\n[{asset_id[:8]}] {question}")
            click.echo(f"  Selling {size:.2f} shares (avg entry ${pos['avg_price']:.4f}, cost ${cost:.2f})...")

            result = await sell_position(
                asset_id, size, max_slippage_cents=slippage_cents,
            )
            if not result["ok"]:
                click.echo(
                    f"  FAILED: status={result['status']} "
                    f"error={result.get('error') or 'unknown'}"
                )
                continue

            ok_count += 1
            total_cost += cost
            proceeds = result["proceeds_usd"]
            total_proceeds += proceeds
            tag = "(dry-run)" if result["dry_run"] else ""
            click.echo(
                f"  OK {tag} order_id={result['order_id'] or '-'}  "
                f"filled={result['filled_size']:.2f}@${result['fill_price']:.4f}  "
                f"proceeds=${proceeds:.2f}  floor=${result['limit_price']:.4f}  "
                f"best_bid=${result['best_bid']:.4f}"
            )
            if proceeds > 0 and cost > 0:
                pnl = proceeds - cost
                pnl_pct = pnl / cost * 100
                sign = "+" if pnl >= 0 else ""
                click.echo(f"  Realised P&L: {sign}{pnl:.2f} ({sign}{pnl_pct:.1f}%)")

        click.echo(f"\n=== Summary ===")
        click.echo(f"  Submitted: {ok_count}/{len(targets)}")
        if ok_count:
            click.echo(f"  Total cost:    ${total_cost:.2f}")
            click.echo(f"  Total proceeds: ${total_proceeds:.2f}")
            net = total_proceeds - total_cost
            sign = "+" if net >= 0 else ""
            click.echo(f"  Net P&L:       {sign}{net:.2f}")

    asyncio.run(_sell())


@bet.command("redeem")
@click.option("--all", "redeem_all", is_flag=True, help="Redeem all resolved positions.")
@click.option("--yes", "-y", "skip_confirm", is_flag=True, help="Skip confirmation prompt.")
@click.option(
    "--reconcile",
    "reconcile_only",
    is_flag=True,
    help="Skip on-chain redemption; only stamp Trade.redeemed_at for WON trades whose tokens are already gone on-chain.",
)
@click.option(
    "--full-scan",
    "full_scan",
    is_flag=True,
    help="Discover positions by pulling the full CLOB trade history instead of querying the local Trade table. Slower; use as a safety net when local DB may have drifted from on-chain state.",
)
def bet_redeem(
    redeem_all: bool,
    skip_confirm: bool,
    reconcile_only: bool,
    full_scan: bool,
) -> None:
    """Redeem winnings from resolved Polymarket positions on-chain."""

    async def _redeem() -> None:
        from src.config import settings
        from src.bet_helpers import (
            compute_positions,
            get_clob_client,
            get_ctf_readonly,
            get_trades_history,
            get_usdc_balance,
            is_transient_rpc_error,
            rpc_call_with_retry,
        )

        if not settings.POLYMARKET_PRIVATE_KEY:
            click.echo("Error: POLYMARKET_PRIVATE_KEY not set in .env")
            raise SystemExit(1)

        if not redeem_all and not reconcile_only:
            click.echo("Usage: bet redeem --all  (or --reconcile)")
            click.echo("  --all        Redeem all WON-unredeemed positions (DB-driven).")
            click.echo("  --full-scan  Use full CLOB trade-history scan (slower; combine with --all).")
            click.echo("  --reconcile  Stamp redeemed_at on WON trades whose tokens are already gone (no on-chain action).")
            raise SystemExit(1)

        # --- Web3 setup (lifted before CLOB fetch so --reconcile can run
        # without needing CLOB trade history).
        from web3 import Web3

        NEG_RISK_ADAPTER_ADDRESS = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"
        PUSD_ADDRESS = "0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB"

        NEG_RISK_ADAPTER_ABI = [
            {
                "inputs": [
                    {"name": "conditionId", "type": "bytes32"},
                    {"name": "amounts", "type": "uint256[]"},
                ],
                "name": "redeemPositions",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function",
            },
        ]

        try:
            w3, ctf, address, rpc_url = get_ctf_readonly()
        except Exception as e:
            click.echo(f"Error: {e}")
            raise SystemExit(1)
        click.echo(f"Connected to {rpc_url}")

        class _Conn:
            def __init__(self, w3, ctf, address, rpc_url):
                self.w3 = w3
                self.ctf = ctf
                self.address = address
                self.rpc_url = rpc_url
                self._build_adapter()

            def _build_adapter(self) -> None:
                self.neg_risk_adapter = self.w3.eth.contract(
                    address=Web3.to_checksum_address(NEG_RISK_ADAPTER_ADDRESS),
                    abi=NEG_RISK_ADAPTER_ABI,
                )

            def reconnect(
                self,
                prev_exc: BaseException | None = None,
                attempt: int | None = None,
            ) -> None:
                old_url = self.rpc_url
                new_w3, new_ctf, _, new_url = get_ctf_readonly(skip_url=old_url)
                self.w3 = new_w3
                self.ctf = new_ctf
                self.rpc_url = new_url
                self._build_adapter()
                why = f" (after {prev_exc.__class__.__name__})" if prev_exc else ""
                click.echo(f"  RPC switch: {old_url} → {new_url}{why}")

        conn = _Conn(w3, ctf, address, rpc_url)

        # --- Reconcile helper: stamp redeemed_at on WON trades whose
        # conditional tokens are already gone on-chain. Catches positions
        # redeemed via the Polymarket UI, redeemed before this stamping
        # logic existed, or that have fallen out of the CLOB trade-history
        # window. Looked-up token_ids are also persisted back so subsequent
        # runs avoid re-fetching from Gamma.
        async def _reconcile_won() -> int:
            from sqlalchemy import select as _select
            from src.db.engine import async_session
            from src.db.models import (
                Trade as TradeModel,
                TradeStatus as TradeStatusEnum,
                TradeDirection,
            )
            from src.execution.polymarket_client import get_token_ids

            async with async_session() as session:
                rows = (
                    await session.execute(
                        _select(
                            TradeModel.id,
                            TradeModel.market_id,
                            TradeModel.token_id,
                            TradeModel.direction,
                        ).where(
                            TradeModel.status == TradeStatusEnum.WON,
                            TradeModel.redeemed_at.is_(None),
                        )
                    )
                ).all()

                if not rows:
                    click.echo("Reconcile: no WON-unredeemed trades.")
                    return 0

                click.echo(f"Reconcile: checking {len(rows)} WON-unredeemed trade(s)...")

                # Cache Gamma lookups across trades sharing a market_id.
                token_id_cache: dict[str, tuple[str, str] | None] = {}
                stamped_trade_ids: list[int] = []
                resolved_token_ids: dict[int, str] = {}
                unresolved = 0
                error_count = 0

                for trade_id, market_id, token_id, direction in rows:
                    if not token_id:
                        if market_id not in token_id_cache:
                            try:
                                token_id_cache[market_id] = await get_token_ids(market_id)
                            except Exception as exc:
                                click.echo(f"  trade {trade_id}: token lookup failed ({exc})")
                                token_id_cache[market_id] = None
                        pair = token_id_cache[market_id]
                        if pair is None:
                            unresolved += 1
                            continue
                        yes_tok, no_tok = pair
                        token_id = (
                            yes_tok if direction == TradeDirection.BUY_YES else no_tok
                        )
                        resolved_token_ids[trade_id] = token_id

                    try:
                        asset_id_int = int(token_id)
                    except (TypeError, ValueError):
                        click.echo(f"  trade {trade_id}: invalid token_id {token_id!r}")
                        unresolved += 1
                        continue

                    def _check_balance(aid: int = asset_id_int) -> int:
                        return conn.ctf.functions.balanceOf(conn.address, aid).call()

                    try:
                        bal = rpc_call_with_retry(
                            _check_balance, on_transient=conn.reconnect
                        )
                    except Exception as exc:
                        error_count += 1
                        click.echo(f"  trade {trade_id}: balance check failed ({exc})")
                        continue

                    if bal == 0:
                        stamped_trade_ids.append(trade_id)

                if resolved_token_ids:
                    from sqlalchemy import update as _update

                    for tid, tok in resolved_token_ids.items():
                        await session.execute(
                            _update(TradeModel)
                            .where(TradeModel.id == tid)
                            .values(token_id=tok)
                        )

                if stamped_trade_ids:
                    from sqlalchemy import update as _update

                    await session.execute(
                        _update(TradeModel)
                        .where(TradeModel.id.in_(stamped_trade_ids))
                        .values(redeemed_at=datetime.now(timezone.utc))
                    )

                await session.commit()

                click.echo(
                    f"Reconcile: stamped {len(stamped_trade_ids)} trade(s); "
                    f"unresolved {unresolved}, errors {error_count}"
                )
                return len(stamped_trade_ids)

        if reconcile_only:
            await _reconcile_won()
            return

        # --- DB-driven position discovery: query WON-unredeemed trades.
        # `resolve_trades` sets status=WON at the moment of resolution, so
        # `(status=WON AND redeemed_at IS NULL)` is the redeemability set.
        # Avoids paginating the entire CLOB trade history every call. The
        # legacy full-history scan is still available via --full-scan as a
        # safety net for cases where local DB has drifted from on-chain
        # state.
        async def _build_redeemable_from_db() -> list[dict]:
            import json as _json
            from sqlalchemy import select as _select, update as _update
            from src.db.engine import async_session
            from src.db.models import (
                Market as MarketModel,
                Trade as TradeModel,
                TradeStatus as TradeStatusEnum,
                TradeDirection,
            )
            from src.execution.polymarket_client import get_token_ids
            import httpx as _httpx

            async with async_session() as session:
                rows = (
                    await session.execute(
                        _select(
                            TradeModel.market_id,
                            TradeModel.direction,
                            TradeModel.token_id,
                        ).where(
                            TradeModel.status == TradeStatusEnum.WON,
                            TradeModel.redeemed_at.is_(None),
                        )
                    )
                ).all()

                if not rows:
                    click.echo("No WON-unredeemed trades in DB.")
                    return []

                # Collapse to one entry per (market_id, direction) — multiple
                # trades on the same side share a token, and one
                # redeemPositions call clears them all. Prefer any
                # already-populated token_id.
                groups: dict[tuple[str, TradeDirection], str | None] = {}
                for market_id, direction, token_id in rows:
                    key = (market_id, direction)
                    if key not in groups:
                        groups[key] = token_id
                    elif groups[key] is None and token_id:
                        groups[key] = token_id

                click.echo(
                    f"Found {len(groups)} WON-unredeemed (market, direction) group(s)."
                )

                # Resolve missing token_ids via Gamma, persist back so future
                # runs skip the hop. Same shape as _reconcile_won's lookup.
                token_id_cache: dict[str, tuple[str, str] | None] = {}
                resolved_back: list[tuple[str, TradeDirection, str]] = []
                for (market_id, direction), token_id in list(groups.items()):
                    if token_id:
                        continue
                    if market_id not in token_id_cache:
                        try:
                            token_id_cache[market_id] = await get_token_ids(market_id)
                        except Exception as exc:
                            click.echo(
                                f"  market {market_id}: token lookup failed ({exc})"
                            )
                            token_id_cache[market_id] = None
                    pair = token_id_cache[market_id]
                    if pair is None:
                        continue
                    yes_tok, no_tok = pair
                    final = (
                        yes_tok if direction == TradeDirection.BUY_YES else no_tok
                    )
                    groups[(market_id, direction)] = final
                    resolved_back.append((market_id, direction, final))

                if resolved_back:
                    for mid, direction, tok in resolved_back:
                        await session.execute(
                            _update(TradeModel)
                            .where(
                                TradeModel.market_id == mid,
                                TradeModel.direction == direction,
                                TradeModel.token_id.is_(None),
                            )
                            .values(token_id=tok)
                        )
                    await session.commit()

            # --- Per-market metadata: DB-first, with data-api fallback.
            # Gamma's `GET /markets?id=<market_id>` drops resolved
            # markets and returns [], so we cache `condition_id` /
            # `neg_risk` / `clob_token_ids` on the `markets` row at scan
            # time (see `ingest_markets`). Redemption reads from there.
            # For legacy rows where those columns are NULL (markets
            # resolved before that caching shipped) we hit
            # data-api.polymarket.com/positions — one bulk call keyed
            # by our wallet returns every on-chain position with
            # `asset` (= token_id), `conditionId`, and `negativeRisk`,
            # including resolved-but-unredeemed. CLOB
            # `/markets/<conditionId>` then fills the sibling token_id
            # for `clob_token_ids`. Result persisted back so subsequent
            # runs hit DB only.
            unique_market_ids = {
                m for (m, _), tok in groups.items() if tok
            }
            market_meta: dict[str, dict] = {}

            async with async_session() as session:
                cached_rows = (
                    await session.execute(
                        _select(
                            MarketModel.id,
                            MarketModel.question,
                            MarketModel.condition_id,
                            MarketModel.neg_risk,
                            MarketModel.clob_token_ids,
                        ).where(MarketModel.id.in_(unique_market_ids))
                    )
                ).all()

                missing_meta_ids: list[str] = []
                cached_by_id: dict[str, dict] = {}
                for mid, question, cond_id, neg_risk_db, cached_clob_ids in cached_rows:
                    cached_by_id[mid] = {
                        "question": question or "",
                        "condition_id": cond_id or "",
                        "neg_risk": neg_risk_db,
                        "clob_token_ids": cached_clob_ids or [],
                    }

                for mid in unique_market_ids:
                    cached = cached_by_id.get(mid)
                    if (
                        cached
                        and cached["condition_id"]
                        and cached["clob_token_ids"]
                    ):
                        tokens = [
                            {"token_id": str(t)}
                            for t in cached["clob_token_ids"]
                            if t
                        ]
                        market_meta[mid] = {
                            "question": cached["question"],
                            "condition_id": cached["condition_id"],
                            "closed": True,  # status=WON locally implies resolved
                            "neg_risk": bool(cached["neg_risk"]),
                            "tokens": tokens,
                        }
                    else:
                        missing_meta_ids.append(mid)

                refreshed: list[tuple[str, str, bool, list[str]]] = []
                if missing_meta_ids:
                    tok_for_mid: dict[str, str] = {}
                    for (g_mid, _g_direction), g_tok in groups.items():
                        if (
                            g_tok
                            and g_mid in missing_meta_ids
                            and g_mid not in tok_for_mid
                        ):
                            tok_for_mid[g_mid] = g_tok

                    # Derive the funder address that holds the
                    # positions (proxy/safe if set, else EOA). Matches
                    # the pattern in src/bet_helpers.py:202.
                    funder_addr: str | None = None
                    if settings.POLYMARKET_PRIVATE_KEY:
                        try:
                            from eth_account import Account as _Account

                            _acct = _Account.from_key(
                                settings.POLYMARKET_PRIVATE_KEY
                            )
                            funder_addr = (
                                settings.POLYMARKET_FUNDER_ADDRESS
                                or _acct.address
                            )
                        except Exception as exc:
                            click.echo(
                                f"  warning: could not derive funder "
                                f"address ({exc})"
                            )

                    # token_id -> {conditionId, negativeRisk}
                    asset_to_meta: dict[str, dict] = {}
                    # conditionId -> list of token_ids we own
                    cond_to_assets: dict[str, list[str]] = {}

                    async with _httpx.AsyncClient(timeout=20) as http:
                        if funder_addr:
                            try:
                                resp = await http.get(
                                    "https://data-api.polymarket.com/positions",
                                    params={
                                        "user": funder_addr,
                                        "sizeThreshold": "0",
                                        "limit": "500",
                                    },
                                )
                                resp.raise_for_status()
                                positions_payload = resp.json()
                                if isinstance(positions_payload, list):
                                    for p in positions_payload:
                                        asset = str(p.get("asset", ""))
                                        cond = str(p.get("conditionId", ""))
                                        if not asset or not cond:
                                            continue
                                        asset_to_meta[asset] = {
                                            "conditionId": cond,
                                            "negativeRisk": bool(
                                                p.get("negativeRisk", False)
                                            ),
                                            "question": p.get("title", "")
                                            or p.get("eventTitle", ""),
                                        }
                                        cond_to_assets.setdefault(
                                            cond, []
                                        ).append(asset)
                            except Exception as exc:
                                click.echo(
                                    f"  warning: data-api positions lookup "
                                    f"failed ({exc})"
                                )

                        # Sibling-token cache keyed by conditionId.
                        sibling_cache: dict[str, list[str]] = {}

                        async def _full_token_pair(
                            cond_id: str, known_tok: str
                        ) -> list[str]:
                            """Return [yes_tok, no_tok] for cond_id.

                            Prefer the data-api response (free when we
                            own both sides); fall back to one CLOB
                            call. Order in the returned list mirrors
                            what _market_to_row would persist — we
                            don't try to assert YES-first because the
                            actual redeemPositions call uses
                            condition_id + indexSet, not token order.
                            """
                            if cond_id in sibling_cache:
                                return sibling_cache[cond_id]
                            owned = cond_to_assets.get(cond_id, [])
                            owned = [str(t) for t in owned if t]
                            if len(owned) >= 2:
                                # Dedupe preserving order.
                                seen: set[str] = set()
                                pair: list[str] = []
                                for t in owned:
                                    if t in seen:
                                        continue
                                    seen.add(t)
                                    pair.append(t)
                                    if len(pair) == 2:
                                        break
                                if len(pair) == 2:
                                    sibling_cache[cond_id] = pair
                                    return pair
                            try:
                                r = await http.get(
                                    f"https://clob.polymarket.com/markets/{cond_id}"
                                )
                                r.raise_for_status()
                                raw = r.json()
                                toks = raw.get("tokens") or []
                                tok_ids = [
                                    str(t.get("token_id", ""))
                                    for t in toks
                                    if t.get("token_id")
                                ]
                                if len(tok_ids) >= 2:
                                    sibling_cache[cond_id] = tok_ids[:2]
                                    return tok_ids[:2]
                            except Exception as exc:
                                click.echo(
                                    f"  warning: CLOB sibling lookup for "
                                    f"{cond_id} failed ({exc})"
                                )
                            sibling_cache[cond_id] = [known_tok]
                            return [known_tok]

                        for mid in missing_meta_ids:
                            fallback_tok = tok_for_mid.get(mid)
                            if not fallback_tok:
                                click.echo(
                                    f"  market {mid}: redemption metadata "
                                    f"unavailable (no token_id to fall back on)"
                                )
                                continue
                            meta = asset_to_meta.get(fallback_tok)
                            if not meta:
                                click.echo(
                                    f"  market {mid}: redemption metadata "
                                    f"unavailable (no position found at "
                                    f"data-api for token {fallback_tok})"
                                )
                                continue
                            cond_id = meta["conditionId"]
                            neg_risk = bool(meta["negativeRisk"])
                            pair = await _full_token_pair(
                                cond_id, fallback_tok
                            )
                            if len(pair) < 2:
                                click.echo(
                                    f"  market {mid}: redemption metadata "
                                    f"unavailable (could not resolve sibling "
                                    f"token for condition {cond_id})"
                                )
                                continue
                            market_meta[mid] = {
                                "question": meta.get("question", "")
                                or (cached_by_id.get(mid) or {}).get(
                                    "question", ""
                                ),
                                "condition_id": cond_id,
                                "closed": True,
                                "neg_risk": neg_risk,
                                "tokens": [
                                    {"token_id": t} for t in pair
                                ],
                            }
                            refreshed.append(
                                (mid, cond_id, neg_risk, pair)
                            )

                if refreshed:
                    for mid, cond_id, neg_risk, clob_ids in refreshed:
                        await session.execute(
                            _update(MarketModel)
                            .where(MarketModel.id == mid)
                            .values(
                                condition_id=cond_id,
                                neg_risk=neg_risk,
                                clob_token_ids=clob_ids,
                            )
                        )
                    await session.commit()

            redeemable: list[dict] = []
            for (market_id, direction), token_id in groups.items():
                if not token_id:
                    continue
                mkt = market_meta.get(market_id)
                if not mkt:
                    continue
                condition_id = (
                    mkt.get("condition_id", "")
                    or mkt.get("conditionId", "")
                    or market_id
                )
                if not condition_id:
                    continue
                # status=WON already implies resolved; treat closed=false
                # as a soft signal but still allow the redemption attempt
                # — the on-chain balanceOf check is authoritative.
                closed = (
                    mkt.get("closed") is True
                    or str(mkt.get("closed", "")).lower() == "true"
                )
                if not closed:
                    click.echo(
                        f"  market {condition_id}: CLOB reports not closed; "
                        f"proceeding anyway (status=WON locally)"
                    )
                neg_risk = (
                    mkt.get("neg_risk") is True
                    or str(mkt.get("neg_risk", "")).lower() == "true"
                )
                tokens = mkt.get("tokens", [])
                clob_ids: list[str] = [
                    t.get("token_id", "")
                    for t in tokens
                    if t.get("token_id")
                ]
                token_side = (
                    "YES" if direction == TradeDirection.BUY_YES else "NO"
                )
                redeemable.append(
                    {
                        "asset_id": token_id,
                        "question": mkt.get("question", ""),
                        "condition_id": condition_id,
                        "token_side": token_side,
                        "neg_risk": neg_risk,
                        "clob_ids": clob_ids,
                        "outcome_prices": [],
                        "size": 0,
                    }
                )
            return redeemable

        if full_scan:
            # --- Legacy: fetch positions from CLOB trade history ---
            click.echo("\nConnecting to Polymarket CLOB (full-scan)...")
            client = get_clob_client()

            click.echo("Fetching trade history...")
            trades = get_trades_history(client)
            if not trades:
                click.echo("No trades found.")
                await _reconcile_won()
                return

            positions = compute_positions(trades)
            if not positions:
                click.echo("No open positions found.")
                await _reconcile_won()
                return

            click.echo(
                f"Found {len(positions)} position(s). Checking market resolution..."
            )

            import httpx

            redeemable: list[dict] = []

            async with httpx.AsyncClient(timeout=15) as http:
                cond_to_assets: dict[str, list[tuple[str, dict]]] = {}
                for asset_id, pos in positions.items():
                    cond = pos.get("market", "")
                    if not cond:
                        continue
                    cond_to_assets.setdefault(cond, []).append((asset_id, pos))

                for cond_id, assets in cond_to_assets.items():
                    mkt = None
                    try:
                        resp = await http.get(
                            f"https://clob.polymarket.com/markets/{cond_id}",
                        )
                        resp.raise_for_status()
                        mkt = resp.json()
                    except Exception:
                        pass

                    if not mkt:
                        continue

                    closed = mkt.get("closed") is True or str(mkt.get("closed", "")).lower() == "true"
                    if not closed:
                        continue

                    condition_id = mkt.get("condition_id", "") or mkt.get("conditionId", "")
                    if not condition_id:
                        continue

                    neg_risk = mkt.get("neg_risk") is True or str(mkt.get("neg_risk", "")).lower() == "true"

                    tokens = mkt.get("tokens", [])
                    token_map: dict[str, str] = {}
                    clob_ids: list[str] = []
                    for tok in tokens:
                        tid = tok.get("token_id", "")
                        outcome = tok.get("outcome", "")
                        if tid:
                            token_map[tid] = outcome.upper()
                            clob_ids.append(tid)

                    for asset_id, pos in assets:
                        token_side = token_map.get(asset_id, "")
                        redeemable.append({
                            "asset_id": asset_id,
                            "question": mkt.get("question", ""),
                            "condition_id": condition_id,
                            "token_side": token_side,
                            "neg_risk": neg_risk,
                            "clob_ids": clob_ids,
                            "outcome_prices": [],
                            "size": pos["size"],
                        })
        else:
            redeemable = await _build_redeemable_from_db()

        if not redeemable:
            click.echo("No resolved positions found to redeem.")
            await _reconcile_won()
            return

        bal = rpc_call_with_retry(
            lambda: conn.w3.eth.get_balance(conn.address),
            on_transient=conn.reconnect,
        )
        click.echo(f"POL balance: {conn.w3.from_wei(bal, 'ether'):.4f} (for gas)")
        if bal == 0:
            click.echo("Error: wallet has no POL for gas fees")
            raise SystemExit(1)

        # --- Check on-chain balances ---
        click.echo("\nChecking on-chain token balances...")
        to_redeem: list[dict] = []

        for item in redeemable:
            asset_id_int = int(item["asset_id"])

            def _check_balance(asset_id: int = asset_id_int) -> int:
                return conn.ctf.functions.balanceOf(conn.address, asset_id).call()

            try:
                token_balance = rpc_call_with_retry(
                    _check_balance, on_transient=conn.reconnect
                )
            except Exception as e:
                click.echo(f"  Could not check balance for {item['question'][:50]}: {e}")
                continue

            if token_balance == 0:
                click.echo(f"  {item['question'][:50]}: already redeemed (balance=0)")
                continue

            # Conditional tokens use 1e6 decimals (same as USDC)
            balance_usdc = token_balance / 1e6
            item["on_chain_balance"] = token_balance
            item["balance_usdc"] = balance_usdc
            to_redeem.append(item)

        if not to_redeem:
            click.echo("\nNo positions with non-zero on-chain balance to redeem.")
            await _reconcile_won()
            return

        # --- Display redeemable positions ---
        click.echo(f"\n=== Redeemable Positions ({len(to_redeem)}) ===")
        for item in to_redeem:
            question = item["question"] or item["asset_id"][:20]
            side = item["token_side"] or "?"
            neg = " [neg-risk]" if item["neg_risk"] else ""
            click.echo(f"\n  {question}")
            click.echo(f"    Side:        {side}")
            click.echo(f"    On-chain:    {item['balance_usdc']:.2f} shares")
            click.echo(f"    Neg risk:    {'Yes' if item['neg_risk'] else 'No'}")

        # --- Confirmation ---
        if not skip_confirm:
            if not click.confirm("\nProceed with on-chain redemption?"):
                click.echo("Aborted.")
                return

        # --- Execute redemptions ---
        click.echo("\n=== Executing Redemptions ===")
        CHAIN_ID = 137
        PARENT_COLLECTION_ID = b"\x00" * 32

        def _wait_for_receipt(tx_hash, total_timeout: float = 180.0):
            """Poll for the receipt across RPC failovers.

            web3's `wait_for_transaction_receipt` keeps re-polling the
            same provider, so a flaky endpoint stalls the whole call.
            We poll manually and rotate to a fresh RPC on every transient
            error — the tx_hash is portable across endpoints.
            """
            deadline = time.time() + total_timeout
            poll_interval = 2.0
            while time.time() < deadline:
                try:
                    return rpc_call_with_retry(
                        lambda: conn.w3.eth.get_transaction_receipt(tx_hash),
                        on_transient=conn.reconnect,
                        max_attempts=2,
                    )
                except Exception as exc:
                    msg = str(exc).lower()
                    # web3 raises TransactionNotFound while the tx is in
                    # the mempool — that's expected, just keep polling.
                    if "not found" in msg or "transactionnotfound" in msg:
                        time.sleep(poll_interval)
                        poll_interval = min(poll_interval * 1.3, 8.0)
                        continue
                    if is_transient_rpc_error(exc):
                        try:
                            conn.reconnect(exc)
                        except Exception:
                            pass
                        time.sleep(poll_interval)
                        continue
                    raise
            raise TimeoutError(
                f"receipt for {tx_hash.hex() if hasattr(tx_hash, 'hex') else tx_hash} "
                f"did not appear within {total_timeout:.0f}s"
            )

        def send_tx(tx_data: dict) -> dict:
            """Sign, broadcast, and wait for receipt with RPC failover.

            Refreshes nonce + gasPrice per attempt against `pending` so a
            previous tx that broadcast-but-receipt-timed-out doesn't
            strand this one with "nonce too low". Forces a legacy
            (type-0) transaction by stripping any auto-filled EIP-1559
            fields — Polygon supports 1559 and `build_transaction` will
            populate `maxFeePerGas`/`maxPriorityFeePerGas` when no
            `gasPrice` is supplied, which then conflicts with our
            `gasPrice` field at sign time (`Unknown kwargs: ['gasPrice']`).

            Captures `signed.hash` BEFORE broadcasting so a
            transient "already known" (the same tx reached the mempool
            on a prior attempt via a different RPC) falls through to
            the receipt poll instead of being treated as a fresh failure.
            """
            tx_data["chainId"] = CHAIN_ID
            tx_data["from"] = conn.address
            tx_data.pop("maxFeePerGas", None)
            tx_data.pop("maxPriorityFeePerGas", None)
            tx_data.pop("type", None)

            last_exc: Exception | None = None
            for attempt in range(1, 4):
                try:
                    tx_data["nonce"] = conn.w3.eth.get_transaction_count(
                        conn.address, "pending"
                    )
                    tx_data["gasPrice"] = conn.w3.eth.gas_price
                    if "gas" not in tx_data:
                        tx_data["gas"] = conn.w3.eth.estimate_gas(tx_data)
                    signed = conn.w3.eth.account.sign_transaction(
                        tx_data, private_key=settings.POLYMARKET_PRIVATE_KEY
                    )
                    tx_hash = signed.hash
                    try:
                        conn.w3.eth.send_raw_transaction(signed.raw_transaction)
                    except Exception as bcast_exc:
                        bcast_msg = str(bcast_exc).lower()
                        if "already known" in bcast_msg or "already exists" in bcast_msg:
                            pass  # tx is in mempool from a prior attempt
                        else:
                            raise
                    return _wait_for_receipt(tx_hash)
                except Exception as exc:
                    last_exc = exc
                    msg = str(exc).lower()
                    if "nonce too low" in msg:
                        # An earlier-loop tx with the same nonce already
                        # landed; let the caller report SKIPPED.
                        raise
                    if is_transient_rpc_error(exc) and attempt < 3:
                        conn.reconnect(exc)
                        time.sleep(1.0 * attempt)
                        continue
                    raise

            raise last_exc or RuntimeError("send_tx exhausted retries")

        success_count = 0
        redeemed_condition_ids: set[str] = set()
        for item in to_redeem:
            question = (item["question"] or item["asset_id"][:20])[:50]
            condition_id_hex = item["condition_id"]
            if condition_id_hex.startswith("0x"):
                condition_id_bytes = bytes.fromhex(condition_id_hex[2:])
            else:
                condition_id_bytes = bytes.fromhex(condition_id_hex)

            # Determine indexSets based on token side
            clob_ids = item["clob_ids"]
            if len(clob_ids) >= 2 and item["asset_id"] == clob_ids[0]:
                index_sets = [1]  # YES = outcome index 0 → 2^0 = 1
            elif len(clob_ids) >= 2 and item["asset_id"] == clob_ids[1]:
                index_sets = [2]  # NO = outcome index 1 → 2^1 = 2
            else:
                index_sets = [1, 2]  # Try both

            click.echo(f"\n  Redeeming: {question}...")
            try:
                if item["neg_risk"]:
                    on_chain = item["on_chain_balance"]
                    clob_ids = item["clob_ids"]
                    if len(clob_ids) >= 2 and item["asset_id"] == clob_ids[0]:
                        amounts = [on_chain, 0]   # holding YES position
                    elif len(clob_ids) >= 2 and item["asset_id"] == clob_ids[1]:
                        amounts = [0, on_chain]   # holding NO position
                    else:
                        amounts = [on_chain, 0]   # fallback

                    def _build_tx() -> dict:
                        # gasPrice forces a legacy (type-0) tx; without
                        # it, web3 v7 auto-fills EIP-1559 fields and
                        # signing then rejects gasPrice as "unknown
                        # kwargs". Retried via rpc_call_with_retry so an
                        # estimate_gas timeout fails over to a fresh RPC
                        # rather than getting reported as a permanent
                        # "Market may not be resolved" error.
                        return conn.neg_risk_adapter.functions.redeemPositions(
                            condition_id_bytes,
                            amounts,
                        ).build_transaction({
                            "from": conn.address,
                            "gasPrice": conn.w3.eth.gas_price,
                        })
                else:
                    pusd_addr = Web3.to_checksum_address(PUSD_ADDRESS)

                    def _build_tx() -> dict:  # noqa: F811
                        return conn.ctf.functions.redeemPositions(
                            pusd_addr,
                            PARENT_COLLECTION_ID,
                            condition_id_bytes,
                            index_sets,
                        ).build_transaction({
                            "from": conn.address,
                            "gasPrice": conn.w3.eth.gas_price,
                        })

                tx = rpc_call_with_retry(_build_tx, on_transient=conn.reconnect)
                receipt = send_tx(tx)
                status = "OK" if receipt["status"] == 1 else "FAILED"
                tx_hash = receipt["transactionHash"].hex()[:16]
                gas_used = receipt["gasUsed"]
                click.echo(f"    Status: {status}  tx: {tx_hash}...  gas: {gas_used}")
                if receipt["status"] == 1:
                    success_count += 1
                    redeemed_condition_ids.add(item["condition_id"])
            except Exception as e:
                msg = str(e).lower()
                if "nonce too low" in msg or "already known" in msg:
                    click.echo(f"    SKIPPED: prior tx already broadcast ({e})")
                    click.echo("    (Re-run `bet redeem` to refresh on-chain balances)")
                else:
                    click.echo(f"    FAILED: {e}")
                    click.echo("    (Market may not be resolved on-chain yet)")

        # --- Stamp Trade.redeemed_at so bankroll stops counting these as
        # unredeemed pending payouts. Market.id == condition_id in our DB
        # (set during ingest in src/ingestion/polymarket.py).
        if redeemed_condition_ids:
            from sqlalchemy import update
            from src.db.engine import async_session
            from src.db.models import Trade as TradeModel, TradeStatus as TradeStatusEnum

            async with async_session() as session:
                stamped = await session.execute(
                    update(TradeModel)
                    .where(
                        TradeModel.market_id.in_(redeemed_condition_ids),
                        TradeModel.status == TradeStatusEnum.WON,
                        TradeModel.redeemed_at.is_(None),
                    )
                    .values(redeemed_at=datetime.now(timezone.utc))
                )
                await session.commit()
                if stamped.rowcount:
                    click.echo(f"  Stamped redeemed_at on {stamped.rowcount} local trade row(s)")

        # --- Reconcile any straggler WON-unredeemed trades whose tokens
        # are already gone (redeemed via UI, redeemed before this stamping
        # logic existed, or out of CLOB trade-history window).
        await _reconcile_won()

        # --- Final balance ---
        click.echo(f"\n=== Results ===")
        click.echo(f"  Redeemed: {success_count}/{len(to_redeem)} positions")
        pusd, usdc_e, usdc_native = await get_usdc_balance(settings.POLYMARKET_PRIVATE_KEY)
        click.echo(f"  pUSD balance:   ${pusd:.2f}")
        click.echo(f"  USDC.e balance: ${usdc_e:.2f}")
        if usdc_native > 0:
            click.echo(f"  Native USDC:    ${usdc_native:.2f}")

    asyncio.run(_redeem())


@main.group()
def admin() -> None:
    """One-shot maintenance commands."""


@admin.command("reset-drawdown-peak")
@click.option("--yes", "-y", "skip_confirm", is_flag=True, help="Skip confirmation prompt.")
@click.option("--dry-run", is_flag=True, help="Show what would change without writing.")
def reset_drawdown_peak(skip_confirm: bool, dry_run: bool) -> None:
    """Reset DrawdownMonitor's peak to current equity.

    Inserts a fresh BankrollLog row with peak = balance = get_current_bankroll().
    The running scheduler reloads DrawdownMonitor's peak from this row on a TTL
    (~5 min), so the change takes effect without a restart; restart only if you
    need it applied immediately.

    Use this after a bankroll-equation correction that left an inflated
    `peak` in the latest BankrollLog row, which pins drawdown_pct >
    PAUSE_THRESHOLD and silently zeros every Kelly stake.
    """

    async def _reset() -> None:
        from datetime import datetime, timezone

        from sqlalchemy import select

        from src.db.engine import async_session
        from src.db.models import BankrollLog
        from src.resolution import get_current_bankroll

        async with async_session() as session:
            prior = (
                await session.execute(
                    select(BankrollLog)
                    .order_by(BankrollLog.timestamp.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            current = await get_current_bankroll(session)

            click.echo("Current state:")
            if prior is not None:
                click.echo(f"  Prior row #{prior.id} @ {prior.timestamp.isoformat()}")
                click.echo(f"    balance={prior.balance:.2f}  peak={prior.peak:.2f}  dd_pct={prior.drawdown_pct:.4f}")
            else:
                click.echo("  No prior BankrollLog row (table empty).")
            click.echo(f"  Computed current bankroll: ${current:.2f}")
            click.echo()
            click.echo("Will insert:")
            click.echo(f"  BankrollLog(balance={current:.2f}, peak={current:.2f}, drawdown_pct=0.0)")
            click.echo()

            if dry_run:
                click.echo("--dry-run: no row written.")
                return

            if not skip_confirm:
                click.confirm("Proceed?", abort=True)

            row = BankrollLog(
                balance=current,
                peak=current,
                drawdown_pct=0.0,
                timestamp=datetime.now(timezone.utc),
            )
            session.add(row)
            await session.commit()
            click.echo(f"Inserted BankrollLog row #{row.id}.")
            click.echo()
            click.echo("✓  The running scheduler reloads the peak on a ~5 min TTL,")
            click.echo("   so this takes effect automatically. Restart only if you")
            click.echo("   need the new peak applied immediately.")

    asyncio.run(_reset())


@admin.command("reconcile-stuck")
@click.option("--yes", "-y", "skip_confirm", is_flag=True, help="Skip confirmation prompt.")
@click.option("--dry-run", is_flag=True, help="Show what would change without writing.")
@click.option(
    "--grace-hours",
    default=4,
    show_default=True,
    type=int,
    help="Only touch trades whose market ended at least this many hours ago.",
)
@click.option(
    "--include-lost",
    is_flag=True,
    default=False,
    help=(
        "Also process LOST trades whose fill_price is NULL — recovers "
        "the 2026-05-20 phantom-LOST class (matched orders prematurely "
        "marked LOST by resolve_trades' null-fill grace fallback). For "
        "each such trade we balanceOf the on-chain token: bal>0 reverts "
        "to OPEN + backfill, bal==0 stays LOST with pnl=0."
    ),
)
def reconcile_stuck(
    skip_confirm: bool,
    dry_run: bool,
    grace_hours: int,
    include_lost: bool,
) -> None:
    """Resolve OPEN trades with no fill_price whose markets have ended.

    Symptom: ``size_position`` returns $0 because ``get_current_exposure``
    sums OPEN ``stake_usd`` and the bot is already over the
    ``MAX_EXPOSURE_PCT`` cap. Root cause: delayed or matched CLOB orders
    whose fill data never landed in our DB — promoted to ``status=OPEN``
    despite ``fill_price IS NULL``. ``job_resolve_trades`` then can't
    settle them because the CLOB drops resolved markets from its
    listings and ``_refresh_market_price`` returns None.

    Scope matches ``job_reconcile_orders`` (exchange_status in
    {delayed, matched, matching}) so the manual and automated reconcile
    paths share the same row-set. Pre-2026-05-18 this filter was
    ``=="delayed"`` only, which silently skipped 24 of 25 stuck rows
    when the matching-engine acked with ``status="matched"`` but
    ``client.get_order`` never returned readable fill data.

    For each affected trade, query on-chain ``balanceOf`` for the
    market's conditional token:
      * balance == 0 → trade never landed; mark ``LOST`` with
        ``pnl = 0`` (post-2026-05-20: no money moved, so no phantom
        loss). Releases exposure budget.
      * balance > 0  → trade DID land; backfill ``fill_price`` from
        ``entry_price`` and ``filled_size`` from ``stake_usd/entry_price``
        and leave OPEN so ``job_resolve_trades`` settles it next tick.

    Use ``--dry-run`` first to confirm on-chain balanceOf +
    payoutNumerators per row before letting the live pass write
    WON/LOST.

    Requires ``POLYMARKET_PRIVATE_KEY`` set (read-only chain access; no
    signed transactions).

    Pass ``--include-lost`` to also process LOST trades with NULL
    fill_price — recovers the 2026-05-20 phantom-LOST class where
    ``resolve_trades`` prematurely marked OPEN trades LOST via the
    null-fill grace fallback. For each balanced > 0 row, status
    reverts to OPEN, ``pnl``/``exit_price``/``closed_at`` clear, and
    ``fill_price``/``filled_size`` backfill — letting normal resolve
    flow re-settle from chain when UMA reports. For bal==0 rows, the
    LOST stays but ``pnl`` resets to 0 (true accounting cleanup, was
    -stake pre-fix).
    """

    async def _reconcile_stuck() -> None:
        from datetime import datetime, timedelta, timezone

        from sqlalchemy import select, update
        from sqlalchemy.orm import joinedload

        from src.config import settings
        from src.db.engine import async_session
        from src.db.models import Market, Trade, TradeDirection, TradeStatus
        from src.bet_helpers import (
            get_ctf_readonly,
            get_payout_outcome,
            rpc_call_with_retry,
        )
        from src.execution.polymarket_client import get_token_ids

        if not settings.POLYMARKET_PRIVATE_KEY:
            click.echo("Error: POLYMARKET_PRIVATE_KEY not set in .env")
            raise SystemExit(1)

        cutoff = datetime.now(timezone.utc) - timedelta(hours=grace_hours)

        async with async_session() as session:
            status_filter = (
                [TradeStatus.OPEN, TradeStatus.LOST]
                if include_lost
                else [TradeStatus.OPEN]
            )
            rows = (
                await session.execute(
                    select(Trade)
                    .options(joinedload(Trade.market))
                    .join(Market, Trade.market_id == Market.id)
                    .where(
                        Trade.status.in_(status_filter),
                        Trade.fill_price.is_(None),
                        Trade.exchange_status.in_(
                            ("delayed", "matched", "matching")
                        ),
                        Market.end_date < cutoff,
                    )
                    .order_by(Trade.opened_at)
                )
            ).scalars().unique().all()

            if not rows:
                scope = (
                    "OPEN/LOST" if include_lost else "OPEN"
                )
                click.echo(
                    f"No stuck trades ({scope} + delayed/matched/matching + "
                    f"null fill_price) with markets ended > {grace_hours}h "
                    f"ago. Nothing to do."
                )
                return

            click.echo(
                f"Found {len(rows)} stuck trade(s) totalling "
                f"${sum(t.stake_usd for t in rows):.2f} in tied-up exposure.\n"
            )

            # Lazy Web3 setup so --dry-run with zero matches doesn't pay
            # the connection cost. (Connection is needed for any non-dry
            # path because we need the on-chain balance.)
            try:
                w3, ctf, address, rpc_url = get_ctf_readonly()
            except Exception as exc:
                click.echo(f"Error connecting to chain: {exc}")
                raise SystemExit(1)
            click.echo(f"Connected to {rpc_url} as {address}\n")

            def _check_balance(token_id_int: int) -> int:
                return ctf.functions.balanceOf(address, token_id_int).call()

            # Cache Gamma lookups across trades sharing market_id.
            # Lookup of YES/NO token pair only happens for trades whose
            # token_id is NULL (legacy rows from before token_id capture).
            token_id_cache: dict[str, tuple[str, str] | None] = {}

            # Backfill missing market.condition_id from data-api positions.
            # Gamma drops resolved markets from /markets?id=… so direct
            # lookup returns []; data-api.polymarket.com/positions is the
            # only feed that still carries conditionId for closed positions.
            # Same fallback path the redeem flow uses for legacy NULL rows.
            funder_addr = (
                settings.POLYMARKET_FUNDER_ADDRESS or address
            )
            asset_to_cond: dict[str, str] = {}
            missing_cond_market_ids = {
                t.market_id for t in rows
                if t.market and not t.market.condition_id
            }
            if missing_cond_market_ids:
                import httpx as _httpx
                try:
                    async with _httpx.AsyncClient(timeout=20) as http:
                        resp = await http.get(
                            "https://data-api.polymarket.com/positions",
                            params={
                                "user": funder_addr,
                                "sizeThreshold": "0",
                                "limit": "500",
                            },
                        )
                        resp.raise_for_status()
                        payload = resp.json()
                    if isinstance(payload, list):
                        for p in payload:
                            asset = str(p.get("asset", ""))
                            cond = str(p.get("conditionId", ""))
                            if asset and cond:
                                asset_to_cond[asset] = cond
                    click.echo(
                        f"data-api positions: fetched {len(asset_to_cond)} "
                        f"asset→conditionId mappings\n"
                    )
                except Exception as exc:
                    click.echo(
                        f"  warning: data-api positions lookup failed ({exc}); "
                        f"payout settlement will fall back to backfill-only"
                    )

            decisions: list[dict] = []

            for trade in rows:
                token_id = trade.token_id
                if not token_id:
                    if trade.market_id not in token_id_cache:
                        try:
                            token_id_cache[trade.market_id] = await get_token_ids(
                                trade.market_id
                            )
                        except Exception as exc:
                            click.echo(
                                f"  trade {trade.id}: token lookup failed ({exc})"
                            )
                            token_id_cache[trade.market_id] = None
                    pair = token_id_cache[trade.market_id]
                    if pair is None:
                        decisions.append({
                            "trade": trade, "action": "skip-no-token",
                            "balance": None,
                        })
                        continue
                    yes_tok, no_tok = pair
                    token_id = (
                        yes_tok if trade.direction == TradeDirection.BUY_YES else no_tok
                    )

                try:
                    asset_id_int = int(token_id)
                except (TypeError, ValueError):
                    click.echo(f"  trade {trade.id}: invalid token_id {token_id!r}")
                    decisions.append({
                        "trade": trade, "action": "skip-bad-token", "balance": None,
                    })
                    continue

                try:
                    bal = rpc_call_with_retry(
                        lambda aid=asset_id_int: _check_balance(aid)
                    )
                except Exception as exc:
                    click.echo(f"  trade {trade.id}: balance check failed ({exc})")
                    decisions.append({
                        "trade": trade, "action": "skip-rpc-error", "balance": None,
                    })
                    continue

                # bal == 0  → order never filled on-chain; mark LOST.
                # bal > 0   → order DID fill. Check on-chain payout
                #             numerators: if reported, settle WON/LOST
                #             inline (the CLOB dropped the market so
                #             resolve_trades would just skip it forever);
                #             otherwise backfill fill_price and leave OPEN.
                if bal == 0:
                    action = "mark-lost"
                    settled_yes_won: bool | None = None
                    cond_id_used = ""
                else:
                    cond_id_used = (
                        (trade.market.condition_id or "") if trade.market else ""
                    )
                    if not cond_id_used:
                        # Legacy market (NULL condition_id). Fallback via
                        # data-api positions feed populated above.
                        cond_id_used = asset_to_cond.get(token_id, "")
                        if cond_id_used and trade.market is not None:
                            trade.market.condition_id = cond_id_used
                    try:
                        settled_yes_won = rpc_call_with_retry(
                            lambda c=cond_id_used: get_payout_outcome(ctf, c)
                        ) if cond_id_used else None
                    except Exception as exc:
                        click.echo(
                            f"  trade {trade.id}: payout check failed ({exc})"
                        )
                        settled_yes_won = None
                    if settled_yes_won is None:
                        action = "backfill-fill"
                    else:
                        trade_won = (
                            (trade.direction == TradeDirection.BUY_YES and settled_yes_won)
                            or (trade.direction == TradeDirection.BUY_NO and not settled_yes_won)
                        )
                        action = "settle-won" if trade_won else "settle-lost"

                decisions.append({
                    "trade": trade,
                    "action": action,
                    "balance": bal,
                    "token_id": token_id,
                    "condition_id": cond_id_used,
                    "yes_won": settled_yes_won,
                })

            # Summary table
            click.echo("Decisions:")
            click.echo(
                f"  {'id':>6} {'opened_at':<25} {'stake':>8} {'action':<18}  market"
            )
            for d in decisions:
                t = d["trade"]
                bal_str = "—" if d["balance"] is None else f"bal={d['balance']}"
                q = (t.market.question[:50] + "…") if t.market and len(t.market.question or "") > 50 else (t.market.question if t.market else "?")
                click.echo(
                    f"  {t.id:>6} {t.opened_at.isoformat():<25} "
                    f"${t.stake_usd:>7.2f} {d['action']:<18}  {bal_str}  {q}"
                )

            to_mark_lost = [d for d in decisions if d["action"] == "mark-lost"]
            to_settle_won = [d for d in decisions if d["action"] == "settle-won"]
            to_settle_lost = [d for d in decisions if d["action"] == "settle-lost"]
            to_backfill = [d for d in decisions if d["action"] == "backfill-fill"]
            released = sum(
                d["trade"].stake_usd
                for d in (to_mark_lost + to_settle_won + to_settle_lost)
            )
            click.echo()
            click.echo(
                f"Summary: {len(to_mark_lost)} mark-LOST (never filled), "
                f"{len(to_settle_won)} settle-WON, "
                f"{len(to_settle_lost)} settle-LOST (filled, on-chain payout), "
                f"{len(to_backfill)} backfill-only (not yet reported on-chain), "
                f"{len(decisions) - len(to_mark_lost) - len(to_settle_won) - len(to_settle_lost) - len(to_backfill)} skipped. "
                f"Will release ${released:.2f} of exposure."
            )

            if dry_run:
                click.echo("\n--dry-run: no changes written.")
                return

            if not skip_confirm and (
                to_mark_lost or to_settle_won or to_settle_lost or to_backfill
            ):
                click.confirm("Apply these changes?", abort=True)

            now = datetime.now(timezone.utc)
            for d in to_mark_lost:
                t = d["trade"]
                t.status = TradeStatus.LOST
                # 2026-05-20: bal==0 means order never landed; no wallet
                # debit, so pnl=0. Pre-fix this was -stake, which
                # invented phantom losses when balanceOf was never
                # checked.
                t.pnl = 0.0
                t.exit_price = 0.0
                t.closed_at = now
            for d in to_settle_won:
                t = d["trade"]
                # Backfill fill_price first so PnL math has a clean basis.
                t.fill_price = float(t.entry_price or 0.0)
                t.filled_size = (
                    float(t.stake_usd) / float(t.entry_price)
                    if t.entry_price and t.entry_price > 0 else 0.0
                )
                t.status = TradeStatus.WON
                entry = float(t.entry_price or 0.0)
                t.pnl = (
                    float(t.stake_usd) * (1.0 / entry - 1.0) if entry > 0 else 0.0
                )
                t.exit_price = 1.0
                t.closed_at = now
            for d in to_settle_lost:
                t = d["trade"]
                t.fill_price = float(t.entry_price or 0.0)
                t.filled_size = (
                    float(t.stake_usd) / float(t.entry_price)
                    if t.entry_price and t.entry_price > 0 else 0.0
                )
                t.status = TradeStatus.LOST
                t.pnl = -float(t.stake_usd)
                t.exit_price = 0.0
                t.closed_at = now
            for d in to_backfill:
                t = d["trade"]
                t.fill_price = float(t.entry_price or 0.0)
                t.filled_size = (
                    float(t.stake_usd) / float(t.entry_price)
                    if t.entry_price and t.entry_price > 0 else 0.0
                )
                # When --include-lost recovers a phantom-LOST row, the
                # trade was already marked terminal (status=LOST, pnl
                # set, exit_price/closed_at populated). Re-opening
                # requires clearing the terminal-state fields so
                # subsequent resolution doesn't see stale data.
                if t.status != TradeStatus.OPEN:
                    t.status = TradeStatus.OPEN
                    t.pnl = None
                    t.exit_price = None
                    t.closed_at = None
                # Status stays OPEN; market not yet reported on-chain.
                # resolve_trades will catch it once CLOB or the E2
                # fallback succeeds.

            await session.commit()
            click.echo(
                f"\nApplied: {len(to_mark_lost)} LOST (never filled), "
                f"{len(to_settle_won)} WON, "
                f"{len(to_settle_lost)} LOST (settled from on-chain payout), "
                f"{len(to_backfill)} backfilled-only. "
                f"Exposure released: ${released:.2f}."
            )

    asyncio.run(_reconcile_stuck())


# --- Measurement-layer readers (Phase 0) ---------------------------------


@main.command("exposure-report")
@click.option("--days", default=7, show_default=True, help="Look-back window in days.")
@click.option(
    "--since-epoch", "since_epoch", type=int, default=None,
    help="Only ticks at/after this config_epochs.id's start (see `epochs`).",
)
def exposure_report(days: int, since_epoch: int | None) -> None:
    """Per-tick capital / headroom report from ``exposure_snapshots`` (M4).

    The Phase-0 capital gate: is there room to fund new bets? Prints
    headroom / exposure / equity quantiles over the window, the fraction
    of ticks where headroom fell below one ``MIN_STAKE_USD`` (a new bet
    couldn't be funded), the bracket-vs-threshold open-position split, and
    a per-UTC-day median-headroom trend so a capital change shows up.
    """
    from sqlalchemy import select

    from src.config import settings
    from src.db.engine import async_session
    from src.db.models import ExposureSnapshot

    async def _run() -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        async with async_session() as session:
            ep = await _epoch_start(session, since_epoch)
            if ep is not None and ep > cutoff:
                cutoff = ep
            snaps = (
                await session.execute(
                    select(ExposureSnapshot)
                    .where(ExposureSnapshot.created_at >= cutoff)
                    .order_by(ExposureSnapshot.created_at)
                )
            ).scalars().all()
            if not snaps:
                click.echo(
                    f"No exposure_snapshots in the last {days}d. "
                    "Has job_unified_pipeline run since the M4 deploy?"
                )
                return

            rows = [
                {
                    "headroom": s.headroom, "exposure": s.exposure,
                    "equity": s.equity, "effective_cap": s.effective_cap,
                    "n_open": s.n_open, "n_open_by_class": s.n_open_by_class,
                    "created_at": s.created_at,
                    "n_sized_to_zero": s.n_sized_to_zero,
                }
                for s in snaps
            ]
            summ = _summarize_exposure(rows, min_stake=settings.MIN_STAKE_USD)

            click.echo(f"# Exposure report — last {days}d ({summ['count']:,} ticks)")
            click.echo()

            def _q(label: str, key: str) -> None:
                q = summ.get(key)
                if q:
                    click.echo(f"- **{label}** p25/p50/p75: ${q[0]:,.0f} / ${q[1]:,.0f} / ${q[2]:,.0f}")

            _q("Headroom", "headroom")
            _q("Exposure", "exposure")
            _q("Equity", "equity")
            _q("Effective cap", "effective_cap")
            click.echo(f"- **Median open positions:** {summ['n_open_median']:.0f}")
            click.echo(
                f"- **Ticks with headroom < ${settings.MIN_STAKE_USD:.0f}:** "
                f"{summ['low_headroom_frac'] * 100:.1f}%  "
                "(can't fund a new bet — the Phase-0 capital gate)"
            )
            if summ["by_class"]:
                split = ", ".join(
                    f"{k}={v}" for k, v in sorted(summ["by_class"].items())
                )
                click.echo(f"- **Open positions by class (summed):** {split}")
            click.echo()

            # Per-UTC-day trend so a capital/floor change is visible over time.
            click.echo("## Daily trend (UTC)")
            click.echo()
            click.echo("| Day | Ticks | Median headroom | Median exposure |")
            click.echo("|---|---:|---:|---:|")
            by_day: dict[object, list[dict]] = {}
            for r in rows:
                by_day.setdefault(r["created_at"].date(), []).append(r)
            for day in sorted(by_day):
                drows = by_day[day]
                hq = _quantiles([float(r["headroom"]) for r in drows])
                eq = _quantiles([float(r["exposure"]) for r in drows])
                click.echo(
                    f"| {day} | {len(drows):,} | "
                    f"${hq[1]:,.0f} | ${eq[1]:,.0f} |"
                )
            click.echo()

    asyncio.run(_run())


@main.command("resolution-report")
@click.option("--days", default=30, show_default=True, help="Look-back window in days.")
@click.option("--station", default=None, help="Restrict to one ICAO.")
@click.option(
    "--unit", default=None, type=click.Choice(["C", "F"], case_sensitive=False),
    help="Restrict to °C or °F markets.",
)
def resolution_report(days: int, station: str | None, unit: str | None) -> None:
    """Resolver-divergence report from ``market_resolutions`` (M3).

    De-circularises filter tuning: shows, per station, how far our
    routine-METAR daily max sits from the bound the actual resolution
    implies (``divergence_f`` = ours − resolved; positive = we read
    hotter). The Phase-3 gate for re-enabling ``RANGE_OVERSHOOT_LOCK_ENABLED``
    is the °C-city mean divergence shrinking toward ~0.5°C (~0.9°F).
    """
    from sqlalchemy import select

    from src.db.engine import async_session
    from src.db.models import MarketResolution

    async def _run() -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        async with async_session() as session:
            stmt = (
                select(MarketResolution)
                .where(MarketResolution.resolved_at >= cutoff)
                .order_by(MarketResolution.resolved_at)
            )
            if station:
                stmt = stmt.where(MarketResolution.station_icao == station.upper())
            if unit:
                stmt = stmt.where(MarketResolution.unit == unit.upper())
            res = (await session.execute(stmt)).scalars().all()
            if not res:
                click.echo(
                    f"No market_resolutions in the last {days}d "
                    f"(station={station or 'any'}, unit={unit or 'any'}). "
                    "Have any held markets settled since the M3 deploy?"
                )
                return

            rows = [
                {
                    "station_icao": r.station_icao, "unit": r.unit,
                    "divergence_f": r.divergence_f,
                    "routine_metar_max_f": r.routine_metar_max_f,
                }
                for r in res
            ]
            pending = sum(1 for r in rows if r["routine_metar_max_f"] is None)
            agg = _aggregate_divergence(rows)

            click.echo(f"# Resolution / divergence report — last {days}d")
            click.echo()
            click.echo(f"- **Settled rows:** {len(res):,}")
            click.echo(
                f"- **Pending backfill** (no routine max yet): {pending:,}  "
                "(filled at the next daily settlement)"
            )
            # Headline: °C-city mean divergence in °C vs the ~0.5°C tolerance.
            c_rows = [
                r["divergence_f"] for r in rows
                if r["unit"] == "C" and r["divergence_f"] is not None
                and r["routine_metar_max_f"] is not None
            ]
            if c_rows:
                mean_c = (sum(c_rows) / len(c_rows)) * 5.0 / 9.0
                click.echo(
                    f"- **°C-city mean divergence:** {mean_c:+.2f}°C "
                    f"(n={len(c_rows)}; Phase-3 target |mean| < ~0.5°C)"
                )
            click.echo()

            if not agg:
                click.echo(
                    "_No rows with both an outcome and a routine max yet — "
                    "run after a daily settlement backfill._"
                )
                return

            click.echo("## Divergence by station (ours − resolved, °F)")
            click.echo()
            click.echo("| Station | Unit | n | mean | std | min | max |")
            click.echo("|---|---|---:|---:|---:|---:|---:|")
            for d in agg:
                click.echo(
                    f"| {d['station_icao'] or '?'} | {d['unit'] or '?'} | "
                    f"{d['n']:,} | {d['mean']:+.2f} | {d['std']:.2f} | "
                    f"{d['min']:+.1f} | {d['max']:+.1f} |"
                )
            click.echo()

    asyncio.run(_run())


@main.command("shadow-report")
@click.option("--days", default=7, show_default=True, help="Look-back window in days.")
@click.option("--key", default=None, help="Only shadow keys with this dotted prefix.")
@click.option(
    "--since-epoch", "since_epoch", type=int, default=None,
    help="Only rows at/after this config_epochs.id's start (see `epochs`).",
)
def shadow_report(days: int, key: str | None, since_epoch: int | None) -> None:
    """Counterfactual telemetry report from ``evaluation_logs.shadow_json`` (M1).

    Generic over whatever an experiment writes: flattens the JSON to
    dotted leaf keys and prints per-leaf occupancy + numeric quantiles.
    Empty until the first experiment emits shadow data (e.g. per-class
    calibration's ``cal.*`` in Phase 1) — prints a clean empty message.
    """
    from sqlalchemy import select

    from src.db.engine import async_session
    from src.db.models import EvaluationLog

    async def _run() -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        async with async_session() as session:
            ep = await _epoch_start(session, since_epoch)
            if ep is not None and ep > cutoff:
                cutoff = ep
            blobs = (
                await session.execute(
                    select(EvaluationLog.shadow_json).where(
                        EvaluationLog.created_at >= cutoff,
                        EvaluationLog.shadow_json.isnot(None),
                    )
                )
            ).scalars().all()
            summ = _summarize_shadow([b for b in blobs if b])
            if key:
                summ = {k: v for k, v in summ.items() if k.startswith(key)}
            if not summ:
                click.echo(
                    f"No shadow telemetry in the last {days}d"
                    + (f" with prefix '{key}'" if key else "")
                    + ". (Populated once a measure-before-flip experiment opts in.)"
                )
                return

            click.echo(f"# Shadow telemetry — last {days}d")
            click.echo()
            click.echo("| Key | Count | p25 | p50 | p75 |")
            click.echo("|---|---:|---:|---:|---:|")
            for k in sorted(summ):
                info = summ[k]
                q = info["quantiles"]
                if q:
                    click.echo(
                        f"| {k} | {info['count']:,} | "
                        f"{q[0]:.4g} | {q[1]:.4g} | {q[2]:.4g} |"
                    )
                else:
                    click.echo(f"| {k} | {info['count']:,} | — | — | — |")
            click.echo()

    asyncio.run(_run())


@main.command("epochs")
@click.option("--limit", default=20, show_default=True, help="Most recent N epochs.")
def epochs(limit: int) -> None:
    """List config epochs (M2) with the flag diff vs the previous one.

    Each row marks a trading-config change; use the ``id`` as
    ``--since-epoch`` on the other reports to scope telemetry to a clean
    before/after-flip window.
    """
    from sqlalchemy import select

    from src.db.engine import async_session
    from src.db.models import ConfigEpoch

    async def _run() -> None:
        async with async_session() as session:
            eps = (
                await session.execute(
                    select(ConfigEpoch).order_by(ConfigEpoch.id.desc()).limit(limit)
                )
            ).scalars().all()
            if not eps:
                click.echo("No config_epochs yet. (Written at scheduler startup.)")
                return

            # Oldest-first so each diff compares to the genuine predecessor.
            eps = list(reversed(eps))
            click.echo(f"# Config epochs (latest {len(eps)})")
            click.echo()
            prev_flags: dict | None = None
            for e in eps:
                started = e.started_at.strftime("%Y-%m-%d %H:%M UTC") if e.started_at else "?"
                click.echo(f"## Epoch {e.id} — {started}" + (f"  ({e.note})" if e.note else ""))
                diff = _flags_diff(prev_flags, e.flags_json)
                if prev_flags is None:
                    click.echo("- _(initial snapshot)_")
                elif not diff:
                    click.echo("- _(no tracked-flag change)_")
                else:
                    for k in sorted(diff):
                        old, new = diff[k]
                        click.echo(f"- `{k}`: {old} → {new}")
                click.echo()
                prev_flags = e.flags_json

    asyncio.run(_run())


if __name__ == "__main__":
    main()
