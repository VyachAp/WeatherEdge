"""Daily "why no trade" funnel aggregators.

Pure dict-in/dict-out functions (no DB / no click / no I/O), same contract as
:mod:`src.analysis.pnl`, so they unit-test without a database. The DB layer
(``cli.no_trade_result``) batch-reads the telemetry, dedups it to one row per
``(market_id, direction)`` per window, and hands plain dicts to these helpers.

The funnel answers "of the markets we could have traded today, where did each
candidate fall out?":

    L0  universe        binary markets active in the window         (denominator)
         └─ never evaluated = universe − markets present in evaluation_logs
    L1  evaluated       distinct (market, side) in evaluation_logs
         ├─ rejected    passes=False → reject_reason rollup (prefix-bucketed)
         └─ passed      passes=True  → feeds L2
    L2  post-filter     decision_logs outcomes (via _throttle_breakdown)
         ├─ throttled   stake_below_min / drawdown_paused / cap_exceeded / …
         └─ traded      trade_pending / trade_filled / signal_written

Known limitation (by design — see the plan): the ~21 "silent" pre-evaluation
drop points (excluded stations, future-day, bias-runaway, near-resolved,
lock-fired-first, circuit-breaker) write no telemetry row, so ``never_evaluated``
is a *count*, not a per-reason breakdown.
"""

from __future__ import annotations

from src.analysis.pnl import _throttle_breakdown
from src.signals.decision_log import (
    OUTCOME_SIGNAL_WRITTEN,
    OUTCOME_TRADE_FILLED,
    OUTCOME_TRADE_PENDING,
)

# Outcomes that mean a Signal/Trade row was actually produced (not throttled).
_TRADED_OUTCOMES = (
    OUTCOME_TRADE_FILLED,
    OUTCOME_TRADE_PENDING,
    OUTCOME_SIGNAL_WRITTEN,
)


def _reason_prefix(reason: str | None) -> str:
    """Bucket a ``reject_reason`` by its leading token.

    ``"edge 0.03 < 0.05"`` → ``"edge"``; ``"bracket-like NO disabled …"`` →
    ``"bracket-like"``; ``"price-valley blocked …"`` → ``"price-valley"``.
    Mirrors the prefix rollup ``evals-report`` already uses so the two reports
    agree. ``None`` → ``"(unknown)"``.
    """
    if not reason:
        return "(unknown)"
    return reason.strip().split(" ", 1)[0] or "(unknown)"


def _reject_rollup(eval_rows: list[dict]) -> dict:
    """Histogram the rejected per-side evals by reason prefix / kind / op-class.

    Each row: ``{passes, reject_reason, signal_kind, op_class}``. Only
    ``passes is False`` rows are counted. Returns
    ``{total, by_reason, by_kind, by_op_class}`` (each inner dict sorted by
    descending count for stable digest output).
    """
    rejected = [r for r in eval_rows if not r.get("passes")]
    by_reason: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    by_op_class: dict[str, int] = {}
    for r in rejected:
        pref = _reason_prefix(r.get("reject_reason"))
        by_reason[pref] = by_reason.get(pref, 0) + 1
        k = str(r.get("signal_kind") or "(none)")
        by_kind[k] = by_kind.get(k, 0) + 1
        oc = str(r.get("op_class") or "(unknown)")
        by_op_class[oc] = by_op_class.get(oc, 0) + 1

    def _sorted(d: dict[str, int]) -> dict[str, int]:
        return dict(sorted(d.items(), key=lambda kv: (-kv[1], kv[0])))

    return {
        "total": len(rejected),
        "by_reason": _sorted(by_reason),
        "by_kind": _sorted(by_kind),
        "by_op_class": _sorted(by_op_class),
    }


def aggregate_no_trade_funnel(
    *,
    universe_n: int,
    eval_rows: list[dict],
    decision_rows: list[dict],
    dd_state: dict | None = None,
) -> dict:
    """Assemble the L0–L3 no-trade funnel from deduped telemetry dicts.

    ``eval_rows``: deduped per ``(market_id, direction)`` —
    ``{market_id, direction, passes, reject_reason, signal_kind, op_class}``.
    ``decision_rows``: deduped per ``(market_id, direction)`` —
    ``{outcome, metadata_json}`` (the L2 histogram reuses
    :func:`src.analysis.pnl._throttle_breakdown`).
    ``dd_state``: ``{dd_level, dd_multiplier}`` of the most recent decision (the
    drawdown gate in force), or ``None``.

    Returns a JSON-serialisable dict consumed by :func:`render_no_trade_digest`
    and persisted as the daily artifact.
    """
    evaluated_n = len(eval_rows)
    distinct_markets = len({r.get("market_id") for r in eval_rows if r.get("market_id")})
    never_evaluated_n = max(0, universe_n - distinct_markets)

    rejects = _reject_rollup(eval_rows)
    passed_n = sum(1 for r in eval_rows if r.get("passes"))

    throttle = _throttle_breakdown(decision_rows)
    traded_n = sum(throttle["outcomes"].get(o, 0) for o in _TRADED_OUTCOMES)

    return {
        "universe_n": universe_n,
        "never_evaluated_n": never_evaluated_n,
        "evaluated_n": evaluated_n,
        "rejected_n": rejects["total"],
        "passed_n": passed_n,
        "traded_n": traded_n,
        "rejects": rejects,
        "throttle": throttle,
        "dd": dd_state or {},
    }


def render_no_trade_digest(result: dict, *, markdown: bool = True) -> str:
    """Render the funnel dict as a compact digest.

    ``markdown=True`` → Telegram MarkdownV2 (dynamic values escaped via
    ``_escape_md2``); ``markdown=False`` → plain text for the console. Same
    rendering shape as :func:`src.analysis.perf.render_perf_digest`.
    """
    from src.execution.alerter import _escape_md2

    e = _escape_md2 if markdown else (lambda x: str(x))
    bold = (lambda t: f"*{t}*") if markdown else (lambda t: t)
    bullet = "• "
    L: list[str] = []

    win = str(result.get("window_days", "?")) + "d"
    since = result.get("since")
    if since:
        win += " since " + str(since)[:10]
    L.append(f"\U0001f6ab {bold('No-trade funnel')} — {e(win)}")

    uni = result.get("universe_n", 0)
    ev = result.get("evaluated_n", 0)
    passed = result.get("passed_n", 0)
    traded = result.get("traded_n", 0)
    L.append(
        f"\U0001f50d {e(uni)} markets → {e(ev)} evaluated → {e(passed)} passed "
        f"→ {e(traded)} traded"
    )
    nev = result.get("never_evaluated_n", 0)
    if nev:
        L.append(f"{bullet}{e(nev)} never evaluated (silent skip — count only)")

    rejects = result.get("rejects") or {}
    by_reason = rejects.get("by_reason") or {}
    if by_reason:
        top = list(by_reason.items())[:5]
        parts = ", ".join(f"{e(k)} {e(v)}" for k, v in top)
        L.append(f"❌ Rejected {e(rejects.get('total', 0))}: {parts}")

    th = result.get("throttle") or {}
    dom = th.get("dominant_throttle")
    if dom:
        fr = th.get("dominant_throttle_frac")
        fr_s = f"{fr*100:.0f}%" if fr is not None else "n/a"
        L.append(
            f"\U0001f6a6 Top throttle: {e(dom)} ({e(fr_s)} of "
            f"{e(th.get('total', 0))} decisions)"
        )
    size_reasons = th.get("size_reasons") or {}
    if size_reasons:
        top_sr = sorted(size_reasons.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
        parts = ", ".join(f"{e(k)} {e(v)}" for k, v in top_sr)
        L.append(f"{bullet}sizing cap bound: {parts}")

    dd = result.get("dd") or {}
    if dd.get("dd_level"):
        mult = dd.get("dd_multiplier")
        mult_s = f"×{mult:.2f}" if mult is not None else ""
        L.append(f"\U0001f4c9 Drawdown: {e(dd['dd_level'])} {e(mult_s)}")

    L.append(
        e("note: 'never evaluated' is a count, not a reason "
          "(silent pre-eval skips write no telemetry).")
    )
    return "\n".join(L)
