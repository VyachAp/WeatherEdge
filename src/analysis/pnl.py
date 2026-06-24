"""Perf-review PnL / throttle / flip-gate aggregators.

Pure functions feeding the ``perf-review`` command + scheduler job. Same
dict-in/dict-out contract as the Phase-0 readers so they unit-test without a
DB. The PnL helpers enforce the project's phantom-LOST-safe methodology: a
LOST row that never filled released exposure but is NOT a realised loss (see
memory project_phantom_losses_2026-05-20), and dry-run rows keep their
requested stake so must be excluded from any $ sum.
"""

from __future__ import annotations

from src.analysis.common import _PERF_RESOLVED, _is_phantom_lost

# Outcomes that represent a throttle / failure (vs the success branches
# signal_written / trade_pending / trade_filled). The dominant throttle is
# the live volume bottleneck.
_PERF_THROTTLE_OUTCOMES = (
    "stake_below_min", "drawdown_paused", "cluster_cap_hit", "cap_exceeded",
    "no_fill", "order_failed", "insufficient_balance", "no_token_ids",
    "no_client", "dup_blocked_inproc", "dup_blocked_db",
)


def _perf_tradeable(rows: list[dict]) -> list[dict]:
    """Rows representing real capital at risk: drop dry-run + phantom-LOST,
    keep only OPEN / WON / LOST (PENDING never reached the book)."""
    out: list[dict] = []
    for r in rows:
        if r.get("exchange_status") == "dry_run":
            continue
        if r.get("status") not in (("open",) + _PERF_RESOLVED):
            continue
        if _is_phantom_lost(r):
            continue
        out.append(r)
    return out


def _realized_pnl(trade_rows: list[dict]) -> dict:
    """Phantom-LOST-safe realised P&L over a window of trade-row dicts.

    Each row: ``{status, pnl, stake_usd, fill_price, exchange_status,
    signal_kind, lock_branch}`` (``status`` lowercased: won/lost/open/...).
    Returns ``{n, n_won, win_pct, pnl, staked, ev_per_usd, n_open, by_kind,
    by_branch}`` where the rate fields cover *resolved* (won/lost) trades only.
    """
    rows = _perf_tradeable(trade_rows)
    resolved = [r for r in rows if r.get("status") in _PERF_RESOLVED]
    n = len(resolved)
    won = sum(1 for r in resolved if r.get("status") == "won")
    pnl = sum(float(r.get("pnl") or 0.0) for r in resolved)
    staked = sum(float(r.get("stake_usd") or 0.0) for r in resolved)

    def _group(key: str) -> dict:
        groups: dict[object, list[dict]] = {}
        for r in resolved:
            g = r.get(key)
            if g is None:
                continue
            groups.setdefault(g, []).append(r)
        out: dict = {}
        for g, rs in groups.items():
            gw = sum(1 for r in rs if r.get("status") == "won")
            gstk = sum(float(r.get("stake_usd") or 0.0) for r in rs)
            gpnl = sum(float(r.get("pnl") or 0.0) for r in rs)
            out[g] = {
                "n": len(rs), "won": gw, "win_pct": gw / len(rs),
                "pnl": gpnl, "ev_per_usd": (gpnl / gstk) if gstk else None,
            }
        return out

    return {
        "n": n, "n_won": won, "win_pct": (won / n) if n else None,
        "pnl": pnl, "staked": staked,
        "ev_per_usd": (pnl / staked) if staked else None,
        "n_open": sum(1 for r in rows if r.get("status") == "open"),
        "by_kind": _group("signal_kind"),
        "by_branch": _group("lock_branch"),
    }


def _throttle_breakdown(decision_rows: list[dict]) -> dict:
    """Funnel histogram over ``decision_logs`` rows + the dominant throttle.

    Each row: ``{outcome, metadata_json}``. Returns ``{total, outcomes,
    dominant_throttle, dominant_throttle_frac, n_throttled, size_reasons,
    floored_up_n}``. ``dominant_throttle`` is the most common skip/fail
    outcome — the live volume bottleneck (e.g. ``stake_below_min``).
    """
    total = len(decision_rows)
    outcomes: dict[str, int] = {}
    size_reasons: dict[str, int] = {}
    floored = 0
    for r in decision_rows:
        oc = r.get("outcome") or "(none)"
        outcomes[oc] = outcomes.get(oc, 0) + 1
        md = r.get("metadata_json") or {}
        if oc in ("stake_below_min", "drawdown_paused"):
            sr = str(md.get("size_reason", "(none)"))
            size_reasons[sr] = size_reasons.get(sr, 0) + 1
        if md.get("floored_up"):
            floored += 1
    throttles = {k: v for k, v in outcomes.items() if k in _PERF_THROTTLE_OUTCOMES}
    dom = max(throttles.items(), key=lambda kv: kv[1]) if throttles else None
    return {
        "total": total, "outcomes": outcomes,
        "dominant_throttle": dom[0] if dom else None,
        "dominant_throttle_frac": (dom[1] / total) if (dom and total) else None,
        "n_throttled": sum(throttles.values()),
        "size_reasons": size_reasons, "floored_up_n": floored,
    }


def _loss_classes(trade_rows: list[dict], min_n: int = 3) -> list[dict]:
    """Per-(signal_kind, lock_branch) realised P&L, worst (most negative)
    first. Phantom-LOST-safe; only cohorts with ``>= min_n`` resolved trades.
    Feeds new/worsening-loss-class detection (the win-small/lose-big shapes)."""
    rows = [r for r in _perf_tradeable(trade_rows)
            if r.get("status") in _PERF_RESOLVED]
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r.get("signal_kind"), r.get("lock_branch")), []).append(r)
    out: list[dict] = []
    for (kind, branch), rs in groups.items():
        if len(rs) < min_n:
            continue
        won = sum(1 for r in rs if r.get("status") == "won")
        pnl = sum(float(r.get("pnl") or 0.0) for r in rs)
        staked = sum(float(r.get("stake_usd") or 0.0) for r in rs)
        out.append({
            "signal_kind": kind, "lock_branch": branch, "n": len(rs),
            "won": won, "win_pct": won / len(rs), "pnl": pnl,
            "ev_per_usd": (pnl / staked) if staked else None,
        })
    out.sort(key=lambda d: d["pnl"])
    return out


def _window_regression(cur: dict, prev: dict) -> dict:
    """Current vs prior equal-length window for higher-is-better metrics
    (``pnl`` / ``win_pct`` / ``ev_per_usd`` / ``volume``). ``regressed`` is
    True when the metric got worse (delta < 0). ``None`` inputs → no verdict."""
    out: dict = {}
    for key in ("pnl", "win_pct", "ev_per_usd", "volume"):
        a, b = cur.get(key), prev.get(key)
        if a is None or b is None:
            out[key] = {"cur": a, "prev": b, "delta": None,
                        "pct_change": None, "regressed": False}
            continue
        delta = a - b
        out[key] = {"cur": a, "prev": b, "delta": delta,
                    "pct_change": (delta / abs(b)) if b else None,
                    "regressed": delta < 0}
    return out


# Minimum resolved-sample sizes before a flip-gate is judged (else
# "insufficient-data"). Codified in CLAUDE.md + docs/improvements.md.
_GATE_MIN_N = {
    "BRACKET_LIKE_NO_DISABLED": 30,
    "SIGMA_FLOOR_LEAD_TIME_ENABLED": 30,
    "PER_OPERATOR_CALIBRATION_ENABLED": 50,
    "VALLEY_MIN_EDGE": 8,
    "THRESHOLD_MIN_PROBABILITY": 20,
}


def _flip_gate_verdicts(ctx: dict) -> list[dict]:
    """Per-dark-flag readiness verdicts from measured telemetry. PURE.

    ``ctx`` carries the current flag values + the measured inputs the
    orchestrator gathered. Each verdict is advisory (``ready`` /
    ``not-ready`` / ``insufficient-data``) — a human still approves the
    ``.env`` flip (propose-only). The criteria mirror the codified gates in
    CLAUDE.md / docs/improvements.md.
    """
    flags = ctx.get("flags") or {}
    out: list[dict] = []

    def add(flag, verdict, measured, threshold, proposed, evidence, n):
        out.append({
            "flag": flag, "current_value": flags.get(flag), "verdict": verdict,
            "measured": measured, "threshold": threshold,
            "proposed_value": proposed, "evidence": evidence, "sample_n": n,
        })

    # 1. BRACKET_LIKE_NO_DISABLED → False once bracket-like NO is +EV again.
    b = ctx.get("bracket_like_no") or {}
    bn, bev = int(b.get("n") or 0), b.get("ev_per_usd")
    # Full readiness (sample floor AND +EV) — reused by the σ gate so a thin
    # bracket-like fluke can't trip the conjunctive condition.
    bracket_ready = (
        bn >= _GATE_MIN_N["BRACKET_LIKE_NO_DISABLED"] and bev is not None and bev > 0
    )
    if bn < _GATE_MIN_N["BRACKET_LIKE_NO_DISABLED"] or bev is None:
        bv = "insufficient-data"
    elif bev > 0:
        bv = "ready"
    else:
        bv = "not-ready"
    add("BRACKET_LIKE_NO_DISABLED", bv, bev, "realised EV/$1 > 0", False,
        "bracket-like NO realised EV/$1 over the window (kill-switch "
        "reactivation gate; ~0 live trades while disabled → insufficient)", bn)

    # 2. SIGMA_FLOOR_LEAD_TIME_ENABLED → True when the σ-arm widens
    #    far-from-peak AND bracket-like baseline is +EV (conjunctive).
    s = ctx.get("sigma") or {}
    sn, sd = int(s.get("n") or 0), s.get("delta_p50")
    if sn < _GATE_MIN_N["SIGMA_FLOOR_LEAD_TIME_ENABLED"] or sd is None:
        sv = "insufficient-data"
    elif sd > 0 and bracket_ready:
        sv = "ready"
    else:
        sv = "not-ready"
    add("SIGMA_FLOOR_LEAD_TIME_ENABLED", sv, sd,
        "sigma.delta p50 > 0 (far-from-peak) AND bracket-like EV>0", True,
        "σ lead-time arm widening on far-from-peak evals; conjunctive with "
        "the bracket-like-NO gate", sn)

    # 3. PER_OPERATOR_CALIBRATION_ENABLED → True when the threshold class fit
    #    un-squashes the [0.78,0.85) band AND the slope is plausible.
    c = ctx.get("cal") or {}
    cn, cd, slope = int(c.get("n") or 0), c.get("delta_p50"), c.get("slope")
    slope_ok = slope is not None and 0.2 <= slope <= 2.0
    if cn < _GATE_MIN_N["PER_OPERATOR_CALIBRATION_ENABLED"] or cd is None or slope is None:
        cv = "insufficient-data"
    elif cd >= 0 and slope_ok:
        cv = "ready"
    else:
        cv = "not-ready"
    add("PER_OPERATOR_CALIBRATION_ENABLED", cv, cd,
        "threshold cal.delta p50 >= 0 AND slope in [0.2,2.0]", True,
        f"per-class fit un-squash in-band (threshold slope={slope})", cn)

    # 4. VALLEY_MIN_EDGE → set when the P2-allows cohort is +EV out-of-sample
    #    and beats P2-blocks.
    val = ctx.get("valley") or {}
    vn, va, vb = int(val.get("p2_allows_n") or 0), val.get("p2_allows_ev"), val.get("p2_blocks_ev")
    if vn < _GATE_MIN_N["VALLEY_MIN_EDGE"] or va is None:
        vv = "insufficient-data"
    elif va > 0 and (vb is None or va > vb):
        vv = "ready"
    else:
        vv = "not-ready"
    add("VALLEY_MIN_EDGE", vv, va, "p2_allows EV/$1 > 0 and > p2_blocks",
        val.get("p2_min_edge") or 0.15, "price-band P2 promotion gate", vn)

    # 5. THRESHOLD_MIN_PROBABILITY → 0.78 only if the recoverable band is +EV
    #    AND it isn't dead-on-arrival at the stake_below_min throttle.
    t = ctx.get("threshold_loosen") or {}
    tn, tev, dom = int(t.get("recoverable_n") or 0), t.get("recoverable_ev"), bool(t.get("stake_below_min_dominant"))
    if dom:
        tv, tevidence = "not-ready", (
            "stake_below_min dominates the throttle — loosening only adds "
            "passing evals that can't fund; fix the capital/depth lever first")
    elif tn < _GATE_MIN_N["THRESHOLD_MIN_PROBABILITY"] or tev is None:
        tv, tevidence = "insufficient-data", (
            "recoverable threshold band EV (run `evals-report --operator threshold`)")
    elif tev > 0:
        tv, tevidence = "ready", "recoverable threshold band +EV and not throttle-blocked"
    else:
        tv, tevidence = "not-ready", "recoverable threshold band not +EV"
    add("THRESHOLD_MIN_PROBABILITY", tv, tev,
        "recoverable band EV/$1 > 0 AND stake_below_min not dominant", 0.78,
        tevidence, tn)

    # 6. NEAR_PEAK_FLOOR_UP_ENABLED — standing note: the real lever is the
    #    depth cap, not the floor (memory project_floorup_inert_2026-06-03).
    add("NEAR_PEAK_FLOOR_UP_ENABLED", "not-ready", None,
        "floored fills win >= break-even", True,
        "inert: depth cap (20%×book) binds tighter than the floor on <$25 "
        "books — raise DEPTH_POSITION_CAP_PCT instead", 0)

    return out


def _summarize_floored_fills(rows: list[dict]) -> dict:
    """Win-rate / EV of near-peak floored-up fills vs normal fills.

    The post-flip gate for ``NEAR_PEAK_FLOOR_UP_ENABLED`` (Phase 1): a
    floored bet can't be shadowed (it's either placed or not), so the
    discipline is to measure realised outcomes after the flip. ``rows``
    are filled-trade dicts ``{floored_up: bool, status: str|None,
    entry_price: float|None}`` where ``status`` is ``won``/``lost``/
    ``open``/None. Returns ``{floored: {...}, normal: {...}}`` each with
    n / resolved / won / won_pct / break_even (mean entry price) /
    ev_per_dollar. The gate passes when the floored bucket's
    ``won_pct >= break_even`` (equivalently ``ev_per_dollar >= 0``) over a
    meaningful resolved count. Pure — unit-tested without a DB.
    """
    def _bucket(rs: list[dict]) -> dict:
        resolved = [r for r in rs if r.get("status") in ("won", "lost")]
        won = [r for r in resolved if r.get("status") == "won"]
        entries = [
            float(r["entry_price"]) for r in resolved if r.get("entry_price")
        ]
        evs = [
            (1.0 / float(r["entry_price"]) - 1.0)
            if r.get("status") == "won" else -1.0
            for r in resolved if r.get("entry_price")
        ]
        return {
            "n": len(rs),
            "resolved": len(resolved),
            "won": len(won),
            "won_pct": (len(won) / len(resolved)) if resolved else None,
            "break_even": (sum(entries) / len(entries)) if entries else None,
            "ev_per_dollar": (sum(evs) / len(evs)) if evs else None,
        }

    return {
        "floored": _bucket([r for r in rows if r.get("floored_up")]),
        "normal": _bucket([r for r in rows if not r.get("floored_up")]),
    }
