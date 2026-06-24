"""Counterfactual mining — the self-improvement substrate.

Pure aggregators (no DB / no click / no I/O) over already-joined telemetry rows
that answer the question the bot needs to learn from: **which trades did we
decline to place that then resolved in our favour?** Two cohorts:

1. **Rejected at evaluation** — `evaluation_logs.passes = False`. A filter
   blocked the side. The "counterfactual win" is whether the *rejected side*
   would have won given the resolution.
2. **Throttled post-filter** — `decision_logs.outcome` in `THROTTLE_OUTCOMES`.
   The side passed every filter but no order was placed (sizing / exposure /
   dedup). Same counterfactual-win question.

Each cohort is grouped by reject-reason / throttle-outcome, station, operator
class and side-buy-price band, and scored for hypothetical win-rate and EV per
$1 (reusing the trades-replay convention in `analysis.common`). Cohorts whose
rejected side would have won *above break-even* (`win_rate > avg price`) are the
actionable signal: a filter (or a station / price band) that is leaving money.

The caller (a CLI report / nightly knowledge-refresh job) does the SQL join
(`evaluation_logs`/`decision_logs` → `market_resolutions.yes_won`) and passes
enriched row dicts here; this module only crunches numbers, so it is fully
unit-testable without a database. Ground truth is `market_resolutions.yes_won`
(the de-circularised on-chain outcome), NOT our own routine-METAR proxy.
"""

from __future__ import annotations

from collections import defaultdict

from src.analysis.common import (
    _op_class,
    _valley_bet_won,
    _valley_ev_per_dollar,
)

# decision_logs outcomes where the side PASSED filters but was not placed —
# the "we wanted to but couldn't / wouldn't size it" funnel. Mirrors the
# OUTCOME_* throttle constants in signals.decision_log.
THROTTLE_OUTCOMES = (
    "stake_below_min",
    "drawdown_paused",
    "cap_exceeded",
    "cluster_cap_hit",
    "insufficient_balance",
)

# Side-buy-price bands — the U-shaped-EV partition used elsewhere (deep-value /
# mid "valley" / near-lock). Half-open [lo, hi).
_PRICE_BANDS = ((0.0, 0.40), (0.40, 0.60), (0.60, 0.80), (0.80, 0.95), (0.95, 1.0))


def price_band(price: float | None) -> str:
    """Label the side-buy-price band, or ``"?"`` when price is missing."""
    if price is None:
        return "?"
    for lo, hi in _PRICE_BANDS:
        if lo <= price < hi:
            return f"[{lo:.2f},{hi:.2f})"
    return "[1.00]" if price >= 1.0 else "?"


def lead_band(hours_until_peak: float | None) -> str:
    """Coarse lead-to-peak bucket for grouping (past / near / far)."""
    if hours_until_peak is None:
        return "?"
    if hours_until_peak <= 0:
        return "past-peak"
    if hours_until_peak <= 2:
        return "0-2h"
    if hours_until_peak <= 6:
        return "2-6h"
    return ">6h"


def cohort_stats(rows: list[dict]) -> dict:
    """Counterfactual stats for one cohort of declined per-side decisions.

    Each row needs ``direction`` ("BUY_YES"/"BUY_NO") and ``yes_won`` (bool, or
    ``None`` when the market hasn't resolved yet); ``side_price`` (the buy price
    of that side) is used for EV when present. Returns counts, the hypothetical
    win-rate of the declined side, average side price, break-even (= avg price),
    and average EV per $1 staked. ``edge_pp`` = (win_rate − break_even) × 100 —
    positive means the declined side was, in hindsight, +EV.
    """
    n = len(rows)
    resolved = [r for r in rows if r.get("yes_won") is not None]
    wins = [r for r in resolved if _valley_bet_won(r["direction"], r["yes_won"])]
    priced = [r for r in resolved if r.get("side_price") not in (None, 0)]
    win_rate = (len(wins) / len(resolved)) if resolved else None
    avg_price = (sum(r["side_price"] for r in priced) / len(priced)) if priced else None
    ev = (
        sum(
            _valley_ev_per_dollar(
                r["side_price"], _valley_bet_won(r["direction"], r["yes_won"])
            )
            for r in priced
        )
        / len(priced)
        if priced
        else None
    )
    edge_pp = (
        (win_rate - avg_price) * 100.0
        if win_rate is not None and avg_price is not None
        else None
    )
    return {
        "n": n,
        "n_resolved": len(resolved),
        "would_win": len(wins),
        "win_rate": win_rate,
        "avg_price": avg_price,
        "break_even": avg_price,
        "edge_pp": edge_pp,
        "ev_per_dollar": ev,
    }


def group_cohorts(rows: list[dict], key: str) -> list[dict]:
    """Group rows by ``row[key]`` and return per-group ``cohort_stats`` (with
    the group label under ``key``), sorted by *foregone EV* — ``ev_per_dollar ×
    n_resolved`` desc — so the cohorts leaving the most money on the table sort
    first. Unresolved-only groups (ev None) sort last."""
    groups: dict[object, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r.get(key)].append(r)
    out = []
    for label, grp in groups.items():
        stats = cohort_stats(grp)
        stats[key] = label
        out.append(stats)
    out.sort(
        key=lambda d: (
            (d["ev_per_dollar"] or -9.9) * d["n_resolved"]
            if d["ev_per_dollar"] is not None
            else -9e9
        ),
        reverse=True,
    )
    return out


def _enrich(rows: list[dict]) -> list[dict]:
    """Attach derived grouping fields (op_class, price band) used by the
    cohort pivots, without mutating the inputs."""
    out = []
    for r in rows:
        out.append({
            **r,
            "op_class": _op_class(r.get("op")),
            "price_band": price_band(r.get("side_price")),
        })
    return out


def mine_rejected(rejected_rows: list[dict]) -> dict:
    """Cohort the at-evaluation rejections (passes=False) several ways.

    ``rejected_rows``: one dict per rejected per-side eval with ``direction``,
    ``reject_reason`` (prefix is fine), ``side_price``, ``op``, ``station_icao``,
    ``hours_until_peak`` and joined ``yes_won`` (None if unresolved).
    """
    rows = _enrich(rejected_rows)
    return {
        "total": len(rows),
        "by_reason": group_cohorts(rows, "reject_reason"),
        "by_station": group_cohorts(rows, "station_icao"),
        "by_op_class": group_cohorts(rows, "op_class"),
        "by_price_band": group_cohorts(rows, "price_band"),
    }


def mine_throttled(throttled_rows: list[dict]) -> dict:
    """Cohort the post-filter throttles (decision_logs throttle outcomes).

    ``throttled_rows``: one dict per throttled per-side decision with
    ``direction``, ``outcome`` (in ``THROTTLE_OUTCOMES``), ``side_price`` (may be
    None), ``op``, ``station_icao`` and joined ``yes_won``.
    """
    rows = _enrich([r for r in throttled_rows if r.get("outcome") in THROTTLE_OUTCOMES])
    return {
        "total": len(rows),
        "by_outcome": group_cohorts(rows, "outcome"),
        "by_station": group_cohorts(rows, "station_icao"),
        "by_op_class": group_cohorts(rows, "op_class"),
    }


def headline_missed(cohorts: list[dict], *, min_resolved: int = 5) -> list[dict]:
    """The actionable subset of a `group_cohorts` list: cohorts with at least
    ``min_resolved`` resolved decisions whose declined side was +EV in hindsight
    (``ev_per_dollar > 0`` and ``edge_pp > 0``). These are "we left money here"
    — candidate filters/stations/bands to revisit (propose-only)."""
    return [
        c
        for c in cohorts
        if c["n_resolved"] >= min_resolved
        and (c["ev_per_dollar"] or 0.0) > 0.0
        and (c["edge_pp"] or 0.0) > 0.0
    ]


def mine_counterfactuals(
    rejected_rows: list[dict],
    throttled_rows: list[dict],
    *,
    min_resolved: int = 5,
) -> dict:
    """Top-level entry: mine both cohorts and surface the headline missed
    winners (by reject-reason for the rejected cohort, by outcome for the
    throttled one). Pure — the caller supplies the joined rows and renders."""
    rejected = mine_rejected(rejected_rows)
    throttled = mine_throttled(throttled_rows)
    return {
        "rejected": rejected,
        "throttled": throttled,
        "headline": {
            "rejected_by_reason": headline_missed(
                rejected["by_reason"], min_resolved=min_resolved
            ),
            "throttled_by_outcome": headline_missed(
                throttled["by_outcome"], min_resolved=min_resolved
            ),
        },
    }
