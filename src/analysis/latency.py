"""Information-latency aggregator (Phase 2, 2026-06-24).

Pure number-crunching for ``latency-report``. Pairs the per-METAR reprice
snapshots — rows sharing a ``(market_id, metar_observed_at)`` key — and measures
how far the market's YES mid moved from the first snapshot (T0, at information
arrival) to the last, bucketed by how much of the day was already observed. A
positive mean move that grows with ``obs_fraction`` is the realized
information-latency edge: the market chasing observations we already held.
"""

from __future__ import annotations

from collections import defaultdict


def _obs_bucket(frac: float | None) -> str:
    """Coarse observation-fraction band label (the thesis axis)."""
    if frac is None:
        return "unknown"
    if frac < 0.34:
        return "low (<0.34)"
    if frac < 0.67:
        return "mid (0.34-0.67)"
    return "high (>=0.67)"


def _pair_move(group: list[dict]) -> dict | None:
    """Reduce one ``(market_id, metar_observed_at)`` group to its T0→last move.

    ``group`` is the list of snapshot dicts for a single triggering METAR. They
    are sorted by ``created_at``; the move is ``last.yes_mid − first.yes_mid``.
    ``obs_fraction`` is taken from whichever row carries it (the fast-poll T0 row
    leaves it None; the unified rows populate it). Returns None when the group has
    fewer than two priced snapshots (no move to measure).
    """
    priced = [r for r in group if r.get("yes_mid") is not None]
    if len(priced) < 2:
        return None
    priced.sort(key=lambda r: r.get("created_at") or "")
    first, last = priced[0], priced[-1]
    move = float(last["yes_mid"]) - float(first["yes_mid"])
    obs_frac = next(
        (r["obs_fraction"] for r in group if r.get("obs_fraction") is not None),
        None,
    )
    span_s = None
    if last.get("seconds_since_obs") is not None and first.get("seconds_since_obs") is not None:
        span_s = float(last["seconds_since_obs"]) - float(first["seconds_since_obs"])
    return {
        "market_id": first.get("market_id"),
        "station_icao": first.get("station_icao"),
        "move": move,
        "abs_move": abs(move),
        "obs_fraction": obs_frac,
        "n_snapshots": len(priced),
        "span_seconds": span_s,
    }


def _summarize_reprice(rows: list[dict]) -> dict:
    """Summarize reprice snapshots into per-obs_fraction-bucket move stats.

    Groups ``rows`` by ``(market_id, metar_observed_at)``, reduces each to its
    T0→last YES-mid move, then aggregates the moves by observation-fraction
    bucket. Returns ``{n_groups, n_measurable, overall: {...}, by_bucket:
    [{bucket, n, mean_move, mean_abs_move, ...}]}``. ``mean_move`` > 0 means the
    market rose toward YES after the observation landed; the falsifiable thesis is
    that it rises with ``obs_fraction``.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r.get("market_id"), r.get("metar_observed_at"))].append(r)

    moves = [m for m in (_pair_move(g) for g in groups.values()) if m is not None]
    n_groups = len(groups)
    if not moves:
        return {"n_groups": n_groups, "n_measurable": 0, "overall": None,
                "by_bucket": []}

    def _stats(ms: list[dict]) -> dict:
        n = len(ms)
        mean_move = sum(m["move"] for m in ms) / n
        mean_abs = sum(m["abs_move"] for m in ms) / n
        spans = [m["span_seconds"] for m in ms if m["span_seconds"] is not None]
        return {
            "n": n,
            "mean_move": mean_move,
            "mean_abs_move": mean_abs,
            "mean_span_seconds": (sum(spans) / len(spans)) if spans else None,
        }

    by_bucket_groups: dict[str, list[dict]] = defaultdict(list)
    for m in moves:
        by_bucket_groups[_obs_bucket(m["obs_fraction"])].append(m)

    # Stable, thesis-ordered bucket sequence.
    order = ["low (<0.34)", "mid (0.34-0.67)", "high (>=0.67)", "unknown"]
    by_bucket = [
        {"bucket": b, **_stats(by_bucket_groups[b])}
        for b in order if b in by_bucket_groups
    ]

    return {
        "n_groups": n_groups,
        "n_measurable": len(moves),
        "overall": _stats(moves),
        "by_bucket": by_bucket,
    }
