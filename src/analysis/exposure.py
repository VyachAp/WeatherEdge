"""Exposure-report aggregator (measurement layer M4)."""

from __future__ import annotations

from src.analysis.common import _quantiles


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
