"""Conviction-weighted EASY-YES lock monitor aggregator."""

from __future__ import annotations


def _summarize_conviction_locks(rows: list[dict]) -> dict:
    """Aggregate conviction-lock decision rows (each already joined to its
    settled Trade, if any). Pure — unit-tested in ``tests/test_cli_reports.py``.

    Each ``row`` carries: ``outcome``, ``requested_stake_usd``,
    ``actual_stake_usd``, ``branch``, ``status`` (Trade status name or None),
    ``pnl`` (None until settled).
    """
    deployed_outcomes = {"trade_filled", "trade_pending"}
    throttle_outcomes = {
        "stake_below_min", "drawdown_paused", "cluster_cap_hit",
        "order_failed", "insufficient_balance",
    }
    filled = [r for r in rows if r["outcome"] in deployed_outcomes]
    settled = [r for r in rows if r.get("status") in ("WON", "LOST")]
    won = [r for r in settled if r["status"] == "WON"]
    by_branch: dict[str, int] = {}
    for r in rows:
        b = r.get("branch") or "?"
        by_branch[b] = by_branch.get(b, 0) + 1
    return {
        "fires": len(rows),
        "filled": len(filled),
        "no_fill": sum(1 for r in rows if r["outcome"] == "no_fill"),
        "throttled": sum(1 for r in rows if r["outcome"] in throttle_outcomes),
        "deployed_actual": sum(r.get("actual_stake_usd") or 0.0 for r in filled),
        "deployed_requested": sum(r.get("requested_stake_usd") or 0.0 for r in filled),
        "settled": len(settled),
        "won": len(won),
        "lost": len(settled) - len(won),
        "net_pnl": sum(r.get("pnl") or 0.0 for r in settled),
        "by_branch": by_branch,
    }
