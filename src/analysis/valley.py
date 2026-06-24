"""Price-band valley policy aggregator (P1→P2 promotion gate)."""

from __future__ import annotations

from src.analysis.common import _valley_bet_won, _valley_ev_per_dollar


def _aggregate_valley(rows: list[dict]) -> dict:
    """Score resolved valley evaluations, split by the P2 decision.

    Each row: ``{direction, price, yes_won, p2_would_block}``. Returns
    ``{"all_valley", "p2_allows", "p2_blocks"}`` — each a summary
    ``{n, won, win_pct, breakeven_pct, edge_pp, ev_per_usd}`` (rate fields are
    ``None`` for an empty cohort). ``p2_allows`` (``p2_would_block=False``) is
    the cohort the P2 policy keeps; the P1→P2 promotion gate is this cohort
    being +EV (and ideally beating ``p2_blocks``).
    """
    def _summ(subset: list[dict]) -> dict:
        n = len(subset)
        if n == 0:
            return {"n": 0, "won": 0, "win_pct": None,
                    "breakeven_pct": None, "edge_pp": None, "ev_per_usd": None}
        wins = [_valley_bet_won(r["direction"], r["yes_won"]) for r in subset]
        won = sum(1 for w in wins if w)
        be = sum(r["price"] for r in subset) / n
        ev = sum(_valley_ev_per_dollar(r["price"], w)
                 for r, w in zip(subset, wins)) / n
        wp = 100.0 * won / n
        return {"n": n, "won": won, "win_pct": wp,
                "breakeven_pct": 100.0 * be, "edge_pp": wp - 100.0 * be,
                "ev_per_usd": ev}

    return {
        "all_valley": _summ(rows),
        "p2_allows": _summ([r for r in rows if not r["p2_would_block"]]),
        "p2_blocks": _summ([r for r in rows if r["p2_would_block"]]),
    }
