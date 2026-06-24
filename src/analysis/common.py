"""Small shared primitives for the analysis aggregators.

Pure functions over plain data (no DB, no click, no I/O) shared across the
report modules. Mirrors the pure-helper discipline of
``signals.market_resolution`` so the number-crunching is unit-testable.
"""

from __future__ import annotations

_OP_THRESHOLD = ("above", "at_least", "below", "at_most")
_OP_BRACKET = ("exactly", "range", "bracket")


def _op_class(op: str | None) -> str:
    if op in _OP_THRESHOLD:
        return "threshold"
    if op in _OP_BRACKET:
        return "bracket-like"
    return "other"


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


def _brier(p: float, y: int) -> float:
    return (p - y) ** 2


def _logloss(p: float, y: int) -> float:
    import math
    eps = 1e-6
    p = min(1.0 - eps, max(eps, p))
    return -(y * math.log(p) + (1 - y) * math.log(1.0 - p))


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


def _valley_bet_won(direction: str, yes_won: bool) -> bool:
    """Did the counterfactual valley bet win? ``BUY_NO`` wins iff the market
    resolved NO; ``BUY_YES`` iff it resolved YES."""
    return (yes_won is False) if direction == "BUY_NO" else (yes_won is True)


def _valley_ev_per_dollar(price: float, won: bool) -> float:
    """Per-$1-staked return of a binary bet at side price ``price``:
    ``(1-p)/p`` profit on win, ``-1`` (full stake lost) on loss. Break-even
    win rate = ``price`` — matches the trades-replay ``sum(pnl)/sum(stake)``
    convention."""
    if price <= 0.0:
        return 0.0
    return ((1.0 - price) / price) if won else -1.0


_PERF_RESOLVED = ("won", "lost")


def _is_phantom_lost(row: dict) -> bool:
    """A LOST trade whose order never filled — exposure-release cleanup, not a
    realised loss. Excluded from every PnL / win-rate sum."""
    return row.get("status") == "lost" and row.get("fill_price") is None
