"""Config-epoch flag-diff aggregator (measurement layer M2)."""

from __future__ import annotations


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
