"""Shadow-json telemetry aggregator (measurement layer M1)."""

from __future__ import annotations

from src.analysis.common import _flatten_shadow, _quantiles


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
