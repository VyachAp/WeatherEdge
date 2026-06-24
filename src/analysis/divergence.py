"""Resolver-divergence aggregator (measurement layer M3)."""

from __future__ import annotations


def _aggregate_divergence(rows: list[dict]) -> list[dict]:
    """Per-(station, unit) resolver-divergence stats from M3 rows (dicts).

    Only rows that have both ``divergence_f`` and ``routine_metar_max_f``
    (i.e. the daily-settlement backfill has run) contribute. Returns a
    list of ``{station_icao, unit, n, mean, std, min, max}`` sorted by
    ``|mean|`` desc — the worst-diverging stations first. This is the
    Phase-3 audit surface.
    """
    from collections import defaultdict

    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("divergence_f") is None or r.get("routine_metar_max_f") is None:
            continue
        groups[(r.get("station_icao"), r.get("unit"))].append(float(r["divergence_f"]))
    out: list[dict] = []
    for (icao, unit), vals in groups.items():
        n = len(vals)
        mean = sum(vals) / n
        std = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5 if n else 0.0
        out.append({
            "station_icao": icao, "unit": unit, "n": n,
            "mean": mean, "std": std, "min": min(vals), "max": max(vals),
        })
    out.sort(key=lambda d: abs(d["mean"]), reverse=True)
    return out
