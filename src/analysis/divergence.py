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


def _aggregate_station_day_divergence(rows: list[dict]) -> list[dict]:
    """Per-(station, unit) continuous resolver-divergence stats (Phase 1).

    Reads ``station_day_resolutions`` rows (dicts). Unlike ``_aggregate_divergence``
    (which scores the per-market bound-violation ``divergence_f``, mostly 0.0 or
    NULL), this scores ``divergence_point_f`` — our routine max minus the
    *intersected* resolved-max point — so the signal is a continuous signed °F on
    (ideally) every laddered station-day, not just the YES-pinned ~33%. Only rows
    with a finite ``divergence_point_f`` (i.e. a two-sided intersected window AND a
    routine max) contribute. Returns ``{station_icao, unit, n, mean, std, min,
    max, abs_mean}`` sorted by ``|mean|`` desc — worst-diverging stations first.
    """
    from collections import defaultdict

    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        d = r.get("divergence_point_f")
        if d is None:
            continue
        groups[(r.get("station_icao"), r.get("unit"))].append(float(d))
    out: list[dict] = []
    for (icao, unit), vals in groups.items():
        n = len(vals)
        mean = sum(vals) / n
        std = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5 if n else 0.0
        out.append({
            "station_icao": icao, "unit": unit, "n": n,
            "mean": mean, "std": std, "min": min(vals), "max": max(vals),
            "abs_mean": sum(abs(v) for v in vals) / n,
        })
    out.sort(key=lambda d: abs(d["mean"]), reverse=True)
    return out
