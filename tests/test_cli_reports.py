"""Unit tests for the Phase 0 measurement-layer report aggregators.

The CLI report commands are thin I/O wrappers; the number-crunching is in
pure module-level functions in ``src.cli`` (no DB, no click), so they're
tested here with plain-dict inputs.
"""

from __future__ import annotations

from src.cli import (
    _aggregate_divergence,
    _flags_diff,
    _flatten_shadow,
    _quantiles,
    _summarize_exposure,
    _summarize_shadow,
)


# --- _quantiles -------------------------------------------------------------


def test_quantiles_empty_is_none():
    assert _quantiles([]) is None


def test_quantiles_nearest_rank():
    p25, p50, p75 = _quantiles([float(i) for i in range(1, 9)])  # 1..8
    assert p25 == 2.0
    assert p50 == 5.0
    assert p75 == 7.0


# --- _summarize_exposure ----------------------------------------------------


def test_summarize_exposure_empty():
    assert _summarize_exposure([]) == {"count": 0}


def test_summarize_exposure_headroom_and_low_frac():
    rows = [
        {"headroom": 2.0, "exposure": 298.0, "equity": 300.0,
         "effective_cap": 300.0, "n_open": 28,
         "n_open_by_class": {"bracket-like": 22, "threshold": 6}},
        {"headroom": 50.0, "exposure": 250.0, "equity": 300.0,
         "effective_cap": 300.0, "n_open": 20,
         "n_open_by_class": {"threshold": 20}},
    ]
    summ = _summarize_exposure(rows, min_stake=5.0)
    assert summ["count"] == 2
    # One of two ticks has headroom < $5 → 50%.
    assert summ["low_headroom_frac"] == 0.5
    # by_class sums across ticks.
    assert summ["by_class"]["threshold"] == 26
    assert summ["by_class"]["bracket-like"] == 22
    assert summ["headroom"] is not None


def test_summarize_exposure_handles_missing_by_class():
    rows = [{"headroom": 10.0, "exposure": 0.0, "equity": 100.0,
             "effective_cap": 300.0, "n_open": 0, "n_open_by_class": None}]
    summ = _summarize_exposure(rows)
    assert summ["by_class"] == {}
    assert summ["low_headroom_frac"] == 0.0


# --- _aggregate_divergence --------------------------------------------------


def test_aggregate_divergence_groups_and_sorts():
    rows = [
        # KAAA: +3 and +5 → mean +4 (read hotter).
        {"station_icao": "KAAA", "unit": "F", "divergence_f": 3.0,
         "routine_metar_max_f": 80.0},
        {"station_icao": "KAAA", "unit": "F", "divergence_f": 5.0,
         "routine_metar_max_f": 82.0},
        # KBBB: -1 → mean -1, smaller magnitude → sorted after KAAA.
        {"station_icao": "KBBB", "unit": "C", "divergence_f": -1.0,
         "routine_metar_max_f": 70.0},
        # Skipped: no routine max yet (pending backfill).
        {"station_icao": "KCCC", "unit": "C", "divergence_f": None,
         "routine_metar_max_f": None},
    ]
    agg = _aggregate_divergence(rows)
    assert [d["station_icao"] for d in agg] == ["KAAA", "KBBB"]
    assert agg[0]["n"] == 2
    assert agg[0]["mean"] == 4.0
    assert agg[0]["min"] == 3.0
    assert agg[0]["max"] == 5.0
    assert agg[1]["mean"] == -1.0


def test_aggregate_divergence_empty():
    assert _aggregate_divergence([]) == []
    # rows present but all pending backfill → nothing aggregable.
    assert _aggregate_divergence(
        [{"station_icao": "K", "unit": "F", "divergence_f": None,
          "routine_metar_max_f": None}]
    ) == []


# --- shadow helpers ---------------------------------------------------------


def test_flatten_shadow_dotted_keys():
    flat = _flatten_shadow({"cal": {"pooled": 0.8, "class": 0.78}, "n": 3})
    assert flat == {"cal.pooled": 0.8, "cal.class": 0.78, "n": 3}


def test_summarize_shadow_counts_and_quantiles():
    blobs = [
        {"cal": {"pooled": 0.80, "class": 0.78}, "flag": True},
        {"cal": {"pooled": 0.90, "class": 0.85}, "flag": False},
        {"cal": {"pooled": 0.70}},  # class missing here
        None,  # null blob ignored
    ]
    summ = _summarize_shadow(blobs)
    assert summ["cal.pooled"]["count"] == 3
    assert summ["cal.class"]["count"] == 2
    # Booleans excluded from numeric quantiles.
    assert summ["flag"]["quantiles"] is None
    assert summ["flag"]["count"] == 2
    # Numeric leaf has quantiles.
    assert summ["cal.pooled"]["quantiles"] is not None


def test_summarize_shadow_empty():
    assert _summarize_shadow([]) == {}
    assert _summarize_shadow([None, {}]) == {}


# --- _flags_diff ------------------------------------------------------------


def test_flags_diff_detects_changes():
    prev = {"A": True, "B": 0.85, "C": "x"}
    cur = {"A": False, "B": 0.85, "C": "x", "D": 1}
    diff = _flags_diff(prev, cur)
    assert diff == {"A": (True, False), "D": (None, 1)}


def test_flags_diff_handles_none():
    assert _flags_diff(None, {"A": 1}) == {"A": (None, 1)}
    assert _flags_diff({"A": 1}, None) == {"A": (1, None)}
    assert _flags_diff(None, None) == {}
