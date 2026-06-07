"""Unit tests for the Phase 0 measurement-layer report aggregators.

The CLI report commands are thin I/O wrappers; the number-crunching is in
pure module-level functions in ``src.cli`` (no DB, no click), so they're
tested here with plain-dict inputs.
"""

from __future__ import annotations

import pytest

from src.cli import (
    _aggregate_divergence,
    _aggregate_valley,
    _flags_diff,
    _flatten_shadow,
    _quantiles,
    _summarize_exposure,
    _summarize_floored_fills,
    _summarize_shadow,
    _valley_bet_won,
    _valley_ev_per_dollar,
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


# --- _summarize_floored_fills (NEAR_PEAK_FLOOR_UP gate) ---------------------


def test_summarize_floored_fills_splits_and_scores():
    rows = [
        # floored bucket: 1 win + 1 loss at entry 0.50 → won%=50%, EV=0.0.
        {"floored_up": True, "status": "won", "entry_price": 0.50},
        {"floored_up": True, "status": "lost", "entry_price": 0.50},
        # open is excluded from resolved.
        {"floored_up": True, "status": "open", "entry_price": None},
        # normal bucket: 1 win at 0.80 → won%=100%, EV = 1/0.8-1 = +0.25.
        {"floored_up": False, "status": "won", "entry_price": 0.80},
    ]
    out = _summarize_floored_fills(rows)
    fl, no = out["floored"], out["normal"]
    assert fl["n"] == 3 and fl["resolved"] == 2 and fl["won"] == 1
    assert fl["won_pct"] == 0.5
    assert fl["break_even"] == 0.5
    assert fl["ev_per_dollar"] == 0.0          # at break-even
    assert no["n"] == 1 and no["resolved"] == 1 and no["won"] == 1
    assert no["won_pct"] == 1.0
    assert no["ev_per_dollar"] == pytest.approx(0.25, abs=1e-9)


def test_summarize_floored_fills_empty_buckets():
    out = _summarize_floored_fills([])
    for b in (out["floored"], out["normal"]):
        assert b["n"] == 0 and b["resolved"] == 0 and b["won"] == 0
        assert b["won_pct"] is None
        assert b["break_even"] is None
        assert b["ev_per_dollar"] is None


def test_summarize_floored_fills_all_open_no_resolved():
    rows = [
        {"floored_up": True, "status": "open", "entry_price": 0.6},
        {"floored_up": True, "status": None, "entry_price": None},
    ]
    fl = _summarize_floored_fills(rows)["floored"]
    assert fl["n"] == 2
    assert fl["resolved"] == 0
    assert fl["won_pct"] is None  # gate not yet readable


# --- valley-report aggregator (P1→P2 promotion gate) ------------------------

def test_valley_bet_won_directions():
    # BUY_NO wins when market resolved NO (yes_won False).
    assert _valley_bet_won("BUY_NO", False) is True
    assert _valley_bet_won("BUY_NO", True) is False
    # BUY_YES wins when market resolved YES.
    assert _valley_bet_won("BUY_YES", True) is True
    assert _valley_bet_won("BUY_YES", False) is False


def test_valley_ev_per_dollar_breakeven_at_price():
    # Win pays (1-p)/p; loss is full -1. Break-even win rate == price, so a
    # 50/50 outcome at price 0.5 nets ~0.
    assert _valley_ev_per_dollar(0.5, True) == pytest.approx(1.0)
    assert _valley_ev_per_dollar(0.5, False) == pytest.approx(-1.0)
    assert _valley_ev_per_dollar(0.0, True) == 0.0  # guard against div-by-zero


def test_aggregate_valley_splits_and_scores():
    rows = [
        # P2 allows (high edge): 3 NO bets at 0.70, 2 win → +EV-ish
        {"direction": "BUY_NO", "price": 0.70, "yes_won": False, "p2_would_block": False},
        {"direction": "BUY_NO", "price": 0.70, "yes_won": False, "p2_would_block": False},
        {"direction": "BUY_NO", "price": 0.70, "yes_won": True, "p2_would_block": False},
        # P2 blocks (low edge): 2 NO bets at 0.70, both lose → -EV
        {"direction": "BUY_NO", "price": 0.70, "yes_won": True, "p2_would_block": True},
        {"direction": "BUY_NO", "price": 0.70, "yes_won": True, "p2_would_block": True},
    ]
    agg = _aggregate_valley(rows)
    assert agg["all_valley"]["n"] == 5
    assert agg["p2_allows"]["n"] == 3
    assert agg["p2_blocks"]["n"] == 2
    # p2_allows: 2/3 win → 66.7%, break-even 70%
    assert agg["p2_allows"]["win_pct"] == pytest.approx(66.667, abs=0.1)
    assert agg["p2_allows"]["breakeven_pct"] == pytest.approx(70.0)
    # p2_blocks all lose → EV -1.0
    assert agg["p2_blocks"]["ev_per_usd"] == pytest.approx(-1.0)
    assert agg["p2_blocks"]["win_pct"] == 0.0


def test_aggregate_valley_empty_cohorts():
    agg = _aggregate_valley([])
    for c in ("all_valley", "p2_allows", "p2_blocks"):
        assert agg[c]["n"] == 0
        assert agg[c]["win_pct"] is None
        assert agg[c]["ev_per_usd"] is None
