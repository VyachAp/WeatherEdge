"""Unit tests for the Phase 0 measurement-layer report aggregators.

The CLI report commands are thin I/O wrappers; the number-crunching is in
pure module-level functions in ``src.cli`` (no DB, no click), so they're
tested here with plain-dict inputs.
"""

from __future__ import annotations

import pytest

from src.cli import (
    _aggregate_divergence,
    _aggregate_station_day_divergence,
    _aggregate_valley,
    _flags_diff,
    _flatten_shadow,
    _quantiles,
    _reject_rollup,
    _summarize_conviction_locks,
    _summarize_exposure,
    _summarize_floored_fills,
    _summarize_forecast_error,
    _summarize_reprice,
    _summarize_shadow,
    _valley_bet_won,
    _valley_ev_per_dollar,
    aggregate_no_trade_funnel,
    render_no_trade_digest,
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


# --- _aggregate_station_day_divergence (Phase 1) ----------------------------


def test_aggregate_station_day_divergence_groups_sorts_and_abs_mean():
    rows = [
        {"station_icao": "KAUS", "unit": "F", "divergence_point_f": 1.0},
        {"station_icao": "KAUS", "unit": "F", "divergence_point_f": 3.0},
        {"station_icao": "KNYC", "unit": "F", "divergence_point_f": -0.5},
        # NULL point divergence (one-sided window) → excluded.
        {"station_icao": "KNYC", "unit": "F", "divergence_point_f": None},
    ]
    agg = _aggregate_station_day_divergence(rows)
    assert [d["station_icao"] for d in agg] == ["KAUS", "KNYC"]  # |mean| desc
    aus = agg[0]
    assert aus["n"] == 2
    assert aus["mean"] == 2.0
    assert aus["abs_mean"] == 2.0
    assert aus["min"] == 1.0
    assert aus["max"] == 3.0
    nyc = agg[1]
    assert nyc["n"] == 1
    assert nyc["mean"] == -0.5
    assert nyc["abs_mean"] == 0.5


def test_aggregate_station_day_divergence_empty():
    assert _aggregate_station_day_divergence([]) == []
    assert _aggregate_station_day_divergence(
        [{"station_icao": "K", "unit": "F", "divergence_point_f": None}]
    ) == []


# --- _summarize_reprice (Phase 2 latency) -----------------------------------


def test_summarize_reprice_empty():
    out = _summarize_reprice([])
    assert out == {"n_groups": 0, "n_measurable": 0, "overall": None,
                   "by_bucket": []}


def test_summarize_reprice_single_snapshot_not_measurable():
    # One snapshot per METAR group → no T0→last move.
    rows = [
        {"market_id": "m1", "metar_observed_at": "t0", "created_at": "c0",
         "yes_mid": 0.5, "obs_fraction": 0.8, "seconds_since_obs": 30.0},
    ]
    out = _summarize_reprice(rows)
    assert out["n_groups"] == 1
    assert out["n_measurable"] == 0
    assert out["by_bucket"] == []


def test_summarize_reprice_pairs_t0_and_later_and_buckets():
    rows = [
        # high obs_fraction group: market rose 0.50 → 0.62 toward YES.
        {"market_id": "m1", "metar_observed_at": "t0", "created_at": "c0",
         "yes_mid": 0.50, "obs_fraction": None, "seconds_since_obs": 30.0},
        {"market_id": "m1", "metar_observed_at": "t0", "created_at": "c1",
         "yes_mid": 0.62, "obs_fraction": 0.9, "seconds_since_obs": 330.0},
        # low obs_fraction group: barely moved.
        {"market_id": "m2", "metar_observed_at": "t9", "created_at": "c0",
         "yes_mid": 0.40, "obs_fraction": 0.1, "seconds_since_obs": 20.0},
        {"market_id": "m2", "metar_observed_at": "t9", "created_at": "c1",
         "yes_mid": 0.41, "obs_fraction": 0.1, "seconds_since_obs": 320.0},
    ]
    out = _summarize_reprice(rows)
    assert out["n_groups"] == 2
    assert out["n_measurable"] == 2
    # obs_fraction read from whichever row carries it (T0 was None for m1).
    buckets = {b["bucket"]: b for b in out["by_bucket"]}
    assert "high (>=0.67)" in buckets
    assert "low (<0.34)" in buckets
    assert buckets["high (>=0.67)"]["mean_move"] == pytest.approx(0.12)
    assert buckets["low (<0.34)"]["mean_move"] == pytest.approx(0.01)
    # high bucket span = 330 - 30 = 300s.
    assert buckets["high (>=0.67)"]["mean_span_seconds"] == pytest.approx(300.0)
    # Thesis-order: low before high.
    assert [b["bucket"] for b in out["by_bucket"]] == [
        "low (<0.34)", "high (>=0.67)"
    ]


# --- _summarize_forecast_error (Phase 3) ------------------------------------


def test_summarize_forecast_error_empty():
    assert _summarize_forecast_error([]) == []


def test_summarize_forecast_error_rmse_by_lead_and_sorting():
    rows = [
        # lead 0: small errors → low RMSE.
        {"lead_bucket_h": 0, "error_vs_resolved_f": 1.0,
         "error_vs_metar_f": 1.0, "forecast_sigma_f": 2.0},
        {"lead_bucket_h": 0, "error_vs_resolved_f": -1.0,
         "error_vs_metar_f": -1.0, "forecast_sigma_f": 2.0},
        # lead 24: larger errors → higher RMSE, warm bias.
        {"lead_bucket_h": 24, "error_vs_resolved_f": 3.0,
         "error_vs_metar_f": 3.0, "forecast_sigma_f": 4.0},
        {"lead_bucket_h": 24, "error_vs_resolved_f": 5.0,
         "error_vs_metar_f": 5.0, "forecast_sigma_f": 4.0},
    ]
    agg = _summarize_forecast_error(rows)
    # Sorted by lead asc.
    assert [d["lead_bucket_h"] for d in agg] == [0, 24]
    lead0, lead24 = agg
    assert lead0["mean_error"] == 0.0
    assert lead0["rmse"] == 1.0
    assert lead0["bias_sign"] == "flat"
    assert lead24["mean_error"] == 4.0
    assert lead24["rmse"] == pytest.approx((34.0 / 2) ** 0.5)
    assert lead24["bias_sign"] == "warm"
    assert lead24["mean_sigma"] == 4.0


def test_summarize_forecast_error_falls_back_to_metar_when_no_resolved():
    rows = [
        {"lead_bucket_h": 6, "error_vs_resolved_f": None,
         "error_vs_metar_f": -2.0, "forecast_sigma_f": None},
    ]
    agg = _summarize_forecast_error(rows)
    assert len(agg) == 1
    assert agg[0]["mean_error"] == -2.0
    assert agg[0]["bias_sign"] == "cool"
    assert agg[0]["mean_sigma"] is None


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


# --- _summarize_conviction_locks --------------------------------------------


def test_summarize_conviction_locks_empty():
    s = _summarize_conviction_locks([])
    assert s["fires"] == 0
    assert s["filled"] == 0 and s["throttled"] == 0
    assert s["deployed_actual"] == 0.0
    assert s["settled"] == 0 and s["net_pnl"] == 0.0
    assert s["by_branch"] == {}


def test_summarize_conviction_locks_funnel_and_settlement():
    rows = [
        # deployed + settled WON
        {"outcome": "trade_filled", "branch": "easy_super",
         "requested_stake_usd": 68.0, "actual_stake_usd": 60.0,
         "status": "WON", "pnl": 9.5},
        # deployed (dry-run pending), not settled
        {"outcome": "trade_pending", "branch": "easy_standard",
         "requested_stake_usd": 20.0, "actual_stake_usd": 20.0,
         "status": "PENDING", "pnl": None},
        # walked the book but empty → no fill
        {"outcome": "no_fill", "branch": "easy_super",
         "requested_stake_usd": 50.0, "actual_stake_usd": None,
         "status": None, "pnl": None},
        # throttled by drawdown
        {"outcome": "drawdown_paused", "branch": "easy_super",
         "requested_stake_usd": 0.0, "actual_stake_usd": 0.0,
         "status": None, "pnl": None},
        # deployed + settled LOST
        {"outcome": "trade_filled", "branch": "easy_super",
         "requested_stake_usd": 60.0, "actual_stake_usd": 55.0,
         "status": "LOST", "pnl": -55.0},
    ]
    s = _summarize_conviction_locks(rows)
    assert s["fires"] == 5
    assert s["filled"] == 3            # 2 trade_filled + 1 trade_pending
    assert s["no_fill"] == 1
    assert s["throttled"] == 1
    # deployed sums over the 3 "filled" rows only
    assert s["deployed_actual"] == pytest.approx(60.0 + 20.0 + 55.0)
    assert s["deployed_requested"] == pytest.approx(68.0 + 20.0 + 60.0)
    assert s["settled"] == 2 and s["won"] == 1 and s["lost"] == 1
    assert s["net_pnl"] == pytest.approx(9.5 - 55.0)
    assert s["by_branch"] == {"easy_super": 4, "easy_standard": 1}


# --- no-trade funnel --------------------------------------------------------

def test_reject_rollup_empty():
    r = _reject_rollup([])
    assert r["total"] == 0
    assert r["by_reason"] == {} and r["by_kind"] == {} and r["by_op_class"] == {}


def test_reject_rollup_buckets_by_prefix_kind_op_class():
    rows = [
        {"passes": True, "reject_reason": None, "signal_kind": "probability",
         "op_class": "threshold"},  # passing → ignored
        {"passes": False, "reject_reason": "edge 0.03 < 0.05",
         "signal_kind": "probability", "op_class": "threshold"},
        {"passes": False, "reject_reason": "edge 0.01 < 0.05",
         "signal_kind": "probability", "op_class": "threshold"},
        {"passes": False, "reject_reason": "bracket-like NO disabled (x)",
         "signal_kind": "probability", "op_class": "bracket-like"},
        {"passes": False, "reject_reason": None,
         "signal_kind": "lock", "op_class": None},
    ]
    r = _reject_rollup(rows)
    assert r["total"] == 4
    # prefix bucketing + descending sort (edge=2 first)
    assert list(r["by_reason"].items())[0] == ("edge", 2)
    assert r["by_reason"]["bracket-like"] == 1
    assert r["by_reason"]["(unknown)"] == 1
    assert r["by_kind"] == {"probability": 3, "lock": 1}
    assert r["by_op_class"]["threshold"] == 2


def test_aggregate_no_trade_funnel_arithmetic():
    eval_rows = [
        {"market_id": "m1", "direction": "BUY_YES", "passes": True,
         "reject_reason": None, "signal_kind": "probability", "op_class": "threshold"},
        {"market_id": "m1", "direction": "BUY_NO", "passes": False,
         "reject_reason": "edge 0.0 < 0.05", "signal_kind": "probability",
         "op_class": "threshold"},
        {"market_id": "m2", "direction": "BUY_YES", "passes": False,
         "reject_reason": "depth $3 < $10", "signal_kind": "probability",
         "op_class": "threshold"},
    ]
    decision_rows = [
        {"outcome": "trade_filled", "metadata_json": None},
        {"outcome": "stake_below_min",
         "metadata_json": {"size_reason": "depth cap"}},
        {"outcome": "stake_below_min",
         "metadata_json": {"size_reason": "depth cap"}},
    ]
    res = aggregate_no_trade_funnel(
        universe_n=5, eval_rows=eval_rows, decision_rows=decision_rows,
        dd_state={"dd_level": "NORMAL", "dd_multiplier": 1.0},
    )
    assert res["universe_n"] == 5
    assert res["evaluated_n"] == 3
    # distinct evaluated markets = {m1, m2} = 2 → never = 5 - 2 = 3
    assert res["never_evaluated_n"] == 3
    assert res["passed_n"] == 1
    assert res["rejected_n"] == 2
    assert res["traded_n"] == 1
    assert res["throttle"]["dominant_throttle"] == "stake_below_min"
    assert res["throttle"]["size_reasons"] == {"depth cap": 2}
    # funnel monotonicity over the evaluated set
    assert res["universe_n"] >= res["evaluated_n"] >= res["passed_n"] >= res["traded_n"]


def test_aggregate_no_trade_funnel_empty():
    res = aggregate_no_trade_funnel(
        universe_n=0, eval_rows=[], decision_rows=[], dd_state=None,
    )
    assert res["universe_n"] == 0 and res["evaluated_n"] == 0
    assert res["never_evaluated_n"] == 0 and res["traded_n"] == 0
    assert res["dd"] == {}


def test_render_no_trade_digest_markdown_and_plain():
    res = aggregate_no_trade_funnel(
        universe_n=10,
        eval_rows=[
            {"market_id": "m1", "direction": "BUY_NO", "passes": False,
             "reject_reason": "price 0.30 < 0.40", "signal_kind": "probability",
             "op_class": "threshold"},
        ],
        decision_rows=[{"outcome": "drawdown_paused",
                        "metadata_json": {"size_reason": "exposure"}}],
        dd_state={"dd_level": "PAUSED", "dd_multiplier": 0.0},
    )
    res["window_days"] = 1
    plain = render_no_trade_digest(res, markdown=False)
    assert "No-trade funnel" in plain and "10 markets" in plain
    assert "PAUSED" in plain
    md = render_no_trade_digest(res, markdown=True)
    # MarkdownV2 escapes the '.' and '-' in dynamic values / headline
    assert md and "\\" in md
