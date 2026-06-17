"""Unit tests for the perf-review aggregators (Layer 1 of the auto-eval loop).

The `perf-review` command + `job_perf_review` are thin I/O wrappers; the
number-crunching is in pure module-level functions in ``src.cli`` (no DB, no
click), tested here with plain-dict inputs — mirroring ``test_cli_reports``.
"""

from __future__ import annotations

import pytest

from src.cli import (
    _flip_gate_verdicts,
    _is_phantom_lost,
    _loss_classes,
    _realized_pnl,
    _throttle_breakdown,
    _window_regression,
    render_perf_digest,
)


# --- _realized_pnl (phantom-LOST-safe) -------------------------------------


def test_realized_pnl_excludes_phantom_and_dry_run():
    rows = [
        {"status": "won", "pnl": 10.0, "stake_usd": 20.0, "fill_price": 0.5,
         "exchange_status": None, "signal_kind": "probability", "lock_branch": None},
        {"status": "lost", "pnl": -5.0, "stake_usd": 5.0, "fill_price": 0.6,
         "exchange_status": None, "signal_kind": "lock", "lock_branch": "easy_standard"},
        # phantom-LOST (no fill) — excluded entirely.
        {"status": "lost", "pnl": None, "stake_usd": 5.0, "fill_price": None,
         "exchange_status": None, "signal_kind": "lock", "lock_branch": "hard"},
        # open — counts toward n_open, not resolved.
        {"status": "open", "pnl": None, "stake_usd": 8.0, "fill_price": 0.4,
         "exchange_status": None, "signal_kind": "probability", "lock_branch": None},
        # dry-run — excluded (stake is requested, never at risk).
        {"status": "won", "pnl": 99.0, "stake_usd": 50.0, "fill_price": 0.5,
         "exchange_status": "dry_run", "signal_kind": "probability", "lock_branch": None},
    ]
    out = _realized_pnl(rows)
    assert out["n"] == 2
    assert out["n_won"] == 1
    assert out["win_pct"] == 0.5
    assert out["pnl"] == pytest.approx(5.0)
    assert out["staked"] == pytest.approx(25.0)
    assert out["ev_per_usd"] == pytest.approx(0.2)
    assert out["n_open"] == 1
    assert out["by_kind"]["probability"]["pnl"] == pytest.approx(10.0)
    assert out["by_kind"]["lock"]["won"] == 0
    assert out["by_branch"]["easy_standard"]["n"] == 1
    # phantom/hard branch never made it into a resolved group.
    assert "hard" not in out["by_branch"]


def test_realized_pnl_empty():
    out = _realized_pnl([])
    assert out["n"] == 0
    assert out["win_pct"] is None
    assert out["ev_per_usd"] is None
    assert out["pnl"] == 0.0


def test_is_phantom_lost():
    assert _is_phantom_lost({"status": "lost", "fill_price": None}) is True
    assert _is_phantom_lost({"status": "lost", "fill_price": 0.5}) is False
    assert _is_phantom_lost({"status": "won", "fill_price": None}) is False


# --- _throttle_breakdown ----------------------------------------------------


def test_throttle_breakdown_dominant_and_size_reasons():
    rows = [
        {"outcome": "trade_filled", "metadata_json": None},
        {"outcome": "stake_below_min", "metadata_json": {"size_reason": "MIN_TRADE_USD floor"}},
        {"outcome": "stake_below_min", "metadata_json": {"size_reason": "depth"}},
        {"outcome": "drawdown_paused", "metadata_json": {"size_reason": "paused"}},
        {"outcome": "trade_filled", "metadata_json": {"floored_up": True}},
    ]
    out = _throttle_breakdown(rows)
    assert out["total"] == 5
    assert out["outcomes"]["trade_filled"] == 2
    # stake_below_min (2) is the dominant *throttle* (trade_filled is a success).
    assert out["dominant_throttle"] == "stake_below_min"
    assert out["dominant_throttle_frac"] == pytest.approx(0.4)
    assert out["n_throttled"] == 3  # 2 stake_below_min + 1 drawdown_paused
    assert out["size_reasons"]["depth"] == 1
    assert out["floored_up_n"] == 1


def test_throttle_breakdown_empty():
    out = _throttle_breakdown([])
    assert out["total"] == 0
    assert out["dominant_throttle"] is None
    assert out["dominant_throttle_frac"] is None
    assert out["floored_up_n"] == 0


# --- _loss_classes ----------------------------------------------------------


def test_loss_classes_min_n_and_sort():
    def _t(kind, branch, pnl):
        return {"status": "lost" if pnl < 0 else "won", "pnl": pnl,
                "stake_usd": 5.0, "fill_price": 0.5,
                "exchange_status": None, "signal_kind": kind, "lock_branch": branch}

    rows = (
        [_t("lock", "range_undershoot", -2.0) for _ in range(3)]   # -6 total
        + [_t("lock", "easy_standard", 1.0) for _ in range(3)]     # +3 total
        + [_t("probability", None, -9.0) for _ in range(2)]        # n<3 → dropped
    )
    out = _loss_classes(rows, min_n=3)
    assert [(c["lock_branch"], c["pnl"]) for c in out] == [
        ("range_undershoot", -6.0), ("easy_standard", 3.0),
    ]
    # the n=2 probability cohort is excluded by min_n.
    assert all(c["n"] >= 3 for c in out)


# --- _window_regression -----------------------------------------------------


def test_window_regression_flags_worse_metrics():
    cur = {"pnl": -10.0, "win_pct": 0.5, "ev_per_usd": -0.1, "volume": 5.0}
    prev = {"pnl": 5.0, "win_pct": 0.6, "ev_per_usd": 0.1, "volume": 8.0}
    out = _window_regression(cur, prev)
    assert out["pnl"]["regressed"] is True
    assert out["pnl"]["delta"] == pytest.approx(-15.0)
    assert out["pnl"]["pct_change"] == pytest.approx(-3.0)
    assert out["volume"]["regressed"] is True
    assert out["ev_per_usd"]["regressed"] is True


def test_window_regression_none_is_not_regressed():
    out = _window_regression(
        {"pnl": 1.0, "win_pct": None, "ev_per_usd": None, "volume": 0.0},
        {"pnl": 1.0, "win_pct": 0.5, "ev_per_usd": 0.1, "volume": 0.0},
    )
    assert out["win_pct"]["regressed"] is False
    assert out["win_pct"]["delta"] is None
    # equal pnl → not regressed; prev volume 0 → pct_change None.
    assert out["pnl"]["regressed"] is False
    assert out["volume"]["pct_change"] is None


# --- _flip_gate_verdicts ----------------------------------------------------


def _verdict(gates, flag):
    return next(g for g in gates if g["flag"] == flag)


def test_flip_gates_empty_ctx_is_insufficient():
    gates = _flip_gate_verdicts({"flags": {}})
    for flag in ("BRACKET_LIKE_NO_DISABLED", "SIGMA_FLOOR_LEAD_TIME_ENABLED",
                 "PER_OPERATOR_CALIBRATION_ENABLED", "VALLEY_MIN_EDGE",
                 "THRESHOLD_MIN_PROBABILITY"):
        assert _verdict(gates, flag)["verdict"] == "insufficient-data"
    # near-peak floor-up is a standing not-ready note.
    assert _verdict(gates, "NEAR_PEAK_FLOOR_UP_ENABLED")["verdict"] == "not-ready"


def test_flip_gate_bracket_like_ready_when_positive_ev():
    gates = _flip_gate_verdicts({
        "flags": {"BRACKET_LIKE_NO_DISABLED": True},
        "bracket_like_no": {"n": 40, "ev_per_usd": 0.04},
    })
    g = _verdict(gates, "BRACKET_LIKE_NO_DISABLED")
    assert g["verdict"] == "ready"
    assert g["proposed_value"] is False
    # below the n floor → insufficient regardless of EV.
    gates2 = _flip_gate_verdicts({
        "flags": {}, "bracket_like_no": {"n": 10, "ev_per_usd": 0.5}})
    assert _verdict(gates2, "BRACKET_LIKE_NO_DISABLED")["verdict"] == "insufficient-data"


def test_flip_gate_per_operator_calibration_slope_guard():
    base = {"flags": {}, "cal": {"n": 60, "delta_p50": 0.02, "slope": 1.0}}
    assert _verdict(_flip_gate_verdicts(base), "PER_OPERATOR_CALIBRATION_ENABLED")["verdict"] == "ready"
    # degenerate slope (out of [0.2, 2.0]) → not-ready even with good delta + n.
    bad = {"flags": {}, "cal": {"n": 60, "delta_p50": 0.02, "slope": 3.64}}
    assert _verdict(_flip_gate_verdicts(bad), "PER_OPERATOR_CALIBRATION_ENABLED")["verdict"] == "not-ready"


def test_flip_gate_valley_ready_when_allows_beats_blocks():
    ctx = {"flags": {}, "valley": {"p2_allows_n": 12, "p2_allows_ev": 0.08,
                                   "p2_blocks_ev": -0.2, "p2_min_edge": 0.15}}
    g = _verdict(_flip_gate_verdicts(ctx), "VALLEY_MIN_EDGE")
    assert g["verdict"] == "ready"
    assert g["proposed_value"] == 0.15


def test_flip_gate_threshold_blocked_by_stake_below_min():
    ctx = {"flags": {}, "threshold_loosen": {
        "stake_below_min_dominant": True, "recoverable_ev": 0.5, "recoverable_n": 100}}
    g = _verdict(_flip_gate_verdicts(ctx), "THRESHOLD_MIN_PROBABILITY")
    # Even with a +EV recoverable band, a dominant stake_below_min throttle
    # means loosening won't add fills → not-ready.
    assert g["verdict"] == "not-ready"


def test_flip_gate_sigma_conjunctive_with_bracket_like():
    # σ arm widens far-from-peak, but bracket-like baseline not yet +EV → not-ready.
    ctx = {"flags": {}, "sigma": {"n": 50, "delta_p50": 0.4},
           "bracket_like_no": {"n": 40, "ev_per_usd": -0.02}}
    assert _verdict(_flip_gate_verdicts(ctx), "SIGMA_FLOOR_LEAD_TIME_ENABLED")["verdict"] == "not-ready"
    # both conditions met → ready.
    ctx2 = {"flags": {}, "sigma": {"n": 50, "delta_p50": 0.4},
            "bracket_like_no": {"n": 40, "ev_per_usd": 0.03}}
    assert _verdict(_flip_gate_verdicts(ctx2), "SIGMA_FLOOR_LEAD_TIME_ENABLED")["verdict"] == "ready"


# --- render_perf_digest -----------------------------------------------------


def _sample_result():
    return {
        "window_days": 7,
        "generated_at": "2026-06-08T22:30:00+00:00",
        "pnl": {"n": 12, "n_won": 9, "win_pct": 0.75, "pnl": 34.5,
                "staked": 120.0, "ev_per_usd": 0.29, "n_open": 4,
                "by_kind": {}, "by_branch": {}},
        "regression": {
            "pnl": {"cur": 34.5, "prev": 40.0, "delta": -5.5,
                    "pct_change": -0.14, "regressed": True},
            "win_pct": {"cur": 0.75, "prev": 0.7, "delta": 0.05,
                        "pct_change": 0.07, "regressed": False},
            "ev_per_usd": {"cur": 0.29, "prev": 0.3, "delta": -0.01,
                           "pct_change": -0.03, "regressed": True},
            "volume": {"cur": 12.0, "prev": 10.0, "delta": 2.0,
                       "pct_change": 0.2, "regressed": False},
        },
        "throttle": {"total": 50, "outcomes": {"trade_filled": 12, "stake_below_min": 30},
                     "dominant_throttle": "stake_below_min",
                     "dominant_throttle_frac": 0.6, "n_throttled": 30,
                     "size_reasons": {"depth": 30}, "floored_up_n": 0},
        "loss_classes": [
            {"signal_kind": "lock", "lock_branch": "range_undershoot", "n": 4,
             "won": 1, "win_pct": 0.25, "pnl": -12.3, "ev_per_usd": -0.4},
        ],
        "exposure": {"count": 200, "low_headroom_frac": 0.1, "n_open_median": 4.0},
        "epoch": {"current_id": 7, "started_at": "2026-06-01T00:00:00+00:00",
                  "flags_diff": {}},
        "flip_gates": [
            {"flag": "VALLEY_MIN_EDGE", "current_value": None, "verdict": "ready",
             "measured": 0.08, "threshold": "p2_allows EV>0", "proposed_value": 0.15,
             "evidence": "x", "sample_n": 12},
            {"flag": "BRACKET_LIKE_NO_DISABLED", "current_value": True,
             "verdict": "insufficient-data", "measured": None, "threshold": "EV>0",
             "proposed_value": False, "evidence": "y", "sample_n": 3},
        ],
        "anomalies": [
            {"kind": "stuck_open", "detail": "2 OPEN past end_date",
             "action": "admin reconcile-stuck"},
        ],
    }


def test_render_perf_digest_markdown_and_plain():
    res = _sample_result()
    md = render_perf_digest(res, markdown=True)
    plain = render_perf_digest(res, markdown=False)
    for text in (md, plain):
        assert "Perf review" in text
        assert "Flip-ready" in text
        assert "VALLEY" in text  # MD2 escapes the underscores → "VALLEY\\_MIN\\_EDGE"
        assert "stuck" in text
    # Plain keeps the raw flag name + dollar sign; MD2 escapes punctuation.
    assert "VALLEY_MIN_EDGE" in plain
    assert "$" in plain
    assert "\\$" in md or "\\." in md


def test_render_perf_digest_since_window_label():
    # Lever 1: --since windows surface the absolute start in the header so the
    # operator knows they're reading the live book, not the rolling window.
    res = _sample_result()
    res["window_days"] = 18
    res["since"] = "2026-05-30T00:00:00+00:00"
    res["window_start"] = "2026-05-30T00:00:00+00:00"
    out = render_perf_digest(res, markdown=False)
    assert "since 2026-05-30" in out
    # Absent `since` → no "since" in the header line.
    res2 = _sample_result()
    assert "since" not in render_perf_digest(res2, markdown=False).splitlines()[0]


def test_render_perf_digest_no_ready_flags():
    res = _sample_result()
    for g in res["flip_gates"]:
        g["verdict"] = "insufficient-data"
    out = render_perf_digest(res, markdown=False)
    assert "No flags flip-ready" in out


def test_render_perf_digest_empty_window_is_nonempty():
    empty = {
        "window_days": 1, "generated_at": "x",
        "pnl": {"n": 0, "n_won": 0, "win_pct": None, "pnl": 0.0,
                "staked": 0.0, "ev_per_usd": None, "n_open": 0,
                "by_kind": {}, "by_branch": {}},
        "regression": {}, "throttle": {"total": 0, "outcomes": {},
                                       "dominant_throttle": None,
                                       "dominant_throttle_frac": None,
                                       "n_throttled": 0, "size_reasons": {},
                                       "floored_up_n": 0},
        "loss_classes": [], "exposure": {"count": 0},
        "epoch": {"current_id": None, "started_at": None, "flags_diff": {}},
        "flip_gates": [], "anomalies": [],
    }
    out = render_perf_digest(empty, markdown=True)
    assert out  # non-empty even with no data
    assert "Perf review" in out
