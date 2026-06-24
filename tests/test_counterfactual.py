"""Unit tests for the counterfactual mining module (self-improvement substrate).

Pure functions over synthetic joined-telemetry rows — no DB.
"""

from __future__ import annotations

from src.analysis.counterfactual import (
    THROTTLE_OUTCOMES,
    cohort_stats,
    group_cohorts,
    headline_missed,
    lead_band,
    mine_counterfactuals,
    mine_rejected,
    mine_throttled,
    price_band,
)


def _rej(direction, reason, price, yes_won, op="at_least", icao="KAUS", htp=0.0):
    return {
        "direction": direction,
        "reject_reason": reason,
        "side_price": price,
        "op": op,
        "station_icao": icao,
        "hours_until_peak": htp,
        "yes_won": yes_won,
    }


class TestPriceBand:
    def test_bands(self):
        assert price_band(0.30) == "[0.00,0.40)"
        assert price_band(0.50) == "[0.40,0.60)"
        assert price_band(0.92) == "[0.80,0.95)"
        assert price_band(0.97) == "[0.95,1.00)"
        assert price_band(1.0) == "[1.00]"
        assert price_band(None) == "?"


class TestLeadBand:
    def test_bands(self):
        assert lead_band(-1.0) == "past-peak"
        assert lead_band(0.0) == "past-peak"
        assert lead_band(1.5) == "0-2h"
        assert lead_band(4.0) == "2-6h"
        assert lead_band(9.0) == ">6h"
        assert lead_band(None) == "?"


class TestCohortStats:
    def test_won_rate_and_ev_for_rejected_no_side(self):
        # 3 rejected BUY_NO at price 0.90; 2 resolved NO (win), 1 resolved YES (loss).
        rows = [
            _rej("BUY_NO", "edge", 0.90, yes_won=False),
            _rej("BUY_NO", "edge", 0.90, yes_won=False),
            _rej("BUY_NO", "edge", 0.90, yes_won=True),
        ]
        s = cohort_stats(rows)
        assert s["n"] == 3
        assert s["n_resolved"] == 3
        assert s["would_win"] == 2
        assert abs(s["win_rate"] - 2 / 3) < 1e-9
        assert abs(s["avg_price"] - 0.90) < 1e-9
        # EV: two wins at (1-.9)/.9=+0.1111, one loss -1 → (0.2222-1)/3 = -0.2593
        assert abs(s["ev_per_dollar"] - ((2 * (0.1 / 0.9) - 1) / 3)) < 1e-6
        # break-even 0.90 > 0.667 win → negative edge
        assert s["edge_pp"] < 0

    def test_yes_side_win_semantics(self):
        # BUY_YES wins when yes_won True.
        rows = [_rej("BUY_YES", "probability", 0.50, yes_won=True),
                _rej("BUY_YES", "probability", 0.50, yes_won=True)]
        s = cohort_stats(rows)
        assert s["would_win"] == 2
        assert s["win_rate"] == 1.0
        assert s["ev_per_dollar"] > 0  # 100% win at 0.50 → +1.0/ea

    def test_unresolved_rows_excluded_from_rate(self):
        rows = [
            _rej("BUY_NO", "edge", 0.80, yes_won=False),
            _rej("BUY_NO", "edge", 0.80, yes_won=None),  # not yet resolved
        ]
        s = cohort_stats(rows)
        assert s["n"] == 2
        assert s["n_resolved"] == 1
        assert s["win_rate"] == 1.0

    def test_missing_price_skips_ev_keeps_winrate(self):
        rows = [_rej("BUY_NO", "x", None, yes_won=False),
                _rej("BUY_NO", "x", None, yes_won=True)]
        s = cohort_stats(rows)
        assert s["win_rate"] == 0.5
        assert s["ev_per_dollar"] is None
        assert s["avg_price"] is None

    def test_empty(self):
        s = cohort_stats([])
        assert s["n"] == 0 and s["win_rate"] is None and s["ev_per_dollar"] is None


class TestGroupCohorts:
    def test_groups_and_sorts_by_foregone_ev(self):
        rows = [
            # reason "price": deep-value YES that won at 0.30 → strongly +EV
            _rej("BUY_YES", "price", 0.30, yes_won=True),
            _rej("BUY_YES", "price", 0.30, yes_won=True),
            # reason "edge": NO at 0.90 that lost → -EV
            _rej("BUY_NO", "edge", 0.90, yes_won=True),
        ]
        out = group_cohorts(rows, "reject_reason")
        assert {c["reject_reason"] for c in out} == {"price", "edge"}
        # +EV "price" cohort sorts before the -EV "edge" cohort
        assert out[0]["reject_reason"] == "price"
        assert out[0]["ev_per_dollar"] > 0


class TestMineRejected:
    def test_pivots_present(self):
        rows = [_rej("BUY_NO", "edge 0.04 < 0.10", 0.88, yes_won=False, icao="KJFK"),
                _rej("BUY_YES", "probability 0.7 < 0.85", 0.45, yes_won=True, icao="KLAX")]
        out = mine_rejected(rows)
        assert out["total"] == 2
        for pivot in ("by_reason", "by_station", "by_op_class", "by_price_band"):
            assert pivot in out and isinstance(out[pivot], list)
        # op_class derived from op
        assert {c["op_class"] for c in out["by_op_class"]} == {"threshold"}


class TestMineThrottled:
    def test_filters_to_throttle_outcomes(self):
        rows = [
            {"direction": "BUY_YES", "outcome": "stake_below_min", "side_price": 0.92,
             "op": "at_least", "station_icao": "KAUS", "yes_won": True},
            {"direction": "BUY_YES", "outcome": "trade_filled", "side_price": 0.9,
             "op": "at_least", "station_icao": "KAUS", "yes_won": True},  # not a throttle
        ]
        out = mine_throttled(rows)
        assert out["total"] == 1  # trade_filled excluded
        assert "stake_below_min" in {c["outcome"] for c in out["by_outcome"]}

    def test_throttle_outcomes_constant(self):
        assert "stake_below_min" in THROTTLE_OUTCOMES
        assert "trade_filled" not in THROTTLE_OUTCOMES


class TestHeadlineMissed:
    def test_surfaces_positive_ev_cohorts_only(self):
        cohorts = [
            {"reject_reason": "price", "n_resolved": 10, "ev_per_dollar": 0.20, "edge_pp": 15.0},
            {"reject_reason": "edge", "n_resolved": 10, "ev_per_dollar": -0.10, "edge_pp": -8.0},
            {"reject_reason": "thin", "n_resolved": 2, "ev_per_dollar": 0.50, "edge_pp": 20.0},  # too few
        ]
        out = headline_missed(cohorts, min_resolved=5)
        assert [c["reject_reason"] for c in out] == ["price"]


class TestMineCounterfactuals:
    def test_end_to_end_shape(self):
        rejected = [
            _rej("BUY_YES", "probability 0.7 < 0.85", 0.40, yes_won=True),
            _rej("BUY_YES", "probability 0.7 < 0.85", 0.40, yes_won=True),
            _rej("BUY_YES", "probability 0.7 < 0.85", 0.40, yes_won=True),
            _rej("BUY_YES", "probability 0.7 < 0.85", 0.40, yes_won=True),
            _rej("BUY_YES", "probability 0.7 < 0.85", 0.40, yes_won=True),
            _rej("BUY_YES", "probability 0.7 < 0.85", 0.40, yes_won=False),
        ]
        throttled = [
            {"direction": "BUY_YES", "outcome": "stake_below_min", "side_price": 0.50,
             "op": "at_least", "station_icao": "KAUS", "yes_won": True},
        ]
        out = mine_counterfactuals(rejected, throttled, min_resolved=5)
        assert out["rejected"]["total"] == 6
        assert out["throttled"]["total"] == 1
        # the loosenable probability-floor cohort (5/6 won at 0.40 → strongly +EV) shows up
        head = out["headline"]["rejected_by_reason"]
        assert any("probability" in c["reject_reason"] for c in head)
