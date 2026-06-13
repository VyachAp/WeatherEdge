"""Unit tests for the pure calibration-report aggregator (promotion gate).

Scores shadow-ledger predictions vs the market baseline. The key property: a
model that's perfect should beat the market; one that's pure noise should not.
"""

from __future__ import annotations

from src.cli import _calibration_report, _score_cohort, _would_trade_ev


def _row(updated_yes, market_mid, yes_won, obs_fraction=0.9,
         operator_class="threshold", would_trade=False, side=None,
         effective_cost=None):
    return dict(
        updated_yes=updated_yes, market_mid=market_mid, yes_won=yes_won,
        obs_fraction=obs_fraction, operator_class=operator_class,
        would_trade=would_trade, side=side, effective_cost=effective_cost,
    )


def test_perfect_model_beats_market():
    # Model nails the outcomes; market is a coin flip.
    rows = [_row(0.99, 0.5, 1) for _ in range(10)] + [_row(0.01, 0.5, 0) for _ in range(10)]
    sc = _score_cohort(rows)
    assert sc["model_beats_market"] is True
    assert sc["model_brier"] < sc["market_brier"]


def test_noise_model_does_not_beat_market():
    # Market is well-calibrated; model is anti-correlated noise.
    rows = [_row(0.2, 0.8, 1) for _ in range(10)] + [_row(0.8, 0.2, 0) for _ in range(10)]
    sc = _score_cohort(rows)
    assert sc["model_beats_market"] is False


def test_empty_cohort_returns_none():
    assert _score_cohort([]) is None
    assert _score_cohort([_row(0.9, None, 1)]) is None  # no market price


def test_buckets_split_by_obs_fraction():
    rows = (
        [_row(0.99, 0.5, 1, obs_fraction=0.1) for _ in range(4)]
        + [_row(0.99, 0.5, 1, obs_fraction=0.9) for _ in range(4)]
    )
    rep = _calibration_report(rows)
    labels = {b["bucket"] for b in rep["buckets"]}
    assert "0.00-0.25" in labels
    assert "0.75-1.00" in labels


def test_would_trade_ev_winner():
    # A NO bet at effective_cost 0.4 that wins (yes_won=0) returns 1/0.4 - 1 = 1.5.
    rows = [_row(0.2, 0.4, 0, would_trade=True, side="NO", effective_cost=0.4)]
    wt = _would_trade_ev(rows)
    assert wt["n"] == 1
    assert wt["win_rate"] == 1.0
    assert abs(wt["ev_per_dollar"] - 1.5) < 1e-9


def test_would_trade_ev_empty():
    assert _would_trade_ev([_row(0.5, 0.5, 1)])["n"] == 0
