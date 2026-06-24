"""Shadow-ledger calibration / promotion-gate aggregators (M1)."""

from __future__ import annotations

from src.analysis.common import _brier, _logloss

_OBS_BINS: tuple[tuple[str, float, float], ...] = (
    ("0.00-0.25", 0.00, 0.25),
    ("0.25-0.50", 0.25, 0.50),
    ("0.50-0.75", 0.50, 0.75),
    ("0.75-1.00", 0.75, 1.0001),  # include the past-peak 1.0 rows
)


def _score_cohort(rows: list[dict]) -> dict | None:
    """Brier/log-loss of our prob vs the market baseline for one cohort.

    Each row needs ``updated_yes`` (our P(YES)), ``market_mid`` (the market's
    implied P(YES)), and ``yes_won`` (the 0/1 label). The promotion gate is
    ``model_beats_market``: our scores strictly lower (better) than just trusting
    the price. Returns None for an empty cohort.
    """
    labeled = [r for r in rows if r.get("yes_won") is not None
               and r.get("market_mid") is not None]
    if not labeled:
        return None
    n = len(labeled)
    mb = sum(_brier(float(r["updated_yes"]), int(r["yes_won"])) for r in labeled) / n
    kb = sum(_brier(float(r["market_mid"]), int(r["yes_won"])) for r in labeled) / n
    ml = sum(_logloss(float(r["updated_yes"]), int(r["yes_won"])) for r in labeled) / n
    kl = sum(_logloss(float(r["market_mid"]), int(r["yes_won"])) for r in labeled) / n
    return {
        "n": n,
        "model_brier": mb, "market_brier": kb,
        "model_logloss": ml, "market_logloss": kl,
        "model_beats_market": (mb < kb and ml < kl),
    }


def _would_trade_ev(rows: list[dict]) -> dict:
    """Realised win-rate + EV/$1 of the shadow bets the model WOULD have placed.

    A bet wins iff its chosen side matches the outcome (YES↔yes_won). Profit per
    $1 staked = ``1/price - 1`` on a win, ``-1`` on a loss, where ``price`` is the
    modelled all-in ``effective_cost``. Empty → ``{"n": 0}``.
    """
    bets = [r for r in rows if r.get("would_trade") and r.get("yes_won") is not None
            and r.get("side") and r.get("effective_cost")]
    if not bets:
        return {"n": 0}
    wins, ev = 0, 0.0
    for r in bets:
        won = (r["side"] == "YES") == bool(r["yes_won"])
        price = min(0.999, max(0.001, float(r["effective_cost"])))
        ev += (1.0 / price - 1.0) if won else -1.0
        wins += 1 if won else 0
    return {"n": len(bets), "win_rate": wins / len(bets), "ev_per_dollar": ev / len(bets)}


def _calibration_report(rows: list[dict]) -> dict:
    """Promotion-gate aggregation for the shadow ledger (pure).

    Buckets every labeled shadow prediction by observation-fraction and scores
    our probability against the market baseline (Brier + log-loss). The
    information-latency thesis predicts ``model_beats_market`` should turn True
    as ``obs_fraction`` rises (we know more than the market near peak) and stay
    False far from peak (forecast disagreement = noise). Also splits by operator
    class and reports the EV of the bets the model would have placed.
    """
    overall = _score_cohort(rows)
    buckets = []
    for label, lo, hi in _OBS_BINS:
        cohort = [r for r in rows if lo <= float(r.get("obs_fraction") or 0.0) < hi]
        sc = _score_cohort(cohort)
        if sc is not None:
            sc["bucket"] = label
            buckets.append(sc)
    by_class: dict[str, dict] = {}
    for cls in ("threshold", "bracket-like"):
        sc = _score_cohort([r for r in rows if r.get("operator_class") == cls])
        if sc is not None:
            by_class[cls] = sc
    return {
        "n_total": len(rows),
        "overall": overall,
        "buckets": buckets,
        "by_class": by_class,
        "would_trade": _would_trade_ev(rows),
    }
