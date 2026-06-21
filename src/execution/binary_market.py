"""Per-market shape helpers — binary-vs-bracket detection, °F bucket grid
construction, unit display, and future-day skip logic.

These are pure functions over ``Market`` rows. No I/O, no DB, no
scheduler state — safe to call from any tick / replay / test.
"""

from __future__ import annotations

import math
from datetime import datetime


def is_binary_market(market) -> bool:
    """True if market is a single-outcome binary YES/NO market.

    All temperature markets we trade are binary at the CLOB level (one
    YES token, one NO token) — the "bracket" operator from the parser
    refers to questions like "Will the highest be between 88-89°F?" which
    are *single binary* markets asking about a 2°F window, not a multi-
    outcome bracket. We unify them here and let downstream routing pick
    threshold-vs-range handling based on market_range_f().
    """
    op = market.parsed_operator
    if op is None:
        return False
    if op in ("above", "at_least", "below", "at_most"):
        return market.parsed_threshold is not None
    if op == "exactly":
        return market.parsed_threshold is not None
    if op in ("range", "bracket"):
        return market_range_f(market) is not None
    return False


def market_range_f(market) -> tuple[int, int] | None:
    """Inclusive integer °F range for a range-style binary market.

    Recognized shapes:
      * "Will the highest temperature in X be between 88-89°F on …?"
        → (88, 89)  — a 2°F-wide window
      * "Will the highest temperature in X be 17°C on …?"
        → (62, 63)  — integer °F values that round to 17°C
      * "Will the highest temperature in X be 88°F on …?"
        → (88, 88)  — single-degree window

    Returns None for one-sided threshold markets (above/below) and for
    questions where neither pattern matches.
    """
    op = market.parsed_operator

    # 1. Explicit "between X-Y°[FC]" in the question text.
    if op in ("range", "bracket"):
        from src.ingestion.polymarket import parse_bracket_from_question

        parsed = parse_bracket_from_question(market.question or "")
        if parsed is not None:
            low_f, high_f_excl = parsed  # high is exclusive (+1)
            return (int(round(low_f)), int(round(high_f_excl)) - 1)
        return None

    # 2. "Exactly" — Celsius single-value or Fahrenheit single-value.
    if op == "exactly" and market.parsed_threshold is not None:
        if market_unit(market) == "°C":
            c_int = round((market.parsed_threshold - 32.0) * 5.0 / 9.0)
            lo_c = c_int - 0.5
            hi_c = c_int + 0.5
            lo_f = lo_c * 9.0 / 5.0 + 32.0
            hi_f = hi_c * 9.0 / 5.0 + 32.0
            f_lo = int(math.ceil(lo_f - 1e-9))
            f_hi = int(math.floor(hi_f - 1e-9))
            if f_hi < f_lo:
                return (f_lo, f_lo)
            return (f_lo, f_hi)
        f = int(round(market.parsed_threshold))
        return (f, f)

    return None


def should_skip_future_day(
    market, now: datetime, station_icao: str | None = None,
) -> bool:
    """True when the market's data day is *strictly later* than today
    in the **station's local timezone**.

    Each Polymarket weather market resolves to a single local-day max at
    the station that resolves it. A market for tomorrow's local data
    still has no observations — skip. A market for today's local data,
    or yesterday's that hasn't yet been settled, stays in scope.

    Falls back to the legacy UTC-date comparison when no station is
    available — keeps the rule defined for callers that don't know the
    station yet.
    """
    if not market.end_date:
        return False
    if station_icao is None:
        return market.end_date.date() > now.date()

    from src.signals.mapper import (
        icao_timezone,
        resolve_target_local_day,
        today_local,
    )
    tz = icao_timezone(station_icao)
    target = resolve_target_local_day(market.end_date, tz)
    if target is None:
        return False
    return target > today_local(tz)


def market_unit(market) -> str:
    """Return '°C' or '°F' based on the market's original question text."""
    q = (market.question or "").upper()
    if "°C" in q or "CELSIUS" in q:
        return "°C"
    return "°F"


def display_bucket(bucket_f: int, unit: str) -> int:
    """Convert an internal °F bucket to the market's display unit, rounded."""
    if unit == "°C":
        return round((bucket_f - 32) * 5 / 9)
    return bucket_f


def make_binary_buckets(market, state) -> list[int]:
    """Generate the integer °F bucket grid for a single-binary market.

    The grid spans from one degree below the observed max up through the
    forecast peak (or the threshold/range upper bound, whichever is
    higher) plus a 10°F headroom — wide enough to capture upside tails
    without wasting compute on far-out buckets.
    """
    rng = market_range_f(market)
    if rng is not None:
        upper = max(rng[1], int(state.forecast_peak_f))
    elif market.parsed_threshold is not None:
        upper = max(int(market.parsed_threshold), int(state.forecast_peak_f))
    else:
        upper = int(state.forecast_peak_f)
    low = int(state.current_max_f) - 1
    return list(range(low, upper + 11))


_THRESHOLD_OPS: frozenset[str] = frozenset({"above", "at_least", "below", "at_most"})
_BRACKET_LIKE_OPS: frozenset[str] = frozenset({"bracket", "range", "exactly"})


def is_bracket_like(market) -> bool:
    """True for multi-bucket window markets (bracket / range / exactly).

    Used by the bracket-disable gate and the cluster-cap helper. Threshold
    markets (above/at_least/below/at_most) are NOT bracket-like — they're
    a single binary outcome, no anti-correlated buckets.
    """
    return market.parsed_operator in _BRACKET_LIKE_OPS


def operator_class(market_or_op) -> str | None:
    """Canonical operator-class label for filter/calibration routing.

    Returns ``"threshold"`` for one-sided binary thresholds, ``"bracket-like"``
    for the single-bucket window family, and ``None`` for unparsed / unknown
    operators. Accepts either a ``Market`` row (reads ``parsed_operator``) or
    a raw operator string. Single source-of-truth so the same split is shared
    by ``cli.py`` (telemetry reports), ``calibration.py`` (per-class fit
    selection), and ``edge_calculator.py`` (per-class filter floors).
    """
    op = (
        getattr(market_or_op, "parsed_operator", None)
        if not isinstance(market_or_op, str)
        else market_or_op
    )
    if op in _THRESHOLD_OPS:
        return "threshold"
    if op in _BRACKET_LIKE_OPS:
        return "bracket-like"
    return None


def near_peak_floor_eligible(
    market,
    *,
    our_probability: float,
    hours_until_peak: float | None,
) -> bool:
    """True if a passing edge may have its sub-min stake floored up.

    Gates the near-peak floor-up (``NEAR_PEAK_FLOOR_UP_ENABLED``). Restricted
    to **threshold** operators — ``exactly``/``range``/``bracket`` are excluded
    (validate-first: that single-bucket NO class is historically a net loser;
    it still trades when Kelly sizes ≥ ``MIN_STAKE_USD`` naturally, we just
    don't artificially floor it up yet). The trade must also be either
    high-confidence (recorded prob ≥ ``NEAR_PEAK_FLOOR_UP_MIN_PROB`` — admits
    an extreme-threshold bet decided by the forecast hours before peak) OR
    genuinely near the forecast peak (``|hours_until_peak| ≤
    NEAR_PEAK_FLOOR_UP_MAX_HOURS``). The OR keeps the "close around the peak"
    principle while still admitting forecast-decisive far-from-peak threshold
    bets.
    """
    # Local import avoids a module-load-time settings dependency in this
    # otherwise-pure shape-helper module.
    from src.config import settings

    if not settings.NEAR_PEAK_FLOOR_UP_ENABLED:
        return False
    if is_bracket_like(market):
        return False
    high_conf = our_probability >= settings.NEAR_PEAK_FLOOR_UP_MIN_PROB
    near_peak = (
        hours_until_peak is not None
        and abs(hours_until_peak) <= settings.NEAR_PEAK_FLOOR_UP_MAX_HOURS
    )
    return high_conf or near_peak


def near_lock_conviction_eligible(
    market,
    *,
    direction,
    anchored_max_f: float | None,
    has_forecast: bool,
) -> bool:
    """True if a passing edge is a genuine observational monotonic lock that may
    bypass ``KELLY_PROB_CAP`` (conviction sizing).

    Gates ``NEAR_LOCK_CONVICTION_SIZING_ENABLED``. Eligible ONLY when the bet has
    already won given the **target-day-anchored** observed max and monotonicity
    (the daily max only rises) — the "already hot, betting hot" direction:

    - ``BUY_YES`` on ``at_least``/``above`` (YES wins when the max is high), or
    - ``BUY_NO`` on ``at_most``/``below`` (NO wins when the max exceeds the bound),

    with ``anchored_max_f`` clearing the threshold by
    ``NEAR_LOCK_CONVICTION_MARGIN_F`` (hedges resolver/station divergence — cf. the
    RCTP exclusion). ``has_forecast`` is required so degenerate Open-Meteo-failure
    states (forecast_peak==current_max, hours_until_peak==0) cannot slip through —
    those are exactly the bets the HARD-lock path refuses, and the original
    prob/hours proxy fired 96% on them (2026-06-21 rebuild). Bracket-like excluded.
    The forecast direction (NO-on-``at_least`` / YES-on-``at_most``) is NOT eligible:
    it is σ-collapse overconfidence and the side resolver divergence hurts.

    ``anchored_max_f`` is the caller's ``_market_daily_max`` result in °F (matching
    ``market.parsed_threshold``); ``direction`` is a ``TradeDirection`` (or its
    ``.value`` string).
    """
    from src.config import settings

    if not settings.NEAR_LOCK_CONVICTION_SIZING_ENABLED:
        return False
    if not has_forecast:
        return False
    if is_bracket_like(market):
        return False
    op = getattr(market, "parsed_operator", None)
    thr = getattr(market, "parsed_threshold", None)
    if thr is None or anchored_max_f is None:
        return False
    dir_val = getattr(direction, "value", direction)
    bets_high = (dir_val == "BUY_YES" and op in ("at_least", "above")) or (
        dir_val == "BUY_NO" and op in ("at_most", "below")
    )
    if not bets_high:
        return False
    return anchored_max_f >= thr + settings.NEAR_LOCK_CONVICTION_MARGIN_F
