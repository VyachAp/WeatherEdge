"""Deterministic "lock-rule" trade decisions.

A complement to the probability engine: instead of estimating a probability
distribution and looking for edge, look for physical conditions that make the
outcome of a binary "daily max temperature" market essentially decided.

The primary edge: routine METARs are the resolution source, published 1-2h
before the Wunderground UI updates. If the current observed daily max already
clears the threshold by a safe margin, YES is mathematically locked because
daily max is monotonic.

The rules intentionally produce a boolean decision + reasons — no probability,
no Kelly. Sizing uses a fixed fraction of bankroll (see `risk.kelly.size_locked_position`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from src.config import settings
from src.db.models import TradeDirection
from src.signals.mapper import icao_timezone, resolve_target_local_day
from src.signals.state_aggregator import WeatherState


Side = Literal["YES", "NO"]


@dataclass
class LockDecision:
    """Result of evaluating physical lock conditions against a market."""

    side: Side | None
    reasons: list[str] = field(default_factory=list)
    margin_f: float = 0.0

    @property
    def direction(self) -> TradeDirection | None:
        if self.side == "YES":
            return TradeDirection.BUY_YES
        if self.side == "NO":
            return TradeDirection.BUY_NO
        return None


_NO_LOCK = LockDecision(side=None)


def _market_daily_max(
    state: WeatherState,
    market_end_date: datetime,
    now_utc: datetime,
) -> tuple[float | None, int]:
    """Compute max + count of routine METARs for the market's target day.

    Target day = ``resolve_target_local_day(market_end_date, station_tz)``,
    which canonically lands on the LOCAL day whose max the market is
    asking about (Polymarket's question-text "April 26" is a UTC-based
    label and does not necessarily mean Apr 26 local). The window is the
    full local calendar day, capped at ``min(market_end_date, now_utc)``.

    Returns (None, 0) when no METARs fall in the window (e.g. market
    hasn't started yet, or station has no recent data).
    """
    from datetime import time as _time

    tz = icao_timezone(state.station_icao)
    target_day = resolve_target_local_day(market_end_date, tz)
    if target_day is None:
        return None, 0

    local_day_start = datetime.combine(target_day, _time.min, tzinfo=tz)
    local_day_end = datetime.combine(target_day, _time.max, tzinfo=tz)
    utc_start = local_day_start.astimezone(timezone.utc)
    local_end_cap = local_day_end.astimezone(timezone.utc)
    utc_end = min(market_end_date, now_utc, local_end_cap)
    if utc_end <= utc_start:
        return None, 0

    temps = [t for (obs, t) in state.routine_history if utc_start <= obs <= utc_end]
    if not temps:
        return None, 0
    return max(temps), len(temps)


def _no_more_heating(state: WeatherState, threshold_f: float) -> tuple[bool, list[str]]:
    """Forward-looking evidence that the daily max will not climb to threshold.

    Requires forecast peak below threshold AND at least one past-peak signal.
    """
    reasons: list[str] = []
    if not state.has_forecast:
        # Hard-direction locks depend on trustworthy forecast evidence. When
        # forecast data is unavailable, fall back to "no lock" rather than
        # opportunistically decide from partial state.
        return False, ["forecast unavailable"]

    if state.forecast_peak_f >= threshold_f:
        return False, [
            f"forecast peak {state.forecast_peak_f:.1f}°F >= threshold {threshold_f:.1f}°F",
        ]

    past_peak = state.solar_declining or state.metar_trend_rate <= 0.0
    if not past_peak:
        return False, [
            f"no past-peak signal (solar_declining={state.solar_declining}, "
            f"trend={state.metar_trend_rate:+.1f}°F/hr)",
        ]

    reasons.append(f"forecast peak {state.forecast_peak_f:.1f}°F < threshold {threshold_f:.1f}°F")
    if state.solar_declining:
        reasons.append(f"solar declining ({state.solar_decline_magnitude:.0%})")
    if state.metar_trend_rate <= 0.0:
        reasons.append(f"temp trend {state.metar_trend_rate:+.1f}°F/hr")
    return True, reasons


def evaluate_lock(
    state: WeatherState,
    market,
    now_utc: datetime | None = None,
) -> LockDecision:
    """Decide whether the market's outcome is physically locked.

    Returns LockDecision(side=None) when no lock conditions are met.

    Supported market shapes:
      * Threshold (above / at_least / below / at_most): EASY (observed max
        already clears threshold by margin) and HARD (observed max well
        below threshold + no_more_heating) directions.
      * Range (range / bracket-with-parseable-window / exactly): NO lock
        when observed max overshoots high+margin or undershoots low-margin
        with no_more_heating; YES lock when current_max sits inside
        [low, high] AND past peak with no upward signals.
    """
    if market.parsed_operator is None or market.end_date is None:
        return _NO_LOCK
    # Defense-in-depth: drop lowest/minimum-temperature markets even if a
    # stale Market row from a pre-fix ingestion still has parsed_operator set.
    # Primary filter is in polymarket.parse_question.
    q = (getattr(market, "question", None) or "").lower()
    if "lowest temperature" in q or "minimum temperature" in q:
        return _NO_LOCK
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    op = market.parsed_operator
    margin = settings.LOCK_MARGIN_F

    current_max_f, routine_count = _market_daily_max(state, market.end_date, now_utc)
    if current_max_f is None:
        return _NO_LOCK
    # Hard floor: even a super-margin lock needs at least 2 routines so the
    # max isn't a single-METAR fluke. Standard EASY/HARD still need
    # `MIN_ROUTINE_COUNT` (3 by default) — see per-branch gates below.
    if routine_count < 2:
        return _NO_LOCK

    # ---- Range / "exactly" markets ----
    if op in ("range", "bracket", "exactly"):
        from src.scheduler import market_range_f

        if routine_count < settings.MIN_ROUTINE_COUNT:
            return _NO_LOCK
        rng = market_range_f(market)
        if rng is None:
            return _NO_LOCK
        return _evaluate_range_lock(
            state=state, market=market,
            low_f=float(rng[0]), high_f=float(rng[1]),
            current_max_f=current_max_f, routine_count=routine_count,
            margin=margin, op=op,
        )

    # ---- Threshold markets ----
    if market.parsed_threshold is None:
        return _NO_LOCK
    threshold_f = float(market.parsed_threshold)

    # EASY: observed max already clears threshold by margin (monotonic lock).
    # Super-margin (>= 2× LOCK_MARGIN_F) is mathematically bulletproof at
    # routine_count=2 — daily max is monotonic, so two confirming obs
    # already overshooting by 4°F can't be undone by a third.
    super_margin = 2.0 * margin
    easy_overshoot = current_max_f - threshold_f
    easy_min_routines = (
        2 if easy_overshoot >= super_margin else settings.MIN_ROUTINE_COUNT
    )
    if op in ("above", "at_least"):
        if (
            current_max_f >= threshold_f + margin
            and routine_count >= easy_min_routines
        ):
            return LockDecision(
                side="YES",
                reasons=[
                    f"market-day max {current_max_f:.1f}°F >= threshold "
                    f"{threshold_f:.0f}°F + {margin:.1f}°F margin ({op})",
                    f"routine_count={routine_count} (min {easy_min_routines})",
                ],
                margin_f=easy_overshoot,
            )
    elif op in ("below", "at_most"):
        if (
            current_max_f >= threshold_f + margin
            and routine_count >= easy_min_routines
        ):
            return LockDecision(
                side="NO",
                reasons=[
                    f"market-day max {current_max_f:.1f}°F exceeds threshold "
                    f"{threshold_f:.0f}°F by {margin:.1f}°F ({op} already violated)",
                    f"routine_count={routine_count} (min {easy_min_routines})",
                ],
                margin_f=easy_overshoot,
            )
    else:
        return _NO_LOCK

    # HARD: max is well below threshold AND no more heating possible.
    # Forecast-dependent — requires the standard MIN_ROUTINE_COUNT.
    if routine_count < settings.MIN_ROUTINE_COUNT:
        return _NO_LOCK
    if current_max_f >= threshold_f - margin:
        return _NO_LOCK
    has_headroom, evidence = _no_more_heating(state, threshold_f)
    if not has_headroom:
        return _NO_LOCK

    side: Side = "NO" if op in ("above", "at_least") else "YES"
    return LockDecision(
        side=side,
        reasons=[
            f"market-day max {current_max_f:.1f}°F < threshold "
            f"{threshold_f:.0f}°F - {margin:.1f}°F margin ({op})",
            *evidence,
            f"routine_count={routine_count}",
        ],
        margin_f=threshold_f - current_max_f,
    )


def _evaluate_range_lock(
    *,
    state: WeatherState,
    market,
    low_f: float,
    high_f: float,
    current_max_f: float,
    routine_count: int,
    margin: float,
    op: str,
) -> LockDecision:
    """Lock evaluation for [low, high] range / single-value-exactly markets.

    Three deterministic outcomes:
      * NO overshoot — observed max already > high + margin.
      * NO undershoot — observed max << low AND no more heating possible.
      * YES in-range — observed max inside [low, high] AND past peak AND
        no upward signal (solar declining + flat/falling METAR trend) AND
        forecast peak does not exceed high.
    """
    # NO overshoot.
    if current_max_f > high_f + margin:
        return LockDecision(
            side="NO",
            reasons=[
                f"market-day max {current_max_f:.1f}°F > range upper "
                f"{high_f:.0f}°F + {margin:.1f}°F margin ({op})",
                f"routine_count={routine_count}",
            ],
            margin_f=current_max_f - high_f,
        )

    # NO undershoot.
    if current_max_f < low_f - margin:
        has_headroom, evidence = _no_more_heating(state, low_f)
        if has_headroom:
            return LockDecision(
                side="NO",
                reasons=[
                    f"market-day max {current_max_f:.1f}°F < range lower "
                    f"{low_f:.0f}°F - {margin:.1f}°F margin ({op})",
                    *evidence,
                    f"routine_count={routine_count}",
                ],
                margin_f=low_f - current_max_f,
            )
        return _NO_LOCK

    # YES in-range — only when *all* upward signals say nothing more is
    # coming and the forecast peak is contained in the range.
    if low_f <= current_max_f <= high_f:
        if not state.has_forecast:
            return _NO_LOCK
        past_peak = state.hours_until_peak <= 0
        no_upward = state.solar_declining and state.metar_trend_rate <= 0
        forecast_caps = state.forecast_peak_f <= high_f + 0.5
        if past_peak and no_upward and forecast_caps:
            return LockDecision(
                side="YES",
                reasons=[
                    f"market-day max {current_max_f:.1f}°F inside "
                    f"[{low_f:.0f}, {high_f:.0f}]°F ({op})",
                    f"past peak (h_to_peak={state.hours_until_peak:.1f}), "
                    f"solar declining ({state.solar_decline_magnitude:.0%}), "
                    f"trend {state.metar_trend_rate:+.1f}°F/hr",
                    f"forecast peak {state.forecast_peak_f:.1f}°F <= "
                    f"{high_f:.0f}°F",
                    f"routine_count={routine_count}",
                ],
                margin_f=min(current_max_f - low_f, high_f - current_max_f),
            )

    return _NO_LOCK
