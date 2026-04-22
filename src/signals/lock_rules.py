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
from src.signals.mapper import icao_timezone
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

    Target day = local calendar day of ``market_end_date`` in the station's
    timezone. Window = [local_midnight, min(market_end_date, now_utc)].

    Returns (None, 0) when no METARs fall in the window (e.g. market hasn't
    started yet, or station has no recent data).
    """
    tz = icao_timezone(state.station_icao)
    local_end = market_end_date.astimezone(tz)
    local_day_start = local_end.replace(hour=0, minute=0, second=0, microsecond=0)
    utc_start = local_day_start.astimezone(timezone.utc)
    utc_end = min(market_end_date, now_utc)
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

    Parameters
    ----------
    state:
        Aggregated per-city weather state. ``routine_history`` supplies the
        raw METAR points used for per-market daily-max computation.
    market:
        A Market row with ``parsed_threshold`` (°F), ``parsed_operator``, and
        ``end_date`` populated. Operator must be above/at_least/below/at_most;
        'exactly' and bracket markets are not in scope.
    now_utc:
        Evaluation time — defaults to the current UTC time. Set explicitly
        in backtest replay so that the max is computed over
        [target_day_local_midnight, min(market.end_date, now_utc)].
    """
    if market.parsed_threshold is None or market.parsed_operator is None:
        return _NO_LOCK
    if market.end_date is None:
        # Can't anchor to a target day without an end_date.
        return _NO_LOCK

    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    threshold_f = float(market.parsed_threshold)
    op = market.parsed_operator
    margin = settings.LOCK_MARGIN_F

    # Per-market daily max (anchored to market.end_date's local day, not the
    # snapshot's local day). This is the critical distinction for markets
    # that close mid-day (e.g. 07:00 local) — in those cases the relevant
    # window is just early-morning observations, not the previous full day.
    current_max_f, routine_count = _market_daily_max(state, market.end_date, now_utc)
    if current_max_f is None:
        return _NO_LOCK

    # Routine count guard shared by both sides — we need confidence in the observation.
    if routine_count < settings.MIN_ROUTINE_COUNT:
        return _NO_LOCK

    # --- Easy direction: observed max already clears threshold by margin ---
    # Daily max is monotonic, so this side is mathematically locked.
    if op in ("above", "at_least"):
        if current_max_f >= threshold_f + margin:
            return LockDecision(
                side="YES",
                reasons=[
                    f"market-day max {current_max_f:.1f}°F >= threshold "
                    f"{threshold_f:.0f}°F + {margin:.1f}°F margin ({op})",
                    f"routine_count={routine_count}",
                ],
                margin_f=current_max_f - threshold_f,
            )
    elif op in ("below", "at_most"):
        if current_max_f >= threshold_f + margin:
            return LockDecision(
                side="NO",
                reasons=[
                    f"market-day max {current_max_f:.1f}°F exceeds threshold "
                    f"{threshold_f:.0f}°F by {margin:.1f}°F ({op} already violated)",
                    f"routine_count={routine_count}",
                ],
                margin_f=current_max_f - threshold_f,
            )
    else:
        # 'exactly' and unknown operators are out of scope.
        return _NO_LOCK

    # --- Hard direction: observed max is below threshold, need forecast+trend ---
    below_margin = current_max_f < threshold_f - margin
    if not below_margin:
        return _NO_LOCK

    has_headroom, evidence = _no_more_heating(state, threshold_f)
    if not has_headroom:
        return _NO_LOCK

    base_reason = (
        f"market-day max {current_max_f:.1f}°F < threshold "
        f"{threshold_f:.0f}°F - {margin:.1f}°F margin ({op})"
    )
    side: Side = "NO" if op in ("above", "at_least") else "YES"
    return LockDecision(
        side=side,
        reasons=[base_reason, *evidence, f"routine_count={routine_count}"],
        margin_f=threshold_f - current_max_f,
    )
