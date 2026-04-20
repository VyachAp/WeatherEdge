"""Backtesting engine — replay historical signals to compute bankroll metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.config import settings
from src.risk.drawdown import DrawdownMonitor
from src.risk.kelly import MIN_TRADE_USD, size_position

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SimSignal:
    """A single signal for the simulator."""

    model_prob: float
    market_prob: float
    outcome: bool  # True = model's predicted direction was correct


@dataclass
class SimResult:
    """Aggregated output of :func:`simulate_bankroll`."""

    final_bankroll: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    num_trades: int
    num_skipped: int
    bankroll_curve: list[float] = field(repr=False)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def simulate_bankroll(
    signals: list[SimSignal],
    initial_bankroll: float | None = None,
    kelly_fraction: float | None = None,
) -> SimResult:
    """Replay *signals* sequentially and return performance metrics.

    Each signal is sized via :func:`size_position` and adjusted by the
    :class:`DrawdownMonitor`.  Trades below the $5 minimum are skipped.
    """
    if initial_bankroll is None:
        initial_bankroll = settings.INITIAL_BANKROLL
    if kelly_fraction is None:
        kelly_fraction = settings.KELLY_FRACTION

    bankroll = initial_bankroll
    monitor = DrawdownMonitor(initial_bankroll)

    curve: list[float] = []
    returns: list[float] = []
    wins = 0
    trades = 0
    skipped = 0
    peak = initial_bankroll

    for sig in signals:
        dd_state = monitor.check(bankroll)

        pos = size_position(
            bankroll,
            sig.model_prob,
            sig.market_prob,
            current_exposure=0.0,
            kelly_fraction=kelly_fraction,
        )

        stake = pos.stake_usd * dd_state.size_multiplier

        if stake < MIN_TRADE_USD:
            skipped += 1
            curve.append(bankroll)
            continue

        # Compute P&L
        payout_ratio = 1.0 / max(0.01, min(0.99, sig.market_prob))
        if sig.outcome:
            pnl = stake * (payout_ratio - 1.0)
            wins += 1
        else:
            pnl = -stake

        bankroll += pnl
        bankroll = max(bankroll, 0.0)  # can't go negative
        trades += 1

        returns.append(pnl / stake if stake > 0 else 0.0)

        # Advance drawdown monitor
        monitor.advance(bankroll)
        if bankroll > peak:
            peak = bankroll

        curve.append(bankroll)

    # --- Max drawdown over the curve ---
    max_dd = _max_drawdown(curve, initial_bankroll)

    # --- Sharpe ratio (per-trade) ---
    sharpe = _sharpe(returns, trades)

    win_rate = wins / trades if trades > 0 else 0.0

    return SimResult(
        final_bankroll=round(bankroll, 2),
        max_drawdown=round(max_dd, 6),
        sharpe_ratio=round(sharpe, 4),
        win_rate=round(win_rate, 4),
        num_trades=trades,
        num_skipped=skipped,
        bankroll_curve=curve,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _max_drawdown(curve: list[float], initial: float) -> float:
    peak = initial
    max_dd = 0.0
    for val in curve:
        if val > peak:
            peak = val
        dd = (peak - val) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _sharpe(returns: list[float], n_trades: int) -> float:
    if n_trades < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(var) if var > 0 else 0.0
    if std == 0:
        return 0.0
    return mean / std * math.sqrt(n_trades)


# ---------------------------------------------------------------------------
# Distribution pipeline backtest (V2)
# ---------------------------------------------------------------------------


@dataclass
class CalibrationBucket:
    """Calibration stats for a single temperature bucket."""

    bucket_value: int
    predicted_avg: float
    actual_rate: float
    count: int


@dataclass
class DistributionSimResult:
    """Result of the distribution pipeline backtest."""

    calibration_error: float  # Mean absolute error across buckets
    brier_score: float
    num_days: int
    per_bucket: list[CalibrationBucket]


async def simulate_distribution_pipeline(
    stations: list[str],
    days_back: int = 30,
) -> DistributionSimResult:
    """Backtest the probability engine against historical METAR outcomes.

    For each station and each historical day:
      1. Reconstruct a WeatherState from stored routine METARs
      2. Run compute_distribution() with typical temperature buckets
      3. Compare predicted probabilities to the actual daily max outcome

    Returns calibration metrics.
    """
    from datetime import datetime, timedelta, timezone
    from src.db.engine import async_session
    from src.db.models import MetarObservation
    from src.signals.probability_engine import compute_distribution
    from src.signals.state_aggregator import WeatherState
    from sqlalchemy import select, func

    # Collect predicted vs actual across all days/stations
    predictions: list[tuple[int, float, bool]] = []  # (bucket, predicted_prob, was_actual)
    num_days = 0

    async with async_session() as session:
        for icao in stations:
            # Get date range with data
            for day_offset in range(days_back, 0, -1):
                day = datetime.now(timezone.utc) - timedelta(days=day_offset)
                day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
                day_end = day_start + timedelta(days=1)

                # Get routine METARs for this day
                stmt = (
                    select(MetarObservation)
                    .where(
                        MetarObservation.station_icao == icao,
                        MetarObservation.observed_at >= day_start,
                        MetarObservation.observed_at < day_end,
                        MetarObservation.is_speci == False,
                        MetarObservation.temp_f.isnot(None),
                    )
                    .order_by(MetarObservation.observed_at)
                )
                result = await session.execute(stmt)
                metars = result.scalars().all()

                if len(metars) < 3:
                    continue

                # Actual daily max from routine METARs
                actual_max = max(m.temp_f for m in metars)
                actual_bucket = int(round(actual_max))

                # Simulate mid-day state (use first half of METARs)
                mid_count = len(metars) // 2
                mid_metars = metars[:mid_count] if mid_count >= 2 else metars[:2]
                mid_max = max(m.temp_f for m in mid_metars)

                # Build a simplified WeatherState
                temps = [m.temp_f for m in mid_metars]
                if len(temps) >= 2:
                    rate = (temps[-1] - temps[0]) / max(1, len(temps) - 1)
                else:
                    rate = 0.0

                state = WeatherState(
                    station_icao=icao,
                    current_max_f=mid_max,
                    metar_trend_rate=rate,
                    dewpoint_trend_rate=0.0,
                    forecast_peak_f=mid_max + 2.0,  # Simple estimate
                    hours_until_peak=3.0,
                    solar_declining=False,
                    solar_decline_magnitude=0.0,
                    cloud_rising=False,
                    cloud_rise_magnitude=0.0,
                    routine_count_today=mid_count,
                )

                # Generate buckets around expected range
                center = int(mid_max)
                buckets = list(range(center - 5, center + 10))

                dist = compute_distribution(state, buckets)

                for bucket, prob in dist.probabilities.items():
                    was_actual = bucket == actual_bucket
                    predictions.append((bucket, prob, was_actual))

                num_days += 1

    # Compute calibration metrics
    if not predictions:
        return DistributionSimResult(
            calibration_error=0.0, brier_score=0.0, num_days=0, per_bucket=[]
        )

    # Brier score
    brier = sum((p - (1.0 if a else 0.0)) ** 2 for _, p, a in predictions) / len(predictions)

    # Per-bucket calibration
    from collections import defaultdict
    bucket_preds: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    for bucket, prob, actual in predictions:
        bucket_preds[bucket].append((prob, actual))

    per_bucket: list[CalibrationBucket] = []
    total_error = 0.0
    n_buckets = 0
    for b in sorted(bucket_preds):
        items = bucket_preds[b]
        pred_avg = sum(p for p, _ in items) / len(items)
        actual_rate = sum(1 for _, a in items if a) / len(items)
        per_bucket.append(CalibrationBucket(b, round(pred_avg, 4), round(actual_rate, 4), len(items)))
        total_error += abs(pred_avg - actual_rate)
        n_buckets += 1

    cal_error = total_error / n_buckets if n_buckets > 0 else 0.0

    return DistributionSimResult(
        calibration_error=round(cal_error, 4),
        brier_score=round(brier, 6),
        num_days=num_days,
        per_bucket=per_bucket,
    )
