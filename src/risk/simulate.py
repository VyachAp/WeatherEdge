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
