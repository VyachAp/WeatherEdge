"""Position sizing using fractional Kelly criterion."""

from __future__ import annotations

from dataclasses import dataclass

from src.config import settings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_TRADE_USD: float = 5.0
MAX_EXPOSURE_PCT: float = 0.25

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class PositionSize:
    """Result of a position-sizing calculation."""

    stake_usd: float
    kelly_pct: float
    capped: bool
    reason: str


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


def size_position(
    bankroll: float,
    model_prob: float,
    market_prob: float,
    current_exposure: float = 0.0,
    kelly_fraction: float | None = None,
    max_position_usd: float | None = None,
    orderbook_depth: float | None = None,
) -> PositionSize:
    """Compute stake using fractional Kelly criterion with hard caps.

    Parameters
    ----------
    bankroll:
        Current bankroll in USD.
    model_prob:
        Model's estimated probability of the outcome.
    market_prob:
        Implied probability from market price.
    current_exposure:
        Sum of ``stake_usd`` across all open positions.
    kelly_fraction:
        Fraction of full Kelly to use (default from settings).
    max_position_usd:
        Hard USD cap per position (default from settings.MAX_POSITION_USD).
    orderbook_depth:
        Visible orderbook depth in USD. Position capped at 20% of depth.
    """
    if kelly_fraction is None:
        kelly_fraction = settings.KELLY_FRACTION

    # Clamp market_prob to avoid division blowup
    market_prob = max(0.01, min(0.99, market_prob))

    payout = 1.0 / market_prob
    edge = model_prob * payout - 1.0

    if edge <= 0:
        return PositionSize(stake_usd=0, kelly_pct=0, capped=False, reason="no edge")

    full_kelly = edge / (payout - 1.0)
    raw_stake = bankroll * full_kelly * kelly_fraction

    stake = raw_stake
    capped = False
    reasons: list[str] = []

    # --- Per-trade cap (5% of bankroll) ---
    max_trade = bankroll * settings.MAX_POSITION_PCT
    if stake > max_trade:
        stake = max_trade
        capped = True
        reasons.append(f"per-trade cap ({settings.MAX_POSITION_PCT:.0%})")

    # --- Total exposure cap (25% of bankroll) ---
    max_remaining = bankroll * MAX_EXPOSURE_PCT - current_exposure
    if max_remaining <= 0:
        return PositionSize(
            stake_usd=0,
            kelly_pct=full_kelly * kelly_fraction,
            capped=True,
            reason="exposure limit reached",
        )
    if stake > max_remaining:
        stake = max_remaining
        capped = True
        reasons.append("total exposure cap")

    # --- Hard USD cap ---
    usd_cap = max_position_usd if max_position_usd is not None else settings.MAX_POSITION_USD
    if stake > usd_cap:
        stake = usd_cap
        capped = True
        reasons.append(f"max position ${usd_cap:.0f}")

    # --- Orderbook depth cap (max 20% of visible depth) ---
    if orderbook_depth is not None and orderbook_depth > 0:
        depth_cap = orderbook_depth * settings.DEPTH_POSITION_CAP_PCT
        if stake > depth_cap:
            stake = depth_cap
            capped = True
            reasons.append(f"depth cap ({settings.DEPTH_POSITION_CAP_PCT:.0%} of ${orderbook_depth:.0f})")

    # --- Minimum viable trade ---
    if 0 < stake < MIN_TRADE_USD:
        return PositionSize(
            stake_usd=0,
            kelly_pct=full_kelly * kelly_fraction,
            capped=True,
            reason=f"below ${MIN_TRADE_USD:.0f} minimum",
        )

    reason = ", ".join(reasons) if reasons else "sized normally"
    return PositionSize(
        stake_usd=round(stake, 2),
        kelly_pct=full_kelly * kelly_fraction,
        capped=capped,
        reason=reason,
    )


def size_locked_position(
    bankroll: float,
    price: float,
    current_exposure: float = 0.0,
    orderbook_depth: float | None = None,
) -> PositionSize:
    """Size a lock-rule trade. No Kelly — no probability to plug in.

    The lock rule is a deterministic "outcome is physically decided" signal.
    Stake uses a fixed fraction of bankroll, capped by the standard exposure,
    max-position, and depth limits. The payout shape (buying near-resolved at
    high price gives tiny gross margin) means sizing must stay small and
    depth-aware to avoid moving the book.
    """
    price = max(0.01, min(0.99, price))

    raw_stake = bankroll * settings.LOCK_POSITION_PCT
    stake = raw_stake
    capped = False
    reasons: list[str] = [f"fixed {settings.LOCK_POSITION_PCT:.0%} bankroll"]

    max_remaining = bankroll * MAX_EXPOSURE_PCT - current_exposure
    if max_remaining <= 0:
        return PositionSize(
            stake_usd=0, kelly_pct=0.0, capped=True,
            reason="exposure limit reached",
        )
    if stake > max_remaining:
        stake = max_remaining
        capped = True
        reasons.append("total exposure cap")

    usd_cap = settings.MAX_POSITION_USD / 2
    if stake > usd_cap:
        stake = usd_cap
        capped = True
        reasons.append(f"lock-rule cap ${usd_cap:.0f}")

    if orderbook_depth is not None and orderbook_depth > 0:
        depth_cap = orderbook_depth * 0.15
        if stake > depth_cap:
            stake = depth_cap
            capped = True
            reasons.append(f"depth cap (15% of ${orderbook_depth:.0f})")

    if 0 < stake < MIN_TRADE_USD:
        return PositionSize(
            stake_usd=0, kelly_pct=0.0, capped=True,
            reason=f"below ${MIN_TRADE_USD:.0f} minimum",
        )

    return PositionSize(
        stake_usd=round(stake, 2),
        kelly_pct=0.0,
        capped=capped,
        reason=", ".join(reasons),
    )
