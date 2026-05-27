"""Position sizing using fractional Kelly criterion."""

from __future__ import annotations

from dataclasses import dataclass

from src.config import settings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_TRADE_USD: float = 5.0
MAX_EXPOSURE_PCT: float = 0.25


def _effective_exposure_cap(bankroll: float) -> float:
    """Return the absolute USD exposure cap for the current bankroll.

    ``max(MAX_EXPOSURE_PCT × bankroll, settings.MAX_EXPOSURE_USD_FLOOR)``
    so small-bankroll mode keeps the bot trading instead of pinning at
    ~$100 cap. The floor only binds when ``bankroll < floor /
    MAX_EXPOSURE_PCT`` (≈ $1200 at defaults). Per-trade caps still apply.
    """
    return max(
        bankroll * MAX_EXPOSURE_PCT,
        settings.MAX_EXPOSURE_USD_FLOOR,
    )


def _apply_floor(floor_to_usd: float | None, *, ceiling: float) -> float | None:
    """Resolve a near-peak floor-up request against the tightest cap ceiling.

    Returns the floored stake (``min(floor_to_usd, ceiling)``) when floor-up is
    requested and the result is still ≥ ``MIN_TRADE_USD``; otherwise ``None``
    (the caller drops the trade). Keeping the ``min`` here guarantees the floor
    never exceeds an exposure / depth / per-trade ceiling.
    """
    if floor_to_usd is None:
        return None
    floored = min(floor_to_usd, ceiling)
    return floored if floored >= MIN_TRADE_USD else None

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
    floor_to_usd: float | None = None,
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
    floor_to_usd:
        When set and the Kelly-and-caps stake lands below ``MIN_TRADE_USD``,
        floor the stake up to this value *instead of dropping it* — but never
        above any cap ceiling (per-trade / exposure / USD / depth), so the
        floor can't breach a limit. If the tightest ceiling is itself below
        ``MIN_TRADE_USD`` (e.g. a thin book) the trade still drops. ``None``
        preserves the original drop-on-sub-min behavior. Caller gates this on
        the near-peak / high-confidence threshold class.
    """
    if kelly_fraction is None:
        kelly_fraction = settings.KELLY_FRACTION

    # Clamp market_prob to avoid division blowup
    market_prob = max(0.01, min(0.99, market_prob))

    # Cap the probability used in Kelly to bound oversizing when the
    # engine claims certainty. The audit on 2026-05-16 showed the
    # `model_prob ≈ 1.0` bin only won 78 % of the time; a 0.90 cap roughly
    # halves Kelly there without affecting well-calibrated bins.
    # The recorded `model_prob` on the Signal/EvaluationLog rows is
    # untouched — this is sizing-only.
    effective_prob = min(model_prob, settings.KELLY_PROB_CAP)
    payout = 1.0 / market_prob
    edge = effective_prob * payout - 1.0

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

    # --- Total exposure cap (max of pct-of-bankroll and USD floor) ---
    max_remaining = _effective_exposure_cap(bankroll) - current_exposure
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
        floored = _apply_floor(
            floor_to_usd,
            ceiling=min(
                max_trade,
                max_remaining,
                usd_cap,
                (
                    orderbook_depth * settings.DEPTH_POSITION_CAP_PCT
                    if orderbook_depth is not None and orderbook_depth > 0
                    else float("inf")
                ),
            ),
        )
        if floored is not None:
            return PositionSize(
                stake_usd=round(floored, 2),
                kelly_pct=full_kelly * kelly_fraction,
                capped=True,
                reason=f"floored to ${floored:.2f} (near-peak)",
            )
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
    floor_to_usd: float | None = None,
) -> PositionSize:
    """Size a lock-rule trade. No Kelly — no probability to plug in.

    The lock rule is a deterministic "outcome is physically decided" signal.
    Stake uses a fixed fraction of bankroll, capped by the standard exposure,
    max-position, and depth limits. The payout shape (buying near-resolved at
    high price gives tiny gross margin) means sizing must stay small and
    depth-aware to avoid moving the book.

    ``floor_to_usd`` mirrors :func:`size_position`: when set, a sub-``MIN_TRADE_USD``
    stake is floored up (respecting every cap ceiling) instead of dropped. At a
    typical bankroll the fixed 2% already clears the floor, so this is mostly a
    no-op here — wired for symmetry and the "bigger stake" knob.
    """
    price = max(0.01, min(0.99, price))

    raw_stake = bankroll * settings.LOCK_POSITION_PCT
    stake = raw_stake
    capped = False
    reasons: list[str] = [f"fixed {settings.LOCK_POSITION_PCT:.0%} bankroll"]

    max_remaining = _effective_exposure_cap(bankroll) - current_exposure
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
        floored = _apply_floor(
            floor_to_usd,
            ceiling=min(
                max_remaining,
                usd_cap,
                (
                    orderbook_depth * 0.15
                    if orderbook_depth is not None and orderbook_depth > 0
                    else float("inf")
                ),
            ),
        )
        if floored is not None:
            return PositionSize(
                stake_usd=round(floored, 2), kelly_pct=0.0, capped=True,
                reason=f"floored to ${floored:.2f} (near-peak)",
            )
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
