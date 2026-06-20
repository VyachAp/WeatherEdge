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
    prob_cap: float | None = None,
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
    prob_cap:
        Overrides ``settings.KELLY_PROB_CAP`` — the clamp applied to
        ``model_prob`` before computing edge. ``None`` uses the global cap.
        Callers pass a higher value (``NEAR_LOCK_CONVICTION_PROB_CAP``) for a
        physically near-locked threshold bet so a buy above the 0.90 global cap
        isn't sized to $0 ("no edge"). Sizing-only; the recorded prob is
        untouched, and the full cap cascade below still bounds the stake.
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
    effective_cap = settings.KELLY_PROB_CAP if prob_cap is None else prob_cap
    effective_prob = min(model_prob, effective_cap)
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
    *,
    win_prob: float | None = None,
    kelly_fraction: float | None = None,
    max_position_pct: float | None = None,
    depth_cap_pct: float = 0.15,
) -> PositionSize:
    """Size a lock-rule trade.

    **Flat path (``win_prob is None`` — the default):** the lock rule is a
    deterministic "outcome is physically decided" signal with no probability
    to plug in, so the stake is a fixed fraction of bankroll
    (``LOCK_POSITION_PCT``), capped by the standard exposure, max-position, and
    depth limits.

    **Conviction path (``win_prob`` set):** an EASY-YES lock on a trusted
    station is genuinely near-certain (monotonic daily max + unbiased resolver
    divergence), so flat 2% leaves large EV on the table. Here the stake is
    sized by *fractional Kelly against the price*::

        b      = (1 - price) / price            # net odds on a YES buy at `price`
        f_star = max(0, win_prob - (1 - win_prob) / b)
        frac   = min(f_star * kelly_fraction, max_position_pct)

    No ``KELLY_PROB_CAP`` is applied — that cap exists to tame the noisy
    probability path; a physically-locked, station-verified bet legitimately
    sits near 1.0. ``max_position_pct`` is the hard concentration backstop.

    ``depth_cap_pct`` is the orderbook-depth cap fraction (0.15 on the flat
    path; relaxed for conviction locks that walk the book). ``floor_to_usd``
    mirrors :func:`size_position`: when set, a sub-``MIN_TRADE_USD`` stake is
    floored up (respecting every cap ceiling) instead of dropped — used by the
    flat path's near-peak floor-up, not by conviction sizing.
    """
    price = max(0.01, min(0.99, price))

    if win_prob is not None:
        kf = kelly_fraction if kelly_fraction is not None else settings.LOCK_KELLY_FRACTION
        cap_pct = (
            max_position_pct if max_position_pct is not None
            else settings.LOCK_POSITION_PCT
        )
        b = (1.0 - price) / price
        f_star = max(0.0, win_prob - (1.0 - win_prob) / b) if b > 0 else 0.0
        frac = min(f_star * kf, cap_pct)
        kelly_pct = frac
        raw_stake = bankroll * frac
        reasons: list[str] = [
            f"conviction kelly {frac:.1%} (p={win_prob:.2f}, f*={f_star:.2f}, "
            f"cap={cap_pct:.0%})"
        ]
    else:
        kelly_pct = 0.0
        raw_stake = bankroll * settings.LOCK_POSITION_PCT
        reasons = [f"fixed {settings.LOCK_POSITION_PCT:.0%} bankroll"]

    stake = raw_stake
    capped = False

    max_remaining = _effective_exposure_cap(bankroll) - current_exposure
    if max_remaining <= 0:
        return PositionSize(
            stake_usd=0, kelly_pct=kelly_pct, capped=True,
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
        depth_cap = orderbook_depth * depth_cap_pct
        if stake > depth_cap:
            stake = depth_cap
            capped = True
            reasons.append(f"depth cap ({depth_cap_pct:.0%} of ${orderbook_depth:.0f})")

    if 0 < stake < MIN_TRADE_USD:
        floored = _apply_floor(
            floor_to_usd,
            ceiling=min(
                max_remaining,
                usd_cap,
                (
                    orderbook_depth * depth_cap_pct
                    if orderbook_depth is not None and orderbook_depth > 0
                    else float("inf")
                ),
            ),
        )
        if floored is not None:
            return PositionSize(
                stake_usd=round(floored, 2), kelly_pct=kelly_pct, capped=True,
                reason=f"floored to ${floored:.2f} (near-peak)",
            )
        return PositionSize(
            stake_usd=0, kelly_pct=kelly_pct, capped=True,
            reason=f"below ${MIN_TRADE_USD:.0f} minimum",
        )

    return PositionSize(
        stake_usd=round(stake, 2),
        kelly_pct=kelly_pct,
        capped=capped,
        reason=", ".join(reasons),
    )
