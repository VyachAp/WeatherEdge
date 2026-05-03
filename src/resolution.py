"""Trade resolution, P&L calculation, and bankroll helpers."""

import logging
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from src.config import settings
from src.db.models import BankrollLog, Market, Trade, TradeDirection, TradeStatus

logger = logging.getLogger(__name__)

# Price thresholds for considering a market resolved.
_YES_RESOLVED_THRESHOLD = 0.95
_NO_RESOLVED_THRESHOLD = 0.05


async def _refresh_market_price(market: Market) -> float | None:
    """Fetch live mid for an expired market's YES token.

    The 5-min ``job_unified_pipeline`` and 15-min ``job_scan_markets`` may
    have left ``market.current_yes_price`` 30+ minutes stale by the time a
    market expires. Re-querying the CLOB before applying the 0.95/0.05
    resolution thresholds tightens the loop so we don't wait an extra tick
    on edge-case 0.94 → 0.96 drift. Failures fall back to the stored value.
    """
    from src.execution.polymarket_client import (
        get_best_bid_ask,
        get_token_ids,
    )

    try:
        token_ids = await get_token_ids(market.id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Token ID fetch failed for %s: %s", market.id, exc)
        return market.current_yes_price

    if not token_ids:
        return market.current_yes_price

    try:
        quote = get_best_bid_ask(token_ids[0])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Price refresh failed for %s: %s", market.id, exc)
        return market.current_yes_price

    if quote is None:
        return market.current_yes_price

    bid, ask = quote
    mid = (bid + ask) / 2.0
    market.current_yes_price = mid
    return mid


async def resolve_trades(session: AsyncSession) -> list[Trade]:
    """Find open trades on expired markets and settle them.

    Resolution heuristic based on final market price:
    - ``current_yes_price >= 0.95`` → YES outcome
    - ``current_yes_price <= 0.05`` → NO outcome
    - otherwise → not yet resolvable, skip

    Returns the list of trades whose status was updated.
    """
    now = datetime.utcnow()

    result = await session.execute(
        select(Trade)
        .options(joinedload(Trade.market))
        .where(Trade.status == TradeStatus.OPEN)
        .join(Market)
        .where(Market.end_date < now)
    )
    trades = list(result.scalars().unique())

    resolved: list[Trade] = []
    refreshed_markets: set[str] = set()
    for trade in trades:
        market = trade.market
        if market.id not in refreshed_markets:
            await _refresh_market_price(market)
            refreshed_markets.add(market.id)
        price = market.current_yes_price
        if price is None:
            continue

        if price >= _YES_RESOLVED_THRESHOLD:
            yes_won = True
        elif price <= _NO_RESOLVED_THRESHOLD:
            yes_won = False
        else:
            # Market hasn't clearly resolved yet.
            continue

        trade_won = (
            (trade.direction == TradeDirection.BUY_YES and yes_won)
            or (trade.direction == TradeDirection.BUY_NO and not yes_won)
        )

        if trade_won:
            trade.status = TradeStatus.WON
            entry = trade.entry_price or 0.0
            trade.pnl = trade.stake_usd * (1.0 / entry - 1.0) if entry > 0 else 0.0
            trade.exit_price = 1.0
        else:
            trade.status = TradeStatus.LOST
            trade.pnl = -trade.stake_usd
            trade.exit_price = 0.0

        trade.closed_at = now
        resolved.append(trade)
        logger.info(
            "Resolved trade %s on market %s → %s (pnl=%.2f)",
            trade.id,
            trade.market_id,
            trade.status.value,
            trade.pnl,
        )

    return resolved


async def calculate_daily_pnl(session: AsyncSession) -> float:
    """Sum P&L of all trades resolved today (UTC)."""
    today = datetime.now(timezone.utc).date()
    result = await session.execute(
        select(func.coalesce(func.sum(Trade.pnl), 0.0)).where(
            Trade.status.in_([TradeStatus.WON, TradeStatus.LOST]),
            func.date(Trade.closed_at) == today,
        )
    )
    return float(result.scalar_one())


async def get_unredeemed_won_payout(session: AsyncSession) -> float:
    """Sum the future-redeem dollar value of unsettled WON trades.

    Polymarket wins do **not** auto-settle into wallet USDC — the user must
    call ``redeemPositions()`` on-chain (via the ``bet redeem`` CLI) to
    convert the conditional tokens into wallet balance. Until then the
    payout sits as conditional tokens; the wallet doesn't reflect it.

    For each WON trade with ``redeemed_at IS NULL``:
        future_payout = stake_usd / entry_price = stake_usd + pnl

    (because ``pnl = stake * (1/entry - 1)``)
    """
    result = await session.execute(
        select(
            func.coalesce(
                func.sum(Trade.stake_usd + func.coalesce(Trade.pnl, 0.0)), 0.0
            )
        ).where(
            Trade.status == TradeStatus.WON,
            Trade.redeemed_at.is_(None),
        )
    )
    return float(result.scalar_one())


async def get_current_bankroll(session: AsyncSession) -> float:
    """Return current spendable-equivalent bankroll in USD.

    Equity, not wallet liquidity: counts won-but-unredeemed positions so
    the drawdown monitor compares against true equity instead of phantom
    drawdown caused by conditional tokens still sitting on-chain.

    Sources, in priority order:
      1. Live USDC wallet balance via the CLOB client (when a private key
         is configured), **plus** unredeemed WON payouts.
      2. Latest BankrollLog row plus unredeemed WON payouts.
      3. ``INITIAL_BANKROLL`` setting as a last-resort fallback.
    """
    from src.execution.polymarket_client import get_wallet_usdc_balance

    unredeemed = await get_unredeemed_won_payout(session)

    wallet = get_wallet_usdc_balance()
    if wallet is not None and wallet > 0:
        return wallet + unredeemed

    result = await session.execute(
        select(BankrollLog.balance)
        .order_by(BankrollLog.timestamp.desc())
        .limit(1)
    )
    row = result.scalar_one_or_none()
    base = row if row is not None else settings.INITIAL_BANKROLL
    return base + unredeemed


async def get_current_exposure(session: AsyncSession) -> float:
    """Sum ``stake_usd`` of all currently open trades."""
    result = await session.execute(
        select(func.coalesce(func.sum(Trade.stake_usd), 0.0)).where(
            Trade.status == TradeStatus.OPEN
        )
    )
    return float(result.scalar_one())
