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


async def resolve_trades(session: AsyncSession) -> list[Trade]:
    """Find open trades on expired markets and settle them.

    Resolution heuristic based on final market price:
    - ``current_yes_price >= 0.95`` → YES outcome
    - ``current_yes_price <= 0.05`` → NO outcome
    - otherwise → not yet resolvable, skip

    Returns the list of trades whose status was updated.
    """
    now = datetime.now(timezone.utc)

    result = await session.execute(
        select(Trade)
        .options(joinedload(Trade.market))
        .where(Trade.status == TradeStatus.OPEN)
        .join(Market)
        .where(Market.end_date < now)
    )
    trades = list(result.scalars().unique())

    resolved: list[Trade] = []
    for trade in trades:
        market = trade.market
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


async def get_current_bankroll(session: AsyncSession) -> float:
    """Return the latest bankroll balance, or ``INITIAL_BANKROLL`` if none."""
    result = await session.execute(
        select(BankrollLog.balance)
        .order_by(BankrollLog.timestamp.desc())
        .limit(1)
    )
    row = result.scalar_one_or_none()
    return row if row is not None else settings.INITIAL_BANKROLL


async def get_current_exposure(session: AsyncSession) -> float:
    """Sum ``stake_usd`` of all currently open trades."""
    result = await session.execute(
        select(func.coalesce(func.sum(Trade.stake_usd), 0.0)).where(
            Trade.status == TradeStatus.OPEN
        )
    )
    return float(result.scalar_one())
