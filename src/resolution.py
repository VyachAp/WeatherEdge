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

    Returns the live CLOB mid, or the stored ``market.current_yes_price`` if
    the live fetch fails. Does not mutate the ORM row — concurrent writes
    to ``markets.current_yes_price`` from multiple jobs were the cause of a
    cross-transaction deadlock; only ``scan_markets`` persists this column.
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
    return (bid + ask) / 2.0


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
    refreshed_prices: dict[str, float | None] = {}
    for trade in trades:
        market = trade.market
        if market.id not in refreshed_prices:
            refreshed_prices[market.id] = await _refresh_market_price(market)
        price = refreshed_prices[market.id]
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


async def get_open_trade_value(session: AsyncSession) -> float:
    """Mark-to-market value of conditional tokens held in OPEN trades.

    For each OPEN trade, the position is worth ``shares × per_share_value``
    where ``shares = stake_usd / entry_price`` and the per-share value is
    ``Market.current_yes_price`` for ``BUY_YES`` or ``1 - current_yes_price``
    for ``BUY_NO``. Falls back to cost basis (``stake_usd``) when the
    cached price or entry price is missing — counting the deployed dollar
    at par is much closer to truth than dropping it from equity entirely.

    Without this, ``get_current_bankroll`` treated the wallet drain from
    placing trades as a realized loss until each trade resolved. With many
    same-day positions in flight, that produced phantom drawdown that
    falsely tripped the PAUSED state and shut new trading down.

    The cached price is refreshed every 5 min by ``job_unified_pipeline``
    so this adds no extra HTTP cost.
    """
    result = await session.execute(
        select(
            Trade.direction,
            Trade.stake_usd,
            Trade.entry_price,
            Market.current_yes_price,
        )
        .join(Market, Trade.market_id == Market.id)
        .where(Trade.status == TradeStatus.OPEN)
    )
    total = 0.0
    for direction, stake, entry, yes_price in result.all():
        if stake is None or stake <= 0:
            continue
        if entry is None or entry <= 0 or yes_price is None:
            total += float(stake)
            continue
        per_share = (
            float(yes_price)
            if direction == TradeDirection.BUY_YES
            else 1.0 - float(yes_price)
        )
        # Clip in case the cached price is briefly outside [0, 1].
        per_share = max(0.0, min(1.0, per_share))
        shares = float(stake) / float(entry)
        total += shares * per_share
    return total


async def get_current_bankroll(session: AsyncSession) -> float:
    """Return current spendable-equivalent bankroll in USD.

    Equity, not wallet liquidity. Three lifecycle stages contribute:
      * **Wallet** — settled USDC balance.
      * **Unredeemed WON** — payouts on won trades whose conditional
        tokens haven't been on-chain redeemed yet (see
        :func:`get_unredeemed_won_payout`).
      * **OPEN trade value** — mark-to-market value of in-flight
        positions (see :func:`get_open_trade_value`). Without this,
        placing trades shows up as drawdown until they resolve.

    Sources for the wallet term, in priority order:
      1. Live USDC wallet balance via the CLOB client (when a private key
         is configured).
      2. Latest BankrollLog row.
      3. ``INITIAL_BANKROLL`` setting as a last-resort fallback.
    """
    from src.execution.polymarket_client import get_wallet_usdc_balance

    unredeemed = await get_unredeemed_won_payout(session)
    open_value = await get_open_trade_value(session)

    wallet = get_wallet_usdc_balance()
    if wallet is not None and wallet > 0:
        return wallet + unredeemed + open_value

    result = await session.execute(
        select(BankrollLog.balance)
        .order_by(BankrollLog.timestamp.desc())
        .limit(1)
    )
    row = result.scalar_one_or_none()
    base = row if row is not None else settings.INITIAL_BANKROLL
    return base + unredeemed + open_value


async def get_current_exposure(session: AsyncSession) -> float:
    """Sum ``stake_usd`` of all currently open trades."""
    result = await session.execute(
        select(func.coalesce(func.sum(Trade.stake_usd), 0.0)).where(
            Trade.status == TradeStatus.OPEN
        )
    )
    return float(result.scalar_one())
