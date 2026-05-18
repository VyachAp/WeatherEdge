"""Trade resolution, P&L calculation, and bankroll helpers."""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from src.config import settings
from src.db.models import BankrollLog, Market, Trade, TradeDirection, TradeStatus

logger = logging.getLogger(__name__)

# Legacy CLOB-mid thresholds, retained for trades on markets without a
# stored ``condition_id`` (pre-caching legacy rows). The primary
# resolution source is on-chain ``payoutDenominator`` — see
# ``resolve_trades`` for the cascade.
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


def _apply_settlement(trade: Trade, yes_won: bool, now: datetime) -> None:
    """Apply terminal WON/LOST state to ``trade`` given the YES outcome."""
    trade_won = (
        (trade.direction == TradeDirection.BUY_YES and yes_won)
        or (trade.direction == TradeDirection.BUY_NO and not yes_won)
    )
    if trade_won:
        trade.status = TradeStatus.WON
        entry = trade.entry_price or 0.0
        trade.pnl = (
            trade.stake_usd * (1.0 / entry - 1.0) if entry > 0 else 0.0
        )
        trade.exit_price = 1.0
    else:
        trade.status = TradeStatus.LOST
        trade.pnl = -float(trade.stake_usd)
        trade.exit_price = 0.0
    trade.closed_at = now


async def resolve_trades(session: AsyncSession) -> list[Trade]:
    """Find open trades on expired markets and settle them.

    Resolution cascade (in priority order):

    1. **On-chain** (authoritative) — call ``payoutDenominator(conditionId)``
       on the Polygon ConditionalTokens contract via
       ``bet_helpers.get_payout_outcome``. If UMA has reported (denom > 0),
       map ``payoutNumerators`` to a YES/NO outcome and settle. This is
       the only path that produces a ``WON`` whose tokens are actually
       redeemable — pre-2026-05-19 we used CLOB-mid 0.95/0.05 as the
       signal, which fires hours-to-days before UMA pushes the result
       on-chain and produced false ``💸 redeem`` nudges that reverted
       with ``result for condition not received yet``.

    2. **On-chain unresolved + null-fill past grace** → mark ``LOST``.
       No on-chain position exists (order never landed); release the
       reserved exposure.

    3. **On-chain unresolved + filled position past grace** → log a
       warning telling the operator to run ``admin reconcile-stuck``.
       UMA may be in dispute or the resolution source has not published.

    4. **Legacy CLOB-mid fallback** — only when ``market.condition_id``
       is NULL (pre-caching legacy rows) or the Polygon RPC connection
       could not be built. Preserves prior behavior for trades the
       chain check cannot reach.

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
    if not trades:
        return []

    ctf = await _build_ctf_readonly()
    chain_outcomes: dict[str, bool | None] = {}
    refreshed_prices: dict[str, float | None] = {}
    grace = timedelta(hours=settings.RESOLVE_NO_PRICE_GRACE_HOURS)
    resolved: list[Trade] = []

    for trade in trades:
        market = trade.market
        # Production stores end_date as TIMESTAMP WITH TIME ZONE
        # (asyncpg returns aware); test mocks often use naive
        # ``datetime.utcnow()``. Normalize to aware UTC so the
        # arithmetic with ``now`` doesn't raise TypeError on either.
        end_date = market.end_date
        if end_date is not None and end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        # ── 1. On-chain payout (authoritative) ─────────────────────
        chain_outcome: bool | None = None
        if ctf is not None and market.condition_id:
            cid = market.condition_id
            if cid not in chain_outcomes:
                chain_outcomes[cid] = await _query_payout_outcome(ctf, cid)
            chain_outcome = chain_outcomes[cid]

        if chain_outcome is not None:
            _apply_settlement(trade, chain_outcome, now)
            resolved.append(trade)
            logger.info(
                "Resolved trade %s on market %s → %s (on-chain payout, pnl=%.2f)",
                trade.id, trade.market_id, trade.status.value, trade.pnl,
            )
            continue

        # ── 2/3. Chain check available but UMA has not reported ───
        chain_reachable_for_market = (
            ctf is not None and bool(market.condition_id)
        )
        if chain_reachable_for_market:
            if end_date is None or end_date > now - grace:
                # Within grace — UMA reporting is just slow; wait.
                continue
            if trade.fill_price is None:
                # Past grace, no on-chain position. The order never
                # landed (delayed → never indexed). Release exposure.
                trade.status = TradeStatus.LOST
                trade.pnl = -float(trade.stake_usd)
                trade.exit_price = 0.0
                trade.closed_at = now
                resolved.append(trade)
                logger.warning(
                    "Resolved trade %s on market %s (null fill, UMA "
                    "unresolved %.0fh past end_date) → LOST",
                    trade.id, trade.market_id,
                    (now - end_date).total_seconds() / 3600.0,
                )
                continue
            logger.warning(
                "Trade %s on market %s: filled position past UMA report "
                "grace (%.0fh past end_date, UMA still unresolved). "
                "Run `admin reconcile-stuck` to settle from on-chain payout.",
                trade.id, trade.market_id,
                (now - end_date).total_seconds() / 3600.0,
            )
            continue

        # ── 4. Legacy CLOB-mid fallback (no condition_id / no RPC) ─
        if market.id not in refreshed_prices:
            refreshed_prices[market.id] = await _refresh_market_price(market)
        price = refreshed_prices[market.id]
        if price is None:
            if end_date is None or end_date > now - grace:
                continue
            if trade.fill_price is None:
                trade.status = TradeStatus.LOST
                trade.pnl = -float(trade.stake_usd)
                trade.exit_price = 0.0
                trade.closed_at = now
                resolved.append(trade)
                logger.warning(
                    "Resolved trade %s (null fill, legacy fallback) on "
                    "market %s → LOST (CLOB dropped market %.0fh past "
                    "end_date; no on-chain position)",
                    trade.id, trade.market_id,
                    (now - end_date).total_seconds() / 3600.0,
                )
                continue
            logger.warning(
                "Trade %s on market %s has CLOB-dropped market %.0fh past "
                "end_date with populated fill_price=%.3f. Run `admin "
                "reconcile-stuck` to settle from on-chain payout.",
                trade.id, trade.market_id,
                (now - end_date).total_seconds() / 3600.0,
                trade.fill_price,
            )
            continue

        if price >= _YES_RESOLVED_THRESHOLD:
            yes_won = True
        elif price <= _NO_RESOLVED_THRESHOLD:
            yes_won = False
        else:
            continue

        _apply_settlement(trade, yes_won, now)
        resolved.append(trade)
        logger.info(
            "Resolved trade %s on market %s → %s (CLOB fallback, pnl=%.2f)",
            trade.id, trade.market_id, trade.status.value, trade.pnl,
        )

    return resolved


async def _build_ctf_readonly():
    """Build a read-only CTF contract handle, or return None on failure.

    Wraps the sync ``get_ctf_readonly`` in ``asyncio.to_thread`` so the
    Polygon RPC handshake doesn't block the event loop. A failure here
    downgrades the whole resolve_trades batch to the legacy CLOB-mid
    path rather than aborting.
    """
    from src.bet_helpers import get_ctf_readonly

    try:
        _, ctf, _, _ = await asyncio.to_thread(get_ctf_readonly)
        return ctf
    except Exception:
        logger.warning(
            "Could not build CTF readonly connection; falling back to "
            "CLOB-mid heuristic for this resolve_trades tick",
            exc_info=True,
        )
        return None


async def _query_payout_outcome(ctf, condition_id: str) -> bool | None:
    """Off-thread wrapper around ``get_payout_outcome``."""
    from src.bet_helpers import get_payout_outcome

    return await asyncio.to_thread(get_payout_outcome, ctf, condition_id)


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
