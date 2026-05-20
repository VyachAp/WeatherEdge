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

    ctf, funder_address = await _build_ctf_readonly()
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
                # Past grace, ``fill_price IS NULL``. Pre-2026-05-20 this
                # path assumed "order never landed" and marked LOST with
                # ``pnl=-stake``, but that conflated two cases:
                #   * order_id IS NULL / exchange_status='exception:*'
                #     → really never landed; safe to flag LOST.
                #   * exchange_status='matched' with a valid order_id
                #     → DID land on the engine; fill_price=None is just
                #       a reconcile-job backfill gap. The on-chain NO
                #       position is still live and waiting for UMA.
                # The Bug 2026-05-20 phantom-LOST incident wrote off
                # $228 of healthy positions this way. Cross-check the
                # on-chain balance before deciding.
                bal = await _query_token_balance(
                    ctf, funder_address, trade.token_id
                )
                if bal is None:
                    logger.warning(
                        "Trade %s on market %s: balanceOf check failed "
                        "(%.0fh past end_date). Leaving OPEN — operator "
                        "should run `admin reconcile-stuck` to settle "
                        "manually.",
                        trade.id, trade.market_id,
                        (now - end_date).total_seconds() / 3600.0,
                    )
                    continue
                if bal > 0:
                    _backfill_null_fill(trade)
                    logger.warning(
                        "Trade %s on market %s: on-chain position "
                        "exists (%.4f shares) — fill_price was NULL but "
                        "order DID land. Backfilled fill_price=%.3f and "
                        "holding OPEN. UMA still unresolved %.0fh past "
                        "end_date.",
                        trade.id, trade.market_id, bal / 1e6,
                        trade.fill_price or 0.0,
                        (now - end_date).total_seconds() / 3600.0,
                    )
                    continue
                # bal == 0 → order truly never landed. Release exposure
                # without inventing a loss. Wallet wasn't debited; this
                # is accounting cleanup, not a P&L event.
                trade.status = TradeStatus.LOST
                trade.pnl = 0.0
                trade.exit_price = 0.0
                trade.closed_at = now
                resolved.append(trade)
                logger.warning(
                    "Resolved trade %s on market %s (null fill, "
                    "on-chain balance=0, UMA unresolved %.0fh past "
                    "end_date) → LOST (no-op, pnl=0; order never "
                    "landed)",
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
                # Same phantom-LOST guard as branch 2 above. Even without
                # a condition_id we can still ``balanceOf`` the trade's
                # token_id when a CTF handle is available — the legacy
                # fallback is only taken on the CTF-up path (we hit this
                # branch when ``market.condition_id`` is NULL but ``ctf``
                # may still exist). The on-chain position check is the
                # only way to distinguish "matched then dropped" from
                # "never landed".
                bal: int | None = None
                if ctf is not None and funder_address:
                    bal = await _query_token_balance(
                        ctf, funder_address, trade.token_id
                    )
                if bal is not None and bal > 0:
                    _backfill_null_fill(trade)
                    logger.warning(
                        "Trade %s (legacy fallback, market %s): "
                        "on-chain position exists (%.4f shares) — "
                        "backfilled fill_price=%.3f and holding OPEN "
                        "%.0fh past end_date (CLOB dropped market).",
                        trade.id, trade.market_id, bal / 1e6,
                        trade.fill_price or 0.0,
                        (now - end_date).total_seconds() / 3600.0,
                    )
                    continue
                trade.status = TradeStatus.LOST
                # bal == 0 → no money moved. bal is None (no ctf) →
                # we can't tell, but pre-fix behavior was pnl=-stake;
                # preserve that conservative posture only when the
                # chain is fully unreachable.
                trade.pnl = (
                    0.0 if bal == 0 else -float(trade.stake_usd)
                )
                trade.exit_price = 0.0
                trade.closed_at = now
                resolved.append(trade)
                logger.warning(
                    "Resolved trade %s (null fill, legacy fallback) on "
                    "market %s → LOST (pnl=%.2f; bal=%s; CLOB dropped "
                    "market %.0fh past end_date)",
                    trade.id, trade.market_id, trade.pnl,
                    "unknown" if bal is None else str(bal),
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


async def _build_ctf_readonly() -> tuple[object | None, str | None]:
    """Build a read-only CTF contract handle + funder address.

    Wraps the sync ``get_ctf_readonly`` in ``asyncio.to_thread`` so the
    Polygon RPC handshake doesn't block the event loop. Returns
    ``(ctf, funder_address)`` where ``funder_address`` is the proxy/safe
    holding the conditional tokens (``POLYMARKET_FUNDER_ADDRESS`` when
    set, else the EOA derived from the private key). A failure here
    returns ``(None, None)`` and downgrades the whole resolve_trades
    batch to the legacy CLOB-mid path rather than aborting.
    """
    from src.bet_helpers import get_ctf_readonly

    try:
        _, ctf, address, _ = await asyncio.to_thread(get_ctf_readonly)
        funder = settings.POLYMARKET_FUNDER_ADDRESS or address
        return ctf, funder
    except Exception:
        logger.warning(
            "Could not build CTF readonly connection; falling back to "
            "CLOB-mid heuristic for this resolve_trades tick",
            exc_info=True,
        )
        return None, None


async def _query_payout_outcome(ctf, condition_id: str) -> bool | None:
    """Off-thread wrapper around ``get_payout_outcome``."""
    from src.bet_helpers import get_payout_outcome

    return await asyncio.to_thread(get_payout_outcome, ctf, condition_id)


async def _query_token_balance(
    ctf, address: str, token_id: str | None
) -> int | None:
    """Off-thread ``balanceOf`` lookup for a Polymarket conditional token.

    Returns the on-chain balance in atomic units (1e6 per share) or
    ``None`` if the lookup failed (invalid token_id, RPC error). A return
    of ``0`` means the order did NOT land on-chain — the wallet still
    holds the would-be stake. A return ``> 0`` means the order DID land
    and the position is real, regardless of what our DB ``fill_price``
    says (likely a ``job_reconcile_orders`` backfill gap).

    Used by ``resolve_trades`` to gate the null-fill grace fallback —
    without this check, every ``fill_price IS NULL`` trade past the
    4h grace window was being flagged ``LOST`` with ``pnl=-stake``
    even when the on-chain position existed and was still tradeable
    (the phantom-loss bug fixed 2026-05-20).
    """
    if not ctf or not address or not token_id:
        return None
    try:
        tid_int = int(token_id)
    except (TypeError, ValueError):
        return None

    from src.bet_helpers import rpc_call_with_retry

    def _call() -> int:
        return ctf.functions.balanceOf(address, tid_int).call()

    try:
        return await asyncio.to_thread(rpc_call_with_retry, _call)
    except Exception:
        logger.warning(
            "balanceOf failed for trade token_id=%s; treating as unknown",
            token_id,
            exc_info=True,
        )
        return None


def _backfill_null_fill(trade: Trade) -> None:
    """Best-effort fill_price / filled_size backfill from entry_price.

    Used when on-chain ``balanceOf`` confirms a trade DID land but our
    DB row still has ``fill_price IS NULL`` because the SDK couldn't
    return fill details at order time. ``job_reconcile_orders`` will
    refine these with the real fill data on its next tick, but the
    backfill is sufficient for ``_apply_settlement`` to compute PnL.
    """
    if trade.fill_price is None:
        trade.fill_price = float(trade.entry_price or 0.0)
    if trade.filled_size is None and trade.entry_price:
        trade.filled_size = float(trade.stake_usd or 0.0) / float(
            trade.entry_price
        )


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
