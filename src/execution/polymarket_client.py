"""Polymarket CLOB order execution client.

Handles order placement, status tracking, and daily spend accounting.
Runs in dry-run mode when POLYMARKET_PRIVATE_KEY is not configured.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from sqlalchemy import func, select

from src.config import settings
from src.db.engine import async_session
from src.db.models import Trade, TradeStatus

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level client singleton
# ---------------------------------------------------------------------------

_client = None
_api_creds_set = False


def _get_client():
    """Lazily initialise the CLOB client singleton."""
    global _client, _api_creds_set  # noqa: PLW0603

    if _client is not None:
        return _client

    if not settings.POLYMARKET_PRIVATE_KEY:
        return None

    from py_clob_client.client import ClobClient

    _client = ClobClient(
        settings.POLYMARKET_HOST,
        key=settings.POLYMARKET_PRIVATE_KEY,
        chain_id=settings.POLYMARKET_CHAIN_ID,
    )

    if not _api_creds_set:
        creds = _client.create_or_derive_api_creds()
        _client.set_api_creds(creds)
        _api_creds_set = True
        logger.info("Polymarket CLOB client initialised (API creds derived)")

    return _client


def is_live() -> bool:
    """Return True if live execution is configured and enabled."""
    return bool(settings.POLYMARKET_PRIVATE_KEY) and settings.AUTO_EXECUTE


# ---------------------------------------------------------------------------
# Token ID resolution
# ---------------------------------------------------------------------------


async def get_token_ids(market_id: str) -> tuple[str, str] | None:
    """Fetch YES/NO conditional token IDs for a Polymarket market.

    Returns (yes_token_id, no_token_id) or None on failure.
    Token IDs come from the Gamma API, keyed by clobTokenIds.
    """
    import httpx

    try:
        async with httpx.AsyncClient() as http:
            resp = await http.get(
                "https://gamma-api.polymarket.com/markets",
                params={"id": market_id},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

        if not data:
            # Try slug-based lookup
            async with httpx.AsyncClient() as http:
                resp = await http.get(
                    f"https://gamma-api.polymarket.com/markets/{market_id}",
                    timeout=15,
                )
                resp.raise_for_status()
                data = [resp.json()]

        market_data = data[0] if isinstance(data, list) else data

        token_ids = market_data.get("clobTokenIds", [])
        if len(token_ids) < 2:
            logger.warning("Market %s has fewer than 2 token IDs: %s", market_id, token_ids)
            return None

        return (token_ids[0], token_ids[1])  # (YES, NO)

    except Exception:
        logger.exception("Failed to fetch token IDs for market %s", market_id)
        return None


# ---------------------------------------------------------------------------
# Daily spend tracking
# ---------------------------------------------------------------------------


async def get_daily_spend(session: AsyncSession) -> float:
    """Sum of stake_usd for all trades opened in the last 24 hours."""
    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=24)
    stmt = (
        select(func.coalesce(func.sum(Trade.stake_usd), 0.0))
        .where(Trade.opened_at >= cutoff)
        .where(Trade.status.in_([TradeStatus.OPEN, TradeStatus.WON, TradeStatus.LOST]))
    )
    result = await session.execute(stmt)
    return float(result.scalar_one())


# ---------------------------------------------------------------------------
# Order execution
# ---------------------------------------------------------------------------


async def place_order(
    trade: Trade,
    session: AsyncSession,
) -> bool:
    """Place a FOK market order on Polymarket for the given Trade.

    Updates the Trade row with order_id, token_id, fill_price,
    exchange_status. Returns True if the order was successfully placed
    (regardless of fill), False on failure.

    In dry-run mode (no private key or AUTO_EXECUTE=False), logs the
    order details and returns True without touching the exchange.
    """
    # --- Dry-run guard ---
    if not is_live():
        logger.info(
            "DRY-RUN: would place %s order on market %s for $%.2f",
            trade.direction.value,
            trade.market_id,
            trade.stake_usd,
        )
        trade.exchange_status = "dry_run"
        return True

    # --- Daily spend cap ---
    daily_spend = await get_daily_spend(session)
    if daily_spend + trade.stake_usd > settings.DAILY_SPEND_CAP_USD:
        logger.warning(
            "Daily spend cap reached: $%.2f spent + $%.2f new > $%.2f cap",
            daily_spend,
            trade.stake_usd,
            settings.DAILY_SPEND_CAP_USD,
        )
        trade.exchange_status = "cap_exceeded"
        return False

    # --- Resolve token IDs ---
    token_ids = await get_token_ids(trade.market_id)
    if token_ids is None:
        logger.error("Cannot resolve token IDs for market %s", trade.market_id)
        trade.exchange_status = "token_id_error"
        return False

    yes_token_id, no_token_id = token_ids

    from py_clob_client.clob_types import MarketOrderArgs, OrderType
    from py_clob_client.order_builder.constants import BUY

    # BUY_YES → buy YES token; BUY_NO → buy NO token
    from src.db.models import TradeDirection

    if trade.direction == TradeDirection.BUY_YES:
        token_id = yes_token_id
    else:
        token_id = no_token_id

    trade.token_id = token_id

    client = _get_client()
    if client is None:
        trade.exchange_status = "no_client"
        return False

    try:
        # Get tick size and neg_risk for this token
        tick_size = client.get_tick_size(token_id)
        neg_risk = client.get_neg_risk(token_id)

        # Create FOK market order
        market_order = MarketOrderArgs(
            token_id=token_id,
            amount=trade.stake_usd,  # USDC amount to spend
        )
        signed = client.create_market_order(market_order)
        resp = client.post_order(
            signed,
            OrderType.FOK,
            options={"tick_size": tick_size, "neg_risk": neg_risk},
        )

        trade.order_id = resp.get("orderID")
        trade.exchange_status = resp.get("status", "unknown")

        if resp.get("success") or resp.get("orderID"):
            logger.info(
                "Order placed: market=%s direction=%s amount=$%.2f order_id=%s status=%s",
                trade.market_id,
                trade.direction.value,
                trade.stake_usd,
                trade.order_id,
                trade.exchange_status,
            )

            # Fetch fill details if matched
            if trade.order_id and trade.exchange_status == "matched":
                await _update_fill_details(trade, client)

            return True

        error_msg = resp.get("errorMsg", "unknown error")
        logger.error(
            "Order failed: market=%s error=%s", trade.market_id, error_msg,
        )
        trade.exchange_status = f"failed: {error_msg}"
        return False

    except Exception:
        logger.exception("Order execution error for market %s", trade.market_id)
        trade.exchange_status = "exception"
        return False


async def _update_fill_details(trade: Trade, client) -> None:
    """Query the CLOB for fill price and size after a matched order."""
    try:
        order = client.get_order(trade.order_id)
        trade.fill_price = float(order.get("associate_trades", [{}])[0].get("price", 0)) if order.get("associate_trades") else None
        trade.filled_size = float(order.get("size_matched", 0))

        if trade.fill_price:
            trade.entry_price = trade.fill_price

    except Exception:
        logger.warning("Could not fetch fill details for order %s", trade.order_id)


async def check_order_status(trade: Trade) -> str | None:
    """Check current status of an open order. Returns the status string."""
    client = _get_client()
    if client is None or not trade.order_id:
        return None

    try:
        order = client.get_order(trade.order_id)
        status = order.get("status", "unknown")
        trade.exchange_status = status

        if status == "matched" and not trade.fill_price:
            await _update_fill_details(trade, client)

        return status
    except Exception:
        logger.warning("Could not check order %s", trade.order_id)
        return None


async def cancel_order(trade: Trade) -> bool:
    """Cancel an open order on the CLOB."""
    client = _get_client()
    if client is None or not trade.order_id:
        return False

    try:
        client.cancel(trade.order_id)
        trade.exchange_status = "cancelled"
        logger.info("Cancelled order %s for market %s", trade.order_id, trade.market_id)
        return True
    except Exception:
        logger.exception("Failed to cancel order %s", trade.order_id)
        return False
