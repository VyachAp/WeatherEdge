"""Polymarket CLOB order execution client.

Handles order placement, status tracking, and daily spend accounting.
Runs in dry-run mode when POLYMARKET_PRIVATE_KEY is not configured.
"""

from __future__ import annotations

import logging
import time as _time
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
# Polymarket V2 migration
# ---------------------------------------------------------------------------
#
# Polymarket migrated CLOB to V2 on 2026-04-28: new exchange contracts,
# new collateral (pUSD instead of USDC.e), new EIP-712 domain version "2",
# and a new Order struct (drops taker/expiration/nonce/feeRateBps; adds
# timestamp/metadata/builder). The legacy ``py-clob-client`` v0.34.x is
# permanently incompatible — every signed order is rejected with
# ``order_version_mismatch``. The replacement SDK is the separate package
# ``py-clob-client-v2``, which we now use everywhere.
#
# V2 ``get_contract_config(137)`` returns one struct with both V1 and V2
# fields populated. Callers that need to know "what address is the active
# regular exchange?" should read ``exchange_v2`` / ``neg_risk_exchange_v2``
# from this single source of truth — do not hard-code addresses elsewhere.

# ---------------------------------------------------------------------------
# Cloudflare HTML noise filter
# ---------------------------------------------------------------------------
#
# When the SDK posts to ``/auth/api-key`` and Cloudflare rate-limits the IP,
# it returns a 403 with the full Cloudflare blocked-page HTML in the body.
# The SDK's request layer dumps that whole HTML body to logger.error and the
# CLI then transparently falls back to ``derive_api_key`` (a GET, which
# Cloudflare doesn't block). Net effect: every CLI invocation prints ~5KB of
# Cloudflare HTML even though the call succeeded. Suppress those specific
# error records while leaving legitimate API errors intact.

class _CloudflareNoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            return True
        return "<!DOCTYPE html>" not in msg and "Cloudflare" not in msg


def _install_cloudflare_filter() -> None:
    """Attach the noise filter once. Idempotent."""
    target = logging.getLogger("py_clob_client_v2.http_helpers.helpers")
    if not any(isinstance(f, _CloudflareNoiseFilter) for f in target.filters):
        target.addFilter(_CloudflareNoiseFilter())
    # The SDK module also uses the package-root logger as a parent in some
    # versions — cover both.
    root = logging.getLogger("py_clob_client_v2")
    if not any(isinstance(f, _CloudflareNoiseFilter) for f in root.filters):
        root.addFilter(_CloudflareNoiseFilter())


_install_cloudflare_filter()


# ---------------------------------------------------------------------------
# Module-level client singleton
# ---------------------------------------------------------------------------

_client = None
_api_creds_set = False
_version_mismatch_alerted = False  # one-shot alert dedup per process


def build_clob_client():
    """Build a fresh V2 ClobClient using the configured signature_type / funder.

    Returns None when ``POLYMARKET_PRIVATE_KEY`` is empty (dry-run mode).

    The funder + signature_type pair must match what Polymarket has registered
    for this wallet. For a wallet that has never logged into polymarket.com,
    leave the defaults (``POLYMARKET_SIGNATURE_TYPE=0`` and an empty
    ``POLYMARKET_FUNDER_ADDRESS``) — funder is then derived from the EOA.
    For a UI-onboarded wallet, set ``POLYMARKET_SIGNATURE_TYPE=2`` and
    ``POLYMARKET_FUNDER_ADDRESS=<proxy address>``.
    """
    if not settings.POLYMARKET_PRIVATE_KEY:
        return None

    from eth_account import Account
    from py_clob_client_v2.client import ClobClient

    eoa_address = Account.from_key(settings.POLYMARKET_PRIVATE_KEY).address
    funder_address = settings.POLYMARKET_FUNDER_ADDRESS or eoa_address
    signature_type = settings.POLYMARKET_SIGNATURE_TYPE

    # V2 ClobClient: chain_id is positional arg #2 (was kwarg in V1).
    temp_client = ClobClient(
        settings.POLYMARKET_HOST,
        settings.POLYMARKET_CHAIN_ID,
        key=settings.POLYMARKET_PRIVATE_KEY,
    )
    creds = temp_client.create_or_derive_api_key()

    return ClobClient(
        settings.POLYMARKET_HOST,
        settings.POLYMARKET_CHAIN_ID,
        key=settings.POLYMARKET_PRIVATE_KEY,
        creds=creds,
        signature_type=signature_type,
        funder=funder_address,
    )


def _get_client():
    """Lazily initialise the CLOB client singleton."""
    global _client, _api_creds_set  # noqa: PLW0603

    if _client is not None:
        return _client

    _client = build_clob_client()
    if _client is None:
        return None

    _api_creds_set = True
    from eth_account import Account
    eoa_address = Account.from_key(settings.POLYMARKET_PRIVATE_KEY).address
    funder = settings.POLYMARKET_FUNDER_ADDRESS or eoa_address
    logger.info(
        "Polymarket CLOB client initialised (eoa=%s funder=%s sig_type=%d)",
        eoa_address,
        funder,
        settings.POLYMARKET_SIGNATURE_TYPE,
    )

    return _client


def is_live() -> bool:
    """Return True if live execution is configured and enabled."""
    return bool(settings.POLYMARKET_PRIVATE_KEY) and settings.AUTO_EXECUTE


# ---------------------------------------------------------------------------
# Wallet balance
# ---------------------------------------------------------------------------

_WALLET_BALANCE_TTL_SEC = 300.0  # cache USDC balance lookups for 5 min
_wallet_balance_cache: tuple[float, float] | None = None  # (fetched_at, usdc)


def get_wallet_usdc_balance(force_refresh: bool = False) -> float | None:
    """Return spendable USDC balance for the configured wallet.

    Returns None when no private key is configured (dry-run) or the lookup
    fails. Cached for ``_WALLET_BALANCE_TTL_SEC`` seconds.

    The CLOB ``get_balance_allowance`` endpoint returns the on-chain USDC
    balance held by the proxy/EOA in 6-decimal base units. We expose it in
    plain USD so the rest of the bot can size positions against the real
    bankroll instead of a static ``INITIAL_BANKROLL`` setting.
    """
    global _wallet_balance_cache  # noqa: PLW0603

    if not settings.POLYMARKET_PRIVATE_KEY:
        return None

    now = _time.monotonic()
    if not force_refresh and _wallet_balance_cache is not None:
        ts, cached = _wallet_balance_cache
        if now - ts < _WALLET_BALANCE_TTL_SEC:
            return cached

    client = _get_client()
    if client is None:
        return None

    try:
        from py_clob_client_v2.clob_types import AssetType, BalanceAllowanceParams

        resp = client.get_balance_allowance(
            BalanceAllowanceParams(asset_type=AssetType.COLLATERAL),
        )
        # USDC has 6 decimals; the API returns the base-unit string.
        raw = resp.get("balance") if isinstance(resp, dict) else None
        if raw is None:
            return _wallet_balance_cache[1] if _wallet_balance_cache else None
        usdc = float(raw) / 1_000_000.0
        _wallet_balance_cache = (now, usdc)
        return usdc
    except Exception:
        logger.warning("Failed to fetch wallet USDC balance", exc_info=True)
        return _wallet_balance_cache[1] if _wallet_balance_cache else None


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
        if isinstance(token_ids, str):
            import json
            token_ids = json.loads(token_ids)
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
    cutoff = datetime.utcnow() - timedelta(hours=24)
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
    max_slippage_cents: float = 2.0,
    *,
    submit_yes_bid: float | None = None,
    submit_yes_ask: float | None = None,
    submit_depth_usd: float | None = None,
) -> bool:
    """Place a price-limited FAK (immediate-or-cancel) order on Polymarket.

    Buy semantics: we set a per-share limit price = ``trade.entry_price +
    max_slippage_cents/100`` (clamped to [0.001, 0.999]) and post a Fill-
    And-Kill order at that limit. The matching engine sweeps every ask
    priced at-or-below the limit, fills as much as is available, then
    cancels the unfilled remainder. Result: the order naturally walks
    across the price ladder (e.g. 0.95 → 0.96 → 0.97) instead of
    rejecting when the top of book can't cover the full size.

    After the order is posted we read back ``size_matched`` and the
    weighted-average fill price; ``trade.stake_usd`` is updated to
    reflect the actual USDC spent so downstream P&L and exposure
    accounting remain accurate even on partial fills.

    The ``submit_*`` kwargs snapshot the caller's view of the live
    market state at the moment we call this function — they land on the
    Trade row regardless of dry-run / live so post-mortems can decompose
    ``fill_price - entry_price`` into "spread we accepted" vs "depth we
    walked". NULL when the caller didn't supply them (older callers,
    backtests).

    Returns True when the order was successfully posted (regardless of
    whether it filled — partial fills and zero fills both return True so
    the caller can decide what to do based on ``trade.filled_size``).
    Returns False on outright failures (cap exceeded, no token IDs,
    client error).
    """
    # Snapshot submit-time market context first — this lands on the row
    # whether the order ultimately fires (live), no-ops (dry-run), or is
    # rejected (cap_exceeded / no_client / etc.). The diagnostic value of
    # "we wanted to fire, here's what the book looked like" is the same
    # in every case.
    trade.submit_yes_bid = submit_yes_bid
    trade.submit_yes_ask = submit_yes_ask
    trade.submit_depth_usd = submit_depth_usd
    trade.submit_at = datetime.now(timezone.utc)

    # --- Dry-run guard ---
    if not is_live():
        logger.info(
            "DRY-RUN: would place %s on %s for $%.2f at limit %.3f (+%.0f¢ slip)",
            trade.direction.value,
            trade.market_id,
            trade.stake_usd,
            trade.entry_price or 0.0,
            max_slippage_cents,
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

    from py_clob_client_v2.clob_types import OrderArgsV2, OrderType
    from py_clob_client_v2.order_builder.constants import BUY

    from src.db.models import TradeDirection

    token_id = yes_token_id if trade.direction == TradeDirection.BUY_YES else no_token_id
    trade.token_id = token_id

    client = _get_client()
    if client is None:
        trade.exchange_status = "no_client"
        return False

    # Build the price-limited FAK order. The limit price is the highest
    # per-share price we'll pay; FAK lets the matching engine sweep all
    # asks at-or-below that limit and cancel the rest.
    base_price = trade.entry_price or 0.0
    if base_price <= 0:
        # Fallback: read live best ask. Without a price we can't size or limit.
        quote = get_best_bid_ask(token_id)
        if quote is None:
            trade.exchange_status = "no_price"
            return False
        base_price = quote[1]

    limit_price = max(0.001, min(0.999, base_price + max_slippage_cents / 100.0))
    # Snap to the token's tick size so the exchange accepts the order.
    try:
        tick_size = float(client.get_tick_size(token_id))
        if tick_size > 0:
            limit_price = round(limit_price / tick_size) * tick_size
            limit_price = max(tick_size, min(1.0 - tick_size, limit_price))
    except Exception:
        logger.debug("tick-size lookup failed; using raw limit_price", exc_info=True)

    # Worst-case sizing: if every share fills at the limit, we spend
    # exactly stake_usd. Better-priced fills leave us underspent — fine.
    raw_size = trade.stake_usd / limit_price
    # Polymarket requires whole-cent share sizes (typically 2-decimal).
    size_shares = round(raw_size, 2)
    if size_shares <= 0:
        trade.exchange_status = "size_zero"
        return False

    try:
        # V2 OrderArgs: no fee_rate_bps / nonce / taker — those have moved
        # out of the signed struct in the V2 exchange contracts.
        order = OrderArgsV2(
            token_id=token_id,
            price=limit_price,
            size=size_shares,
            side=BUY,
        )
        signed = client.create_order(order)
        resp = client.post_order(signed, OrderType.FAK)

        trade.order_id = resp.get("orderID")
        trade.exchange_status = resp.get("status", "unknown")

        if resp.get("success") or resp.get("orderID"):
            logger.info(
                "Order posted: market=%s side=%s limit=%.3f size=%.2f stake=$%.2f order=%s status=%s",
                trade.market_id, trade.direction.value,
                limit_price, size_shares, trade.stake_usd,
                trade.order_id, trade.exchange_status,
            )
            if trade.order_id:
                await _update_fill_details(trade, client)
                if trade.filled_size and trade.filled_size > 0:
                    logger.info(
                        "Fill: %.2f shares @ avg %.3f → spent $%.2f (limit was %.3f)",
                        trade.filled_size,
                        trade.fill_price or 0.0,
                        trade.stake_usd,
                        limit_price,
                    )
                else:
                    logger.info(
                        "FAK order posted but no fill at limit %.3f (book empty at/below limit)",
                        limit_price,
                    )
            return True

        error_msg = resp.get("errorMsg", "unknown error")
        logger.error("Order failed: market=%s error=%s", trade.market_id, error_msg)
        trade.exchange_status = f"failed: {error_msg}"
        await _maybe_alert_version_mismatch(error_msg, trade.market_id)
        return False

    except Exception as exc:
        logger.exception("Order execution error for market %s", trade.market_id)
        trade.exchange_status = "exception"
        await _maybe_alert_version_mismatch(str(exc), trade.market_id)
        return False


async def _maybe_alert_version_mismatch(message: str, market_id: str) -> None:
    """Fire a one-shot Telegram alert when Polymarket rejects with
    ``order_version_mismatch``.

    Means Polymarket migrated their exchange protocol again and our pinned SDK
    no longer matches the API. The bot will silently reject every signal until
    the SDK is bumped; this alert exists so the operator can react fast.
    """
    global _version_mismatch_alerted  # noqa: PLW0603
    if _version_mismatch_alerted or "order_version_mismatch" not in message.lower():
        return
    _version_mismatch_alerted = True

    try:
        from src.execution.alerter import get_alerter

        alerter = get_alerter()
        err = RuntimeError(
            f"Polymarket rejected order with order_version_mismatch on market {market_id}. "
            f"Polymarket has likely migrated CLOB protocol again. The currently-pinned "
            f"py-clob-client-v2 SDK is no longer compatible. Run "
            f"`python -m src.cli bet diagnose --post-test` to inspect and check PyPI "
            f"for a newer SDK release."
        )
        await alerter.send_system_error(err, context="Polymarket protocol version mismatch")
    except Exception:
        logger.exception("Failed to send order_version_mismatch alert")


async def _update_fill_details(trade: Trade, client) -> None:
    """Read fill details (size, weighted avg price, actual spend) from CLOB.

    For FAK orders that walked the book, ``associate_trades`` is a list
    of individual fills at different prices. We weight by size to get
    the true average and update ``trade.stake_usd`` to the actual amount
    spent — partial fills (or no-fills) reduce stake_usd from the
    requested amount, so exposure and bankroll accounting stay correct.
    """
    try:
        order = client.get_order(trade.order_id)
    except Exception:
        logger.warning("Could not fetch fill details for order %s", trade.order_id)
        return

    fills = order.get("associate_trades") or []
    total_size = 0.0
    total_cost = 0.0
    for f in fills:
        try:
            p = float(f.get("price", 0))
            s = float(f.get("size", 0))
        except (TypeError, ValueError):
            continue
        if p > 0 and s > 0:
            total_size += s
            total_cost += p * s

    # Fall back to size_matched if associate_trades is missing.
    if total_size == 0.0:
        sm = order.get("size_matched")
        if sm:
            try:
                total_size = float(sm)
            except (TypeError, ValueError):
                total_size = 0.0

    trade.filled_size = total_size
    if total_size > 0 and total_cost > 0:
        avg_price = total_cost / total_size
        trade.fill_price = avg_price
        trade.entry_price = avg_price  # the price for P&L
        trade.stake_usd = round(total_cost, 2)
    elif total_size == 0.0:
        # FAK order rejected with no fill. Zero out stake so it doesn't
        # contaminate exposure or daily-spend accounting.
        trade.stake_usd = 0.0


async def check_order_status(trade: Trade) -> str | None:
    """Check current status of an open order. Returns the status string."""
    client = _get_client()
    if client is None or not trade.order_id:
        return None

    try:
        order = client.get_order(trade.order_id)
        status = order.get("status", "unknown")
        trade.exchange_status = status

        if status.lower() == "matched" and not trade.fill_price:
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


# ---------------------------------------------------------------------------
# Orderbook depth – cache, throttle, helpers
# ---------------------------------------------------------------------------

_orderbook_cache: dict[str, tuple[float, object]] = {}
_ORDERBOOK_TTL = 30.0  # seconds

_last_clob_request: float = 0.0
_CLOB_MIN_INTERVAL: float = 0.15  # ~6-7 req/s max

_OB_MAX_RETRIES = 2
_OB_RETRY_BACKOFF = (0.5, 1.5)


def _get_cached_orderbook(token_id: str) -> object | None:
    entry = _orderbook_cache.get(token_id)
    if entry is None:
        return None
    ts, book = entry
    if _time.monotonic() - ts > _ORDERBOOK_TTL:
        del _orderbook_cache[token_id]
        return None
    return book


def _throttle_clob() -> None:
    global _last_clob_request  # noqa: PLW0603
    now = _time.monotonic()
    elapsed = now - _last_clob_request
    if elapsed < _CLOB_MIN_INTERVAL:
        _time.sleep(_CLOB_MIN_INTERVAL - elapsed)
    _last_clob_request = _time.monotonic()


def _compute_depth(book: object, price: float) -> float:
    """USD of liquidity available to a market BUY at limit *price*.

    For a BUY we take the ASK side: any ask with `ask_price <= price`
    is fillable at our limit. Cost is `price * size` per level. Summing
    across all matching asks gives the total USDC we could spend if we
    swept the book at-or-below the limit.

    Verified against py_clob_client semantics 2026-04-28: bids are buy
    orders, asks are sell orders, and the binary CLOB invariant
    `YES.bid + NO.ask = 1.0` holds. See `scripts/inspect_orderbook.py`.
    """
    asks = book.asks if hasattr(book, "asks") else book.get("asks", [])  # type: ignore[union-attr]
    depth = 0.0
    for ask in asks or []:
        ask_price = float(ask.price if hasattr(ask, "price") else ask.get("price", 0))
        ask_size = float(ask.size if hasattr(ask, "size") else ask.get("size", 0))
        if 0 < ask_price <= price:
            depth += ask_price * ask_size
    return depth


# ---------------------------------------------------------------------------
# Orderbook depth – public API
# ---------------------------------------------------------------------------


def get_orderbook_depth(token_id: str, price: float) -> float:
    """Return USD of ask-side liquidity fillable by a market BUY at *price*.

    Sums asks where `ask_price <= price` (cost = ask_price * size per level).
    Use the side we'd actually take when buying: e.g., to BUY NO at 0.45,
    pass `(no_token, 0.45)` and we read NO.asks <= 0.45.

    Uses the CLOB client's ``get_order_book`` method with a 30-second TTL
    cache, inter-request throttling, and retry with backoff.  Returns 0.0
    if the client is unavailable or all attempts fail.
    """
    book = _fetch_orderbook(token_id)
    if book is None:
        return 0.0
    return _compute_depth(book, price)


def _fetch_orderbook(token_id: str):
    """Return a cached or freshly fetched orderbook for *token_id*, or None."""
    client = _get_client()
    if client is None:
        return None

    cached = _get_cached_orderbook(token_id)
    if cached is not None:
        return cached

    for attempt in range(_OB_MAX_RETRIES + 1):
        _throttle_clob()
        try:
            book = client.get_order_book(token_id)
            _orderbook_cache[token_id] = (_time.monotonic(), book)
            return book
        except Exception as exc:
            if attempt < _OB_MAX_RETRIES:
                _time.sleep(_OB_RETRY_BACKOFF[attempt])
            else:
                logger.warning(
                    "Could not fetch orderbook for token %s after %d attempts: %s",
                    token_id, _OB_MAX_RETRIES + 1, exc,
                )
                return None
    return None


def get_best_bid_ask(token_id: str) -> tuple[float, float] | None:
    """Return (best_bid, best_ask) for *token_id*, or None if unavailable.

    Reuses the 30-second orderbook cache, so this piggybacks on the same
    HTTP request already made by ``get_orderbook_depth`` when both are
    called in sequence.
    """
    book = _fetch_orderbook(token_id)
    if book is None:
        return None

    bids = book.bids if hasattr(book, "bids") else book.get("bids", [])  # type: ignore[union-attr]
    asks = book.asks if hasattr(book, "asks") else book.get("asks", [])  # type: ignore[union-attr]
    if not bids or not asks:
        return None

    try:
        best_bid = max(
            float(b.price if hasattr(b, "price") else b.get("price", 0)) for b in bids
        )
        best_ask = min(
            float(a.price if hasattr(a, "price") else a.get("price", 0)) for a in asks
        )
    except (ValueError, TypeError):
        return None

    if best_bid <= 0 or best_ask <= 0 or best_bid > best_ask:
        return None
    return best_bid, best_ask
