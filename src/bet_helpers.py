"""Helpers for the manual bet CLI commands.

Handles market resolution from URLs/slugs/IDs, balance checks,
CLOB client init, and display formatting.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Callable, Optional, TypeVar
from urllib.parse import urlparse

import httpx

from src.config import settings

logger = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"


# ---------------------------------------------------------------------------
# Market identifier parsing
# ---------------------------------------------------------------------------


def parse_market_identifier(raw: str) -> str:
    """Extract a usable identifier from a URL, slug, or condition_id.

    Supports:
      - https://polymarket.com/event/<slug>
      - https://polymarket.com/event/<slug>/<sub-slug>
      - bare slug string
      - condition_id (hex string)
    """
    raw = raw.strip()
    if raw.startswith("http"):
        parsed = urlparse(raw)
        # Path like /event/slug or /event/slug/sub-slug
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) >= 2:
            return parts[-1]  # last segment
        if parts:
            return parts[0]
    return raw


# ---------------------------------------------------------------------------
# Gamma API lookups
# ---------------------------------------------------------------------------


async def resolve_market(raw: str) -> dict | None:
    """Resolve a market identifier to a Gamma API market dict.

    Tries multiple lookup strategies: by id, by slug, by direct path.
    """
    identifier = parse_market_identifier(raw)

    async with httpx.AsyncClient(timeout=15) as http:
        # Try by numeric id
        try:
            resp = await http.get(
                f"{GAMMA_BASE}/markets",
                params={"id": identifier},
            )
            resp.raise_for_status()
            data = resp.json()
            if data:
                return data[0] if isinstance(data, list) else data
        except Exception:
            pass

        # Try by conditionId (0x... hex string)
        if identifier.startswith("0x"):
            try:
                resp = await http.get(
                    f"{GAMMA_BASE}/markets",
                    params={"conditionId": identifier},
                )
                resp.raise_for_status()
                data = resp.json()
                if data:
                    return data[0] if isinstance(data, list) else data
            except Exception:
                pass

        # Try by slug param
        try:
            resp = await http.get(
                f"{GAMMA_BASE}/markets",
                params={"slug": identifier},
            )
            resp.raise_for_status()
            data = resp.json()
            if data:
                return data[0] if isinstance(data, list) else data
        except Exception:
            pass

        # Try direct path lookup
        try:
            resp = await http.get(f"{GAMMA_BASE}/markets/{identifier}")
            resp.raise_for_status()
            data = resp.json()
            if data:
                return data if isinstance(data, dict) else data[0]
        except Exception:
            pass

    return None


async def search_markets(query: str, limit: int = 20) -> list[dict]:
    """Search active Polymarket markets by keyword.

    Paginates through all active markets and filters client-side,
    since the Gamma API has no server-side text search.
    """
    query_lower = query.lower()
    results: list[dict] = []
    page_size = 100
    offset = 0

    async with httpx.AsyncClient(timeout=30) as http:
        while len(results) < limit:
            resp = await http.get(
                f"{GAMMA_BASE}/markets",
                params={
                    "active": "true",
                    "closed": "false",
                    "limit": page_size,
                    "offset": offset,
                },
            )
            resp.raise_for_status()
            batch = resp.json()

            if not batch:
                break

            for m in batch:
                if query_lower in (m.get("question") or "").lower():
                    results.append(m)
                    if len(results) >= limit:
                        break

            if len(batch) < page_size:
                break
            offset += page_size

    return results


# ---------------------------------------------------------------------------
# Token ID extraction
# ---------------------------------------------------------------------------


def extract_token_ids(market: dict) -> tuple[str, str] | None:
    """Parse clobTokenIds from a Gamma API market dict.

    Returns (yes_token_id, no_token_id) or None.
    """
    token_ids = market.get("clobTokenIds", [])
    if isinstance(token_ids, str):
        token_ids = json.loads(token_ids)
    if len(token_ids) < 2 or not token_ids[0]:
        return None
    return (token_ids[0], token_ids[1])


# ---------------------------------------------------------------------------
# USDC balance
# ---------------------------------------------------------------------------


async def get_usdc_balance(private_key: str) -> tuple[float, float, float]:
    """Check on-chain pUSD, USDC.e, and native USDC balances at the funder
    address (proxy/safe if configured, else the EOA).

    Returns (pusd_usd, usdc_e_usd, native_usdc_usd).
    """
    from eth_account import Account
    from web3 import Web3

    PUSD = "0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB"
    USDC_E = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
    USDC_NATIVE = "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359"
    BAL_ABI = [
        {
            "inputs": [{"name": "account", "type": "address"}],
            "name": "balanceOf",
            "outputs": [{"name": "", "type": "uint256"}],
            "stateMutability": "view",
            "type": "function",
        }
    ]

    account = Account.from_key(private_key)
    funder = settings.POLYMARKET_FUNDER_ADDRESS or account.address
    addr = Web3.to_checksum_address(funder)

    w3 = Web3(
        Web3.HTTPProvider(
            "https://polygon-bor-rpc.publicnode.com",
            request_kwargs={"timeout": _RPC_TIMEOUT_SECONDS},
        )
    )

    pusd = w3.eth.contract(
        address=Web3.to_checksum_address(PUSD), abi=BAL_ABI
    )
    usdc_e = w3.eth.contract(
        address=Web3.to_checksum_address(USDC_E), abi=BAL_ABI
    )
    usdc_n = w3.eth.contract(
        address=Web3.to_checksum_address(USDC_NATIVE), abi=BAL_ABI
    )

    pusd_bal = pusd.functions.balanceOf(addr).call() / 1e6
    usdc_e_bal = usdc_e.functions.balanceOf(addr).call() / 1e6
    usdc_n_bal = usdc_n.functions.balanceOf(addr).call() / 1e6

    return (pusd_bal, usdc_e_bal, usdc_n_bal)


# ---------------------------------------------------------------------------
# CLOB client (manual mode — ignores AUTO_EXECUTE)
# ---------------------------------------------------------------------------


CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"

_POLYGON_RPC_URLS = (
    "https://polygon-bor-rpc.publicnode.com",
    "https://rpc.ankr.com/polygon",
    "https://polygon.drpc.org",
    "https://polygon-rpc.com",
)

# 30s allows slow eth_getTransactionReceipt polls to complete on a
# congested RPC; the previous 10s tripped on every receipt wait against
# publicnode and was the headline cause of "Read timed out" errors in
# `bet redeem`.
_RPC_TIMEOUT_SECONDS = 30

_TRANSIENT_RPC_SIGNALS = (
    "timed out",
    "timeout",
    "connection aborted",
    "connection reset",
    "remote end closed",
    "max retries exceeded",
    "bad gateway",
    "service unavailable",
    "too many requests",
    "read timed out",
)


def is_transient_rpc_error(exc: BaseException) -> bool:
    """Heuristic: True when `exc` is a retryable HTTP/RPC failure."""
    msg = str(exc).lower()
    return any(s in msg for s in _TRANSIENT_RPC_SIGNALS)


_T = TypeVar("_T")


def rpc_call_with_retry(
    call_fn: Callable[[], _T],
    *,
    on_transient: Optional[Callable[[BaseException, int], None]] = None,
    max_attempts: int = 3,
    initial_backoff: float = 1.5,
) -> _T:
    """Run `call_fn()` with exponential-backoff retry on transient RPC errors.

    `on_transient(exc, attempt)` runs between attempts so the caller can
    rotate to a different RPC endpoint before the next try. Permanent
    errors (revert, "nonce too low", invalid signature) propagate up
    immediately so the caller can react.
    """
    backoff = initial_backoff
    for attempt in range(1, max_attempts + 1):
        try:
            return call_fn()
        except Exception as exc:
            if not is_transient_rpc_error(exc) or attempt == max_attempts:
                raise
            time.sleep(backoff)
            if on_transient is not None:
                try:
                    on_transient(exc, attempt)
                except Exception:
                    pass
            backoff *= 2
    raise RuntimeError("rpc_call_with_retry: unreachable")

_CTF_BALANCE_ABI = [
    {
        "inputs": [
            {"name": "account", "type": "address"},
            {"name": "id", "type": "uint256"},
        ],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"name": "collateralToken", "type": "address"},
            {"name": "parentCollectionId", "type": "bytes32"},
            {"name": "conditionId", "type": "bytes32"},
            {"name": "indexSets", "type": "uint256[]"},
        ],
        "name": "redeemPositions",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
]


def get_ctf_readonly(skip_url: str | None = None):
    """Connect to Polygon with RPC failover and return (w3, ctf, address, rpc_url).

    Read-only bootstrap shared by portfolio and redeem flows — does not
    check the POL gas balance; callers that submit transactions must do
    that themselves.

    Pass `skip_url` to rotate the failover order so the previous (failing)
    endpoint is tried last on a reconnect attempt.
    """
    from eth_account import Account
    from web3 import Web3

    account = Account.from_key(settings.POLYMARKET_PRIVATE_KEY)
    address = account.address

    urls = list(_POLYGON_RPC_URLS)
    if skip_url and skip_url in urls:
        idx = urls.index(skip_url)
        urls = urls[idx + 1:] + urls[: idx + 1]

    last_err: Exception | None = None
    for rpc_url in urls:
        try:
            w3 = Web3(
                Web3.HTTPProvider(
                    rpc_url, request_kwargs={"timeout": _RPC_TIMEOUT_SECONDS}
                )
            )
            w3.eth.get_balance(address)
            ctf = w3.eth.contract(
                address=Web3.to_checksum_address(CTF_ADDRESS),
                abi=_CTF_BALANCE_ABI,
            )
            return w3, ctf, address, rpc_url
        except Exception as exc:
            last_err = exc
            continue

    raise RuntimeError(f"all Polygon RPC endpoints failed (last: {last_err})")


def get_clob_client():
    """Initialise a ClobClient for manual bet placement.

    Only requires POLYMARKET_PRIVATE_KEY (AUTO_EXECUTE is irrelevant).
    Honours POLYMARKET_SIGNATURE_TYPE / POLYMARKET_FUNDER_ADDRESS for
    UI-onboarded wallets.
    """
    from src.execution.polymarket_client import build_clob_client

    client = build_clob_client()
    if client is None:
        raise RuntimeError("POLYMARKET_PRIVATE_KEY is not set")
    return client


# ---------------------------------------------------------------------------
# Display formatting
# ---------------------------------------------------------------------------


def format_market_info(market: dict) -> str:
    """Format a Gamma API market dict for terminal display."""
    question = market.get("question", "Unknown")
    slug = market.get("slug", market.get("id", ""))

    # Parse prices
    outcome_prices = market.get("outcomePrices", "[]")
    if isinstance(outcome_prices, str):
        try:
            outcome_prices = json.loads(outcome_prices)
        except (json.JSONDecodeError, TypeError):
            outcome_prices = []

    yes_price = float(outcome_prices[0]) if len(outcome_prices) > 0 else None
    no_price = float(outcome_prices[1]) if len(outcome_prices) > 1 else None

    volume = market.get("volume") or market.get("volumeNum")
    liquidity = market.get("liquidity") or market.get("liquidityNum")
    end_date = market.get("endDate", "")
    condition_id = market.get("conditionId", market.get("id", ""))

    lines = [
        f"  Question:  {question}",
    ]
    if yes_price is not None:
        lines.append(f"  YES price: {yes_price:.2%}")
    if no_price is not None:
        lines.append(f"  NO price:  {no_price:.2%}")
    if volume:
        lines.append(f"  Volume:    ${float(volume):,.0f}")
    if liquidity:
        lines.append(f"  Liquidity: ${float(liquidity):,.0f}")
    if end_date:
        lines.append(f"  End date:  {end_date}")
    lines.append(f"  ID:        {condition_id}")
    lines.append(f"  URL:       https://polymarket.com/event/{slug}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Portfolio / trade history
# ---------------------------------------------------------------------------


def get_trades_history(client) -> list[dict]:
    """Fetch all trade history from the CLOB API."""
    from py_clob_client_v2.clob_types import TradeParams
    return client.get_trades(TradeParams())


def get_open_orders(client) -> list[dict]:
    """Fetch all open orders from the CLOB API."""
    from py_clob_client_v2.clob_types import OpenOrderParams
    return client.get_open_orders(OpenOrderParams())


async def enrich_trades_with_markets(trades: list[dict]) -> dict[str, dict]:
    """Fetch market info for all unique condition_ids in trades.

    Returns {condition_id: market_dict}.
    """
    condition_ids = {t.get("market") or t.get("condition_id", "") for t in trades}
    condition_ids.discard("")

    markets: dict[str, dict] = {}

    async with httpx.AsyncClient(timeout=15) as http:
        for cid in condition_ids:
            # CLOB API reliably resolves by condition_id (works for neg-risk)
            try:
                resp = await http.get(
                    f"https://clob.polymarket.com/markets/{cid}",
                )
                resp.raise_for_status()
                mkt = resp.json()
                if mkt and "condition_id" in mkt:
                    markets[cid] = mkt
                    continue
            except Exception:
                pass

            # Fallback: Gamma API
            try:
                resp = await http.get(
                    f"{GAMMA_BASE}/markets",
                    params={"condition_id": cid},
                )
                resp.raise_for_status()
                data = resp.json()
                if data:
                    mkt = data[0] if isinstance(data, list) else data
                    markets[cid] = mkt
            except Exception:
                pass

    return markets


def compute_positions(trades: list[dict]) -> dict[str, dict]:
    """Aggregate trades into net positions by asset_id (token).

    Returns {asset_id: {side, size, avg_price, cost, market}}.
    """
    positions: dict[str, dict] = {}

    for t in trades:
        asset_id = t.get("asset_id", "")
        if not asset_id:
            continue

        side = t.get("side", "")
        size = float(t.get("size", 0))
        price = float(t.get("price", 0))
        market = t.get("market", "")

        if asset_id not in positions:
            positions[asset_id] = {
                "asset_id": asset_id,
                "side": side,
                "size": 0.0,
                "cost": 0.0,
                "market": market,
            }

        pos = positions[asset_id]
        if side == "BUY":
            pos["cost"] += size * price
            pos["size"] += size
        else:  # SELL
            pos["cost"] -= size * price
            pos["size"] -= size

    # Calculate average entry price and filter out closed positions
    active: dict[str, dict] = {}
    for asset_id, pos in positions.items():
        if abs(pos["size"]) < 0.001:
            continue  # fully closed
        if pos["size"] > 0:
            pos["avg_price"] = pos["cost"] / pos["size"] if pos["size"] else 0
            pos["side"] = "LONG"
        else:
            pos["avg_price"] = pos["cost"] / pos["size"] if pos["size"] else 0
            pos["side"] = "SHORT"
        active[asset_id] = pos

    return active
