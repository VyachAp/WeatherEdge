"""CLI entry point for WeatherEdge."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import click


@click.group()
def main() -> None:
    """WeatherEdge — weather-driven Polymarket edge detection."""


@main.command()
def run() -> None:
    """Start the scheduler daemon."""
    from src.scheduler import run_scheduler

    asyncio.run(run_scheduler())


@main.command()
def scan() -> None:
    """One-shot: fetch markets, forecasts, and generate signals."""

    async def _scan() -> None:
        from src.ingestion.polymarket import scan_and_ingest
        from src.scheduler import configure_logging

        configure_logging()
        count = await scan_and_ingest()
        click.echo(f"Scanned {count} weather markets")

    asyncio.run(_scan())


@main.command()
@click.option("--days", default=30, show_default=True, help="Days to backfill.")
def backfill(days: int) -> None:
    """Backfill historical market snapshots."""

    async def _backfill() -> None:
        from src.scheduler import backfill_markets, configure_logging

        configure_logging()
        count = await backfill_markets(days)
        click.echo(f"Backfilled {count} markets over {days} days")

    asyncio.run(_backfill())


@main.command()
def status() -> None:
    """Show current bankroll, open positions, and recent signals."""

    async def _status() -> None:
        from sqlalchemy import select

        from src.config import settings
        from src.db.engine import async_session
        from src.db.models import (
            BankrollLog,
            Signal,
            Trade,
            TradeStatus,
        )

        async with async_session() as session:
            # Bankroll
            row = (
                await session.execute(
                    select(BankrollLog)
                    .order_by(BankrollLog.timestamp.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            bankroll = row.balance if row else settings.INITIAL_BANKROLL
            peak = row.peak if row else bankroll
            dd = row.drawdown_pct if row else 0.0

            click.echo("=== Bankroll ===")
            click.echo(f"  Balance:  ${bankroll:,.2f}")
            click.echo(f"  Peak:     ${peak:,.2f}")
            click.echo(f"  Drawdown: {dd * 100:.1f}%")
            click.echo()

            # Open positions
            open_trades = (
                await session.execute(
                    select(Trade).where(Trade.status == TradeStatus.OPEN)
                )
            ).scalars().all()
            click.echo(f"=== Open Positions ({len(open_trades)}) ===")
            total_exposure = 0.0
            for t in open_trades:
                click.echo(
                    f"  {t.market_id[:12]}…  {t.direction.value:<8} "
                    f"${t.stake_usd:>7.2f}  entry={t.entry_price:.3f}"
                )
                total_exposure += t.stake_usd
            if open_trades:
                click.echo(f"  Total exposure: ${total_exposure:,.2f}")
            click.echo()

            # Recent signals
            signals = (
                await session.execute(
                    select(Signal).order_by(Signal.created_at.desc()).limit(10)
                )
            ).scalars().all()
            click.echo(f"=== Recent Signals ({len(signals)}) ===")
            for s in signals:
                age = datetime.now(timezone.utc) - (
                    s.created_at.replace(tzinfo=timezone.utc)
                    if s.created_at.tzinfo is None
                    else s.created_at
                )
                ago = f"{age.total_seconds() / 3600:.0f}h ago"
                click.echo(
                    f"  {s.market_id[:12]}…  edge={s.edge:+.3f}  "
                    f"{s.direction.value:<8}  conf={s.confidence:.2f}  {ago}"
                )

    asyncio.run(_status())


@main.command("paper-trade")
@click.option("--days", default=30, show_default=True, help="Simulation period in days.")
def paper_trade(days: int) -> None:
    """Simulate trading over historical signals."""

    async def _paper_trade() -> None:
        from sqlalchemy import select

        from src.db.engine import async_session
        from src.db.models import Signal, Trade, TradeStatus
        from src.risk.simulate import SimSignal, simulate_bankroll

        async with async_session() as session:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            signals = (
                await session.execute(
                    select(Signal)
                    .where(Signal.created_at >= cutoff)
                    .order_by(Signal.created_at)
                )
            ).scalars().all()

            if not signals:
                click.echo("No signals found in the given period.")
                return

            sim_signals: list[SimSignal] = []
            for sig in signals:
                trades = (
                    await session.execute(
                        select(Trade).where(Trade.signal_id == sig.id)
                    )
                ).scalars().all()
                outcome = any(t.status == TradeStatus.WON for t in trades)
                sim_signals.append(
                    SimSignal(
                        model_prob=sig.model_prob,
                        market_prob=sig.market_prob,
                        outcome=outcome,
                    )
                )

        result = simulate_bankroll(sim_signals)

        click.echo("=== Paper Trade Results ===")
        click.echo(f"  Period:       {days} days")
        click.echo(f"  Signals:      {len(sim_signals)}")
        click.echo(f"  Trades taken: {result.num_trades}")
        click.echo(f"  Skipped:      {result.num_skipped}")
        click.echo(f"  Final bankroll: ${result.final_bankroll:,.2f}")
        click.echo(f"  Max drawdown:   {result.max_drawdown:.1%}")
        click.echo(f"  Sharpe ratio:   {result.sharpe_ratio:.2f}")
        click.echo(f"  Win rate:       {result.win_rate:.1%}")

    asyncio.run(_paper_trade())


@main.command()
def migrate() -> None:
    """Run pending database migrations (adds missing columns)."""

    async def _migrate() -> None:
        from sqlalchemy import text

        from src.db.engine import engine

        columns = [
            ("trades", "order_id", "VARCHAR"),
            ("trades", "token_id", "VARCHAR"),
            ("trades", "fill_price", "FLOAT"),
            ("trades", "filled_size", "FLOAT"),
            ("trades", "exchange_status", "VARCHAR"),
        ]

        async with engine.begin() as conn:
            for table, col, dtype in columns:
                stmt = f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {dtype}"
                await conn.execute(text(stmt))
                click.echo(f"  OK: {table}.{col}")

        click.echo("Migration complete.")

    asyncio.run(_migrate())


@main.command()
def approve() -> None:
    """Approve USDC + Conditional Token contracts for Polymarket trading.

    Sends 6 on-chain transactions (2 tokens x 3 spender contracts).
    Requires POLYMARKET_PRIVATE_KEY in .env and POL for gas.
    """
    from src.config import settings

    if not settings.POLYMARKET_PRIVATE_KEY:
        click.echo("Error: POLYMARKET_PRIVATE_KEY not set in .env")
        raise SystemExit(1)

    from eth_account import Account
    from web3 import Web3

    RPC_URLS = [
        "https://polygon-bor-rpc.publicnode.com",
        "https://rpc.ankr.com/polygon",
        "https://polygon.drpc.org",
        "https://polygon-rpc.com",
    ]
    CHAIN_ID = 137

    # Token contracts
    USDC = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
    CTF = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"

    # Spender contracts that need approval
    SPENDERS = {
        "CTF Exchange": "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E",
        "Neg Risk CTF Exchange": "0xC5d563A36AE78145C45a50134d48A1215220f80a",
        "Neg Risk Adapter": "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296",
    }

    ERC20_ABI = [
        {
            "inputs": [
                {"name": "spender", "type": "address"},
                {"name": "amount", "type": "uint256"},
            ],
            "name": "approve",
            "outputs": [{"name": "", "type": "bool"}],
            "stateMutability": "nonpayable",
            "type": "function",
        },
        {
            "inputs": [
                {"name": "owner", "type": "address"},
                {"name": "spender", "type": "address"},
            ],
            "name": "allowance",
            "outputs": [{"name": "", "type": "uint256"}],
            "stateMutability": "view",
            "type": "function",
        },
    ]

    ERC1155_ABI = [
        {
            "inputs": [
                {"name": "operator", "type": "address"},
                {"name": "approved", "type": "bool"},
            ],
            "name": "setApprovalForAll",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function",
        },
        {
            "inputs": [
                {"name": "account", "type": "address"},
                {"name": "operator", "type": "address"},
            ],
            "name": "isApprovedForAll",
            "outputs": [{"name": "", "type": "bool"}],
            "stateMutability": "view",
            "type": "function",
        },
    ]

    account = Account.from_key(settings.POLYMARKET_PRIVATE_KEY)
    address = account.address
    max_uint = 2**256 - 1

    click.echo(f"Wallet: {address}")

    # Try RPC endpoints until one works
    w3 = None
    for rpc_url in RPC_URLS:
        try:
            _w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 10}))
            _w3.eth.get_balance(address)  # test connection
            w3 = _w3
            click.echo(f"Connected to {rpc_url}")
            break
        except Exception:
            click.echo(f"  RPC {rpc_url} failed, trying next...")

    if w3 is None:
        click.echo("Error: all Polygon RPC endpoints failed")
        raise SystemExit(1)

    bal = w3.eth.get_balance(address)
    click.echo(f"POL balance: {w3.from_wei(bal, 'ether'):.4f} (for gas)")
    if bal == 0:
        click.echo("Error: wallet has no POL for gas fees")
        raise SystemExit(1)

    usdc = w3.eth.contract(address=Web3.to_checksum_address(USDC), abi=ERC20_ABI)
    ctf = w3.eth.contract(address=Web3.to_checksum_address(CTF), abi=ERC1155_ABI)

    gas_price = w3.eth.gas_price  # eth_gasPrice RPC — no extraData issue
    nonce = w3.eth.get_transaction_count(address)

    def send_tx(tx_data):
        nonlocal nonce
        tx_data["nonce"] = nonce
        tx_data["chainId"] = CHAIN_ID
        tx_data["from"] = address
        tx_data["gasPrice"] = gas_price
        if "gas" not in tx_data:
            tx_data["gas"] = w3.eth.estimate_gas(tx_data)
        signed = w3.eth.account.sign_transaction(tx_data, private_key=settings.POLYMARKET_PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        nonce += 1
        return receipt

    # --- Check existing approvals first ---
    click.echo("\n=== Checking existing approvals ===")
    needs_usdc = []
    needs_ctf = []
    for name, spender in SPENDERS.items():
        spender_cs = Web3.to_checksum_address(spender)
        usdc_ok = usdc.functions.allowance(address, spender_cs).call() > 0
        ctf_ok = ctf.functions.isApprovedForAll(address, spender_cs).call()
        status_u = "OK" if usdc_ok else "NEEDED"
        status_c = "OK" if ctf_ok else "NEEDED"
        click.echo(f"  {name}: USDC={status_u}, CTF={status_c}")
        if not usdc_ok:
            needs_usdc.append((name, spender_cs))
        if not ctf_ok:
            needs_ctf.append((name, spender_cs))

    if not needs_usdc and not needs_ctf:
        click.echo("\nAll approvals already in place!")
        return

    total_txs = len(needs_usdc) + len(needs_ctf)
    click.echo(f"\nNeed {total_txs} approval transaction(s). Proceeding...")

    # --- USDC approvals ---
    for name, spender in needs_usdc:
        click.echo(f"  Approving USDC for {name}...", nl=False)
        tx = usdc.functions.approve(spender, max_uint).build_transaction({"from": address, "gasPrice": gas_price})
        receipt = send_tx(tx)
        ok = "OK" if receipt["status"] == 1 else "FAILED"
        click.echo(f" {ok} (tx: {receipt['transactionHash'].hex()[:16]}...)")

    # --- CTF approvals ---
    for name, spender in needs_ctf:
        click.echo(f"  Approving CTF for {name}...", nl=False)
        tx = ctf.functions.setApprovalForAll(spender, True).build_transaction({"from": address, "gasPrice": gas_price})
        receipt = send_tx(tx)
        ok = "OK" if receipt["status"] == 1 else "FAILED"
        click.echo(f" {ok} (tx: {receipt['transactionHash'].hex()[:16]}...)")

    # --- Notify CLOB server ---
    click.echo("\nNotifying Polymarket CLOB server...")
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import AssetType, BalanceAllowanceParams

    client = ClobClient(
        settings.POLYMARKET_HOST,
        key=settings.POLYMARKET_PRIVATE_KEY,
        chain_id=settings.POLYMARKET_CHAIN_ID,
    )
    creds = client.create_or_derive_api_creds()
    client.set_api_creds(creds)
    client.update_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
    click.echo("Done! You can now trade on Polymarket.")


@main.command("test-trade")
@click.option("--amount", default=1.0, show_default=True, help="USDC amount to spend.")
def test_trade(amount: float) -> None:
    """Place a tiny test trade on the most liquid weather market.

    Uses a FOK market order. Verifies the full execution pipeline works.
    """

    async def _test_trade() -> None:
        from src.config import settings

        if not settings.POLYMARKET_PRIVATE_KEY:
            click.echo("Error: POLYMARKET_PRIVATE_KEY not set in .env")
            raise SystemExit(1)

        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import MarketOrderArgs, OrderType

        client = ClobClient(
            settings.POLYMARKET_HOST,
            key=settings.POLYMARKET_PRIVATE_KEY,
            chain_id=settings.POLYMARKET_CHAIN_ID,
        )
        creds = client.create_or_derive_api_creds()
        client.set_api_creds(creds)

        import httpx

        click.echo("Finding a tradeable market...")

        # Search for markets with valid CLOB token IDs
        found_market = None
        found_token = None
        for search_tag in ["weather", "climate", None]:
            params = {"limit": 50, "active": "true", "order": "liquidity", "ascending": "false"}
            if search_tag:
                params["tag"] = search_tag

            async with httpx.AsyncClient() as http:
                resp = await http.get(
                    "https://gamma-api.polymarket.com/markets",
                    params=params,
                    timeout=15,
                )
                resp.raise_for_status()
                markets = resp.json()

            for m in markets:
                token_ids = m.get("clobTokenIds") or []
                if len(token_ids) >= 2 and token_ids[0]:
                    found_market = m
                    found_token = token_ids[0]
                    break

            if found_market:
                if search_tag:
                    click.echo(f"Found via tag '{search_tag}'")
                break

        if not found_market or not found_token:
            click.echo("No tradeable market found with valid token IDs.")
            return

        click.echo(f"Market: {found_market.get('question', 'unknown')[:80]}")
        click.echo(f"YES token: {found_token[:20]}...")
        click.echo(f"Amount: ${amount:.2f}")

        yes_token = found_token
        tick_size = client.get_tick_size(yes_token)
        neg_risk = client.get_neg_risk(yes_token)

        from py_clob_client.order_builder.constants import BUY

        # Place a small FOK market buy on YES
        click.echo("Placing FOK market order...")
        market_order = MarketOrderArgs(token_id=yes_token, amount=amount, side=BUY)
        signed = client.create_market_order(market_order)
        resp = client.post_order(signed, OrderType.FOK)

        click.echo(f"Response: {resp}")

        if resp.get("orderID"):
            click.echo(f"\nOrder ID: {resp['orderID']}")
            click.echo(f"Status: {resp.get('status', 'unknown')}")
            click.echo("\nTest trade successful! The execution pipeline works.")
        else:
            click.echo(f"\nOrder failed: {resp.get('errorMsg', 'unknown error')}")
            click.echo("This may be normal for FOK on a thin book. The API connection works.")

    asyncio.run(_test_trade())


if __name__ == "__main__":
    main()
