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
            cutoff = datetime.utcnow() - timedelta(days=days)
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


@main.command("backtest-v2")
@click.option("--days", default=30, show_default=True, help="Days of history to backtest.")
@click.option("--stations", default="", help="Comma-separated ICAO codes (default: all stations with data).")
def backtest_v2(days: int, stations: str) -> None:
    """Backtest the distribution probability engine against historical outcomes."""

    async def _backtest() -> None:
        from src.risk.simulate import simulate_distribution_pipeline

        if stations:
            station_list = [s.strip().upper() for s in stations.split(",")]
        else:
            # Discover stations from DB
            from sqlalchemy import select, distinct
            from src.db.engine import async_session
            from src.db.models import MetarObservation

            async with async_session() as session:
                result = await session.execute(
                    select(distinct(MetarObservation.station_icao))
                )
                station_list = [row[0] for row in result.all() if row[0]]

        if not station_list:
            click.echo("No stations found. Provide --stations or ensure METAR data exists.")
            return

        click.echo(f"Backtesting {len(station_list)} stations over {days} days...")
        result = await simulate_distribution_pipeline(station_list, days_back=days)

        click.echo(f"\n=== Distribution Backtest Results ===")
        click.echo(f"  Days evaluated:     {result.num_days}")
        click.echo(f"  Calibration error:  {result.calibration_error:.4f}")
        click.echo(f"  Brier score:        {result.brier_score:.6f}")

        if result.per_bucket:
            click.echo(f"\n  Per-bucket calibration (top 10 by count):")
            top = sorted(result.per_bucket, key=lambda b: b.count, reverse=True)[:10]
            for b in top:
                click.echo(
                    f"    {b.bucket_value:3d}°F: predicted={b.predicted_avg:.3f} "
                    f"actual={b.actual_rate:.3f} (n={b.count})"
                )

        threshold = 0.03
        if result.calibration_error <= threshold:
            click.echo(f"\n  PASS: calibration error {result.calibration_error:.4f} <= {threshold}")
        else:
            click.echo(f"\n  FAIL: calibration error {result.calibration_error:.4f} > {threshold}")

    asyncio.run(_backtest())


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
    else:
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
    from eth_account import Account
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import AssetType, BalanceAllowanceParams

    account = Account.from_key(settings.POLYMARKET_PRIVATE_KEY)
    funder_address = account.address

    temp_client = ClobClient(
        settings.POLYMARKET_HOST,
        key=settings.POLYMARKET_PRIVATE_KEY,
        chain_id=settings.POLYMARKET_CHAIN_ID,
    )
    creds = temp_client.create_or_derive_api_creds()

    client = ClobClient(
        settings.POLYMARKET_HOST,
        key=settings.POLYMARKET_PRIVATE_KEY,
        chain_id=settings.POLYMARKET_CHAIN_ID,
        creds=creds,
        signature_type=0,
        funder=funder_address,
    )
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

        from eth_account import Account
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import MarketOrderArgs, OrderType

        account = Account.from_key(settings.POLYMARKET_PRIVATE_KEY)
        funder_address = account.address

        temp_client = ClobClient(
            settings.POLYMARKET_HOST,
            key=settings.POLYMARKET_PRIVATE_KEY,
            chain_id=settings.POLYMARKET_CHAIN_ID,
        )
        creds = temp_client.create_or_derive_api_creds()

        client = ClobClient(
            settings.POLYMARKET_HOST,
            key=settings.POLYMARKET_PRIVATE_KEY,
            chain_id=settings.POLYMARKET_CHAIN_ID,
            creds=creds,
            signature_type=0,
            funder=funder_address,
        )

        import httpx
        from web3 import Web3

        # --- Diagnostic: check on-chain USDC.e balance ---
        USDC_E = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
        USDC_NATIVE = "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359"
        BAL_ABI = [{"inputs": [{"name": "account", "type": "address"}], "name": "balanceOf", "outputs": [{"name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"}]

        w3 = Web3(Web3.HTTPProvider("https://polygon-bor-rpc.publicnode.com", request_kwargs={"timeout": 10}))
        addr = Web3.to_checksum_address(funder_address)
        usdc_e_bal = w3.eth.contract(address=Web3.to_checksum_address(USDC_E), abi=BAL_ABI).functions.balanceOf(addr).call()
        usdc_n_bal = w3.eth.contract(address=Web3.to_checksum_address(USDC_NATIVE), abi=BAL_ABI).functions.balanceOf(addr).call()

        click.echo(f"Wallet: {funder_address}")
        click.echo(f"USDC.e balance:      ${usdc_e_bal / 1e6:.2f}  (required by Polymarket)")
        click.echo(f"Native USDC balance: ${usdc_n_bal / 1e6:.2f}")

        if usdc_e_bal == 0 and usdc_n_bal > 0:
            click.echo("\nError: Your USDC is native USDC, but Polymarket requires USDC.e.")
            click.echo("Swap native USDC -> USDC.e on a DEX (e.g. Uniswap on Polygon).")
            raise SystemExit(1)
        if usdc_e_bal == 0:
            click.echo("\nError: No USDC.e on this wallet. Deposit USDC.e to trade.")
            raise SystemExit(1)

        click.echo("Finding a tradeable market...")

        # Search for markets with valid CLOB token IDs
        found_market = None
        found_token = None
        found_tick_size = None
        found_neg_risk = None

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
                if isinstance(token_ids, str):
                    import json
                    token_ids = json.loads(token_ids)
                if len(token_ids) < 2 or not token_ids[0]:
                    continue
                # Validate the token is actually tradeable on the CLOB
                try:
                    ts = client.get_tick_size(token_ids[0])
                    nr = client.get_neg_risk(token_ids[0])
                    found_market = m
                    found_token = token_ids[0]
                    found_tick_size = ts
                    found_neg_risk = nr
                    break
                except Exception as exc:
                    click.echo(f"  Skipping {m.get('question', '?')[:40]}... ({exc})")
                    continue

            if found_market:
                if search_tag:
                    click.echo(f"Found tradeable market via tag '{search_tag}'")
                break

        if not found_market or not found_token:
            click.echo("No tradeable market found with valid CLOB token IDs.")
            return

        click.echo(f"Market: {found_market.get('question', 'unknown')[:80]}")
        click.echo(f"YES token: {found_token[:20]}...")
        click.echo(f"Tick size: {found_tick_size}, Neg risk: {found_neg_risk}")
        click.echo(f"Amount: ${amount:.2f}")

        from py_clob_client.order_builder.constants import BUY

        # Place a small FOK market buy on YES
        click.echo("Placing FOK market order...")
        market_order = MarketOrderArgs(token_id=found_token, amount=amount, side=BUY)
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


@main.group()
def bet() -> None:
    """Place and manage manual bets on Polymarket."""


@bet.command("place")
@click.argument("market")
@click.option("--side", required=True, type=click.Choice(["yes", "no"], case_sensitive=False), help="Buy YES or NO.")
@click.option("--amount", required=True, type=float, help="USDC amount to spend.")
@click.option("--yes", "-y", "skip_confirm", is_flag=True, help="Skip confirmation prompt.")
@click.option("--ignore-cap", is_flag=True, help="Bypass daily spend cap check.")
def bet_place(market: str, side: str, amount: float, skip_confirm: bool, ignore_cap: bool) -> None:
    """Place a FOK market order on any Polymarket market.

    MARKET can be a Polymarket URL, slug, or condition ID.
    """

    async def _place() -> None:
        from src.config import settings
        from src.bet_helpers import (
            extract_token_ids,
            format_market_info,
            get_clob_client,
            get_usdc_balance,
            resolve_market,
        )

        if not settings.POLYMARKET_PRIVATE_KEY:
            click.echo("Error: POLYMARKET_PRIVATE_KEY not set in .env")
            raise SystemExit(1)

        # --- Resolve market ---
        click.echo(f"Resolving market: {market}")
        mkt = await resolve_market(market)
        if mkt is None:
            click.echo("Error: market not found.")
            raise SystemExit(1)

        # --- Extract token IDs ---
        token_pair = extract_token_ids(mkt)
        if token_pair is None:
            click.echo("Error: market has no tradeable token IDs.")
            raise SystemExit(1)

        yes_token, no_token = token_pair
        token_id = yes_token if side.lower() == "yes" else no_token

        # --- Check USDC balance ---
        click.echo("Checking wallet balance...")
        pusd, usdc_e, usdc_native = await get_usdc_balance(settings.POLYMARKET_PRIVATE_KEY)
        click.echo(f"  pUSD:        ${pusd:.2f}")
        click.echo(f"  USDC.e:      ${usdc_e:.2f}")
        click.echo(f"  Native USDC: ${usdc_native:.2f}")

        if usdc_e < amount:
            click.echo(f"\nError: insufficient USDC.e (${usdc_e:.2f} < ${amount:.2f}).")
            if usdc_native >= amount:
                click.echo("You have native USDC — swap to USDC.e on a DEX first.")
            raise SystemExit(1)

        # --- Daily spend cap ---
        if not ignore_cap:
            from src.db.engine import async_session
            from src.execution.polymarket_client import get_daily_spend

            async with async_session() as session:
                daily_spend = await get_daily_spend(session)

            remaining = settings.DAILY_SPEND_CAP_USD - daily_spend
            click.echo(f"  24h spend:   ${daily_spend:.2f} / ${settings.DAILY_SPEND_CAP_USD:.2f}")
            if daily_spend + amount > settings.DAILY_SPEND_CAP_USD:
                click.echo(
                    f"\nError: daily spend cap would be exceeded "
                    f"(${daily_spend:.2f} + ${amount:.2f} > ${settings.DAILY_SPEND_CAP_USD:.2f}). "
                    f"Use --ignore-cap to override."
                )
                raise SystemExit(1)

        # --- Display market info ---
        click.echo(f"\n=== Market ===")
        click.echo(format_market_info(mkt))
        click.echo(f"\n=== Order ===")
        click.echo(f"  Side:   BUY {side.upper()}")
        click.echo(f"  Amount: ${amount:.2f}")
        click.echo(f"  Type:   FOK (Fill-or-Kill)")

        # --- Confirmation ---
        if not skip_confirm:
            click.echo()
            if not click.confirm("Place this order?"):
                click.echo("Cancelled.")
                return

        # --- Place order ---
        click.echo("\nInitialising CLOB client...")
        client = get_clob_client()

        # Validate token is tradeable
        try:
            tick_size = client.get_tick_size(token_id)
            neg_risk = client.get_neg_risk(token_id)
        except Exception as exc:
            click.echo(f"Error: token not tradeable on CLOB ({exc})")
            raise SystemExit(1)

        click.echo(f"  Tick size: {tick_size}, Neg risk: {neg_risk}")

        from py_clob_client.clob_types import MarketOrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY

        click.echo("Placing FOK market order...")
        market_order = MarketOrderArgs(token_id=token_id, amount=amount, side=BUY)
        signed = client.create_market_order(market_order)
        resp = client.post_order(signed, OrderType.FOK)

        # --- Display result ---
        click.echo(f"\n=== Result ===")
        order_id = resp.get("orderID")
        status = (resp.get("status") or "unknown").lower()

        if order_id:
            click.echo(f"  Order ID: {order_id}")
            click.echo(f"  Status:   {status}")

            if status == "matched":
                try:
                    order = client.get_order(order_id)
                    trades = order.get("associate_trades", [])
                    if trades:
                        fill_price = float(trades[0].get("price", 0))
                        click.echo(f"  Fill price: {fill_price:.4f}")
                    size_matched = order.get("size_matched")
                    if size_matched:
                        click.echo(f"  Shares:     {float(size_matched):.2f}")
                except Exception:
                    click.echo("  (could not fetch fill details)")

                click.echo("\nOrder filled successfully.")

            elif status == "delayed":
                import time

                click.echo("\nOrder accepted but not yet filled. Polling for fill...")
                filled = False
                for attempt in range(3):
                    time.sleep(2)
                    try:
                        order = client.get_order(order_id)
                        current_status = order.get("status", "unknown")
                        click.echo(f"  [{attempt + 1}/3] Status: {current_status}")
                        if current_status.lower() == "matched":
                            try:
                                trades = order.get("associate_trades", [])
                                if trades:
                                    fill_price = float(trades[0].get("price", 0))
                                    click.echo(f"  Fill price: {fill_price:.4f}")
                                size_matched = order.get("size_matched")
                                if size_matched:
                                    click.echo(f"  Shares:     {float(size_matched):.2f}")
                            except Exception:
                                click.echo("  (could not fetch fill details)")
                            click.echo("\nOrder filled successfully.")
                            filled = True
                            break
                    except Exception:
                        click.echo(f"  [{attempt + 1}/3] (could not fetch status)")

                if not filled:
                    click.echo(f"\nOrder is still delayed (not filled).")
                    click.echo(f"  To cancel: python -m src.cli bet cancel {order_id}")
                    if click.confirm("Cancel this order now?"):
                        try:
                            client.cancel(order_id)
                            click.echo("Order cancelled.")
                        except Exception as exc:
                            click.echo(f"Failed to cancel: {exc}")
                    else:
                        click.echo("Order left open. Check later with: python -m src.cli bet orders")
            else:
                click.echo(f"\nOrder placed (status: {status}).")
        else:
            error_msg = resp.get("errorMsg", "unknown error")
            click.echo(f"  Status: FAILED")
            click.echo(f"  Error:  {error_msg}")
            click.echo("\nThe order was not filled. This may be normal for FOK on a thin order book.")

    asyncio.run(_place())


@bet.command("info")
@click.argument("market")
def bet_info(market: str) -> None:
    """Display details about a Polymarket market.

    MARKET can be a Polymarket URL, slug, or condition ID.
    """

    async def _info() -> None:
        from src.bet_helpers import (
            extract_token_ids,
            format_market_info,
            resolve_market,
        )

        click.echo(f"Resolving market: {market}")
        mkt = await resolve_market(market)
        if mkt is None:
            click.echo("Error: market not found.")
            raise SystemExit(1)

        click.echo(f"\n=== Market Info ===")
        click.echo(format_market_info(mkt))

        token_pair = extract_token_ids(mkt)
        if token_pair:
            yes_token, no_token = token_pair
            click.echo(f"\n=== Token IDs ===")
            click.echo(f"  YES: {yes_token}")
            click.echo(f"  NO:  {no_token}")
        else:
            click.echo("\n  No tradeable token IDs found.")

    asyncio.run(_info())


@bet.command("search")
@click.argument("query")
@click.option("--limit", default=10, show_default=True, help="Max results to display.")
def bet_search(query: str, limit: int) -> None:
    """Search active Polymarket markets by keyword."""

    async def _search() -> None:
        from src.bet_helpers import search_markets

        click.echo(f"Searching for: {query}")
        results = await search_markets(query, limit=limit)

        if not results:
            click.echo("No markets found.")
            return

        click.echo(f"\n{'#':<4} {'Question':<60} {'YES':>6} {'Volume':>12} {'ID'}")
        click.echo("-" * 110)

        for i, m in enumerate(results, 1):
            question = (m.get("question") or "")[:58]
            outcome_prices = m.get("outcomePrices", "[]")
            if isinstance(outcome_prices, str):
                import json
                try:
                    outcome_prices = json.loads(outcome_prices)
                except (json.JSONDecodeError, TypeError):
                    outcome_prices = []

            yes_str = f"{float(outcome_prices[0]):.0%}" if outcome_prices else "?"
            vol = m.get("volume") or m.get("volumeNum") or 0
            vol_str = f"${float(vol):,.0f}"
            mid = m.get("id") or m.get("conditionId") or ""

            click.echo(f"{i:<4} {question:<60} {yes_str:>6} {vol_str:>12} {mid}")

    asyncio.run(_search())


@bet.command("find")
@click.option("--city", default=None, help="City name (e.g. Phoenix, Austin).")
@click.option("--station", default=None, help="ICAO station code (e.g. KPHX, KAUS).")
@click.option("--date", "date_str", default=None, help="Date filter (e.g. 'April 19').")
@click.option("--variable", default=None, help="Weather variable (temperature, precipitation, wind_speed).")
@click.option("--hours", default=72.0, show_default=True, help="Look-ahead window in hours.")
def bet_find(city: str | None, station: str | None, date_str: str | None, variable: str | None, hours: float) -> None:
    """Find Polymarket markets matching weather location/station (DB lookup).

    Much faster than 'bet search' — queries the local markets database
    populated by the 15-minute scanner instead of paginating the Gamma API.

    Examples:
      bet find --city Phoenix
      bet find --city Austin --date "April 19"
      bet find --station KPHX
      bet find --city Denver --variable temperature
    """

    async def _find() -> None:
        from src.db.engine import async_session
        from src.signals.reverse_lookup import (
            find_markets_for_city,
            find_markets_for_observation,
            find_markets_for_station,
        )

        if not city and not station:
            click.echo("Error: provide --city or --station")
            raise SystemExit(1)

        async with async_session() as session:
            if station:
                from src.ingestion.wx import get_buffer_history
                from src.signals.mapper import cities_for_icao

                city_names = cities_for_icao(station)
                click.echo(f"Station {station} -> cities: {', '.join(c.title() for c in city_names) or 'none'}")

                # Try observation-enriched lookup
                buf = get_buffer_history(station, count=1)
                if buf:
                    obs = buf[-1]
                    current_f = obs.temp_f
                    click.echo(f"Current observation: {current_f:.1f}F")
                    matches = await find_markets_for_observation(
                        session, station, current_f, hours_ahead=hours,
                    )
                    if not matches:
                        click.echo("No markets found.")
                        return

                    click.echo(f"\n{'#':<4} {'Question':<52} {'YES':>5} {'Thresh':>7} {'Dist':>6} {'Dir':>11} {'Liq':>10}")
                    click.echo("-" * 100)

                    for i, mm in enumerate(matches, 1):
                        m = mm.market
                        q = (m.question or "")[:50]
                        yes_str = f"{m.current_yes_price:.0%}" if m.current_yes_price else "?"
                        thresh = f"{m.parsed_threshold:.0f}F" if m.parsed_threshold is not None else "?"
                        dist = f"{mm.distance_to_threshold:+.1f}" if mm.distance_to_threshold is not None else "?"
                        liq = f"${m.liquidity:,.0f}" if m.liquidity else "?"
                        click.echo(f"{i:<4} {q:<52} {yes_str:>5} {thresh:>7} {dist:>6} {mm.direction:>11} {liq:>10}")
                    return

                # No observation buffer — plain station lookup
                markets = await find_markets_for_station(
                    session, station, variable=variable, hours_ahead=hours,
                )
            else:
                click.echo(f"Looking up markets for: {city}")
                markets = await find_markets_for_city(
                    session, city, variable=variable, date_str=date_str,
                )

            if not markets:
                click.echo("No markets found.")
                return

            click.echo(f"\n{'#':<4} {'Question':<52} {'YES':>5} {'Var':>14} {'Thresh':>7} {'Op':>8} {'Date':>12} {'Liq':>10}")
            click.echo("-" * 116)

            for i, m in enumerate(markets, 1):
                q = (m.question or "")[:50]
                yes_str = f"{m.current_yes_price:.0%}" if m.current_yes_price else "?"
                var = (m.parsed_variable or "?")[:12]
                thresh = f"{m.parsed_threshold:.0f}F" if m.parsed_threshold is not None else "?"
                op = (m.parsed_operator or "?")[:6]
                date = (m.parsed_target_date or "?")[:10]
                liq = f"${m.liquidity:,.0f}" if m.liquidity else "?"
                click.echo(f"{i:<4} {q:<52} {yes_str:>5} {var:>14} {thresh:>7} {op:>8} {date:>12} {liq:>10}")

    asyncio.run(_find())


@bet.command("cancel")
@click.argument("order_id")
def bet_cancel(order_id: str) -> None:
    """Cancel an open order on Polymarket."""

    async def _cancel() -> None:
        from src.config import settings
        from src.bet_helpers import get_clob_client

        if not settings.POLYMARKET_PRIVATE_KEY:
            click.echo("Error: POLYMARKET_PRIVATE_KEY not set in .env")
            raise SystemExit(1)

        click.echo(f"Cancelling order: {order_id}")
        client = get_clob_client()

        try:
            client.cancel(order_id)
            click.echo("Order cancelled successfully.")
        except Exception as exc:
            click.echo(f"Error: failed to cancel order ({exc})")
            raise SystemExit(1)

    asyncio.run(_cancel())


@bet.command("orders")
@click.option("--limit", "max_orders", default=20, show_default=True, help="Max orders to display.")
def bet_orders(max_orders: int) -> None:
    """List recent orders with their statuses (matched, delayed, cancelled, etc.)."""

    async def _orders() -> None:
        from src.config import settings
        from src.bet_helpers import get_clob_client

        if not settings.POLYMARKET_PRIVATE_KEY:
            click.echo("Error: POLYMARKET_PRIVATE_KEY not set in .env")
            raise SystemExit(1)

        click.echo("Fetching orders...")
        client = get_clob_client()

        from py_clob_client.clob_types import OpenOrderParams
        orders = client.get_orders(OpenOrderParams())

        if not orders:
            click.echo("No orders found.")
            return

        # Sort by timestamp descending if available
        orders = orders[:max_orders]

        click.echo(f"\n=== Orders ({len(orders)}) ===")
        click.echo(f"{'ID':<14} {'Status':<12} {'Side':<6} {'Size':>8} {'Price':>8} {'Matched':>8}  Token")
        click.echo("-" * 80)
        for o in orders:
            oid = (o.get("id") or "?")[:12]
            status = o.get("status", "?")
            side = o.get("side", "?")
            size = o.get("original_size", o.get("size", "?"))
            price = o.get("price", "?")
            matched = o.get("size_matched", "0")
            asset = (o.get("asset_id") or "")[:16]
            click.echo(f"  {oid}..  {status:<12} {side:<6} {size:>8} {price:>8} {matched:>8}  {asset}...")

        # Show cancel hint for non-terminal orders
        active = [o for o in orders if o.get("status") in ("live", "delayed")]
        if active:
            click.echo(f"\n{len(active)} active order(s). Cancel with:")
            for o in active:
                click.echo(f"  python -m src.cli bet cancel {o.get('id', '?')}")

    asyncio.run(_orders())


@bet.command("portfolio")
@click.option("--all", "show_all", is_flag=True, help="Show all positions ever held (including redeemed).")
@click.option("--history", is_flag=True, help="Show full trade history instead of positions.")
def bet_portfolio(show_all: bool, history: bool) -> None:
    """Show open positions and P&L from your Polymarket trades."""

    async def _portfolio() -> None:
        from src.config import settings
        from src.bet_helpers import (
            compute_positions,
            get_clob_client,
            get_ctf_readonly,
            get_open_orders,
            get_trades_history,
            get_usdc_balance,
        )

        if not settings.POLYMARKET_PRIVATE_KEY:
            click.echo("Error: POLYMARKET_PRIVATE_KEY not set in .env")
            raise SystemExit(1)

        click.echo("Connecting to Polymarket CLOB...")
        client = get_clob_client()

        # --- Wallet balance ---
        click.echo("Fetching wallet balance...")
        pusd, usdc_e, usdc_native = await get_usdc_balance(settings.POLYMARKET_PRIVATE_KEY)
        click.echo(f"\n=== Wallet ===")
        click.echo(f"  pUSD:        ${pusd:.2f}")
        click.echo(f"  USDC.e:      ${usdc_e:.2f}")
        if usdc_native > 0:
            click.echo(f"  Native USDC: ${usdc_native:.2f}")

        # --- Open orders ---
        click.echo("\nFetching open orders...")
        open_orders = get_open_orders(client)
        click.echo(f"\n=== Open Orders ({len(open_orders)}) ===")
        if open_orders:
            for o in open_orders:
                oid = o.get("id", "?")[:12]
                side = o.get("side", "?")
                size = o.get("original_size", o.get("size", "?"))
                price = o.get("price", "?")
                status = o.get("status", "?")
                asset = o.get("asset_id", "")[:16]
                click.echo(f"  {oid}...  {side:<5} size={size} price={price} status={status}  token={asset}...")
        else:
            click.echo("  No open orders.")

        # --- Trade history & positions ---
        click.echo("\nFetching trade history...")
        trades = get_trades_history(client)

        if not trades:
            click.echo("\n  No trades found.")
            return

        if history:
            # Resolve market questions for all unique asset_ids
            import httpx
            asset_ids = {t.get("asset_id", "") for t in trades}
            asset_ids.discard("")
            token_questions: dict[str, str] = {}
            async with httpx.AsyncClient(timeout=15) as http:
                for aid in asset_ids:
                    try:
                        resp = await http.get(
                            "https://gamma-api.polymarket.com/markets",
                            params={"clob_token_ids": aid},
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        if data:
                            mkt = data[0] if isinstance(data, list) else data
                            token_questions[aid] = (mkt.get("question") or "?")[:50]
                    except Exception:
                        pass

            click.echo(f"\n=== Trade History ({len(trades)}) ===")
            for t in trades:
                ts = t.get("match_time") or t.get("created_at") or t.get("timestamp", "?")
                if isinstance(ts, str) and len(ts) > 19:
                    ts = ts[:19]
                side = t.get("side", "?")
                size = float(t.get("size", 0))
                price = float(t.get("price", 0))
                cost = size * price
                aid = t.get("asset_id", "")
                question = token_questions.get(aid, aid[:20] + "...")
                click.echo(f"  {ts}  {side:<4} {size:>7.2f} @ ${price:.4f}  (${cost:.2f})  {question}")
            return

        # --- Positions ---
        positions = compute_positions(trades)

        if not positions:
            click.echo("\n=== Positions (0) ===")
            click.echo("  No open positions (all trades fully closed).")
            return

        # Resolve market questions via CLOB API (condition_id from trade data)
        import httpx
        token_to_market: dict[str, dict] = {}
        async with httpx.AsyncClient(timeout=15) as http:
            # Deduplicate: group by condition_id to avoid repeated lookups
            seen_conds: dict[str, dict | None] = {}
            for asset_id, pos in positions.items():
                cond = pos.get("market", "")
                if not cond:
                    continue
                if cond not in seen_conds:
                    try:
                        resp = await http.get(
                            f"https://clob.polymarket.com/markets/{cond}",
                        )
                        resp.raise_for_status()
                        seen_conds[cond] = resp.json()
                    except Exception:
                        seen_conds[cond] = None
                if seen_conds[cond]:
                    token_to_market[asset_id] = seen_conds[cond]

        # Default: hide positions in resolved markets whose on-chain balance
        # is zero (already redeemed). `--all` keeps the raw CLOB-history view.
        if not show_all:
            resolved_assets = [
                asset_id for asset_id, mkt in token_to_market.items()
                if mkt.get("closed") is True
                or str(mkt.get("closed", "")).lower() == "true"
            ]
            if resolved_assets:
                try:
                    _w3, ctf, wallet_addr, _rpc = get_ctf_readonly()
                except Exception as e:
                    click.echo(f"  Warning: could not reach Polygon RPC ({e}); showing unfiltered.")
                else:
                    for asset_id in resolved_assets:
                        try:
                            bal = ctf.functions.balanceOf(
                                wallet_addr, int(asset_id)
                            ).call()
                        except Exception:
                            continue
                        if bal == 0:
                            positions.pop(asset_id, None)
                            token_to_market.pop(asset_id, None)

        mode_label = "all-time" if show_all else "active+unredeemed"
        click.echo(f"\n=== Positions ({len(positions)}) [{mode_label}] ===")

        if not positions:
            click.echo(
                "  No active or unredeemed positions. "
                "Use --all to show redeemed positions as well."
            )
            return

        total_cost = 0.0
        total_value = 0.0

        for asset_id, pos in positions.items():
            size = pos["size"]
            avg_price = pos["avg_price"]
            cost = abs(pos["cost"])
            total_cost += cost

            # Resolve market question and determine YES/NO side
            mkt = token_to_market.get(asset_id)
            question = ""
            token_side = ""
            if mkt:
                question = mkt.get("question", "")
                # Determine if this token is YES or NO
                tokens = mkt.get("tokens", [])
                for tok in tokens:
                    if tok.get("token_id") == asset_id:
                        token_side = (tok.get("outcome") or "").upper()
                        break

            # Fetch current price for this token
            current_price = None
            try:
                last = client.get_last_trade_price(asset_id)
                if last and last.get("price"):
                    current_price = float(last["price"])
            except Exception:
                pass

            current_value = abs(size) * current_price if current_price else None
            if current_value is not None:
                total_value += current_value
                pnl = current_value - cost
                pnl_pct = (pnl / cost * 100) if cost else 0
                pnl_str = f"{'+'if pnl>=0 else ''}{pnl:.2f} ({pnl_pct:+.1f}%)"
            else:
                pnl_str = "?"

            # Display
            if question:
                click.echo(f"\n  {question}")
            else:
                click.echo(f"\n  Token: {asset_id[:20]}...")
            side_label = f"LONG {token_side}" if token_side else pos["side"]
            click.echo(f"    Side:      {side_label}")
            click.echo(f"    Size:      {abs(size):.2f} shares")
            click.echo(f"    Avg entry: ${avg_price:.4f}")
            click.echo(f"    Cost:      ${cost:.2f}")
            if current_price is not None:
                click.echo(f"    Current:   ${current_price:.4f}")
                click.echo(f"    Value:     ${current_value:.2f}")
            click.echo(f"    P&L:       {pnl_str}")
            if mkt:
                slug = mkt.get("slug", "")
                if slug:
                    click.echo(f"    URL:       https://polymarket.com/event/{slug}")

        # --- Summary ---
        click.echo(f"\n=== Summary ===")
        click.echo(f"  Total cost:     ${total_cost:.2f}")
        if total_value > 0:
            click.echo(f"  Current value:  ${total_value:.2f}")
            total_pnl = total_value - total_cost
            click.echo(f"  Total P&L:      {'+'if total_pnl>=0 else ''}{total_pnl:.2f}")
        click.echo(f"  USDC.e balance: ${usdc_e:.2f}")

    asyncio.run(_portfolio())


@bet.command("redeem")
@click.option("--all", "redeem_all", is_flag=True, help="Redeem all resolved positions.")
@click.option("--yes", "-y", "skip_confirm", is_flag=True, help="Skip confirmation prompt.")
def bet_redeem(redeem_all: bool, skip_confirm: bool) -> None:
    """Redeem winnings from resolved Polymarket positions on-chain."""

    async def _redeem() -> None:
        from src.config import settings
        from src.bet_helpers import (
            compute_positions,
            get_clob_client,
            get_ctf_readonly,
            get_trades_history,
            get_usdc_balance,
        )

        if not settings.POLYMARKET_PRIVATE_KEY:
            click.echo("Error: POLYMARKET_PRIVATE_KEY not set in .env")
            raise SystemExit(1)

        if not redeem_all:
            click.echo("Usage: bet redeem --all")
            click.echo("  Redeems all resolved positions with non-zero on-chain balance.")
            raise SystemExit(1)

        # --- Fetch positions from CLOB ---
        click.echo("Connecting to Polymarket CLOB...")
        client = get_clob_client()

        click.echo("Fetching trade history...")
        trades = get_trades_history(client)
        if not trades:
            click.echo("No trades found.")
            return

        positions = compute_positions(trades)
        if not positions:
            click.echo("No open positions found.")
            return

        click.echo(f"Found {len(positions)} position(s). Checking market resolution...")

        # --- Resolve market info via CLOB API (reliable for neg-risk) ---
        import httpx
        import json as _json

        redeemable: list[dict] = []

        async with httpx.AsyncClient(timeout=15) as http:
            # Group positions by condition_id (market) to avoid duplicate lookups
            cond_to_assets: dict[str, list[tuple[str, dict]]] = {}
            for asset_id, pos in positions.items():
                cond = pos.get("market", "")
                if not cond:
                    continue
                cond_to_assets.setdefault(cond, []).append((asset_id, pos))

            for cond_id, assets in cond_to_assets.items():
                # CLOB API reliably resolves markets by condition_id
                mkt = None
                try:
                    resp = await http.get(
                        f"https://clob.polymarket.com/markets/{cond_id}",
                    )
                    resp.raise_for_status()
                    mkt = resp.json()
                except Exception:
                    pass

                if not mkt:
                    continue

                # Check if market is closed/resolved
                closed = mkt.get("closed") is True or str(mkt.get("closed", "")).lower() == "true"
                if not closed:
                    continue

                condition_id = mkt.get("condition_id", "") or mkt.get("conditionId", "")
                if not condition_id:
                    continue

                neg_risk = mkt.get("neg_risk") is True or str(mkt.get("neg_risk", "")).lower() == "true"

                # Build token_id → outcome mapping from CLOB tokens array
                tokens = mkt.get("tokens", [])
                token_map: dict[str, str] = {}
                clob_ids: list[str] = []
                for tok in tokens:
                    tid = tok.get("token_id", "")
                    outcome = tok.get("outcome", "")
                    if tid:
                        token_map[tid] = outcome.upper()  # "Yes" → "YES"
                        clob_ids.append(tid)

                for asset_id, pos in assets:
                    token_side = token_map.get(asset_id, "")

                    redeemable.append({
                        "asset_id": asset_id,
                        "question": mkt.get("question", ""),
                        "condition_id": condition_id,
                        "token_side": token_side,
                        "neg_risk": neg_risk,
                        "clob_ids": clob_ids,
                        "outcome_prices": [],
                        "size": pos["size"],
                    })

        if not redeemable:
            click.echo("No resolved positions found to redeem.")
            return

        # --- Web3 setup with RPC failover ---
        from web3 import Web3

        NEG_RISK_ADAPTER_ADDRESS = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"
        PUSD_ADDRESS = "0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB"

        NEG_RISK_ADAPTER_ABI = [
            {
                "inputs": [
                    {"name": "conditionId", "type": "bytes32"},
                    {"name": "amounts", "type": "uint256[]"},
                ],
                "name": "redeemPositions",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function",
            },
        ]

        try:
            w3, ctf, address, rpc_url = get_ctf_readonly()
        except Exception as e:
            click.echo(f"Error: {e}")
            raise SystemExit(1)
        click.echo(f"Connected to {rpc_url}")

        bal = w3.eth.get_balance(address)
        click.echo(f"POL balance: {w3.from_wei(bal, 'ether'):.4f} (for gas)")
        if bal == 0:
            click.echo("Error: wallet has no POL for gas fees")
            raise SystemExit(1)

        neg_risk_adapter = w3.eth.contract(
            address=Web3.to_checksum_address(NEG_RISK_ADAPTER_ADDRESS),
            abi=NEG_RISK_ADAPTER_ABI,
        )

        # --- Check on-chain balances ---
        click.echo("\nChecking on-chain token balances...")
        to_redeem: list[dict] = []

        for item in redeemable:
            try:
                token_balance = ctf.functions.balanceOf(
                    address, int(item["asset_id"])
                ).call()
            except Exception as e:
                click.echo(f"  Could not check balance for {item['question'][:50]}: {e}")
                continue

            if token_balance == 0:
                click.echo(f"  {item['question'][:50]}: already redeemed (balance=0)")
                continue

            # Conditional tokens use 1e6 decimals (same as USDC)
            balance_usdc = token_balance / 1e6
            item["on_chain_balance"] = token_balance
            item["balance_usdc"] = balance_usdc
            to_redeem.append(item)

        if not to_redeem:
            click.echo("\nNo positions with non-zero on-chain balance to redeem.")
            return

        # --- Display redeemable positions ---
        click.echo(f"\n=== Redeemable Positions ({len(to_redeem)}) ===")
        for item in to_redeem:
            question = item["question"] or item["asset_id"][:20]
            side = item["token_side"] or "?"
            neg = " [neg-risk]" if item["neg_risk"] else ""
            click.echo(f"\n  {question}")
            click.echo(f"    Side:        {side}")
            click.echo(f"    On-chain:    {item['balance_usdc']:.2f} shares")
            click.echo(f"    Neg risk:    {'Yes' if item['neg_risk'] else 'No'}")

        # --- Confirmation ---
        if not skip_confirm:
            if not click.confirm("\nProceed with on-chain redemption?"):
                click.echo("Aborted.")
                return

        # --- Execute redemptions ---
        click.echo("\n=== Executing Redemptions ===")
        gas_price = w3.eth.gas_price
        nonce = w3.eth.get_transaction_count(address)
        CHAIN_ID = 137
        PARENT_COLLECTION_ID = b"\x00" * 32

        def send_tx(tx_data: dict) -> dict:
            nonlocal nonce
            tx_data["nonce"] = nonce
            tx_data["chainId"] = CHAIN_ID
            tx_data["from"] = address
            tx_data["gasPrice"] = gas_price
            if "gas" not in tx_data:
                tx_data["gas"] = w3.eth.estimate_gas(tx_data)
            signed = w3.eth.account.sign_transaction(
                tx_data, private_key=settings.POLYMARKET_PRIVATE_KEY
            )
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            nonce += 1
            return receipt

        success_count = 0
        for item in to_redeem:
            question = (item["question"] or item["asset_id"][:20])[:50]
            condition_id_hex = item["condition_id"]
            if condition_id_hex.startswith("0x"):
                condition_id_bytes = bytes.fromhex(condition_id_hex[2:])
            else:
                condition_id_bytes = bytes.fromhex(condition_id_hex)

            # Determine indexSets based on token side
            clob_ids = item["clob_ids"]
            if len(clob_ids) >= 2 and item["asset_id"] == clob_ids[0]:
                index_sets = [1]  # YES = outcome index 0 → 2^0 = 1
            elif len(clob_ids) >= 2 and item["asset_id"] == clob_ids[1]:
                index_sets = [2]  # NO = outcome index 1 → 2^1 = 2
            else:
                index_sets = [1, 2]  # Try both

            click.echo(f"\n  Redeeming: {question}...")
            try:
                if item["neg_risk"]:
                    on_chain = item["on_chain_balance"]
                    clob_ids = item["clob_ids"]
                    if len(clob_ids) >= 2 and item["asset_id"] == clob_ids[0]:
                        amounts = [on_chain, 0]   # holding YES position
                    elif len(clob_ids) >= 2 and item["asset_id"] == clob_ids[1]:
                        amounts = [0, on_chain]   # holding NO position
                    else:
                        amounts = [on_chain, 0]   # fallback
                    tx = neg_risk_adapter.functions.redeemPositions(
                        condition_id_bytes,
                        amounts,
                    ).build_transaction({"from": address, "gasPrice": gas_price})
                else:
                    pusd_addr = Web3.to_checksum_address(PUSD_ADDRESS)
                    tx = ctf.functions.redeemPositions(
                        pusd_addr,
                        PARENT_COLLECTION_ID,
                        condition_id_bytes,
                        index_sets,
                    ).build_transaction({"from": address, "gasPrice": gas_price})
                receipt = send_tx(tx)
                status = "OK" if receipt["status"] == 1 else "FAILED"
                tx_hash = receipt["transactionHash"].hex()[:16]
                gas_used = receipt["gasUsed"]
                click.echo(f"    Status: {status}  tx: {tx_hash}...  gas: {gas_used}")
                if receipt["status"] == 1:
                    success_count += 1
            except Exception as e:
                click.echo(f"    FAILED: {e}")
                click.echo("    (Market may not be resolved on-chain yet)")

        # --- Final balance ---
        click.echo(f"\n=== Results ===")
        click.echo(f"  Redeemed: {success_count}/{len(to_redeem)} positions")
        pusd, usdc_e, usdc_native = await get_usdc_balance(settings.POLYMARKET_PRIVATE_KEY)
        click.echo(f"  pUSD balance:   ${pusd:.2f}")
        click.echo(f"  USDC.e balance: ${usdc_e:.2f}")
        if usdc_native > 0:
            click.echo(f"  Native USDC:    ${usdc_native:.2f}")

    asyncio.run(_redeem())


if __name__ == "__main__":
    main()
