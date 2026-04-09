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


if __name__ == "__main__":
    main()
