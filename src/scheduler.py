"""APScheduler-based pipeline orchestration and health-check server."""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import select

from src.config import settings
from src.db.engine import async_session, engine
from src.db.models import Market, Signal, Trade, TradeStatus
from src.execution.alerter import get_alerter
from src.ingestion.polymarket import scan_and_ingest
from src.resolution import (
    get_current_bankroll,
    get_current_exposure,
    resolve_trades,
)
from src.risk.drawdown import DrawdownLevel, DrawdownMonitor
from src.risk.kelly import size_position
from src.signals.consensus import get_calibration_coefficients
from src.execution.polymarket_client import is_live, place_order
from src.signals.detector import detect_signals_short_range

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession as AsyncSessionType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_scheduler: AsyncIOScheduler | None = None
_drawdown_monitor: DrawdownMonitor | None = None
_shutdown_event: asyncio.Event | None = None

# ---------------------------------------------------------------------------
# Structured JSON logging
# ---------------------------------------------------------------------------


class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, object] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry)


def configure_logging() -> None:
    """Replace root handlers with a single JSON-to-stdout handler."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JSONFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)
    # Suppress noisy third-party loggers (httpx URLs can leak Telegram tokens, etc.)
    from src.logging_utils import NOISY_LOGGERS
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Health-check HTTP server (stdlib only)
# ---------------------------------------------------------------------------


async def _health_handler(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    try:
        await reader.read(4096)  # consume request
        body = json.dumps({
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scheduler_running": _scheduler is not None and _scheduler.running,
        })
        response = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            "\r\n"
            f"{body}"
        )
        writer.write(response.encode())
        await writer.drain()
    finally:
        writer.close()
        await writer.wait_closed()


async def start_health_server(port: int = 8080) -> asyncio.Server:
    server = await asyncio.start_server(_health_handler, "0.0.0.0", port)
    logger.info("Health-check server listening on port %d", port)
    return server


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get_drawdown_monitor() -> DrawdownMonitor:
    global _drawdown_monitor  # noqa: PLW0603
    if _drawdown_monitor is None:
        _drawdown_monitor = DrawdownMonitor(settings.INITIAL_BANKROLL)
        async with async_session() as session:
            await _drawdown_monitor.load_state(session)
    return _drawdown_monitor


async def _latest_signal_id(
    session: AsyncSessionType,
    market_id: str,
) -> int | None:
    """Return the most recently created Signal id for a market."""
    result = await session.execute(
        select(Signal.id)
        .where(Signal.market_id == market_id)
        .order_by(Signal.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


# ---------------------------------------------------------------------------
# Scheduled jobs
# ---------------------------------------------------------------------------


async def job_scan_markets() -> None:
    """Every 15 min — fetch Polymarket weather markets snapshot."""
    try:
        count = await scan_and_ingest()
        logger.info("Market scan complete: %d markets", count)
    except Exception as exc:
        logger.exception("Market scan failed")
        await get_alerter().send_system_error(exc, "market scan")


async def _size_and_create_trades(signals: list, alerter, monitor) -> None:
    """Shared logic: size positions, create trades, send alerts."""
    async with async_session() as session:
        bankroll = await get_current_bankroll(session)
        exposure = await get_current_exposure(session)
        dd_state = monitor.check(bankroll)

        for sig in signals:
            position = size_position(
                bankroll=bankroll,
                model_prob=sig.consensus_prob,
                market_prob=sig.market_prob,
                current_exposure=exposure,
            )

            # Apply drawdown multiplier
            adjusted_stake = position.stake_usd * dd_state.size_multiplier
            if adjusted_stake < 5.0:
                logger.info(
                    "Skipping signal on %s: stake $%.2f after drawdown multiplier",
                    sig.market_id,
                    adjusted_stake,
                )
                continue

            position.stake_usd = adjusted_stake

            # Look up the Signal DB row for the FK
            signal_id = await _latest_signal_id(session, sig.market_id)
            if signal_id is None:
                logger.warning("No Signal row found for market %s", sig.market_id)
                continue

            # Create trade
            trade = Trade(
                signal_id=signal_id,
                market_id=sig.market_id,
                direction=sig.direction,
                stake_usd=position.stake_usd,
                entry_price=sig.market_prob,
                status=TradeStatus.PENDING,
            )
            session.add(trade)
            await session.flush()  # assign trade.id before execution

            # Execute order on Polymarket
            order_ok = await place_order(trade, session)
            if order_ok:
                trade.status = TradeStatus.OPEN
                exposure += position.stake_usd
            else:
                logger.warning(
                    "Order failed for market %s (status: %s); trade stays PENDING",
                    sig.market_id,
                    trade.exchange_status,
                )
                continue

            # Fetch Market row for the alert
            market = await session.get(Market, sig.market_id)

            await alerter.send_signal_alert(sig, position, dd_state, market)

        if dd_state.level in (DrawdownLevel.CAUTION, DrawdownLevel.PAUSED):
            await alerter.send_drawdown_warning(dd_state)

        await session.commit()


async def job_short_range_pipeline() -> None:
    """Every 60 min — aviation-focused pipeline for markets ≤30h out."""
    alerter = get_alerter()
    try:
        signals = await detect_signals_short_range()
        logger.info("Short-range pipeline: %d actionable signals", len(signals))

        if not signals:
            return

        monitor = await _get_drawdown_monitor()
        await _size_and_create_trades(signals, alerter, monitor)

    except Exception as exc:
        logger.exception("Short-range pipeline failed")
        await alerter.send_system_error(exc, "short-range pipeline")


async def job_wx_rapid_pipeline() -> None:
    """Every 1 min — Weather Company observation-driven rapid pipeline.

    Polls airport ICAO stations for markets ≤12h out, deduplicates by
    validTimeLocal, analyzes trends, and detects threshold crossings
    and peak events for temporal edge trading.
    """
    from src.ingestion.wx import analyze_trend, detect_threshold_events, poll_stations
    from src.signals.mapper import icao_for_location

    alerter = get_alerter()
    try:
        async with async_session() as session:
            # 1. Get ultra-short markets (≤12h)
            now = datetime.now(timezone.utc)
            cutoff = now + timedelta(hours=12)
            stmt = select(Market).where(
                Market.end_date.isnot(None),
                Market.end_date <= cutoff,
                Market.end_date > now,
                Market.parsed_location.isnot(None),
            )
            result = await session.execute(stmt)
            markets = result.scalars().all()

            if not markets:
                logger.debug("WX rapid: no ultra-short markets")
                return

            # 2. Map markets to ICAO stations
            icao_set: set[str] = set()
            market_icao: dict[str, str] = {}  # market_id → icao
            market_thresholds: dict[str, list[tuple[float, str]]] = {}  # icao → [(threshold_f, operator)]

            for m in markets:
                icao = icao_for_location(m.parsed_location)
                if not icao:
                    continue
                icao_set.add(icao)
                market_icao[m.id] = icao

                if m.parsed_threshold is not None and m.parsed_operator:
                    key = icao
                    if key not in market_thresholds:
                        market_thresholds[key] = []
                    market_thresholds[key].append(
                        (m.parsed_threshold, m.parsed_operator)
                    )

            if not icao_set:
                logger.debug("WX rapid: no ICAO stations for active markets")
                return

            # 3. Poll all stations (dedup + persist happens inside)
            new_obs = await poll_stations(list(icao_set), session)
            logger.debug(
                "WX rapid: polled %d stations, %d new observations",
                len(icao_set), len(new_obs),
            )

            # 3b. Confirmation pipeline
            if settings.WU_CONFIRM_ENABLED:
                from src.ingestion.confirmation import (
                    ConfirmationState,
                    get_confirmation_tracker,
                )

                tracker = get_confirmation_tracker()

                # Feed new observations to the tracker
                stations_to_scrape: list[str] = []
                for icao, obs in new_obs.items():
                    new_state = tracker.on_new_observation(icao, obs)
                    if new_state == ConfirmationState.DECREASE_DETECTED:
                        stations_to_scrape.append(icao)

                # Add stations waiting for WU data retry
                stations_to_scrape.extend(tracker.get_stations_needing_wu_retry())

                # Scrape WU sequentially (rate limited)
                for icao in stations_to_scrape:
                    try:
                        wu_state = await tracker.check_wu_confirmation(icao)
                        sc = tracker.get(icao)
                        if sc is None:
                            continue
                        if wu_state == ConfirmationState.WU_CONFIRMED:
                            matches = await tracker.match_markets(icao, session)
                            await alerter.send_confirmation_alert(
                                icao,
                                sc.peak_temp_f,
                                sc.wu_high_f,
                                sc.db_vs_wu_delta,
                                matches,
                            )
                            sc.state = ConfirmationState.ALERTED
                        elif wu_state == ConfirmationState.WU_WAITING:
                            if not sc._wu_waiting_alerted:
                                await alerter.send_wu_waiting_alert(
                                    icao, sc.peak_temp_f, sc.wu_latest_hour,
                                )
                                sc._wu_waiting_alerted = True
                    except Exception:
                        logger.exception(
                            "Confirmation pipeline error for %s", icao,
                        )

            # 4. Analyze trends and detect events for each station
            all_events = []
            for icao in icao_set:
                thresholds = market_thresholds.get(icao, [])
                if thresholds:
                    events = detect_threshold_events(icao, thresholds)
                    all_events.extend(events)

                trend = analyze_trend(icao)
                if trend is not None:
                    logger.info(
                        "WX trend %s: %.1f°F (max=%.1f min=%.1f) "
                        "rate=%.1f°F/hr rising=%s falling=%s peak_done=%s",
                        icao, trend.current_temp_f,
                        trend.observed_max_f, trend.observed_min_f,
                        trend.temp_rate_per_hour or 0.0,
                        trend.is_rising, trend.is_falling,
                        trend.peak_likely_passed,
                    )

            # 5. Log threshold events and run signal detection if needed
            if all_events:
                for ev in all_events:
                    logger.info(
                        "WX EVENT %s [%s]: %s (confidence=%.2f)",
                        ev.station_icao, ev.event_type, ev.detail, ev.confidence,
                    )

                # Run full signal detection for affected markets
                affected_icaos = {ev.station_icao for ev in all_events}
                affected_market_ids = {
                    mid for mid, icao in market_icao.items()
                    if icao in affected_icaos
                }

                if affected_market_ids:
                    signals = await detect_signals_short_range(session)
                    signals = [s for s in signals if s.market_id in affected_market_ids]

                    logger.info("WX rapid pipeline: %d actionable signals", len(signals))

                    if signals:
                        monitor = await _get_drawdown_monitor()
                        await _size_and_create_trades(signals, alerter, monitor)

                # Backward resolution: discover additional markets beyond 12h
                from src.signals.reverse_lookup import find_markets_for_event

                for ev in all_events:
                    extra = await find_markets_for_event(
                        session, ev.station_icao, ev.temp_f, hours_ahead=48.0,
                    )
                    extra_ids = {m.id for m in extra} - affected_market_ids
                    if extra_ids:
                        logger.info(
                            "WX backward resolution: %s found %d additional market(s) "
                            "beyond 12h window",
                            ev.station_icao, len(extra_ids),
                        )
                        await alerter.send_market_discovery(
                            ev.station_icao, ev.temp_f, extra,
                        )

            await session.commit()

    except Exception as exc:
        logger.exception("WX rapid pipeline failed")
        await alerter.send_system_error(exc, "wx-rapid pipeline")


async def job_daily_settlement() -> None:
    """Daily 22:00 UTC — resolve markets, P&L, daily summary."""
    alerter = get_alerter()
    try:
        monitor = await _get_drawdown_monitor()

        async with async_session() as session:
            # 1. Resolve expired trades
            resolved = await resolve_trades(session)
            for trade in resolved:
                market = await session.get(Market, trade.market_id)
                await alerter.send_resolution(trade, market)

            # 2. Update bankroll & drawdown
            bankroll = await get_current_bankroll(session)
            total_pnl = sum(t.pnl or 0.0 for t in resolved)
            new_bankroll = bankroll + total_pnl
            dd_state = await monitor.update(new_bankroll, session)

            if dd_state.level in (DrawdownLevel.CAUTION, DrawdownLevel.PAUSED):
                await alerter.send_drawdown_warning(dd_state)

            # 3. Daily summary
            await alerter.send_daily_summary(session)

            # 4. WX observation retention cleanup
            if settings.WX_API_KEY:
                from sqlalchemy import delete

                from src.db.models import WxObservation as WxObservationModel

                wx_cutoff = datetime.now(timezone.utc) - timedelta(
                    hours=settings.WX_RETENTION_HOURS,
                )
                result = await session.execute(
                    delete(WxObservationModel).where(
                        WxObservationModel.valid_time_utc < wx_cutoff,
                    )
                )
                if result.rowcount:
                    logger.info("WX cleanup: deleted %d old observations", result.rowcount)

            # 5. Weekly calibration check (Sundays)
            if datetime.now(timezone.utc).weekday() == 6:
                coeffs = await get_calibration_coefficients(session)
                if coeffs is not None:
                    logger.info(
                        "Calibration active: slope=%.4f intercept=%.4f",
                        coeffs[0],
                        coeffs[1],
                    )
                else:
                    logger.info("Calibration: insufficient data (<50 resolved signals)")

            await session.commit()

    except Exception as exc:
        logger.exception("Daily settlement failed")
        await alerter.send_system_error(exc, "daily settlement")


async def job_startup() -> None:
    """Run once on startup — initial data load and Telegram status."""
    alerter = get_alerter()
    await alerter.start()

    await _get_drawdown_monitor()
    logger.info("Drawdown monitor loaded")

    await job_scan_markets()
    await job_short_range_pipeline()

    try:
        async with async_session() as session:
            bankroll = await get_current_bankroll(session)
        await alerter._enqueue(
            f"\U0001f680 *WeatherEdge started*\n"
            f"Bankroll: ${bankroll:,.2f}\n"
            f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        )
    except Exception:
        logger.info("Pipeline started (Telegram notification skipped)")


# ---------------------------------------------------------------------------
# Backfill (used by CLI)
# ---------------------------------------------------------------------------


async def backfill_markets(days: int) -> int:
    """Fetch historical market data going back *days* days.

    Fetches both active and closed weather markets from the Gamma API and
    stores Market + MarketSnapshot rows via :func:`ingest_markets`.
    """
    import httpx

    from src.ingestion.polymarket import (
        GAMMA_BASE,
        ingest_markets,
        is_weather_market,
    )

    MARKETS_URL = f"{GAMMA_BASE}/markets"
    total = 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    async with httpx.AsyncClient(timeout=30) as client:
        for closed in (False, True):
            offset = 0
            limit = 100
            while True:
                params: dict[str, object] = {
                    "limit": limit,
                    "offset": offset,
                    "active": str(not closed).lower(),
                    "closed": str(closed).lower(),
                }
                resp = await client.get(MARKETS_URL, params=params)
                resp.raise_for_status()
                batch = resp.json()

                if not isinstance(batch, list) or not batch:
                    break

                # Filter to weather markets within the date range.
                weather: list[dict] = []
                past_cutoff = False
                for raw in batch:
                    end_str = raw.get("endDate") or raw.get("end_date_iso")
                    if end_str:
                        try:
                            end_dt = datetime.fromisoformat(
                                end_str.replace("Z", "+00:00")
                            )
                            if end_dt < cutoff:
                                past_cutoff = True
                                break
                        except (ValueError, TypeError):
                            pass

                    if is_weather_market(raw):
                        weather.append(raw)

                if weather:
                    async with async_session() as session:
                        total += await ingest_markets(session, weather)

                if past_cutoff or len(batch) < limit:
                    break
                offset += limit

    logger.info("Backfill complete: %d markets over %d days", total, days)
    return total


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def setup_scheduler() -> AsyncIOScheduler:
    """Create and configure the APScheduler instance."""
    scheduler = AsyncIOScheduler(timezone="UTC")

    scheduler.add_job(
        job_scan_markets,
        IntervalTrigger(minutes=15),
        id="scan_markets",
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        job_daily_settlement,
        CronTrigger(hour=22, minute=0, timezone="UTC"),
        id="daily_settlement",
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        job_short_range_pipeline,
        IntervalTrigger(minutes=settings.SR_PIPELINE_INTERVAL_MINUTES),
        id="short_range_pipeline",
        max_instances=1,
        coalesce=True,
    )

    if settings.WX_API_KEY:
        scheduler.add_job(
            job_wx_rapid_pipeline,
            IntervalTrigger(minutes=settings.WX_PIPELINE_INTERVAL_MINUTES),
            id="wx_rapid_pipeline",
            max_instances=1,
            coalesce=True,
        )

    return scheduler


async def run_scheduler() -> None:
    """Start the full pipeline daemon."""
    global _scheduler, _shutdown_event  # noqa: PLW0603

    configure_logging()
    logger.info("Starting WeatherEdge scheduler")

    _shutdown_event = asyncio.Event()

    loop = asyncio.get_running_loop()
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, _shutdown_event.set)
    except NotImplementedError:
        # Some runtimes (e.g. DO App Platform) don't support signal handlers
        logger.warning("Signal handlers not supported; will run until cancelled")

    health_server = await start_health_server()

    # Ensure all tables exist (first deploy / fresh database)
    from src.db.models import Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    _scheduler = setup_scheduler()
    _scheduler.start()

    try:
        await job_startup()
    except Exception as exc:
        logger.exception("Startup job failed, continuing scheduler")
        await get_alerter().send_system_error(exc, "startup")

    await _shutdown_event.wait()

    # Graceful shutdown
    logger.info("Shutting down…")
    _scheduler.shutdown(wait=True)
    await get_alerter().shutdown()
    health_server.close()
    await health_server.wait_closed()
    logger.info("Shutdown complete")
