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


_UNIFIED_CONCURRENCY = 8  # Max concurrent city aggregations


async def job_unified_pipeline() -> None:
    """Every 5 min — unified METAR + Open-Meteo + market pipeline.

    Replaces job_short_range_pipeline and job_wx_rapid_pipeline when
    UNIFIED_PIPELINE_ENABLED is True.

    Performance design:
      Phase 1 — fetch all weather states concurrently (bounded semaphore).
                Each city's METAR + Open-Meteo calls run in parallel.
      Phase 2 — evaluate edges and execute trades sequentially
                (bankroll/exposure tracking requires serial execution).
    """
    alerter = get_alerter()
    try:
        from src.ingestion.polymarket import get_active_weather_markets
        from src.ingestion.station_bias import is_bias_runaway
        from src.risk.circuit_breakers import check_circuit_breakers
        from src.signals.state_aggregator import WeatherState, aggregate_state
        from src.signals.probability_engine import compute_distribution
        from src.signals.edge_calculator import compute_edges
        from src.signals.mapper import icao_for_location, geocode
        from src.execution.polymarket_client import get_orderbook_depth, get_token_ids

        async with async_session() as session:
            # 1. Circuit breaker check
            cb = await check_circuit_breakers(session)
            if not cb.can_trade:
                logger.info("Circuit breaker active: %s", cb.reason)
                return

            # 2. Get active temperature markets
            markets = await get_active_weather_markets(session)
            if not markets:
                logger.debug("Unified pipeline: no active markets")
                return

            # 3. Group markets by city/ICAO and resolve coordinates
            city_markets: dict[str, list] = {}
            city_coords: dict[str, tuple[float, float]] = {}
            for m in markets:
                loc = m.parsed_location
                if not loc:
                    continue
                icao = icao_for_location(loc)
                if not icao:
                    continue
                city_markets.setdefault(icao, []).append(m)
                if icao not in city_coords:
                    coords = geocode(loc)
                    if coords:
                        city_coords[icao] = coords

            city_summary = ", ".join(f"{k}:{len(v)}" for k, v in sorted(city_markets.items(), key=lambda x: -len(x[1]))[:8])
            logger.info(
                "Unified pipeline: %d markets across %d cities (%s)",
                sum(len(v) for v in city_markets.values()), len(city_markets), city_summary,
            )

            # ---- Phase 1: concurrent weather state aggregation ----
            sem = asyncio.Semaphore(_UNIFIED_CONCURRENCY)

            async def _aggregate_one(icao: str) -> tuple[str, WeatherState | None]:
                if icao not in city_coords:
                    return icao, None
                try:
                    if await is_bias_runaway(session, icao):
                        logger.warning("Skipping %s: bias runaway", icao)
                        return icao, None
                except Exception:
                    logger.warning("Bias check failed for %s, proceeding", icao, exc_info=True)
                lat, lon = city_coords[icao]
                async with sem:
                    try:
                        state = await aggregate_state(session, icao, lat, lon)
                    except Exception:
                        logger.warning("Unified pipeline: aggregate failed for %s", icao, exc_info=True)
                        state = None
                return icao, state

            agg_results = await asyncio.gather(
                *[_aggregate_one(icao) for icao in city_markets],
            )

            city_states: dict[str, WeatherState] = {}
            skipped_cities = 0
            for icao, state in agg_results:
                if state is not None:
                    city_states[icao] = state
                else:
                    skipped_cities += 1

            logger.info(
                "Unified pipeline phase 1: %d/%d cities aggregated (%.0f%% success)",
                len(city_states), len(city_markets),
                100 * len(city_states) / max(1, len(city_markets)),
            )

            # ---- Phase 2: sequential edge evaluation and trading ----
            monitor = await _get_drawdown_monitor()
            bankroll = await get_current_bankroll(session)
            exposure = await get_current_exposure(session)
            total_trades = 0

            for icao, state in city_states.items():
              try:
                for market in city_markets[icao]:
                    end_time = market.end_date or datetime.now(timezone.utc) + timedelta(hours=24)

                    # Orderbook depth at market YES price
                    mkt_depth = 0.0
                    token_ids = await get_token_ids(market.id)
                    if token_ids and market.current_yes_price:
                        mkt_depth = get_orderbook_depth(token_ids[0], market.current_yes_price)

                    if _is_binary_market(market):
                        # --- Binary threshold market ---
                        buckets = _make_binary_buckets(market, state)
                        dist = compute_distribution(state, buckets)
                        edge_result = _binary_market_edge(
                            dist, market, end_time,
                            state.routine_count_today, mkt_depth,
                        )
                        if edge_result is None:
                            continue

                        op_symbol = {"above": "≥", "at_least": "≥", "below": "<", "at_most": "≤", "exactly": "="}.get(market.parsed_operator, "?")
                        tag = "PASS" if edge_result.passes else edge_result.reject_reason
                        logger.info(
                            "[%s] binary %s%d°F: P(YES)=%.3f, mkt=%.3f, edge=%+.3f %s | %s",
                            icao, op_symbol, int(market.parsed_threshold),
                            edge_result.our_probability, edge_result.market_price,
                            edge_result.edge, tag, dist.reasoning,
                        )

                        edges = [edge_result]
                        passing = [edge_result] if edge_result.passes else []
                        depths = {int(market.parsed_threshold): mkt_depth}
                    else:
                        # --- Bracket market ---
                        buckets = _extract_bracket_buckets(market)
                        if not buckets:
                            continue

                        dist = compute_distribution(state, buckets)
                        market_prices = _extract_market_prices(market, buckets)

                        depths: dict[int, float] = {}
                        if token_ids:
                            for b in buckets:
                                price = market_prices.get(b, 0)
                                if price > 0:
                                    depths[b] = get_orderbook_depth(token_ids[0], price)

                        edges = compute_edges(
                            dist, market_prices, state.routine_count_today, end_time, depths
                        )
                        passing = [e for e in edges if e.passes]
                        logger.info(
                            "Unified pipeline [%s/%s]: dist=%s, passing_edges=%d/%d, reasoning=%s",
                            icao, market.id[:12],
                            {k: f"{v:.3f}" for k, v in dist.probabilities.items() if v > 0.01},
                            len(passing), len(edges), dist.reasoning,
                        )

                    dd_state = monitor.check(bankroll)
                    for edge in passing:
                        pos = size_position(
                            bankroll=bankroll,
                            model_prob=edge.our_probability,
                            market_prob=edge.market_price,
                            current_exposure=exposure,
                            max_position_usd=settings.MAX_POSITION_USD,
                            orderbook_depth=depths.get(edge.bucket_value),
                        )

                        adjusted_stake = pos.stake_usd * dd_state.size_multiplier
                        logger.info(
                            "[%s] sizing bucket=%d: kelly_stake=$%.2f, dd_mult=%.2f, adjusted=$%.2f, bankroll=$%.0f, exposure=$%.0f",
                            icao, edge.bucket_value, pos.stake_usd, dd_state.size_multiplier,
                            adjusted_stake, bankroll, exposure,
                        )
                        if adjusted_stake < settings.MIN_STAKE_USD:
                            logger.info(
                                "[%s] skip bucket=%d: adjusted_stake=$%.2f < min $%.2f",
                                icao, edge.bucket_value, adjusted_stake, settings.MIN_STAKE_USD,
                            )
                            continue

                        from src.db.models import Signal, TradeDirection
                        sig_row = Signal(
                            market_id=market.id,
                            model_prob=edge.our_probability,
                            market_prob=edge.market_price,
                            edge=edge.edge,
                            direction=TradeDirection.BUY_YES,
                            confidence=edge.our_probability,
                        )
                        session.add(sig_row)
                        await session.flush()

                        trade = Trade(
                            signal_id=sig_row.id,
                            market_id=market.id,
                            direction=TradeDirection.BUY_YES,
                            stake_usd=adjusted_stake,
                            entry_price=edge.market_price,
                            status=TradeStatus.PENDING,
                        )
                        session.add(trade)
                        await session.flush()

                        order_ok = await place_order(trade, session)
                        if order_ok:
                            trade.status = TradeStatus.OPEN
                            exposure += adjusted_stake
                            total_trades += 1
                            await alerter._enqueue(
                                f"\U0001f4b0 *Unified trade* [{icao}]\n"
                                f"Bucket: {edge.bucket_value}°F | Edge: {edge.edge:.3f}\n"
                                f"Stake: ${adjusted_stake:.2f} | Price: {edge.market_price:.2f}\n"
                                f"Market: {market.question[:60]}",
                            )

              except Exception:
                skipped_cities += 1
                logger.warning("Unified pipeline: error evaluating %s, skipping", icao, exc_info=True)
                continue

            await session.commit()
            if skipped_cities:
                logger.warning("Unified pipeline: %d/%d cities skipped due to errors", skipped_cities, len(city_markets))
            logger.info("Unified pipeline complete: %d trades placed", total_trades)

    except Exception as exc:
        logger.exception("Unified pipeline failed")
        await alerter.send_system_error(exc, "unified pipeline")


def _is_binary_market(market) -> bool:
    """True if market is a binary threshold market (not a bracket)."""
    return (
        market.parsed_threshold is not None
        and market.parsed_operator is not None
        and market.parsed_operator != "bracket"
    )


def _make_binary_buckets(market, state) -> list[int]:
    """Generate integer temperature range for distribution over a binary market."""
    from src.signals.state_aggregator import WeatherState

    threshold = int(market.parsed_threshold)
    low = int(state.current_max_f) - 1
    high = max(threshold, int(state.forecast_peak_f)) + 10
    return list(range(low, high + 1))


def _binary_market_edge(dist, market, end_time, routine_count, depth):
    """Collapse distribution into cumulative probability for a binary market.

    Returns a BucketEdge or None if the market can't be evaluated.
    """
    from src.signals.edge_calculator import BucketEdge, _check_filters

    threshold = int(market.parsed_threshold)
    op = market.parsed_operator
    price = market.current_yes_price or 0.0

    # Cumulative probability based on operator
    if op in ("above", "at_least"):
        our_prob = sum(p for b, p in dist.probabilities.items() if b >= threshold)
    elif op in ("below", "at_most"):
        our_prob = sum(p for b, p in dist.probabilities.items() if b < threshold)
    elif op == "exactly":
        our_prob = dist.probabilities.get(threshold, 0.0)
    else:
        return None

    our_prob = round(our_prob, 4)
    edge = round(our_prob - price, 4)

    now = datetime.now(timezone.utc)
    minutes_to_close = (end_time - now).total_seconds() / 60.0

    reason = _check_filters(
        edge=edge, prob=our_prob, price=price,
        routine_count=routine_count,
        minutes_to_close=minutes_to_close,
        depth=depth,
    )

    return BucketEdge(
        bucket_value=threshold,
        our_probability=our_prob,
        market_price=price,
        edge=edge,
        passes=reason is None,
        reject_reason=reason,
    )


def _extract_bracket_buckets(market) -> list[int]:
    """Extract temperature bucket values from a bracket market's outcomes."""
    import re
    buckets: list[int] = []
    outcomes = market.outcomes or []
    for outcome in outcomes:
        if isinstance(outcome, str):
            match = re.search(r"(\d+)", outcome)
            if match:
                buckets.append(int(match.group(1)))
        elif isinstance(outcome, dict):
            val = outcome.get("value") or outcome.get("title", "")
            match = re.search(r"(\d+)", str(val))
            if match:
                buckets.append(int(match.group(1)))
    return sorted(set(buckets))


def _extract_market_prices(market, buckets: list[int]) -> dict[int, float]:
    """Map bucket values to current YES prices for bracket markets."""
    prices: dict[int, float] = {}
    if market.current_yes_price and buckets:
        prices[buckets[0]] = market.current_yes_price
    return prices


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

            # 5. Station bias recording (unified pipeline)
            if settings.UNIFIED_PIPELINE_ENABLED:
                try:
                    from src.ingestion.station_bias import record_daily_outcome
                    from src.ingestion.aviation import get_routine_daily_max
                    from src.ingestion.openmeteo import fetch_forecast
                    from src.signals.mapper import icao_for_location, geocode, CITY_ICAO

                    # Collect bias for each station that had active markets today
                    seen_icaos: set[str] = set()
                    stmt_markets = select(Market).where(
                        Market.parsed_location.isnot(None),
                        Market.parsed_variable == "temperature",
                    )
                    market_result = await session.execute(stmt_markets)
                    for mkt in market_result.scalars():
                        icao = icao_for_location(mkt.parsed_location) if mkt.parsed_location else None
                        if not icao or icao in seen_icaos:
                            continue
                        seen_icaos.add(icao)

                        max_f, count = await get_routine_daily_max(icao)
                        if max_f is None or count < 3:
                            continue

                        coords = geocode(mkt.parsed_location) if mkt.parsed_location else None
                        if not coords:
                            continue

                        forecast = await fetch_forecast(coords[0], coords[1])
                        if forecast is None:
                            continue

                        max_c = (max_f - 32.0) * 5.0 / 9.0
                        await record_daily_outcome(
                            session, icao,
                            datetime.now(timezone.utc),
                            max_c, forecast.peak_temp_c,
                        )
                    logger.info("Station bias recorded for %d stations", len(seen_icaos))
                except Exception:
                    logger.exception("Station bias recording failed (non-fatal)")

            # 6. Weekly calibration check (Sundays)
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
    if settings.UNIFIED_PIPELINE_ENABLED:
        scheduler.add_job(
            job_unified_pipeline,
            IntervalTrigger(minutes=settings.UNIFIED_PIPELINE_INTERVAL_MINUTES),
            id="unified_pipeline",
            max_instances=1,
            coalesce=True,
        )
        logger.info(
            "Unified pipeline enabled (every %dm); legacy pipelines disabled",
            settings.UNIFIED_PIPELINE_INTERVAL_MINUTES,
        )
    else:
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
