"""APScheduler-based pipeline orchestration and health-check server."""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import select

from src.config import settings
from src.db.engine import async_session, engine
from src.db.models import EvaluationLog, Market, Signal, Trade, TradeStatus
from src.execution.alerter import get_alerter
from src.ingestion.polymarket import scan_and_ingest
from src.resolution import (
    get_current_bankroll,
    get_current_exposure,
    resolve_trades,
)
from src.risk.drawdown import DrawdownLevel, DrawdownMonitor
from src.risk.kelly import size_locked_position, size_position
from src.signals.consensus import get_calibration_coefficients
from src.signals.lock_rules import evaluate_lock
from src.execution.polymarket_client import is_live, place_order

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession as AsyncSessionType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_scheduler: AsyncIOScheduler | None = None
_drawdown_monitor: DrawdownMonitor | None = None
_shutdown_event: asyncio.Event | None = None

# Fast-lock-poll bookkeeping. In-process only; reset on process restart.
# `_locked_markets_fired_today` prevents re-firing on a market we've already
# placed a lock order for. Cleared per-station at the station's local-day
# rollover by `_maybe_clear_per_station_caches` (run from the unified
# pipeline tick), not at a single global UTC time.
# `_unified_fired_today` is the dry-run-only sibling for the probability path
# (keyed by `(market_id, direction, bucket)` since brackets can fire several
# buckets per market). In dry-run, `place_order` is a no-op and the resulting
# Trade row stays `status=PENDING` (with `exchange_status='dry_run'`), so
# `current_exposure` — which filters on `status=OPEN` — doesn't grow and
# nothing else blocks the next 5-min tick from re-emitting the same alert.
# This set closes that gap. Live trades remain undeduped — the OPEN-trade
# exposure is the source of truth there.
# `_last_routine_seen` skips METARs the fast loop already processed.
# `_market_to_icao` lets the per-station rollover find which market_ids
# in `_locked_markets_fired_today` / `_unified_fired_today` belong to a
# given station.
# `_local_day_seen` tracks each station's last-observed local date so we
# can detect rollover.
_locked_markets_fired_today: set[str] = set()
_unified_fired_today: set[tuple[str, str, int]] = set()
_last_routine_seen: dict[str, datetime] = {}
_market_to_icao: dict[str, str] = {}
_local_day_seen: dict[str, date] = {}

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


_UNIFIED_CONCURRENCY = 8  # Max concurrent city aggregations

# Stations whose Polymarket resolution source diverges from the routine-METAR
# stream we consume (e.g. uses a different airport, an HKO-style city station,
# or a different rounding convention) — skip them entirely in the pipeline.
_EXCLUDED_ICAOS: set[str] = {"VHHH", "LLBG"}  # Hong Kong, Tel Aviv


def _maybe_clear_per_station_caches() -> None:
    """Drop per-station dedup / cache entries when a station's local day
    has rolled over since we last looked.

    Replaces the legacy global wipe at 22:00 UTC: each station now resets
    on its own local midnight, which is the actual data-day boundary the
    routine-METAR / lock-rule logic uses (see lock_rules._market_daily_max).

    Cheap to call every unified-pipeline tick — most calls are no-ops
    because nothing rolled over in the last 5 minutes.
    """
    from src.signals.mapper import ICAO_TIMEZONE, icao_timezone, today_local
    from src.signals.state_aggregator import clear_state_cache_for_icao

    for icao in ICAO_TIMEZONE.keys():
        try:
            tz = icao_timezone(icao)
            today = today_local(tz)
        except Exception:
            continue
        prev = _local_day_seen.get(icao)
        if prev is None:
            # First time we're seeing this station — record today's
            # local date but don't wipe state. Avoids clobbering dedup
            # entries written by fast-poll before the first unified tick.
            _local_day_seen[icao] = today
            continue
        if prev == today:
            continue
        # Local day rolled over — drop dedup market_ids that belong to
        # this station, the per-station routine cursor, and the cached
        # forecast/bias inputs so the next tick re-fetches.
        mids_for_icao = [m for m, ic in _market_to_icao.items() if ic == icao]
        for mid in mids_for_icao:
            _locked_markets_fired_today.discard(mid)
            _market_to_icao.pop(mid, None)
        if mids_for_icao:
            mids_set = set(mids_for_icao)
            _unified_fired_today.difference_update(
                {k for k in _unified_fired_today if k[0] in mids_set}
            )
        _last_routine_seen.pop(icao, None)
        clear_state_cache_for_icao(icao)
        _local_day_seen[icao] = today


async def job_resolve_trades() -> None:
    """Every 5 min — resolve any expired markets.

    Replaces the once-a-day resolution that used to live inside
    ``job_daily_settlement``. Markets close at different UTC instants
    depending on their station's local timezone; batching at 22:00 UTC left
    morning-UTC closures sitting in OPEN status for up to ~22h, polluting
    bankroll/exposure accounting in the meantime.

    Idempotent: ``resolve_trades`` already filters on
    ``Market.end_date < now`` and ``Trade.status == OPEN``; repeated calls
    only touch newly-elapsed markets.
    """
    alerter = get_alerter()
    try:
        async with async_session() as session:
            resolved = await resolve_trades(session)
            for trade in resolved:
                market = await session.get(Market, trade.market_id)
                await alerter.send_resolution(trade, market)
            await session.commit()
            if resolved:
                logger.info("Resolved %d trade(s)", len(resolved))
    except Exception as exc:
        logger.exception("resolve_trades failed")
        try:
            await alerter.send_system_error(exc, "resolve_trades")
        except Exception:
            logger.warning("Alerter failed to send resolve_trades error")


async def job_unified_pipeline() -> None:
    """Every 5 min — unified METAR + Open-Meteo + market pipeline.

    Runs every UNIFIED_PIPELINE_INTERVAL_MINUTES.

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
        from src.execution.polymarket_client import get_best_bid_ask, get_orderbook_depth, get_token_ids

        # Reset per-station dedup / cache for any station whose local day
        # has just rolled over. Cheap, runs every tick.
        _maybe_clear_per_station_caches()

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
                    logger.debug("skip %s: parsed_location is None", m.id[:12])
                    continue
                icao = icao_for_location(loc)
                if not icao:
                    continue
                if icao in _EXCLUDED_ICAOS:
                    logger.debug(
                        "skip %s: ICAO %s excluded from pipeline (resolution source divergence)",
                        m.id[:12], icao,
                    )
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

            # Refresh consensus calibration once per tick when enabled.
            # The fit is cached for `_CACHE_TTL_SEC` (30 min) so back-to-back
            # ticks don't re-query; `apply_calibration` reads sync from the
            # cache inside `_binary_market_edge`.
            if settings.APPLY_CALIBRATION:
                try:
                    from src.signals.consensus import refresh_calibration
                    await refresh_calibration(session)
                except Exception:
                    logger.warning(
                        "calibration refresh failed; falling back to uncalibrated",
                        exc_info=True,
                    )

            for icao, state in city_states.items():
              try:
                for market in city_markets[icao]:
                    end_time = market.end_date or datetime.now(timezone.utc) + timedelta(hours=24)

                    now_utc = datetime.now(timezone.utc)
                    if _should_skip_future_day(market, now_utc, station_icao=icao):
                        from src.signals.mapper import (
                            icao_timezone as _tz_for,
                            resolve_target_local_day as _target_day,
                            today_local as _today_local,
                        )
                        _tz = _tz_for(icao)
                        logger.info(
                            "[%s] skip %s: target_local=%s today_local=%s — future-day",
                            icao, market.id[:12],
                            _target_day(market.end_date, _tz),
                            _today_local(_tz),
                        )
                        continue

                    # Orderbook depth at market YES price; refresh live price from CLOB.
                    # We keep the mid as `current_yes_price` for display + the
                    # near-resolved skip below, but downstream edge/lock evaluation
                    # uses the per-side BUY price (yes_ask for BUY YES, 1-yes_bid
                    # for BUY NO) so wide post-move spreads don't invent fake edges.
                    mkt_depth = 0.0
                    yes_bid: float | None = None
                    yes_ask: float | None = None
                    live_price: float | None = None
                    token_ids = await get_token_ids(market.id)
                    if token_ids:
                        quote = get_best_bid_ask(token_ids[0])
                        if quote:
                            yes_bid, yes_ask = quote
                            live_price = (yes_bid + yes_ask) / 2
                        if live_price:
                            mkt_depth = get_orderbook_depth(token_ids[0], live_price)

                    # Don't persist the live mid back onto the ORM row —
                    # concurrent writers caused cross-transaction deadlocks.
                    # Use the live quote in-tick; the stored value (refreshed
                    # by job_scan_markets every 15 min) is the durable copy.
                    price = live_price if live_price is not None else (market.current_yes_price or 0.0)

                    # --- Lock-rule fast path (deterministic physical lock-in) ---
                    # Evaluate before the near-resolved skip so the 0.90-0.95
                    # "we read the max on METAR before Wunderground updates" zone
                    # is tradeable rather than filtered out as "too close to resolved".
                    if (
                        settings.LOCK_RULE_ENABLED
                        and _is_binary_market(market)
                        and price > 0
                    ):
                        lock_executed = await _try_lock_rule_trade(
                            session=session, market=market, state=state,
                            yes_price=price, token_ids=token_ids,
                            yes_depth=mkt_depth, end_time=end_time,
                            bankroll=bankroll, exposure=exposure,
                            monitor=monitor, alerter=alerter, icao=icao,
                            yes_bid=yes_bid, yes_ask=yes_ask,
                        )
                        if lock_executed is not None:
                            if lock_executed:
                                exposure += lock_executed
                                total_trades += 1
                            continue

                    # Skip near-resolved markets — any "edge" at the tails is noise.
                    # Prefer bid/ask over mid: a dust order on the dead side can
                    # leave mid in the middle of the book even when bid≈1.0 or
                    # ask≈0.0 already locks the outcome.
                    near_yes_lock = yes_bid is not None and yes_bid >= 0.99
                    near_no_lock = (
                        yes_ask is not None and 0.0 < yes_ask <= 0.01
                    )
                    if near_yes_lock or near_no_lock or price >= 0.99 or (price > 0 and price <= 0.01):
                        logger.info(
                            "[%s] skip %s: market effectively resolved "
                            "(mid=%.2f bid=%s ask=%s)",
                            icao, market.id[:12], price,
                            f"{yes_bid:.3f}" if yes_bid is not None else "?",
                            f"{yes_ask:.3f}" if yes_ask is not None else "?",
                        )
                        continue

                    if not _is_binary_market(market):
                        logger.debug(
                            "[%s] skip %s: not a binary/range market (operator=%r)",
                            icao, market.id[:12], market.parsed_operator,
                        )
                        continue

                    buckets = _make_binary_buckets(market, state)
                    dist = compute_distribution(state, buckets)

                    # Build a deferred NO-depth fetcher; only invoked when
                    # _binary_market_edge actually picks the NO side, so
                    # we don't pay an extra CLOB call on the (more common)
                    # YES branch.
                    def _no_depth_for_market(
                        market=market, token_ids=token_ids,
                        price=price, yes_bid=yes_bid,
                    ):
                        if not token_ids:
                            return 0.0
                        # Use the actual NO buy price (1 - yes_bid) when we
                        # have a quote — measures depth at the price we'd
                        # really pay, not at the dust-spread mid.
                        if yes_bid is not None and yes_bid > 0:
                            no_price = max(0.001, 1.0 - yes_bid)
                        else:
                            no_price = max(0.001, 1.0 - price)
                        return get_orderbook_depth(token_ids[1], no_price)

                    edge_result = _binary_market_edge(
                        dist, market, end_time,
                        state.routine_count_today,
                        depth_yes=mkt_depth,
                        depth_no_fn=_no_depth_for_market,
                        yes_bid=yes_bid, yes_ask=yes_ask,
                    )
                    if edge_result is None:
                        logger.info(
                            "[%s] skip %s: edge None (operator=%r)",
                            icao, market.id[:12], market.parsed_operator,
                        )
                        continue

                    op_symbol = {
                        "above": "≥", "at_least": "≥",
                        "below": "<", "at_most": "≤",
                        "exactly": "=", "range": "∈", "bracket": "∈",
                    }.get(market.parsed_operator, "?")
                    tag = "PASS" if edge_result.passes else edge_result.reject_reason
                    unit = _market_unit(market)
                    rng = market_range_f(market)
                    if rng is not None and rng[0] != rng[1]:
                        label = f"[{_display_bucket(rng[0], unit)}-{_display_bucket(rng[1], unit)}]{unit}"
                    elif market.parsed_threshold is not None:
                        label = f"{_display_bucket(int(market.parsed_threshold), unit)}{unit}"
                    else:
                        label = f"{edge_result.bucket_value}{unit}"
                    side_label = "YES" if edge_result.direction.value == "BUY_YES" else "NO"
                    logger.info(
                        "[%s] %s %s%s [%s]: P=%.3f, mkt=%.3f, edge=%+.3f %s | %s",
                        icao, market.parsed_operator, op_symbol, label, side_label,
                        edge_result.our_probability, edge_result.market_price,
                        edge_result.edge, tag, dist.reasoning,
                    )

                    # Telemetry: log every evaluation (pass or fail) so
                    # filter-tuning can be backtested against rejected
                    # candidates. Depth on the actual buy side; minutes
                    # to close from end_time. See _log_evaluation docstring.
                    eval_side_depth = (
                        mkt_depth if edge_result.direction.value == "BUY_YES"
                        else _no_depth_for_market()
                    )
                    eval_minutes_to_close = (
                        (end_time - now_utc).total_seconds() / 60.0
                        if end_time else None
                    )
                    await _log_evaluation(
                        session,
                        market_id=market.id,
                        direction=edge_result.direction,
                        signal_kind="probability",
                        model_prob=edge_result.our_probability,
                        market_prob=edge_result.market_price,
                        edge=edge_result.edge,
                        passes=edge_result.passes,
                        reject_reason=edge_result.reject_reason,
                        depth_usd=eval_side_depth or None,
                        minutes_to_close=eval_minutes_to_close,
                        routine_count=state.routine_count_today,
                    )

                    edges = [edge_result]
                    passing = [edge_result] if edge_result.passes else []

                    dd_state = monitor.check(bankroll)
                    for edge in passing:
                        # Side-effective depth for the chosen direction —
                        # the NO branch already fetched its own depth via
                        # _no_depth_for_market during edge evaluation; the
                        # YES branch reuses mkt_depth.
                        side_depth = (
                            mkt_depth
                            if edge.direction.value == "BUY_YES"
                            else _no_depth_for_market()
                        )
                        pos = size_position(
                            bankroll=bankroll,
                            model_prob=edge.our_probability,
                            market_prob=edge.market_price,
                            current_exposure=exposure,
                            max_position_usd=settings.MAX_POSITION_USD,
                            orderbook_depth=side_depth or None,
                        )

                        adjusted_stake = pos.stake_usd * dd_state.size_multiplier
                        logger.info(
                            "[%s] sizing %s bucket=%d: kelly_stake=$%.2f, dd_mult=%.2f, adjusted=$%.2f, bankroll=$%.0f, exposure=$%.0f",
                            icao, side_label, edge.bucket_value, pos.stake_usd,
                            dd_state.size_multiplier, adjusted_stake, bankroll, exposure,
                        )
                        if adjusted_stake < settings.MIN_STAKE_USD:
                            logger.info(
                                "[%s] skip %s bucket=%d: adjusted_stake=$%.2f < min $%.2f",
                                icao, side_label, edge.bucket_value,
                                adjusted_stake, settings.MIN_STAKE_USD,
                            )
                            continue

                        # Hard guard: don't double-bet a (market, direction).
                        # ``_has_active_trade`` checks the DB for any
                        # PENDING/OPEN trade on this pair — catches both
                        # live re-fires (after restart, when in-process
                        # dedup is empty) and dry-run repeats.
                        # ``_unified_fired_today`` is a same-tick speed-up
                        # that lets us skip the DB roundtrip when we know
                        # we've already fired this tick.
                        dedup_key = (
                            market.id, edge.direction.value, edge.bucket_value,
                        )
                        if dedup_key in _unified_fired_today:
                            logger.info(
                                "[%s] skip %s bucket=%d: already fired this tick",
                                icao, side_label, edge.bucket_value,
                            )
                            continue
                        if await _has_active_trade(session, market.id, edge.direction):
                            logger.info(
                                "[%s] skip %s bucket=%d: active trade exists for this market+side",
                                icao, side_label, edge.bucket_value,
                            )
                            _unified_fired_today.add(dedup_key)
                            _market_to_icao[market.id] = icao
                            continue

                        # Side-effective probability lands in Signal.model_prob
                        # so consensus calibration treats YES and NO uniformly.
                        # See src/signals/consensus.py. The (market, direction)
                        # uniqueness constraint means this is INSERT-or-REFRESH:
                        # repeat evaluations of the same market+side update the
                        # existing row instead of inserting duplicates.
                        sig_row = await _upsert_signal(
                            session,
                            market_id=market.id,
                            direction=edge.direction,
                            model_prob=edge.our_probability,
                            market_prob=edge.market_price,
                            edge=edge.edge,
                            confidence=edge.our_probability,
                        )

                        trade = Trade(
                            signal_id=sig_row.id,
                            market_id=market.id,
                            direction=edge.direction,
                            stake_usd=adjusted_stake,
                            entry_price=edge.market_price,
                            status=TradeStatus.PENDING,
                        )
                        session.add(trade)
                        await session.flush()

                        order_ok = await place_order(
                            trade, session,
                            submit_yes_bid=yes_bid,
                            submit_yes_ask=yes_ask,
                            submit_depth_usd=side_depth or None,
                        )
                        # In dry-run, ``place_order`` is a no-op and never
                        # updates fill fields, so ``trade.stake_usd`` is still
                        # the requested value. Keep it populated for paper-
                        # trade analysis; ``status=PENDING`` + ``exchange_status
                        # ='dry_run'`` keep it out of OPEN-filtered exposure
                        # / PnL math. Any future query that sums stake_usd
                        # must filter on one of those.
                        is_dry_run = trade.exchange_status == "dry_run"
                        if order_ok and is_dry_run:
                            trade.status = TradeStatus.PENDING
                            unit = _market_unit(market)
                            display_bucket = _display_bucket(edge.bucket_value, unit)
                            await alerter._enqueue(
                                f"\U0001f4b0 *Unified trade (dry-run)* [{icao}] {side_label}\n"
                                f"Bucket: {display_bucket}{unit} | Edge: {edge.edge:.3f}\n"
                                f"Indicative: ${trade.stake_usd:.2f} @ {trade.entry_price or edge.market_price:.3f}\n"
                                f"Market: {market.question[:60]}",
                            )
                            _unified_fired_today.add(dedup_key)
                            _market_to_icao[market.id] = icao
                        elif order_ok and (trade.stake_usd or 0.0) > 0:
                            trade.status = TradeStatus.OPEN
                            actual_stake = trade.stake_usd
                            exposure += actual_stake
                            total_trades += 1
                            unit = _market_unit(market)
                            display_bucket = _display_bucket(edge.bucket_value, unit)
                            fill = trade.fill_price or edge.market_price
                            await alerter._enqueue(
                                f"\U0001f4b0 *Unified trade* [{icao}] {side_label}\n"
                                f"Bucket: {display_bucket}{unit} | Edge: {edge.edge:.3f}\n"
                                f"Filled: ${actual_stake:.2f} (req ${adjusted_stake:.2f}) @ {fill:.3f}\n"
                                f"Market: {market.question[:60]}",
                            )
                        elif order_ok:
                            # FAK posted but didn't match — leave PENDING so
                            # the next pipeline tick can re-evaluate. No
                            # exposure increment, no dedup.
                            trade.status = TradeStatus.PENDING
                            logger.info(
                                "[%s] %s %s bucket=%d: FAK posted, no fill at quoted price",
                                icao, side_label, market.id[:12], edge.bucket_value,
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
    """True if market is a single-outcome binary YES/NO market.

    All temperature markets we trade are binary at the CLOB level (one
    YES token, one NO token) — the "bracket" operator from the parser
    refers to questions like "Will the highest be between 88-89°F?" which
    are *single binary* markets asking about a 2°F window, not a multi-
    outcome bracket. We unify them here and let downstream routing pick
    threshold-vs-range handling based on market_range_f().
    """
    op = market.parsed_operator
    if op is None:
        return False
    if op in ("above", "at_least", "below", "at_most"):
        return market.parsed_threshold is not None
    if op == "exactly":
        return market.parsed_threshold is not None
    if op in ("range", "bracket"):
        # bracket-operator markets need a parseable °F window in the
        # question text; otherwise they're true multi-outcome brackets
        # (very rare in practice for weather markets).
        return market_range_f(market) is not None
    return False


def market_range_f(market) -> tuple[int, int] | None:
    """Inclusive integer °F range for a range-style binary market.

    Recognized shapes:
      * "Will the highest temperature in X be between 88-89°F on …?"
        → (88, 89)  — a 2°F-wide window
      * "Will the highest temperature in X be 17°C on …?"
        → (62, 63)  — integer °F values that round to 17°C
      * "Will the highest temperature in X be 88°F on …?"
        → (88, 88)  — single-degree window

    Returns None for one-sided threshold markets (above/below) and for
    questions where neither pattern matches.
    """
    import math

    op = market.parsed_operator

    # 1. Explicit "between X-Y°[FC]" in the question text.
    if op in ("range", "bracket"):
        from src.ingestion.polymarket import parse_bracket_from_question

        parsed = parse_bracket_from_question(market.question or "")
        if parsed is not None:
            low_f, high_f_excl = parsed  # high is exclusive (+1)
            return (int(round(low_f)), int(round(high_f_excl)) - 1)
        return None

    # 2. "Exactly" — Celsius single-value or Fahrenheit single-value.
    if op == "exactly" and market.parsed_threshold is not None:
        if _market_unit(market) == "°C":
            c_int = round((market.parsed_threshold - 32.0) * 5.0 / 9.0)
            lo_c = c_int - 0.5
            hi_c = c_int + 0.5
            lo_f = lo_c * 9.0 / 5.0 + 32.0
            hi_f = hi_c * 9.0 / 5.0 + 32.0
            f_lo = int(math.ceil(lo_f - 1e-9))
            f_hi = int(math.floor(hi_f - 1e-9))
            if f_hi < f_lo:
                return (f_lo, f_lo)
            return (f_lo, f_hi)
        f = int(round(market.parsed_threshold))
        return (f, f)

    return None


def _should_skip_future_day(market, now: datetime, station_icao: str | None = None) -> bool:
    """True when the market's data day is *strictly later* than today
    in the **station's local timezone**.

    Each Polymarket weather market resolves to a single local-day max at
    the station that resolves it. The data day is computed by
    ``resolve_target_local_day(end_date, station_tz)`` (see mapper for the
    derivation). A market for tomorrow's local data still has no
    observations — skip. A market for today's local data, or yesterday's
    that hasn't yet been settled, stays in scope.

    Falls back to the legacy UTC-date comparison when no station is
    available — keeps the rule defined for callers that don't know the
    station yet.
    """
    if not market.end_date:
        return False
    if station_icao is None:
        return market.end_date.date() > now.date()

    from src.signals.mapper import (
        icao_timezone,
        resolve_target_local_day,
        today_local,
    )
    tz = icao_timezone(station_icao)
    target = resolve_target_local_day(market.end_date, tz)
    if target is None:
        return False
    return target > today_local(tz)


def _market_unit(market) -> str:
    """Return '°C' or '°F' based on the market's original question text."""
    q = (market.question or "").upper()
    if "°C" in q or "CELSIUS" in q:
        return "°C"
    return "°F"


def _display_bucket(bucket_f: int, unit: str) -> int:
    """Convert an internal °F bucket to the market's display unit, rounded."""
    if unit == "°C":
        return round((bucket_f - 32) * 5 / 9)
    return bucket_f


def _make_binary_buckets(market, state) -> list[int]:
    """Generate the integer °F bucket grid for a single-binary market.

    The grid spans from one degree below the observed max up through the
    forecast peak (or the threshold/range upper bound, whichever is
    higher) plus a 10°F headroom — wide enough to capture upside tails
    without wasting compute on far-out buckets.
    """
    rng = market_range_f(market)
    if rng is not None:
        upper = max(rng[1], int(state.forecast_peak_f))
    elif market.parsed_threshold is not None:
        upper = max(int(market.parsed_threshold), int(state.forecast_peak_f))
    else:
        upper = int(state.forecast_peak_f)
    low = int(state.current_max_f) - 1
    return list(range(low, upper + 11))


def _binary_market_edge(
    dist,
    market,
    end_time,
    routine_count,
    depth_yes: float,
    depth_no_fn=None,
    yes_bid: float | None = None,
    yes_ask: float | None = None,
):
    """Pick the best side (YES or NO) of a binary market and gate it.

    Computes ``our_prob_yes`` from the distribution under the operator,
    then evaluates *both* sides at their actual BUY-side cost:

      * YES: price = ``yes_ask`` (what a YES buyer pays)
      * NO:  price = ``1 - yes_bid`` (what a NO buyer pays; equivalent to
             the NO-token ask given the ``YES + NO = 1`` constraint)

    The (yes_bid, yes_ask) quote is optional. When omitted, both sides
    fall back to ``market.current_yes_price`` symmetrically — preserves
    legacy behavior for callers (and tests) that don't have a quote.

    Why asymmetric pricing matters: after a sharp move the orderbook can
    have stale dust on the dead side (e.g. bid=0.20, ask=0.55 on a
    market that's actually trading near YES=0). The arithmetic mid then
    invents a phantom "edge" that wouldn't fill. Charging each side its
    real ask cost makes both sides correctly fail the MIN_EDGE filter.

    For a binary market ``edge_NO == -edge_YES`` only when the spread is
    zero; with a real spread the two sides see independent edges.

    ``depth_no_fn`` is an optional zero-arg callable returning NO-side
    orderbook depth in USD; only invoked when the chosen direction is
    NO, so the additional CLOB call is skipped on the (more common) YES
    side.

    Returns the passing-side ``BucketEdge`` if one passes; otherwise the
    higher-edge candidate (with ``passes=False`` and a reject reason),
    so callers can still log what was attempted.
    """
    from src.db.models import TradeDirection
    from src.signals.edge_calculator import BucketEdge, _check_filters

    op = market.parsed_operator
    mid_price = market.current_yes_price or 0.0
    # Per-side BUY costs. Fall back to the symmetric mid when a real
    # quote isn't supplied (keeps legacy callers + tests working).
    yes_buy_price = yes_ask if (yes_ask is not None and yes_ask > 0) else mid_price
    no_buy_price = (1.0 - yes_bid) if (yes_bid is not None and yes_bid > 0) else (1.0 - mid_price)
    bucket_value: int

    if op in ("above", "at_least"):
        threshold = int(market.parsed_threshold)
        bucket_value = threshold
        our_prob_yes = sum(p for b, p in dist.probabilities.items() if b >= threshold)
    elif op in ("below", "at_most"):
        threshold = int(market.parsed_threshold)
        bucket_value = threshold
        our_prob_yes = sum(p for b, p in dist.probabilities.items() if b < threshold)
    elif op in ("exactly", "range", "bracket"):
        rng = market_range_f(market)
        if rng is None:
            return None
        low, high = rng
        bucket_value = (low + high) // 2
        our_prob_yes = sum(p for b, p in dist.probabilities.items() if low <= b <= high)
    else:
        return None

    our_prob_yes = round(our_prob_yes, 4)
    yes_edge = round(our_prob_yes - yes_buy_price, 4)
    no_prob = round(1.0 - our_prob_yes, 4)
    no_price = round(no_buy_price, 4)
    no_edge = round(no_prob - no_price, 4)
    # Keep the variable name `yes_price` for the BucketEdge.market_price
    # field on the YES branch — reads as "what a YES buyer pays".
    yes_price = round(yes_buy_price, 4)

    now = datetime.now(timezone.utc)
    minutes_to_close = (end_time - now).total_seconds() / 60.0

    # Pick the side whose edge is positive. If both are non-positive,
    # the higher-edge side is still returned (with passes=False) so the
    # caller's log line shows what was considered.
    if no_edge > yes_edge:
        direction = TradeDirection.BUY_NO
        side_prob = no_prob
        side_price = no_price
        side_edge = no_edge
        side_depth = depth_no_fn() if depth_no_fn is not None else 0.0
    else:
        direction = TradeDirection.BUY_YES
        side_prob = our_prob_yes
        side_price = yes_price
        side_edge = yes_edge
        side_depth = depth_yes

    # Apply consensus calibration when enabled — corrects the
    # side-effective probability based on resolved-signal history. No-op
    # when `APPLY_CALIBRATION=False`, when fewer than
    # `MIN_CALIBRATION_SAMPLES` resolved signals exist, or when the cache
    # is stale (refresh happens at the top of each tick).
    from src.signals.consensus import apply_calibration
    side_prob_raw = side_prob
    side_prob, calibrated = apply_calibration(side_prob_raw)
    if calibrated:
        side_edge = round(side_prob - side_price, 4)
        logger.debug(
            "calibrated %s: prob %.3f→%.3f, edge %.3f→%.3f",
            direction.value, side_prob_raw, side_prob,
            round(side_prob_raw - side_price, 4), side_edge,
        )

    reason = _check_filters(
        edge=side_edge, prob=side_prob, price=side_price,
        routine_count=routine_count,
        minutes_to_close=minutes_to_close,
        depth=side_depth,
    )

    return BucketEdge(
        bucket_value=bucket_value,
        our_probability=side_prob,
        market_price=side_price,
        edge=side_edge,
        passes=reason is None,
        reject_reason=reason,
        direction=direction,
    )


async def _log_evaluation(
    session,
    *,
    market_id: str,
    direction,
    signal_kind: str,
    model_prob: float,
    market_prob: float,
    edge: float,
    passes: bool,
    reject_reason: str | None,
    depth_usd: float | None,
    minutes_to_close: float | None,
    routine_count: int | None,
) -> None:
    """Append one ``EvaluationLog`` row capturing this edge evaluation.

    Called by BOTH the probability path and ``_try_lock_rule_trade`` for
    EVERY candidate (passing or rejected). Without this row, MIN_EDGE /
    MIN_PROBABILITY / MIN_DEPTH_USD tuning is blind because ``signals``
    only carries passing edges and is now de-duplicated to one row per
    (market, side). Append-only, no UPSERT — each tick's evaluation is a
    separate calibration data point.
    """
    session.add(EvaluationLog(
        market_id=market_id,
        direction=direction,
        signal_kind=signal_kind,
        model_prob=model_prob,
        market_prob=market_prob,
        edge=edge,
        passes=passes,
        reject_reason=reject_reason,
        depth_usd=depth_usd,
        minutes_to_close=minutes_to_close,
        routine_count=routine_count,
    ))
    # Don't flush here — the next session.flush() in the caller (e.g.
    # _upsert_signal or commit) will batch these along with other writes.


async def _has_active_trade(
    session, market_id: str, direction,
) -> bool:
    """True iff a PENDING or OPEN Trade row already exists for this pair.

    Hard guard against duplicate firing. Replaces the load-bearing role of
    the in-process ``_unified_fired_today`` / ``_locked_markets_fired_today``
    sets — those are kept as same-tick speed-ups but no longer the only
    line of defence. In live mode this is the only thing that prevents
    the bot from re-betting the same market+side every tick until the
    Kelly exposure cap kicks in. In dry-run it stops the Signal/Trade
    table from accumulating one row per tick.

    Notes:
      - Filters on status only, no Market.end_date check needed: PENDING/
        OPEN trades on resolved markets are fixed up by ``resolve_trades``
        within minutes of expiry, so a stale row blocking an already-
        resolved market is a non-issue (we wouldn't trade into it anyway).
      - The migration ``i9j0k1l2m3n4`` collapsed pre-existing duplicate
        PENDING rows so this guard doesn't permanently lock out markets
        that already accumulated multiple dry-run attempts.
    """
    result = await session.execute(
        select(Trade.id).where(
            Trade.market_id == market_id,
            Trade.direction == direction,
            Trade.status.in_([TradeStatus.PENDING, TradeStatus.OPEN]),
        ).limit(1)
    )
    return result.scalar_one_or_none() is not None


async def _upsert_signal(
    session,
    *,
    market_id: str,
    direction,
    model_prob: float,
    market_prob: float,
    edge: float,
    confidence: float | None,
    signal_kind: str = "probability",
    lock_branch: str | None = None,
    lock_routine_count: int | None = None,
    lock_observed_max_f: float | None = None,
) -> Signal:
    """Insert-or-refresh the unique ``(market_id, direction)`` Signal row.

    Schema-level ``uq_signals_market_direction`` (migration
    ``i9j0k1l2m3n4``) means we'd otherwise collide on every re-evaluation
    tick. SELECT-then-INSERT-or-UPDATE keeps the ORM identity intact so
    callers can still take ``sig_row.id`` for the Trade FK. Refreshes
    ``created_at`` so callers can see "this signal was last evaluated at
    X" in the DB.

    ``signal_kind`` and the ``lock_*`` fields land on the row regardless
    of whether it's INSERT or UPDATE; a market that flips between the
    probability and lock paths between ticks (e.g. early-day probability
    edge → lock fires when the threshold gets crossed) will overwrite
    the kind/branch on the next tick. That's the intended semantic:
    Signal reflects the current trading rationale, not history.
    """
    existing = await session.execute(
        select(Signal).where(
            Signal.market_id == market_id,
            Signal.direction == direction,
        )
    )
    sig_row = existing.scalar_one_or_none()
    if sig_row is None:
        sig_row = Signal(
            market_id=market_id,
            direction=direction,
            model_prob=model_prob,
            market_prob=market_prob,
            edge=edge,
            confidence=confidence,
            signal_kind=signal_kind,
            lock_branch=lock_branch,
            lock_routine_count=lock_routine_count,
            lock_observed_max_f=lock_observed_max_f,
        )
        session.add(sig_row)
    else:
        sig_row.model_prob = model_prob
        sig_row.market_prob = market_prob
        sig_row.edge = edge
        sig_row.confidence = confidence
        sig_row.signal_kind = signal_kind
        sig_row.lock_branch = lock_branch
        sig_row.lock_routine_count = lock_routine_count
        sig_row.lock_observed_max_f = lock_observed_max_f
        sig_row.created_at = datetime.now(timezone.utc)
    await session.flush()
    return sig_row


async def _try_lock_rule_trade(
    *,
    session,
    market,
    state,
    yes_price: float,
    token_ids,
    yes_depth: float,
    end_time: datetime,
    bankroll: float,
    exposure: float,
    monitor: DrawdownMonitor,
    alerter,
    icao: str,
    yes_bid: float | None = None,
    yes_ask: float | None = None,
) -> float | None:
    """Evaluate lock-rule conditions and place order if triggered.

    ``yes_bid`` / ``yes_ask``: optional live quote. When supplied, the
    side we're buying is charged its real ask cost (yes_ask for YES,
    1-yes_bid for NO) instead of the symmetric mid carried in
    ``yes_price``. This prevents a wide post-move spread from making a
    locked market look mid-priced and slipping through the
    ``LOCK_RULE_MAX_PRICE`` guard.

    Returns:
      None — no lock fired; caller should fall through to probability path.
      0.0  — lock fired but was not executable (price out of range, order
             failed, depth insufficient, etc.). Caller should `continue`.
      >0   — stake in USD that was actually placed. Caller should `continue`
             and add to exposure counter.
    """
    from src.execution.polymarket_client import get_orderbook_depth
    from src.signals.edge_calculator import _check_filters

    decision = evaluate_lock(state, market)
    if decision.side is None or decision.direction is None:
        # No lock fired — nothing to log; the probability path will emit
        # its own EvaluationLog row for this market on this tick.
        return None

    # Effective price needs to land before _has_active_trade so the
    # EvaluationLog row carries the actual market_prob even on early
    # rejections (active trade exists, etc.).
    if decision.side == "YES":
        effective_price = (
            yes_ask if (yes_ask is not None and yes_ask > 0) else yes_price
        )
    else:
        effective_price = (
            (1.0 - yes_bid) if (yes_bid is not None and yes_bid > 0)
            else (1.0 - yes_price)
        )

    now = datetime.now(timezone.utc)
    minutes_to_close = (end_time - now).total_seconds() / 60.0

    async def _log_lock_eval(passes: bool, reject_reason: str | None, depth_usd: float | None) -> None:
        await _log_evaluation(
            session,
            market_id=market.id,
            direction=decision.direction,
            signal_kind="lock",
            model_prob=1.0,
            market_prob=effective_price,
            edge=1.0 - effective_price,
            passes=passes,
            reject_reason=reject_reason,
            depth_usd=depth_usd,
            minutes_to_close=minutes_to_close,
            routine_count=decision.routine_count,
        )

    # Hard guard against double-betting. Mirrors the probability path.
    # ``_locked_markets_fired_today`` is the same-tick speed-up; the DB
    # check is the durable line of defence (survives restarts and gates
    # both modes). Returning 0.0 (not None) so the caller treats this as
    # "lock evaluated but not executed" and skips the probability path
    # for this market on this tick.
    if market.id in _locked_markets_fired_today:
        logger.info(
            "[%s] LOCK %s %s: already fired this tick (in-process dedup)",
            icao, decision.side, market.id[:12],
        )
        await _log_lock_eval(False, "fired this tick", None)
        return 0.0
    if await _has_active_trade(session, market.id, decision.direction):
        logger.info(
            "[%s] LOCK %s %s: active trade exists for this market+side, skipping",
            icao, decision.side, market.id[:12],
        )
        _locked_markets_fired_today.add(market.id)
        _market_to_icao[market.id] = icao
        await _log_lock_eval(False, "active trade exists", None)
        return 0.0
    if not (
        settings.LOCK_RULE_MIN_PRICE
        <= effective_price
        <= settings.LOCK_RULE_MAX_PRICE
    ):
        logger.info(
            "[%s] lock %s %s: price %.2f outside [%.2f, %.2f]",
            icao, decision.side, market.id[:12], effective_price,
            settings.LOCK_RULE_MIN_PRICE, settings.LOCK_RULE_MAX_PRICE,
        )
        await _log_lock_eval(
            False,
            f"price {effective_price:.2f} outside [{settings.LOCK_RULE_MIN_PRICE}, {settings.LOCK_RULE_MAX_PRICE}]",
            None,
        )
        return 0.0

    # Depth against the side we're actually buying.
    if decision.side == "YES":
        buy_depth = yes_depth
    else:
        buy_depth = (
            get_orderbook_depth(token_ids[1], effective_price)
            if token_ids else 0.0
        )

    # Reuse the existing filter helper for routine-count / close-buffer / depth.
    # Pass stub prob/edge/price values that will pass those specific checks; we're
    # not edge-gating here, only piggy-backing on the shared sanity filters.
    # The lock-rule already gates routine_count per its own rules (allowing
    # 2 routines for super-margin EASY locks), so the filter just guards
    # the floor of 2 here — preventing single-METAR fluke trades regardless.
    reject = _check_filters(
        edge=1.0,
        prob=1.0,
        price=max(settings.MIN_ENTRY_PRICE, min(settings.MAX_ENTRY_PRICE, effective_price)),
        routine_count=state.routine_count_today,
        minutes_to_close=minutes_to_close,
        depth=buy_depth,
        min_routine_count=2,
    )
    if reject is not None:
        logger.info(
            "[%s] lock %s %s rejected by filter: %s",
            icao, decision.side, market.id[:12], reject,
        )
        await _log_lock_eval(False, reject, buy_depth or None)
        return 0.0

    # Lock candidate cleared all gates — emit the "passes" log row before
    # the order goes out so backtests can correlate evaluations with trade
    # outcomes via market_id+direction+created_at.
    await _log_lock_eval(True, None, buy_depth or None)

    pos = size_locked_position(
        bankroll=bankroll,
        price=effective_price,
        current_exposure=exposure,
        orderbook_depth=buy_depth or None,
    )
    dd_state = monitor.check(bankroll)
    stake = pos.stake_usd * dd_state.size_multiplier

    logger.info(
        "[%s] LOCK %s %s: margin=%.1f°F, price=%.2f, stake=$%.2f "
        "(raw=$%.2f, dd_mult=%.2f) | %s",
        icao, decision.side, market.id[:12], decision.margin_f,
        effective_price, stake, pos.stake_usd, dd_state.size_multiplier,
        "; ".join(decision.reasons),
    )

    if stake < settings.MIN_STAKE_USD:
        logger.info(
            "[%s] LOCK %s %s: stake $%.2f < min $%.2f, skipping",
            icao, decision.side, market.id[:12], stake, settings.MIN_STAKE_USD,
        )
        return 0.0

    # model_prob=1.0 because the lock rule is deterministic (no probability
    # estimate to record). confidence carries the lock margin in °F so the
    # detail view can show "how locked was this". Lock fields tag the
    # branch + observation context so post-mortems can split realised P&L
    # by which lock path produced the signal.
    sig_row = await _upsert_signal(
        session,
        market_id=market.id,
        direction=decision.direction,
        model_prob=1.0,
        market_prob=effective_price,
        edge=1.0 - effective_price,
        confidence=decision.margin_f,
        signal_kind="lock",
        lock_branch=decision.branch,
        lock_routine_count=decision.routine_count,
        lock_observed_max_f=decision.observed_max_f,
    )

    trade = Trade(
        signal_id=sig_row.id,
        market_id=market.id,
        direction=decision.direction,
        stake_usd=stake,
        entry_price=effective_price,
        status=TradeStatus.PENDING,
    )
    session.add(trade)
    await session.flush()

    order_ok = await place_order(
        trade, session,
        submit_yes_bid=yes_bid,
        submit_yes_ask=yes_ask,
        submit_depth_usd=buy_depth or None,
    )
    if not order_ok:
        logger.warning(
            "[%s] LOCK %s %s: order placement failed",
            icao, decision.side, market.id[:12],
        )
        return 0.0

    # In dry-run, ``place_order`` is a no-op and never updates fill fields.
    # Don't pretend the trade opened: keep status PENDING and emit a
    # clearly-labelled indicative alert. ``stake_usd`` stays at the requested
    # value so paper-trade analysis can read it directly; OPEN-filtered
    # exposure / PnL math is unaffected because the row is PENDING and
    # ``exchange_status='dry_run'``. Return positive so the caller's
    # in-process dedup blocks repeat firings on the same market today.
    is_dry_run = trade.exchange_status == "dry_run"

    if is_dry_run:
        indicative_price = trade.entry_price or effective_price
        trade.status = TradeStatus.PENDING
        indicative_stake = trade.stake_usd
    else:
        # FAK orders may fill partially or not at all when liquidity is thin.
        # ``_update_fill_details`` already replaced trade.stake_usd with the
        # actual filled cost (zeroed when nothing matched). Use that value as
        # the source of truth for exposure / dedup so we don't book an order
        # that never landed.
        actual_stake = trade.stake_usd or 0.0
        if actual_stake <= 0:
            trade.status = TradeStatus.PENDING
            logger.info(
                "[%s] LOCK %s %s: order posted but no fill (book empty at limit); "
                "leaving open for next-tick retry",
                icao, decision.side, market.id[:12],
            )
            return 0.0
        trade.status = TradeStatus.OPEN
        indicative_stake = actual_stake
        indicative_price = trade.fill_price or effective_price

    unit = _market_unit(market)
    rng = market_range_f(market)
    if rng is not None and rng[0] != rng[1]:
        threshold_disp = (
            f"[{_display_bucket(rng[0], unit)}-{_display_bucket(rng[1], unit)}]"
        )
        op_symbol = "∈"
    elif market.parsed_threshold is not None:
        threshold_disp = str(_display_bucket(int(market.parsed_threshold), unit))
        op_symbol = {
            "above": "≥", "at_least": "≥",
            "below": "<", "at_most": "≤",
            "exactly": "=",
        }.get(market.parsed_operator, "?")
    else:
        threshold_disp = "?"
        op_symbol = "?"
    header = (
        "\U0001f512 *LOCK trade (dry-run)*" if is_dry_run
        else "\U0001f512 *LOCK trade*"
    )
    fill_label = "Indicative" if is_dry_run else "Filled"
    await alerter._enqueue(
        f"{header} [{icao}] {decision.side}\n"
        f"Threshold: {op_symbol}{threshold_disp}{unit} | Margin: {decision.margin_f:+.1f}°F\n"
        f"{fill_label}: ${indicative_stake:.2f} (req ${stake:.2f}) @ {indicative_price:.3f}\n"
        f"Reason: {decision.reasons[0] if decision.reasons else 'locked'}\n"
        f"Market: {market.question[:60]}",
    )
    return indicative_stake


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


async def job_daily_settlement() -> None:
    """Daily 22:00 UTC — bankroll/drawdown bookkeeping, daily digest.

    Trade resolution moved to ``job_resolve_trades`` (5-min interval) so
    markets settle within minutes of their actual ``end_date`` rather than
    sitting in OPEN until the next 22:00 UTC tick. Per-station cache
    rollover moved to ``_maybe_clear_per_station_caches``.
    """
    alerter = get_alerter()
    try:
        monitor = await _get_drawdown_monitor()

        async with async_session() as session:
            # 1. Update bankroll & drawdown.
            # ``get_current_bankroll`` already includes unredeemed WON pnl,
            # so we don't add it twice here — just record the current equity.
            new_bankroll = await get_current_bankroll(session)
            dd_state = await monitor.update(new_bankroll, session)

            if dd_state.level in (DrawdownLevel.CAUTION, DrawdownLevel.PAUSED):
                await alerter.send_drawdown_warning(dd_state)

            # 2. Daily summary
            await alerter.send_daily_summary(session)

            # 3. WX observation retention cleanup
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

            # 4. Station bias recording
            try:
                from src.ingestion.station_bias import record_daily_outcome
                from src.ingestion.aviation import get_routine_daily_max
                from src.ingestion.openmeteo import fetch_deterministic_forecast
                from src.signals.mapper import icao_for_location, geocode, CITY_ICAO

                # Only markets that resolved in the last ~36h. Settlement runs
                # at 22:00 UTC, markets close at 12:00 UTC the same day → 10h
                # old; 36h covers yesterday's run being missed. Anchoring to a
                # specific market (instead of "any market with this ICAO")
                # lets us pass end_date into the observation window so eastern
                # TZs stop recording the pre-dawn hours of the next local day.
                now_utc = datetime.now(timezone.utc)
                recent_cutoff = now_utc - timedelta(hours=36)
                seen_icaos: set[str] = set()
                stmt_markets = (
                    select(Market)
                    .where(
                        Market.parsed_location.isnot(None),
                        Market.parsed_variable == "temperature",
                        Market.end_date.isnot(None),
                        Market.end_date <= now_utc,
                        Market.end_date > recent_cutoff,
                    )
                    .order_by(Market.end_date.desc())
                )
                market_result = await session.execute(stmt_markets)
                for mkt in market_result.scalars():
                    icao = icao_for_location(mkt.parsed_location) if mkt.parsed_location else None
                    if not icao or icao in seen_icaos:
                        continue
                    seen_icaos.add(icao)

                    max_f, count = await get_routine_daily_max(
                        icao, reference_utc=mkt.end_date,
                    )
                    if max_f is None or count < 3:
                        continue

                    coords = geocode(mkt.parsed_location) if mkt.parsed_location else None
                    if not coords:
                        continue

                    # Bias is anchored to the deterministic single-source
                    # peak — not the aggregated ensemble — so the recorded
                    # offset stays stable when the ensemble model list or
                    # aggregation function changes (e.g. swapping mean→median,
                    # adding/removing a model). The pipeline still trades on
                    # the bias-corrected ensemble peak; this just keeps the
                    # correction reference frame fixed.
                    forecast = await fetch_deterministic_forecast(coords[0], coords[1])
                    if forecast is None:
                        continue

                    max_c = (max_f - 32.0) * 5.0 / 9.0
                    await record_daily_outcome(
                        session, icao,
                        mkt.end_date,
                        max_c, forecast.peak_temp_c,
                    )
                logger.info("Station bias recorded for %d stations", len(seen_icaos))
            except Exception:
                logger.exception("Station bias recording failed (non-fatal)")

            # (Per-station fast-lock-poll dedup is now reset at each
            # station's local-day rollover by ``_maybe_clear_per_station_caches``;
            # no global wipe at 22:00 UTC.)

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
# Fast lock poll — between-tick latency fix
# ---------------------------------------------------------------------------


def _minimal_state_for_easy_lock(
    icao: str,
    routine_points: list[tuple[datetime, float]],
):
    """Build a WeatherState with only routine history — sufficient for the
    EASY lock direction (observed max already clears threshold), which doesn't
    read forecast/solar/trend fields. HARD-direction locks still need the
    main pipeline's forecast context and run there.
    """
    from src.signals.state_aggregator import WeatherState

    return WeatherState(
        station_icao=icao,
        current_max_f=max(t for _, t in routine_points),
        metar_trend_rate=0.0,
        dewpoint_trend_rate=0.0,
        forecast_peak_f=0.0,
        hours_until_peak=0.0,
        solar_declining=False,
        solar_decline_magnitude=0.0,
        cloud_rising=False,
        cloud_rise_magnitude=0.0,
        routine_count_today=len(routine_points),
        has_forecast=False,
        routine_history=tuple(sorted(routine_points, key=lambda p: p[0])),
    )


async def _fast_poll_projection_check(
    *,
    icao: str,
    station_metars: list[dict],
) -> None:
    """Run forecast-exceedance projection from fast-poll using cached inputs.

    Closes the 5-minute cadence gap on projection alerts: when a new routine
    METAR lands at HH:51, the fast-poll loop catches it within 30s instead
    of waiting for the next unified tick. No extra HTTP — reuses the
    forecast / bias / climate-normal snapshot from the last unified tick.

    No-op when no cache exists (unified pipeline hasn't run for this station
    yet, or the cache aged past the 30-min window).
    """
    from src.signals.state_aggregator import (
        build_state_from_metars,
        get_cached_aggregation_inputs,
    )
    from src.signals.forecast_exceedance import check_and_record_daily_max_alert

    cached = get_cached_aggregation_inputs(icao)
    if cached is None:
        return

    seen_at: set[datetime] = set()
    for m in cached.history:
        obs = m.get("observed_at")
        if isinstance(obs, datetime):
            seen_at.add(obs)
    merged = list(cached.history)
    for m in station_metars:
        obs = m.get("observed_at")
        if isinstance(obs, datetime) and obs not in seen_at:
            merged.append(m)
            seen_at.add(obs)

    new_state = build_state_from_metars(
        icao, merged, cached.forecast, cached.bias_c,
        datetime.now(timezone.utc),
        climate_prior_mean_f=cached.climate_prior_mean_f,
        climate_prior_std_f=cached.climate_prior_std_f,
    )
    if new_state is None:
        return

    try:
        await check_and_record_daily_max_alert(
            icao, new_state, merged, cached.forecast,
        )
    except Exception:
        logger.warning(
            "[fast-poll %s] projection check failed", icao, exc_info=True,
        )


async def job_fast_lock_poll() -> None:
    """Between-tick lock-rule check. Catches new routine METARs seconds after
    publication so the order lands before Polymarket market-makers react.

    Only the EASY lock direction fires here — observed daily max already
    clears threshold + LOCK_MARGIN_F. HARD direction (below-threshold,
    no-more-heating) needs forecast/solar context and stays on the main
    5-min pipeline.

    Scope: one bulk `fetch_latest_metars` per tick (AWC accepts up to 20
    stations per request), CLOB price/depth only when a lock actually fires.
    Dedup via `_locked_markets_fired_today`; cleared at daily settlement.
    """
    if not settings.LOCK_RULE_ENABLED or not settings.FAST_LOCK_POLL_ENABLED:
        return

    alerter = get_alerter()
    try:
        from src.ingestion.aviation import fetch_latest_metars
        from src.ingestion.polymarket import get_active_weather_markets
        from src.risk.circuit_breakers import check_circuit_breakers
        from src.signals.mapper import icao_for_location
        from src.execution.polymarket_client import (
            get_best_bid_ask,
            get_orderbook_depth,
            get_token_ids,
        )

        async with async_session() as session:
            cb = await check_circuit_breakers(session)
            if not cb.can_trade:
                return

            markets = await get_active_weather_markets(session)
            if not markets:
                return

            # Group active binary markets by ICAO. Skip markets the fast loop
            # already fired on and non-binary brackets (lock rule scope).
            by_icao: dict[str, list] = {}
            for m in markets:
                if m.id in _locked_markets_fired_today:
                    continue
                if not _is_binary_market(m):
                    continue
                if not m.parsed_location:
                    continue
                icao = icao_for_location(m.parsed_location)
                if not icao:
                    continue
                by_icao.setdefault(icao, []).append(m)

            if not by_icao:
                return

            try:
                latest = await fetch_latest_metars(list(by_icao.keys()), session)
            except Exception:
                logger.warning("fast-poll: fetch_latest_metars failed", exc_info=True)
                return

            metars_by_icao: dict[str, list] = {}
            for row in latest:
                icao = row.get("station_icao")
                if icao:
                    metars_by_icao.setdefault(icao, []).append(row)

            monitor = await _get_drawdown_monitor()
            bankroll = await get_current_bankroll(session)
            exposure = await get_current_exposure(session)
            now_utc = datetime.now(timezone.utc)
            epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)

            for icao, city_markets in by_icao.items():
                station_metars = metars_by_icao.get(icao, [])
                if not station_metars:
                    continue

                last_seen = _last_routine_seen.get(icao, epoch)
                new_routine = [
                    m for m in station_metars
                    if not m.get("is_speci")
                    and isinstance(m.get("observed_at"), datetime)
                    and m["observed_at"] > last_seen
                    and m.get("temp_f") is not None
                ]
                if not new_routine:
                    continue

                _last_routine_seen[icao] = max(m["observed_at"] for m in new_routine)

                routine_points = [
                    (m["observed_at"], float(m["temp_f"]))
                    for m in station_metars
                    if not m.get("is_speci")
                    and isinstance(m.get("observed_at"), datetime)
                    and m.get("temp_f") is not None
                ]
                if not routine_points:
                    continue

                state = _minimal_state_for_easy_lock(icao, routine_points)
                latest_obs = max(new_routine, key=lambda m: m["observed_at"])
                logger.info(
                    "[fast-poll %s] new routine METAR obs=%s temp=%.1f°F max=%.1f°F | %d market(s)",
                    icao, latest_obs["observed_at"].isoformat(),
                    latest_obs["temp_f"], state.current_max_f, len(city_markets),
                )

                # Re-run forecast-exceedance projection with the fresh METAR.
                # Reuses cached forecast/bias/normals from the last unified
                # tick — same call signature as the unified path, so the DB
                # row + Telegram cooldown logic dedupe correctly when both
                # fast-poll and unified see the same observation.
                await _fast_poll_projection_check(
                    icao=icao,
                    station_metars=station_metars,
                )

                for market in city_markets:
                    if market.id in _locked_markets_fired_today:
                        continue

                    decision = evaluate_lock(state, market)
                    if decision.side is None or decision.direction is None:
                        continue

                    token_ids = await get_token_ids(market.id)
                    if not token_ids:
                        logger.info(
                            "[fast-poll %s] lock %s on %s but no token IDs",
                            icao, decision.side, market.id[:12],
                        )
                        continue

                    quote = get_best_bid_ask(token_ids[0])
                    if quote is None:
                        logger.info(
                            "[fast-poll %s] lock %s on %s but no price",
                            icao, decision.side, market.id[:12],
                        )
                        continue
                    yes_bid, yes_ask = quote
                    yes_price = (yes_bid + yes_ask) / 2

                    yes_depth = get_orderbook_depth(token_ids[0], yes_price) if yes_price > 0 else 0.0
                    end_time = market.end_date or now_utc + timedelta(hours=24)

                    stake = await _try_lock_rule_trade(
                        session=session, market=market, state=state,
                        yes_price=yes_price, token_ids=token_ids,
                        yes_depth=yes_depth, end_time=end_time,
                        bankroll=bankroll, exposure=exposure,
                        monitor=monitor, alerter=alerter, icao=icao,
                        yes_bid=yes_bid, yes_ask=yes_ask,
                    )
                    if stake is None:
                        continue
                    if stake > 0:
                        # Real fill — dedup so we don't keep adding to the
                        # same market on every poll. A second fresh entry
                        # later in the day would require a new lock signal.
                        _locked_markets_fired_today.add(market.id)
                        _market_to_icao[market.id] = icao
                        exposure += stake
                    # If stake == 0 the FAK didn't fill (book empty at the
                    # limit) or the filter rejected. Don't dedup — fresh
                    # liquidity may appear on the next METAR tick and we
                    # want to take it.

            await session.commit()

    except Exception as exc:
        logger.exception("Fast lock poll failed")
        try:
            await alerter.send_system_error(exc, "fast lock poll")
        except Exception:
            logger.warning("Alerter failed to send fast-poll error")


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
        job_resolve_trades,
        IntervalTrigger(minutes=5),
        id="resolve_trades",
        next_run_time=datetime.now(timezone.utc) + timedelta(seconds=30),
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        job_unified_pipeline,
        IntervalTrigger(minutes=settings.UNIFIED_PIPELINE_INTERVAL_MINUTES),
        id="unified_pipeline",
        max_instances=1,
        coalesce=True,
    )
    logger.info(
        "Unified pipeline enabled (every %dm)",
        settings.UNIFIED_PIPELINE_INTERVAL_MINUTES,
    )

    if settings.LOCK_RULE_ENABLED and settings.FAST_LOCK_POLL_ENABLED:
        scheduler.add_job(
            job_fast_lock_poll,
            IntervalTrigger(seconds=settings.FAST_LOCK_POLL_INTERVAL_SECONDS),
            id="fast_lock_poll",
            max_instances=1,
            coalesce=True,
        )
        logger.info(
            "Fast lock poll enabled (every %ds)",
            settings.FAST_LOCK_POLL_INTERVAL_SECONDS,
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
