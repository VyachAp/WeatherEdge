"""APScheduler-based pipeline orchestration and health-check server."""

from __future__ import annotations

import asyncio
import functools
import logging
import signal
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import ProgrammingError

from src.config import settings
from src.db.engine import async_session, engine
from src.db.models import BotState, EvaluationLog, Market, Signal, Trade, TradeStatus
from src.execution.alerter import get_alerter
from src.execution.binary_market import (
    display_bucket as _display_bucket,
    is_binary_market as _is_binary_market,
    is_bracket_like as _is_bracket_like,
    make_binary_buckets as _make_binary_buckets,
    market_range_f,
    market_unit as _market_unit,
    near_peak_floor_eligible as _near_peak_floor_eligible,
    should_skip_future_day as _should_skip_future_day,
)
from src.execution.lock_rule_executor import (
    extract_bracket_buckets as _extract_bracket_buckets,
    extract_market_prices as _extract_market_prices,
    minimal_state_for_easy_lock as _minimal_state_for_easy_lock,
    try_lock_rule_trade as _try_lock_rule_trade,
)
from src.ingestion.polymarket import scan_and_ingest
from src.monitoring.health import start_health_server as _start_health_server
from src.monitoring.logging import configure_logging
from src.persistence import cache_rollover
from src.persistence.cache_rollover import (
    last_routine_seen as _last_routine_seen,
    local_day_seen as _local_day_seen,
    locked_markets_fired_today as _locked_markets_fired_today,
    market_to_icao as _market_to_icao,
    maybe_clear_per_station_caches as _maybe_clear_per_station_caches,
    unified_fired_today as _unified_fired_today,
)
from src.persistence.dedup import (
    has_active_trade as _has_active_trade,
    upsert_signal as _upsert_signal,
)
from src.resolution import (
    get_current_bankroll,
    get_current_exposure,
    resolve_trades,
)
from src.risk.cluster_cap import cluster_stake_used as _cluster_stake_used
from src.risk.drawdown import DrawdownLevel, DrawdownMonitor
from src.risk.kelly import MAX_EXPOSURE_PCT, size_locked_position, size_position
from src.signals.calibration import get_calibration_coefficients
from src.signals.edge_calculator import binary_market_edge as _binary_market_edge
from src.signals.decision_log import (
    OUTCOME_CLUSTER_CAP,
    OUTCOME_DRAWDOWN_PAUSED,
    OUTCOME_DUP_DB,
    OUTCOME_DUP_INPROC,
    OUTCOME_INSUFFICIENT_BALANCE,
    OUTCOME_NO_FILL,
    OUTCOME_ORDER_FAILED,
    OUTCOME_STAKE_BELOW_MIN,
    OUTCOME_TRADE_FILLED,
    OUTCOME_TRADE_PENDING,
    log_decision,
)
from src.signals.evaluation_log import log_evaluation as _log_evaluation
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
# Last time the monitor's persisted peak was reloaded from bankroll_log.
# Reloaded on a TTL so `admin reset-drawdown-peak` takes effect on the running
# process without a restart (the peak was previously loaded once at boot only).
_drawdown_peak_loaded_at: datetime | None = None
_DRAWDOWN_PEAK_RELOAD_TTL = timedelta(seconds=300)
_shutdown_event: asyncio.Event | None = None

# A note on the ``_X`` aliases in the import block above.
#
# Many of those imports use ``from X import name as _name``. The
# underscore prefix is **historical**: these symbols were once defined
# locally in this file as private helpers (with underscore prefix), then
# refactored out into focused submodules (``execution.binary_market``,
# ``execution.lock_rule_executor``, ``persistence.cache_rollover``,
# ``persistence.dedup``, ``signals.edge_calculator``,
# ``signals.evaluation_log``, ``risk.cluster_cap``, ``monitoring.health``).
# The aliases preserve the in-file callsites that already read ``_name``
# AND keep the small set of tests that import via the underscore form
# working (see ``tests/test_scheduler.py``: ``_binary_market_edge``,
# ``_has_active_trade``, ``_log_evaluation``, ``_minimal_state_for_easy_lock``,
# ``_should_skip_future_day``, ``_upsert_signal``).
#
# These are NOT private to the scheduler — they're public functions of
# their canonical home modules. New code should prefer importing from
# the canonical home (``from src.signals.edge_calculator import
# binary_market_edge`` etc.) rather than via the scheduler's alias.
# The aliases are not the scheduler's public API surface; only
# ``run_scheduler``, ``configure_logging``, ``backfill_markets``, and
# ``start_health_server`` are.
#
# ``_load_stuck_alert_cooldown`` / ``_persist_stuck_alert_cooldown`` are
# different — those are genuinely scheduler-local helpers (defined below
# in this file) that tests import for unit coverage of the reconcile
# heartbeat. The underscore there means "private to scheduler", and that
# meaning is correct.

# ---------------------------------------------------------------------------
# Health server — module-local thin wrapper that closes over `_scheduler`
# ---------------------------------------------------------------------------


async def start_health_server(port: int = 8080) -> asyncio.Server:
    """Backcompat wrapper. Delegates to ``monitoring.health`` but supplies
    the scheduler-running callable so the extracted module stays free of
    a circular import on this file."""
    return await _start_health_server(
        scheduler_running=lambda: _scheduler is not None and _scheduler.running,
        port=port,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get_drawdown_monitor() -> DrawdownMonitor:
    global _drawdown_monitor, _drawdown_peak_loaded_at  # noqa: PLW0603
    now = datetime.now(timezone.utc)
    if _drawdown_monitor is None:
        _drawdown_monitor = DrawdownMonitor(settings.INITIAL_BANKROLL)
        async with async_session() as session:
            await _drawdown_monitor.load_state(session)
        _drawdown_peak_loaded_at = now
    elif (
        _drawdown_peak_loaded_at is None
        or now - _drawdown_peak_loaded_at > _DRAWDOWN_PEAK_RELOAD_TTL
    ):
        # Re-read the persisted peak so an operator's `admin
        # reset-drawdown-peak` (a newer bankroll_log row with peak=equity)
        # takes effect within the TTL — no scheduler restart needed. Safe to
        # pull in a *lower* peak: ``check()`` re-maxes with live equity, so a
        # reload can only relax an over-tight pause, never under-protect.
        async with async_session() as session:
            await _drawdown_monitor.load_state(session)
        _drawdown_peak_loaded_at = now
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

            # Refresh calibration once per tick when enabled.
            # The fit is cached for `_CACHE_TTL_SEC` (30 min) so back-to-back
            # ticks don't re-query; `apply_calibration` reads sync from the
            # cache inside `_binary_market_edge`.
            if settings.APPLY_CALIBRATION:
                try:
                    from src.signals.calibration import refresh_calibration
                    await refresh_calibration(session)
                except Exception:
                    logger.warning(
                        "calibration refresh failed; falling back to uncalibrated",
                        exc_info=True,
                    )

            for icao, state in city_states.items():
              try:
                for market in city_markets[icao]:
                    # Bracket-class markets gated off by default — see
                    # ``settings.BRACKET_MARKETS_ENABLED``. Bracket bets
                    # were the dominant live-PnL bleed pre-2026-05-08.
                    if (
                        not settings.BRACKET_MARKETS_ENABLED
                        and market.parsed_operator in ("bracket", "range")
                    ):
                        logger.debug(
                            "[%s] skip %s: bracket markets disabled (BRACKET_MARKETS_ENABLED=False)",
                            icao, market.id[:12],
                        )
                        continue

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
                        forecast_peak_f=state.forecast_peak_f,
                        current_max_f=state.current_max_f,
                        hours_until_peak=state.hours_until_peak,
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

                    # Redesigned decision flow (information-latency thesis):
                    # run the pure four-stage shadow model for this market and
                    # record it to the unselected shadow_ledger. Places NO
                    # orders and never mutates the Market row — it runs purely
                    # alongside the live probability + lock paths so the
                    # calibration-report can prove (or refute) it OOS before any
                    # capital is routed to it. Depth_no reuses the YES-side
                    # depth as a proxy to avoid an extra CLOB call per market
                    # (slippage rarely binds at shadow stakes). Best-effort.
                    try:
                        from src.signals.shadow_ledger import record_shadow_decision
                        await record_shadow_decision(
                            session, market, state,
                            yes_bid=yes_bid, yes_ask=yes_ask, yes_mid=price,
                            depth_yes_usd=mkt_depth, depth_no_usd=mkt_depth,
                            minutes_to_close=eval_minutes_to_close,
                            bankroll=bankroll,
                        )
                    except Exception:  # noqa: BLE001 — shadow must not break live
                        logger.debug("shadow ledger hook failed", exc_info=True)

                    # M1/Phase-1: shadow-log the pooled-vs-per-class
                    # calibration counterfactual for this raw prob (pure
                    # telemetry — can't affect the trade). Lets shadow-report
                    # validate the PER_OPERATOR_CALIBRATION_ENABLED flip
                    # before it touches live trading. Best-effort.
                    eval_shadow = None
                    try:
                        from src.signals.calibration import shadow_calibration
                        from src.signals.probability_engine import shadow_sigma_fields
                        from src.signals.edge_calculator import shadow_valley_fields
                        from src.execution.binary_market import operator_class
                        shadow_parts: dict = {}
                        cal_shadow = shadow_calibration(
                            edge_result.raw_probability, operator_class(market),
                        )
                        if cal_shadow is not None:
                            shadow_parts["cal"] = cal_shadow
                        sig_shadow = shadow_sigma_fields(state)
                        if sig_shadow is not None:
                            shadow_parts["sigma"] = sig_shadow
                        # P2 price-band refinement counterfactual — banded on
                        # the chosen side's BUY cost (== BucketEdge.market_price).
                        val_shadow = shadow_valley_fields(
                            edge_result.market_price, edge_result.edge,
                        )
                        if val_shadow is not None:
                            shadow_parts["valley"] = val_shadow
                        eval_shadow = shadow_parts or None
                    except Exception:
                        eval_shadow = None
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
                        raw_model_prob=edge_result.raw_probability,
                        calibrated=edge_result.calibrated,
                        forecast_peak_f=state.forecast_peak_f,
                        current_max_f=state.current_max_f,
                        hours_until_peak=state.hours_until_peak,
                        forecast_sigma_f=state.forecast_sigma_f,
                        shadow=eval_shadow,
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
                        # Near-peak floor-up: threshold-class, high-confidence
                        # OR near-peak passers get their sub-min stake floored
                        # up instead of dropped (gated off by default).
                        floor_to_usd = (
                            settings.NEAR_PEAK_FLOOR_STAKE_USD
                            if _near_peak_floor_eligible(
                                market,
                                our_probability=edge.our_probability,
                                hours_until_peak=state.hours_until_peak,
                            )
                            else None
                        )
                        pos = size_position(
                            bankroll=bankroll,
                            model_prob=edge.our_probability,
                            market_prob=edge.market_price,
                            current_exposure=exposure,
                            max_position_usd=settings.MAX_POSITION_USD,
                            orderbook_depth=side_depth or None,
                            floor_to_usd=floor_to_usd,
                        )
                        floored_up = (pos.reason or "").startswith("floored")

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
                            # Split outcome by *cause*: PAUSED drawdown
                            # produces multiplier=0 → adjusted=0; Kelly +
                            # caps producing a tiny stake is a different
                            # signal. Separate outcomes let dashboards
                            # attribute zero-throughput days correctly.
                            paused = dd_state.size_multiplier == 0.0
                            await log_decision(
                                session,
                                market_id=market.id,
                                direction=edge.direction,
                                signal_kind="probability",
                                outcome=OUTCOME_DRAWDOWN_PAUSED if paused else OUTCOME_STAKE_BELOW_MIN,
                                requested_stake_usd=pos.stake_usd,
                                actual_stake_usd=adjusted_stake,
                                dd_multiplier=dd_state.size_multiplier,
                                dd_level=dd_state.level.value,
                                metadata={
                                    "bucket": edge.bucket_value,
                                    "edge": edge.edge,
                                    "side": side_label,
                                    # Which cap actually zeroed the stake —
                                    # per-trade / exposure / USD / depth /
                                    # MIN_TRADE_USD floor. Without this the
                                    # decision_log only tells us the trade
                                    # was killed, not by which constraint.
                                    "size_reason": pos.reason,
                                    "kelly_pct": pos.kelly_pct,
                                    "depth_usd": side_depth,
                                },
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
                            await log_decision(
                                session,
                                market_id=market.id,
                                direction=edge.direction,
                                signal_kind="probability",
                                outcome=OUTCOME_DUP_INPROC,
                                requested_stake_usd=adjusted_stake,
                                metadata={"bucket": edge.bucket_value, "side": side_label},
                            )
                            continue
                        if await _has_active_trade(session, market.id, edge.direction):
                            logger.info(
                                "[%s] skip %s bucket=%d: active trade exists for this market+side",
                                icao, side_label, edge.bucket_value,
                            )
                            _unified_fired_today.add(dedup_key)
                            _market_to_icao[market.id] = icao
                            await log_decision(
                                session,
                                market_id=market.id,
                                direction=edge.direction,
                                signal_kind="probability",
                                outcome=OUTCOME_DUP_DB,
                                requested_stake_usd=adjusted_stake,
                                metadata={"bucket": edge.bucket_value, "side": side_label},
                            )
                            continue

                        # Cluster cap: cap total stake across all
                        # bucket markets in the same parsed_location +
                        # target-day cluster. Outcomes are anti-correlated
                        # (only one bucket wins) so independent Kelly
                        # sizing per bucket over-allocates. Threshold
                        # markets are unaffected (helper returns 0.0).
                        if settings.CLUSTER_STAKE_CAP_USD > 0:
                            cluster_used = await _cluster_stake_used(session, market)
                            if cluster_used + adjusted_stake > settings.CLUSTER_STAKE_CAP_USD:
                                logger.info(
                                    "[%s] skip %s bucket=%d: cluster stake $%.2f + new $%.2f > cap $%.2f",
                                    icao, side_label, edge.bucket_value,
                                    cluster_used, adjusted_stake,
                                    settings.CLUSTER_STAKE_CAP_USD,
                                )
                                await log_decision(
                                    session,
                                    market_id=market.id,
                                    direction=edge.direction,
                                    signal_kind="probability",
                                    outcome=OUTCOME_CLUSTER_CAP,
                                    requested_stake_usd=adjusted_stake,
                                    metadata={
                                        "bucket": edge.bucket_value,
                                        "cluster_used_usd": cluster_used,
                                        "cap_usd": settings.CLUSTER_STAKE_CAP_USD,
                                    },
                                )
                                continue

                        # Side-effective probability lands in Signal.model_prob
                        # so calibration treats YES and NO uniformly.
                        # See src/signals/calibration.py. The (market, direction)
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
                            raw_model_prob=edge.raw_probability,
                            calibrated=edge.calibrated,
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
                            await log_decision(
                                session,
                                market_id=market.id,
                                direction=edge.direction,
                                signal_kind="probability",
                                outcome=OUTCOME_TRADE_PENDING,
                                requested_stake_usd=adjusted_stake,
                                actual_stake_usd=trade.stake_usd,
                                dd_multiplier=dd_state.size_multiplier,
                                dd_level=dd_state.level.value,
                                metadata={
                                    "bucket": edge.bucket_value,
                                    "edge": edge.edge,
                                    "is_dry_run": True,
                                    "floored_up": floored_up,
                                },
                            )
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
                            await log_decision(
                                session,
                                market_id=market.id,
                                direction=edge.direction,
                                signal_kind="probability",
                                outcome=OUTCOME_TRADE_FILLED,
                                requested_stake_usd=adjusted_stake,
                                actual_stake_usd=actual_stake,
                                dd_multiplier=dd_state.size_multiplier,
                                dd_level=dd_state.level.value,
                                metadata={
                                    "bucket": edge.bucket_value,
                                    "edge": edge.edge,
                                    "fill_price": fill,
                                    "floored_up": floored_up,
                                },
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
                            await log_decision(
                                session,
                                market_id=market.id,
                                direction=edge.direction,
                                signal_kind="probability",
                                outcome=OUTCOME_NO_FILL,
                                requested_stake_usd=adjusted_stake,
                                metadata={"bucket": edge.bucket_value, "edge": edge.edge},
                            )
                        else:
                            # place_order returned False — the FAK never
                            # posted (no token IDs, no client, raised, or
                            # wallet pre-flight tripped). Trade row stays
                            # PENDING. Distinguish `insufficient_balance` —
                            # an intentional pre-flight skip — from real
                            # `order_failed` so dashboards can separate
                            # "we held back" from "we tried and lost".
                            failed_outcome = (
                                OUTCOME_INSUFFICIENT_BALANCE
                                if trade.exchange_status == "insufficient_balance"
                                else OUTCOME_ORDER_FAILED
                            )
                            await log_decision(
                                session,
                                market_id=market.id,
                                direction=edge.direction,
                                signal_kind="probability",
                                outcome=failed_outcome,
                                requested_stake_usd=adjusted_stake,
                                metadata={
                                    "bucket": edge.bucket_value,
                                    "edge": edge.edge,
                                    "exchange_status": trade.exchange_status,
                                },
                            )

              except Exception:
                skipped_cities += 1
                logger.warning("Unified pipeline: error evaluating %s, skipping", icao, exc_info=True)
                continue

            # M4: per-tick capital/headroom snapshot (best-effort telemetry).
            # Uses the top-of-tick equity/exposure the sizer actually saw.
            try:
                from src.signals.exposure_snapshot import record_exposure_snapshot
                await record_exposure_snapshot(
                    session, equity=bankroll, exposure=exposure,
                )
            except Exception:
                logger.warning("exposure snapshot failed (non-fatal)", exc_info=True)

            await session.commit()
            if skipped_cities:
                logger.warning("Unified pipeline: %d/%d cities skipped due to errors", skipped_cities, len(city_markets))
            logger.info("Unified pipeline complete: %d trades placed", total_trades)

    except Exception as exc:
        logger.exception("Unified pipeline failed")
        await alerter.send_system_error(exc, "unified pipeline")


# Cooldown key + window for the calibration-squash diagnostic (section 7 of
# job_daily_settlement). Persisted in bot_state so a restart doesn't re-alert.
_CALIB_SQUASH_ALERT_KEY = "settlement.calib_squash_last_pushed_at"
_CALIB_SQUASH_COOLDOWN_HOURS = 48
# Minimum squashed-eval count before the diagnostic fires (avoid noise on a
# quiet day that genuinely had no edged candidates).
_CALIB_SQUASH_MIN_COUNT = 20


def _detect_calibration_squash(
    eval_rows: list[dict], fill_count: int, *, min_squashed: int = _CALIB_SQUASH_MIN_COUNT,
) -> dict | None:
    """Diagnose a calibration-induced trade halt. Pure (no I/O).

    Returns ``{n_squashed, avg_raw, avg_cal}`` when the day saw ~no fills AND a
    meaningful set of edged evals were rejected on the probability floor
    *because calibration pulled an otherwise-passing raw prob below it* — i.e.
    the 2026-06-08 "bot correctly sitting out an overconfident regime" signature
    (raw≈0.92 → cal≈0.72 < 0.75 floor). Returns ``None`` when there were fills,
    too few squashed evals, or the data is incomplete.

    ``eval_rows`` items carry ``reject_reason`` (``"probability X < FLOOR"``),
    ``raw_model_prob`` (pre-calibration), ``model_prob`` (post-calibration). A
    row counts as squashed when calibration moved it from clearing the floor
    (raw ≥ FLOOR) to below it (cal < FLOOR).
    """
    if fill_count > 0:
        return None
    squashed: list[tuple[float, float]] = []
    for r in eval_rows:
        rr = r.get("reject_reason") or ""
        if not rr.startswith("probability"):
            continue
        raw, cal = r.get("raw_model_prob"), r.get("model_prob")
        if raw is None or cal is None:
            continue
        try:
            floor = float(rr.rsplit("<", 1)[1].strip())
        except (IndexError, ValueError):
            continue
        if raw >= floor > cal:
            squashed.append((float(raw), float(cal)))
    if len(squashed) < min_squashed:
        return None
    n = len(squashed)
    return {
        "n_squashed": n,
        "avg_raw": sum(s[0] for s in squashed) / n,
        "avg_cal": sum(s[1] for s in squashed) / n,
    }


async def _load_alert_cooldown(
    session: "AsyncSessionType", key: str,
) -> datetime | None:
    """Generic bot_state cooldown reader (``{"at_iso": ...}``). Fail-open on a
    missing bot_state table (mirrors ``_load_stuck_alert_cooldown``)."""
    try:
        row = await session.get(BotState, key)
    except ProgrammingError:
        await session.rollback()
        logger.warning("bot_state table missing — %s cooldown not persisted.", key)
        return None
    if row is None or not isinstance(row.value, dict):
        return None
    iso = row.value.get("at_iso")
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso)
    except ValueError:
        return None


async def _persist_alert_cooldown(
    session: "AsyncSessionType", key: str, at: datetime,
) -> None:
    """Generic bot_state cooldown writer (idempotent upsert)."""
    try:
        stmt = (
            pg_insert(BotState)
            .values(key=key, value={"at_iso": at.isoformat()}, updated_at=at)
            .on_conflict_do_update(
                index_elements=["key"],
                set_={"value": {"at_iso": at.isoformat()}, "updated_at": at},
            )
        )
        await session.execute(stmt)
    except ProgrammingError:
        await session.rollback()
        logger.warning("bot_state table missing — %s cooldown not persisted.", key)


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

            # 2b. Nudge to run `bet redeem` when WON trades sit on-chain.
            # Skipped in pure dry-run wallets — there's nothing to redeem.
            if settings.POLYMARKET_PRIVATE_KEY:
                await alerter.send_redeem_nudge(session)

            # 3. Station bias recording (WX pipeline + its cleanup were
            # retired 2026-05-30 — see docs/graveyard.md)
            try:
                from src.ingestion.station_bias import record_daily_outcome
                from src.ingestion.aviation import get_routine_daily_max
                from src.ingestion.openmeteo import fetch_deterministic_forecast
                from src.signals.mapper import (
                    icao_for_location, geocode, CITY_ICAO, resolve_target_local_day,
                )
                from src.signals.market_resolution import backfill_routine_max

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

                    # M3: backfill the routine daily max + recomputed
                    # divergence onto any market_resolution rows the
                    # resolver seeded for this station-day. Best-effort,
                    # runs before the bias path's coords/forecast gates so a
                    # missing forecast can't skip the divergence fill.
                    try:
                        from src.signals.mapper import icao_timezone
                        target_local = resolve_target_local_day(
                            mkt.end_date, icao_timezone(icao)
                        )
                        if target_local is not None:
                            await backfill_routine_max(
                                session,
                                station_icao=icao,
                                target_date_local=target_local,
                                routine_metar_max_f=max_f,
                            )
                    except Exception:
                        logger.warning(
                            "market_resolution backfill failed for %s (non-fatal)",
                            icao, exc_info=True,
                        )

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

            # 4b. M3 straggler sweep — backfill routine_metar_max_f for any
            # MarketResolution row the ~36h live-fetch loop above missed
            # (settlement outage, data-starved day, or aged-out live history),
            # reading the daily max from stored METARs. Self-heals stragglers so
            # they don't stay NULL forever. Best-effort; never blocks settlement.
            try:
                from src.signals.market_resolution import sweep_stranded_routine_max
                filled = await sweep_stranded_routine_max(session)
                if filled:
                    logger.info(
                        "M3 straggler sweep filled %d routine-max rows", filled
                    )
            except Exception:
                logger.warning("M3 straggler sweep failed (non-fatal)", exc_info=True)

            # (Per-station fast-lock-poll dedup is now reset at each
            # station's local-day rollover by ``_maybe_clear_per_station_caches``;
            # no global wipe at 22:00 UTC.)

            # 5. Weekly calibration check (Sundays)
            if datetime.now(timezone.utc).weekday() == 6:
                coeffs = await get_calibration_coefficients(session)
                if coeffs is not None:
                    # Always log the pooled fit; surface per-class fits
                    # only when they were actually fit (≥ MIN_CALIBRATION_SAMPLES
                    # in that class). Helps the weekly review attribute
                    # any drift to a specific operator class.
                    for cls, (slope, intercept) in coeffs.items():
                        logger.info(
                            "Calibration active [%s]: slope=%.4f intercept=%.4f",
                            cls, slope, intercept,
                        )
                else:
                    logger.info("Calibration: insufficient data (<50 resolved signals)")

            # 6. Exposure-cap proximity alert. Catches the silent-silencer
            # failure mode (incident 2026-05-17): stuck OPEN trades pin the
            # ``MAX_EXPOSURE_PCT × bankroll`` cap, ``size_position`` returns
            # $0 for every passing eval, the bot stops trading but logs
            # nothing operationally urgent. The alert fires at 80% of the
            # effective cap (matching the ``MAX_EXPOSURE_USD_FLOOR`` floor
            # when bankroll is small) so the operator sees it BEFORE
            # throughput dies, not after.
            try:
                exposure = await get_current_exposure(session)
                bankroll = await get_current_bankroll(session)
                effective_cap = max(
                    MAX_EXPOSURE_PCT * bankroll,
                    settings.MAX_EXPOSURE_USD_FLOOR,
                )
                if effective_cap > 0 and exposure > 0.80 * effective_cap:
                    await alerter._enqueue(
                        f"⚠ *Exposure approaching cap*\n"
                        f"Current: ${exposure:,.2f}  "
                        f"Cap: ${effective_cap:,.2f}  "
                        f"({100 * exposure / effective_cap:.0f}%)\n"
                        f"New sizes will start collapsing to $0 once cap is hit. "
                        f"Inspect OPEN trades and run `admin reconcile-stuck` "
                        f"if any are stuck delayed."
                    )
            except Exception:
                logger.warning("Exposure-cap alert check failed (non-fatal)", exc_info=True)

            # 7. Calibration-squash / low-throughput diagnostic. When the day
            # saw ~no fills AND a meaningful set of edged evals were rejected on
            # the probability floor *because calibration pulled an otherwise-
            # passing raw prob below it*, surface it — so a "correctly sitting
            # out an overconfident regime" halt (2026-06-08 investigation) is
            # distinguishable from a real bug. Best-effort; bot_state cooldown.
            try:
                day_ago = datetime.now(timezone.utc) - timedelta(hours=24)
                fills = (
                    await session.execute(
                        select(Trade.id).where(
                            Trade.opened_at >= day_ago,
                            Trade.fill_price.isnot(None),
                        )
                    )
                ).scalars().all()
                cand = (
                    await session.execute(
                        select(
                            EvaluationLog.edge,
                            EvaluationLog.raw_model_prob,
                            EvaluationLog.model_prob,
                            EvaluationLog.reject_reason,
                        ).where(
                            EvaluationLog.signal_kind == "probability",
                            EvaluationLog.created_at >= day_ago,
                            EvaluationLog.raw_model_prob.isnot(None),
                            EvaluationLog.model_prob.isnot(None),
                            EvaluationLog.reject_reason.like("probability%"),
                        )
                    )
                ).all()
                eval_rows = [
                    {"edge": e, "raw_model_prob": rp,
                     "model_prob": mp, "reject_reason": rr}
                    for (e, rp, mp, rr) in cand
                ]
                squash = _detect_calibration_squash(eval_rows, len(fills))
                if squash is not None:
                    now = datetime.now(timezone.utc)
                    last_at = await _load_alert_cooldown(
                        session, _CALIB_SQUASH_ALERT_KEY
                    )
                    if last_at is None or (now - last_at) >= timedelta(
                        hours=_CALIB_SQUASH_COOLDOWN_HOURS
                    ):
                        await alerter._enqueue(
                            f"\U0001f4c9 *Low throughput — calibration squash*\n"
                            f"0 fills in 24h; {squash['n_squashed']} edged evals "
                            f"gated by the probability floor after calibration "
                            f"pulled them below it "
                            f"(raw≈{squash['avg_raw']:.2f} → "
                            f"cal≈{squash['avg_cal']:.2f}).\n"
                            f"Likely CORRECT (model overconfidence), not a bug. "
                            f"Forcing volume is −EV; track "
                            f"`shadow-report --key sigma` for the σ-floor fix."
                        )
                        await _persist_alert_cooldown(
                            session, _CALIB_SQUASH_ALERT_KEY, now
                        )
            except Exception:
                logger.warning(
                    "Calibration-squash diagnostic failed (non-fatal)",
                    exc_info=True,
                )

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
            # M2: stamp a config epoch so telemetry can be bucketed into
            # before/after-flip windows. Best-effort, only inserts on change.
            try:
                from src.signals.config_epoch import record_config_epoch
                epoch_id = await record_config_epoch(session)
                await session.commit()
                if epoch_id is not None:
                    logger.info("Config epoch active: id=%s", epoch_id)
            except Exception:
                logger.warning("config epoch record failed (non-fatal)", exc_info=True)
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
        station_rmse_f=cached.station_rmse_f,
        station_rmse_sample_days=cached.station_rmse_sample_days,
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
                # Bracket-class markets gated off by default — see
                # ``settings.BRACKET_MARKETS_ENABLED``. Mirror the unified
                # pipeline's gate so range-overshoot lock fires can't
                # bypass the brake.
                if (
                    not settings.BRACKET_MARKETS_ENABLED
                    and m.parsed_operator in ("bracket", "range")
                ):
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
                    # Defense-in-depth mirror of the unified pipeline's
                    # future-day filter. _market_daily_max also rejects the
                    # strictly-future case via its utc_end <= utc_start
                    # guard, but keeping the two job paths symmetric prevents
                    # drift if _market_daily_max semantics ever change.
                    if _should_skip_future_day(market, now_utc, station_icao=icao):
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
# Stuck-OPEN heartbeat cooldown helpers — mirror the
# ``circuit_breakers._load_paused_until`` / ``_persist_paused_until`` pattern
# so a process restart doesn't reset the cooldown timer, and the missing
# ``bot_state`` table fails open (rather than killing the reconcile tick).
# ---------------------------------------------------------------------------

_STUCK_ALERT_KEY = "reconcile.stuck_alert_last_pushed_at"


async def _load_stuck_alert_cooldown(
    session: "AsyncSessionType",
) -> datetime | None:
    try:
        row = await session.get(BotState, _STUCK_ALERT_KEY)
    except ProgrammingError:
        await session.rollback()
        logger.warning(
            "bot_state table missing — stuck-trade alert cooldown not "
            "persisted. Run `alembic upgrade head`."
        )
        return None
    if row is None or row.value is None:
        return None
    iso = row.value.get("at_iso") if isinstance(row.value, dict) else None
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso)
    except ValueError:
        logger.warning(
            "Invalid stuck-alert cooldown value in bot_state: %r", row.value
        )
        return None


async def _persist_stuck_alert_cooldown(
    session: "AsyncSessionType", at: datetime,
) -> None:
    try:
        stmt = (
            pg_insert(BotState)
            .values(
                key=_STUCK_ALERT_KEY,
                value={"at_iso": at.isoformat()},
                updated_at=datetime.now(timezone.utc),
            )
            .on_conflict_do_update(
                index_elements=["key"],
                set_={
                    "value": {"at_iso": at.isoformat()},
                    "updated_at": datetime.now(timezone.utc),
                },
            )
        )
        await session.execute(stmt)
    except ProgrammingError:
        await session.rollback()
        logger.warning(
            "bot_state table missing — stuck-alert cooldown not persisted."
        )


async def job_reconcile_orders() -> None:
    """Poll PENDING/OPEN trades whose fill data hasn't been written yet.

    The CLOB returns ``status=delayed`` with a real ``order_id`` when the
    matching engine queues an order; ``client.get_order`` returns ``None``
    until the engine picks it up. ``_update_fill_details`` correctly
    skips that case in the live order path, but no other job re-tries
    the lookup, so partial fills on delayed orders never made it into
    ``trades.fill_price``/``filled_size`` — slippage analytics were
    blind across the whole live history.

    Scope: trades within the last ``ORDER_RECONCILE_LOOKBACK_HOURS`` whose
    ``order_id`` is set, ``fill_price`` is NULL, and ``exchange_status``
    is one of {``delayed``, ``matched``, ``matching``}. Status filter is
    ``PENDING/OPEN/WON``: WON is included so historical rows that
    ``resolve_trades`` flipped terminal before fill details landed
    (pre-2026-05-19 CLOB-mid path) can still be backfilled. When a
    WON row gets fill data populated here, the per-trade pnl is
    recomputed against the corrected entry_price. For each, calls
    ``check_order_status`` (which itself triggers ``_update_fill_details``
    when the order has reached ``matched``).

    After the per-row reconcile, runs a **log-only stale-OPEN sweep**:
    any OPEN+null-fill rows whose market ended more than
    ``STALE_OPEN_RECONCILE_GRACE_HOURS`` ago are summarised as a high-
    signal warning, and a Telegram heartbeat is pushed when the stuck
    set crosses the count or exposure-fraction threshold (cooldown in
    ``bot_state``). The sweep NEVER marks rows LOST — that requires
    on-chain ``balanceOf`` which is the operator's
    ``admin reconcile-stuck`` flow. Added 2026-05-18 after a 12+ hour
    silencer driven by ``matched``-but-no-fill rows that
    ``check_order_status`` couldn't drive forward (it was silently
    crashing on ``None`` from the CLOB; that bug also fixed in the
    same change).

    No-op in dry-run / no-private-key mode.
    """
    if settings.ORDER_RECONCILE_INTERVAL_MINUTES <= 0:
        return

    from src.execution.polymarket_client import check_order_status, is_live

    if not is_live():
        return

    try:
        async with async_session() as session:
            now = datetime.now(timezone.utc)
            recent_cutoff = (
                now - timedelta(hours=settings.ORDER_RECONCILE_LOOKBACK_HOURS)
            )
            # Long-tail cutoff: catch delayed orders whose fill never
            # landed and whose market is still recent enough that
            # resolution is plausible. Pre-2026-05-17 we only looked
            # back 24h, so the May-13/14 batch of 16 delayed-no-fill
            # trades drifted out of the window before resolution and
            # ended up pinning the exposure cap (incident notes).
            longtail_cutoff = now - timedelta(days=7)
            stmt = (
                select(Trade)
                .join(Market, Trade.market_id == Market.id)
                .where(
                    Trade.order_id.is_not(None),
                    Trade.fill_price.is_(None),
                    Trade.exchange_status.in_(["delayed", "matched", "matching"]),
                    # PENDING / OPEN: still waiting for fill data.
                    # WON: pre-2026-05-19 ``resolve_trades`` could flip
                    # OPEN→WON on CLOB-mid before fill details landed,
                    # leaving the row terminal with NULL fill_price.
                    # After Fix 1 that resolution now waits for the
                    # on-chain payout, but we still need to backfill
                    # the historical WON+null-fill rows so pnl/stake
                    # accounting catches up. LOST stays excluded —
                    # fill_price=NULL there means the order genuinely
                    # never landed and there's nothing to fix.
                    Trade.status.in_(
                        [TradeStatus.PENDING, TradeStatus.OPEN, TradeStatus.WON]
                    ),
                    (
                        (Trade.opened_at >= recent_cutoff)
                        | (Market.end_date >= longtail_cutoff)
                    ),
                )
            )
            result = await session.execute(stmt)
            trades = result.scalars().all()
            if not trades:
                return

            updated = 0
            for trade in trades:
                try:
                    prior_fill = trade.fill_price
                    prior_stake = float(trade.stake_usd or 0.0)
                    status = await check_order_status(trade)
                    if trade.fill_price is not None and prior_fill is None:
                        updated += 1
                        # ``_update_fill_details`` rewrote entry_price /
                        # stake_usd to the actual fill values. For trades
                        # already flipped to WON by ``resolve_trades``,
                        # the pnl was computed against the pre-backfill
                        # entry_price — recompute against the corrected
                        # one so downstream accounting (daily PnL,
                        # drawdown, calibration) sees real numbers.
                        if trade.status == TradeStatus.WON:
                            entry = trade.entry_price or 0.0
                            trade.pnl = (
                                trade.stake_usd * (1.0 / entry - 1.0)
                                if entry > 0 else 0.0
                            )
                            logger.info(
                                "reconcile: backfilled WON trade %s "
                                "(stake %.2f→%.2f, fill_price=%.3f, pnl=%.2f)",
                                trade.id, prior_stake, trade.stake_usd,
                                trade.fill_price, trade.pnl,
                            )
                except Exception:
                    logger.warning(
                        "reconcile: lookup failed for order %s",
                        trade.order_id, exc_info=True,
                    )
                    continue
            if updated:
                await session.commit()
                logger.info(
                    "reconcile: filled %d/%d delayed/matched orders",
                    updated, len(trades),
                )

            # ---------------- Stale-OPEN sweep + heartbeat ----------------
            # Rows that ``check_order_status`` couldn't drive forward AND
            # whose market has ended past grace. Log-only here; the
            # operator's ``admin reconcile-stuck`` flow is what touches
            # on-chain state. Mirrors the analogous fallback in
            # ``resolution.resolve_trades`` but is reachable WITHOUT
            # requiring ``_refresh_market_price`` to return None first —
            # that gap is exactly what hid incident 2026-05-18.
            try:
                stale_cutoff = now - timedelta(
                    hours=settings.STALE_OPEN_RECONCILE_GRACE_HOURS
                )
                stale_stmt = (
                    select(Trade)
                    .join(Market, Trade.market_id == Market.id)
                    .where(
                        Trade.status == TradeStatus.OPEN,
                        Trade.fill_price.is_(None),
                        Trade.stake_usd > 0,
                        Market.end_date.is_not(None),
                        Market.end_date < stale_cutoff,
                    )
                )
                stale_rows = (
                    await session.execute(stale_stmt)
                ).scalars().all()
                if stale_rows:
                    stuck_stake = sum(
                        float(t.stake_usd) for t in stale_rows
                    )
                    logger.warning(
                        "reconcile: %d stuck OPEN trades past %dh grace "
                        "(stake $%.2f); run `admin reconcile-stuck "
                        "--dry-run` to inspect, then live to settle",
                        len(stale_rows),
                        settings.STALE_OPEN_RECONCILE_GRACE_HOURS,
                        stuck_stake,
                    )

                    bankroll = await get_current_bankroll(session)
                    effective_cap = max(
                        MAX_EXPOSURE_PCT * bankroll,
                        settings.MAX_EXPOSURE_USD_FLOOR,
                    )
                    crosses_count = (
                        len(stale_rows) >= settings.STUCK_ALERT_MIN_COUNT
                    )
                    crosses_exposure = (
                        effective_cap > 0
                        and stuck_stake
                        >= settings.STUCK_ALERT_EXPOSURE_FRACTION
                        * effective_cap
                    )
                    if crosses_count or crosses_exposure:
                        last_at = await _load_stuck_alert_cooldown(session)
                        cooldown = timedelta(
                            hours=settings.STUCK_ALERT_COOLDOWN_HOURS
                        )
                        if last_at is None or (now - last_at) >= cooldown:
                            pct = (
                                100 * stuck_stake / effective_cap
                                if effective_cap > 0
                                else 0.0
                            )
                            alerter = get_alerter()
                            await alerter._enqueue(
                                f"🚨 *Stuck OPEN trades pinning exposure*\n"
                                f"Count: {len(stale_rows)} "
                                f"(stake ${stuck_stake:,.2f})\n"
                                f"Effective cap: ${effective_cap:,.2f}  "
                                f"({pct:.0f}% pinned)\n"
                                f"Bankroll: ${bankroll:,.2f}\n"
                                f"Action: run "
                                f"`python -m src.cli admin reconcile-stuck "
                                f"--dry-run` to inspect, then live to release."
                            )
                            await _persist_stuck_alert_cooldown(session, now)
                            await session.commit()
            except Exception:
                logger.warning(
                    "Stuck-trade heartbeat check failed (non-fatal)",
                    exc_info=True,
                )
    except Exception:
        logger.warning("Order reconciliation job failed", exc_info=True)


async def job_perf_review(days: int) -> None:
    """Periodic performance digest — Layer 1 of the auto-evaluation loop.

    Computes the window digest (``cli.perf_review_result``), persists the
    JSON artifact for the Layer-2 analyst, and pushes a Telegram summary —
    throttled per cadence via a ``bot_state`` cooldown so the daily / 3d / 7d
    jobs don't spam. Propose-only: never changes config. Registered (gated by
    ``PERF_REVIEW_ENABLED``) for days ∈ {1, 3, 7} in ``setup_scheduler``.
    """
    # Lazy import: the CLI module is only needed when this job runs, and
    # importing it at module load would create a scheduler⇄cli cycle.
    from src.cli import (
        _persist_perf_artifact,
        perf_review_result,
        render_perf_digest,
    )

    alerter = get_alerter()
    try:
        async with async_session() as session:
            key = f"perf_review.last_pushed_at.{days}d"
            now = datetime.now(timezone.utc)
            last = await _load_alert_cooldown(session, key)
            # Window just under the cadence so a coalesced/late run still fires
            # but a same-day double-trigger is suppressed.
            cooldown = timedelta(
                hours={1: 20, 3: 68, 7: 164}.get(days, max(1, days * 24 - 4))
            )
            if last is not None and (now - last) < cooldown:
                return
            result = await perf_review_result(session, days)
            await _persist_perf_artifact(session, days, result)
            await alerter._enqueue(render_perf_digest(result, markdown=True))
            await _persist_alert_cooldown(session, key, now)
            await session.commit()
    except Exception as exc:
        logger.exception("Perf-review job failed (%dd)", days)
        await alerter.send_system_error(exc, f"perf review {days}d")


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

    if settings.ORDER_RECONCILE_INTERVAL_MINUTES > 0:
        scheduler.add_job(
            job_reconcile_orders,
            IntervalTrigger(minutes=settings.ORDER_RECONCILE_INTERVAL_MINUTES),
            id="reconcile_orders",
            next_run_time=datetime.now(timezone.utc) + timedelta(minutes=2),
            max_instances=1,
            coalesce=True,
        )
        logger.info(
            "Order reconciliation enabled (every %dm)",
            settings.ORDER_RECONCILE_INTERVAL_MINUTES,
        )

    if settings.PERF_REVIEW_ENABLED:
        # One parametrised job, three cadences. functools.partial (not a loop
        # closure) avoids APScheduler late-binding of `days`. Staggered after
        # the 22:00 settlement so the digest reflects the freshly-settled book.
        for _days, _trigger, _jid in (
            (1, CronTrigger(hour=22, minute=30, timezone="UTC"), "perf_review_daily"),
            (3, CronTrigger(day_of_week="mon,thu", hour=22, minute=40, timezone="UTC"), "perf_review_3d"),
            (7, CronTrigger(day_of_week="mon", hour=22, minute=50, timezone="UTC"), "perf_review_7d"),
        ):
            scheduler.add_job(
                functools.partial(job_perf_review, _days),
                _trigger,
                id=_jid,
                max_instances=1,
                coalesce=True,
            )
        logger.info("Perf-review jobs enabled (daily / 3d / 7d)")

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
