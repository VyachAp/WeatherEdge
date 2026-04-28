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
# placed a lock order for; cleared by `job_daily_settlement` at 22:00 UTC.
# `_last_routine_seen` skips METARs the fast loop already processed.
_locked_markets_fired_today: set[str] = set()
_last_routine_seen: dict[str, datetime] = {}

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
                    token_ids = await get_token_ids(market.id)
                    if token_ids:
                        quote = get_best_bid_ask(token_ids[0])
                        if quote:
                            yes_bid, yes_ask = quote
                            live_price = (yes_bid + yes_ask) / 2
                            market.current_yes_price = live_price
                        if market.current_yes_price:
                            mkt_depth = get_orderbook_depth(token_ids[0], market.current_yes_price)

                    price = market.current_yes_price or 0.0

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

                        from src.db.models import Signal
                        # Store the side-effective probability so the
                        # consensus calibration regression (which reads
                        # `model_prob` against won={0,1}) treats both
                        # YES and NO trades uniformly. See
                        # src/signals/consensus.py.
                        sig_row = Signal(
                            market_id=market.id,
                            model_prob=edge.our_probability,
                            market_prob=edge.market_price,
                            edge=edge.edge,
                            direction=edge.direction,
                            confidence=edge.our_probability,
                        )
                        session.add(sig_row)
                        await session.flush()

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

                        order_ok = await place_order(trade, session)
                        if order_ok and (trade.stake_usd or 0.0) > 0:
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
        return None

    if decision.side == "YES":
        effective_price = (
            yes_ask if (yes_ask is not None and yes_ask > 0) else yes_price
        )
    else:
        effective_price = (
            (1.0 - yes_bid) if (yes_bid is not None and yes_bid > 0)
            else (1.0 - yes_price)
        )
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
        return 0.0

    # Depth against the side we're actually buying.
    if decision.side == "YES":
        buy_depth = yes_depth
    else:
        buy_depth = (
            get_orderbook_depth(token_ids[1], effective_price)
            if token_ids else 0.0
        )

    now = datetime.now(timezone.utc)
    minutes_to_close = (end_time - now).total_seconds() / 60.0

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
        return 0.0

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

    sig_row = Signal(
        market_id=market.id,
        model_prob=1.0,  # Lock rule is deterministic; carry no prob estimate.
        market_prob=effective_price,
        edge=1.0 - effective_price,
        direction=decision.direction,
        confidence=decision.margin_f,
    )
    session.add(sig_row)
    await session.flush()

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

    order_ok = await place_order(trade, session)
    if not order_ok:
        logger.warning(
            "[%s] LOCK %s %s: order placement failed",
            icao, decision.side, market.id[:12],
        )
        return 0.0

    # In dry-run, ``place_order`` is a no-op and never updates fill fields.
    # Don't pretend the trade opened: keep status PENDING, zero stake_usd
    # so DB-backed exposure/PnL stays clean, and emit a clearly-labelled
    # indicative alert. We still return a positive value so the caller's
    # in-process dedup blocks repeat firings on the same market today.
    is_dry_run = trade.exchange_status == "dry_run"

    if is_dry_run:
        indicative_stake = stake
        indicative_price = trade.entry_price or effective_price
        trade.stake_usd = 0.0
        trade.status = TradeStatus.PENDING
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

            # 5. Station bias recording
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

            # 6. Reset fast-lock-poll dedup so the next trading day starts clean.
            _locked_markets_fired_today.clear()
            _last_routine_seen.clear()
            from src.signals.state_aggregator import clear_state_cache
            clear_state_cache()

            # 7. Weekly calibration check (Sundays)
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
                    market.current_yes_price = yes_price

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
