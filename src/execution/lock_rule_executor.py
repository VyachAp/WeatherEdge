"""Lock-rule trade executor — the side-effect wrapper around
``signals.lock_rules.evaluate_lock``.

Mirrors the split in the probability path: pure decision lives in
``signals.lock_rules`` (LockDecision), order placement + sizing +
filter-gating + Signal/Trade persistence lives here. Pulled out of
``scheduler.py`` so the scheduler stays an orchestration layer.

Flow per call:
  1. ``evaluate_lock`` — decide YES/NO/None.
  2. Compute side-effective price from yes_bid/ask (avoid mid-price
     phantom-edge on wide spreads).
  3. Same-tick dedup (``locked_markets_fired_today``) + durable dedup
     (``has_active_trade``).
  4. Price + filter gates (``_check_filters`` with min_routine_count=2
     for super-margin EASY).
  5. Cluster-cap gate for bracket/exactly markets.
  6. ``size_locked_position`` × drawdown multiplier.
  7. ``upsert_signal`` + ``Trade(PENDING)`` row + ``place_order`` FAK BUY.
  8. Dry-run leaves status=PENDING with stake_usd populated; live path
     advances to OPEN on partial/full fill, stays PENDING on no-fill.
  9. Emit Telegram 🔒 alert with "(dry-run)" tag when applicable.

Return contract (matches scheduler caller):
  ``None`` — no lock fired; caller should fall through to probability path.
  ``0.0``  — lock fired but not executable (price, depth, dedup, sizing).
             Caller should ``continue`` (skip the probability path on this
             market this tick).
  ``>0``   — stake in USD actually placed; caller should add to exposure.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from src.config import settings
from src.db.models import Trade, TradeStatus
from src.execution.binary_market import (
    display_bucket,
    market_range_f,
    market_unit,
)
from src.execution.polymarket_client import get_orderbook_depth, place_order
from src.persistence import cache_rollover
from src.persistence.dedup import has_active_trade, upsert_signal
from src.risk.cluster_cap import cluster_stake_used
from src.risk.drawdown import DrawdownMonitor
from src.risk.kelly import size_locked_position
from src.signals.evaluation_log import log_evaluation
from src.signals.lock_rules import evaluate_lock

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


async def try_lock_rule_trade(
    *,
    session: "AsyncSession",
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
    """
    # Local import: avoids circular ``edge_calculator`` → ``binary_market``
    # → polymarket-parser cycle at module load.
    from src.signals.edge_calculator import _check_filters

    decision = evaluate_lock(state, market)
    if decision.side is None or decision.direction is None:
        # No lock fired — nothing to log; the probability path will emit
        # its own EvaluationLog row for this market on this tick.
        return None

    # Effective price needs to land before has_active_trade so the
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
        await log_evaluation(
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
    # ``locked_markets_fired_today`` is the same-tick speed-up; the DB
    # check is the durable line of defence (survives restarts and gates
    # both modes). Returning 0.0 (not None) so the caller treats this as
    # "lock evaluated but not executed" and skips the probability path
    # for this market on this tick.
    if market.id in cache_rollover.locked_markets_fired_today:
        logger.info(
            "[%s] LOCK %s %s: already fired this tick (in-process dedup)",
            icao, decision.side, market.id[:12],
        )
        await _log_lock_eval(False, "fired this tick", None)
        return 0.0
    if await has_active_trade(session, market.id, decision.direction):
        logger.info(
            "[%s] LOCK %s %s: active trade exists for this market+side, skipping",
            icao, decision.side, market.id[:12],
        )
        cache_rollover.record_lock_fire(market.id, icao)
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

    # Cluster cap also applies to lock-rule fires on bracket/exactly
    # markets (range_overshoot/undershoot/in_window branches). Threshold
    # markets are unaffected because ``cluster_stake_used`` returns 0.0
    # for non-bracket-like operators.
    if settings.CLUSTER_STAKE_CAP_USD > 0:
        cluster_used = await cluster_stake_used(session, market)
        if cluster_used + stake > settings.CLUSTER_STAKE_CAP_USD:
            logger.info(
                "[%s] LOCK %s %s: cluster stake $%.2f + new $%.2f > cap $%.2f, skipping",
                icao, decision.side, market.id[:12],
                cluster_used, stake, settings.CLUSTER_STAKE_CAP_USD,
            )
            return 0.0

    # model_prob=1.0 because the lock rule is deterministic (no probability
    # estimate to record). confidence carries the lock margin in °F so the
    # detail view can show "how locked was this". Lock fields tag the
    # branch + observation context so post-mortems can split realised P&L
    # by which lock path produced the signal.
    sig_row = await upsert_signal(
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

    unit = market_unit(market)
    rng = market_range_f(market)
    if rng is not None and rng[0] != rng[1]:
        threshold_disp = (
            f"[{display_bucket(rng[0], unit)}-{display_bucket(rng[1], unit)}]"
        )
        op_symbol = "∈"
    elif market.parsed_threshold is not None:
        threshold_disp = str(display_bucket(int(market.parsed_threshold), unit))
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


def extract_bracket_buckets(market) -> list[int]:
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


def extract_market_prices(market, buckets: list[int]) -> dict[int, float]:
    """Map bucket values to current YES prices for bracket markets."""
    prices: dict[int, float] = {}
    if market.current_yes_price and buckets:
        prices[buckets[0]] = market.current_yes_price
    return prices


def minimal_state_for_easy_lock(
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
