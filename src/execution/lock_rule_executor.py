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
    near_peak_floor_eligible,
)
from src.execution.polymarket_client import get_orderbook_depth, place_order
from src.persistence import cache_rollover
from src.persistence.dedup import has_active_trade, upsert_signal
from src.risk.cluster_cap import cluster_stake_used
from src.risk.drawdown import DrawdownMonitor
from src.risk.kelly import size_locked_position
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
from src.signals.evaluation_log import log_evaluation
from src.signals.lock_rules import evaluate_lock

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


def _lock_big_size_stations() -> set[str]:
    """Parse ``settings.LOCK_BIG_SIZE_STATIONS`` (comma-separated ICAOs) into a
    set of upper-cased codes. Empty when unset — conviction sizing then applies
    to nobody, even with ``LOCK_CONVICTION_SIZING_ENABLED`` on."""
    raw = settings.LOCK_BIG_SIZE_STATIONS or ""
    return {s.strip().upper() for s in raw.split(",") if s.strip()}


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
    live_quote: bool = True,
    seconds_since_obs: float | None = None,
) -> float | None:
    """Evaluate lock-rule conditions and place order if triggered.

    ``yes_bid`` / ``yes_ask``: optional live quote. When supplied, the
    side we're buying is charged its real ask cost (yes_ask for YES,
    1-yes_bid for NO) instead of the symmetric mid carried in
    ``yes_price``. This prevents a wide post-move spread from making a
    locked market look mid-priced and slipping through the
    ``LOCK_RULE_MAX_PRICE`` guard.

    ``live_quote``: False when ``get_best_bid_ask`` returned None — i.e. the
    book is one-sided or empty, so there is no executable price. We refuse to
    price off ``yes_price`` in that case (it would be the stale Gamma snapshot)
    and emit the ``lock_unexecutable`` telemetry line instead.
    """
    # Local import: avoids circular ``edge_calculator`` → ``binary_market``
    # → polymarket-parser cycle at module load.
    from src.signals.edge_calculator import _check_filters

    decision = evaluate_lock(state, market)
    if decision.side is None or decision.direction is None:
        # No lock fired — nothing to log; the probability path will emit
        # its own EvaluationLog row for this market on this tick.
        return None

    # A lock fired but the book is empty/one-sided, so nothing is buyable.
    # This is the NORMAL state of a long-dead bucket: once it is worthless the
    # YES bids vanish, and by the CLOB mirror invariant (YES.bid + NO.ask = 1)
    # no YES bids means no NO asks — nobody sells a certain $1 for less.
    # We cannot write an EvaluationLog row (market_prob/edge are NOT NULL and
    # inventing a price is precisely the stale-Gamma bug this replaced), so
    # emit structured telemetry instead. THIS IS THE DECISIVE MEASUREMENT for
    # whether bucket_overshoot is executable at all: grep `lock_unexecutable`
    # and compare `seconds_since_obs` against the fills. If even *fresh* kills
    # (low seconds_since_obs) always show an empty book, the edge does not
    # exist at any latency and the 07-10 study — which priced candidates off
    # CLOB prices-history (trades/mids), not resting offers — was measuring a
    # price that was never executable.
    if not live_quote:
        logger.info(
            "[%s] lock_unexecutable %s %s [%s]: empty/one-sided book, nothing to buy",
            icao, decision.side, market.id[:12], decision.branch,
            extra={
                "event": "lock_unexecutable",
                "icao": icao,
                "market_id": market.id,
                "lock_branch": decision.branch,
                "side": decision.side,
                "routine_count": decision.routine_count,
                "seconds_since_obs": seconds_since_obs,
            },
        )
        return 0.0

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
            forecast_peak_f=state.forecast_peak_f,
            current_max_f=state.current_max_f,
            hours_until_peak=state.hours_until_peak,
            forecast_sigma_f=state.forecast_sigma_f,
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
        await log_decision(
            session,
            market_id=market.id,
            direction=decision.direction,
            signal_kind="lock",
            outcome=OUTCOME_DUP_INPROC,
            metadata={"branch": decision.branch, "side": decision.side},
        )
        return 0.0
    if await has_active_trade(session, market.id, decision.direction):
        logger.info(
            "[%s] LOCK %s %s: active trade exists for this market+side, skipping",
            icao, decision.side, market.id[:12],
        )
        cache_rollover.record_lock_fire(market.id, icao)
        await _log_lock_eval(False, "active trade exists", None)
        await log_decision(
            session,
            market_id=market.id,
            direction=decision.direction,
            signal_kind="lock",
            outcome=OUTCOME_DUP_DB,
            metadata={"branch": decision.branch, "side": decision.side},
        )
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

    # range_in_window (YES) min-price gate: the only realized fill ever was a
    # 0.40-priced "exactly" YES gamble that lost (2026-06-30 audit). Require a
    # near-certain YES price so the branch stops firing cheap low-conviction
    # bets. effective_price == yes_price for a YES side here.
    if (
        decision.branch == "range_in_window"
        and decision.side == "YES"
        and effective_price < settings.RANGE_IN_WINDOW_MIN_YES_PRICE
    ):
        logger.info(
            "[%s] lock %s %s: range_in_window price %.2f < min %.2f",
            icao, decision.side, market.id[:12], effective_price,
            settings.RANGE_IN_WINDOW_MIN_YES_PRICE,
        )
        await _log_lock_eval(
            False,
            f"range_in_window price {effective_price:.2f} < min {settings.RANGE_IN_WINDOW_MIN_YES_PRICE}",
            None,
        )
        return 0.0

    # bucket_overshoot max-cost gate. The rule is certain; the *price* decides
    # whether the residual edge still covers resolver-divergence risk. Above
    # BUCKET_OVERSHOOT_MAX_COST the market has already repriced and break-even
    # needs a lower divergence rate than we can guarantee. (This is the mistake
    # the old range_overshoot branch made: it bought NO at an average 0.901.)
    if (
        decision.branch == "bucket_overshoot"
        and effective_price > settings.BUCKET_OVERSHOOT_MAX_COST
    ):
        logger.info(
            "[%s] lock %s %s: bucket_overshoot cost %.3f > max %.2f (already repriced)",
            icao, decision.side, market.id[:12], effective_price,
            settings.BUCKET_OVERSHOOT_MAX_COST,
        )
        await _log_lock_eval(
            False,
            f"bucket_overshoot cost {effective_price:.3f} > max "
            f"{settings.BUCKET_OVERSHOOT_MAX_COST}",
            None,
        )
        return 0.0

    # Without token IDs there is no book to price or probe against, and
    # `effective_price` would have been derived from the stale Gamma snapshot
    # rather than the live CLOB quote. Reject explicitly: the old code fell
    # through with `buy_depth = 0.0`, so this failure mode disguised itself as
    # "depth $0 < $10" and quietly filled evaluation_logs with NO costs that
    # never existed on the book.
    if not token_ids:
        await _log_lock_eval(False, "no token IDs (cannot price the live book)", None)
        return 0.0

    # Depth against the side we're actually buying.
    if decision.side == "YES":
        buy_depth = yes_depth
    else:
        buy_depth = get_orderbook_depth(token_ids[1], effective_price)

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

    # Conviction-weighted sizing: only the truly monotonic EASY-YES direction,
    # only on a station whose routine-METAR max is trusted to match the
    # resolver (whitelist), only when the master flag is on. Everything else
    # falls through to the flat 2% path. See `risk.kelly.size_locked_position`.
    conviction = (
        settings.LOCK_CONVICTION_SIZING_ENABLED
        and decision.side == "YES"
        and decision.branch in ("easy_super", "easy_standard")
        and icao.upper() in _lock_big_size_stations()
    )

    if conviction:
        win_prob = (
            settings.LOCK_WIN_PROB_SUPER
            if decision.branch == "easy_super"
            else settings.LOCK_WIN_PROB_STANDARD
        )
        max_position_pct = (
            settings.LOCK_MAX_POSITION_PCT_SUPER
            if decision.branch == "easy_super"
            else settings.LOCK_MAX_POSITION_PCT_STANDARD
        )
        pos = size_locked_position(
            bankroll=bankroll,
            price=effective_price,
            current_exposure=exposure,
            orderbook_depth=buy_depth or None,
            win_prob=win_prob,
            kelly_fraction=settings.LOCK_KELLY_FRACTION,
            max_position_pct=max_position_pct,
            depth_cap_pct=settings.LOCK_DEPTH_CAP_PCT_BIG,
        )
    else:
        win_prob = None
        # Near-peak floor-up: lock decisions are deterministic (prob ≡ 1.0), so
        # the confidence arm of the gate always passes for threshold lock
        # markets; bracket-like lock branches (range_*) are excluded
        # (validate-first). Redundant under conviction sizing (stakes already
        # clear the floor), so the flat path owns it.
        floor_to_usd = (
            settings.NEAR_PEAK_FLOOR_STAKE_USD
            if near_peak_floor_eligible(
                market,
                our_probability=1.0,
                hours_until_peak=state.hours_until_peak,
            )
            else None
        )
        pos = size_locked_position(
            bankroll=bankroll,
            price=effective_price,
            current_exposure=exposure,
            orderbook_depth=buy_depth or None,
            floor_to_usd=floor_to_usd,
        )
    dd_state = monitor.check(bankroll)
    stake = pos.stake_usd * dd_state.size_multiplier
    floored_up = (pos.reason or "").startswith("floored")

    logger.info(
        "[%s] LOCK %s %s: margin=%.1f°F, price=%.2f, stake=$%.2f "
        "(raw=$%.2f, dd_mult=%.2f) | %s",
        icao, decision.side, market.id[:12], decision.margin_f,
        effective_price, stake, pos.stake_usd, dd_state.size_multiplier,
        "; ".join(decision.reasons),
        extra={
            "event": "lock_sized",
            "icao": icao,
            "market_id": market.id,
            "side": decision.side,
            "branch": decision.branch,
            "margin_f": decision.margin_f,
            "effective_price": effective_price,
            "stake_usd": stake,
            "raw_stake_usd": pos.stake_usd,
            "dd_multiplier": dd_state.size_multiplier,
            "routine_count": decision.routine_count,
            "conviction": conviction,
            "win_prob": win_prob,
            "kelly_pct": pos.kelly_pct,
            "walk_max_price": (
                settings.LOCK_WALK_MAX_PRICE if conviction else None
            ),
        },
    )

    if stake < settings.MIN_STAKE_USD:
        logger.info(
            "[%s] LOCK %s %s: stake $%.2f < min $%.2f, skipping",
            icao, decision.side, market.id[:12], stake, settings.MIN_STAKE_USD,
        )
        paused = dd_state.size_multiplier == 0.0
        await log_decision(
            session,
            market_id=market.id,
            direction=decision.direction,
            signal_kind="lock",
            outcome=OUTCOME_DRAWDOWN_PAUSED if paused else OUTCOME_STAKE_BELOW_MIN,
            requested_stake_usd=pos.stake_usd,
            actual_stake_usd=stake,
            dd_multiplier=dd_state.size_multiplier,
            dd_level=dd_state.level.value,
            metadata={
                "branch": decision.branch,
                "margin_f": decision.margin_f,
                # Which cap actually zeroed the stake — see the parallel
                # comment in scheduler.__init__ for the probability path.
                "size_reason": pos.reason,
                "depth_usd": buy_depth,
                "effective_price": effective_price,
                "conviction": conviction,
                "win_prob": win_prob,
            },
        )
        return 0.0

    # Cluster cap also applies to lock-rule fires on bracket/exactly
    # markets (range_undershoot/in_window branches). Threshold
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
            await log_decision(
                session,
                market_id=market.id,
                direction=decision.direction,
                signal_kind="lock",
                outcome=OUTCOME_CLUSTER_CAP,
                requested_stake_usd=stake,
                metadata={
                    "branch": decision.branch,
                    "cluster_used_usd": cluster_used,
                    "cap_usd": settings.CLUSTER_STAKE_CAP_USD,
                    "conviction": conviction,
                    "win_prob": win_prob,
                },
            )
            return 0.0

    # model_prob=1.0 because the lock rule is deterministic (no probability
    # estimate to record). lock_margin_f carries the °F margin from
    # threshold ("how locked was this") so the detail view can surface it.
    # Lock fields tag the branch + observation context so post-mortems can
    # split realised P&L by which lock path produced the signal.
    sig_row = await upsert_signal(
        session,
        market_id=market.id,
        direction=decision.direction,
        model_prob=1.0,
        market_prob=effective_price,
        edge=1.0 - effective_price,
        signal_kind="lock",
        lock_branch=decision.branch,
        lock_routine_count=decision.routine_count,
        lock_observed_max_f=decision.observed_max_f,
        lock_margin_f=decision.margin_f,
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

    # Walk the book for conviction locks: widen the FAK slippage budget so the
    # limit reaches LOCK_WALK_MAX_PRICE and the order sweeps every ask at/below
    # it (place_order already reconciles stake_usd to the actual fill). Clamped
    # ≥ the 2¢ default so a near-ceiling price never narrows the budget. The
    # flat path omits the kwarg entirely → place_order's 2¢ default stands.
    order_kwargs: dict = {
        "submit_yes_bid": yes_bid,
        "submit_yes_ask": yes_ask,
        "submit_depth_usd": buy_depth or None,
    }
    if conviction:
        order_kwargs["max_slippage_cents"] = max(
            2.0, (settings.LOCK_WALK_MAX_PRICE - effective_price) * 100.0
        )
    order_ok = await place_order(trade, session, **order_kwargs)
    if not order_ok:
        # Distinguish wallet pre-flight skips from real submission failures
        # so dashboards can separate "held back intentionally" from "tried
        # and lost". See decision_log.OUTCOME_INSUFFICIENT_BALANCE.
        failed_outcome = (
            OUTCOME_INSUFFICIENT_BALANCE
            if trade.exchange_status == "insufficient_balance"
            else OUTCOME_ORDER_FAILED
        )
        logger.warning(
            "[%s] LOCK %s %s: order placement failed (%s)",
            icao, decision.side, market.id[:12], trade.exchange_status,
            extra={
                "event": "lock_order_failed",
                "icao": icao,
                "market_id": market.id,
                "side": decision.side,
                "stake_usd": stake,
                "exchange_status": trade.exchange_status,
            },
        )
        await log_decision(
            session,
            market_id=market.id,
            direction=decision.direction,
            signal_kind="lock",
            outcome=failed_outcome,
            requested_stake_usd=stake,
            dd_multiplier=dd_state.size_multiplier,
            dd_level=dd_state.level.value,
            metadata={
                "branch": decision.branch,
                "exchange_status": trade.exchange_status,
                "conviction": conviction,
                "win_prob": win_prob,
            },
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
        await log_decision(
            session,
            market_id=market.id,
            direction=decision.direction,
            signal_kind="lock",
            outcome=OUTCOME_TRADE_PENDING,
            requested_stake_usd=stake,
            actual_stake_usd=indicative_stake,
            dd_multiplier=dd_state.size_multiplier,
            dd_level=dd_state.level.value,
            metadata={
                "branch": decision.branch,
                "margin_f": decision.margin_f,
                "is_dry_run": True,
                "floored_up": floored_up,
                "conviction": conviction,
                "win_prob": win_prob,
            },
        )
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
                extra={
                    "event": "lock_no_fill",
                    "icao": icao,
                    "market_id": market.id,
                    "side": decision.side,
                    "requested_stake_usd": stake,
                },
            )
            await log_decision(
                session,
                market_id=market.id,
                direction=decision.direction,
                signal_kind="lock",
                outcome=OUTCOME_NO_FILL,
                requested_stake_usd=stake,
                metadata={
                    "branch": decision.branch,
                    "margin_f": decision.margin_f,
                    "conviction": conviction,
                    "win_prob": win_prob,
                },
            )
            return 0.0
        trade.status = TradeStatus.OPEN
        indicative_stake = actual_stake
        indicative_price = trade.fill_price or effective_price
        await log_decision(
            session,
            market_id=market.id,
            direction=decision.direction,
            signal_kind="lock",
            outcome=OUTCOME_TRADE_FILLED,
            requested_stake_usd=stake,
            actual_stake_usd=actual_stake,
            dd_multiplier=dd_state.size_multiplier,
            dd_level=dd_state.level.value,
            metadata={
                "branch": decision.branch,
                "margin_f": decision.margin_f,
                "fill_price": indicative_price,
                "floored_up": floored_up,
                "conviction": conviction,
                "win_prob": win_prob,
            },
        )

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
