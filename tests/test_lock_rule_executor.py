"""Tests for ``execution.lock_rule_executor.try_lock_rule_trade``.

The executor wraps ``signals.lock_rules.evaluate_lock`` with everything
that touches the outside world: filter gates, sizing, dedup, Signal /
Trade persistence, FAK order placement, Telegram alert. It's the
load-bearing live-trading entrypoint for the lock-rule path, so each
exit point matters:

  - ``None``  → no lock fired; caller falls through to probability path.
  - ``0.0``   → lock fired but not executable (price out of range, depth,
                dedup, sizing). Caller `continue`s.
  - ``> 0``   → stake actually placed; caller adds to exposure.

The tests mock the executor's collaborators at the *module-level
boundary* (``evaluate_lock``, ``has_active_trade``, ``upsert_signal``,
``cluster_stake_used``, ``place_order``, ``get_orderbook_depth``,
``size_locked_position``, ``log_evaluation``, and the alerter) so each
test exercises a single return point with no DB / network involvement.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.db.models import TradeDirection, TradeStatus
from src.execution import lock_rule_executor as lre
from src.execution.lock_rule_executor import try_lock_rule_trade
from src.risk.drawdown import DrawdownLevel
from src.signals.lock_rules import LockDecision


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_dedup_state():
    """Each test starts with empty in-process dedup. Otherwise a previous
    test's ``record_lock_fire`` would block this test's market id."""
    from src.persistence import cache_rollover
    cache_rollover.locked_markets_fired_today.clear()
    cache_rollover.market_to_icao.clear()
    yield
    cache_rollover.locked_markets_fired_today.clear()
    cache_rollover.market_to_icao.clear()


def _market(
    *, market_id: str = "0x" + "a" * 64,
    operator: str = "above",
    threshold: float | None = 80.0,
    question: str = "Will the highest temp in Phoenix be 80°F or higher today?",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=market_id,
        question=question,
        parsed_operator=operator,
        parsed_threshold=threshold,
        parsed_location="Phoenix",
        end_date=datetime.now(timezone.utc) + timedelta(hours=4),
    )


def _state(*, routine_count: int = 5) -> SimpleNamespace:
    """Minimal WeatherState stand-in. The executor reads
    ``routine_count_today`` directly plus the forecast-context fields
    that get logged on the EvaluationLog row; ``evaluate_lock`` is mocked
    so none of the other fields matter."""
    return SimpleNamespace(
        routine_count_today=routine_count,
        forecast_peak_f=80.0,
        current_max_f=78.0,
        hours_until_peak=2.0,
        forecast_sigma_f=2.0,
    )


def _monitor(multiplier: float = 1.0) -> MagicMock:
    monitor = MagicMock()
    monitor.check = MagicMock(return_value=SimpleNamespace(
        level=DrawdownLevel.NORMAL,
        size_multiplier=multiplier,
    ))
    return monitor


def _alerter() -> MagicMock:
    a = MagicMock()
    a._enqueue = AsyncMock()
    return a


async def _invoke(
    *,
    decision: LockDecision,
    # Default inside [LOCK_RULE_MIN_PRICE=0.05, LOCK_RULE_MAX_PRICE=0.95]
    # so tests that aren't *about* the price gate don't fail at it.
    yes_price: float = 0.92,
    yes_depth: float = 200.0,
    state: SimpleNamespace | None = None,
    market: SimpleNamespace | None = None,
    monitor: MagicMock | None = None,
    bankroll: float = 1000.0,
    has_active: bool = False,
    cluster_used: float = 0.0,
    sized_stake: float = 20.0,
    place_returns: bool = True,
    place_side_effect=None,
    upsert_sig_id: int = 42,
    depth_no: float = 100.0,
    cluster_cap: float = 100.0,
    yes_bid: float | None = None,
    yes_ask: float | None = None,
    alerter: MagicMock | None = None,
    token_ids: tuple[str, str] | None = ("yes_token", "no_token"),
) -> tuple[float | None, dict]:
    """Drive ``try_lock_rule_trade`` with every collaborator stubbed.

    Returns ``(return_value, captures)`` where ``captures`` is a dict
    holding the mocks/AsyncMocks so individual tests can assert on calls.
    """
    market = market or _market()
    state = state or _state()
    monitor = monitor or _monitor()
    alerter = alerter or _alerter()
    session = AsyncMock()
    # session.add is sync on a real Session — match shape so we don't
    # accumulate "coroutine never awaited" warnings.
    session.add = MagicMock()

    # signal_id is read off the row returned by ``upsert_signal``.
    sig_row = SimpleNamespace(id=upsert_sig_id)

    captures: dict = {
        "alerter": alerter,
        "session": session,
        "place_order": AsyncMock(
            return_value=place_returns, side_effect=place_side_effect,
        ),
        "upsert_signal": AsyncMock(return_value=sig_row),
        "has_active_trade": AsyncMock(return_value=has_active),
        "log_evaluation": AsyncMock(),
        "cluster_stake_used": AsyncMock(return_value=cluster_used),
        "evaluate_lock": MagicMock(return_value=decision),
        "size_locked_position": MagicMock(return_value=SimpleNamespace(
            stake_usd=sized_stake, kelly_pct=0.02, capped=False, reason=None,
        )),
        "get_orderbook_depth": MagicMock(return_value=depth_no),
    }

    with patch.object(lre, "evaluate_lock", captures["evaluate_lock"]), \
         patch.object(lre, "has_active_trade", captures["has_active_trade"]), \
         patch.object(lre, "upsert_signal", captures["upsert_signal"]), \
         patch.object(lre, "cluster_stake_used", captures["cluster_stake_used"]), \
         patch.object(lre, "log_evaluation", captures["log_evaluation"]), \
         patch.object(lre, "place_order", captures["place_order"]), \
         patch.object(lre, "size_locked_position", captures["size_locked_position"]), \
         patch.object(lre, "get_orderbook_depth", captures["get_orderbook_depth"]), \
         patch.object(lre.settings, "CLUSTER_STAKE_CAP_USD", cluster_cap), \
         patch.object(lre.settings, "MIN_STAKE_USD", 5.0):
        result = await try_lock_rule_trade(
            session=session,
            market=market,
            state=state,
            yes_price=yes_price,
            token_ids=token_ids,
            yes_depth=yes_depth,
            end_time=market.end_date,
            bankroll=bankroll,
            exposure=0.0,
            monitor=monitor,
            alerter=alerter,
            icao="KPHX",
            yes_bid=yes_bid,
            yes_ask=yes_ask,
        )
    return result, captures


def _yes_decision(*, branch: str = "easy_super", margin: float = 4.0) -> LockDecision:
    return LockDecision(
        side="YES",
        reasons=[f"observed max +{margin}°F over threshold"],
        margin_f=margin,
        branch=branch,
        routine_count=3,
        observed_max_f=85.0,
    )


def _no_decision(*, branch: str = "easy_super", margin: float = 4.0) -> LockDecision:
    return LockDecision(
        side="NO",
        reasons=["forecast peak < threshold and past peak"],
        margin_f=margin,
        branch=branch,
        routine_count=3,
        observed_max_f=72.0,
    )


# ---------------------------------------------------------------------------
# Decision = None — caller should fall through to probability path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_returns_none_when_no_lock_fires():
    """No DB write, no log row — letting the probability path own the
    EvaluationLog entry for this market on this tick."""
    result, c = await _invoke(decision=LockDecision(side=None))

    assert result is None
    c["log_evaluation"].assert_not_called()
    c["has_active_trade"].assert_not_called()
    c["upsert_signal"].assert_not_called()
    c["place_order"].assert_not_called()


# ---------------------------------------------------------------------------
# Dedup paths — return 0.0
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_returns_zero_when_in_process_dedup_hits():
    """Same-tick dedup — `locked_markets_fired_today` membership skips
    re-firing and logs a reject row so the EvaluationLog audit is
    complete."""
    from src.persistence import cache_rollover
    market = _market(market_id="0xMKT")
    cache_rollover.locked_markets_fired_today.add(market.id)

    result, c = await _invoke(decision=_yes_decision(), market=market)

    assert result == 0.0
    c["has_active_trade"].assert_not_called()
    c["log_evaluation"].assert_awaited_once()
    assert c["log_evaluation"].await_args.kwargs["reject_reason"] == "fired this tick"


@pytest.mark.asyncio
async def test_returns_zero_when_active_trade_exists_and_records_dedup():
    """DB-backed dedup. Future-proof guarantee: even if the in-process
    set was wiped (restart, rollover), an existing PENDING/OPEN trade
    blocks. Records the lock_fire so the next per-station rollover can
    find this market id."""
    from src.persistence import cache_rollover

    market = _market(market_id="0xACTIVE")
    result, c = await _invoke(
        decision=_yes_decision(), market=market, has_active=True,
    )

    assert result == 0.0
    c["has_active_trade"].assert_awaited_once()
    c["log_evaluation"].assert_awaited_once()
    assert c["log_evaluation"].await_args.kwargs["reject_reason"] == "active trade exists"
    # In-process map updated for rollover discovery.
    assert market.id in cache_rollover.locked_markets_fired_today
    assert cache_rollover.market_to_icao[market.id] == "KPHX"


# ---------------------------------------------------------------------------
# Price gate — return 0.0
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_returns_zero_when_yes_price_above_lock_max():
    """LOCK_RULE_MAX_PRICE=0.95 caps how much we'll pay even when the
    outcome is "locked". A market trading at 0.99 has ~no upside left."""
    result, c = await _invoke(decision=_yes_decision(), yes_price=0.99)

    assert result == 0.0
    c["log_evaluation"].assert_awaited_once()
    assert "outside" in c["log_evaluation"].await_args.kwargs["reject_reason"]
    c["upsert_signal"].assert_not_called()


@pytest.mark.asyncio
async def test_no_side_uses_one_minus_yes_bid_for_effective_price():
    """NO-side effective price = 1 - yes_bid (what a NO buyer pays).
    With yes_bid=0.02 → NO buyer pays 0.98, which exceeds LOCK_RULE_MAX_PRICE
    → reject. Guards against wide-spread dust-bid markets slipping through."""
    result, c = await _invoke(
        decision=_no_decision(),
        yes_price=0.50,  # ignored when yes_bid supplied
        yes_bid=0.02,
        yes_ask=0.55,
    )

    assert result == 0.0
    reason = c["log_evaluation"].await_args.kwargs["reject_reason"]
    assert "0.98" in reason  # effective NO price
    assert "outside" in reason


# ---------------------------------------------------------------------------
# Filter rejection — return 0.0
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_returns_zero_when_filter_rejects_low_routine_count():
    """Filter floor of 2 routines applies to lock-rule too; a single
    METAR fluke can't trigger a trade even if evaluate_lock fired."""
    state = _state(routine_count=1)
    result, c = await _invoke(decision=_yes_decision(), state=state)

    assert result == 0.0
    reason = c["log_evaluation"].await_args.kwargs["reject_reason"]
    assert "routine count" in reason
    c["upsert_signal"].assert_not_called()


@pytest.mark.asyncio
async def test_returns_zero_when_filter_rejects_low_depth():
    """Depth gate uses settings.MIN_DEPTH_USD; thin books shouldn't get
    FAK orders that won't fill."""
    result, c = await _invoke(decision=_yes_decision(), yes_depth=1.0)

    assert result == 0.0
    reason = c["log_evaluation"].await_args.kwargs["reject_reason"]
    assert "depth" in reason


# ---------------------------------------------------------------------------
# Sizing rejections — return 0.0
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_returns_zero_when_stake_below_min_after_drawdown():
    """Drawdown PAUSED → multiplier 0.0 → stake collapses to zero → skip.
    Important: the "passes" eval row went out before sizing, so the
    rejection here is not double-counted in the EvaluationLog."""
    monitor = _monitor(multiplier=0.0)
    result, c = await _invoke(
        decision=_yes_decision(), monitor=monitor, sized_stake=20.0,
    )

    assert result == 0.0
    # Passing row fired before sizing; no second reject row appended.
    assert c["log_evaluation"].await_count == 1
    assert c["log_evaluation"].await_args.kwargs["passes"] is True
    c["upsert_signal"].assert_not_called()
    c["place_order"].assert_not_called()


@pytest.mark.asyncio
async def test_returns_zero_when_cluster_cap_would_be_exceeded():
    """Anti-correlation guard: cluster_used + new_stake > cap → skip.
    Defends against multiple bracket buckets sizing independently to
    over-allocate on outcomes that can't all win."""
    result, c = await _invoke(
        decision=_yes_decision(),
        market=_market(operator="exactly"),  # bracket-like → cluster active
        cluster_used=90.0, sized_stake=20.0, cluster_cap=100.0,
    )

    assert result == 0.0
    c["cluster_stake_used"].assert_awaited_once()
    c["upsert_signal"].assert_not_called()
    c["place_order"].assert_not_called()


# ---------------------------------------------------------------------------
# Order-placement failure — return 0.0
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_returns_zero_when_place_order_returns_false():
    """`place_order` returning False = transport / pre-CLOB failure
    (cap exceeded, missing keys). Trade row was already added —
    intentional, the next reconcile run will clean it up."""
    result, c = await _invoke(
        decision=_yes_decision(), place_returns=False,
    )

    assert result == 0.0
    c["upsert_signal"].assert_awaited_once()
    c["place_order"].assert_awaited_once()
    c["alerter"]._enqueue.assert_not_called()


# ---------------------------------------------------------------------------
# Happy paths — return > 0
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dry_run_returns_stake_and_keeps_pending():
    """Dry-run: `place_order` flips trade.exchange_status='dry_run' but
    doesn't update fill fields. Trade stays PENDING with the requested
    stake_usd intact; alert is tagged (dry-run)."""

    async def _flip_to_dry_run(trade, session, **kwargs):
        trade.exchange_status = "dry_run"
        return True

    result, c = await _invoke(
        decision=_yes_decision(), place_side_effect=_flip_to_dry_run,
        sized_stake=20.0,
    )

    assert result == 20.0  # indicative stake
    c["alerter"]._enqueue.assert_awaited_once()
    sent = c["alerter"]._enqueue.await_args.args[0]
    assert "(dry-run)" in sent
    assert "Indicative" in sent
    c["upsert_signal"].assert_awaited_once()


@pytest.mark.asyncio
async def test_live_filled_returns_actual_stake_and_marks_open():
    """Live path: `place_order` populates trade.stake_usd with actual
    fill cost (may differ from sized stake when FAK partial-fills).
    Trade advances to OPEN; alert reads "Filled" not "Indicative"."""

    async def _fill(trade, session, **kwargs):
        trade.stake_usd = 18.5  # FAK got a partial fill
        trade.fill_price = 0.96
        return True

    result, c = await _invoke(
        decision=_yes_decision(), place_side_effect=_fill, sized_stake=20.0,
    )

    assert result == 18.5
    c["alerter"]._enqueue.assert_awaited_once()
    sent = c["alerter"]._enqueue.await_args.args[0]
    assert "Filled" in sent
    assert "(dry-run)" not in sent


@pytest.mark.asyncio
async def test_live_no_fill_returns_zero_and_keeps_pending():
    """Live path with empty book at limit: `place_order` succeeds (order
    posted) but fill is zero. Trade kept PENDING for reconcile / next-tick
    retry; caller should NOT treat this as exposure."""

    async def _no_fill(trade, session, **kwargs):
        trade.stake_usd = 0.0
        return True

    result, c = await _invoke(
        decision=_yes_decision(), place_side_effect=_no_fill, sized_stake=20.0,
    )

    assert result == 0.0
    c["alerter"]._enqueue.assert_not_called()


@pytest.mark.asyncio
async def test_no_side_uses_no_token_depth_lookup():
    """For NO trades, depth must be queried on the NO token (token_ids[1])
    at the effective NO price, not on the YES token's depth carried in
    by the caller."""
    # Effective NO price = 1 - 0.10 (yes_bid) = 0.90 → within [0.05, 0.95]
    result, c = await _invoke(
        decision=_no_decision(),
        yes_price=0.10,
        yes_bid=0.10,
        depth_no=500.0,  # generous so we pass the depth filter
    )

    assert result is not None and result > 0
    # NO-side depth was fetched via the orderbook for the NO token.
    c["get_orderbook_depth"].assert_called_once_with("no_token", 0.9)


@pytest.mark.asyncio
async def test_missing_token_ids_rejects_explicitly_and_never_probes_depth():
    """No token IDs → reject with a *named* reason, not a phantom depth failure.

    Regression for 2026-07-14. The unified pipeline ran the lock path whenever
    `price > 0` without requiring token_ids. With token_ids=None the executor
    fell through to `buy_depth = 0.0`, so the trade died on "depth $0 < $10" —
    while `effective_price` had been derived from `market.current_yes_price`,
    the *stale* Gamma snapshot. Gamma keeps quoting the pre-move price through
    the post-METAR repricing window, so a dying YES bucket still looked
    expensive and NO looked cheap: evaluation_logs filled with NO costs of
    0.11-0.93 that never existed on the book (the live CLOB was at 0.91+).
    Only the accidental depth-0 veto kept us from betting fictional prices.
    """
    result, c = await _invoke(
        decision=_no_decision(),
        yes_price=0.295,          # the STALE Gamma price
        yes_bid=None,             # no live quote — that's the whole point
        token_ids=None,
    )

    assert result == 0.0
    # Never probe a book we have no token for.
    c["get_orderbook_depth"].assert_not_called()
    # And never place an order off a stale price.
    c["place_order"].assert_not_called()

    reason = c["log_evaluation"].await_args.kwargs["reject_reason"]
    assert "token" in reason.lower()
    assert "depth" not in reason.lower()  # must not masquerade as a depth veto


@pytest.mark.asyncio
async def test_signal_row_records_lock_branch_and_observed_max():
    """The Signal row written on the happy path must carry the lock
    branch + observed_max so post-mortems can split realised P&L by
    which deterministic path produced the trade."""

    async def _fill(trade, session, **kwargs):
        trade.stake_usd = 20.0
        return True

    decision = _yes_decision(branch="easy_super", margin=5.0)
    decision = LockDecision(
        side="YES", branch="easy_super", margin_f=5.0,
        routine_count=4, observed_max_f=88.0,
        reasons=decision.reasons,
    )
    result, c = await _invoke(
        decision=decision, place_side_effect=_fill, sized_stake=20.0,
    )

    assert result == 20.0
    kwargs = c["upsert_signal"].await_args.kwargs
    assert kwargs["signal_kind"] == "lock"
    assert kwargs["lock_branch"] == "easy_super"
    assert kwargs["lock_routine_count"] == 4
    assert kwargs["lock_observed_max_f"] == 88.0
    # model_prob=1.0 is the lock-rule convention (deterministic decision).
    assert kwargs["model_prob"] == 1.0
    # lock_margin_f = °F margin from threshold ("how locked"); supersedes
    # the overloaded `confidence` column retired 2026-05-30.
    assert kwargs["lock_margin_f"] == 5.0


# ---------------------------------------------------------------------------
# Near-peak floor-up wiring — the gate result flows into size_locked_position
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_floor_up_passed_to_sizer_for_threshold_lock(monkeypatch):
    """Enabled + threshold op + near peak → floor_to_usd flows to the sizer."""
    monkeypatch.setattr(lre.settings, "NEAR_PEAK_FLOOR_UP_ENABLED", True)
    monkeypatch.setattr(lre.settings, "NEAR_PEAK_FLOOR_STAKE_USD", 6.0)
    # _market() default operator="above" (threshold), _state() hours_until_peak=2.0.
    _, c = await _invoke(decision=_yes_decision())
    assert c["size_locked_position"].call_args.kwargs["floor_to_usd"] == 6.0


@pytest.mark.asyncio
async def test_floor_up_excluded_for_bracket_like_lock(monkeypatch):
    """Enabled but bracket-like op (range) stays validate-first → no floor."""
    monkeypatch.setattr(lre.settings, "NEAR_PEAK_FLOOR_UP_ENABLED", True)
    market = _market(operator="range")
    _, c = await _invoke(decision=_no_decision(branch="range_in_window"), market=market)
    assert c["size_locked_position"].call_args.kwargs["floor_to_usd"] is None


# ---------------------------------------------------------------------------
# range_in_window (YES) min-price gate — cheap YES gambles are rejected
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_range_in_window_yes_below_min_price_rejected(monkeypatch):
    """YES range_in_window below RANGE_IN_WINDOW_MIN_YES_PRICE → no sizing."""
    monkeypatch.setattr(lre.settings, "RANGE_IN_WINDOW_MIN_YES_PRICE", 0.80)
    market = _market(operator="range")
    result, c = await _invoke(
        decision=_yes_decision(branch="range_in_window"),
        market=market,
        yes_price=0.40,
    )
    assert result == 0.0
    assert c["size_locked_position"].call_args is None


@pytest.mark.asyncio
async def test_range_in_window_yes_above_min_price_sizes(monkeypatch):
    """YES range_in_window at/above the min price still sizes normally."""
    monkeypatch.setattr(lre.settings, "RANGE_IN_WINDOW_MIN_YES_PRICE", 0.80)
    market = _market(operator="range")
    _, c = await _invoke(
        decision=_yes_decision(branch="range_in_window"),
        market=market,
        yes_price=0.90,
    )
    assert c["size_locked_position"].call_args is not None


# ---------------------------------------------------------------------------
# Conviction-weighted sizing wiring — only EASY-YES on a whitelisted station,
# only with the master flag on, gets the Kelly params + a widened walk budget.
# ---------------------------------------------------------------------------


def _enable_conviction(monkeypatch, *, stations: str = "KPHX"):
    monkeypatch.setattr(lre.settings, "LOCK_CONVICTION_SIZING_ENABLED", True)
    monkeypatch.setattr(lre.settings, "LOCK_BIG_SIZE_STATIONS", stations)
    monkeypatch.setattr(lre.settings, "LOCK_KELLY_FRACTION", 0.25)
    monkeypatch.setattr(lre.settings, "LOCK_WIN_PROB_SUPER", 0.99)
    monkeypatch.setattr(lre.settings, "LOCK_WIN_PROB_STANDARD", 0.97)
    monkeypatch.setattr(lre.settings, "LOCK_MAX_POSITION_PCT_SUPER", 0.15)
    monkeypatch.setattr(lre.settings, "LOCK_MAX_POSITION_PCT_STANDARD", 0.07)
    monkeypatch.setattr(lre.settings, "LOCK_WALK_MAX_PRICE", 0.95)
    monkeypatch.setattr(lre.settings, "LOCK_DEPTH_CAP_PCT_BIG", 0.50)


@pytest.mark.asyncio
async def test_conviction_super_passes_kelly_params_and_walks(monkeypatch):
    """EASY-YES super-margin on a whitelisted station → conviction Kelly
    params flow to the sizer (no floor_to_usd), and place_order gets a
    widened slippage budget to walk the book up to LOCK_WALK_MAX_PRICE."""
    _enable_conviction(monkeypatch)
    _, c = await _invoke(decision=_yes_decision(branch="easy_super"), yes_price=0.80)

    kwargs = c["size_locked_position"].call_args.kwargs
    assert kwargs["win_prob"] == 0.99
    assert kwargs["max_position_pct"] == 0.15
    assert kwargs["kelly_fraction"] == 0.25
    assert kwargs["depth_cap_pct"] == 0.50
    assert "floor_to_usd" not in kwargs  # conviction path bypasses floor-up
    # Walk budget = (0.95 - 0.80) × 100 = 15¢, passed as a keyword.
    assert c["place_order"].call_args.kwargs["max_slippage_cents"] == pytest.approx(15.0)


@pytest.mark.asyncio
async def test_conviction_standard_uses_standard_params(monkeypatch):
    """easy_standard branch draws the standard win-prob + concentration cap."""
    _enable_conviction(monkeypatch)
    _, c = await _invoke(decision=_yes_decision(branch="easy_standard"))

    kwargs = c["size_locked_position"].call_args.kwargs
    assert kwargs["win_prob"] == 0.97
    assert kwargs["max_position_pct"] == 0.07


@pytest.mark.asyncio
async def test_conviction_skipped_off_whitelist(monkeypatch):
    """Flag on but station not whitelisted → flat path (floor_to_usd present,
    no win_prob), and the default 2¢ slippage."""
    _enable_conviction(monkeypatch, stations="KJFK")  # not KPHX
    _, c = await _invoke(decision=_yes_decision(branch="easy_super"), yes_price=0.80)

    kwargs = c["size_locked_position"].call_args.kwargs
    assert kwargs.get("win_prob") is None
    assert "floor_to_usd" in kwargs
    # Flat path omits the walk kwarg → place_order's 2¢ default stands.
    assert "max_slippage_cents" not in c["place_order"].call_args.kwargs


@pytest.mark.asyncio
async def test_conviction_skipped_when_flag_off(monkeypatch):
    """Whitelisted but master flag off (default) → flat path."""
    monkeypatch.setattr(lre.settings, "LOCK_BIG_SIZE_STATIONS", "KPHX")
    # LOCK_CONVICTION_SIZING_ENABLED left at its default (False).
    _, c = await _invoke(decision=_yes_decision(branch="easy_super"))
    assert c["size_locked_position"].call_args.kwargs.get("win_prob") is None


@pytest.mark.asyncio
async def test_conviction_skipped_for_no_side(monkeypatch):
    """Only the monotonic YES direction is upsized; a NO lock stays flat."""
    _enable_conviction(monkeypatch)
    _, c = await _invoke(decision=_no_decision(branch="easy_super"), yes_bid=0.10)
    assert c["size_locked_position"].call_args.kwargs.get("win_prob") is None


@pytest.mark.asyncio
async def test_conviction_skipped_for_hard_branch(monkeypatch):
    """HARD (forecast-driven, not pure monotonicity) stays flat."""
    _enable_conviction(monkeypatch)
    _, c = await _invoke(decision=_yes_decision(branch="hard"))
    assert c["size_locked_position"].call_args.kwargs.get("win_prob") is None


# ---------------------------------------------------------------------------
# bucket_overshoot max-cost gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bucket_overshoot_rejected_above_max_cost(monkeypatch):
    """The dead-bucket rule is certain, but paying 0.95 for it is not +EV.

    Break-even at cost c needs a resolver-divergence loss rate below (1-c).
    Above BUCKET_OVERSHOOT_MAX_COST the market has already repriced. This is
    precisely the mistake the old `range_overshoot` branch made — it bought NO
    at an average 0.901 and its edge vanished.
    """
    from src.config import settings as _s

    monkeypatch.setattr(_s, "BUCKET_OVERSHOOT_MAX_COST", 0.93)
    # yes_bid=0.05 → NO buyer pays 0.95, inside LOCK_RULE_MAX_PRICE but above
    # the bucket_overshoot cap.
    result, c = await _invoke(
        decision=_no_decision(branch="bucket_overshoot"),
        yes_price=0.50,
        yes_bid=0.05,
        yes_ask=0.55,
    )

    assert result == 0.0
    reason = c["log_evaluation"].await_args.kwargs["reject_reason"]
    assert "bucket_overshoot cost" in reason
    c["upsert_signal"].assert_not_called()


@pytest.mark.asyncio
async def test_bucket_overshoot_allowed_below_max_cost(monkeypatch):
    """A cheap NO on a dead bucket is the whole strategy — it must get through."""
    from src.config import settings as _s

    monkeypatch.setattr(_s, "BUCKET_OVERSHOOT_MAX_COST", 0.93)
    # yes_bid=0.30 → NO buyer pays 0.70, well under the cap.
    result, c = await _invoke(
        decision=_no_decision(branch="bucket_overshoot"),
        yes_price=0.50,
        yes_bid=0.30,
        yes_ask=0.35,
    )

    assert result != 0.0
    c["upsert_signal"].assert_called()


@pytest.mark.asyncio
async def test_max_cost_gate_does_not_touch_other_branches(monkeypatch):
    """The cap is scoped to bucket_overshoot; EASY locks keep LOCK_RULE_MAX_PRICE."""
    from src.config import settings as _s

    monkeypatch.setattr(_s, "BUCKET_OVERSHOOT_MAX_COST", 0.50)
    result, c = await _invoke(
        decision=_no_decision(branch="easy_super"),
        yes_price=0.50,
        yes_bid=0.30,
        yes_ask=0.35,
    )
    assert result != 0.0
