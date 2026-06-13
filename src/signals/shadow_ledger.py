"""Writer + glue for the four-stage shadow decision ledger.

Bridges the live pipeline to the pure ``shadow_decision`` module: builds a
``MarketSpec`` from a Market row, runs all four stages, and appends a
``ShadowLedger`` row capturing the full decision + feature snapshot. Places NO
orders and never mutates the Market row (read-only — see the
``Market.current_yes_price`` autoflush gotcha).

Best-effort by contract: ``record_shadow_decision`` swallows its own errors and
logs a warning, so a shadow-ledger failure can never abort or alter the live
trading paths it runs alongside.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import TYPE_CHECKING

from src.db.models import ShadowLedger
from src.execution.binary_market import market_range_f
from src.signals.shadow_decision import (
    DEFAULT_PARAMS,
    MarketSpec,
    ShadowParams,
    ShadowResult,
    evaluate_shadow,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.signals.state_aggregator import WeatherState

logger = logging.getLogger(__name__)

_THRESHOLD_OPS = frozenset({"above", "at_least", "below", "at_most"})
_BRACKET_LIKE_OPS = frozenset({"bracket", "range", "exactly"})


def build_market_spec(market) -> MarketSpec | None:
    """Translate a Market row into the °F outcome spec the shadow model needs.

    Returns None for unparsed / non-binary markets the model can't score.
    ``parsed_threshold`` and the ``market_range_f`` bounds are already in °F
    (the same frame the legacy edge calculator's buckets use).
    """
    op = market.parsed_operator
    if op in _THRESHOLD_OPS:
        if market.parsed_threshold is None:
            return None
        return MarketSpec(operator=op, threshold_f=float(market.parsed_threshold))
    if op in _BRACKET_LIKE_OPS:
        rng = market_range_f(market)
        if rng is None:
            return None
        return MarketSpec(operator=op, low_f=float(rng[0]), high_f=float(rng[1]))
    return None


def _feature_snapshot(state: "WeatherState", result: ShadowResult) -> dict:
    """JSON-serialisable snapshot of the inputs + reasoning behind the decision.

    This is the "why" the legacy ``evaluation_logs`` never persisted: the full
    WeatherState the model consumed plus the per-stage reasoning trail, so any
    past shadow decision is fully reconstructable for calibration analysis.
    """
    snap = {k: v for k, v in asdict(state).items() if k != "routine_history"}
    snap["reasoning"] = list(result.estimate.reasoning)
    snap["decision_reason"] = result.decision.reason
    snap["size_reason"] = result.size.reason
    # routine_history is (datetime, float) tuples — store just the count.
    snap["routine_history_len"] = len(state.routine_history)
    return snap


async def record_shadow_decision(
    session: "AsyncSession",
    market,
    state: "WeatherState",
    *,
    yes_bid: float | None,
    yes_ask: float | None,
    yes_mid: float,
    depth_yes_usd: float | None,
    depth_no_usd: float | None,
    minutes_to_close: float | None,
    bankroll: float,
    params: ShadowParams = DEFAULT_PARAMS,
) -> ShadowResult | None:
    """Run the four-stage shadow model for one market and append a ledger row.

    Best-effort: returns the ``ShadowResult`` on success (for optional logging
    by the caller) or None if the market can't be scored or anything fails.
    Does not flush — the caller's surrounding flush batches the insert.
    """
    try:
        spec = build_market_spec(market)
        if spec is None:
            return None

        result = evaluate_shadow(
            state, spec,
            yes_bid=yes_bid, yes_ask=yes_ask, yes_mid=yes_mid,
            depth_yes_usd=depth_yes_usd, depth_no_usd=depth_no_usd,
            minutes_to_close=minutes_to_close, bankroll=bankroll, params=params,
        )

        side_depth = depth_yes_usd if result.decision.side == "YES" else depth_no_usd
        target_day = market.end_date.date() if market.end_date is not None else None

        session.add(ShadowLedger(
            market_id=market.id,
            station_icao=state.station_icao,
            target_date_local=target_day,
            prior_yes=result.estimate.prior_yes,
            updated_yes=result.estimate.updated_yes,
            info_advantage=result.estimate.info_advantage,
            obs_fraction=result.estimate.obs_fraction,
            mean_f=result.estimate.mean_f,
            prior_sigma_f=result.estimate.prior_sigma_f,
            updated_sigma_f=result.estimate.updated_sigma_f,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            market_mid=yes_mid,
            depth_usd=side_depth,
            minutes_to_close=minutes_to_close,
            side=result.decision.side,
            effective_cost=result.decision.cost.effective_cost,
            net_edge=result.decision.net_edge,
            sigma_edge=result.decision.sigma_edge,
            would_trade=result.decision.should_trade,
            shadow_stake_usd=result.size.stake_usd,
            decision_reason=result.decision.reason,
            feature_json=_feature_snapshot(state, result),
        ))
        return result
    except Exception:  # noqa: BLE001 — shadow must never break the live paths
        logger.warning("shadow ledger record failed for market %s",
                       getattr(market, "id", "?"), exc_info=True)
        return None
