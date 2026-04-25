"""Reverse-lookup the Polymarket binary market matching a projected daily max.

Used by the forecast-exceedance Telegram alert to enrich the message with the
live YES price for the binary "≥X°F today" market whose threshold is closest
to the projected daily max.

Only operator-aligned markets are returned: for ``above``/``at_least``/``exceed``
operators the threshold must sit at or below the projection (so YES would
resolve TRUE under our model); for ``below``/``at_most`` it must sit at or
above. Markets whose threshold is farther than ``MAX_THRESHOLD_DISTANCE_F``
(≈2°C) from the projection are also dropped — at that distance the alert
isn't actionable and the line just adds noise.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.db.engine import async_session
from src.db.models import Market
from src.execution.polymarket_client import get_best_bid_ask, get_token_ids
from src.signals.reverse_lookup import find_markets_for_station

logger = logging.getLogger(__name__)

_ABOVE_OPS = {"above", "at_least", "exceed"}
_BELOW_OPS = {"below", "at_most"}
_BINARY_OPS = _ABOVE_OPS | _BELOW_OPS

MAX_THRESHOLD_DISTANCE_F = 3.6  # ≈2°C plausibility band around the projection


def _is_operator_aligned(operator: str, threshold_f: float, projected_f: float) -> bool:
    """True when buying YES on this market is consistent with our projection.

    For ``above``/``at_least``/``exceed``: projection must reach or exceed the
    threshold (YES resolves TRUE). For ``below``/``at_most``: projection must
    sit at or below the threshold.
    """
    if operator in _ABOVE_OPS:
        return projected_f >= threshold_f
    if operator in _BELOW_OPS:
        return projected_f <= threshold_f
    return False


async def lookup_projected_binary(
    icao: str,
    projected_max_f: float,
) -> tuple[Market, float, str, float] | None:
    """Return (market, threshold_f, operator, yes_price) for the binary market
    resolving today at *icao* whose threshold is closest to *projected_max_f*
    *and* whose operator direction is consistent with that projection.

    *yes_price* is the live best ask from the CLOB orderbook (the price you'd
    pay to buy YES). Falls back to ``market.current_yes_price`` if the CLOB
    call fails. Returns None when no operator-aligned market sits within
    ``MAX_THRESHOLD_DISTANCE_F`` of the projection or the lookup raises —
    the alert is sent without a Polymarket line in that case.
    """
    try:
        async with async_session() as session:
            markets = await find_markets_for_station(
                session, icao, variable="temperature", hours_ahead=24,
            )

            today = datetime.now(timezone.utc).date()
            candidates: list[Market] = []
            for m in markets:
                if m.parsed_threshold is None or m.end_date is None:
                    continue
                if m.end_date.date() != today:
                    continue
                operator = (m.parsed_operator or "").lower()
                if operator not in _BINARY_OPS:
                    continue
                threshold = float(m.parsed_threshold)
                if not _is_operator_aligned(operator, threshold, projected_max_f):
                    continue
                if abs(threshold - projected_max_f) > MAX_THRESHOLD_DISTANCE_F:
                    continue
                candidates.append(m)
            if not candidates:
                return None

            candidates.sort(
                key=lambda m: (
                    abs((m.parsed_threshold or 0.0) - projected_max_f),
                    -(m.liquidity or 0.0),
                )
            )
            best = candidates[0]
            threshold = float(best.parsed_threshold or 0.0)
            operator = (best.parsed_operator or "").lower()

        token_ids = await get_token_ids(best.id)
        yes_price: float | None = None
        if token_ids is not None:
            quote = get_best_bid_ask(token_ids[0])
            if quote is not None:
                _, yes_price = quote  # (best_bid, best_ask) — pay the ask
        if yes_price is None:
            yes_price = best.current_yes_price
        if yes_price is None:
            return None

        return best, threshold, operator, float(yes_price)
    except Exception:
        logger.warning("projected-market lookup failed for %s", icao, exc_info=True)
        return None
