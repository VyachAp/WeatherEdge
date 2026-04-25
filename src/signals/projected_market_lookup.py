"""Reverse-lookup the Polymarket binary market matching a projected daily max.

Used by the forecast-exceedance Telegram alert to enrich the message with the
live YES price for the binary "≥X°F today" market whose threshold is closest
to the projected daily max.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.db.engine import async_session
from src.db.models import Market
from src.execution.polymarket_client import get_best_bid_ask, get_token_ids
from src.signals.reverse_lookup import find_markets_for_station

logger = logging.getLogger(__name__)

_BINARY_OPS = {"above", "at_least", "exceed", "below", "at_most"}


async def lookup_projected_binary(
    icao: str,
    projected_max_f: float,
) -> tuple[Market, float, str, float] | None:
    """Return (market, threshold_f, operator, yes_price) for the binary market
    resolving today at *icao* whose threshold is closest to *projected_max_f*.

    *yes_price* is the live best ask from the CLOB orderbook (the price you'd
    pay to buy YES). Falls back to ``market.current_yes_price`` if the CLOB
    call fails. Returns None when no qualifying market exists or the lookup
    raises — the alert is sent without a Polymarket line in that case.
    """
    try:
        async with async_session() as session:
            markets = await find_markets_for_station(
                session, icao, variable="temperature", hours_ahead=24,
            )

            today = datetime.now(timezone.utc).date()
            candidates: list[Market] = [
                m for m in markets
                if m.parsed_threshold is not None
                and (m.parsed_operator or "").lower() in _BINARY_OPS
                and m.end_date is not None
                and m.end_date.date() == today
            ]
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
