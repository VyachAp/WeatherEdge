"""Cluster stake cap for bracket / range / exactly markets.

A "cluster" is the set of bucket markets with the same ``parsed_location``
and the same UTC ``end_date.date()``. For a multi-bucket question only
one bucket can ultimately resolve YES, so each independent Kelly bucket
bet over-states diversification — the cluster's combined exposure is the
right unit for the cap. ``settings.CLUSTER_STAKE_CAP_USD`` (default $100,
half of ``MAX_POSITION_USD``) enforces it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import func, select

from src.db.models import Market, Trade, TradeStatus
from src.execution.binary_market import is_bracket_like

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def cluster_stake_used(
    session: "AsyncSession", market,
) -> float:
    """Sum of currently-staked $ across the same bracket/exactly cluster.

    Excludes dry-run rows and resolved (WON/LOST) rows; includes PENDING
    + OPEN. Returns 0.0 for non-bracket-like markets so threshold
    markets aren't accidentally clustered.
    """
    if not is_bracket_like(market) or not market.parsed_location or not market.end_date:
        return 0.0

    end_day = market.end_date.date()
    stmt = (
        select(func.coalesce(func.sum(Trade.stake_usd), 0.0))
        .join(Market, Trade.market_id == Market.id)
        .where(
            Market.parsed_location == market.parsed_location,
            func.date(Market.end_date) == end_day,
            Market.parsed_operator.in_(("bracket", "range", "exactly")),
            Trade.status.in_([TradeStatus.PENDING, TradeStatus.OPEN]),
            # Live trades only — dry-run rows would over-state the cap.
            (Trade.exchange_status.is_(None) | (Trade.exchange_status != "dry_run")),
        )
    )
    result = await session.execute(stmt)
    return float(result.scalar() or 0.0)
