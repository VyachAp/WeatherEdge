"""Circuit breakers for risk management.

Checks daily P&L and consecutive loss streaks to halt or pause trading.
Per-city checks (routine METAR count, bias runaway) are handled at the
edge-calculator and state-aggregator level, not here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select

from src.config import settings
from src.db.models import Trade, TradeStatus

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Persistent pause state (resets on process restart, which is acceptable
# since consecutive losses are re-queried from DB each check).
_paused_until: datetime | None = None


@dataclass
class CircuitBreakerState:
    """Result of circuit breaker evaluation."""

    can_trade: bool
    paused_until: datetime | None
    reason: str | None


async def check_circuit_breakers(session: AsyncSession) -> CircuitBreakerState:
    """Evaluate all circuit breaker conditions.

    Returns a state indicating whether trading is allowed.
    """
    global _paused_until

    # Check if we're still in a consecutive-loss pause
    now = datetime.now(timezone.utc)
    if _paused_until is not None and now < _paused_until:
        remaining = (_paused_until - now).total_seconds() / 60
        return CircuitBreakerState(
            can_trade=False,
            paused_until=_paused_until,
            reason=f"consecutive loss pause ({remaining:.0f}m remaining)",
        )
    _paused_until = None

    # 1. Daily loss stop
    daily_pnl = await _get_daily_pnl(session)
    if daily_pnl < -settings.DAILY_LOSS_STOP_USD:
        return CircuitBreakerState(
            can_trade=False,
            paused_until=None,
            reason=f"daily loss stop: ${daily_pnl:.2f} (limit -${settings.DAILY_LOSS_STOP_USD:.0f})",
        )

    # 2. Consecutive loss stop
    consecutive = await _get_consecutive_losses(session)
    if consecutive >= settings.CONSECUTIVE_LOSS_PAUSE_COUNT:
        _paused_until = now + timedelta(hours=settings.CONSECUTIVE_LOSS_PAUSE_HOURS)
        return CircuitBreakerState(
            can_trade=False,
            paused_until=_paused_until,
            reason=f"{consecutive} consecutive losses — pausing {settings.CONSECUTIVE_LOSS_PAUSE_HOURS}h",
        )

    return CircuitBreakerState(can_trade=True, paused_until=None, reason=None)


async def _get_daily_pnl(session: AsyncSession) -> float:
    """Sum P&L of trades closed today (UTC)."""
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    stmt = select(func.coalesce(func.sum(Trade.pnl), 0.0)).where(
        Trade.closed_at >= today_start,
        Trade.status.in_([TradeStatus.WON, TradeStatus.LOST]),
    )
    result = await session.execute(stmt)
    return float(result.scalar())


async def _get_consecutive_losses(session: AsyncSession) -> int:
    """Count consecutive LOST trades from the most recent backward."""
    stmt = (
        select(Trade.status)
        .where(Trade.status.in_([TradeStatus.WON, TradeStatus.LOST]))
        .order_by(Trade.closed_at.desc())
        .limit(settings.CONSECUTIVE_LOSS_PAUSE_COUNT + 5)
    )
    result = await session.execute(stmt)
    statuses = [row[0] for row in result.all()]

    count = 0
    for status in statuses:
        if status == TradeStatus.LOST:
            count += 1
        else:
            break
    return count
