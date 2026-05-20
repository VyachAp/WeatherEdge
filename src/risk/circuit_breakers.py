"""Circuit breakers for risk management.

Checks daily P&L and consecutive loss streaks to halt or pause trading.
Per-city checks (routine METAR count, bias runaway) are handled at the
edge-calculator and state-aggregator level, not here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import and_, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import ProgrammingError

from src.config import settings
from src.db.models import BotState, Trade, TradeStatus

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Persisted in ``bot_state`` so the protective window survives a process
# restart. In-process variable acts as a cache: ``None`` means "not yet
# hydrated from DB this process" — the first call to
# ``check_circuit_breakers`` after boot reads from DB once, then in-memory
# until set/clear writes back through.
_PAUSED_UNTIL_KEY = "circuit_breakers.paused_until"
_paused_until: datetime | None = None
_hydrated_from_db: bool = False


async def _load_paused_until(session: AsyncSession) -> datetime | None:
    # Fail-open if the bot_state migration hasn't been applied yet: missing
    # table means there's no persisted pause to honor, so trading proceeds
    # under the in-process default. The alternative — bubbling the
    # ProgrammingError up — silently kills every pipeline tick.
    try:
        row = await session.get(BotState, _PAUSED_UNTIL_KEY)
    except ProgrammingError:
        await session.rollback()
        logger.warning(
            "bot_state table missing — circuit-breaker pause not persisted. "
            "Run `alembic upgrade head` to enable cross-restart pauses."
        )
        return None
    if row is None or row.value is None:
        return None
    iso = row.value.get("until_iso") if isinstance(row.value, dict) else None
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso)
    except ValueError:
        logger.warning("Invalid paused_until value in bot_state: %r", row.value)
        return None


async def _persist_paused_until(session: AsyncSession, until: datetime | None) -> None:
    try:
        if until is None:
            existing = await session.get(BotState, _PAUSED_UNTIL_KEY)
            if existing is not None:
                await session.delete(existing)
            return
        stmt = (
            pg_insert(BotState)
            .values(
                key=_PAUSED_UNTIL_KEY,
                value={"until_iso": until.isoformat()},
                updated_at=datetime.now(timezone.utc),
            )
            .on_conflict_do_update(
                index_elements=["key"],
                set_={
                    "value": {"until_iso": until.isoformat()},
                    "updated_at": datetime.now(timezone.utc),
                },
            )
        )
        await session.execute(stmt)
    except ProgrammingError:
        await session.rollback()
        logger.warning("bot_state table missing — pause not persisted across restarts.")


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
    global _paused_until, _hydrated_from_db

    # First call after process boot: hydrate from bot_state so a restart
    # mid-pause doesn't silently drop the protective window.
    if not _hydrated_from_db:
        _paused_until = await _load_paused_until(session)
        _hydrated_from_db = True

    # Check if we're still in a consecutive-loss pause
    now = datetime.now(timezone.utc)
    if _paused_until is not None and now < _paused_until:
        remaining = (_paused_until - now).total_seconds() / 60
        return CircuitBreakerState(
            can_trade=False,
            paused_until=_paused_until,
            reason=f"consecutive loss pause ({remaining:.0f}m remaining)",
        )
    if _paused_until is not None:
        # Pause expired — clear both in-process and DB so a future restart
        # doesn't see a stale "active pause" row.
        _paused_until = None
        await _persist_paused_until(session, None)

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
        await _persist_paused_until(session, _paused_until)
        return CircuitBreakerState(
            can_trade=False,
            paused_until=_paused_until,
            reason=f"{consecutive} consecutive losses — pausing {settings.CONSECUTIVE_LOSS_PAUSE_HOURS}h",
        )

    # 3. Submission-failure stop (2026-05-20). Many ``exception:*`` rows
    # in the last window means the CLOB / SDK is rejecting orders en
    # masse — usually a Polymarket protocol migration or wallet
    # allowance issue. Pause new submissions and Telegram-alert so the
    # operator can react before the failure mode buries the decision
    # logs in noise.
    submit_fails = await _get_recent_submit_failures(session)
    if submit_fails >= settings.SUBMIT_FAIL_PAUSE_COUNT:
        _paused_until = now + timedelta(
            minutes=settings.SUBMIT_FAIL_PAUSE_MINUTES
        )
        await _persist_paused_until(session, _paused_until)
        await _maybe_alert_submit_failures(submit_fails)
        return CircuitBreakerState(
            can_trade=False,
            paused_until=_paused_until,
            reason=(
                f"{submit_fails} submission failures in last "
                f"{settings.SUBMIT_FAIL_PAUSE_WINDOW_MINUTES}m — "
                f"pausing {settings.SUBMIT_FAIL_PAUSE_MINUTES}m"
            ),
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


async def _get_recent_submit_failures(session: AsyncSession) -> int:
    """Count trades whose order submission raised an exception within
    the recent window (``SUBMIT_FAIL_PAUSE_WINDOW_MINUTES``).

    ``polymarket_client.place_order`` sets ``exchange_status`` to
    ``f"exception:{type(exc).__name__}"`` (e.g. ``exception:
    PolyApiException``) when the SDK / CLOB raises. Counting rows
    opened in the recent window with that prefix lets us detect a
    burst of submission failures (Polymarket protocol migration,
    wallet allowance drop, etc.) before they bury the decision logs.
    """
    window_start = datetime.now(timezone.utc) - timedelta(
        minutes=settings.SUBMIT_FAIL_PAUSE_WINDOW_MINUTES
    )
    stmt = select(func.count()).select_from(Trade).where(
        and_(
            Trade.opened_at >= window_start,
            Trade.exchange_status.startswith("exception"),
        )
    )
    result = await session.execute(stmt)
    return int(result.scalar() or 0)


_submit_fail_alerted_at: datetime | None = None


async def _maybe_alert_submit_failures(count: int) -> None:
    """One-shot-per-window Telegram alert for the submission-failure
    circuit breaker. Re-arms after ``SUBMIT_FAIL_PAUSE_MINUTES`` so the
    alert isn't a one-time-ever event (rebooting the bot would clear
    in-memory state anyway)."""
    global _submit_fail_alerted_at

    now = datetime.now(timezone.utc)
    if _submit_fail_alerted_at is not None and (
        now - _submit_fail_alerted_at
        < timedelta(minutes=settings.SUBMIT_FAIL_PAUSE_MINUTES)
    ):
        return
    _submit_fail_alerted_at = now

    try:
        from src.execution.alerter import get_alerter

        alerter = get_alerter()
        await alerter._enqueue(
            f"\U0001f6a8 *Submission failure circuit breaker tripped*\n"
            f"{count} CLOB submissions raised exceptions in the last "
            f"{settings.SUBMIT_FAIL_PAUSE_WINDOW_MINUTES}m. Trading "
            f"paused for {settings.SUBMIT_FAIL_PAUSE_MINUTES}m. Check "
            f"`trades.exchange_error` for the SDK / CLOB response — "
            f"likely a Polymarket protocol migration or wallet "
            f"allowance issue."
        )
    except Exception:
        logger.warning(
            "Failed to send submit-failure circuit-breaker alert",
            exc_info=True,
        )
