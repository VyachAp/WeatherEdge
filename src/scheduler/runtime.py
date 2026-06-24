"""Scheduler runtime singleton state.

Leaf module: imports only from ``src.risk.*`` / ``src.resolution`` / ``src.db.*``
/ ``src.config`` so it can be imported by both ``setup`` and ``jobs`` without
creating an import cycle. Holds the drawdown-monitor singleton + its TTL reload
accessor — the only "outside" runtime state the jobs need.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from src.config import settings
from src.db.engine import async_session
from src.risk.drawdown import DrawdownMonitor

if TYPE_CHECKING:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler


def scheduler_is_running() -> bool:
    """Whether the live scheduler instance exists and is running.

    Shared accessor so ``jobs.start_health_server`` can report status without
    importing ``setup`` (which imports ``jobs`` — that would be a cycle).
    ``setup`` owns and assigns the singleton.
    """
    sched = _get_scheduler()
    return sched is not None and sched.running


_scheduler: "AsyncIOScheduler | None" = None


def _get_scheduler() -> "AsyncIOScheduler | None":
    return _scheduler


def _set_scheduler(sched: "AsyncIOScheduler | None") -> None:
    global _scheduler  # noqa: PLW0603
    _scheduler = sched


_drawdown_monitor: DrawdownMonitor | None = None
# Last time the monitor's persisted peak was reloaded from bankroll_log.
# Reloaded on a TTL so `admin reset-drawdown-peak` takes effect on the running
# process without a restart (the peak was previously loaded once at boot only).
_drawdown_peak_loaded_at: datetime | None = None
_DRAWDOWN_PEAK_RELOAD_TTL = timedelta(seconds=300)


async def _get_drawdown_monitor() -> DrawdownMonitor:
    global _drawdown_monitor, _drawdown_peak_loaded_at  # noqa: PLW0603
    now = datetime.now(timezone.utc)
    if _drawdown_monitor is None:
        _drawdown_monitor = DrawdownMonitor(settings.INITIAL_BANKROLL)
        async with async_session() as session:
            await _drawdown_monitor.load_state(session)
        _drawdown_peak_loaded_at = now
    elif (
        _drawdown_peak_loaded_at is None
        or now - _drawdown_peak_loaded_at > _DRAWDOWN_PEAK_RELOAD_TTL
    ):
        # Re-read the persisted peak so an operator's `admin
        # reset-drawdown-peak` (a newer bankroll_log row with peak=equity)
        # takes effect within the TTL — no scheduler restart needed. Safe to
        # pull in a *lower* peak: ``check()`` re-maxes with live equity, so a
        # reload can only relax an over-tight pause, never under-protect.
        async with async_session() as session:
            await _drawdown_monitor.load_state(session)
        _drawdown_peak_loaded_at = now
    return _drawdown_monitor
