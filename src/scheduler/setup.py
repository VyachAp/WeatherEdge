"""Scheduler construction + daemon entry point.

Owns the APScheduler instance + shutdown event, ``setup_scheduler`` (job
registration), and ``run_scheduler`` (the daemon main loop). Imports the
``job_*`` callables from ``.jobs``. ``configure_logging`` is re-exported here
from ``src.monitoring.logging`` to preserve the historical public-API surface
(``src.scheduler.configure_logging``).
"""

from __future__ import annotations

import asyncio
import functools
import logging
import signal
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from src.config import settings
from src.db.engine import engine
from src.execution.alerter import get_alerter
from src.monitoring.logging import configure_logging
from src.scheduler import runtime
from src.scheduler.jobs import (
    job_daily_settlement,
    job_fast_lock_poll,
    job_no_trade_review,
    job_perf_review,
    job_reconcile_orders,
    job_resolve_trades,
    job_scan_markets,
    job_startup,
    job_unified_pipeline,
    start_health_server,
)

logger = logging.getLogger(__name__)

_shutdown_event: asyncio.Event | None = None


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def setup_scheduler() -> AsyncIOScheduler:
    """Create and configure the APScheduler instance."""
    scheduler = AsyncIOScheduler(timezone="UTC")

    scheduler.add_job(
        job_scan_markets,
        IntervalTrigger(minutes=15),
        id="scan_markets",
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        job_daily_settlement,
        CronTrigger(hour=22, minute=0, timezone="UTC"),
        id="daily_settlement",
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        job_resolve_trades,
        IntervalTrigger(minutes=5),
        id="resolve_trades",
        next_run_time=datetime.now(timezone.utc) + timedelta(seconds=30),
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        job_unified_pipeline,
        IntervalTrigger(minutes=settings.UNIFIED_PIPELINE_INTERVAL_MINUTES),
        id="unified_pipeline",
        max_instances=1,
        coalesce=True,
    )
    logger.info(
        "Unified pipeline enabled (every %dm)",
        settings.UNIFIED_PIPELINE_INTERVAL_MINUTES,
    )

    if settings.LOCK_RULE_ENABLED and settings.FAST_LOCK_POLL_ENABLED:
        scheduler.add_job(
            job_fast_lock_poll,
            IntervalTrigger(seconds=settings.FAST_LOCK_POLL_INTERVAL_SECONDS),
            id="fast_lock_poll",
            max_instances=1,
            coalesce=True,
        )
        logger.info(
            "Fast lock poll enabled (every %ds)",
            settings.FAST_LOCK_POLL_INTERVAL_SECONDS,
        )

    if settings.ORDER_RECONCILE_INTERVAL_MINUTES > 0:
        scheduler.add_job(
            job_reconcile_orders,
            IntervalTrigger(minutes=settings.ORDER_RECONCILE_INTERVAL_MINUTES),
            id="reconcile_orders",
            next_run_time=datetime.now(timezone.utc) + timedelta(minutes=2),
            max_instances=1,
            coalesce=True,
        )
        logger.info(
            "Order reconciliation enabled (every %dm)",
            settings.ORDER_RECONCILE_INTERVAL_MINUTES,
        )

    if settings.PERF_REVIEW_ENABLED:
        # One parametrised job, three cadences. functools.partial (not a loop
        # closure) avoids APScheduler late-binding of `days`. Staggered after
        # the 22:00 settlement so the digest reflects the freshly-settled book.
        for _days, _trigger, _jid in (
            (1, CronTrigger(hour=22, minute=30, timezone="UTC"), "perf_review_daily"),
            (3, CronTrigger(day_of_week="mon,thu", hour=22, minute=40, timezone="UTC"), "perf_review_3d"),
            (7, CronTrigger(day_of_week="mon", hour=22, minute=50, timezone="UTC"), "perf_review_7d"),
        ):
            scheduler.add_job(
                functools.partial(job_perf_review, _days),
                _trigger,
                id=_jid,
                max_instances=1,
                coalesce=True,
            )
        logger.info("Perf-review jobs enabled (daily / 3d / 7d)")

    if settings.NO_TRADE_REVIEW_ENABLED:
        # Daily "why no trade" funnel, staggered after the perf-review daily.
        scheduler.add_job(
            functools.partial(job_no_trade_review, 1),
            CronTrigger(hour=22, minute=35, timezone="UTC"),
            id="no_trade_review_daily",
            max_instances=1,
            coalesce=True,
        )
        logger.info("No-trade-review job enabled (daily)")

    return scheduler


async def run_scheduler() -> None:
    """Start the full pipeline daemon."""
    global _shutdown_event  # noqa: PLW0603

    configure_logging()
    logger.info("Starting WeatherEdge scheduler")

    _shutdown_event = asyncio.Event()

    loop = asyncio.get_running_loop()
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, _shutdown_event.set)
    except NotImplementedError:
        # Some runtimes (e.g. DO App Platform) don't support signal handlers
        logger.warning("Signal handlers not supported; will run until cancelled")

    health_server = await start_health_server()

    # Ensure all tables exist (first deploy / fresh database)
    from src.db.models import Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    scheduler = setup_scheduler()
    runtime._set_scheduler(scheduler)
    scheduler.start()

    try:
        await job_startup()
    except Exception as exc:
        logger.exception("Startup job failed, continuing scheduler")
        await get_alerter().send_system_error(exc, "startup")

    await _shutdown_event.wait()

    # Graceful shutdown
    logger.info("Shutting down…")
    scheduler.shutdown(wait=True)
    await get_alerter().shutdown()
    health_server.close()
    await health_server.wait_closed()
    logger.info("Shutdown complete")
