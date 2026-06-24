"""APScheduler-based pipeline orchestration and health-check server.

Phase 2b split this former 2165-line module into focused submodules:

* ``runtime``  — drawdown-monitor singleton + scheduler-running accessor (leaf)
* ``jobs``     — all ``job_*`` functions, ``backfill_markets``,
                 ``start_health_server``, and scheduler-private helpers
* ``setup``    — ``setup_scheduler`` / ``run_scheduler`` + ``configure_logging``

This package module is a thin facade that re-exports the public API plus the
scheduler's OWN tested private helpers (``_detect_calibration_squash``,
``_load_stuck_alert_cooldown``, ``_persist_stuck_alert_cooldown``,
``_fast_poll_projection_check``) for back-compat. It also re-exports the
per-station cache-rollover state that ``tests/test_scheduler.py`` reaches via
``src.scheduler``. The foreign ``_X`` aliases that used to live here (e.g.
``_binary_market_edge``) are NOT re-exported — import them from their canonical
home modules.
"""

from __future__ import annotations

# Public API
from src.scheduler.jobs import backfill_markets, start_health_server
from src.scheduler.setup import configure_logging, run_scheduler

# Scheduler-OWN private helpers tested via `from src.scheduler import ...`
from src.scheduler.jobs import (
    _detect_calibration_squash,
    _fast_poll_projection_check,
    _load_stuck_alert_cooldown,
    _persist_stuck_alert_cooldown,
)

# Per-station cache-rollover state reached by tests via `src.scheduler.<name>`
# (the dedup dicts + the rollover helper). These are re-exports of
# ``persistence.cache_rollover`` — not scheduler-private — but kept on this
# namespace for back-compat with the existing scheduler tests.
from src.persistence.cache_rollover import (  # noqa: F401
    last_routine_seen as _last_routine_seen,
    local_day_seen as _local_day_seen,
    locked_markets_fired_today as _locked_markets_fired_today,
    market_to_icao as _market_to_icao,
    maybe_clear_per_station_caches as _maybe_clear_per_station_caches,
    unified_fired_today as _unified_fired_today,
)

__all__ = [
    "run_scheduler",
    "configure_logging",
    "backfill_markets",
    "start_health_server",
]
