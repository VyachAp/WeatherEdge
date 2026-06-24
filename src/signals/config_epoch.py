"""Config-epoch telemetry (M2) — Phase 0 measurement layer.

Writes one ``ConfigEpoch`` row at scheduler startup, but only when the
experiment-relevant settings differ from the latest stored epoch — so
each row marks a *real* config change. Telemetry rows
(``evaluation_logs`` / ``decision_logs`` / ``exposure_snapshots``) are
bucketed into an epoch at read time by
``created_at >= started_at`` of the latest epoch at or before them,
giving clean before/after-flip A/B without stamping a foreign key on the
hot-path tables.

Why this matters: the profitability roadmap flips dark features one at a
time and reads the effect from telemetry. Without an epoch marker the
"before" and "after" windows are guessed from wall-clock; with it they're
exact. See the roadmap's measurement layer (M2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from src.config import settings
from src.db.models import ConfigEpoch

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


# The flags / thresholds whose flips this roadmap gates. ``getattr`` with a
# ``None`` default so a renamed or not-yet-added setting records as ``None``
# rather than raising — this is telemetry, it must never break startup.
_TRACKED_FLAGS: tuple[str, ...] = (
    "PER_OPERATOR_CALIBRATION_ENABLED",
    "NEAR_PEAK_FLOOR_UP_ENABLED",
    "NEAR_PEAK_FLOOR_STAKE_USD",
    "THRESHOLD_MIN_PROBABILITY",
    "THRESHOLD_MIN_EDGE",
    "SIGMA_FLOOR_LEAD_TIME_ENABLED",
    "BRACKET_LIKE_NO_DISABLED",
    "CLIMATE_PRIOR_ENABLED",
    "BRACKET_MARKETS_ENABLED",
    "APPLY_CALIBRATION",
    "MIN_EDGE",
    "MIN_PROBABILITY",
    "MAX_EXPOSURE_USD_FLOOR",
    "DRAWDOWN_PAUSE_THRESHOLD",
    "DRAWDOWN_CAUTION_THRESHOLD",
    "MIN_STAKE_USD",
    "KELLY_FRACTION",
    "KELLY_PROB_CAP",
    "MAX_POSITION_USD",
    "DAILY_SPEND_CAP_USD",
    "AUTO_EXECUTE",
)


def current_flags() -> dict[str, Any]:
    """Snapshot the tracked settings into a JSON-serialisable dict."""
    return {name: getattr(settings, name, None) for name in _TRACKED_FLAGS}


async def record_config_epoch(session: "AsyncSession") -> int | None:
    """Insert a new epoch iff the tracked flags changed since the last one.

    Returns the epoch id (newly inserted, or the unchanged latest), or
    ``None`` on failure. Does not commit — the caller batches it.
    Best-effort: never raises.
    """
    try:
        flags = current_flags()
        latest = (
            await session.execute(
                select(ConfigEpoch).order_by(ConfigEpoch.id.desc()).limit(1)
            )
        ).scalars().first()
        if latest is not None and latest.flags_json == flags:
            return latest.id
        epoch = ConfigEpoch(flags_json=flags)
        session.add(epoch)
        await session.flush()
        return epoch.id
    except Exception:
        return None
