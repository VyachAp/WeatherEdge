"""Per-tick exposure / headroom snapshot telemetry (M4) — Phase 0.

One ``ExposureSnapshot`` row per ``job_unified_pipeline`` tick (~288/day).
Turns the one-off 2026-05-23 capital diagnosis ("exposure cap saturated,
~$25-32 room per tick") into a standing timeseries, so the Phase-0 gate —
does raising ``MAX_EXPOSURE_USD_FLOOR`` / growing bankroll actually open
per-tick headroom? — is read from data, not eyeballed once.

The effective cap reuses ``risk.kelly._effective_exposure_cap`` so the
recorded ceiling always matches the one the sizer actually applies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import func, select

from src.db.models import ExposureSnapshot, Market, Trade, TradeStatus
from src.execution.binary_market import operator_class
from src.risk.kelly import _effective_exposure_cap

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def record_exposure_snapshot(
    session: "AsyncSession",
    *,
    equity: float,
    exposure: float,
    dd_level: str | None = None,
    dd_mult: float | None = None,
    n_candidates: int | None = None,
    n_sized_to_zero: int | None = None,
) -> None:
    """Append one ``ExposureSnapshot`` row. Best-effort: never raises.

    ``equity`` / ``exposure`` are the top-of-tick values the sizer saw.
    ``n_open`` and the per-operator-class breakdown come from one grouped
    query over OPEN trades. Does not commit — the caller batches it.
    """
    try:
        effective_cap = _effective_exposure_cap(equity)
        headroom = max(0.0, effective_cap - exposure)

        rows = (
            await session.execute(
                select(Market.parsed_operator, func.count(Trade.id))
                .join(Market, Trade.market_id == Market.id)
                .where(Trade.status == TradeStatus.OPEN)
                .group_by(Market.parsed_operator)
            )
        ).all()
        n_open = 0
        by_class: dict[str, int] = {}
        for op, cnt in rows:
            cnt = int(cnt)
            n_open += cnt
            cls = operator_class(op) or "unknown"
            by_class[cls] = by_class.get(cls, 0) + cnt

        session.add(
            ExposureSnapshot(
                equity=float(equity),
                exposure=float(exposure),
                effective_cap=float(effective_cap),
                headroom=float(headroom),
                n_open=n_open,
                n_open_by_class=by_class,
                dd_level=dd_level,
                dd_mult=dd_mult,
                n_candidates=n_candidates,
                n_sized_to_zero=n_sized_to_zero,
            )
        )
    except Exception:
        return
