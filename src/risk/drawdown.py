"""Drawdown protection with state-machine tracking."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from src.db.models import BankrollLog

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CAUTION_THRESHOLD: float = 0.10
PAUSE_THRESHOLD: float = 0.20

# ---------------------------------------------------------------------------
# Enum & dataclass
# ---------------------------------------------------------------------------


class DrawdownLevel(str, enum.Enum):
    NORMAL = "normal"
    CAUTION = "caution"
    PAUSED = "paused"
    RECOVERY = "recovery"


_ACTIONS = {
    DrawdownLevel.NORMAL: "full size",
    DrawdownLevel.CAUTION: "reduce to half size",
    DrawdownLevel.PAUSED: "no new trades",
    DrawdownLevel.RECOVERY: "half size (recovering)",
}

_MULTIPLIERS = {
    DrawdownLevel.NORMAL: 1.0,
    DrawdownLevel.CAUTION: 0.5,
    DrawdownLevel.PAUSED: 0.0,
    DrawdownLevel.RECOVERY: 0.5,
}


@dataclass
class DrawdownState:
    """Snapshot of drawdown status."""

    level: DrawdownLevel
    drawdown_pct: float
    peak: float
    current: float
    size_multiplier: float
    action: str


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


class DrawdownMonitor:
    """Tracks bankroll peak/drawdown and enforces position-size limits."""

    def __init__(self, initial_bankroll: float) -> None:
        self._peak = initial_bankroll
        self._current = initial_bankroll
        self._level = DrawdownLevel.NORMAL

    # -- Pure computation (no side effects) ---------------------------------

    def check(self, current_bankroll: float) -> DrawdownState:
        """Return the drawdown state for *current_bankroll* without mutating."""
        peak = max(self._peak, current_bankroll)
        dd_pct = (peak - current_bankroll) / peak if peak > 0 else 0.0

        if current_bankroll >= self._peak:
            level = DrawdownLevel.NORMAL
        elif dd_pct < CAUTION_THRESHOLD:
            if self._level in (
                DrawdownLevel.CAUTION,
                DrawdownLevel.PAUSED,
                DrawdownLevel.RECOVERY,
            ):
                level = DrawdownLevel.RECOVERY
            else:
                level = DrawdownLevel.NORMAL
        elif dd_pct <= PAUSE_THRESHOLD:
            level = DrawdownLevel.CAUTION
        else:
            level = DrawdownLevel.PAUSED

        return DrawdownState(
            level=level,
            drawdown_pct=round(dd_pct, 6),
            peak=peak,
            current=current_bankroll,
            size_multiplier=_MULTIPLIERS[level],
            action=_ACTIONS[level],
        )

    # -- State mutation (sync, no DB) ---------------------------------------

    def advance(self, current_bankroll: float) -> DrawdownState:
        """Update internal state and return the new :class:`DrawdownState`."""
        state = self.check(current_bankroll)
        self._peak = state.peak
        self._current = current_bankroll
        self._level = state.level
        return state

    # -- Async DB operations ------------------------------------------------

    async def update(
        self,
        current_bankroll: float,
        session: AsyncSession,
    ) -> DrawdownState:
        """Advance state and persist a :class:`BankrollLog` row."""
        state = self.advance(current_bankroll)
        row = BankrollLog(
            balance=state.current,
            peak=state.peak,
            drawdown_pct=state.drawdown_pct,
            timestamp=datetime.now(timezone.utc),
        )
        session.add(row)
        return state

    async def load_state(self, session: AsyncSession) -> DrawdownState:
        """Restore internal state from the most recent :class:`BankrollLog`."""
        from sqlalchemy import select

        stmt = (
            select(BankrollLog)
            .order_by(BankrollLog.timestamp.desc())
            .limit(1)
        )
        result = await session.execute(stmt)
        row = result.scalar_one_or_none()

        if row is not None:
            self._peak = row.peak
            self._current = row.balance
            # Derive level from stored values
            state = self.check(self._current)
            self._level = state.level
            return state

        return self.check(self._current)
