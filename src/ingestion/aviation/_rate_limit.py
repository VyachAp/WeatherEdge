"""Per-provider async rate limiters for aviation data sources."""

from __future__ import annotations

import asyncio
import logging
import time as _time

logger = logging.getLogger(__name__)


class RateLimitExhausted(Exception):
    """Raised when a provider's daily budget is exhausted."""

    def __init__(self, provider: str) -> None:
        self.provider = provider
        super().__init__(f"Rate limit exhausted for provider: {provider}")


class ProviderRateLimiter:
    """Async token-bucket rate limiter with optional daily budget.

    Parameters
    ----------
    name:
        Provider name for logging.
    max_per_second:
        Maximum requests per second. Use fractions for slower rates
        (e.g. 0.2 = 1 request per 5 seconds).
    daily_budget:
        If set, hard cap on total requests per 24-hour rolling window.
        None means unlimited.
    """

    def __init__(
        self,
        name: str,
        max_per_second: float = 1.0,
        daily_budget: int | None = None,
    ) -> None:
        self._name = name
        self._interval = 1.0 / max_per_second
        self._semaphore = asyncio.Semaphore(max(1, int(max_per_second)))
        self._daily_budget = daily_budget
        self._daily_count = 0
        self._budget_reset_at = _time.monotonic() + 86400

    async def acquire(self) -> None:
        """Acquire a rate-limit slot. Raises RateLimitExhausted if daily budget spent."""
        if self._daily_budget is not None:
            now = _time.monotonic()
            if now > self._budget_reset_at:
                self._daily_count = 0
                self._budget_reset_at = now + 86400
            if self._daily_count >= self._daily_budget:
                raise RateLimitExhausted(self._name)
            self._daily_count += 1

        await self._semaphore.acquire()

        async def _release() -> None:
            await asyncio.sleep(self._interval)
            self._semaphore.release()

        asyncio.create_task(_release())

    @property
    def remaining_budget(self) -> int | None:
        """Return remaining daily budget, or None if unlimited."""
        if self._daily_budget is None:
            return None
        return max(0, self._daily_budget - self._daily_count)

    def reset_budget(self) -> None:
        """Manually reset the daily budget counter."""
        self._daily_count = 0
        self._budget_reset_at = _time.monotonic() + 86400
