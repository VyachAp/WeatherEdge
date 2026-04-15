"""TTL cache for aviation weather data with per-data-type instances."""

from __future__ import annotations

import time as _time
from typing import Any, Hashable


class _TTLCache:
    """Simple in-memory TTL cache. Not thread-safe, but fine for asyncio."""

    def __init__(self, ttl_seconds: float = 300.0) -> None:
        self._ttl = ttl_seconds
        self._store: dict[Hashable, tuple[float, Any]] = {}

    def get(self, key: Hashable) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        ts, value = entry
        if _time.monotonic() - ts > self._ttl:
            del self._store[key]
            return None
        return value

    def set(self, key: Hashable, value: Any) -> None:
        self._store[key] = (_time.monotonic(), value)

    def clear(self) -> None:
        self._store.clear()


# Per-data-type cache instances
metar_cache = _TTLCache(ttl_seconds=300)        # 5 min — real-time needs fresh data
metar_hist_cache = _TTLCache(ttl_seconds=3600)   # 1 hour — historical doesn't change
taf_cache = _TTLCache(ttl_seconds=1800)          # 30 min — recheck on amendments
synop_cache = _TTLCache(ttl_seconds=10800)       # 3 hours — SYNOP updates every 3-6h
station_cache = _TTLCache(ttl_seconds=86400)     # 24 hours — rarely changes
sigmet_cache = _TTLCache(ttl_seconds=300)        # 5 min
pirep_cache = _TTLCache(ttl_seconds=300)         # 5 min

_ALL_CACHES = [
    metar_cache, metar_hist_cache, taf_cache,
    synop_cache, station_cache, sigmet_cache, pirep_cache,
]


def clear_all_caches() -> None:
    """Clear all aviation data caches."""
    for cache in _ALL_CACHES:
        cache.clear()
