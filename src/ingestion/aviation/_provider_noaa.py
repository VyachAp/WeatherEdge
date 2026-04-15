"""NOAA/NWS direct raw METAR provider.

https://tgftp.nws.noaa.gov — FREE, no API key needed.
Provides raw METAR text, updated every cycle.
Simple but reliable fallback source.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.ingestion.aviation._base_provider import AviationProvider
from src.ingestion.aviation._caching import metar_cache
from src.ingestion.aviation._parsers import parse_raw_metar
from src.ingestion.aviation._rate_limit import ProviderRateLimiter
from src.ingestion.aviation._types import Observation

logger = logging.getLogger(__name__)

NOAA_METAR_URL = "https://tgftp.nws.noaa.gov/data/observations/metar/stations"

_rate_limiter = ProviderRateLimiter(name="noaa", max_per_second=1.0)


def _make_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={"User-Agent": "WeatherEdge/1.0"},
        timeout=15,
    )


class NOAAProvider(AviationProvider):
    """NOAA/NWS direct raw METAR provider."""

    name = "noaa"

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
        reraise=True,
    )
    async def _fetch_raw(self, client: httpx.AsyncClient, station: str) -> str:
        await _rate_limiter.acquire()
        url = f"{NOAA_METAR_URL}/{station.upper()}.TXT"
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.text

    async def fetch_metar(self, station: str) -> Observation | None:
        cache_key = ("noaa_metar", station)
        cached = metar_cache.get(cache_key)
        if cached is not None:
            return cached

        async with _make_client() as client:
            try:
                text = await self._fetch_raw(client, station)
            except Exception:
                logger.warning("NOAA METAR fetch failed for %s", station, exc_info=True)
                return None

        obs = _parse_noaa_metar_text(text, station)
        if obs is not None:
            metar_cache.set(cache_key, obs)
        return obs

    async def fetch_metar_history(self, station: str, hours: int = 24) -> list[Observation]:
        # NOAA direct only serves latest METAR, not history
        return []

    async def fetch_taf(self, station: str) -> dict[str, Any] | None:
        # NOAA TXT endpoint doesn't serve TAFs
        return None

    async def health_check(self) -> bool:
        try:
            async with _make_client() as client:
                resp = await client.get(f"{NOAA_METAR_URL}/KJFK.TXT")
                return resp.status_code == 200
        except Exception:
            return False


def _parse_noaa_metar_text(text: str, station: str) -> Observation | None:
    """Parse NOAA raw METAR text file.

    NOAA format is:
    2026/04/14 18:53
    KPHX 141853Z 22012G18KT 10SM FEW250 32/08 A2992 RMK...
    """
    lines = text.strip().split("\n")
    if len(lines) < 2:
        return None

    # The METAR line is typically the last non-empty line
    metar_line = ""
    for line in reversed(lines):
        line = line.strip()
        if line and not line[0].isdigit() or (len(line) > 10 and "/" not in line[:11]):
            metar_line = line
            break
        # Could also be on the second line
        if station.upper() in line.upper():
            metar_line = line
            break

    if not metar_line:
        # Fallback: take the second line
        if len(lines) >= 2:
            metar_line = lines[1].strip()
        else:
            return None

    return parse_raw_metar(metar_line, station=station, source="noaa")
