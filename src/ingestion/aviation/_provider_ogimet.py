"""OGIMET provider — primary for global SYNOP + METAR archive.

https://www.ogimet.com — FREE, no API key, global coverage.
Strict rate limit: max 1 request per 5 seconds.
Provides METAR, SPECI, TAF, and SYNOP reports worldwide.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.ingestion.aviation._base_provider import AviationProvider
from src.ingestion.aviation._caching import metar_hist_cache, synop_cache, taf_cache
from src.ingestion.aviation._parsers import parse_raw_metar, parse_raw_synop
from src.ingestion.aviation._rate_limit import ProviderRateLimiter
from src.ingestion.aviation._types import Observation, SynopObs

logger = logging.getLogger(__name__)

OGIMET_METAR_URL = "https://www.ogimet.com/cgi-bin/getmetar"
OGIMET_SYNOP_URL = "https://www.ogimet.com/cgi-bin/getsynop"
OGIMET_TAF_URL = "https://www.ogimet.com/cgi-bin/gettaf"

_rate_limiter = ProviderRateLimiter(name="ogimet", max_per_second=0.2)  # 1 per 5 sec


def _make_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={"User-Agent": "WeatherEdge/1.0 (weather research)"},
        timeout=30,
    )


def _dt_to_ogimet(dt: datetime) -> str:
    """Format datetime for OGIMET begin/end params: YYYYMMDDHH00."""
    return dt.strftime("%Y%m%d%H00")


# Regex to extract raw METAR lines from OGIMET HTML
_METAR_LINE_RE = re.compile(r"(\d{12})\s+((?:METAR|SPECI)\s+\S{4}\s+.+?)(?=<br>|</pre>|\n\d{12})", re.DOTALL)
# Regex to extract SYNOP lines
_SYNOP_LINE_RE = re.compile(r"(\d{12})\s+(AAXX\s+.+?)(?=<br>|</pre>|\n\d{12})", re.DOTALL)


class OGIMETProvider(AviationProvider):
    """OGIMET METAR/SYNOP archive provider."""

    name = "ogimet"

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=2, min=5, max=15),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
        reraise=True,
    )
    async def _get_html(self, client: httpx.AsyncClient, url: str, params: dict) -> str:
        await _rate_limiter.acquire()
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.text

    async def fetch_metar(self, station: str) -> Observation | None:
        history = await self.fetch_metar_history(station, hours=3)
        return history[0] if history else None

    async def fetch_metar_history(self, station: str, hours: int = 24) -> list[Observation]:
        cache_key = ("ogimet_metar_hist", station, hours)
        cached = metar_hist_cache.get(cache_key)
        if cached is not None:
            return cached

        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=hours)

        async with _make_client() as client:
            try:
                html = await self._get_html(client, OGIMET_METAR_URL, {
                    "icao": station,
                    "begin": _dt_to_ogimet(start),
                    "end": _dt_to_ogimet(now),
                })
            except Exception:
                logger.warning("OGIMET METAR fetch failed for %s", station, exc_info=True)
                return []

        observations = _parse_ogimet_metar_html(html, station)
        metar_hist_cache.set(cache_key, observations)
        return observations

    async def fetch_taf(self, station: str) -> dict[str, Any] | None:
        cache_key = ("ogimet_taf", station)
        cached = taf_cache.get(cache_key)
        if cached is not None:
            return cached

        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=12)

        async with _make_client() as client:
            try:
                html = await self._get_html(client, OGIMET_TAF_URL, {
                    "icao": station,
                    "begin": _dt_to_ogimet(start),
                    "end": _dt_to_ogimet(now),
                })
            except Exception:
                logger.warning("OGIMET TAF fetch failed for %s", station, exc_info=True)
                return None

        taf = _parse_ogimet_taf_html(html, station)
        if taf is not None:
            taf_cache.set(cache_key, taf)
        return taf

    async def fetch_synop(self, wmo_id: str, hours: int = 24) -> list[SynopObs]:
        cache_key = ("ogimet_synop", wmo_id, hours)
        cached = synop_cache.get(cache_key)
        if cached is not None:
            return cached

        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=hours)

        async with _make_client() as client:
            try:
                html = await self._get_html(client, OGIMET_SYNOP_URL, {
                    "block": wmo_id,
                    "begin": _dt_to_ogimet(start),
                    "end": _dt_to_ogimet(now),
                })
            except Exception:
                logger.warning("OGIMET SYNOP fetch failed for %s", wmo_id, exc_info=True)
                return []

        results = _parse_ogimet_synop_html(html, wmo_id)
        synop_cache.set(cache_key, results)
        return results

    async def health_check(self) -> bool:
        try:
            async with _make_client() as client:
                resp = await client.get(OGIMET_METAR_URL, params={"icao": "KJFK", "begin": "202604140000", "end": "202604140100"})
                return resp.status_code == 200
        except Exception:
            return False


def _parse_ogimet_metar_html(html: str, station: str) -> list[Observation]:
    """Parse OGIMET METAR response (CSV format).

    OGIMET getmetar returns CSV lines like:
        EFHK,2026,04,15,00,20,METAR EFHK 150020Z 31003KT CAVOK M02/M03 Q1028 NOSIG=
    Fields: ICAO, YYYY, MM, DD, HH, MM, METAR_TEXT
    """
    observations: list[Observation] = []

    for line in html.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("<"):
            continue

        parts = line.split(",", 6)  # Split into at most 7 fields
        if len(parts) < 7:
            continue

        icao, year_s, month_s, day_s, hour_s, minute_s, raw_metar = parts
        raw_metar = raw_metar.strip().rstrip("=")

        # Must contain METAR or SPECI
        if "METAR" not in raw_metar and "SPECI" not in raw_metar:
            continue

        obs = parse_raw_metar(raw_metar, station=icao.strip() or station, source="ogimet")
        if obs is not None:
            observations.append(obs)

    observations.sort(key=lambda o: o.observed_at, reverse=True)
    return observations


def _parse_ogimet_taf_html(html: str, station: str) -> dict[str, Any] | None:
    """Extract latest TAF from OGIMET HTML response.

    Returns in the same dict format as AWC for compatibility.
    """
    # Find TAF text in the HTML
    taf_re = re.compile(r"(TAF(?:\s+AMD)?(?:\s+COR)?\s+" + re.escape(station) + r"\s+.+?)(?:=|</pre>)", re.DOTALL)
    m = taf_re.search(html)
    if not m:
        return None

    raw_taf = m.group(1).strip()
    raw_taf = re.sub(r"<[^>]+>", "", raw_taf)
    raw_taf = re.sub(r"\s+", " ", raw_taf)

    amendment_number = 1 if (" AMD " in raw_taf or " COR " in raw_taf) else 0

    return {
        "station_icao": station,
        "issued_at": datetime.now(timezone.utc),
        "valid_from": datetime.now(timezone.utc),
        "valid_to": datetime.now(timezone.utc),
        "periods": [],  # Raw TAF not parsed into periods — use for raw text only
        "amendment_number": amendment_number,
        "raw_taf": raw_taf,
        "fetched_at": datetime.now(timezone.utc),
    }


def _parse_ogimet_synop_html(html: str, wmo_id: str) -> list[SynopObs]:
    """Parse OGIMET SYNOP response (CSV format).

    OGIMET getsynop returns CSV lines like:
        02974,2026,04,14,00,00,AAXX 14001 02974 24981 00802 10009 ...==
    Fields: WMO_ID, YYYY, MM, DD, HH, MM, SYNOP_MESSAGE
    """
    results: list[SynopObs] = []

    for line in html.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("<"):
            continue

        parts = line.split(",", 6)  # Split into at most 7 fields
        if len(parts) < 7:
            continue

        station_id, year_s, month_s, day_s, hour_s, minute_s, raw_synop = parts
        station_id = station_id.strip()
        raw_synop = raw_synop.strip().rstrip("=")

        # Must contain AAXX to be a valid SYNOP
        if "AAXX" not in raw_synop:
            continue

        try:
            observed_at = datetime(
                int(year_s), int(month_s), int(day_s),
                int(hour_s), int(minute_s),
                tzinfo=timezone.utc,
            )
        except (ValueError, TypeError):
            continue

        obs = parse_raw_synop(raw_synop, wmo_id=station_id or wmo_id, observed_at=observed_at)
        if obs is not None:
            results.append(obs)

    results.sort(key=lambda o: o.observed_at, reverse=True)
    return results
