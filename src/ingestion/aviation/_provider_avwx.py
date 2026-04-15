"""AVWX REST API provider — backup parsed METAR/TAF source.

https://avwx.rest — free hobby tier provides basic METAR/TAF parsing.
Good error handling for malformed reports.
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
from src.ingestion.aviation._caching import metar_cache, metar_hist_cache, taf_cache
from src.ingestion.aviation._conversions import c_to_f
from src.ingestion.aviation._rate_limit import ProviderRateLimiter
from src.ingestion.aviation._types import Observation

logger = logging.getLogger(__name__)

AVWX_BASE = "https://avwx.rest/api"


class AVWXProvider(AviationProvider):
    """AVWX REST API provider."""

    name = "avwx"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._rate_limiter = ProviderRateLimiter(name="avwx", max_per_second=1.0)

    def _make_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers={
                "Authorization": f"BEARER {self._api_key}",
                "Accept": "application/json",
            },
            timeout=15,
        )

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
        reraise=True,
    )
    async def _get_json(self, client: httpx.AsyncClient, url: str, params: dict | None = None) -> Any:
        await self._rate_limiter.acquire()
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    def _parse_metar(self, raw: dict[str, Any]) -> Observation:
        """Parse AVWX METAR response into Observation."""
        # Temperature
        temp_c = _avwx_float(raw.get("temperature", {}).get("value"))
        dewpoint_c = _avwx_float(raw.get("dewpoint", {}).get("value"))

        # Wind
        wind_data = raw.get("wind_direction", {})
        wind_dir = str(wind_data.get("value")) if wind_data.get("value") is not None else None
        wind_speed_kts = _avwx_float(raw.get("wind_speed", {}).get("value"))
        wind_gust_kts = _avwx_float(raw.get("wind_gust", {}).get("value"))

        # Visibility (AVWX returns in meters by default)
        vis_repr = raw.get("visibility", {}).get("repr", "")
        vis_m = _avwx_float(raw.get("visibility", {}).get("value"))
        vis_miles = vis_m / 1609.34 if vis_m is not None else None

        # Pressure
        altimeter = raw.get("altimeter", {}).get("value")
        pressure_hpa = _avwx_float(altimeter)

        # Sky
        clouds_raw = raw.get("clouds", []) or []
        sky_condition: list[dict[str, Any]] = []
        ceiling_ft: int | None = None
        for c in clouds_raw:
            cover = c.get("type")
            base_ft = c.get("altitude")
            if base_ft is not None:
                base_ft = int(base_ft) * 100  # AVWX returns altitude in hundreds of feet
            sky_condition.append({"cover": cover, "base_ft": base_ft})
            if cover in ("BKN", "OVC", "OVX") and ceiling_ft is None:
                ceiling_ft = base_ft

        # Time
        obs_str = raw.get("time", {}).get("dt", "")
        try:
            observed_at = datetime.fromisoformat(obs_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            observed_at = datetime.now(timezone.utc)

        return Observation(
            station_icao=raw.get("station", ""),
            observed_at=observed_at,
            temp_c=temp_c,
            temp_f=c_to_f(temp_c),
            dewpoint_c=dewpoint_c,
            dewpoint_f=c_to_f(dewpoint_c),
            wind_speed_kts=wind_speed_kts,
            wind_gust_kts=wind_gust_kts,
            wind_dir=wind_dir,
            visibility_m=vis_m,
            visibility_miles=vis_miles,
            pressure_hpa=pressure_hpa,
            sky_condition=tuple(sky_condition),
            ceiling_ft=ceiling_ft,
            flight_category=raw.get("flight_rules"),
            is_speci=raw.get("type") == "SPECI",
            raw_metar=raw.get("raw"),
            source="avwx",
        )

    async def fetch_metar(self, station: str) -> Observation | None:
        cache_key = ("avwx_metar", station)
        cached = metar_cache.get(cache_key)
        if cached is not None:
            return cached

        async with self._make_client() as client:
            try:
                data = await self._get_json(client, f"{AVWX_BASE}/metar/{station}")
            except Exception:
                logger.warning("AVWX METAR fetch failed for %s", station, exc_info=True)
                return None

        if not data or "error" in data:
            return None

        obs = self._parse_metar(data)
        metar_cache.set(cache_key, obs)
        return obs

    async def fetch_metar_history(self, station: str, hours: int = 24) -> list[Observation]:
        # AVWX free tier doesn't support history well; return empty
        return []

    async def fetch_taf(self, station: str) -> dict[str, Any] | None:
        cache_key = ("avwx_taf", station)
        cached = taf_cache.get(cache_key)
        if cached is not None:
            return cached

        async with self._make_client() as client:
            try:
                data = await self._get_json(client, f"{AVWX_BASE}/taf/{station}")
            except Exception:
                logger.warning("AVWX TAF fetch failed for %s", station, exc_info=True)
                return None

        if not data or "error" in data:
            return None

        # Convert AVWX TAF format to our standard format
        periods: list[dict[str, Any]] = []
        for fc in data.get("forecast", []) or []:
            wind_dir_val = fc.get("wind_direction", {}).get("value")
            period: dict[str, Any] = {
                "type": fc.get("type", "FM"),
                "from": fc.get("start_time", {}).get("dt", ""),
                "to": fc.get("end_time", {}).get("dt", ""),
                "wind_dir": wind_dir_val,
                "wind_speed_kts": _avwx_float(fc.get("wind_speed", {}).get("value")),
                "wind_gust_kts": _avwx_float(fc.get("wind_gust", {}).get("value")),
                "visibility_miles": None,
                "sky_condition": [
                    {"cover": c.get("type"), "base_ft": (c.get("altitude") or 0) * 100}
                    for c in (fc.get("clouds") or [])
                ],
                "weather": " ".join(
                    w.get("repr", "") for w in (fc.get("wx_codes") or [])
                ) or None,
                "prob": fc.get("probability"),
            }
            periods.append(period)

        raw_taf = data.get("raw", "")
        amendment_number = 1 if (" AMD " in raw_taf or " COR " in raw_taf) else 0

        result = {
            "station_icao": data.get("station", station),
            "issued_at": datetime.now(timezone.utc),
            "valid_from": datetime.now(timezone.utc),
            "valid_to": datetime.now(timezone.utc),
            "periods": periods,
            "amendment_number": amendment_number,
            "raw_taf": raw_taf,
            "fetched_at": datetime.now(timezone.utc),
        }

        taf_cache.set(cache_key, result)
        return result

    async def health_check(self) -> bool:
        try:
            async with self._make_client() as client:
                resp = await client.get(f"{AVWX_BASE}/metar/KJFK")
                return resp.status_code == 200
        except Exception:
            return False


def _avwx_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
