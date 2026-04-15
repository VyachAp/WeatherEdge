"""CheckWX API provider — best for real-time parsed METAR/TAF.

https://api.checkwxapi.com — free tier: 2,000 requests/day.
Returns beautifully parsed JSON with all fields pre-decoded.
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
from src.ingestion.aviation._caching import metar_cache, taf_cache
from src.ingestion.aviation._conversions import c_to_f
from src.ingestion.aviation._rate_limit import ProviderRateLimiter
from src.ingestion.aviation._types import Observation

logger = logging.getLogger(__name__)

CHECKWX_BASE = "https://api.checkwxapi.com"


class CheckWXProvider(AviationProvider):
    """CheckWX REST API provider."""

    name = "checkwx"

    def __init__(self, api_key: str, daily_budget: int = 2000) -> None:
        self._api_key = api_key
        self._rate_limiter = ProviderRateLimiter(
            name="checkwx",
            max_per_second=1.0,
            daily_budget=daily_budget,
        )

    def _make_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers={
                "X-API-Key": self._api_key,
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
    async def _get_json(self, client: httpx.AsyncClient, url: str) -> Any:
        await self._rate_limiter.acquire()
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()

    def _parse_decoded_metar(self, raw: dict[str, Any]) -> Observation:
        """Parse CheckWX decoded METAR response into Observation."""
        temp_c = raw.get("temperature", {}).get("celsius")
        dewpoint_c = raw.get("dewpoint", {}).get("celsius")

        wind = raw.get("wind", {})
        wind_speed_kts = wind.get("speed_kts")
        wind_gust_kts = wind.get("gust_kts")
        wind_dir = str(wind.get("degrees")) if wind.get("degrees") is not None else None

        vis = raw.get("visibility", {})
        vis_m = vis.get("meters")
        vis_miles = vis.get("miles")

        baro = raw.get("barometer", {})
        pressure_hpa = baro.get("hpa") or baro.get("mb")

        clouds_raw = raw.get("clouds", []) or []
        sky_condition: list[dict[str, Any]] = []
        ceiling_ft: int | None = None
        for c in clouds_raw:
            cover = c.get("code")
            base_ft = c.get("base_feet_agl")
            sky_condition.append({"cover": cover, "base_ft": base_ft})
            if cover in ("BKN", "OVC", "OVX") and ceiling_ft is None:
                ceiling_ft = base_ft

        obs_str = raw.get("observed", "")
        try:
            observed_at = datetime.fromisoformat(obs_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            observed_at = datetime.now(timezone.utc)

        return Observation(
            station_icao=raw.get("icao", raw.get("station", {}).get("icao", "")),
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
            flight_category=raw.get("flight_category"),
            is_speci=raw.get("type") == "SPECI",
            raw_metar=raw.get("raw_text"),
            source="checkwx",
        )

    async def fetch_metar(self, station: str) -> Observation | None:
        cache_key = ("checkwx_metar", station)
        cached = metar_cache.get(cache_key)
        if cached is not None:
            return cached

        async with self._make_client() as client:
            try:
                data = await self._get_json(client, f"{CHECKWX_BASE}/metar/{station}/decoded")
            except Exception:
                logger.warning("CheckWX METAR fetch failed for %s", station, exc_info=True)
                return None

        results = data.get("data", [])
        if not results:
            return None

        obs = self._parse_decoded_metar(results[0])
        metar_cache.set(cache_key, obs)
        return obs

    async def fetch_metar_history(self, station: str, hours: int = 24) -> list[Observation]:
        # CheckWX doesn't support historical queries well on the free tier
        return []

    async def fetch_metar_bulk(self, stations: list[str]) -> dict[str, Observation]:
        """CheckWX supports comma-separated ICAOs (up to 20)."""
        result: dict[str, Observation] = {}
        for i in range(0, len(stations), 20):
            batch = stations[i: i + 20]
            ids_str = ",".join(batch)

            async with self._make_client() as client:
                try:
                    data = await self._get_json(client, f"{CHECKWX_BASE}/metar/{ids_str}/decoded")
                except Exception:
                    logger.warning("CheckWX bulk METAR failed for %s", ids_str, exc_info=True)
                    continue

            for entry in data.get("data", []):
                try:
                    obs = self._parse_decoded_metar(entry)
                    result[obs.station_icao] = obs
                except Exception:
                    logger.warning("CheckWX METAR parse failed", exc_info=True)

        return result

    async def fetch_taf(self, station: str) -> dict[str, Any] | None:
        cache_key = ("checkwx_taf", station)
        cached = taf_cache.get(cache_key)
        if cached is not None:
            return cached

        async with self._make_client() as client:
            try:
                data = await self._get_json(client, f"{CHECKWX_BASE}/taf/{station}/decoded")
            except Exception:
                logger.warning("CheckWX TAF fetch failed for %s", station, exc_info=True)
                return None

        results = data.get("data", [])
        if not results:
            return None

        raw = results[0]
        # Convert CheckWX TAF format to our standard format
        periods: list[dict[str, Any]] = []
        for fc in raw.get("forecast", []) or []:
            period: dict[str, Any] = {
                "type": fc.get("change", {}).get("indicator", "FM"),
                "from": fc.get("timestamp", {}).get("from", ""),
                "to": fc.get("timestamp", {}).get("to", ""),
                "wind_dir": fc.get("wind", {}).get("degrees"),
                "wind_speed_kts": fc.get("wind", {}).get("speed_kts"),
                "wind_gust_kts": fc.get("wind", {}).get("gust_kts"),
                "visibility_miles": fc.get("visibility", {}).get("miles"),
                "sky_condition": [
                    {"cover": c.get("code"), "base_ft": c.get("base_feet_agl")}
                    for c in (fc.get("clouds") or [])
                ],
                "weather": " ".join(
                    w.get("text", "") for w in (fc.get("conditions") or [])
                ) or None,
                "prob": fc.get("change", {}).get("probability"),
            }
            periods.append(period)

        result = {
            "station_icao": raw.get("icao", station),
            "issued_at": datetime.now(timezone.utc),
            "valid_from": datetime.now(timezone.utc),
            "valid_to": datetime.now(timezone.utc),
            "periods": periods,
            "amendment_number": 0,
            "raw_taf": raw.get("raw_text", ""),
            "fetched_at": datetime.now(timezone.utc),
        }

        taf_cache.set(cache_key, result)
        return result

    async def health_check(self) -> bool:
        try:
            async with self._make_client() as client:
                resp = await client.get(f"{CHECKWX_BASE}/metar/KJFK")
                return resp.status_code == 200
        except Exception:
            return False
