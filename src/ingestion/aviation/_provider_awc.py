"""Aviation Weather Center (aviationweather.gov) provider.

This is the original data source, extracted from the monolithic aviation.py.
Supports METAR, TAF, PIREP, and SIGMET/AIRMET via the AWC REST API.
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

from src.config import settings
from src.ingestion.aviation._base_provider import AviationProvider
from src.ingestion.aviation._caching import metar_cache, metar_hist_cache, pirep_cache, sigmet_cache, taf_cache
from src.ingestion.aviation._conversions import c_to_f
from src.ingestion.aviation._rate_limit import ProviderRateLimiter
from src.ingestion.aviation._types import Observation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AWC_BASE = "https://aviationweather.gov/api/data"
METAR_URL = f"{AWC_BASE}/metar"
TAF_URL = f"{AWC_BASE}/taf"
PIREP_URL = f"{AWC_BASE}/pirep"
AIRSIGMET_URL = f"{AWC_BASE}/airsigmet"

_BULK_BATCH_SIZE = 20

# Precipitation weather codes in TAFs
PRECIP_CODES = frozenset({
    "RA", "SN", "TS", "TSRA", "TSSN", "DZ", "FZRA", "FZDZ",
    "SG", "GR", "GS", "PL", "IC", "SHRA", "SHSN", "SHGR",
})

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

_rate_limiter = ProviderRateLimiter(
    name="awc",
    max_per_second=settings.AWC_RATE_LIMIT_RPS,
)

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _make_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={
            "User-Agent": settings.AWC_USER_AGENT,
            "Accept-Encoding": "gzip",
        },
        timeout=30,
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
    reraise=True,
)
async def _awc_get_json(
    client: httpx.AsyncClient,
    url: str,
    params: dict[str, Any] | None = None,
) -> Any:
    """Fetch JSON from AWC API with rate limiting and retry."""
    await _rate_limiter.acquire()
    resp = await client.get(url, params=params)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Parsers — AWC JSON to DB-ready dicts
# ---------------------------------------------------------------------------


def _parse_awc_observed_at(raw: dict[str, Any]) -> datetime:
    """Extract observation timestamp from an AWC API object.

    AWC returns ``obsTime`` as a Unix epoch integer and ``reportTime`` as an
    ISO8601 string. Try ISO first, then epoch, then fall back to now().
    """
    report_time = raw.get("reportTime")
    if report_time:
        try:
            return datetime.fromisoformat(str(report_time).replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            pass
    obs_time = raw.get("obsTime")
    if obs_time is not None:
        try:
            return datetime.fromtimestamp(int(obs_time), tz=timezone.utc)
        except (ValueError, TypeError):
            try:
                return datetime.fromisoformat(str(obs_time).replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass
    return datetime.now(timezone.utc)


def parse_metar_json(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert AWC METAR JSON object to DB-ready dict."""
    temp_c = raw.get("temp")
    dewpoint_c = raw.get("dwpt")

    clouds = raw.get("clouds", []) or []
    sky_condition = [
        {"cover": c.get("cover"), "base_ft": c.get("base")}
        for c in clouds
    ]

    ceiling_ft = None
    for c in clouds:
        cover = c.get("cover")
        if cover in ("BKN", "OVC", "OVX"):
            ceiling_ft = c.get("base")
            break

    vis_raw = raw.get("visib")
    try:
        vis_miles = float(str(vis_raw).rstrip("+").lstrip("P")) if vis_raw is not None else None
    except (ValueError, TypeError):
        vis_miles = None
    vis_m = vis_miles * 1609.34 if vis_miles is not None else None

    observed_at = _parse_awc_observed_at(raw)

    return {
        "station_icao": raw.get("icaoId"),
        "observed_at": observed_at,
        "temp_c": temp_c,
        "dewpoint_c": dewpoint_c,
        "temp_f": c_to_f(temp_c),
        "dewpoint_f": c_to_f(dewpoint_c),
        "wind_speed_kts": raw.get("wspd"),
        "wind_dir": str(raw.get("wdir")) if raw.get("wdir") is not None else None,
        "wind_gust_kts": raw.get("wgst"),
        "visibility_m": vis_m,
        "visibility_miles": vis_miles,
        "pressure_hpa": raw.get("altim"),
        "sky_condition": sky_condition,
        "ceiling_ft": ceiling_ft,
        "flight_category": raw.get("fltcat"),
        "is_speci": raw.get("metar_type") == "SPECI",
        "raw_metar": raw.get("rawOb"),
        "fetched_at": datetime.now(timezone.utc),
    }


def parse_taf_json(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert AWC TAF JSON object to DB-ready dict."""
    periods: list[dict[str, Any]] = []
    for p in raw.get("fcsts", []) or []:
        periods.append({
            "type": p.get("fcstType"),
            "from": p.get("timeFrom"),
            "to": p.get("timeTo"),
            "wind_dir": p.get("wdir"),
            "wind_speed_kts": p.get("wspd"),
            "wind_gust_kts": p.get("wgst"),
            "visibility_miles": p.get("visib"),
            "sky_condition": [
                {"cover": c.get("cover"), "base_ft": c.get("base")}
                for c in (p.get("clouds") or [])
            ],
            "weather": p.get("wxString"),
            "prob": p.get("probability"),
        })

    raw_text = raw.get("rawTAF", "")
    amendment_number = 1 if (" AMD " in raw_text or " COR " in raw_text) else 0

    issue_str = raw.get("issueTime", "")
    from_str = raw.get("validTimeFrom", "")
    to_str = raw.get("validTimeTo", "")

    def _parse_dt(s: str) -> datetime:
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return datetime.now(timezone.utc)

    return {
        "station_icao": raw.get("icaoId"),
        "issued_at": _parse_dt(issue_str),
        "valid_from": _parse_dt(from_str),
        "valid_to": _parse_dt(to_str),
        "periods": periods,
        "amendment_number": amendment_number,
        "raw_taf": raw_text,
        "fetched_at": datetime.now(timezone.utc),
    }


def parse_pirep_json(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert AWC PIREP JSON object to DB-ready dict."""
    observed_at = _parse_awc_observed_at(raw)

    return {
        "report_id": raw.get("airepId") or raw.get("pirepId"),
        "observed_at": observed_at,
        "lat": raw.get("lat"),
        "lon": raw.get("lon"),
        "altitude_ft": raw.get("altFt") or raw.get("fltlvl"),
        "icing_type": raw.get("icgType1"),
        "icing_intensity": raw.get("icgInt1"),
        "turbulence_type": raw.get("tbType1"),
        "turbulence_intensity": raw.get("tbInt1"),
        "weather": raw.get("wxString"),
        "raw_text": raw.get("rawOb"),
        "fetched_at": datetime.now(timezone.utc),
    }


def parse_airsigmet_json(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert AWC SIGMET/AIRMET JSON to DB-ready dict."""
    coords = raw.get("coords") or []
    area = [{"lat": c.get("lat"), "lon": c.get("lon")} for c in coords]

    from_str = raw.get("validTimeFrom", "")
    to_str = raw.get("validTimeTo", "")

    def _dt(s: str) -> datetime | None:
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None

    return {
        "alert_id": raw.get("airSigmetId"),
        "alert_type": raw.get("airsigmetType"),
        "hazard": raw.get("hazard"),
        "severity": raw.get("severity"),
        "area": area,
        "valid_from": _dt(from_str),
        "valid_to": _dt(to_str),
        "raw_text": raw.get("rawAirSigmet"),
        "fetched_at": datetime.now(timezone.utc),
    }


def _metar_dict_to_observation(d: dict[str, Any]) -> Observation:
    """Convert a legacy parsed METAR dict to an Observation dataclass."""
    sky = d.get("sky_condition", [])
    return Observation(
        station_icao=d.get("station_icao", ""),
        observed_at=d.get("observed_at", datetime.now(timezone.utc)),
        temp_c=d.get("temp_c"),
        temp_f=d.get("temp_f"),
        dewpoint_c=d.get("dewpoint_c"),
        dewpoint_f=d.get("dewpoint_f"),
        wind_speed_kts=d.get("wind_speed_kts"),
        wind_gust_kts=d.get("wind_gust_kts"),
        wind_dir=d.get("wind_dir"),
        visibility_m=d.get("visibility_m"),
        visibility_miles=d.get("visibility_miles"),
        pressure_hpa=d.get("pressure_hpa"),
        sky_condition=tuple(sky) if sky else (),
        ceiling_ft=d.get("ceiling_ft"),
        flight_category=d.get("flight_category"),
        is_speci=d.get("is_speci", False),
        raw_metar=d.get("raw_metar"),
        source="awc",
    )


# ---------------------------------------------------------------------------
# AWC Provider class
# ---------------------------------------------------------------------------


class AWCProvider(AviationProvider):
    """aviationweather.gov REST API provider."""

    name = "awc"

    async def fetch_metar(self, station: str) -> Observation | None:
        cached = metar_cache.get(("awc_metar", station))
        if cached is not None:
            return cached

        async with _make_client() as client:
            try:
                data = await _awc_get_json(
                    client, METAR_URL, {"ids": station, "format": "json"},
                )
            except Exception:
                logger.warning("AWC METAR fetch failed for %s", station, exc_info=True)
                return None

        if not isinstance(data, list) or not data:
            return None

        parsed = parse_metar_json(data[0])
        obs = _metar_dict_to_observation(parsed)
        metar_cache.set(("awc_metar", station), obs)
        return obs

    async def fetch_metar_history(self, station: str, hours: int = 24) -> list[Observation]:
        cache_key = ("awc_metar_hist", station, hours)
        cached = metar_hist_cache.get(cache_key)
        if cached is not None:
            return cached

        async with _make_client() as client:
            try:
                data = await _awc_get_json(
                    client, METAR_URL, {"ids": station, "format": "json", "hours": hours},
                )
            except Exception:
                logger.warning("AWC METAR history failed for %s", station, exc_info=True)
                return []

        if not isinstance(data, list):
            return []

        results = []
        for entry in data:
            try:
                parsed = parse_metar_json(entry)
                results.append(_metar_dict_to_observation(parsed))
            except Exception:
                logger.warning("AWC METAR history parse failed", exc_info=True)

        metar_hist_cache.set(cache_key, results)
        return results

    async def fetch_metar_bulk(self, stations: list[str]) -> dict[str, Observation]:
        result: dict[str, Observation] = {}
        async with _make_client() as client:
            for i in range(0, len(stations), _BULK_BATCH_SIZE):
                batch = stations[i: i + _BULK_BATCH_SIZE]
                ids_str = ",".join(batch)
                try:
                    data = await _awc_get_json(
                        client, METAR_URL, {"ids": ids_str, "format": "json"},
                    )
                except Exception:
                    logger.warning("AWC bulk METAR failed for %s", ids_str, exc_info=True)
                    continue

                if not isinstance(data, list):
                    continue

                for entry in data:
                    try:
                        parsed = parse_metar_json(entry)
                        obs = _metar_dict_to_observation(parsed)
                        result[obs.station_icao] = obs
                    except Exception:
                        logger.warning("AWC bulk METAR parse failed", exc_info=True)

        return result

    async def fetch_taf(self, station: str) -> dict[str, Any] | None:
        cached = taf_cache.get(("awc_taf", station))
        if cached is not None:
            return cached

        async with _make_client() as client:
            try:
                data = await _awc_get_json(
                    client, TAF_URL, {"ids": station, "format": "json"},
                )
            except Exception:
                logger.warning("AWC TAF fetch failed for %s", station, exc_info=True)
                return None

        if not isinstance(data, list) or not data:
            return None

        parsed = parse_taf_json(data[0])
        taf_cache.set(("awc_taf", station), parsed)
        return parsed

    async def fetch_taf_bulk(self, stations: list[str]) -> list[dict[str, Any]]:
        cache_key = ("awc_tafs", tuple(sorted(stations)))
        cached = taf_cache.get(cache_key)
        if cached is not None:
            return cached

        results: list[dict[str, Any]] = []
        async with _make_client() as client:
            for i in range(0, len(stations), _BULK_BATCH_SIZE):
                batch = stations[i: i + _BULK_BATCH_SIZE]
                ids_str = ",".join(batch)
                try:
                    data = await _awc_get_json(
                        client, TAF_URL, {"ids": ids_str, "format": "json"},
                    )
                except Exception:
                    logger.warning("AWC TAF fetch failed for %s", ids_str, exc_info=True)
                    continue

                if not isinstance(data, list):
                    continue

                for entry in data:
                    try:
                        results.append(parse_taf_json(entry))
                    except Exception:
                        logger.warning("AWC TAF parse failed", exc_info=True)

        taf_cache.set(cache_key, results)
        return results

    async def fetch_pireps(
        self, lat: float, lon: float, radius_nm: int = 100,
    ) -> list[dict[str, Any]]:
        cache_key = ("awc_pireps", round(lat, 2), round(lon, 2), radius_nm)
        cached = pirep_cache.get(cache_key)
        if cached is not None:
            return cached

        async with _make_client() as client:
            try:
                data = await _awc_get_json(
                    client, PIREP_URL,
                    {"format": "json", "age": 6, "dist": radius_nm, "lat": lat, "lon": lon},
                )
            except Exception:
                logger.warning("AWC PIREP fetch failed", exc_info=True)
                return []

        if not isinstance(data, list):
            return []

        results = []
        for entry in data:
            try:
                results.append(parse_pirep_json(entry))
            except Exception:
                logger.warning("AWC PIREP parse failed", exc_info=True)

        pirep_cache.set(cache_key, results)
        return results

    async def fetch_sigmets(self) -> list[dict[str, Any]]:
        cache_key = ("awc_airsigmets",)
        cached = sigmet_cache.get(cache_key)
        if cached is not None:
            return cached

        async with _make_client() as client:
            try:
                data = await _awc_get_json(
                    client, AIRSIGMET_URL, {"format": "json"},
                )
            except Exception:
                logger.warning("AWC AIRSIGMET fetch failed", exc_info=True)
                return []

        if not isinstance(data, list):
            return []

        results = []
        for entry in data:
            try:
                results.append(parse_airsigmet_json(entry))
            except Exception:
                logger.warning("AWC AIRSIGMET parse failed", exc_info=True)

        sigmet_cache.set(cache_key, results)
        return results

    async def health_check(self) -> bool:
        try:
            async with _make_client() as client:
                resp = await client.get(METAR_URL, params={"ids": "KJFK", "format": "json"})
                return resp.status_code == 200
        except Exception:
            return False
