"""Iowa Environmental Mesonet (IEM) provider — primary for global METAR history.

https://mesonet.agron.iastate.edu — FREE, no API key needed, global coverage.
Also provides 1-minute ASOS data for US stations.
"""

from __future__ import annotations

import csv
import io
import logging
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
from src.ingestion.aviation._caching import metar_hist_cache
from src.ingestion.aviation._conversions import c_to_f, miles_to_m
from src.ingestion.aviation._rate_limit import ProviderRateLimiter
from src.ingestion.aviation._types import MinuteObs, Observation

logger = logging.getLogger(__name__)

IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
IEM_1MIN_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos1min.py"

_rate_limiter = ProviderRateLimiter(name="iem", max_per_second=1.0)


def _make_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={"User-Agent": "WeatherEdge/1.0"},
        timeout=30,
    )


class IEMProvider(AviationProvider):
    """Iowa Environmental Mesonet provider."""

    name = "iem"

    async def fetch_metar(self, station: str) -> Observation | None:
        # IEM is better for history than real-time; delegate to history and take latest
        history = await self.fetch_metar_history(station, hours=2)
        return history[0] if history else None

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
        reraise=True,
    )
    async def fetch_metar_history(self, station: str, hours: int = 24) -> list[Observation]:
        cache_key = ("iem_hist", station, hours)
        cached = metar_hist_cache.get(cache_key)
        if cached is not None:
            return cached

        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=hours)

        params = {
            "station": station,
            "data": ["tmpf", "dwpf", "drct", "sknt", "gust", "vsby", "skyc1",
                      "skyc2", "skyc3", "skyl1", "skyl2", "skyl3", "p01i",
                      "alti", "metar", "mslp", "feel"],
            "tz": "Etc/UTC",
            "format": "comma",
            "latlon": "yes",
            "year1": str(start.year),
            "month1": str(start.month),
            "day1": str(start.day),
            "hour1": str(start.hour),
            "year2": str(now.year),
            "month2": str(now.month),
            "day2": str(now.day),
            "hour2": str(now.hour),
        }

        await _rate_limiter.acquire()

        async with _make_client() as client:
            resp = await client.get(IEM_ASOS_URL, params=params)
            resp.raise_for_status()
            text = resp.text

        observations = _parse_iem_csv(text, station)
        metar_hist_cache.set(cache_key, observations)
        return observations

    async def fetch_taf(self, station: str) -> dict[str, Any] | None:
        # IEM doesn't provide TAF data
        return None

    async def fetch_one_minute(self, station: str, hours: int = 6) -> list[MinuteObs]:
        """Fetch 1-minute ASOS data (US stations only)."""
        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=hours)

        params = {
            "station": station,
            "tz": "Etc/UTC",
            "year1": str(start.year),
            "month1": str(start.month),
            "day1": str(start.day),
            "hour1": str(start.hour),
            "minute1": "0",
            "year2": str(now.year),
            "month2": str(now.month),
            "day2": str(now.day),
            "hour2": str(now.hour),
            "minute2": str(now.minute),
            "vars[]": ["tmpf", "dwpf", "drct", "sknt", "pres", "precip"],
        }

        await _rate_limiter.acquire()

        async with _make_client() as client:
            try:
                resp = await client.get(IEM_1MIN_URL, params=params)
                resp.raise_for_status()
            except Exception:
                logger.warning("IEM 1-min fetch failed for %s", station, exc_info=True)
                return []

        return _parse_iem_1min_csv(resp.text, station)

    async def health_check(self) -> bool:
        try:
            async with _make_client() as client:
                resp = await client.get(
                    IEM_ASOS_URL,
                    params={"station": "KJFK", "data": "tmpf", "tz": "Etc/UTC",
                            "format": "comma", "hours": "1"},
                )
                return resp.status_code == 200
        except Exception:
            return False


def _parse_iem_csv(text: str, station: str) -> list[Observation]:
    """Parse IEM ASOS CSV response into Observation list."""
    observations: list[Observation] = []

    lines = text.strip().split("\n")
    # Skip comment lines (start with #)
    data_lines = [l for l in lines if not l.startswith("#")]
    if len(data_lines) < 2:
        return []

    reader = csv.DictReader(io.StringIO("\n".join(data_lines)))

    for row in reader:
        try:
            valid_str = row.get("valid", "")
            try:
                observed_at = datetime.strptime(valid_str, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            except ValueError:
                continue

            temp_f = _safe_float(row.get("tmpf"))
            temp_c = (temp_f - 32.0) * 5.0 / 9.0 if temp_f is not None else None
            dewpoint_f = _safe_float(row.get("dwpf"))
            dewpoint_c = (dewpoint_f - 32.0) * 5.0 / 9.0 if dewpoint_f is not None else None

            wind_speed_kts = _safe_float(row.get("sknt"))
            wind_gust_kts = _safe_float(row.get("gust"))
            wind_dir_raw = row.get("drct", "")
            wind_dir = wind_dir_raw if wind_dir_raw and wind_dir_raw != "M" else None

            vis_miles = _safe_float(row.get("vsby"))
            vis_m = miles_to_m(vis_miles)

            # Altimeter setting (inHg) → approximate hPa
            alti = _safe_float(row.get("alti"))
            pressure_hpa = alti / 0.02953 if alti is not None else None

            # Sky condition from IEM fields
            sky_condition: list[dict[str, Any]] = []
            ceiling_ft: int | None = None
            for i in range(1, 4):
                cover = row.get(f"skyc{i}", "")
                base_str = row.get(f"skyl{i}", "")
                if not cover or cover == "M":
                    continue
                base_ft = _safe_int(base_str)
                sky_condition.append({"cover": cover, "base_ft": base_ft})
                if cover in ("BKN", "OVC", "OVX") and ceiling_ft is None:
                    ceiling_ft = base_ft

            raw_metar = row.get("metar", "")
            is_speci = raw_metar.strip().startswith("SPECI") if raw_metar else False

            obs = Observation(
                station_icao=station,
                observed_at=observed_at,
                temp_c=temp_c,
                temp_f=temp_f,
                dewpoint_c=dewpoint_c,
                dewpoint_f=dewpoint_f,
                wind_speed_kts=wind_speed_kts,
                wind_gust_kts=wind_gust_kts,
                wind_dir=wind_dir,
                visibility_m=vis_m,
                visibility_miles=vis_miles,
                pressure_hpa=pressure_hpa,
                sky_condition=tuple(sky_condition),
                ceiling_ft=ceiling_ft,
                flight_category=None,  # IEM doesn't provide this
                is_speci=is_speci,
                raw_metar=raw_metar or None,
                source="iem",
            )
            observations.append(obs)
        except Exception:
            logger.debug("IEM CSV row parse failed", exc_info=True)
            continue

    # Sort by time descending (most recent first)
    observations.sort(key=lambda o: o.observed_at, reverse=True)
    return observations


def _parse_iem_1min_csv(text: str, station: str) -> list[MinuteObs]:
    """Parse IEM 1-minute ASOS CSV response."""
    results: list[MinuteObs] = []

    lines = text.strip().split("\n")
    data_lines = [l for l in lines if not l.startswith("#")]
    if len(data_lines) < 2:
        return []

    reader = csv.DictReader(io.StringIO("\n".join(data_lines)))

    for row in reader:
        try:
            valid_str = row.get("valid", "")
            try:
                observed_at = datetime.strptime(valid_str, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            except ValueError:
                continue

            temp_f = _safe_float(row.get("tmpf"))
            temp_c = (temp_f - 32.0) * 5.0 / 9.0 if temp_f is not None else None
            dewpoint_f = _safe_float(row.get("dwpf"))
            dewpoint_c = (dewpoint_f - 32.0) * 5.0 / 9.0 if dewpoint_f is not None else None

            wind_speed_kts = _safe_float(row.get("sknt"))
            wind_dir_raw = row.get("drct", "")
            wind_dir = int(float(wind_dir_raw)) if wind_dir_raw and wind_dir_raw != "M" else None

            precip_raw = _safe_float(row.get("precip"))
            precip_mm = precip_raw * 25.4 if precip_raw is not None else None  # inches to mm

            pres_raw = _safe_float(row.get("pres"))
            pressure_hpa = pres_raw / 0.02953 if pres_raw is not None else None

            results.append(MinuteObs(
                station_icao=station,
                observed_at=observed_at,
                temp_c=temp_c,
                dewpoint_c=dewpoint_c,
                wind_speed_kts=wind_speed_kts,
                wind_dir=wind_dir,
                precip_mm=precip_mm,
                pressure_hpa=pressure_hpa,
            ))
        except Exception:
            continue

    results.sort(key=lambda o: o.observed_at, reverse=True)
    return results


def _safe_float(val: str | None) -> float | None:
    if val is None or val == "" or val == "M":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(val: str | None) -> int | None:
    if val is None or val == "" or val == "M":
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None
