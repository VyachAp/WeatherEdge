"""Provider aggregator with priority-based failover routing."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.ingestion.aviation._base_provider import AviationProvider
from src.ingestion.aviation._rate_limit import RateLimitExhausted
from src.ingestion.aviation._types import MinuteObs, Observation, SynopObs

logger = logging.getLogger(__name__)

# Failover chains: data type → ordered list of provider names
FAILOVER_CHAINS: dict[str, list[str]] = {
    "metar_realtime": ["checkwx", "awc", "noaa", "avwx"],
    "metar_history":  ["iem", "ogimet", "awc", "avwx"],
    "metar_bulk":     ["checkwx", "awc", "avwx"],
    "taf":            ["checkwx", "awc", "ogimet", "avwx"],
    "taf_bulk":       ["awc", "checkwx", "avwx"],
    "synop":          ["ogimet"],
    "pirep":          ["awc"],
    "sigmet":         ["awc"],
    "one_minute":     ["iem"],
}

# Health recovery delay in seconds
_HEALTH_RECOVERY_SECONDS = 60


class ProviderAggregator:
    """Routes requests to aviation data providers with automatic failover.

    Providers are tried in priority order per data type. Failed providers
    are marked unhealthy for 60 seconds, then auto-restored.
    """

    def __init__(self) -> None:
        self._providers: dict[str, AviationProvider] = {}
        self._health: dict[str, bool] = {}

    def register(self, provider: AviationProvider) -> None:
        """Register a provider. Can be called multiple times."""
        self._providers[provider.name] = provider
        self._health[provider.name] = True

    def _restore_health(self, name: str) -> None:
        self._health[name] = True
        logger.info("Provider %s health restored", name)

    def _mark_unhealthy(self, name: str) -> None:
        self._health[name] = False
        logger.warning("Provider %s marked unhealthy for %ds", name, _HEALTH_RECOVERY_SECONDS)
        try:
            loop = asyncio.get_running_loop()
            loop.call_later(_HEALTH_RECOVERY_SECONDS, self._restore_health, name)
        except RuntimeError:
            pass  # No running loop (e.g. during shutdown)

    def _get_chain(self, chain_name: str) -> list[AviationProvider]:
        """Return healthy, registered providers for the given chain."""
        names = FAILOVER_CHAINS.get(chain_name, [])
        return [
            self._providers[n]
            for n in names
            if n in self._providers and self._health.get(n, False)
        ]

    # ------------------------------------------------------------------
    # METAR
    # ------------------------------------------------------------------

    async def get_metar(self, station: str) -> Observation | None:
        for provider in self._get_chain("metar_realtime"):
            try:
                result = await provider.fetch_metar(station)
                if result is not None:
                    return result
            except RateLimitExhausted:
                logger.info("Provider %s rate limit exhausted, trying next", provider.name)
                continue
            except Exception:
                logger.warning("Provider %s.fetch_metar failed", provider.name, exc_info=True)
                self._mark_unhealthy(provider.name)
        return None

    async def get_metar_history(self, station: str, hours: int = 24) -> list[Observation]:
        for provider in self._get_chain("metar_history"):
            try:
                result = await provider.fetch_metar_history(station, hours)
                if result:
                    return result
            except RateLimitExhausted:
                continue
            except Exception:
                logger.warning("Provider %s.fetch_metar_history failed", provider.name, exc_info=True)
                self._mark_unhealthy(provider.name)
        return []

    async def get_metar_bulk(self, stations: list[str]) -> dict[str, Observation]:
        for provider in self._get_chain("metar_bulk"):
            try:
                result = await provider.fetch_metar_bulk(stations)
                if result:
                    return result
            except RateLimitExhausted:
                continue
            except Exception:
                logger.warning("Provider %s.fetch_metar_bulk failed", provider.name, exc_info=True)
                self._mark_unhealthy(provider.name)
        return {}

    # ------------------------------------------------------------------
    # TAF
    # ------------------------------------------------------------------

    async def get_taf(self, station: str) -> dict[str, Any] | None:
        for provider in self._get_chain("taf"):
            try:
                result = await provider.fetch_taf(station)
                if result is not None:
                    return result
            except RateLimitExhausted:
                continue
            except Exception:
                logger.warning("Provider %s.fetch_taf failed", provider.name, exc_info=True)
                self._mark_unhealthy(provider.name)
        return None

    async def get_taf_bulk(self, stations: list[str]) -> list[dict[str, Any]]:
        for provider in self._get_chain("taf_bulk"):
            try:
                result = await provider.fetch_taf_bulk(stations)
                if result:
                    return result
            except RateLimitExhausted:
                continue
            except Exception:
                logger.warning("Provider %s.fetch_taf_bulk failed", provider.name, exc_info=True)
                self._mark_unhealthy(provider.name)
        return []

    # ------------------------------------------------------------------
    # SYNOP / PIREP / SIGMET / 1-min
    # ------------------------------------------------------------------

    async def get_synop(self, wmo_id: str, hours: int = 24) -> list[SynopObs]:
        for provider in self._get_chain("synop"):
            try:
                result = await provider.fetch_synop(wmo_id, hours)
                if result:
                    return result
            except Exception:
                logger.warning("Provider %s.fetch_synop failed", provider.name, exc_info=True)
                self._mark_unhealthy(provider.name)
        return []

    async def get_pireps(self, lat: float, lon: float, radius_nm: int = 100) -> list[dict[str, Any]]:
        for provider in self._get_chain("pirep"):
            try:
                result = await provider.fetch_pireps(lat, lon, radius_nm)
                if result:
                    return result
            except Exception:
                logger.warning("Provider %s.fetch_pireps failed", provider.name, exc_info=True)
                self._mark_unhealthy(provider.name)
        return []

    async def get_sigmets(self) -> list[dict[str, Any]]:
        for provider in self._get_chain("sigmet"):
            try:
                result = await provider.fetch_sigmets()
                if result:
                    return result
            except Exception:
                logger.warning("Provider %s.fetch_sigmets failed", provider.name, exc_info=True)
                self._mark_unhealthy(provider.name)
        return []

    async def get_one_minute(self, station: str, hours: int = 6) -> list[MinuteObs]:
        for provider in self._get_chain("one_minute"):
            try:
                result = await provider.fetch_one_minute(station, hours)
                if result:
                    return result
            except Exception:
                logger.warning("Provider %s.fetch_one_minute failed", provider.name, exc_info=True)
                self._mark_unhealthy(provider.name)
        return []


# ---------------------------------------------------------------------------
# Module-level singleton (lazy initialization)
# ---------------------------------------------------------------------------

_aggregator: ProviderAggregator | None = None


def get_aggregator() -> ProviderAggregator:
    """Return the global aggregator, initializing providers on first call."""
    global _aggregator
    if _aggregator is not None:
        return _aggregator

    from src.config import settings
    from src.ingestion.aviation._provider_awc import AWCProvider

    agg = ProviderAggregator()
    agg.register(AWCProvider())

    # Register optional providers if API keys are configured
    try:
        checkwx_key = getattr(settings, "CHECKWX_API_KEY", "")
        if checkwx_key:
            from src.ingestion.aviation._provider_checkwx import CheckWXProvider
            agg.register(CheckWXProvider(api_key=checkwx_key))
    except Exception:
        logger.debug("CheckWX provider not available", exc_info=True)

    try:
        avwx_key = getattr(settings, "AVWX_API_KEY", "")
        if avwx_key:
            from src.ingestion.aviation._provider_avwx import AVWXProvider
            agg.register(AVWXProvider(api_key=avwx_key))
    except Exception:
        logger.debug("AVWX provider not available", exc_info=True)

    # Providers that don't need API keys
    try:
        from src.ingestion.aviation._provider_iem import IEMProvider
        agg.register(IEMProvider())
    except Exception:
        logger.debug("IEM provider not available", exc_info=True)

    try:
        from src.ingestion.aviation._provider_ogimet import OGIMETProvider
        agg.register(OGIMETProvider())
    except Exception:
        logger.debug("OGIMET provider not available", exc_info=True)

    try:
        from src.ingestion.aviation._provider_noaa import NOAAProvider
        agg.register(NOAAProvider())
    except Exception:
        logger.debug("NOAA provider not available", exc_info=True)

    _aggregator = agg
    return _aggregator


def reset_aggregator() -> None:
    """Reset the aggregator singleton. Useful for testing."""
    global _aggregator
    _aggregator = None
