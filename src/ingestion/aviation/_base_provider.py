"""Abstract base class for aviation weather data providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.ingestion.aviation._types import MinuteObs, Observation, SynopObs


class AviationProvider(ABC):
    """Base class for aviation weather data providers.

    Each provider implements the methods it supports.  Methods that a
    provider does not support return ``None`` / empty list by default so
    the aggregator can skip to the next provider in the chain.
    """

    name: str = "base"

    # ------------------------------------------------------------------
    # METAR
    # ------------------------------------------------------------------

    @abstractmethod
    async def fetch_metar(self, station: str) -> Observation | None:
        """Fetch the latest METAR for a station."""
        ...

    @abstractmethod
    async def fetch_metar_history(self, station: str, hours: int = 24) -> list[Observation]:
        """Fetch historical METARs for a station."""
        ...

    async def fetch_metar_bulk(self, stations: list[str]) -> dict[str, Observation]:
        """Fetch latest METARs for multiple stations. Default: sequential."""
        result: dict[str, Observation] = {}
        for stn in stations:
            obs = await self.fetch_metar(stn)
            if obs is not None:
                result[stn] = obs
        return result

    # ------------------------------------------------------------------
    # TAF
    # ------------------------------------------------------------------

    @abstractmethod
    async def fetch_taf(self, station: str) -> dict[str, Any] | None:
        """Fetch the latest TAF for a station."""
        ...

    async def fetch_taf_bulk(self, stations: list[str]) -> list[dict[str, Any]]:
        """Fetch latest TAFs for multiple stations. Default: sequential."""
        result: list[dict[str, Any]] = []
        for stn in stations:
            taf = await self.fetch_taf(stn)
            if taf is not None:
                result.append(taf)
        return result

    # ------------------------------------------------------------------
    # Optional capabilities (providers override what they support)
    # ------------------------------------------------------------------

    async def fetch_pireps(
        self, lat: float, lon: float, radius_nm: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch PIREPs near a location. Only AWC supports this."""
        return []

    async def fetch_sigmets(self) -> list[dict[str, Any]]:
        """Fetch active SIGMETs/AIRMETs. Only AWC supports this."""
        return []

    async def fetch_synop(self, wmo_id: str, hours: int = 24) -> list[SynopObs]:
        """Fetch SYNOP history. Only OGIMET supports this."""
        return []

    async def fetch_one_minute(self, station: str, hours: int = 6) -> list[MinuteObs]:
        """Fetch 1-minute ASOS data. Only IEM supports this (US stations)."""
        return []

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the provider is reachable and functional."""
        ...
