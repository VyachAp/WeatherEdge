"""Tests for the provider aggregator failover logic."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from src.ingestion.aviation import _aggregator
from src.ingestion.aviation._aggregator import FAILOVER_CHAINS, ProviderAggregator, reset_aggregator
from src.ingestion.aviation._base_provider import AviationProvider
from src.ingestion.aviation._caching import clear_all_caches
from src.ingestion.aviation._types import Observation


@pytest.fixture(autouse=True)
def _cleanup():
    clear_all_caches()
    reset_aggregator()
    yield
    clear_all_caches()
    reset_aggregator()


@pytest.fixture
def _override_chains():
    """Override failover chains for test providers, restore after."""
    original = {k: v[:] for k, v in FAILOVER_CHAINS.items()}
    FAILOVER_CHAINS["metar_realtime"] = ["provider_a", "provider_b"]
    FAILOVER_CHAINS["metar_history"] = ["provider_a", "provider_b"]
    FAILOVER_CHAINS["taf"] = ["provider_a", "provider_b"]
    yield
    FAILOVER_CHAINS.update(original)


def _make_observation(station: str = "KPHX", source: str = "test") -> Observation:
    return Observation(
        station_icao=station,
        observed_at=datetime.now(timezone.utc),
        temp_c=30.0,
        temp_f=86.0,
        source=source,
    )


class MockProviderA(AviationProvider):
    name = "provider_a"

    def __init__(self, fail: bool = False, return_none: bool = False):
        self._fail = fail
        self._return_none = return_none

    async def fetch_metar(self, station):
        if self._fail:
            raise ConnectionError("Provider A down")
        if self._return_none:
            return None
        return _make_observation(station, "provider_a")

    async def fetch_metar_history(self, station, hours=24):
        if self._fail:
            raise ConnectionError("Provider A down")
        return [_make_observation(station, "provider_a")]

    async def fetch_taf(self, station):
        if self._fail:
            raise ConnectionError("Provider A down")
        return {"station_icao": station, "periods": [], "source": "provider_a"}

    async def health_check(self):
        return not self._fail


class MockProviderB(AviationProvider):
    name = "provider_b"

    async def fetch_metar(self, station):
        return _make_observation(station, "provider_b")

    async def fetch_metar_history(self, station, hours=24):
        return [_make_observation(station, "provider_b")]

    async def fetch_taf(self, station):
        return {"station_icao": station, "periods": [], "source": "provider_b"}

    async def health_check(self):
        return True


class TestAggregatorFailover:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_primary_succeeds(self, _override_chains):
        agg = ProviderAggregator()
        agg.register(MockProviderA())
        agg.register(MockProviderB())

        result = await agg.get_metar("KPHX")
        assert result is not None
        assert result.source == "provider_a"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_failover_on_error(self, _override_chains):
        agg = ProviderAggregator()
        agg.register(MockProviderA(fail=True))
        agg.register(MockProviderB())

        result = await agg.get_metar("KPHX")
        assert result is not None
        assert result.source == "provider_b"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_failover_on_none(self, _override_chains):
        agg = ProviderAggregator()
        agg.register(MockProviderA(return_none=True))
        agg.register(MockProviderB())

        result = await agg.get_metar("KPHX")
        assert result is not None
        assert result.source == "provider_b"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_all_providers_fail(self, _override_chains):
        FAILOVER_CHAINS["metar_realtime"] = ["provider_a"]
        agg = ProviderAggregator()
        agg.register(MockProviderA(fail=True))

        result = await agg.get_metar("KPHX")
        assert result is None

    @pytest.mark.asyncio(loop_scope="function")
    async def test_health_marking(self, _override_chains):
        FAILOVER_CHAINS["metar_realtime"] = ["provider_a"]
        agg = ProviderAggregator()
        agg.register(MockProviderA(fail=True))

        await agg.get_metar("KPHX")
        assert agg._health["provider_a"] is False

    @pytest.mark.asyncio(loop_scope="function")
    async def test_no_providers_registered(self):
        agg = ProviderAggregator()
        result = await agg.get_metar("KPHX")
        assert result is None

    @pytest.mark.asyncio(loop_scope="function")
    async def test_taf_failover(self, _override_chains):
        agg = ProviderAggregator()
        agg.register(MockProviderA(fail=True))
        agg.register(MockProviderB())

        result = await agg.get_taf("KPHX")
        assert result is not None
        assert result["source"] == "provider_b"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_history_failover(self, _override_chains):
        agg = ProviderAggregator()
        agg.register(MockProviderA(fail=True))
        agg.register(MockProviderB())

        result = await agg.get_metar_history("KPHX")
        assert len(result) == 1
        assert result[0].source == "provider_b"
