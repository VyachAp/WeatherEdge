"""Tests for the Polymarket weather-market scanner."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.db.models import Market, MarketSnapshot
from src.ingestion.polymarket import (
    ParsedQuestion,
    fetch_weather_markets,
    ingest_markets,
    is_weather_market,
    parse_question,
)

# ---------------------------------------------------------------------------
# Fixtures — sample Gamma API responses
# ---------------------------------------------------------------------------

SAMPLE_TEMP_MARKET = {
    "id": "0xabc1",
    "question": "Will the temperature in Phoenix exceed 115°F on July 15, 2026?",
    "slug": "phoenix-temp-115-jul-2026",
    "outcomes": ["Yes", "No"],
    "outcomePrices": ["0.35", "0.65"],
    "volume": "125000",
    "liquidity": "48000",
    "endDate": "2026-07-16T00:00:00Z",
    "resolutionSource": "NOAA",
    "tags": [{"label": "weather"}, {"label": "temperature"}],
    "active": True,
    "closed": False,
}

SAMPLE_HURRICANE_MARKET = {
    "id": "0xabc2",
    "question": "Will a hurricane make landfall in Florida in 2026?",
    "slug": "hurricane-landfall-florida-2026",
    "outcomes": ["Yes", "No"],
    "outcomePrices": ["0.42", "0.58"],
    "volume": "300000",
    "liquidity": "92000",
    "endDate": "2026-12-01T00:00:00Z",
    "resolutionSource": "NHC",
    "tags": [{"label": "weather"}],
    "active": True,
    "closed": False,
}

SAMPLE_SNOW_MARKET = {
    "id": "0xabc3",
    "question": "Will snowfall in Denver exceed 12 inches in January 2026?",
    "slug": "denver-snow-12in-jan-2026",
    "outcomes": ["Yes", "No"],
    "outcomePrices": ["0.55", "0.45"],
    "volume": "80000",
    "liquidity": "31000",
    "endDate": "2026-02-01T00:00:00Z",
    "tags": [{"label": "weather"}],
    "active": True,
    "closed": False,
}

SAMPLE_RAIN_MARKET = {
    "id": "0xabc4",
    "question": "Will rainfall in Houston exceed 5 inches on September 12?",
    "slug": "houston-rain-5in-sep",
    "outcomes": ["Yes", "No"],
    "outcomePrices": ["0.20", "0.80"],
    "volume": "50000",
    "liquidity": "19000",
    "endDate": "2026-09-13T00:00:00Z",
    "tags": [],
    "active": True,
    "closed": False,
}

SAMPLE_HEATWAVE_MARKET = {
    "id": "0xabc5",
    "question": "Will Chicago have 5 consecutive days above 95°F in July?",
    "slug": "chicago-heatwave-jul",
    "outcomes": ["Yes", "No"],
    "outcomePrices": ["0.15", "0.85"],
    "volume": "35000",
    "liquidity": "12000",
    "endDate": "2026-08-01T00:00:00Z",
    "tags": [{"label": "climate"}],
    "active": True,
    "closed": False,
}

SAMPLE_TORNADO_MARKET = {
    "id": "0xabc6",
    "question": "Will there be more than 20 tornadoes in Oklahoma in April 2026?",
    "slug": "oklahoma-tornadoes-apr-2026",
    "outcomes": ["Yes", "No"],
    "outcomePrices": ["0.30", "0.70"],
    "volume": "60000",
    "liquidity": "22000",
    "endDate": "2026-05-01T00:00:00Z",
    "tags": [{"label": "weather"}],
    "active": True,
    "closed": False,
}

SAMPLE_FREEZE_MARKET = {
    "id": "0xabc7",
    "question": "Will there be a freeze in Atlanta before November 15, 2026?",
    "slug": "atlanta-freeze-nov-2026",
    "outcomes": ["Yes", "No"],
    "outcomePrices": ["0.25", "0.75"],
    "volume": "40000",
    "liquidity": "15000",
    "endDate": "2026-11-16T00:00:00Z",
    "tags": [],
    "active": True,
    "closed": False,
}

SAMPLE_DROUGHT_MARKET = {
    "id": "0xabc8",
    "question": "Will a drought be declared in California in 2026?",
    "slug": "california-drought-2026",
    "outcomes": ["Yes", "No"],
    "outcomePrices": ["0.50", "0.50"],
    "volume": "200000",
    "liquidity": "75000",
    "endDate": "2027-01-01T00:00:00Z",
    "tags": [{"label": "climate"}],
    "active": True,
    "closed": False,
}

SAMPLE_WILDFIRE_MARKET = {
    "id": "0xabc9",
    "question": "Will wildfires burn more than 500,000 acres in California in 2026?",
    "slug": "california-wildfire-500k-2026",
    "outcomes": ["Yes", "No"],
    "outcomePrices": ["0.40", "0.60"],
    "volume": "150000",
    "liquidity": "55000",
    "endDate": "2027-01-01T00:00:00Z",
    "tags": [{"label": "weather"}],
    "active": True,
    "closed": False,
}

SAMPLE_WIND_MARKET = {
    "id": "0xabc10",
    "question": "Will wind speeds in Miami exceed 100 mph during hurricane season?",
    "slug": "miami-wind-100mph",
    "outcomes": ["Yes", "No"],
    "outcomePrices": ["0.22", "0.78"],
    "volume": "70000",
    "liquidity": "26000",
    "endDate": "2026-12-01T00:00:00Z",
    "tags": [{"label": "weather"}],
    "active": True,
    "closed": False,
}

SAMPLE_RECORD_MARKET = {
    "id": "0xabc11",
    "question": "Will 2026 be the hottest year on record in the US?",
    "slug": "2026-hottest-year-us",
    "outcomes": ["Yes", "No"],
    "outcomePrices": ["0.60", "0.40"],
    "volume": "500000",
    "liquidity": "180000",
    "endDate": "2027-02-01T00:00:00Z",
    "tags": [{"label": "climate"}],
    "active": True,
    "closed": False,
}

SAMPLE_HIGH_RECORD_MARKET = {
    "id": "0xabc12",
    "question": "Will New York City record a high above 100°F in July 2026?",
    "slug": "nyc-high-100f-jul-2026",
    "outcomes": ["Yes", "No"],
    "outcomePrices": ["0.10", "0.90"],
    "volume": "90000",
    "liquidity": "33000",
    "endDate": "2026-08-01T00:00:00Z",
    "tags": [{"label": "weather"}],
    "active": True,
    "closed": False,
}

SAMPLE_HURRICANE_CATEGORY_MARKET = {
    "id": "0xabc13",
    "question": "Will Hurricane Milton reach Category 5?",
    "slug": "hurricane-milton-cat5",
    "outcomes": ["Yes", "No"],
    "outcomePrices": ["0.08", "0.92"],
    "volume": "250000",
    "liquidity": "95000",
    "endDate": "2026-11-30T00:00:00Z",
    "tags": [{"label": "weather"}],
    "active": True,
    "closed": False,
}

SAMPLE_GENERIC_ABOVE_MARKET = {
    "id": "0xabc14",
    "question": "Will it be above 110°F in Death Valley on August 1?",
    "slug": "death-valley-110f-aug1",
    "outcomes": ["Yes", "No"],
    "outcomePrices": ["0.70", "0.30"],
    "volume": "45000",
    "liquidity": "17000",
    "endDate": "2026-08-02T00:00:00Z",
    "tags": [],
    "active": True,
    "closed": False,
}

SAMPLE_NON_WEATHER_MARKET = {
    "id": "0xfff1",
    "question": "Will Bitcoin reach $100,000 by December 2026?",
    "slug": "btc-100k-dec-2026",
    "outcomes": ["Yes", "No"],
    "outcomePrices": ["0.45", "0.55"],
    "volume": "1000000",
    "liquidity": "400000",
    "endDate": "2027-01-01T00:00:00Z",
    "tags": [{"label": "crypto"}],
    "active": True,
    "closed": False,
}

ALL_WEATHER_SAMPLES = [
    SAMPLE_TEMP_MARKET,
    SAMPLE_HURRICANE_MARKET,
    SAMPLE_SNOW_MARKET,
    SAMPLE_RAIN_MARKET,
    SAMPLE_HEATWAVE_MARKET,
    SAMPLE_TORNADO_MARKET,
    SAMPLE_FREEZE_MARKET,
    SAMPLE_DROUGHT_MARKET,
    SAMPLE_WILDFIRE_MARKET,
    SAMPLE_WIND_MARKET,
    SAMPLE_RECORD_MARKET,
    SAMPLE_HIGH_RECORD_MARKET,
    SAMPLE_HURRICANE_CATEGORY_MARKET,
    SAMPLE_GENERIC_ABOVE_MARKET,
]


# ---------------------------------------------------------------------------
# is_weather_market
# ---------------------------------------------------------------------------


class TestIsWeatherMarket:
    def test_keyword_in_question(self):
        assert is_weather_market(SAMPLE_TEMP_MARKET)

    def test_tag_match(self):
        m = {"question": "Something ambiguous", "tags": [{"label": "weather"}]}
        assert is_weather_market(m)

    def test_non_weather(self):
        assert not is_weather_market(SAMPLE_NON_WEATHER_MARKET)

    def test_empty_question_with_climate_tag(self):
        m = {"question": "", "tags": [{"label": "climate"}]}
        assert is_weather_market(m)

    def test_keyword_case_insensitive(self):
        m = {"question": "HURRICANE SEASON outlook?", "tags": []}
        assert is_weather_market(m)


# ---------------------------------------------------------------------------
# parse_question — one test per pattern
# ---------------------------------------------------------------------------


class TestParseQuestion:
    def test_temperature_threshold(self):
        p = parse_question(SAMPLE_TEMP_MARKET["question"])
        assert p.matched
        assert p.variable == "temperature"
        assert p.location == "Phoenix"
        assert p.threshold == 115.0
        assert p.operator == "above"

    def test_high_record(self):
        p = parse_question(SAMPLE_HIGH_RECORD_MARKET["question"])
        assert p.matched
        assert p.variable == "temperature"
        assert "New York" in p.location
        assert p.threshold == 100.0
        assert p.operator == "above"

    def test_precipitation(self):
        p = parse_question(SAMPLE_RAIN_MARKET["question"])
        assert p.matched
        assert p.variable == "precipitation"
        assert p.location == "Houston"
        assert p.threshold == 5.0
        assert p.operator == "above"

    def test_snowfall(self):
        p = parse_question(SAMPLE_SNOW_MARKET["question"])
        assert p.matched
        assert p.variable == "snowfall"
        assert p.location == "Denver"
        assert p.threshold == 12.0

    def test_hurricane_landfall(self):
        p = parse_question(SAMPLE_HURRICANE_MARKET["question"])
        assert p.matched
        assert p.variable == "hurricane_landfall"
        assert p.location == "Florida"
        assert p.operator == "occurs"

    def test_hurricane_category(self):
        p = parse_question(SAMPLE_HURRICANE_CATEGORY_MARKET["question"])
        assert p.matched
        assert p.variable == "hurricane_category"
        assert p.threshold == 5.0

    def test_heat_wave(self):
        p = parse_question(SAMPLE_HEATWAVE_MARKET["question"])
        assert p.matched
        assert p.variable == "heat_wave"
        assert p.location == "Chicago"
        assert p.threshold == 95.0
        assert p.operator == "above"

    def test_freeze(self):
        p = parse_question(SAMPLE_FREEZE_MARKET["question"])
        assert p.matched
        assert p.variable == "freeze"
        assert p.location == "Atlanta"
        assert p.operator == "occurs"

    def test_drought(self):
        p = parse_question(SAMPLE_DROUGHT_MARKET["question"])
        assert p.matched
        assert p.variable == "drought"
        assert p.location == "California"
        assert p.operator == "occurs"

    def test_tornado_count(self):
        p = parse_question(SAMPLE_TORNADO_MARKET["question"])
        assert p.matched
        assert p.variable == "tornado_count"
        assert p.location == "Oklahoma"
        assert p.threshold == 20.0

    def test_wildfire_acreage(self):
        p = parse_question(SAMPLE_WILDFIRE_MARKET["question"])
        assert p.matched
        assert p.variable == "wildfire_acreage"
        assert p.location == "California"
        assert p.threshold == 500000.0
        assert p.operator == "above"

    def test_wind_speed(self):
        p = parse_question(SAMPLE_WIND_MARKET["question"])
        assert p.matched
        assert p.variable == "wind_speed"
        assert p.location == "Miami"
        assert p.threshold == 100.0

    def test_generic_above(self):
        p = parse_question(SAMPLE_GENERIC_ABOVE_MARKET["question"])
        assert p.matched
        assert p.variable == "temperature"
        assert p.threshold == 110.0
        assert p.operator == "above"

    def test_record_breaking(self):
        p = parse_question(SAMPLE_RECORD_MARKET["question"])
        assert p.matched
        assert p.variable == "record"
        assert "US" in p.location

    def test_no_match_fallback(self):
        p = parse_question("Something completely unrelated to weather patterns")
        assert not p.matched
        assert p.raw == "Something completely unrelated to weather patterns"

    def test_date_extraction_on_fallback(self):
        p = parse_question("Will something happen in July 2026?")
        assert p.target_date == "July 2026"


# ---------------------------------------------------------------------------
# fetch_weather_markets (mocked HTTP)
# ---------------------------------------------------------------------------


class TestFetchWeatherMarkets:
    @pytest.mark.asyncio
    async def test_fetches_and_filters(self):
        mixed_batch = ALL_WEATHER_SAMPLES + [SAMPLE_NON_WEATHER_MARKET]

        async def mock_get(url, params=None, timeout=None):
            resp = MagicMock(spec=httpx.Response)
            resp.status_code = 200
            # First paginated call returns data, second returns empty
            tag = (params or {}).get("tag")
            offset = (params or {}).get("offset", 0)
            if tag:
                resp.json.return_value = []
            elif offset == 0:
                resp.json.return_value = mixed_batch
            else:
                resp.json.return_value = []
            resp.raise_for_status = MagicMock()
            return resp

        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = mock_get
        client.aclose = AsyncMock()

        markets = await fetch_weather_markets(client=client)
        ids = {m["id"] for m in markets}

        # All weather markets found, non-weather excluded
        for s in ALL_WEATHER_SAMPLES:
            assert s["id"] in ids
        assert SAMPLE_NON_WEATHER_MARKET["id"] not in ids

    @pytest.mark.asyncio
    async def test_deduplicates_tag_results(self):
        async def mock_get(url, params=None, timeout=None):
            resp = MagicMock(spec=httpx.Response)
            resp.status_code = 200
            tag = (params or {}).get("tag")
            offset = (params or {}).get("offset", 0)
            if tag == "weather":
                resp.json.return_value = [SAMPLE_TEMP_MARKET]
            elif tag == "climate":
                resp.json.return_value = [SAMPLE_TEMP_MARKET]  # duplicate
            elif offset == 0:
                resp.json.return_value = [SAMPLE_TEMP_MARKET]
            else:
                resp.json.return_value = []
            resp.raise_for_status = MagicMock()
            return resp

        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = mock_get
        client.aclose = AsyncMock()

        markets = await fetch_weather_markets(client=client)
        assert len(markets) == 1

    @pytest.mark.asyncio
    async def test_handles_api_error_gracefully(self):
        call_count = 0

        async def mock_get(url, params=None, timeout=None):
            nonlocal call_count
            call_count += 1
            raise httpx.TransportError("connection failed")

        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = mock_get
        client.aclose = AsyncMock()

        # Should not raise — returns empty list after retries exhaust
        markets = await fetch_weather_markets(client=client)
        assert markets == []
        # 3 retries for first paginated call + 3 each for 2 tag calls = 9
        assert call_count == 9


# ---------------------------------------------------------------------------
# ingest_markets (mocked DB session)
# ---------------------------------------------------------------------------


class TestIngestMarkets:
    @pytest.mark.asyncio
    async def test_creates_market_and_snapshot(self):
        session = AsyncMock(spec_set=["get", "add", "commit"])
        session.get = AsyncMock(return_value=None)  # no existing market
        added = []
        session.add = lambda obj: added.append(obj)
        session.commit = AsyncMock()

        count = await ingest_markets(session, [SAMPLE_TEMP_MARKET])
        assert count == 1
        # Should have added a Market and a MarketSnapshot
        types = {type(o).__name__ for o in added}
        assert "Market" in types
        assert "MarketSnapshot" in types

    @pytest.mark.asyncio
    async def test_updates_existing_market(self):
        existing = Market(id="0xabc1", question="old question", fetched_at=datetime.utcnow())
        session = AsyncMock(spec_set=["get", "add", "commit"])
        session.get = AsyncMock(return_value=existing)
        session.add = MagicMock()
        session.commit = AsyncMock()

        await ingest_markets(session, [SAMPLE_TEMP_MARKET])
        assert existing.question == SAMPLE_TEMP_MARKET["question"]

    @pytest.mark.asyncio
    async def test_snapshot_prices(self):
        session = AsyncMock(spec_set=["get", "add", "commit"])
        session.get = AsyncMock(return_value=None)
        added = []
        session.add = lambda obj: added.append(obj)
        session.commit = AsyncMock()

        await ingest_markets(session, [SAMPLE_TEMP_MARKET])
        snapshots = [o for o in added if isinstance(o, MarketSnapshot)]
        assert len(snapshots) == 1
        assert snapshots[0].yes_price == 0.35
        assert snapshots[0].no_price == pytest.approx(0.65)

    @pytest.mark.asyncio
    async def test_skips_market_without_id(self):
        bad_market = {**SAMPLE_TEMP_MARKET, "id": None, "conditionId": None}
        session = AsyncMock(spec_set=["get", "add", "commit"])
        session.get = AsyncMock(return_value=None)
        session.add = MagicMock()
        session.commit = AsyncMock()

        count = await ingest_markets(session, [bad_market])
        assert count == 0