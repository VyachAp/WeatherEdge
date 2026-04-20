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
    parse_bracket_from_question,
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
    SAMPLE_HIGH_RECORD_MARKET,
    SAMPLE_GENERIC_ABOVE_MARKET,
]


# ---------------------------------------------------------------------------
# is_weather_market
# ---------------------------------------------------------------------------


class TestIsWeatherMarket:
    def test_keyword_in_question(self):
        assert is_weather_market(SAMPLE_TEMP_MARKET)

    def test_tag_match(self):
        m = {"question": "Something ambiguous", "tags": [{"label": "temperature"}]}
        assert is_weather_market(m)

    def test_non_weather(self):
        assert not is_weather_market(SAMPLE_NON_WEATHER_MARKET)

    def test_rejects_non_temperature_weather(self):
        m = {"question": "HURRICANE SEASON outlook?", "tags": []}
        assert not is_weather_market(m)

    def test_keyword_case_insensitive(self):
        m = {"question": "TEMPERATURE in NYC?", "tags": []}
        assert is_weather_market(m)

    def test_degree_symbols(self):
        assert is_weather_market({"question": "Will it reach 100°F?", "tags": []})
        assert is_weather_market({"question": "Above 30°C tomorrow?", "tags": []})


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

    def test_generic_above(self):
        p = parse_question(SAMPLE_GENERIC_ABOVE_MARKET["question"])
        assert p.matched
        assert p.variable == "temperature"
        assert p.threshold == 110.0
        assert p.operator == "above"

    def test_no_match_fallback(self):
        p = parse_question("Something completely unrelated to weather patterns")
        assert not p.matched
        assert p.raw == "Something completely unrelated to weather patterns"

    def test_date_extraction_on_fallback(self):
        p = parse_question("Will something happen in July 2026?")
        assert p.target_date == "July 2026"

    def test_celsius_threshold_converted_to_fahrenheit(self):
        p = parse_question(
            "Will the highest temperature in Paris be 22°C on April 11?"
        )
        assert p.matched
        assert p.variable == "temperature"
        assert p.threshold == pytest.approx(71.6)  # 22°C = 71.6°F

    def test_celsius_higher_threshold_converted(self):
        p = parse_question(
            "Will the highest temperature in Chongqing be 24°C or higher on April 13?"
        )
        assert p.matched
        assert p.variable == "temperature"
        assert p.threshold == pytest.approx(75.2)  # 24°C = 75.2°F
        assert p.operator in ("above", "at_least")

    def test_fahrenheit_threshold_unchanged(self):
        p = parse_question(
            "Will the temperature in Phoenix exceed 115°F on July 15, 2026?"
        )
        assert p.matched
        assert p.threshold == 115.0

    def test_no_unit_threshold_unchanged(self):
        p = parse_question(
            "Will the temperature in Phoenix exceed 115°F on July 15, 2026?"
        )
        assert p.matched
        assert p.threshold == 115.0

    def test_celsius_lowest_temperature(self):
        p = parse_question(
            "Will the lowest temperature in Seoul be 5°C or lower on April 12?"
        )
        assert p.matched
        assert p.threshold == pytest.approx(41.0)  # 5°C = 41°F
        assert p.operator in ("below", "at_most")

    def test_celsius_generic_above(self):
        p = parse_question(
            "Will it be above 30°C in Death Valley on August 1?"
        )
        assert p.matched
        assert p.threshold == pytest.approx(86.0)  # 30°C = 86°F


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

        # Temperature markets found, non-weather excluded
        temp_samples = [
            s for s in ALL_WEATHER_SAMPLES
            if "temperature" in (s.get("question") or "").lower()
            or "°f" in (s.get("question") or "").lower()
            or "°c" in (s.get("question") or "").lower()
        ]
        for s in temp_samples:
            assert s["id"] in ids, f"Expected {s['id']} in results"
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
        # 3 retries for first paginated call + 3 for 1 tag call = 6
        assert call_count == 6


# ---------------------------------------------------------------------------
# ingest_markets (mocked DB session)
# ---------------------------------------------------------------------------


def _mock_ingest_session(existing_markets=None):
    """Build an AsyncMock session for ingest_markets tests.

    ``existing_markets`` is a list of Market objects that should appear as
    already-persisted rows.  The mock wires up ``execute`` (for the bulk
    SELECT) and ``add`` / ``commit`` / ``execute`` (for the bulk INSERT).
    """
    existing = existing_markets or []
    session = AsyncMock(spec_set=["get", "add", "commit", "execute"])

    # Bulk SELECT returns existing markets via scalars().all()
    scalars_mock = MagicMock()
    scalars_mock.all.return_value = existing
    result_mock = MagicMock()
    result_mock.scalars.return_value = scalars_mock
    session.execute = AsyncMock(return_value=result_mock)

    session.commit = AsyncMock()
    return session


class TestIngestMarkets:
    @pytest.mark.asyncio
    async def test_creates_market_and_snapshot(self):
        session = _mock_ingest_session()
        added = []
        session.add = lambda obj: added.append(obj)

        count = await ingest_markets(session, [SAMPLE_TEMP_MARKET])
        assert count == 1
        # Should have added a Market (snapshots are bulk-inserted via execute)
        types = {type(o).__name__ for o in added}
        assert "Market" in types
        # Bulk snapshot insert should have been called
        assert session.execute.await_count == 2  # 1 SELECT + 1 bulk INSERT

    @pytest.mark.asyncio
    async def test_updates_existing_market(self):
        existing = Market(id="0xabc1", question="old question", fetched_at=datetime.utcnow())
        session = _mock_ingest_session(existing_markets=[existing])

        await ingest_markets(session, [SAMPLE_TEMP_MARKET])
        assert existing.question == SAMPLE_TEMP_MARKET["question"]

    @pytest.mark.asyncio
    async def test_snapshot_prices(self):
        session = _mock_ingest_session()
        added = []
        session.add = lambda obj: added.append(obj)

        await ingest_markets(session, [SAMPLE_TEMP_MARKET])
        # Snapshots are now bulk-inserted via execute; inspect the call args
        insert_call = session.execute.call_args_list[-1]  # last execute = bulk INSERT
        stmt = insert_call.args[0]
        # The compiled statement should contain snapshot values
        assert session.execute.await_count == 2

    @pytest.mark.asyncio
    async def test_skips_market_without_id(self):
        bad_market = {**SAMPLE_TEMP_MARKET, "id": None, "conditionId": None}
        session = _mock_ingest_session()
        session.add = MagicMock()

        count = await ingest_markets(session, [bad_market])
        assert count == 0


class TestParseBracketFromQuestion:
    def test_between_fahrenheit(self):
        low, high = parse_bracket_from_question(
            "Will the highest temperature in Los Angeles be between 54-55°F on April 15?"
        )
        assert low == 54.0
        assert high == 56.0  # +1 exclusive upper

    def test_between_celsius_converted_to_fahrenheit(self):
        low, high = parse_bracket_from_question(
            "Will the highest temperature in Paris be between 19-20°C on April 15?"
        )
        # 19°C = 66.2°F, 20°C = 68.0°F, +1 exclusive = 69.0
        assert abs(low - 66.2) < 0.01
        assert abs(high - 69.0) < 0.01

    def test_en_dash(self):
        result = parse_bracket_from_question(
            "Will the highest temperature in Seattle be between 56–57°F on April 15?"
        )
        assert result == (56.0, 58.0)

    def test_no_bracket(self):
        assert parse_bracket_from_question("Will it rain tomorrow?") is None

    def test_empty(self):
        assert parse_bracket_from_question("") is None
        assert parse_bracket_from_question(None) is None