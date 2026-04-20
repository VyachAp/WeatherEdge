"""Tests for the signal generation pipeline (mapper, consensus, detector)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.db.models import TradeDirection
from src.signals.consensus import (
    ConsensusResult,
    _compute_weights,
    compute_consensus,
)
from src.signals.detector import (
    ActionableSignal,
    compute_edge,
    passes_filters,
)
from src.signals.mapper import (
    AviationContext,
    convert_threshold,
    geocode,
    normalize_operator,
    parse_target_date,
)


# ===================================================================
# mapper.py – geocoding
# ===================================================================


class TestGeocode:
    def test_known_city(self):
        coords = geocode("Phoenix")
        assert coords is not None
        assert pytest.approx(coords[0], abs=0.01) == 33.45

    def test_case_insensitive(self):
        assert geocode("phoenix") == geocode("Phoenix")

    def test_unknown_location_returns_none(self):
        assert geocode("Atlantis") is None

    def test_state_name(self):
        coords = geocode("Florida")
        assert coords is not None
        # Should resolve to state capital (Tallahassee)
        assert pytest.approx(coords[0], abs=0.1) == 30.4

    def test_substring_match(self):
        # "New York City" should match "new york city"
        coords = geocode("New York City")
        assert coords is not None


# ===================================================================
# mapper.py – operator mapping
# ===================================================================


class TestNormalizeOperator:
    def test_above(self):
        assert normalize_operator("above") == "above"

    def test_below(self):
        assert normalize_operator("below") == "below"

    def test_at_least_maps_to_above(self):
        assert normalize_operator("at_least") == "above"

    def test_at_most_maps_to_below(self):
        assert normalize_operator("at_most") == "below"

    def test_unknown_operator_returns_none(self):
        assert normalize_operator("record_breaking") is None


# ===================================================================
# mapper.py – date parsing
# ===================================================================


class TestParseDateTarget:
    def test_full_date(self):
        dt = parse_target_date("July 15, 2026")
        assert dt is not None
        assert dt.month == 7
        assert dt.day == 15
        assert dt.year == 2026
        assert dt.hour == 23
        assert dt.minute == 59

    def test_month_year_defaults_to_15th(self):
        dt = parse_target_date("January 2026")
        assert dt is not None
        assert dt.day == 15
        assert dt.month == 1
        assert dt.year == 2026
        assert dt.hour == 23
        assert dt.minute == 59

    def test_invalid_returns_none(self):
        assert parse_target_date("someday maybe") is None

    def test_returns_utc(self):
        dt = parse_target_date("July 15, 2026")
        assert dt is not None
        assert dt.tzinfo is not None

    def test_end_of_day(self):
        dt = parse_target_date("April 12")
        assert dt is not None
        assert dt.hour == 23
        assert dt.minute == 59
        assert dt.second == 59


# ===================================================================
# mapper.py – unit conversion
# ===================================================================


class TestConvertThreshold:
    def test_temperature_f_to_k(self):
        # 32°F = 273.15K
        assert pytest.approx(convert_threshold(32.0, "temperature"), abs=0.01) == 273.15

    def test_temperature_100f(self):
        # 100°F = 310.928K
        assert pytest.approx(convert_threshold(100.0, "temperature"), abs=0.1) == 310.93



# ===================================================================
# mapper.py – map_market
# ===================================================================


def _future_date_str(days: int = 3) -> str:
    """Return a date string N days from now, parseable by dateutil."""
    dt = datetime.now(tz=timezone.utc) + timedelta(days=days)
    return dt.strftime("%B %d, %Y")


def _short_range_date_str() -> str:
    """Return today's date string — parsed as end-of-day, within 30h window."""
    dt = datetime.now(tz=timezone.utc)
    return dt.strftime("%B %d, %Y")


def _make_market(**overrides):
    """Create a mock Market ORM object with sensible defaults."""
    m = MagicMock()
    m.id = overrides.get("id", "mkt_001")
    m.question = overrides.get("question", "Will it exceed 100°F in Phoenix?")
    m.parsed_location = overrides.get("parsed_location", "Phoenix")
    m.parsed_variable = overrides.get("parsed_variable", "temperature")
    m.parsed_threshold = overrides.get("parsed_threshold", 100.0)
    m.parsed_operator = overrides.get("parsed_operator", "above")
    m.parsed_target_date = overrides.get("parsed_target_date", _future_date_str(3))
    m.current_yes_price = overrides.get("current_yes_price", 0.45)
    m.volume = overrides.get("volume", 5000.0)
    m.liquidity = overrides.get("liquidity", 1000.0)
    m.end_date = overrides.get("end_date", None)
    return m


class TestMapMarket:
    @pytest.mark.asyncio
    @patch("src.signals.mapper.get_realtime_probability", new_callable=AsyncMock, return_value=0.70)
    async def test_valid_market_returns_signal(self, mock_avx):
        from src.signals.mapper import map_market

        market = _make_market()
        result = await map_market(market)

        assert result is not None
        assert result.market_id == "mkt_001"
        assert result.aviation_prob == 0.70
        assert result.market_prob == 0.45
        assert result.days_to_resolution > 0

    @pytest.mark.asyncio
    async def test_unsupported_variable_returns_none(self):
        from src.signals.mapper import map_market

        market = _make_market(parsed_variable="precipitation")
        result = await map_market(market)
        assert result is None

    @pytest.mark.asyncio
    async def test_unknown_location_returns_none(self):
        from src.signals.mapper import map_market

        market = _make_market(parsed_location="Nowhere Special")
        result = await map_market(market)
        assert result is None

    @pytest.mark.asyncio
    @patch("src.signals.mapper.get_realtime_probability", new_callable=AsyncMock, side_effect=Exception("Aviation down"))
    async def test_aviation_failure_returns_none(self, mock_avx):
        from src.signals.mapper import map_market

        market = _make_market()
        result = await map_market(market)
        assert result is None

    @pytest.mark.asyncio
    @patch("src.signals.mapper.get_realtime_probability", new_callable=AsyncMock, return_value=None)
    async def test_no_aviation_data_returns_none(self, mock_avx):
        from src.signals.mapper import map_market

        market = _make_market()
        result = await map_market(market)
        assert result is None

    @pytest.mark.asyncio
    @patch("src.signals.mapper.get_realtime_probability", new_callable=AsyncMock, return_value=0.70)
    async def test_occurs_operator_maps_to_below(self, mock_avx):
        from src.signals.mapper import map_market

        market = _make_market(parsed_operator="occurs")
        result = await map_market(market)
        # "occurs" now maps to "below" via OPERATOR_MAP
        assert result is not None


# ===================================================================
# consensus.py
# ===================================================================


class TestComputeConsensus:
    def test_aviation_only_short_range(self):
        result = compute_consensus(aviation_prob=0.75, hours_to_resolution=4.0)
        assert pytest.approx(result.consensus_prob, abs=0.001) == 0.75
        assert result.confidence == 0.80

    def test_aviation_only_medium_range(self):
        result = compute_consensus(aviation_prob=0.65, hours_to_resolution=10.0)
        assert pytest.approx(result.consensus_prob, abs=0.001) == 0.65
        assert result.confidence == 0.70

    def test_none_raises(self):
        with pytest.raises(ValueError):
            compute_consensus(None)

    def test_probability_passthrough_high(self):
        # Consensus no longer clamps to [0.01, 0.99]; raw aviation_prob passes through
        # so that narrow-bracket longshots (genuine ~0) aren't floored into fake edge.
        result = compute_consensus(aviation_prob=1.0, hours_to_resolution=4.0)
        assert result.consensus_prob == 1.0

    def test_probability_passthrough_low(self):
        result = compute_consensus(aviation_prob=0.0, hours_to_resolution=4.0)
        assert result.consensus_prob == 0.0


# ===================================================================
# detector.py – edge computation
# ===================================================================


class TestComputeEdge:
    def test_buy_yes_when_model_higher(self):
        edge, direction = compute_edge(0.70, 0.50)
        assert pytest.approx(edge, abs=0.001) == 0.20
        assert direction == TradeDirection.BUY_YES

    def test_buy_no_when_model_lower(self):
        edge, direction = compute_edge(0.30, 0.50)
        assert pytest.approx(edge, abs=0.001) == 0.20
        assert direction == TradeDirection.BUY_NO

    def test_zero_edge(self):
        edge, direction = compute_edge(0.50, 0.50)
        assert pytest.approx(edge, abs=0.001) == 0.0


# ===================================================================
# detector.py – filters
# ===================================================================


class TestPassesFilters:
    def test_passes_all(self):
        market = _make_market(liquidity=500.0, volume=200.0)
        assert passes_filters(0.15, 0.70, 3, market) is True

    def test_fails_min_edge(self):
        market = _make_market(liquidity=500.0, volume=200.0)
        assert passes_filters(0.05, 0.70, 3, market) is False

    def test_fails_low_liquidity(self):
        market = _make_market(liquidity=200.0, volume=200.0)
        assert passes_filters(0.15, 0.70, 3, market) is False

    def test_fails_low_confidence(self):
        market = _make_market(liquidity=500.0, volume=200.0)
        assert passes_filters(0.15, 0.40, 3, market) is False

    def test_fails_days_too_far(self):
        market = _make_market(liquidity=500.0, volume=200.0)
        assert passes_filters(0.15, 0.70, 10, market) is False

    def test_fails_days_too_soon(self):
        market = _make_market(liquidity=500.0, volume=200.0)
        assert passes_filters(0.15, 0.70, 0, market) is False

    def test_fails_low_volume(self):
        market = _make_market(liquidity=500.0, volume=50.0)
        assert passes_filters(0.15, 0.70, 3, market) is False


# ===================================================================
# End-to-end: detect_signals_short_range
# ===================================================================


class TestDetectSignalsE2E:
    """End-to-end test with mocked market data and aviation probabilities."""

    @staticmethod
    def _mock_session():
        """Create a mock session that handles dedup query."""
        session = AsyncMock()
        session.flush = AsyncMock()
        # Mock the dedup query: session.execute(stmt) → result with .all() returning []
        mock_result = MagicMock()
        mock_result.all.return_value = []
        session.execute = AsyncMock(return_value=mock_result)
        return session

    @pytest.mark.asyncio
    @patch("src.signals.mapper.get_active_weather_markets", new_callable=AsyncMock)
    @patch("src.signals.mapper.get_realtime_probability", new_callable=AsyncMock)
    @patch("src.signals.consensus.get_calibration_coefficients", new_callable=AsyncMock, return_value=None)
    async def test_end_to_end(self, mock_calib, mock_avx, mock_markets):
        from src.signals.detector import detect_signals_short_range

        # --- 3 markets: 1 good, 1 small edge, 1 unsupported variable ---
        good_market = _make_market(
            id="good_001",
            parsed_location="Phoenix",
            parsed_variable="temperature",
            parsed_threshold=100.0,
            parsed_operator="above",
            parsed_target_date=_short_range_date_str(),
            current_yes_price=0.45,
            liquidity=1000.0,
            volume=5000.0,
        )
        small_edge_market = _make_market(
            id="small_002",
            parsed_location="Denver",
            parsed_variable="temperature",
            parsed_threshold=90.0,
            parsed_operator="above",
            parsed_target_date=_short_range_date_str(),
            current_yes_price=0.48,
            liquidity=1000.0,
            volume=5000.0,
        )
        unsupported_market = _make_market(
            id="unsup_003",
            parsed_variable="precipitation",
        )

        mock_markets.return_value = [good_market, small_edge_market, unsupported_market]
        mock_avx.return_value = 0.70

        session = self._mock_session()

        result = await detect_signals_short_range(session)

        assert len(result) >= 1
        assert all(s.market_id != "unsup_003" for s in result)

        if len(result) > 1:
            assert result[0].ev_score >= result[1].ev_score

        good_sig = next(s for s in result if s.market_id == "good_001")
        assert good_sig.direction == TradeDirection.BUY_YES
        assert good_sig.edge > 0
        assert good_sig.confidence > 0.3

        assert session.add.called
        assert session.flush.called

    @pytest.mark.asyncio
    @patch("src.signals.mapper.get_active_weather_markets", new_callable=AsyncMock)
    @patch("src.signals.mapper.get_realtime_probability", new_callable=AsyncMock)
    @patch("src.signals.consensus.get_calibration_coefficients", new_callable=AsyncMock, return_value=None)
    async def test_no_markets_returns_empty(self, mock_calib, mock_avx, mock_markets):
        from src.signals.detector import detect_signals_short_range

        mock_markets.return_value = []
        session = self._mock_session()

        result = await detect_signals_short_range(session)
        assert result == []

    @pytest.mark.asyncio
    @patch("src.signals.mapper.get_active_weather_markets", new_callable=AsyncMock)
    @patch("src.signals.mapper.get_realtime_probability", new_callable=AsyncMock, return_value=0.46)
    @patch("src.signals.consensus.get_calibration_coefficients", new_callable=AsyncMock, return_value=None)
    async def test_insufficient_edge_filtered_out(self, mock_calib, mock_avx, mock_markets):
        from src.signals.detector import detect_signals_short_range

        # Market prob = 0.45, aviation ≈ 0.46 → edge < 0.10
        market = _make_market(
            current_yes_price=0.45,
            liquidity=1000.0,
            volume=5000.0,
            parsed_target_date=_short_range_date_str(),
        )
        mock_markets.return_value = [market]

        session = self._mock_session()

        result = await detect_signals_short_range(session)
        assert result == []


# ===================================================================
# Aviation integration – consensus weighting
# ===================================================================


class TestComputeWeights:
    def test_short_range_aviation(self):
        w = _compute_weights(4.0, has_aviation=True)
        assert w["aviation"] == 1.0

    def test_beyond_30h_no_aviation(self):
        w = _compute_weights(48.0, has_aviation=True)
        assert w["aviation"] == 0.0

    def test_no_aviation_flag(self):
        w = _compute_weights(4.0, has_aviation=False)
        assert w["aviation"] == 0.0

    def test_aviation_only(self):
        w = _compute_weights(4.0, has_aviation=True)
        assert w["aviation"] == 1.0


class TestConsensusWithAviation:
    def test_aviation_only_works(self):
        result = compute_consensus(aviation_prob=0.75, hours_to_resolution=4.0)
        assert pytest.approx(result.consensus_prob, abs=0.001) == 0.75
        assert result.confidence == 0.80  # aviation-only, <=6h

    def test_none_raises(self):
        with pytest.raises(ValueError):
            compute_consensus(None)

    def test_confidence_by_lead_time_6h(self):
        result = compute_consensus(aviation_prob=0.60, hours_to_resolution=4.0)
        assert result.confidence == 0.80

    def test_confidence_by_lead_time_12h(self):
        result = compute_consensus(aviation_prob=0.60, hours_to_resolution=10.0)
        assert result.confidence == 0.70

    def test_confidence_by_lead_time_24h(self):
        result = compute_consensus(aviation_prob=0.60, hours_to_resolution=18.0)
        assert result.confidence == 0.55

    def test_confidence_by_lead_time_30h(self):
        result = compute_consensus(aviation_prob=0.60, hours_to_resolution=28.0)
        assert result.confidence == 0.40


# ===================================================================
# Aviation integration – filter relaxation
# ===================================================================


class TestPassesFiltersAviation:
    def test_same_day_allowed_with_aviation(self):
        market = _make_market(liquidity=500.0, volume=200.0)
        assert passes_filters(0.15, 0.70, 0, market, aviation_prob=0.70, hours_to_resolution=4.0) is True

    def test_same_day_blocked_without_aviation(self):
        market = _make_market(liquidity=500.0, volume=200.0)
        assert passes_filters(0.15, 0.70, 0, market) is False

    def test_reduced_min_edge_with_aviation(self):
        market = _make_market(liquidity=500.0, volume=200.0)
        # Default MIN_EDGE=0.10, with aviation discount (0.75) → 0.075
        assert passes_filters(0.08, 0.70, 3, market, aviation_prob=0.70, hours_to_resolution=4.0) is True
        assert passes_filters(0.08, 0.70, 3, market) is False

    def test_aviation_beyond_30h_no_relaxation(self):
        market = _make_market(liquidity=500.0, volume=200.0)
        assert passes_filters(0.08, 0.70, 3, market, aviation_prob=0.70, hours_to_resolution=48.0) is False


# ===================================================================
# Aviation E2E: short-range market through full pipeline
# ===================================================================


class TestDetectSignalsAviationE2E:
    @pytest.mark.asyncio
    @patch("src.signals.mapper.get_active_weather_markets", new_callable=AsyncMock)
    @patch("src.signals.mapper.get_realtime_probability", new_callable=AsyncMock, return_value=0.75)
    @patch("src.signals.consensus.get_calibration_coefficients", new_callable=AsyncMock, return_value=None)
    async def test_short_range_with_aviation(self, mock_calib, mock_avx, mock_markets):
        from src.signals.detector import detect_signals_short_range

        # 1-day-out market with aviation data
        market = _make_market(
            id="short_001",
            parsed_location="Phoenix",
            parsed_variable="temperature",
            parsed_threshold=100.0,
            parsed_operator="above",
            parsed_target_date=_short_range_date_str(),
            current_yes_price=0.45,
            liquidity=1000.0,
            volume=5000.0,
        )
        mock_markets.return_value = [market]

        session = AsyncMock()
        session.flush = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = []
        session.execute = AsyncMock(return_value=mock_result)

        result = await detect_signals_short_range(session)

        assert len(result) >= 1
        sig = result[0]
        assert sig.market_id == "short_001"
        assert sig.aviation_prob == 0.75
        assert sig.hours_to_resolution > 0

        # Verify aviation_prob was persisted
        assert session.add.called
        call_args = session.add.call_args[0][0]
        assert call_args.aviation_prob == 0.75


# ===================================================================
# Tiered filters (ultra-short vs short-range vs standard)
# ===================================================================


class TestTieredFilters:
    def test_ultra_short_low_liquidity_passes(self):
        market = _make_market(liquidity=120.0, volume=60.0)
        assert passes_filters(
            0.08, 0.70, 0, market, aviation_prob=0.70, hours_to_resolution=4.0,
        ) is True

    def test_ultra_short_very_low_liquidity_fails(self):
        market = _make_market(liquidity=50.0, volume=60.0)
        assert passes_filters(
            0.08, 0.70, 0, market, aviation_prob=0.70, hours_to_resolution=4.0,
        ) is False

    def test_short_range_moderate_liquidity_passes(self):
        market = _make_market(liquidity=210.0, volume=80.0)
        assert passes_filters(
            0.08, 0.70, 1, market, aviation_prob=0.70, hours_to_resolution=20.0,
        ) is True

    def test_short_range_low_liquidity_fails(self):
        market = _make_market(liquidity=150.0, volume=80.0)
        assert passes_filters(
            0.08, 0.70, 1, market, aviation_prob=0.70, hours_to_resolution=20.0,
        ) is False

    def test_standard_filters_unchanged_without_aviation(self):
        market = _make_market(liquidity=500.0, volume=200.0)
        assert passes_filters(0.15, 0.70, 3, market) is True
        # Below standard thresholds
        market_low = _make_market(liquidity=200.0, volume=200.0)
        assert passes_filters(0.15, 0.70, 3, market_low) is False


# ===================================================================
# Aviation context confidence adjustments
# ===================================================================


class TestAviationContextConsensus:
    def test_taf_amendments_reduce_confidence(self):
        ctx = AviationContext(taf_amendment_count=5)
        result = compute_consensus(
            aviation_prob=0.60, hours_to_resolution=4.0,
            aviation_context=ctx,
        )
        result_no_ctx = compute_consensus(
            aviation_prob=0.60, hours_to_resolution=4.0,
        )
        # 5 amendments → 3 extra → -0.15
        assert result.confidence < result_no_ctx.confidence

    def test_speci_events_boost_confidence(self):
        ctx = AviationContext(speci_events_2h=2)
        result = compute_consensus(
            aviation_prob=0.80, hours_to_resolution=4.0,
            aviation_context=ctx,
        )
        result_no_ctx = compute_consensus(
            aviation_prob=0.80, hours_to_resolution=4.0,
        )
        assert result.confidence > result_no_ctx.confidence

    def test_severe_pireps_boost_confidence(self):
        ctx = AviationContext(has_severe_pireps=True)
        result = compute_consensus(
            aviation_prob=0.80, hours_to_resolution=4.0,
            aviation_context=ctx,
        )
        result_no_ctx = compute_consensus(
            aviation_prob=0.80, hours_to_resolution=4.0,
        )
        assert result.confidence > result_no_ctx.confidence

    def test_sigmet_boosts_confidence(self):
        ctx = AviationContext(active_sigmet_count=1)
        result = compute_consensus(
            aviation_prob=0.80, hours_to_resolution=4.0,
            aviation_context=ctx,
        )
        result_no_ctx = compute_consensus(
            aviation_prob=0.80, hours_to_resolution=4.0,
        )
        assert result.confidence > result_no_ctx.confidence

    def test_no_context_unchanged(self):
        result = compute_consensus(
            aviation_prob=0.60, hours_to_resolution=4.0,
            aviation_context=None,
        )
        result2 = compute_consensus(
            aviation_prob=0.60, hours_to_resolution=4.0,
        )
        assert result.confidence == result2.confidence


# ===================================================================
# mapper.py – map_exactly_market
# ===================================================================


class TestMapExactlyMarket:
    @pytest.mark.asyncio
    @patch("src.signals.mapper.aviation.get_bracket_probability", new_callable=AsyncMock, return_value=0.15)
    @patch("src.signals.mapper.aviation.taf_amendment_count", new_callable=AsyncMock, return_value=0)
    @patch("src.signals.mapper.aviation.detect_speci_events", new_callable=AsyncMock, return_value=[])
    @patch("src.signals.mapper.aviation.has_severe_weather_reports", new_callable=AsyncMock, return_value=False)
    @patch("src.signals.mapper.aviation.alerts_affecting_location", new_callable=AsyncMock, return_value=[])
    async def test_happy_path(self, _alerts, _severe, _speci, _amend, mock_bracket):
        from src.signals.mapper import map_exactly_market

        market = _make_market(parsed_operator="exactly", parsed_threshold=75.0)
        result = await map_exactly_market(market)

        assert result is not None
        assert result.market_id == "mkt_001"
        assert result.aviation_prob == 0.15
        mock_bracket.assert_called_once()
        call_args = mock_bracket.call_args
        assert call_args[0][1] == 74.0  # low_f = 75 - 1
        assert call_args[0][2] == 76.0  # high_f = 75 + 1

    @pytest.mark.asyncio
    async def test_missing_threshold_returns_none(self):
        from src.signals.mapper import map_exactly_market

        market = _make_market(parsed_operator="exactly", parsed_threshold=None)
        result = await map_exactly_market(market)
        assert result is None

    @pytest.mark.asyncio
    @patch("src.signals.mapper.aviation.get_bracket_probability", new_callable=AsyncMock, return_value=None)
    async def test_no_aviation_returns_none(self, _mock):
        from src.signals.mapper import map_exactly_market

        market = _make_market(parsed_operator="exactly", parsed_threshold=75.0)
        result = await map_exactly_market(market)
        assert result is None

    def test_choose_mapper_routes_exactly(self):
        from src.signals.mapper import _choose_mapper, map_exactly_market

        market = _make_market(parsed_operator="exactly")
        assert _choose_mapper(market) is map_exactly_market

    def test_choose_mapper_routes_bracket(self):
        from src.signals.mapper import _choose_mapper, map_bracket_market

        market = _make_market(parsed_operator="bracket")
        assert _choose_mapper(market) is map_bracket_market

    def test_choose_mapper_routes_default(self):
        from src.signals.mapper import _choose_mapper, map_market

        market = _make_market(parsed_operator="above")
        assert _choose_mapper(market) is map_market
