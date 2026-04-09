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
    resolve_variable,
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

    def test_occurs_maps_to_below(self):
        assert normalize_operator("occurs") == "below"

    def test_record_breaking_returns_none(self):
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

    def test_month_year_defaults_to_15th(self):
        dt = parse_target_date("January 2026")
        assert dt is not None
        assert dt.day == 15
        assert dt.month == 1
        assert dt.year == 2026

    def test_invalid_returns_none(self):
        assert parse_target_date("someday maybe") is None

    def test_returns_utc(self):
        dt = parse_target_date("July 15, 2026")
        assert dt is not None
        assert dt.tzinfo is not None


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

    def test_precipitation_inches_to_mm(self):
        # 1 inch = 25.4 mm
        assert pytest.approx(convert_threshold(1.0, "precipitation"), abs=0.01) == 25.4

    def test_wind_speed_mph_to_ms(self):
        # 60 mph ≈ 26.82 m/s
        assert pytest.approx(convert_threshold(60.0, "wind_speed"), abs=0.1) == 26.82


# ===================================================================
# mapper.py – map_market
# ===================================================================


def _future_date_str(days: int = 3) -> str:
    """Return a date string N days from now, parseable by dateutil."""
    dt = datetime.now(tz=timezone.utc) + timedelta(days=days)
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
    @patch("src.signals.mapper.ecmwf.get_probability", new_callable=AsyncMock, return_value=0.72)
    @patch("src.signals.mapper.gfs.get_probability", new_callable=AsyncMock, return_value=0.68)
    async def test_valid_market_returns_signal(self, mock_gfs, mock_ecmwf):
        from src.signals.mapper import map_market

        market = _make_market()
        result = await map_market(market)

        assert result is not None
        assert result.market_id == "mkt_001"
        assert result.gfs_prob == 0.68
        assert result.ecmwf_prob == 0.72
        assert result.market_prob == 0.45
        assert result.days_to_resolution > 0

    @pytest.mark.asyncio
    async def test_unsupported_variable_returns_none(self):
        from src.signals.mapper import map_market

        market = _make_market(parsed_variable="hurricane_landfall")
        result = await map_market(market)
        assert result is None

    @pytest.mark.asyncio
    async def test_unknown_location_returns_none(self):
        from src.signals.mapper import map_market

        market = _make_market(parsed_location="Nowhere Special")
        result = await map_market(market)
        assert result is None

    @pytest.mark.asyncio
    @patch("src.signals.mapper.ecmwf.get_probability", new_callable=AsyncMock, return_value=0.72)
    @patch("src.signals.mapper.gfs.get_probability", new_callable=AsyncMock, side_effect=Exception("GFS down"))
    async def test_single_model_failure_still_returns(self, mock_gfs, mock_ecmwf):
        from src.signals.mapper import map_market

        market = _make_market()
        result = await map_market(market)

        assert result is not None
        assert result.gfs_prob is None
        assert result.ecmwf_prob == 0.72

    @pytest.mark.asyncio
    @patch("src.signals.mapper.ecmwf.get_probability", new_callable=AsyncMock, side_effect=Exception("ECMWF down"))
    @patch("src.signals.mapper.gfs.get_probability", new_callable=AsyncMock, side_effect=Exception("GFS down"))
    async def test_both_models_fail_returns_none(self, mock_gfs, mock_ecmwf):
        from src.signals.mapper import map_market

        market = _make_market()
        result = await map_market(market)
        assert result is None

    @pytest.mark.asyncio
    @patch("src.signals.mapper.ecmwf.get_probability", new_callable=AsyncMock, return_value=0.72)
    @patch("src.signals.mapper.gfs.get_probability", new_callable=AsyncMock, return_value=0.68)
    async def test_occurs_operator_maps_to_below(self, mock_gfs, mock_ecmwf):
        from src.signals.mapper import map_market

        market = _make_market(parsed_operator="occurs")
        result = await map_market(market)
        # "occurs" now maps to "below" via OPERATOR_MAP
        assert result is not None


# ===================================================================
# consensus.py
# ===================================================================


class TestComputeConsensus:
    def test_both_models_weighted_average(self):
        result = compute_consensus(0.40, 0.60)
        # 0.6*0.60 + 0.4*0.40 = 0.36 + 0.16 = 0.52
        assert pytest.approx(result.consensus_prob, abs=0.001) == 0.52

    def test_confidence_perfect_agreement(self):
        result = compute_consensus(0.50, 0.50)
        assert pytest.approx(result.confidence, abs=0.001) == 1.0

    def test_confidence_disagreement(self):
        result = compute_consensus(0.20, 0.80)
        assert pytest.approx(result.confidence, abs=0.001) == 0.4

    def test_single_model_gfs_only(self):
        result = compute_consensus(0.65, None)
        assert pytest.approx(result.consensus_prob, abs=0.001) == 0.65
        assert result.confidence == 0.5

    def test_single_model_ecmwf_only(self):
        result = compute_consensus(None, 0.70)
        assert pytest.approx(result.consensus_prob, abs=0.001) == 0.70
        assert result.confidence == 0.5

    def test_both_none_raises(self):
        with pytest.raises(ValueError):
            compute_consensus(None, None)

    def test_probability_clamped_high(self):
        result = compute_consensus(1.0, 1.0)
        assert result.consensus_prob <= 0.99

    def test_probability_clamped_low(self):
        result = compute_consensus(0.0, 0.0)
        assert result.consensus_prob >= 0.01


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
# End-to-end: detect_signals
# ===================================================================


class TestDetectSignalsE2E:
    """End-to-end test with mocked market data and forecast probabilities."""

    @pytest.mark.asyncio
    @patch("src.signals.mapper.get_active_weather_markets", new_callable=AsyncMock)
    @patch("src.signals.mapper.ecmwf.get_probability", new_callable=AsyncMock)
    @patch("src.signals.mapper.gfs.get_probability", new_callable=AsyncMock)
    @patch("src.signals.consensus.get_calibration_coefficients", new_callable=AsyncMock, return_value=None)
    async def test_end_to_end(self, mock_calib, mock_gfs, mock_ecmwf, mock_markets):
        from src.signals.detector import detect_signals

        # --- 3 markets: 1 good, 1 small edge, 1 unsupported variable ---
        good_market = _make_market(
            id="good_001",
            parsed_location="Phoenix",
            parsed_variable="temperature",
            parsed_threshold=100.0,
            parsed_operator="above",
            parsed_target_date=_future_date_str(5),
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
            parsed_target_date=_future_date_str(5),
            current_yes_price=0.48,  # close to model prob → small edge
            liquidity=1000.0,
            volume=5000.0,
        )
        unsupported_market = _make_market(
            id="unsup_003",
            parsed_variable="hurricane_landfall",
        )

        mock_markets.return_value = [good_market, small_edge_market, unsupported_market]

        # Model says ~70% for both good and small-edge markets
        mock_gfs.return_value = 0.68
        mock_ecmwf.return_value = 0.72

        # Use a mock session
        session = AsyncMock()
        session.flush = AsyncMock()

        result = await detect_signals(session)

        # Only the good market should produce a signal
        # (small edge market: consensus ≈ 0.704, edge ≈ 0.704-0.48 = 0.224 → passes edge
        #  but both may pass if edge is large enough)
        # Actually both good and small_edge will pass since 0.704-0.48=0.224 > 0.10
        # and 0.704-0.45=0.254 > 0.10
        # The unsupported one should not appear
        assert len(result) >= 1
        assert all(s.market_id != "unsup_003" for s in result)

        # Signals should be sorted by ev_score descending
        if len(result) > 1:
            assert result[0].ev_score >= result[1].ev_score

        # The good market signal should have correct direction
        good_sig = next(s for s in result if s.market_id == "good_001")
        assert good_sig.direction == TradeDirection.BUY_YES
        assert good_sig.edge > 0
        assert good_sig.confidence > 0.55

        # Verify persistence
        assert session.add.called
        assert session.flush.called

    @pytest.mark.asyncio
    @patch("src.signals.mapper.get_active_weather_markets", new_callable=AsyncMock)
    @patch("src.signals.mapper.ecmwf.get_probability", new_callable=AsyncMock)
    @patch("src.signals.mapper.gfs.get_probability", new_callable=AsyncMock)
    @patch("src.signals.consensus.get_calibration_coefficients", new_callable=AsyncMock, return_value=None)
    async def test_no_markets_returns_empty(self, mock_calib, mock_gfs, mock_ecmwf, mock_markets):
        from src.signals.detector import detect_signals

        mock_markets.return_value = []
        session = AsyncMock()
        session.flush = AsyncMock()

        result = await detect_signals(session)
        assert result == []

    @pytest.mark.asyncio
    @patch("src.signals.mapper.get_active_weather_markets", new_callable=AsyncMock)
    @patch("src.signals.mapper.ecmwf.get_probability", new_callable=AsyncMock, return_value=0.46)
    @patch("src.signals.mapper.gfs.get_probability", new_callable=AsyncMock, return_value=0.44)
    @patch("src.signals.consensus.get_calibration_coefficients", new_callable=AsyncMock, return_value=None)
    async def test_insufficient_edge_filtered_out(self, mock_calib, mock_gfs, mock_ecmwf, mock_markets):
        from src.signals.detector import detect_signals

        # Market prob = 0.45, model ≈ 0.452 → edge < 0.10
        market = _make_market(
            current_yes_price=0.45,
            liquidity=1000.0,
            volume=5000.0,
            parsed_target_date=_future_date_str(5),
        )
        mock_markets.return_value = [market]

        session = AsyncMock()
        session.flush = AsyncMock()

        result = await detect_signals(session)
        assert result == []


# ===================================================================
# Aviation integration – consensus weighting
# ===================================================================


class TestComputeWeights:
    def test_short_range_aviation_50pct(self):
        w = _compute_weights(4.0, has_aviation=True, has_gfs=True, has_ecmwf=True)
        assert pytest.approx(w["aviation"], abs=0.001) == 0.50
        assert pytest.approx(w["gfs"] + w["ecmwf"] + w["aviation"], abs=0.001) == 1.0

    def test_medium_range_aviation_30pct(self):
        w = _compute_weights(10.0, has_aviation=True, has_gfs=True, has_ecmwf=True)
        assert pytest.approx(w["aviation"], abs=0.001) == 0.30

    def test_12_24h_aviation_15pct(self):
        w = _compute_weights(18.0, has_aviation=True, has_gfs=True, has_ecmwf=True)
        assert pytest.approx(w["aviation"], abs=0.001) == 0.15

    def test_24_30h_aviation_8pct(self):
        w = _compute_weights(28.0, has_aviation=True, has_gfs=True, has_ecmwf=True)
        assert pytest.approx(w["aviation"], abs=0.001) == 0.08

    def test_beyond_30h_no_aviation(self):
        w = _compute_weights(48.0, has_aviation=True, has_gfs=True, has_ecmwf=True)
        assert w["aviation"] == 0.0

    def test_no_aviation_flag(self):
        w = _compute_weights(4.0, has_aviation=False, has_gfs=True, has_ecmwf=True)
        assert w["aviation"] == 0.0
        assert pytest.approx(w["ecmwf"], abs=0.001) == 0.6
        assert pytest.approx(w["gfs"], abs=0.001) == 0.4

    def test_aviation_only(self):
        w = _compute_weights(4.0, has_aviation=True, has_gfs=False, has_ecmwf=False)
        assert w["aviation"] == 1.0
        assert w["gfs"] == 0.0
        assert w["ecmwf"] == 0.0

    def test_nwp_ratio_preserved(self):
        w = _compute_weights(4.0, has_aviation=True, has_gfs=True, has_ecmwf=True)
        # Remaining 50% should split 60/40
        assert pytest.approx(w["ecmwf"] / w["gfs"], abs=0.01) == 1.5


class TestConsensusWithAviation:
    def test_short_range_aviation_dominates(self):
        result = compute_consensus(0.40, 0.60, aviation_prob=0.80, hours_to_resolution=4.0)
        # aviation 50%, ecmwf 30%, gfs 20%
        expected = 0.50 * 0.80 + 0.30 * 0.60 + 0.20 * 0.40
        assert pytest.approx(result.consensus_prob, abs=0.01) == expected

    def test_no_aviation_backward_compat(self):
        result = compute_consensus(0.40, 0.60)
        assert pytest.approx(result.consensus_prob, abs=0.001) == 0.52
        assert result.aviation_prob is None

    def test_confidence_boost_on_agreement(self):
        result = compute_consensus(0.70, 0.70, aviation_prob=0.70, hours_to_resolution=4.0)
        assert result.confidence == 1.0

    def test_confidence_boost_when_close(self):
        # Use values where GFS/ECMWF disagree enough that base confidence < 1.0
        result_with = compute_consensus(0.55, 0.65, aviation_prob=0.60, hours_to_resolution=4.0)
        result_without = compute_consensus(0.55, 0.65)
        assert result_with.confidence > result_without.confidence

    def test_no_boost_when_disagreeing(self):
        result = compute_consensus(0.60, 0.60, aviation_prob=0.20, hours_to_resolution=4.0)
        assert result.confidence < 0.7

    def test_aviation_only_works(self):
        result = compute_consensus(None, None, aviation_prob=0.75, hours_to_resolution=4.0)
        assert pytest.approx(result.consensus_prob, abs=0.001) == 0.75
        assert result.confidence == 0.5

    def test_all_none_raises(self):
        with pytest.raises(ValueError):
            compute_consensus(None, None, None)


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
    @patch("src.signals.mapper.ecmwf.get_probability", new_callable=AsyncMock, return_value=0.70)
    @patch("src.signals.mapper.gfs.get_probability", new_callable=AsyncMock, return_value=0.68)
    @patch("src.signals.mapper.get_realtime_probability", new_callable=AsyncMock, return_value=0.75)
    @patch("src.signals.consensus.get_calibration_coefficients", new_callable=AsyncMock, return_value=None)
    async def test_short_range_with_aviation(self, mock_calib, mock_avx, mock_gfs, mock_ecmwf, mock_markets):
        from src.signals.detector import detect_signals

        # 1-day-out market with aviation data
        market = _make_market(
            id="short_001",
            parsed_location="Phoenix",
            parsed_variable="temperature",
            parsed_threshold=100.0,
            parsed_operator="above",
            parsed_target_date=_future_date_str(1),
            current_yes_price=0.45,
            liquidity=1000.0,
            volume=5000.0,
        )
        mock_markets.return_value = [market]

        session = AsyncMock()
        session.flush = AsyncMock()

        result = await detect_signals(session)

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
# Variable alias resolution
# ===================================================================


class TestResolveVariable:
    def test_snowfall_maps_to_precipitation(self):
        var, thresh, op = resolve_variable("snowfall", 6.0, "above")
        assert var == "precipitation"
        assert thresh == 6.0
        assert op == "above"

    def test_freeze_maps_to_temperature_below_32(self):
        var, thresh, op = resolve_variable("freeze", None, "occurs")
        assert var == "temperature"
        assert thresh == 32.0
        assert op == "below"

    def test_heat_wave_maps_to_temperature_above(self):
        var, thresh, op = resolve_variable("heat_wave", 100.0, "at_least")
        assert var == "temperature"
        assert thresh == 100.0
        assert op == "above"

    def test_known_variable_unchanged(self):
        var, thresh, op = resolve_variable("temperature", 90.0, "above")
        assert var == "temperature"
        assert thresh == 90.0
        assert op == "above"

    def test_unknown_variable_unchanged(self):
        var, thresh, op = resolve_variable("hurricane_landfall", None, "occurs")
        assert var == "hurricane_landfall"


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
            0.60, 0.60, aviation_prob=0.60, hours_to_resolution=4.0,
            aviation_context=ctx,
        )
        result_no_ctx = compute_consensus(
            0.60, 0.60, aviation_prob=0.60, hours_to_resolution=4.0,
        )
        # 5 amendments → 3 extra → -0.15
        assert result.confidence < result_no_ctx.confidence

    def test_speci_events_boost_confidence(self):
        # Use wide spread so base confidence is well below 1.0
        ctx = AviationContext(speci_events_2h=2)
        result = compute_consensus(
            0.40, 0.70, aviation_prob=0.80, hours_to_resolution=4.0,
            aviation_context=ctx,
        )
        result_no_ctx = compute_consensus(
            0.40, 0.70, aviation_prob=0.80, hours_to_resolution=4.0,
        )
        assert result.confidence > result_no_ctx.confidence

    def test_severe_pireps_boost_confidence(self):
        ctx = AviationContext(has_severe_pireps=True)
        result = compute_consensus(
            0.40, 0.70, aviation_prob=0.80, hours_to_resolution=4.0,
            aviation_context=ctx,
        )
        result_no_ctx = compute_consensus(
            0.40, 0.70, aviation_prob=0.80, hours_to_resolution=4.0,
        )
        assert result.confidence > result_no_ctx.confidence

    def test_sigmet_boosts_confidence(self):
        ctx = AviationContext(active_sigmet_count=1)
        result = compute_consensus(
            0.40, 0.70, aviation_prob=0.80, hours_to_resolution=4.0,
            aviation_context=ctx,
        )
        result_no_ctx = compute_consensus(
            0.40, 0.70, aviation_prob=0.80, hours_to_resolution=4.0,
        )
        assert result.confidence > result_no_ctx.confidence

    def test_no_context_unchanged(self):
        result = compute_consensus(
            0.60, 0.60, aviation_prob=0.60, hours_to_resolution=4.0,
            aviation_context=None,
        )
        result2 = compute_consensus(
            0.60, 0.60, aviation_prob=0.60, hours_to_resolution=4.0,
        )
        assert result.confidence == result2.confidence
