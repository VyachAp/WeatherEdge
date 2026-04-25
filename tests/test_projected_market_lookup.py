"""Tests for the projected-binary market reverse lookup."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.signals.projected_market_lookup import (
    MAX_THRESHOLD_DISTANCE_F,
    _is_operator_aligned,
    lookup_projected_binary,
)


def _market(
    *,
    threshold: float | None,
    operator: str | None,
    end_date: datetime | None,
    market_id: str = "m",
    liquidity: float = 0.0,
    current_yes_price: float | None = 0.5,
):
    return SimpleNamespace(
        id=market_id,
        parsed_threshold=threshold,
        parsed_operator=operator,
        end_date=end_date,
        liquidity=liquidity,
        current_yes_price=current_yes_price,
    )


def _today_end() -> datetime:
    return datetime.now(timezone.utc).replace(hour=23, minute=59)


class _FakeSession:
    """Minimal async-context-manager stand-in for async_session()."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.fixture
def _stub_session():
    """Replace the real DB session opener with a no-op async context manager."""
    with patch(
        "src.signals.projected_market_lookup.async_session",
        return_value=_FakeSession(),
    ):
        yield


class TestIsOperatorAligned:
    def test_above_aligned_when_projection_meets_threshold(self):
        assert _is_operator_aligned("above", 75.0, 75.0)
        assert _is_operator_aligned("above", 75.0, 76.0)
        assert _is_operator_aligned("at_least", 75.0, 80.0)
        assert _is_operator_aligned("exceed", 75.0, 80.0)

    def test_above_not_aligned_when_projection_below_threshold(self):
        assert not _is_operator_aligned("above", 80.0, 75.0)
        assert not _is_operator_aligned("at_least", 90.0, 75.0)

    def test_below_aligned_when_projection_under_threshold(self):
        assert _is_operator_aligned("below", 80.0, 75.0)
        assert _is_operator_aligned("at_most", 80.0, 80.0)

    def test_below_not_aligned_when_projection_exceeds_threshold(self):
        assert not _is_operator_aligned("below", 75.0, 80.0)
        assert not _is_operator_aligned("at_most", 70.0, 80.0)

    def test_unknown_operator_never_aligned(self):
        assert not _is_operator_aligned("bracket", 75.0, 75.0)
        assert not _is_operator_aligned("exactly", 75.0, 75.0)


@pytest.mark.usefixtures("_stub_session")
class TestLookupProjectedBinary:
    @pytest.mark.asyncio
    async def test_picks_operator_aligned_below_projection_for_above(self):
        # Projection 77.7°F. Markets at ≥75 (aligned, within band) and ≥82
        # (not aligned — above projection). Should pick ≥75.
        end = _today_end()
        markets = [
            _market(threshold=82.0, operator="above", end_date=end, market_id="hi"),
            _market(threshold=75.0, operator="above", end_date=end, market_id="lo",
                    current_yes_price=0.7),
        ]

        with patch(
            "src.signals.projected_market_lookup.find_markets_for_station",
            new=AsyncMock(return_value=markets),
        ), patch(
            "src.signals.projected_market_lookup.get_token_ids",
            new=AsyncMock(return_value=None),
        ):
            result = await lookup_projected_binary("KLAX", 77.7)

        assert result is not None
        market, threshold, operator, yes_price = result
        assert market.id == "lo"
        assert threshold == 75.0
        assert operator == "above"
        assert yes_price == 0.7

    @pytest.mark.asyncio
    async def test_drops_above_market_with_threshold_above_projection(self):
        # Projection 25.4°C ≈ 77.7°F. Only market available is ≥82.4°F (≈28°C).
        # The lookup must return None — buying YES makes no sense.
        end = _today_end()
        markets = [
            _market(threshold=82.4, operator="above", end_date=end),
        ]

        with patch(
            "src.signals.projected_market_lookup.find_markets_for_station",
            new=AsyncMock(return_value=markets),
        ):
            result = await lookup_projected_binary("LIMC", 77.7)

        assert result is None

    @pytest.mark.asyncio
    async def test_picks_operator_aligned_above_projection_for_below(self):
        end = _today_end()
        markets = [
            _market(threshold=70.0, operator="below", end_date=end, market_id="under"),
            _market(threshold=78.0, operator="at_most", end_date=end, market_id="over",
                    current_yes_price=0.6),
        ]

        with patch(
            "src.signals.projected_market_lookup.find_markets_for_station",
            new=AsyncMock(return_value=markets),
        ), patch(
            "src.signals.projected_market_lookup.get_token_ids",
            new=AsyncMock(return_value=None),
        ):
            result = await lookup_projected_binary("KLAX", 75.0)

        assert result is not None
        market, threshold, operator, _ = result
        assert market.id == "over"
        assert threshold == 78.0
        assert operator == "at_most"

    @pytest.mark.asyncio
    async def test_drops_aligned_but_far_threshold(self):
        # Aligned (≥X with X below projection) but ≥10°F away → outside band.
        end = _today_end()
        far = MAX_THRESHOLD_DISTANCE_F + 5.0
        markets = [
            _market(threshold=77.7 - far, operator="above", end_date=end),
        ]

        with patch(
            "src.signals.projected_market_lookup.find_markets_for_station",
            new=AsyncMock(return_value=markets),
        ):
            result = await lookup_projected_binary("KLAX", 77.7)

        assert result is None

    @pytest.mark.asyncio
    async def test_keeps_threshold_at_band_edge(self):
        # Exactly MAX_THRESHOLD_DISTANCE_F away, aligned → still kept.
        end = _today_end()
        threshold = 77.7 - MAX_THRESHOLD_DISTANCE_F
        markets = [
            _market(threshold=threshold, operator="above", end_date=end,
                    current_yes_price=0.55),
        ]

        with patch(
            "src.signals.projected_market_lookup.find_markets_for_station",
            new=AsyncMock(return_value=markets),
        ), patch(
            "src.signals.projected_market_lookup.get_token_ids",
            new=AsyncMock(return_value=None),
        ):
            result = await lookup_projected_binary("KLAX", 77.7)

        assert result is not None
        _, t, _, _ = result
        assert t == pytest.approx(threshold)

    @pytest.mark.asyncio
    async def test_skips_non_today_markets(self):
        tomorrow = datetime.now(timezone.utc) + timedelta(days=1)
        markets = [
            _market(threshold=75.0, operator="above", end_date=tomorrow),
        ]

        with patch(
            "src.signals.projected_market_lookup.find_markets_for_station",
            new=AsyncMock(return_value=markets),
        ):
            result = await lookup_projected_binary("KLAX", 77.7)

        assert result is None

    @pytest.mark.asyncio
    async def test_skips_bracket_operators(self):
        end = _today_end()
        markets = [
            _market(threshold=75.0, operator="bracket", end_date=end),
        ]

        with patch(
            "src.signals.projected_market_lookup.find_markets_for_station",
            new=AsyncMock(return_value=markets),
        ):
            result = await lookup_projected_binary("KLAX", 77.7)

        assert result is None

    @pytest.mark.asyncio
    async def test_picks_closest_among_aligned(self):
        # Two aligned ≥X markets; should pick the one closer to projection.
        end = _today_end()
        markets = [
            _market(threshold=70.0, operator="above", end_date=end, market_id="far",
                    current_yes_price=0.9),
            _market(threshold=76.5, operator="above", end_date=end, market_id="near",
                    current_yes_price=0.6),
        ]

        with patch(
            "src.signals.projected_market_lookup.find_markets_for_station",
            new=AsyncMock(return_value=markets),
        ), patch(
            "src.signals.projected_market_lookup.get_token_ids",
            new=AsyncMock(return_value=None),
        ):
            result = await lookup_projected_binary("KLAX", 77.7)

        assert result is not None
        market, _, _, _ = result
        assert market.id == "near"

    @pytest.mark.asyncio
    async def test_uses_clob_ask_when_available(self):
        end = _today_end()
        markets = [
            _market(threshold=75.0, operator="above", end_date=end,
                    current_yes_price=0.5),
        ]

        with patch(
            "src.signals.projected_market_lookup.find_markets_for_station",
            new=AsyncMock(return_value=markets),
        ), patch(
            "src.signals.projected_market_lookup.get_token_ids",
            new=AsyncMock(return_value=("yes", "no")),
        ), patch(
            "src.signals.projected_market_lookup.get_best_bid_ask",
            return_value=(0.71, 0.73),
        ):
            result = await lookup_projected_binary("KLAX", 77.7)

        assert result is not None
        _, _, _, yes_price = result
        # Pays the ask (0.73), not the bid or stored mid.
        assert yes_price == pytest.approx(0.73)

    @pytest.mark.asyncio
    async def test_falls_back_to_stored_yes_price_when_clob_fails(self):
        end = _today_end()
        markets = [
            _market(threshold=75.0, operator="above", end_date=end,
                    current_yes_price=0.42),
        ]

        with patch(
            "src.signals.projected_market_lookup.find_markets_for_station",
            new=AsyncMock(return_value=markets),
        ), patch(
            "src.signals.projected_market_lookup.get_token_ids",
            new=AsyncMock(return_value=("yes", "no")),
        ), patch(
            "src.signals.projected_market_lookup.get_best_bid_ask",
            return_value=None,
        ):
            result = await lookup_projected_binary("KLAX", 77.7)

        assert result is not None
        _, _, _, yes_price = result
        assert yes_price == pytest.approx(0.42)

    @pytest.mark.asyncio
    async def test_returns_none_when_lookup_raises(self):
        with patch(
            "src.signals.projected_market_lookup.find_markets_for_station",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ):
            result = await lookup_projected_binary("KLAX", 77.7)

        assert result is None
