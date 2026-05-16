"""Tests for ``ingestion.station_bias.get_station_rmse`` and the
per-station σ floor it feeds in ``probability_engine``.

The RMSE helper computes ``sqrt(avg(bias_c**2))`` over the rolling
``STATION_BIAS_WINDOW_DAYS`` window from the existing ``station_biases``
table — no new schema, no nightly job. The floor logic in
``_effective_sigma_floor`` then replaces the global
``ENSEMBLE_MIN_SIGMA_F`` with the per-station RMSE when the station
has enough sample days; below the threshold we keep the global floor
so cold-start stations aren't over- or under-padded by a flimsy fit.

Contracts asserted:

* ``get_station_rmse`` returns ``(None, 0)`` when no rows exist and the
  TTL cache short-circuits subsequent calls within the window.
* RMSE is reported in °F (converted from °C via ×9/5).
* ``_effective_sigma_floor`` honors the ``[1.5, 5.0]`` clamp and the
  ``_PER_STATION_SIGMA_MIN_DAYS`` cold-start fallback.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.ingestion import station_bias as sb_mod
from src.ingestion.station_bias import (
    clear_station_rmse_cache,
    get_station_rmse,
)
from src.signals.probability_engine import _effective_sigma_floor
from src.signals.state_aggregator import WeatherState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(rmse_f: float | None, sample_days: int) -> WeatherState:
    """Minimal WeatherState with only the RMSE fields populated."""
    return WeatherState(
        station_icao="TEST",
        current_max_f=80.0,
        metar_trend_rate=0.0,
        dewpoint_trend_rate=0.0,
        forecast_peak_f=85.0,
        hours_until_peak=3.0,
        solar_declining=False,
        solar_decline_magnitude=0.0,
        cloud_rising=False,
        cloud_rise_magnitude=0.0,
        routine_count_today=4,
        station_rmse_f=rmse_f,
        station_rmse_sample_days=sample_days,
    )


@pytest.fixture(autouse=True)
def _reset_rmse_cache():
    """Cache lives at module scope and would leak across tests."""
    clear_station_rmse_cache()
    yield
    clear_station_rmse_cache()


# ---------------------------------------------------------------------------
# get_station_rmse
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_station_rmse_returns_none_when_no_bias_rows() -> None:
    """No rows in station_biases for the station → (None, 0). Caller
    should fall back to the global σ floor."""
    session = MagicMock()
    result = MagicMock()
    result.one.return_value = (None, 0)
    session.execute = AsyncMock(return_value=result)

    rmse_f, days = await get_station_rmse(session, "KXXX")
    assert rmse_f is None
    assert days == 0


@pytest.mark.asyncio
async def test_get_station_rmse_converts_c_to_f_and_returns_count() -> None:
    """Stored bias is in °C; consumer expects °F (×9/5). Sample-day count
    is preserved as int."""
    session = MagicMock()
    result = MagicMock()
    # 2.5 °C RMSE → 4.5 °F
    result.one.return_value = (2.5, 30)
    session.execute = AsyncMock(return_value=result)

    rmse_f, days = await get_station_rmse(session, "KAUS")
    assert rmse_f == pytest.approx(4.5)
    assert days == 30


@pytest.mark.asyncio
async def test_get_station_rmse_cache_short_circuits_within_ttl() -> None:
    """Second call within the TTL must not touch the session — the
    cache is the whole point of having it for ~50-city hot-path use."""
    session = MagicMock()
    result = MagicMock()
    result.one.return_value = (2.0, 25)
    session.execute = AsyncMock(return_value=result)

    await get_station_rmse(session, "KSEA")
    await get_station_rmse(session, "KSEA")
    assert session.execute.await_count == 1


@pytest.mark.asyncio
async def test_get_station_rmse_cache_refreshes_after_ttl() -> None:
    """After ``_RMSE_TTL``, the cached entry must be discarded and
    re-queried — otherwise stale RMSE persists across daily_settlement
    bias writes."""
    session = MagicMock()
    result = MagicMock()
    result.one.return_value = (2.0, 25)
    session.execute = AsyncMock(return_value=result)

    await get_station_rmse(session, "KSEA")
    # Backdate the cache entry beyond the TTL.
    rmse_f, days, _ = sb_mod._rmse_cache["KSEA"]
    sb_mod._rmse_cache["KSEA"] = (
        rmse_f, days, datetime.now(timezone.utc) - sb_mod._RMSE_TTL - timedelta(seconds=1),
    )
    await get_station_rmse(session, "KSEA")
    assert session.execute.await_count == 2


# ---------------------------------------------------------------------------
# _effective_sigma_floor — per-station vs global fallback
# ---------------------------------------------------------------------------


def test_sigma_floor_uses_per_station_rmse_when_enough_days() -> None:
    """≥14 days → use RMSE directly when inside the [1.5, 5.0] band."""
    floor, source = _effective_sigma_floor(_make_state(3.2, 20))
    assert floor == pytest.approx(3.2)
    assert "per-station RMSE" in source


def test_sigma_floor_clamps_low_rmse_to_min() -> None:
    """Very low RMSE (calibration-noise-dominated stations like RJTT)
    is clamped at 1.5°F. Without this clamp the distribution would
    collapse → spurious narrow NO bets on modal buckets."""
    floor, source = _effective_sigma_floor(_make_state(1.0, 30))
    assert floor == 1.5
    assert "per-station RMSE" in source


def test_sigma_floor_clamps_high_rmse_to_max() -> None:
    """Very high RMSE (chaotic stations like KORD) is clamped at 5.0°F
    — beyond that, widening σ doesn't recover edge; the trade just has
    no signal worth taking."""
    floor, source = _effective_sigma_floor(_make_state(8.3, 30))
    assert floor == 5.0
    assert "per-station RMSE" in source


def test_sigma_floor_falls_back_to_global_when_cold_start() -> None:
    """< ``_PER_STATION_SIGMA_MIN_DAYS`` days → use the global
    ``ENSEMBLE_MIN_SIGMA_F`` floor. Avoids over-trusting a noisy
    short-window RMSE estimate."""
    from src.config import settings
    floor, source = _effective_sigma_floor(_make_state(4.5, 10))
    assert floor == settings.ENSEMBLE_MIN_SIGMA_F
    assert "global" in source


def test_sigma_floor_falls_back_to_global_when_rmse_none() -> None:
    """No station_biases rows at all → no RMSE → global floor."""
    from src.config import settings
    floor, source = _effective_sigma_floor(_make_state(None, 0))
    assert floor == settings.ENSEMBLE_MIN_SIGMA_F
    assert "global" in source
