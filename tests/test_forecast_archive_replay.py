"""Tests for the forecast-archive replay adapter (Phase 5).

Covers the pure ``archive_to_forecast_fields`` transform — the unit
conversions (especially σ-as-delta, NOT point) and the point-in-time
``hours_until_peak`` — plus the best-effort no-lookahead query shape via
an AsyncMock. The DB-backed query is exercised lightly; the math is the
part that must be right.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from src.signals.forecast_archive_replay import (
    _c_to_f_abs,
    _c_to_f_delta,
    archive_to_forecast_fields,
    latest_archive_as_of,
)


def _row(peak_c=25.0, std_c=1.0, peak_hour_utc=18, model_count=4,
         target=date(2026, 5, 30)):
    return SimpleNamespace(
        peak_temp_c=peak_c,
        peak_temp_std_c=std_c,
        peak_hour_utc=peak_hour_utc,
        model_count=model_count,
        target_date_local=target,
        station_icao="KAUS",
    )


# --- unit conversions -------------------------------------------------------


def test_c_to_f_abs_is_a_point_conversion():
    assert _c_to_f_abs(0.0) == 32.0
    assert _c_to_f_abs(100.0) == 212.0
    assert _c_to_f_abs(25.0) == 77.0


def test_c_to_f_delta_has_no_offset():
    # A std/interval converts with the 9/5 ratio only — NO +32.
    assert _c_to_f_delta(1.0) == 1.8
    assert _c_to_f_delta(0.0) == 0.0
    assert _c_to_f_delta(2.5) == 4.5


# --- archive_to_forecast_fields --------------------------------------------


def test_fields_basic_conversion_and_sigma_is_delta():
    # peak 25°C → 77°F; std 1°C → 1.8°F (delta, not 33.8).
    as_of = datetime(2026, 5, 30, 14, 0, tzinfo=timezone.utc)  # 4h before 18:00
    out = archive_to_forecast_fields(_row(peak_c=25.0, std_c=1.0), as_of)
    assert out["forecast_peak_f"] == 77.0
    assert out["forecast_sigma_f"] == 1.8
    assert out["ensemble_model_count"] == 4
    assert out["has_forecast"] is True
    assert out["hours_until_peak"] == 4.0  # 18:00 - 14:00


def test_zero_std_yields_none_sigma_not_zero():
    # Single-model snapshot → sigma None so the live `is None` fallback path
    # is mirrored, not a fabricated 0°F σ.
    as_of = datetime(2026, 5, 30, 12, 0, tzinfo=timezone.utc)
    out = archive_to_forecast_fields(_row(std_c=0.0), as_of)
    assert out["forecast_sigma_f"] is None


def test_hours_until_peak_negative_past_peak():
    # as_of 20:00, peak 18:00 → -2h (past peak), matching the live signal.
    as_of = datetime(2026, 5, 30, 20, 0, tzinfo=timezone.utc)
    out = archive_to_forecast_fields(_row(peak_hour_utc=18), as_of)
    assert out["hours_until_peak"] == -2.0


def test_naive_as_of_treated_as_utc():
    naive = datetime(2026, 5, 30, 14, 0)  # no tzinfo
    out = archive_to_forecast_fields(_row(peak_hour_utc=18), naive)
    assert out["hours_until_peak"] == 4.0


def test_peak_hour_24_wraps_to_zero():
    # Defensive: peak_hour_utc=24 shouldn't raise; wraps to 0.
    out = archive_to_forecast_fields(
        _row(peak_hour_utc=24), datetime(2026, 5, 30, 0, 0, tzinfo=timezone.utc)
    )
    assert out["hours_until_peak"] == 0.0


# --- latest_archive_as_of (query shape) ------------------------------------


def test_latest_archive_as_of_returns_first_scalar():
    async def _run():
        session = AsyncMock()
        res = MagicMock()
        sentinel = _row()
        res.scalars.return_value.first.return_value = sentinel
        session.execute.return_value = res
        got = await latest_archive_as_of(
            session, "KAUS", date(2026, 5, 30),
            datetime(2026, 5, 30, 14, 0, tzinfo=timezone.utc),
        )
        assert got is sentinel
        session.execute.assert_awaited_once()

    asyncio.run(_run())


def test_latest_archive_as_of_none_when_no_rows():
    async def _run():
        session = AsyncMock()
        res = MagicMock()
        res.scalars.return_value.first.return_value = None
        session.execute.return_value = res
        got = await latest_archive_as_of(
            session, "KAUS", date(2026, 5, 30),
            datetime(2026, 5, 30, 14, 0, tzinfo=timezone.utc),
        )
        assert got is None

    asyncio.run(_run())
