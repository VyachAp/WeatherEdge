"""Unit tests for the Phase-3 forecast-error lead-bucketing (pure logic)."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

from src.signals.forecast_error import compute_forecast_errors

_DAY = date(2026, 6, 20)
# Canonical peak instant: 2026-06-20 18:00 UTC.
_PEAK = datetime(2026, 6, 20, 18, 0, tzinfo=timezone.utc)


def _snap(hours_before_peak: float, peak_c: float, std_c: float = 1.0):
    """A ForecastArchive-like stub fetched `hours_before_peak` before peak."""
    return SimpleNamespace(
        peak_temp_c=peak_c,
        peak_temp_std_c=std_c,
        peak_hour_utc=18,
        target_date_local=_DAY,
        fetched_at=_PEAK - timedelta(hours=hours_before_peak),
    )


def test_empty_or_no_realized_returns_empty():
    assert compute_forecast_errors([], realized_max_f=90.0) == []
    assert compute_forecast_errors([_snap(0, 32.0)], realized_max_f=None) == []


def test_lead_buckets_pick_point_in_time_snapshot():
    # Forecast warms as peak approaches: 30°C at 24h out, 33°C at peak.
    rows = [_snap(24, 30.0), _snap(12, 31.0), _snap(0, 33.0)]
    # 33°C = 91.4°F; realized 90°F.
    out = compute_forecast_errors(
        rows, realized_max_f=90.0, lead_buckets=(0, 12, 24)
    )
    by_lead = {r["lead_bucket_h"]: r for r in out}
    assert set(by_lead) == {0, 12, 24}
    # lead 0 → the 33°C snapshot (91.4°F).
    assert by_lead[0]["forecast_peak_f"] == 91.4
    assert by_lead[0]["error_vs_metar_f"] == round(91.4 - 90.0, 2)
    # lead 24 → the 30°C snapshot (86.0°F).
    assert by_lead[24]["forecast_peak_f"] == 86.0
    assert by_lead[24]["error_vs_metar_f"] == round(86.0 - 90.0, 2)


def test_sigma_delta_conversion_and_none_when_zero():
    rows = [_snap(0, 30.0, std_c=2.0), _snap(6, 30.0, std_c=0.0)]
    out = compute_forecast_errors(
        rows, realized_max_f=86.0, lead_buckets=(0, 6)
    )
    by_lead = {r["lead_bucket_h"]: r for r in out}
    # 2°C std → 3.6°F (delta conversion, NO +32).
    assert by_lead[0]["forecast_sigma_f"] == 3.6
    # 0 std → None (single-model snapshot), mirroring the live fallback.
    assert by_lead[6]["forecast_sigma_f"] is None


def test_resolved_error_present_when_resolved_given():
    rows = [_snap(0, 30.0)]  # 86.0°F
    out = compute_forecast_errors(
        rows, realized_max_f=85.0, resolved_max_f=84.0, lead_buckets=(0,)
    )
    r = out[0]
    assert r["error_vs_metar_f"] == 1.0      # 86 - 85
    assert r["error_vs_resolved_f"] == 2.0   # 86 - 84


def test_bucket_skipped_when_no_snapshot_old_enough():
    # Only a snapshot 5h before peak → the 24h bucket has nothing ≤ peak-24h.
    rows = [_snap(5, 30.0)]
    out = compute_forecast_errors(
        rows, realized_max_f=86.0, lead_buckets=(0, 24)
    )
    leads = {r["lead_bucket_h"] for r in out}
    assert leads == {0}
