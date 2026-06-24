"""Unit tests for the Phase-1 resolver ground-truth back-solve (pure logic)."""

from __future__ import annotations

from src.signals.station_day_resolution import (
    _MAX_INTERVAL_WIDTH_F,
    _divergence_point,
    back_solve_resolved_max,
)


# --- back_solve_resolved_max ------------------------------------------------


def test_single_exactly_bucket_pins_a_point():
    # An `exactly 86°F` YES implies max ∈ [86, 86].
    lo, hi, point, n = back_solve_resolved_max([(86.0, 86.0)])
    assert (lo, hi, point, n) == (86.0, 86.0, 86.0, 1)


def test_threshold_sandwich_intersects_to_interval():
    # YES "above 85" → max ≥ 85; NO "above 88" → max < 88. Each is one-sided
    # alone but together they sandwich the resolved max to [85, 88), point 86.5.
    lo, hi, point, n = back_solve_resolved_max([(85.0, None), (None, 88.0)])
    assert lo == 85.0
    assert hi == 88.0
    assert point == 86.5
    assert n == 2


def test_one_sided_only_yields_no_point():
    # Only a lower bound (no market caps the top) → no continuous estimate.
    lo, hi, point, n = back_solve_resolved_max([(85.0, None), (80.0, None)])
    assert lo == 85.0  # tightest lower bound
    assert hi is None
    assert point is None
    assert n == 2


def test_multiple_bounds_take_tightest_window():
    bounds = [(80.0, None), (85.0, None), (None, 90.0), (None, 88.0)]
    lo, hi, point, n = back_solve_resolved_max(bounds)
    assert lo == 85.0   # max of lowers
    assert hi == 88.0   # min of uppers
    assert point == 86.5
    assert n == 4


def test_no_bounds_returns_all_none():
    lo, hi, point, n = back_solve_resolved_max([(None, None), (None, None)])
    assert (lo, hi, point, n) == (None, None, None, 0)


def test_inverted_window_keeps_edges_but_no_point():
    # Contradictory labels: lower 88 above upper 85 → keep edges, no point.
    lo, hi, point, n = back_solve_resolved_max([(88.0, None), (None, 85.0)])
    assert lo == 88.0
    assert hi == 85.0
    assert point is None
    assert n == 2


def test_too_wide_window_suppresses_point():
    # A window wider than the trust cap keeps its edges but emits no point.
    lo, hi, point, n = back_solve_resolved_max(
        [(60.0, None), (None, 60.0 + _MAX_INTERVAL_WIDTH_F + 2.0)]
    )
    assert lo == 60.0
    assert hi == 60.0 + _MAX_INTERVAL_WIDTH_F + 2.0
    assert point is None


def test_window_exactly_at_width_cap_keeps_point():
    lo, hi, point, n = back_solve_resolved_max(
        [(60.0, None), (None, 60.0 + _MAX_INTERVAL_WIDTH_F)]
    )
    assert point == round(60.0 + _MAX_INTERVAL_WIDTH_F / 2.0, 2)


# --- _divergence_point ------------------------------------------------------


def test_divergence_point_positive_when_we_read_hotter():
    assert _divergence_point(88.0, 86.0) == 2.0


def test_divergence_point_negative_when_we_read_cooler():
    assert _divergence_point(84.0, 86.5) == -2.5


def test_divergence_point_none_without_point_or_routine():
    assert _divergence_point(None, 86.0) is None
    assert _divergence_point(88.0, None) is None
