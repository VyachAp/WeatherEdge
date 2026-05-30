"""Phase 1.2 tests: in-process calibration cache + apply_calibration.

Cache shape changed 2026-05-30: the module-level ``_cached_coeffs`` is
now ``dict[str, (slope, intercept)] | None`` keyed by operator class
(plus a ``"pooled"`` fallback). The pre-split tuple form is no longer
valid; tests construct dict fixtures inline.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from src.signals.calibration import (
    POOLED_KEY,
    apply_calibration,
    get_cached_calibration,
    reset_calibration_cache,
)
from src.signals import calibration as calibration_mod


@pytest.fixture(autouse=True)
def _clean_cache():
    reset_calibration_cache()
    yield
    reset_calibration_cache()


def test_apply_calibration_disabled_returns_input_unchanged():
    """When `APPLY_CALIBRATION=False` the helper is a no-op even if a fit
    is cached. Default-off is the safe-by-default story."""
    calibration_mod._cached_coeffs = {POOLED_KEY: (0.8, 0.05)}
    calibration_mod._cached_at = time.time()

    with patch.object(calibration_mod.settings, "APPLY_CALIBRATION", False):
        out, applied = apply_calibration(0.7)

    assert out == 0.7
    assert applied is False


def test_apply_calibration_enabled_uses_cached_coefficients():
    calibration_mod._cached_coeffs = {POOLED_KEY: (0.8, 0.05)}
    calibration_mod._cached_at = time.time()

    with patch.object(calibration_mod.settings, "APPLY_CALIBRATION", True):
        out, applied = apply_calibration(0.7)

    assert applied is True
    assert out == pytest.approx(0.8 * 0.7 + 0.05, abs=1e-9)


def test_apply_calibration_clamps_to_unit_interval():
    """A pathological fit (e.g. slope > 1, intercept > 0) can push values
    above 1.0; the helper clamps so downstream filters don't trip on
    impossible probabilities."""
    calibration_mod._cached_coeffs = {POOLED_KEY: (1.5, 0.4)}  # 1.5*0.8 + 0.4 = 1.6
    calibration_mod._cached_at = time.time()

    with patch.object(calibration_mod.settings, "APPLY_CALIBRATION", True):
        out, applied = apply_calibration(0.8)

    assert applied is True
    assert out == 1.0  # clamped


def test_apply_calibration_clamps_negative_lower_bound():
    calibration_mod._cached_coeffs = {POOLED_KEY: (1.0, -0.5)}  # 1.0*0.2 - 0.5 = -0.3
    calibration_mod._cached_at = time.time()

    with patch.object(calibration_mod.settings, "APPLY_CALIBRATION", True):
        out, applied = apply_calibration(0.2)

    assert applied is True
    assert out == 0.0  # clamped


def test_get_cached_calibration_returns_none_when_unset():
    assert get_cached_calibration() is None


def test_get_cached_calibration_returns_none_when_stale():
    """Beyond TTL, the cache is treated as empty so a stale fit doesn't
    silently keep applying after the data has shifted under it."""
    calibration_mod._cached_coeffs = {POOLED_KEY: (0.9, 0.0)}
    calibration_mod._cached_at = time.time() - (calibration_mod._CACHE_TTL_SEC + 1)

    assert get_cached_calibration() is None


def test_apply_calibration_returns_input_when_no_cache():
    """`APPLY_CALIBRATION=True` but cache is empty (no fit yet) — fall
    back to raw probability rather than blocking trades."""
    with patch.object(calibration_mod.settings, "APPLY_CALIBRATION", True):
        out, applied = apply_calibration(0.65)

    assert out == 0.65
    assert applied is False


class TestPerOperatorCalibration:
    """Per-operator-class calibration split (added 2026-05-30).

    The cache holds separate fits per operator class plus a pooled
    fallback. The ``PER_OPERATOR_CALIBRATION_ENABLED`` flag gates the
    class-aware lookup; off = pooled always (legacy bit-for-bit).
    """

    def test_disabled_falls_back_to_pooled(self):
        """Flag off → class-specific cache is ignored, pooled wins."""
        calibration_mod._cached_coeffs = {
            POOLED_KEY: (0.8, 0.05),
            "threshold": (0.5, 0.10),
        }
        calibration_mod._cached_at = time.time()

        with patch.object(calibration_mod.settings, "APPLY_CALIBRATION", True), \
             patch.object(
                 calibration_mod.settings,
                 "PER_OPERATOR_CALIBRATION_ENABLED",
                 False,
             ):
            out, applied = apply_calibration(0.7, operator_class="threshold")

        # Pooled wins: 0.8 * 0.7 + 0.05 = 0.61.
        assert applied is True
        assert out == pytest.approx(0.61, abs=1e-9)

    def test_class_specific_used_when_enabled(self):
        """Flag on + matching class key → class fit wins."""
        calibration_mod._cached_coeffs = {
            POOLED_KEY: (0.8, 0.05),
            "threshold": (0.5, 0.10),
        }
        calibration_mod._cached_at = time.time()

        with patch.object(calibration_mod.settings, "APPLY_CALIBRATION", True), \
             patch.object(
                 calibration_mod.settings,
                 "PER_OPERATOR_CALIBRATION_ENABLED",
                 True,
             ):
            out, applied = apply_calibration(0.7, operator_class="threshold")

        # Class fit: 0.5 * 0.7 + 0.10 = 0.45.
        assert applied is True
        assert out == pytest.approx(0.45, abs=1e-9)

    def test_class_falls_back_to_pooled_when_class_missing(self):
        """Flag on but class key absent → pooled fallback applies."""
        calibration_mod._cached_coeffs = {POOLED_KEY: (0.8, 0.05)}
        calibration_mod._cached_at = time.time()

        with patch.object(calibration_mod.settings, "APPLY_CALIBRATION", True), \
             patch.object(
                 calibration_mod.settings,
                 "PER_OPERATOR_CALIBRATION_ENABLED",
                 True,
             ):
            out, applied = apply_calibration(0.7, operator_class="bracket-like")

        assert applied is True
        assert out == pytest.approx(0.8 * 0.7 + 0.05, abs=1e-9)

    def test_no_class_uses_pooled(self):
        """When the caller passes ``operator_class=None`` (unknown / legacy
        callers) the pooled fit wins regardless of the flag."""
        calibration_mod._cached_coeffs = {
            POOLED_KEY: (0.8, 0.05),
            "threshold": (0.5, 0.10),
        }
        calibration_mod._cached_at = time.time()

        with patch.object(calibration_mod.settings, "APPLY_CALIBRATION", True), \
             patch.object(
                 calibration_mod.settings,
                 "PER_OPERATOR_CALIBRATION_ENABLED",
                 True,
             ):
            out, applied = apply_calibration(0.7, operator_class=None)

        # Pooled wins: 0.8 * 0.7 + 0.05 = 0.61.
        assert applied is True
        assert out == pytest.approx(0.61, abs=1e-9)

    def test_get_cached_calibration_returns_pooled_when_class_missing(self):
        """Sync accessor falls back to pooled identically."""
        calibration_mod._cached_coeffs = {POOLED_KEY: (0.8, 0.05)}
        calibration_mod._cached_at = time.time()

        with patch.object(
            calibration_mod.settings,
            "PER_OPERATOR_CALIBRATION_ENABLED",
            True,
        ):
            assert get_cached_calibration("bracket-like") == (0.8, 0.05)
            assert get_cached_calibration("threshold") == (0.8, 0.05)
            assert get_cached_calibration() == (0.8, 0.05)
