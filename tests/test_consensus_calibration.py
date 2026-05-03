"""Phase 1.2 tests: in-process calibration cache + apply_calibration."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from src.signals.consensus import (
    apply_calibration,
    get_cached_calibration,
    reset_calibration_cache,
)
from src.signals import consensus as consensus_mod


@pytest.fixture(autouse=True)
def _clean_cache():
    reset_calibration_cache()
    yield
    reset_calibration_cache()


def test_apply_calibration_disabled_returns_input_unchanged():
    """When `APPLY_CALIBRATION=False` the helper is a no-op even if a fit
    is cached. Default-off is the safe-by-default story."""
    consensus_mod._cached_coeffs = (0.8, 0.05)
    consensus_mod._cached_at = time.time()

    with patch.object(consensus_mod.settings, "APPLY_CALIBRATION", False):
        out, applied = apply_calibration(0.7)

    assert out == 0.7
    assert applied is False


def test_apply_calibration_enabled_uses_cached_coefficients():
    consensus_mod._cached_coeffs = (0.8, 0.05)
    consensus_mod._cached_at = time.time()

    with patch.object(consensus_mod.settings, "APPLY_CALIBRATION", True):
        out, applied = apply_calibration(0.7)

    assert applied is True
    assert out == pytest.approx(0.8 * 0.7 + 0.05, abs=1e-9)


def test_apply_calibration_clamps_to_unit_interval():
    """A pathological fit (e.g. slope > 1, intercept > 0) can push values
    above 1.0; the helper clamps so downstream filters don't trip on
    impossible probabilities."""
    consensus_mod._cached_coeffs = (1.5, 0.4)  # 1.5*0.8 + 0.4 = 1.6
    consensus_mod._cached_at = time.time()

    with patch.object(consensus_mod.settings, "APPLY_CALIBRATION", True):
        out, applied = apply_calibration(0.8)

    assert applied is True
    assert out == 1.0  # clamped


def test_apply_calibration_clamps_negative_lower_bound():
    consensus_mod._cached_coeffs = (1.0, -0.5)  # 1.0*0.2 - 0.5 = -0.3
    consensus_mod._cached_at = time.time()

    with patch.object(consensus_mod.settings, "APPLY_CALIBRATION", True):
        out, applied = apply_calibration(0.2)

    assert applied is True
    assert out == 0.0  # clamped


def test_get_cached_calibration_returns_none_when_unset():
    assert get_cached_calibration() is None


def test_get_cached_calibration_returns_none_when_stale():
    """Beyond TTL, the cache is treated as empty so a stale fit doesn't
    silently keep applying after the data has shifted under it."""
    consensus_mod._cached_coeffs = (0.9, 0.0)
    consensus_mod._cached_at = time.time() - (consensus_mod._CACHE_TTL_SEC + 1)

    assert get_cached_calibration() is None


def test_apply_calibration_returns_input_when_no_cache():
    """`APPLY_CALIBRATION=True` but cache is empty (no fit yet) — fall
    back to raw probability rather than blocking trades."""
    with patch.object(consensus_mod.settings, "APPLY_CALIBRATION", True):
        out, applied = apply_calibration(0.65)

    assert out == 0.65
    assert applied is False
