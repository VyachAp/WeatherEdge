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


class TestShadowCalibration:
    """Phase 1 measure-before-flip telemetry (``shadow_calibration``).

    Pure telemetry helper that returns BOTH the pooled and the per-class
    calibrated value for a raw prob, bypassing the
    ``PER_OPERATOR_CALIBRATION_ENABLED`` flag, so ``shadow-report`` can
    validate the flip before it touches live trading. Never influences a
    trade. The autouse ``_clean_cache`` fixture resets the cache around
    each test.
    """

    def test_disabled_returns_none(self):
        calibration_mod._cached_coeffs = {POOLED_KEY: (1.0, 0.0)}
        calibration_mod._cached_at = time.time()
        with patch.object(
            calibration_mod.settings, "SHADOW_CALIBRATION_ENABLED", False
        ):
            assert calibration_mod.shadow_calibration(0.8, "threshold") is None

    def test_none_when_no_pooled_fit(self):
        # Cache empty → nothing to compare against.
        with patch.object(
            calibration_mod.settings, "SHADOW_CALIBRATION_ENABLED", True
        ):
            assert calibration_mod.shadow_calibration(0.8, "threshold") is None

    def test_none_when_raw_prob_missing(self):
        calibration_mod._cached_coeffs = {POOLED_KEY: (1.0, 0.0)}
        calibration_mod._cached_at = time.time()
        with patch.object(
            calibration_mod.settings, "SHADOW_CALIBRATION_ENABLED", True
        ):
            assert calibration_mod.shadow_calibration(None, "threshold") is None

    def test_pooled_and_class_with_delta(self):
        # pooled squashes 0.80 → 0.85; threshold class maps 0.80 → 0.78.
        calibration_mod._cached_coeffs = {
            POOLED_KEY: (0.5, 0.45),   # 0.5*0.8 + 0.45 = 0.85
            "threshold": (1.0, -0.02),  # 1.0*0.8 - 0.02 = 0.78
        }
        calibration_mod._cached_at = time.time()
        with patch.object(
            calibration_mod.settings, "SHADOW_CALIBRATION_ENABLED", True
        ):
            out = calibration_mod.shadow_calibration(0.80, "threshold")
        assert out["raw"] == 0.8
        assert out["op_class"] == "threshold"
        assert out["pooled"] == pytest.approx(0.85, abs=1e-6)
        assert out["class"] == pytest.approx(0.78, abs=1e-6)
        # delta = class - pooled (negative: class un-squashes downward).
        assert out["delta"] == pytest.approx(-0.07, abs=1e-6)

    def test_omits_class_when_no_class_fit(self):
        calibration_mod._cached_coeffs = {POOLED_KEY: (1.0, 0.0)}
        calibration_mod._cached_at = time.time()
        with patch.object(
            calibration_mod.settings, "SHADOW_CALIBRATION_ENABLED", True
        ):
            out = calibration_mod.shadow_calibration(0.80, "bracket-like")
        assert "pooled" in out
        # No bracket-like fit cached → flag-on would fall back to pooled.
        assert "class" not in out
        assert "delta" not in out

    def test_respects_ttl(self):
        calibration_mod._cached_coeffs = {POOLED_KEY: (1.0, 0.0)}
        calibration_mod._cached_at = (
            time.time() - calibration_mod._CACHE_TTL_SEC - 1
        )
        with patch.object(
            calibration_mod.settings, "SHADOW_CALIBRATION_ENABLED", True
        ):
            assert calibration_mod.shadow_calibration(0.80, "threshold") is None


class TestDegenerateFitGuardrail:
    """Per-class fits with a runaway slope are rejected → pooled fallback.

    Motivated by the live 2026-05-31 observation: the threshold class at
    n=55 fit slope +3.64 / intercept -2.80, mapping raw 0.78 → 0.04, which
    would destroy threshold trading. ``_is_plausible_fit`` is pure; the
    integration is checked via ``get_calibration_coefficients``.
    """

    def test_is_plausible_fit_band(self):
        f = calibration_mod._is_plausible_fit
        assert f(1.0) is True            # identity
        assert f(0.6) is True            # healthy pooled-like
        assert f(calibration_mod.CALIBRATION_MIN_SLOPE) is True   # inclusive
        assert f(calibration_mod.CALIBRATION_MAX_SLOPE) is True   # inclusive
        assert f(3.64) is False          # the observed runaway
        assert f(0.0) is False           # collapsed
        assert f(-0.5) is False          # inverted

    @pytest.mark.asyncio
    async def test_runaway_class_fit_falls_back_to_pooled(self):
        """A class whose fit slope is out of band is omitted from the cache
        so ``get_cached_calibration`` falls back to pooled for it."""
        from unittest.mock import AsyncMock, MagicMock
        from src.db.models import TradeStatus

        # 60 threshold signals whose raw probs cluster high (0.9-0.99) but
        # mostly lose → polyfit produces a steep extrapolating slope.
        sigs = []
        for i in range(60):
            won = i % 4 == 0  # 25% win rate on high-confidence → steep fit
            t = MagicMock()
            t.status = TradeStatus.WON if won else TradeStatus.LOST
            sig = MagicMock()
            sig.raw_model_prob = 0.90 + (i % 10) * 0.009
            sig.model_prob = sig.raw_model_prob
            sig.trades = [t]
            sig.market = MagicMock(parsed_operator="above")  # threshold class
            sigs.append(sig)

        session = AsyncMock()
        res = MagicMock()
        res.unique.return_value.scalars.return_value.all.return_value = sigs
        session.execute.return_value = res

        coeffs = await calibration_mod.get_calibration_coefficients(session)
        assert coeffs is not None
        assert POOLED_KEY in coeffs  # pooled always present
        # The degenerate threshold fit must NOT be cached as its own key.
        assert "threshold" not in coeffs
