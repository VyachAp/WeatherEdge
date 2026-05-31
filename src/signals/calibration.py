"""Signal calibration utilities.

Provides linear recalibration coefficients from resolved signal history,
plus an in-process cache + sync `apply_calibration` helper used by the
edge evaluator. Calibration is gated by ``settings.APPLY_CALIBRATION`` so
it can be A/B-toggled without code changes while we validate that the
fitted slope/intercept actually improve Brier on a held-out window.

Per-operator split (2026-05-30): when
``settings.PER_OPERATOR_CALIBRATION_ENABLED`` is True, the cache holds
separate fits keyed by ``"threshold"`` / ``"bracket-like"`` (plus a
``"pooled"`` fallback). The threshold-class pool is no longer dragged up
by bracket-like's 0.99-dominated tail, which was the documented reason
``THRESHOLD_MIN_PROBABILITY=0.78`` produced zero fills in the
[0.78, 0.85) band.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from src.config import settings
from src.db.models import Signal, Trade, TradeStatus
from src.execution.binary_market import operator_class as _operator_class

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

MIN_CALIBRATION_SAMPLES: int = 50

# Cache keys. ``"pooled"`` is the always-fitted global regression that
# preserves legacy single-fit behavior; the per-class keys are populated
# only when each class has ≥ ``MIN_CALIBRATION_SAMPLES`` of its own.
POOLED_KEY: str = "pooled"
_OP_CLASS_KEYS: tuple[str, ...] = ("threshold", "bracket-like")

# In-process cache for fitted (slope, intercept) — refreshed by
# ``refresh_calibration`` (called from the scheduler tick) and read
# synchronously by ``apply_calibration`` from the edge evaluator. The
# value is a dict keyed by operator class (plus the ``POOLED_KEY``
# fallback) so the same cache backs single- and multi-fit modes.
_CACHE_TTL_SEC: float = 1800.0
_cached_coeffs: dict[str, tuple[float, float]] | None = None
_cached_at: float | None = None


def _fit_linear(predicted: list[float], actual: list[float]) -> tuple[float, float]:
    """Fit ``actual ≈ slope * predicted + intercept`` and return floats."""
    slope, intercept = np.polyfit(np.array(predicted), np.array(actual), 1)
    return float(slope), float(intercept)


async def get_calibration_coefficients(
    session: AsyncSession,
) -> dict[str, tuple[float, float]] | None:
    """Compute linear recalibration coefficients from resolved signals.

    Queries signals that have associated trades with status WON or LOST,
    partitions rows by operator class (``"threshold"`` vs
    ``"bracket-like"``), fits ``actual = slope * predicted + intercept``
    per class with ≥ :data:`MIN_CALIBRATION_SAMPLES` rows, and always
    additionally fits a pooled regression over the union. Returns a dict
    of ``{class_key: (slope, intercept)}`` — always includes
    ``POOLED_KEY`` when the pooled fit met the minimum, plus any
    per-class key that met it independently. Returns ``None`` when the
    pooled fit itself can't be made.

    Convention: probabilities are stored in the **side-effective frame**
    — i.e. the model's probability of the side the trade actually bet on
    (= P(YES) for BUY_YES, = 1-P(YES) for BUY_NO). That keeps the
    regression's input range consistent across both directions:
    ``predicted`` is "model's confidence in winning", ``actual`` is "did
    we win", and the slope/intercept describe how well-calibrated those
    confidences are.

    ``predicted`` reads from ``Signal.raw_model_prob`` — the engine
    output *before* calibration was applied. Reading ``Signal.model_prob``
    here (the calibrated value) created a feedback loop that defeated
    the point of recalibration. Legacy rows where ``raw_model_prob IS
    NULL`` fall back to ``model_prob`` so they continue to contribute,
    with the caveat that those values are already calibrated.
    """
    stmt = (
        select(Signal)
        .options(joinedload(Signal.trades), joinedload(Signal.market))
        .join(Trade, Trade.signal_id == Signal.id)
        .where(Trade.status.in_([TradeStatus.WON, TradeStatus.LOST]))
    )
    result = await session.execute(stmt)
    signals = result.unique().scalars().all()

    if len(signals) < MIN_CALIBRATION_SAMPLES:
        return None

    # Bucket rows into per-class pools + the global pool. The signed
    # rows-without-a-class case (parsed_operator NULL or unknown) is
    # benign: they still contribute to the pooled fit but not to any
    # class-specific fit.
    per_class: dict[str, tuple[list[float], list[float]]] = {
        key: ([], []) for key in _OP_CLASS_KEYS
    }
    pooled_pred: list[float] = []
    pooled_actual: list[float] = []
    for sig in signals:
        # Cast through ``float()`` so mypy narrows past the SQLAlchemy
        # ``Column[float]`` type (the same pattern that the legacy
        # single-fit code used implicitly via ``np.array``).
        raw = float(
            sig.raw_model_prob
            if sig.raw_model_prob is not None
            else sig.model_prob
        )
        won = any(t.status == TradeStatus.WON for t in sig.trades)
        actual_val = 1.0 if won else 0.0

        pooled_pred.append(raw)
        pooled_actual.append(actual_val)

        cls = _operator_class(sig.market) if sig.market is not None else None
        if cls in per_class:
            per_class[cls][0].append(raw)
            per_class[cls][1].append(actual_val)

    coeffs: dict[str, tuple[float, float]] = {}

    pooled = _fit_linear(pooled_pred, pooled_actual)
    coeffs[POOLED_KEY] = pooled
    logger.info(
        "Calibration fitted (pooled) on %d signals: slope=%.4f intercept=%.4f",
        len(pooled_pred), pooled[0], pooled[1],
    )

    for cls, (pred, actual) in per_class.items():
        if len(pred) >= MIN_CALIBRATION_SAMPLES:
            coeffs[cls] = _fit_linear(pred, actual)
            logger.info(
                "Calibration fitted (%s) on %d signals: slope=%.4f intercept=%.4f",
                cls, len(pred), coeffs[cls][0], coeffs[cls][1],
            )
        else:
            logger.info(
                "Calibration: %s class has %d signals (< %d) — using pooled",
                cls, len(pred), MIN_CALIBRATION_SAMPLES,
            )

    return coeffs


async def refresh_calibration(
    session: AsyncSession,
) -> dict[str, tuple[float, float]] | None:
    """Refresh the in-process calibration cache.

    Called from the unified pipeline tick. Cheap when called more often
    than `_CACHE_TTL_SEC` (returns the cached value); fits a fresh
    regression otherwise.
    """
    global _cached_coeffs, _cached_at
    now = time.time()
    if _cached_at is not None and (now - _cached_at) < _CACHE_TTL_SEC:
        return _cached_coeffs

    coeffs = await get_calibration_coefficients(session)
    _cached_coeffs = coeffs
    _cached_at = now
    return coeffs


def get_cached_calibration(
    operator_class: str | None = None,
) -> tuple[float, float] | None:
    """Sync read of the in-process calibration cache.

    Lookup order:
      1. ``operator_class`` key when ``PER_OPERATOR_CALIBRATION_ENABLED``
         is True and the class has its own cached fit;
      2. the ``POOLED_KEY`` fallback (used whenever the class lookup is
         disabled, missing, or the caller didn't supply a class);
      3. ``None`` when the cache is empty, stale, or no fit exists.

    Returns ``None`` when no fit has succeeded yet, when the cache has
    aged out, or when too few resolved signals exist (the underlying fit
    needs ``MIN_CALIBRATION_SAMPLES`` resolved trades).
    """
    if _cached_at is None or _cached_coeffs is None:
        return None
    if (time.time() - _cached_at) > _CACHE_TTL_SEC:
        return None

    if (
        operator_class is not None
        and getattr(settings, "PER_OPERATOR_CALIBRATION_ENABLED", False)
        and operator_class in _cached_coeffs
    ):
        return _cached_coeffs[operator_class]
    return _cached_coeffs.get(POOLED_KEY)


def apply_calibration(
    prob: float,
    operator_class: str | None = None,
) -> tuple[float, bool]:
    """Apply the cached linear calibration to a side-effective probability.

    Returns ``(corrected_prob, applied)``. When the
    ``APPLY_CALIBRATION`` setting is False or no coefficients are
    cached, returns ``(prob, False)`` so the caller can log/branch.

    ``operator_class`` is forwarded to :func:`get_cached_calibration` so
    the caller can opt into the per-class fit when the
    ``PER_OPERATOR_CALIBRATION_ENABLED`` flag is set. The flag-off path
    always uses the pooled fit — bit-for-bit identical to pre-split
    behavior.

    The regression is fit on ``Signal.raw_model_prob`` (the engine output
    *before* this regression was applied — see
    :func:`get_calibration_coefficients` for the feedback-loop reasoning),
    in the side-effective frame: the model's confidence on the side a
    trade actually bet on. Applying it to a raw P(YES) for a NO trade
    would mix two distributions; callers must apply it after side
    selection.
    """
    if not getattr(settings, "APPLY_CALIBRATION", False):
        return prob, False
    coeffs = get_cached_calibration(operator_class)
    if coeffs is None:
        return prob, False
    slope, intercept = coeffs
    corrected = max(0.0, min(1.0, slope * prob + intercept))
    return corrected, True


def _apply_coeffs(prob: float, coeffs: tuple[float, float] | None) -> float | None:
    """Clamp ``slope*prob + intercept`` to [0,1]; None when no coeffs."""
    if coeffs is None:
        return None
    slope, intercept = coeffs
    return max(0.0, min(1.0, slope * prob + intercept))


def _fresh_coeffs_for_key(key: str) -> tuple[float, float] | None:
    """Cache read for one key, bypassing the per-class enable FLAG.

    Unlike :func:`get_cached_calibration` this does NOT consult
    ``PER_OPERATOR_CALIBRATION_ENABLED`` — it returns whatever fit is
    cached under ``key`` so the shadow logger can compare pooled vs
    per-class regardless of which one is live. Honours the TTL.
    """
    if _cached_at is None or _cached_coeffs is None:
        return None
    if (time.time() - _cached_at) > _CACHE_TTL_SEC:
        return None
    return _cached_coeffs.get(key)


def shadow_calibration(
    raw_prob: float | None, operator_class: str | None
) -> dict | None:
    """Counterfactual calibration outputs for ``evaluation_logs.shadow_json``.

    Pure **telemetry** — never influences a trade. Returns a dict (to be
    stored under the ``cal`` namespace) carrying the raw probability plus
    what BOTH the pooled fit and this operator class's own fit would
    produce for it, so ``shadow-report`` can measure whether flipping
    ``PER_OPERATOR_CALIBRATION_ENABLED`` would meaningfully move the
    threshold [0.78,0.85) band BEFORE the flip touches live trading.

    Returns ``None`` when shadow logging is disabled, the input is
    missing, or no pooled fit is cached yet (nothing to compare).
    ``class`` is omitted when this class has no ≥``MIN_CALIBRATION_SAMPLES``
    fit of its own (the flag-on path would fall back to pooled there).
    """
    if not getattr(settings, "SHADOW_CALIBRATION_ENABLED", False):
        return None
    if raw_prob is None:
        return None
    pooled = _fresh_coeffs_for_key(POOLED_KEY)
    if pooled is None:
        return None
    raw = float(raw_prob)
    out: dict = {
        "raw": round(raw, 4),
        "op_class": operator_class,
        "pooled": round(_apply_coeffs(raw, pooled) or 0.0, 4),
    }
    if operator_class is not None:
        cls = _apply_coeffs(raw, _fresh_coeffs_for_key(operator_class))
        if cls is not None:
            out["class"] = round(cls, 4)
            out["delta"] = round(out["class"] - out["pooled"], 4)
    return out


def reset_calibration_cache() -> None:
    """Test helper — drop the in-process cache."""
    global _cached_coeffs, _cached_at
    _cached_coeffs = None
    _cached_at = None
