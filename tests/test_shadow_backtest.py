"""Unit tests for the pure helpers of the shadow-backtest replay harness."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from src.risk.shadow_backtest import _price_as_of, _spec_from_resolution


def _res(**kw):
    base = dict(parsed_operator=None, parsed_threshold=None,
               bucket_low_f=None, bucket_high_f=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_spec_from_threshold_resolution():
    spec = _spec_from_resolution(_res(parsed_operator="at_least", parsed_threshold=85.0))
    assert spec is not None
    assert spec.operator == "at_least"
    assert spec.threshold_f == 85.0


def test_spec_from_bracket_resolution():
    spec = _spec_from_resolution(
        _res(parsed_operator="range", bucket_low_f=70.0, bucket_high_f=72.0))
    assert spec is not None
    assert (spec.low_f, spec.high_f) == (70.0, 72.0)


def test_spec_none_when_missing_fields():
    assert _spec_from_resolution(_res(parsed_operator="above")) is None  # no threshold
    assert _spec_from_resolution(_res(parsed_operator="range")) is None  # no bounds
    assert _spec_from_resolution(_res(parsed_operator="weird")) is None


def test_price_as_of_picks_latest_prior():
    def t(h):
        return datetime(2026, 6, 1, h, tzinfo=timezone.utc)
    snaps = [(t(8), 0.30), (t(10), 0.45), (t(12), 0.60)]
    assert _price_as_of(snaps, t(11)) == 0.45   # latest at/before 11:00
    assert _price_as_of(snaps, t(12)) == 0.60   # inclusive
    assert _price_as_of(snaps, t(7)) is None     # nothing before
