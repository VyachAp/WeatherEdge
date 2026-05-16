"""Tests for ``monitoring.logging.JSONFormatter``.

Locks in three contracts that affect every aggregator query downstream:

1. The fixed envelope (timestamp / level / logger / message) is always
   present.
2. ``extra={...}`` fields land alongside the envelope as top-level JSON
   keys so they're queryable without parsing the message string.
3. Built-in keys win on collision — an ``extra={"level": ...}`` can't
   silently corrupt the level field consumers grep by.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from src.monitoring.logging import JSONFormatter


def _record(msg: str, *, level: int = logging.INFO, extra: dict | None = None) -> logging.LogRecord:
    rec = logging.LogRecord(
        name="t", level=level, pathname=__file__, lineno=1,
        msg=msg, args=(), exc_info=None,
    )
    if extra:
        for k, v in extra.items():
            setattr(rec, k, v)
    return rec


def test_envelope_fields_always_present():
    out = json.loads(JSONFormatter().format(_record("hello")))
    assert out["level"] == "INFO"
    assert out["logger"] == "t"
    assert out["message"] == "hello"
    # ISO-8601 with timezone — parseable as a datetime.
    parsed = datetime.fromisoformat(out["timestamp"])
    assert parsed.tzinfo is not None


def test_extras_merge_into_payload():
    rec = _record(
        "lock fired",
        extra={"icao": "KPHX", "market_id": "0xabc", "stake_usd": 12.5},
    )
    out = json.loads(JSONFormatter().format(rec))
    assert out["icao"] == "KPHX"
    assert out["market_id"] == "0xabc"
    assert out["stake_usd"] == 12.5
    # Envelope still intact.
    assert out["message"] == "lock fired"


def test_envelope_keys_win_on_collision():
    """If a caller accidentally passes ``extra={"level": "DEBUG"}``,
    the real level must win — log aggregator queries on ``level``
    have to be trustworthy."""
    rec = _record(
        "boom", level=logging.ERROR,
        extra={"level": "TRACE", "message": "spoof"},
    )
    out = json.loads(JSONFormatter().format(rec))
    assert out["level"] == "ERROR"
    assert out["message"] == "boom"


def test_non_jsonable_extras_serialize_via_str():
    """Datetime / Decimal / UUID-shaped values shouldn't crash the
    formatter — production code uses datetimes liberally."""
    from datetime import datetime, timezone

    rec = _record("evt", extra={"observed_at": datetime(2026, 5, 16, 12, 30, tzinfo=timezone.utc)})
    out = json.loads(JSONFormatter().format(rec))
    assert out["observed_at"] == "2026-05-16 12:30:00+00:00"


def test_exception_is_captured_in_envelope():
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        import sys
        exc_info = sys.exc_info()
    rec = logging.LogRecord(
        name="t", level=logging.ERROR, pathname=__file__, lineno=1,
        msg="failed", args=(), exc_info=exc_info,
    )
    out = json.loads(JSONFormatter().format(rec))
    assert "RuntimeError: boom" in out["exception"]
