"""Structured JSON logging for the production scheduler.

Distinct from ``src.logging_utils`` (which is plain-text configuration
for notebooks / one-shot scripts). This module ships a single
JSON-to-stdout handler so log lines are aggregator-friendly when the bot
runs as a daemon.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone


# Keys present on every LogRecord by stdlib. Anything not in this set
# AND not in ``_STDLIB_RESERVED`` is treated as an ``extra=`` field and
# merged into the JSON payload so callers can use the standard
# ``logger.info("...", extra={"icao": ..., "market_id": ...})`` pattern
# without each call site having to format identifiers into the message.
_STDLIB_RESERVED = frozenset({
    "args", "asctime", "created", "exc_info", "exc_text", "filename",
    "funcName", "levelname", "levelno", "lineno", "message", "module",
    "msecs", "msg", "name", "pathname", "process", "processName",
    "relativeCreated", "stack_info", "thread", "threadName",
    "taskName",  # py3.12+
})


class JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line on stdout.

    Recognises ``extra={...}`` fields and merges them into the payload.
    Built-in keys (``timestamp``, ``level``, ``logger``, ``message``,
    ``exception``) take precedence so an ``extra`` field can't clobber
    them by accident.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Anything attached to the record that isn't a standard LogRecord
        # attribute came from ``extra=`` (or a custom adapter). Take a
        # shallow snapshot so the consumer's dict isn't mutated.
        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in _STDLIB_RESERVED and not k.startswith("_")
        }
        entry: dict[str, object] = dict(extras)
        entry.update({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        })
        if record.exc_info and record.exc_info[0]:
            entry["exception"] = self.formatException(record.exc_info)
        # ``default=str`` keeps the formatter resilient to things like
        # Decimal / datetime / UUID extras a future caller might pass.
        return json.dumps(entry, default=str)


def configure_logging() -> None:
    """Replace root handlers with a single JSON-to-stdout handler.

    Suppresses noisy third-party loggers (``httpx`` URLs can leak Telegram
    tokens; ``sqlalchemy.engine`` is wall-of-text at INFO) by raising them
    to WARNING. The list lives in ``src.logging_utils.NOISY_LOGGERS`` so
    both this module and notebook-style ``logging_utils.configure_logging``
    stay in sync.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)
    from src.logging_utils import NOISY_LOGGERS
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
