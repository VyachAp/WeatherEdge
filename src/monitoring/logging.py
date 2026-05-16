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


class JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line on stdout."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, object] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry)


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
