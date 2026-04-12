"""Reusable logging configuration for WeatherEdge.

Call ``configure_logging()`` from notebook code or scheduler to suppress
noisy third-party loggers while keeping WeatherEdge loggers at the desired level.
"""

import logging
import sys

NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "sqlalchemy",
    "sqlalchemy.engine",
    "asyncpg",
    "urllib3",
    "aiosqlite",
    "hpack",
    "h2",
)


def configure_logging(level: int = logging.INFO) -> None:
    """Set up root logger and suppress noisy third-party loggers.

    Args:
        level: Log level for the ``src`` logger hierarchy.  Third-party
            loggers are always set to WARNING regardless of this value.
    """
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        ))
        root.addHandler(handler)

    root.setLevel(logging.WARNING)
    logging.getLogger("src").setLevel(level)

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
