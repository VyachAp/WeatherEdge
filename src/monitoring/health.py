"""HTTP health-check server using stdlib asyncio.

Tiny single-handler server that returns ``{"status": "ok", ...}`` on any
request. Used by container orchestrators / uptime monitors to verify the
scheduler process is alive.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Callable

logger = logging.getLogger(__name__)


def _make_handler(
    scheduler_running: Callable[[], bool],
) -> Callable[[asyncio.StreamReader, asyncio.StreamWriter], asyncio.Future]:
    """Build a connection handler closed over a scheduler-status callable.

    The callable indirection avoids importing the scheduler module here
    (which would be a circular import) and lets tests pass a stub.
    """

    async def _handler(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            await reader.read(4096)  # consume request
            body = json.dumps({
                "status": "ok",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "scheduler_running": scheduler_running(),
            })
            response = (
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\n"
                "\r\n"
                f"{body}"
            )
            writer.write(response.encode())
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    return _handler


async def start_health_server(
    scheduler_running: Callable[[], bool],
    port: int = 8080,
) -> asyncio.Server:
    """Start the health server on ``port``. ``scheduler_running`` should
    return True when the APScheduler instance is actively running."""
    server = await asyncio.start_server(
        _make_handler(scheduler_running), "0.0.0.0", port,
    )
    logger.info("Health-check server listening on port %d", port)
    return server
