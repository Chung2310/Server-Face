"""Persistent outbound WebSocket connection to the ERP server.

Identification events produced by the live stream are pushed here so the ERP
records attendance without polling. The connection is held open and
re-established automatically; events produced while it is down are buffered in a
*bounded* queue and the oldest are dropped once it fills, so a long ERP outage
cannot exhaust memory.

Security note: messages arriving *from* the ERP are treated as data, never as
instructions. Only an explicit allow-list of command names is acted on; anything
else is logged and ignored.
"""

import asyncio
import json
import logging
import random
from collections import deque
from datetime import datetime, timezone
from typing import Optional

from app.config import settings

logger = logging.getLogger("uvicorn.error")

# The only inbound commands this service will act on. Everything else is ignored.
ALLOWED_INBOUND_COMMANDS = frozenset({"refresh_registry", "ping"})


class ErpBridge:
    """Singleton bridge, following the pattern used by the other services here."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._queue: deque = deque(maxlen=settings.ERP_WS_QUEUE_MAX)
        self._task: Optional[asyncio.Task] = None
        self._wakeup = asyncio.Event()
        self._connected = False
        self._dropped = 0
        self._initialized = True

    @property
    def enabled(self) -> bool:
        return bool((settings.ERP_WS_URL or "").strip())

    @property
    def connected(self) -> bool:
        return self._connected

    # ── lifecycle ──

    async def start(self) -> None:
        if not self.enabled:
            logger.info("ERP bridge disabled (ERP_WS_URL is not configured)")
            return
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._run(), name="erp-bridge")
        logger.info("ERP bridge starting")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("ERP bridge stopped with error: %s", type(exc).__name__)
        self._connected = False
        logger.info("ERP bridge stopped")

    # ── producing ──

    def publish(self, event: dict) -> None:
        """Queue an event for delivery. Never blocks and never raises."""
        if not self.enabled:
            return
        if len(self._queue) == self._queue.maxlen:
            self._dropped += 1
            if self._dropped % 100 == 1:
                logger.warning(
                    "ERP bridge queue full (max %d); dropping oldest events (%d dropped)",
                    self._queue.maxlen,
                    self._dropped,
                )
        self._queue.append(event)
        self._wakeup.set()

    def publish_identification(self, user_id: str, similarity: float, live: Optional[bool]) -> None:
        self.publish(
            {
                "type": "attendance",
                "user_id": user_id,
                "similarity": round(float(similarity), 4),
                "live": live,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    # ── connection loop ──

    async def _run(self) -> None:
        try:
            import websockets
        except ImportError:
            logger.error("ERP bridge requires the 'websockets' package; bridge will not run")
            return

        url = settings.ERP_WS_URL.strip()
        delay = settings.ERP_WS_BACKOFF_MIN_SECONDS
        headers = {}
        if settings.ERP_WS_API_KEY:
            headers["X-API-Key"] = settings.ERP_WS_API_KEY

        # websockets renamed extra_headers -> additional_headers in v14.
        import inspect
        header_kwarg = (
            "additional_headers"
            if "additional_headers" in inspect.signature(websockets.connect).parameters
            else "extra_headers"
        )

        while True:
            try:
                connect_kwargs = {
                    "ping_interval": settings.ERP_WS_PING_INTERVAL_SECONDS,
                    "open_timeout": 10,
                }
                if headers:
                    connect_kwargs[header_kwarg] = headers
                async with websockets.connect(url, **connect_kwargs) as socket:
                    self._connected = True
                    delay = settings.ERP_WS_BACKOFF_MIN_SECONDS
                    logger.info("ERP bridge connected")
                    await self._pump(socket)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Log the type only - the URL may embed credentials.
                logger.warning("ERP bridge connection failed: %s", type(exc).__name__)
            finally:
                self._connected = False

            # Full jitter, so a fleet of workers does not reconnect in lockstep.
            await asyncio.sleep(random.uniform(0, delay))
            delay = min(delay * 2, settings.ERP_WS_BACKOFF_MAX_SECONDS)

    async def _pump(self, socket) -> None:
        """Send queued events and handle inbound messages until the socket drops."""
        sender = asyncio.create_task(self._send_loop(socket))
        receiver = asyncio.create_task(self._receive_loop(socket))
        done, pending = await asyncio.wait(
            {sender, receiver}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            # Re-raise so the outer loop sees the disconnect and backs off.
            task.result()

    async def _send_loop(self, socket) -> None:
        while True:
            while self._queue:
                event = self._queue[0]
                await socket.send(json.dumps(event))
                # Only drop it once the send succeeded, so a mid-send failure
                # leaves the event queued for the next connection.
                if self._queue and self._queue[0] is event:
                    self._queue.popleft()
            self._wakeup.clear()
            await self._wakeup.wait()

    async def _receive_loop(self, socket) -> None:
        async for raw in socket:
            try:
                message = json.loads(raw)
            except (TypeError, ValueError):
                logger.warning("ERP bridge ignored a non-JSON inbound message")
                continue
            if not isinstance(message, dict):
                logger.warning("ERP bridge ignored an inbound message that was not an object")
                continue

            command = message.get("command")
            if command not in ALLOWED_INBOUND_COMMANDS:
                logger.warning("ERP bridge ignored unrecognised inbound command: %r", command)
                continue

            if command == "refresh_registry":
                from app.services.face_index import FaceIndex
                try:
                    await FaceIndex().refresh()
                except Exception as exc:
                    logger.warning("Registry refresh failed: %s", type(exc).__name__)
