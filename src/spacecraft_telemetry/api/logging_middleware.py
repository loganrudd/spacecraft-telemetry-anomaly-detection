"""Correlation ID middleware for the FastAPI serving layer.

Attaches a ``X-Correlation-Id`` header to every response and binds it to the
structlog context for the duration of the request so that all log lines emitted
within a single request carry the same ``request_id`` field.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
import structlog.contextvars
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from spacecraft_telemetry.core.logging import get_logger


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Attach a correlation ID to every request/response cycle.

    - If the client sends ``X-Correlation-Id``, it is reused verbatim.
    - Otherwise a new UUID4 is generated.
    - The ID is bound to the structlog context so every log line in the
      request handler includes ``request_id``.
    - The ID is echoed back in the ``X-Correlation-Id`` response header.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        correlation_id = request.headers.get("X-Correlation-Id") or str(uuid.uuid4())
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=correlation_id,
            route=str(request.url.path),
            method=request.method,
        )
        log = get_logger("api.request")
        log.info("request.start")
        start = time.perf_counter()
        try:
            response: Response = await call_next(request)
        except Exception:
            log.exception("request.error")
            raise
        else:
            response.headers["X-Correlation-Id"] = correlation_id
            log.info(
                "request.end",
                status_code=response.status_code,
                duration_ms=round((time.perf_counter() - start) * 1000, 2),
            )
            return response
        finally:
            structlog.contextvars.clear_contextvars()
