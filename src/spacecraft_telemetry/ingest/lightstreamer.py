"""Shared Lightstreamer session helpers used by both the collector daemon and
the live pump.

The standalone collector VM is retired (Phase 17 folds collection into the
always-on live pump — see CLAUDE.md Phase 17), but ``collector.py`` is kept for
local dry-runs and shares its Lightstreamer session plumbing with
``api/live/pump.py``. Factoring the shared pieces out here means neither
module imports underscore-private names from the other.

Log event keys below are prefixed ``collector.*`` for historical reasons (this
module predates the live pump); they are intentionally left unchanged when
reused by the pump rather than renamed, to avoid silently breaking any
log-based dashboards or alerts built against them.
"""

from __future__ import annotations

import os

from spacecraft_telemetry.core.logging import get_logger

log = get_logger(__name__)

# Maximum rows buffered/archived per channel when the destination (GCS) is
# unavailable. At ~2 s median cadence and 28 channels this caps memory at
# roughly 6 hours of ticks per channel (< 50 MB total) before old-first rows
# are dropped and an overflow warning is logged.
MAX_BUFFERED_ROWS = 10_800  # ~6 h at 2 s cadence


# ---------------------------------------------------------------------------
# TLS certificate bundle
# ---------------------------------------------------------------------------


def ensure_ssl_cert_env() -> None:
    """Point SSL_CERT_FILE at certifi's bundle if it is not already set.

    uv-managed CPython on macOS does not load the system keychain, so
    ``ssl.create_default_context()`` (used by the Lightstreamer client's aiohttp
    transport) fails verification against push.lightstreamer.com and the
    connection silently retries forever. certifi is always present (httpx,
    a core dependency, requires it). ``setdefault`` semantics mean an explicit
    SSL_CERT_FILE — including the Debian system store inside the Docker image —
    still wins.
    """
    if os.environ.get("SSL_CERT_FILE"):
        return
    import certifi

    os.environ["SSL_CERT_FILE"] = certifi.where()
    log.info("collector.ssl_cert_file_set", path=certifi.where())


# ---------------------------------------------------------------------------
# Client listener — logs every connection state transition
# ---------------------------------------------------------------------------


class ISSClientListener:
    """Lightstreamer ClientListener that logs connection status changes.

    Lightstreamer's connect() is non-blocking; all connection events arrive
    here on a library thread. Without this listener, callers are silent while
    connecting/retrying, making it impossible to distinguish "still
    connecting" from "stuck".

    Status strings emitted by the library:
        CONNECTING, CONNECTED:WS-STREAMING, CONNECTED:HTTP-STREAMING,
        STALLED, DISCONNECTED:WILL-RETRY, DISCONNECTED:TRYING-RECOVERY,
        DISCONNECTED
    """

    def onStatusChange(self, status: str) -> None:
        connected = status.startswith("CONNECTED")
        level = "info" if connected else "warning"
        getattr(log, level)("collector.connection_status", status=status)

    def onServerError(self, code: int, message: str) -> None:
        log.error("collector.server_error", code=code, message=message)

    def onPropertyChange(self, prop: str) -> None:
        pass

    def onListenEnd(self) -> None:
        pass

    def onListenStart(self) -> None:
        pass
