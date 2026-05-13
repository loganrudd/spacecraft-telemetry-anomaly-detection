"""Tests for GET /api/stream/telemetry (SSE endpoint).

Design notes
------------
- Uses ``TestClient.stream()`` (sync) — no asyncio required.  The ASGI app
  runs in anyio under the hood, so asyncio tasks inside the streaming generator
  work correctly.
- The ``running_app`` fixture (conftest.py) injects a _ZeroModel engine and a
  100-row Parquet with 20 anomaly rows in the tail (rows 80-99).
- At speed=1000 / tick_interval=0.001, each tick sleeps ~1 µs — the full
  stream completes in well under 1 second.
- SSE lines: ``event: telemetry`` / ``data: <JSON>`` / (empty separator).
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STREAM_URL = "/api/stream/telemetry?speed=1000"
_REQUIRED_FIELDS = {
    "timestamp",
    "mission",
    "channel",
    "value_normalized",
    "prediction",
    "residual",
    "smoothed_error",
    "threshold",
    "is_anomaly_predicted",
    "is_anomaly_true",
}


def _collect_events(
    client: TestClient, url: str = _STREAM_URL, max_events: int | None = None
) -> list[dict[str, object]]:
    """Open the SSE stream, parse data lines, return a list of event dicts."""
    events: list[dict[str, object]] = []
    with client.stream("GET", url) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line[5:].strip()))
                if max_events is not None and len(events) >= max_events:
                    break
    return events


# ---------------------------------------------------------------------------
# Basic connectivity and response format
# ---------------------------------------------------------------------------


class TestStreamBasic:
    def test_returns_200(self, running_app: FastAPI) -> None:
        with TestClient(running_app).stream("GET", _STREAM_URL) as resp:
            assert resp.status_code == 200

    def test_content_type_is_event_stream(self, running_app: FastAPI) -> None:
        with TestClient(running_app).stream("GET", _STREAM_URL) as resp:
            assert "text/event-stream" in resp.headers["content-type"]

    def test_emits_events(self, running_app: FastAPI) -> None:
        """Stream should emit at least 30 events for a 100-row Parquet."""
        events = _collect_events(TestClient(running_app))
        assert len(events) >= 30

    def test_events_have_required_fields(self, running_app: FastAPI) -> None:
        events = _collect_events(TestClient(running_app), max_events=5)
        for ev in events:
            missing = _REQUIRED_FIELDS - set(ev)
            assert not missing, f"Event missing fields: {missing}"

    def test_channel_field_matches_fixture(self, running_app: FastAPI) -> None:
        events = _collect_events(TestClient(running_app), max_events=5)
        for ev in events:
            assert ev["channel"] == "test-ch"


# ---------------------------------------------------------------------------
# Event content correctness
# ---------------------------------------------------------------------------


class TestStreamContent:
    @pytest.fixture()
    def all_events(self, running_app: FastAPI) -> list[dict[str, object]]:
        """Collect all 100 events from the stream (Parquet has 100 rows)."""
        return _collect_events(TestClient(running_app))

    def test_at_least_100_events(self, all_events: list[dict[str, object]]) -> None:
        assert len(all_events) == 100

    def test_warmup_events_have_none_prediction(
        self, all_events: list[dict[str, object]]
    ) -> None:
        """First window_size-1 events must have prediction=None."""
        # window_size = 10 in test config → first 9 ticks are warmup
        for ev in all_events[:9]:
            assert ev["prediction"] is None

    def test_post_warmup_events_have_prediction(
        self, all_events: list[dict[str, object]]
    ) -> None:
        """Events past warmup must have a float prediction."""
        # From tick 10 onward (index 9) predictions exist
        for ev in all_events[9:]:
            assert ev["prediction"] is not None
            assert isinstance(ev["prediction"], float)

    def test_anomaly_labeled_rows_visible(
        self, all_events: list[dict[str, object]]
    ) -> None:
        """Last 20 rows of the fixture Parquet are labeled anomalies."""
        # Parquet has 100 rows; last 20 have is_anomaly=True
        anomaly_events = [ev for ev in all_events[80:] if ev["is_anomaly_true"]]
        assert len(anomaly_events) >= 1

    def test_is_anomaly_predicted_is_bool(
        self, all_events: list[dict[str, object]]
    ) -> None:
        for ev in all_events:
            assert isinstance(ev["is_anomaly_predicted"], bool)

    def test_is_anomaly_true_is_bool(
        self, all_events: list[dict[str, object]]
    ) -> None:
        for ev in all_events:
            assert isinstance(ev["is_anomaly_true"], bool)


# ---------------------------------------------------------------------------
# Unknown channel rejection
# ---------------------------------------------------------------------------


class TestStreamChannelValidation:
    def test_unknown_channel_returns_400(self, running_app: FastAPI) -> None:
        resp = TestClient(running_app).get(
            "/api/stream/telemetry?speed=1000&channels=no-such-channel"
        )
        assert resp.status_code == 400

    def test_error_body_mentions_unknown_channel(self, running_app: FastAPI) -> None:
        resp = TestClient(running_app).get(
            "/api/stream/telemetry?speed=1000&channels=no-such-channel"
        )
        assert "no-such-channel" in resp.json()["detail"]

    def test_explicit_valid_channel_accepted(self, running_app: FastAPI) -> None:
        events = _collect_events(
            TestClient(running_app),
            url="/api/stream/telemetry?speed=1000&channels=test-ch",
            max_events=5,
        )
        assert len(events) == 5


# ---------------------------------------------------------------------------
# Mid-stream disconnect does not raise
# ---------------------------------------------------------------------------


class TestStreamDisconnect:
    def test_partial_read_no_exception(self, running_app: FastAPI) -> None:
        """Reading only a few events and closing should not raise."""
        events = _collect_events(TestClient(running_app), max_events=3)
        assert len(events) == 3
