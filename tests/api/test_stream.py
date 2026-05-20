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
    "is_anomaly",
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
        """Stream should emit exactly 100 events for a 100-row Parquet."""
        events = _collect_events(TestClient(running_app))
        assert len(events) == 100

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
        anomaly_events = [ev for ev in all_events[80:] if ev["is_anomaly"]]
        assert len(anomaly_events) == 20

    def test_is_anomaly_predicted_is_bool(
        self, all_events: list[dict[str, object]]
    ) -> None:
        for ev in all_events:
            assert isinstance(ev["is_anomaly_predicted"], bool)

    def test_is_anomaly_is_bool(
        self, all_events: list[dict[str, object]]
    ) -> None:
        for ev in all_events:
            assert isinstance(ev["is_anomaly"], bool)


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


# ---------------------------------------------------------------------------
# Predicted anomaly flag reaches True end-to-end
# ---------------------------------------------------------------------------


class TestStreamPredictedAnomaly:
    def test_is_anomaly_predicted_true_in_spike_region(
        self, running_app_with_spike: FastAPI
    ) -> None:
        """Engine produces is_anomaly_predicted=True for a spike series.

        Verifies the end-to-end chain: Parquet row → engine.step() →
        TelemetryEvent.is_anomaly_predicted → JSON → SSE → client parse.
        Fixture uses _ZeroModel + test.yaml params; spike at rows 31-59
        triggers the flag from tick 32 onwards (K=2 trailing True flags).
        """
        events = _collect_events(
            TestClient(running_app_with_spike),
            url=_STREAM_URL,
        )
        assert any(ev["is_anomaly_predicted"] for ev in events), (
            "Expected at least one is_anomaly_predicted=True event in the spike stream"
        )


# ---------------------------------------------------------------------------
# Mid-stream disconnect does not raise
# ---------------------------------------------------------------------------


class TestStreamMultiChannel:
    """Multi-channel stream regression guard for the per-channel queue design.

    The old single-queue design would head-of-line-block all pumps when any
    one channel's consumer stalled.  These tests verify that events from all
    requested channels arrive, which would fail if the merger drained only
    one channel's queue.
    """

    _TWO_CH_URL = "/api/stream/telemetry?speed=1000&channels=test-ch,test-ch-b"

    def test_both_channels_emit_events(self, running_app_multi_ch: FastAPI) -> None:
        events = _collect_events(TestClient(running_app_multi_ch), url=self._TWO_CH_URL)
        channels_seen = {ev["channel"] for ev in events}
        assert "test-ch" in channels_seen
        assert "test-ch-b" in channels_seen

    def test_total_event_count_matches_two_channels(
        self, running_app_multi_ch: FastAPI
    ) -> None:
        """50 rows x 2 channels = 100 total events."""
        events = _collect_events(TestClient(running_app_multi_ch), url=self._TWO_CH_URL)
        assert len(events) == 100

    def test_events_interleaved_not_sequential(
        self, running_app_multi_ch: FastAPI
    ) -> None:
        """Channel values in the stream should alternate rather than run
        all-ch-a then all-ch-b, confirming the merger races both queues.

        Checks that neither channel dominates the first half of the stream.
        """
        events = _collect_events(TestClient(running_app_multi_ch), url=self._TWO_CH_URL)
        first_half = events[:50]
        ch_a = sum(1 for e in first_half if e["channel"] == "test-ch")
        ch_b = sum(1 for e in first_half if e["channel"] == "test-ch-b")
        # Both channels should contribute to the first half of events.
        # Allow some skew (each contributes at least 5 of 50) but reject
        # the 50/0 split that a sequential (non-merged) design would produce.
        assert ch_a >= 5, f"test-ch underrepresented in first half: {ch_a}/50"
        assert ch_b >= 5, f"test-ch-b underrepresented in first half: {ch_b}/50"


class TestStreamDisconnect:
    def test_partial_read_no_exception(self, running_app: FastAPI, mocker) -> None:
        """Client disconnect should not surface as a server-side error.

        Pump cancellation raises CancelledError, which the cleanup path must
        NOT forward to the error logger (issue 2.3 regression guard).
        """
        mock_log = mocker.patch("spacecraft_telemetry.api.streaming._log")
        events = _collect_events(TestClient(running_app), max_events=3)
        assert len(events) == 3
        mock_log.error.assert_not_called()
