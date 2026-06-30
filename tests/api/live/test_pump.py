"""Unit tests for LivePump.

Drives _on_tick directly (no live Lightstreamer connection) and asserts:
1. Normalized ticks reach the broadcaster as event: raw payloads.
2. Telemetry events (event: telemetry) are emitted after a grid bucket closes.
3. Fault injection choreography: injected raw value differs from nominal;
   is_anomaly=True flows through to the telemetry event.
4. LOS onset → status event emitted, fallback task started.
5. LOS recovery → status event, replay task cancelled, engines re-primed.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest

torch = pytest.importorskip("torch")

from spacecraft_telemetry.api.broadcast import EventBroadcaster  # noqa: E402
from spacecraft_telemetry.api.inference import ChannelInferenceEngine  # noqa: E402
from spacecraft_telemetry.api.live.pump import LivePump  # noqa: E402
from spacecraft_telemetry.core.config import CollectorConfig, load_settings  # noqa: E402
from spacecraft_telemetry.model.io import ScoringParams  # noqa: E402

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

_CH = "S1000003"
_MEAN = 20.5
_STD = 3.2
_INTERVAL = 30  # grid_interval_seconds

_BASE_TS = datetime(2000, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ZeroModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], 1)


def _make_engine(window_size: int = 5) -> ChannelInferenceEngine:
    m = _ZeroModel()
    m.eval()
    return ChannelInferenceEngine(
        mission="ISS",
        channel=_CH,
        model=m,  # type: ignore[arg-type]
        window_size=window_size,
        params=ScoringParams(
            threshold_window=8,
            threshold_z=1.0,
            error_smoothing_window=4,
            threshold_min_anomaly_len=3,
        ),
        device=torch.device("cpu"),
    )


def _make_collect_config() -> CollectorConfig:
    settings = load_settings("test")
    return settings.collect


def _make_pump(
    broadcaster: EventBroadcaster,
    window_size: int = 5,
    fallback_fn: object = None,
    archive_to_gcs: bool = False,
    raw_ticks_dir: object = None,
) -> LivePump:
    """Build a LivePump wired to a single test channel, no Lightstreamer."""
    loop = asyncio.get_event_loop()
    engine = _make_engine(window_size)
    norm_params = {_CH: {"mean": _MEAN, "std": _STD}}
    config = _make_collect_config()

    pump = LivePump(
        loop=loop,
        broadcaster=broadcaster,
        engines={_CH: engine},
        norm_params=norm_params,
        collect_config=config,
        state=None,
        archive_to_gcs=archive_to_gcs,
        raw_ticks_dir=raw_ticks_dir,  # type: ignore[arg-type]
        los_stats_median_s=240.0,
        _fallback_start_fn=fallback_fn,  # type: ignore[arg-type]
    )
    # Override window deque size to match the engine's window_size.
    from collections import deque

    pump._recent_buckets[_CH] = deque(maxlen=window_size)
    return pump


def _published_event_types(payloads: list[bytes]) -> list[str]:
    """Extract event type names from SSE payloads."""
    types: list[str] = []
    for p in payloads:
        for line in p.decode().splitlines():
            if line.startswith("event:"):
                types.append(line.split(":", 1)[1].strip())
    return types


# ---------------------------------------------------------------------------
# Spy broadcaster
# ---------------------------------------------------------------------------


class _SpyBroadcaster(EventBroadcaster):
    """EventBroadcaster that records published payloads and status events."""

    def __init__(self) -> None:
        super().__init__()
        self.published: list[tuple[str, bytes]] = []
        self.status_calls: list[dict[str, object]] = []

    def publish(self, channel: str, payload: bytes) -> None:
        self.published.append((channel, payload))
        super().publish(channel, payload)

    def publish_status(
        self,
        event_type: str,
        *,
        mode: str | None = None,
        expected_resume_in_s: float | None = None,
    ) -> None:
        self.status_calls.append(
            {
                "type": event_type,
                "mode": mode,
                "expected_resume_in_s": expected_resume_in_s,
            }
        )
        super().publish_status(event_type, mode=mode, expected_resume_in_s=expected_resume_in_s)


# ---------------------------------------------------------------------------
# event: raw tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_tick_publishes_raw_event() -> None:
    """Each raw tick produces an event: raw payload for the channel."""
    spy = _SpyBroadcaster()
    pump = _make_pump(spy)

    await pump._on_tick(_CH, _BASE_TS, 23.7)

    raw_payloads = [p for ch, p in spy.published if ch == _CH]
    assert raw_payloads, "expected at least one published payload"
    raw_types = _published_event_types(raw_payloads)
    assert "raw" in raw_types


@pytest.mark.asyncio
async def test_raw_event_value_is_normalized() -> None:
    """event: raw value_normalized equals (raw - mean) / std."""
    spy = _SpyBroadcaster()
    pump = _make_pump(spy)

    raw_physical = 23.7
    expected_normalized = (raw_physical - _MEAN) / _STD

    await pump._on_tick(_CH, _BASE_TS, raw_physical)

    raw_payloads = [p for ch, p in spy.published if ch == _CH]
    raw_events = [
        json.loads(line.split("data:", 1)[1].strip())
        for p in raw_payloads
        for line in p.decode().splitlines()
        if line.startswith("data:") and "value_normalized" in line
    ]
    assert raw_events, "no raw event data found"
    assert raw_events[0]["value_normalized"] == pytest.approx(expected_normalized, rel=1e-5)


@pytest.mark.asyncio
async def test_non_served_channel_no_raw_event() -> None:
    """Ticks from non-served channels are archived but do not emit raw events."""
    spy = _SpyBroadcaster()
    pump = _make_pump(spy)
    # Populate archive buffer for the unknown channel
    pump._archive_buffers["UNKNOWN"] = []

    await pump._on_tick("UNKNOWN", _BASE_TS, 1.0)

    raw_payloads = [p for ch, p in spy.published if ch == "UNKNOWN"]
    assert raw_payloads == [], "non-served channel must not publish raw events"


# ---------------------------------------------------------------------------
# event: telemetry tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bucket_close_publishes_telemetry_event() -> None:
    """A telemetry event is published when a 30-second bucket closes."""
    spy = _SpyBroadcaster()
    pump = _make_pump(spy)

    # Two ticks in different buckets — the first bucket closes when the second arrives.
    await pump._on_tick(_CH, _BASE_TS + timedelta(seconds=5), 20.0)
    await pump._on_tick(_CH, _BASE_TS + timedelta(seconds=35), 22.0)

    published_types = _published_event_types([p for _, p in spy.published])
    assert "telemetry" in published_types, "expected a telemetry event after bucket close"


# ---------------------------------------------------------------------------
# LOS state tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_los_onset_emits_status_event() -> None:
    """_on_los_onset publishes a status event with type='los' and mode='replay'."""
    spy = _SpyBroadcaster()
    pump = _make_pump(spy)

    await pump._on_los_onset()

    assert any(s["type"] == "los" and s["mode"] == "replay" for s in spy.status_calls), (
        f"expected los status event, got {spy.status_calls}"
    )


@pytest.mark.asyncio
async def test_los_onset_includes_eta() -> None:
    """LOS status event includes expected_resume_in_s when provided."""
    spy = _SpyBroadcaster()
    pump = _make_pump(spy)

    await pump._on_los_onset()

    los_event = next(s for s in spy.status_calls if s["type"] == "los")
    assert los_event["expected_resume_in_s"] == pytest.approx(240.0)


@pytest.mark.asyncio
async def test_los_onset_starts_fallback_task() -> None:
    """_on_los_onset starts the fallback coroutine as an asyncio Task."""
    spy = _SpyBroadcaster()
    fallback_called = False

    async def _fake_fallback() -> None:
        nonlocal fallback_called
        fallback_called = True
        await asyncio.sleep(0)

    pump = _make_pump(spy, fallback_fn=_fake_fallback)
    await pump._on_los_onset()

    # Let the event loop run the fallback task.
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert fallback_called, "fallback coroutine was not started"


@pytest.mark.asyncio
async def test_los_onset_suppresses_raw_events() -> None:
    """After LOS onset, _on_tick does not publish raw events for served channels."""
    spy = _SpyBroadcaster()
    pump = _make_pump(spy)

    await pump._on_los_onset()
    spy.published.clear()  # clear any status payloads

    await pump._on_tick(_CH, _BASE_TS, 22.0)

    assert spy.published == [], "raw events must not be emitted during LOS"


@pytest.mark.asyncio
async def test_los_recovery_emits_resumed_status() -> None:
    """_on_los_recovery publishes status event with type='resumed'."""
    spy = _SpyBroadcaster()
    pump = _make_pump(spy)
    pump._in_los = True

    await pump._on_los_recovery()

    assert any(s["type"] == "resumed" for s in spy.status_calls), (
        f"expected resumed status event, got {spy.status_calls}"
    )


@pytest.mark.asyncio
async def test_los_recovery_cancels_replay_task() -> None:
    """_on_los_recovery cancels the active replay task."""
    spy = _SpyBroadcaster()
    pump = _make_pump(spy)

    # Create a long-running fake replay task.
    async def _long_running() -> None:
        await asyncio.sleep(3600)

    pump._replay_task = asyncio.create_task(_long_running())
    pump._in_los = True

    await pump._on_los_recovery()

    assert pump._replay_task is None, "replay task reference should be cleared"


@pytest.mark.asyncio
async def test_los_recovery_reprimes_engine() -> None:
    """_on_los_recovery calls engine.prime_with_scoring() from recent_buckets."""
    spy = _SpyBroadcaster()
    W = 5
    pump = _make_pump(spy, window_size=W)
    pump._in_los = True

    # Populate recent_buckets with W values so prime_with_scoring() gets a
    # full input window (scoring warmup is partial — only W values — but that's
    # the best available after a short AOS window; prime_with_scoring handles it).
    seed = [float(i) for i in range(W)]
    pump._recent_buckets[_CH].extend(seed)

    # Spy on prime_with_scoring().
    primed_values: list[list[float]] = []
    original_prime_ws = pump._engines[_CH].prime_with_scoring

    def _spy_prime_ws(values: list[float]) -> None:
        primed_values.append(values)
        original_prime_ws(values)

    pump._engines[_CH].prime_with_scoring = _spy_prime_ws  # type: ignore[method-assign]

    await pump._on_los_recovery()

    assert primed_values, "prime_with_scoring() was not called"
    assert primed_values[0] == seed


@pytest.mark.asyncio
async def test_los_recovery_resets_resamplers_no_backfill_flood() -> None:
    """Regression: LOS recovery must not backfill one telemetry event per
    grid bucket spanned by the outage.

    Before the resampler-reset fix, OnlineGridResampler._current_bucket stayed
    frozen at the last pre-LOS bucket for the whole outage (ticks return early
    in _on_tick during LOS, before reaching the resampler). The first
    post-recovery tick then landed far ahead of that stale bucket, and
    push()'s gap-fill loop emitted one backfilled telemetry event per grid
    step across the entire outage. A single post-recovery tick should never
    close a bucket by itself -- it can only ever open one.
    """
    spy = _SpyBroadcaster()
    pump = _make_pump(spy)

    # One pre-LOS tick opens (but does not close) a bucket, so the resampler's
    # _current_bucket is non-None going into LOS -- the exact precondition
    # for the flood bug.
    await pump._on_tick(_CH, _BASE_TS, 20.0)

    pump._in_los = True
    await pump._on_los_recovery()
    spy.published.clear()

    # One post-recovery tick, far ahead of the stale pre-LOS bucket (simulates
    # a long outage). With the resampler reset, this only opens a fresh
    # bucket -- it cannot close anything yet, so zero telemetry events fire.
    far_ts = _BASE_TS + timedelta(hours=1)
    await pump._on_tick(_CH, far_ts, 21.0)

    telemetry_events = [
        t for t in _published_event_types([p for _, p in spy.published]) if t == "telemetry"
    ]
    assert telemetry_events == [], (
        f"expected no telemetry events from a single post-recovery tick that can "
        f"only open a fresh bucket, got {len(telemetry_events)} -- resampler was "
        "not reset on recovery, so the stale bucket closed and backfilled the gap"
    )


@pytest.mark.asyncio
async def test_los_onset_uses_default_fallback_when_state_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With state set and no injected fallback, _on_los_onset starts the
    real run_shared_loop() default.

    This documents the A1 decision: LOS triggers a transparent replay
    fallback (labeled via mode="replay" in the status event), not silence.
    All other LOS tests in this file inject a fake _fallback_start_fn; this
    one exercises the actual production wiring (state=app_state in app.py).
    """
    import spacecraft_telemetry.api.broadcast as broadcast_mod

    called_with: list[object] = []

    async def _fake_run_shared_loop(state: object) -> None:
        called_with.append(state)
        await asyncio.sleep(3600)

    monkeypatch.setattr(broadcast_mod, "run_shared_loop", _fake_run_shared_loop)

    spy = _SpyBroadcaster()
    sentinel_state = object()
    pump = LivePump(
        loop=asyncio.get_event_loop(),
        broadcaster=spy,
        engines={_CH: _make_engine()},
        norm_params={_CH: {"mean": _MEAN, "std": _STD}},
        collect_config=_make_collect_config(),
        state=sentinel_state,  # type: ignore[arg-type]
        archive_to_gcs=False,
        raw_ticks_dir=None,
        los_stats_median_s=240.0,
        # No _fallback_start_fn — exercise the real default wiring.
    )

    await pump._on_los_onset()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert called_with == [sentinel_state], (
        "expected the default fallback to call run_shared_loop(state)"
    )


# ---------------------------------------------------------------------------
# Fault injection choreography
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_injection_modifies_raw_event_value() -> None:
    """With an active spike injection, the raw event value is modified."""
    spy = _SpyBroadcaster()
    pump = _make_pump(spy)

    # Queue a spike injection via the broadcaster.
    spy.request_injection(
        fault_type="spike",
        channels=frozenset(),  # all channels
        magnitude_sigma=3.0,
        total_ticks=5,
    )

    raw_physical = 20.5  # == mean → normalized = 0.0
    await pump._on_tick(_CH, _BASE_TS, raw_physical)

    raw_payloads = [p for ch, p in spy.published if ch == _CH]
    raw_data = [
        json.loads(line.split("data:", 1)[1].strip())
        for p in raw_payloads
        for line in p.decode().splitlines()
        if line.startswith("data:") and "value_normalized" in line
    ]
    assert raw_data, "no raw event data"
    # Spike adds +-3 sigma; nominal normalized value is 0.0, so result should be +-3.0
    assert abs(raw_data[0]["value_normalized"]) == pytest.approx(3.0, abs=1e-5)


@pytest.mark.asyncio
async def test_injected_bucket_sets_is_anomaly_flag() -> None:
    """When injection is active, telemetry event is_anomaly=True after bucket close."""
    spy = _SpyBroadcaster()
    W = 5
    pump = _make_pump(spy, window_size=W)

    # Arm a spike injection before any ticks.
    spy.request_injection(
        fault_type="spike",
        channels=frozenset(),
        magnitude_sigma=3.0,
        total_ticks=50,
    )

    ts0 = _BASE_TS
    # Fill first bucket with injected ticks.
    for i in range(3):
        await pump._on_tick(_CH, ts0 + timedelta(seconds=i * 5), 20.5)

    # Move to next bucket to close the first.
    await pump._on_tick(_CH, ts0 + timedelta(seconds=35), 20.5)

    telemetry_payloads = [
        json.loads(line.split("data:", 1)[1].strip())
        for _, p in spy.published
        for line in p.decode().splitlines()
        if line.startswith("data:") and "is_anomaly" in line
    ]
    assert any(t["is_anomaly"] for t in telemetry_payloads), (
        "expected at least one telemetry event with is_anomaly=True"
    )


# ---------------------------------------------------------------------------
# Archive flush tests (thread-safety split: drain on loop, write off-loop)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flush_archive_writes_and_clears_buffer(tmp_path: object) -> None:
    """_flush_archive_once drains a populated buffer and writes it via flush_buffer."""
    spy = _SpyBroadcaster()
    pump = _make_pump(spy, archive_to_gcs=True, raw_ticks_dir=tmp_path)
    pump._archive_buffers[_CH] = []  # normally populated by start() from subscription_items

    await pump._on_tick(_CH, _BASE_TS, 20.5)
    assert pump._archive_buffers[_CH], "tick should have been archived in-memory"

    await pump._flush_archive_once()

    assert pump._archive_buffers[_CH] == [], "buffer should be drained after a successful flush"


@pytest.mark.asyncio
async def test_flush_archive_requeues_rows_on_write_failure(monkeypatch: object) -> None:
    """A failed Parquet write re-queues its rows instead of dropping them."""
    import spacecraft_telemetry.api.live.pump as pump_mod

    spy = _SpyBroadcaster()
    pump = _make_pump(spy, archive_to_gcs=True, raw_ticks_dir="gs://fake-bucket")
    pump._archive_buffers[_CH] = []  # normally populated by start() from subscription_items

    await pump._on_tick(_CH, _BASE_TS, 20.5)
    assert pump._archive_buffers[_CH]

    def _boom(rows: object, dest_dir: object, channel_id: object, bucket_ts: object) -> None:
        raise OSError("GCS unavailable")

    monkeypatch.setattr(pump_mod, "flush_buffer", _boom)  # type: ignore[attr-defined]

    await pump._flush_archive_once()

    assert pump._archive_buffers[_CH], "rows must be re-queued, not dropped, on write failure"


@pytest.mark.asyncio
async def test_flush_archive_drain_runs_on_loop_thread() -> None:
    """_drain_archive and _requeue_failed never run inside asyncio.to_thread.

    Regression guard for the thread-safety fix: only _write_archive (pure I/O)
    may run off the event loop. Verified by asserting drain leaves no window
    where a concurrently-appended tick could be silently dropped: appending a
    second tick for a different bucket right after drain (simulating a tick
    arriving between drain and write) must still be picked up by the next flush.
    """
    spy = _SpyBroadcaster()
    pump = _make_pump(spy, archive_to_gcs=True, raw_ticks_dir="gs://fake-bucket")
    pump._archive_buffers[_CH] = []  # normally populated by start() from subscription_items

    await pump._on_tick(_CH, _BASE_TS, 20.5)
    drained = pump._drain_archive()
    assert drained[_CH], "drain should have captured the first tick"
    assert pump._archive_buffers[_CH] == [], "drain must clear the buffer immediately"

    # A tick arriving after drain (simulating the write-in-flight window) lands
    # in the now-empty buffer rather than being lost or mixed into `drained`.
    await pump._on_tick(_CH, _BASE_TS + timedelta(seconds=5), 21.0)
    assert pump._archive_buffers[_CH], "post-drain tick must still be archived"
    assert len(drained[_CH]) == 1, "the snapshot taken by drain must not grow after the fact"
