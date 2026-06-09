import { describe, it, expect, vi, beforeEach } from "vitest";
import { telemetryStore } from "../state/telemetryStore";
import type { TelemetryEvent } from "../api/types";

function makeEvent(channel: string, idx: number): TelemetryEvent {
  return {
    timestamp: `2000-01-01T00:${String(idx).padStart(2, "0")}:00Z`,
    mission: "test",
    channel,
    value_normalized: idx * 0.1,
    prediction: null,
    residual: null,
    smoothed_error: null,
    threshold: null,
    is_anomaly_predicted: false,
    is_anomaly: false,
  };
}

describe("TelemetryStore", () => {
  beforeEach(() => {
    telemetryStore.clear();
  });

  it("buffers up to 600 events per channel, evicting oldest", () => {
    for (let i = 0; i < 700; i++) {
      telemetryStore.push(makeEvent("ch-a", i));
    }
    telemetryStore.flushForTest();
    const buf = telemetryStore.snapshot("ch-a");
    expect(buf).toHaveLength(600);
    // Oldest 100 evicted — first remaining event is index 100
    expect(buf[0].value_normalized).toBeCloseTo(100 * 0.1);
  });

  it("keeps per-channel buffers isolated", () => {
    for (let i = 0; i < 5; i++) telemetryStore.push(makeEvent("ch-a", i));
    for (let i = 0; i < 3; i++) telemetryStore.push(makeEvent("ch-b", i));
    telemetryStore.flushForTest();

    expect(telemetryStore.snapshot("ch-a")).toHaveLength(5);
    expect(telemetryStore.snapshot("ch-b")).toHaveLength(3);
  });

  it("returns empty array for unknown channel", () => {
    expect(telemetryStore.snapshot("no-such-channel")).toEqual([]);
  });

  it("notifies subscribers once per rAF flush, not once per push", () => {
    const listener = vi.fn();
    const unsub = telemetryStore.subscribe(listener);

    telemetryStore.push(makeEvent("ch-a", 0));
    telemetryStore.push(makeEvent("ch-a", 1));
    // Both pushes batched into one rAF — one notify call.
    telemetryStore.flushForTest();
    expect(listener).toHaveBeenCalledTimes(1);

    unsub();
    telemetryStore.push(makeEvent("ch-a", 2));
    telemetryStore.flushForTest();
    // After unsubscribe, listener should not be called again
    expect(listener).toHaveBeenCalledTimes(1);
  });

  it("clear empties all buffers and notifies", () => {
    telemetryStore.push(makeEvent("ch-a", 0));
    const listener = vi.fn();
    const unsub = telemetryStore.subscribe(listener);

    telemetryStore.clear();

    expect(telemetryStore.snapshot("ch-a")).toEqual([]);
    expect(listener).toHaveBeenCalledTimes(1);
    unsub();
  });

  it("treats a backwards timestamp as a replay-loop wrap and clears all channels", () => {
    // One replay pass across two channels.
    for (let i = 0; i < 10; i++) telemetryStore.push(makeEvent("ch-a", i));
    for (let i = 0; i < 10; i++) telemetryStore.push(makeEvent("ch-b", i));
    telemetryStore.flushForTest();
    expect(telemetryStore.snapshot("ch-a")).toHaveLength(10);
    expect(telemetryStore.snapshot("ch-b")).toHaveLength(10);

    // The shared loop wraps: ch-a's timestamp jumps back to the slice start.
    // The whole store clears so the chart restarts cleanly instead of mixing the
    // tail of the old pass with the head of the new one. (The SSE connection
    // stays open across the wrap, so onOpen can't drive this.)
    telemetryStore.push(makeEvent("ch-a", 0));
    telemetryStore.flushForTest();

    expect(telemetryStore.snapshot("ch-a")).toHaveLength(1); // repopulated from i=0
    expect(telemetryStore.snapshot("ch-b")).toHaveLength(0); // sibling cleared too
  });
});

describe("TelemetryStore — EMPTY sentinel", () => {
  it("snapshot returns the same reference for unknown channels", () => {
    // useSyncExternalStore compares snapshots by reference equality.
    // If snapshot() returns a new [] literal each call, React enters an
    // infinite re-render loop for channels with no data. This test pins
    // that invariant so a well-meaning `?? []` simplification is caught.
    expect(telemetryStore.snapshot("no-such-channel")).toBe(
      telemetryStore.snapshot("no-such-channel"),
    );
  });
});
