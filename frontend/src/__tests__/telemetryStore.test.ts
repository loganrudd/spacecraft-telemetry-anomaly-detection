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
    const buf = telemetryStore.snapshot("ch-a");
    expect(buf).toHaveLength(600);
    // Oldest 100 evicted — first remaining event is index 100
    expect(buf[0].value_normalized).toBeCloseTo(100 * 0.1);
  });

  it("keeps per-channel buffers isolated", () => {
    for (let i = 0; i < 5; i++) telemetryStore.push(makeEvent("ch-a", i));
    for (let i = 0; i < 3; i++) telemetryStore.push(makeEvent("ch-b", i));

    expect(telemetryStore.snapshot("ch-a")).toHaveLength(5);
    expect(telemetryStore.snapshot("ch-b")).toHaveLength(3);
  });

  it("returns empty array for unknown channel", () => {
    expect(telemetryStore.snapshot("no-such-channel")).toEqual([]);
  });

  it("notifies subscribers on every push", () => {
    const listener = vi.fn();
    const unsub = telemetryStore.subscribe(listener);

    telemetryStore.push(makeEvent("ch-a", 0));
    telemetryStore.push(makeEvent("ch-a", 1));

    expect(listener).toHaveBeenCalledTimes(2);
    unsub();

    telemetryStore.push(makeEvent("ch-a", 2));
    // After unsubscribe, listener should not be called again
    expect(listener).toHaveBeenCalledTimes(2);
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
});

describe("openTelemetryStream", () => {
  it("is a callable function", async () => {
    const mod = await import("../api/telemetryStream");
    expect(typeof mod.openTelemetryStream).toBe("function");
  });
});
