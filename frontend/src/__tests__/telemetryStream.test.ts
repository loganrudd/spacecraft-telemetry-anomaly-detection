import { describe, it, expect, beforeEach, vi } from "vitest";
import { openTelemetryStream } from "../api/telemetryStream";
import type { TelemetryEvent } from "../api/types";

// The global EventSource is the MockEventSource installed by setup.ts.
// Cast it to access the static lastInstance tracking field.
interface MockESInstance {
  url: string;
  readyState: number;
  onopen: ((e: Event) => void) | null;
  onerror: ((e: Event) => void) | null;
  dispatchEvent(e: Event): boolean;
}
interface MockESConstructor {
  lastInstance: MockESInstance | null;
  CLOSED: number;
  CONNECTING: number;
}
const MockES = EventSource as unknown as MockESConstructor;

function makeTelemetryEvent(channel = "ch-a"): TelemetryEvent {
  return {
    timestamp: "2000-01-01T00:00:00Z",
    mission: "test",
    channel,
    value_normalized: 0.5,
    prediction: 0.4,
    residual: 0.1,
    smoothed_error: 0.2,
    threshold: 0.5,
    is_anomaly_predicted: false,
    is_anomaly: false,
  };
}

describe("openTelemetryStream", () => {
  beforeEach(() => {
    MockES.lastInstance = null;
  });

  it("builds URL with channels joined by comma and speed param", () => {
    openTelemetryStream({
      channels: ["ch-a", "ch-b"],
      speed: 100,
      onEvent: () => {},
    });
    const url = new URL(MockES.lastInstance!.url, "http://localhost");
    expect(url.searchParams.get("channels")).toBe("ch-a,ch-b");
    expect(url.searchParams.get("speed")).toBe("100");
  });

  it("omits channels param when channel list is empty", () => {
    openTelemetryStream({ channels: [], onEvent: () => {} });
    const url = new URL(MockES.lastInstance!.url, "http://localhost");
    expect(url.searchParams.has("channels")).toBe(false);
  });

  it("wires onOpen to EventSource.onopen so it fires on connection open", () => {
    const onOpen = vi.fn();
    openTelemetryStream({ channels: ["ch-a"], onEvent: () => {}, onOpen });
    const instance = MockES.lastInstance!;
    // Simulate the browser firing the open event
    instance.onopen?.(new Event("open"));
    expect(onOpen).toHaveBeenCalledOnce();
  });

  it("parses JSON and passes a TelemetryEvent to onEvent", () => {
    const onEvent = vi.fn();
    openTelemetryStream({ channels: ["ch-a"], onEvent });
    const payload = makeTelemetryEvent("ch-a");
    MockES.lastInstance!.dispatchEvent(
      new MessageEvent("telemetry", { data: JSON.stringify(payload) }),
    );
    expect(onEvent).toHaveBeenCalledOnce();
    expect(onEvent).toHaveBeenCalledWith(payload);
  });

  it("throws on malformed JSON in a telemetry event", () => {
    openTelemetryStream({ channels: ["ch-a"], onEvent: () => {} });
    expect(() => {
      MockES.lastInstance!.dispatchEvent(
        new MessageEvent("telemetry", { data: "not-json" }),
      );
    }).toThrow(SyntaxError);
  });

  it("closes the underlying EventSource when handle.close() is called", () => {
    const handle = openTelemetryStream({ channels: ["ch-a"], onEvent: () => {} });
    const instance = MockES.lastInstance!;
    expect(instance.readyState).toBe(MockES.CONNECTING);
    handle.close();
    expect(instance.readyState).toBe(MockES.CLOSED);
  });
});
