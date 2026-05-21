import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useSubsystemRollup } from "../hooks/useSubsystemRollup";
import { telemetryStore } from "../state/telemetryStore";
import { driftStore } from "../state/driftStore";
import type { TelemetryEvent, DriftEvent } from "../api/types";

function makeTelemetryEvent(
  channel: string,
  is_anomaly_predicted = false,
): TelemetryEvent {
  return {
    timestamp: new Date().toISOString(),
    mission: "test",
    channel,
    value_normalized: 0.5,
    prediction: null,
    residual: null,
    smoothed_error: null,
    threshold: null,
    is_anomaly_predicted,
    is_anomaly: false,
  };
}

function makeDriftEvent(channel: string, drifted: boolean): DriftEvent {
  return {
    timestamp: new Date().toISOString(),
    mission: "test",
    channel,
    features: [],
    percent_drifted: drifted ? 0.5 : 0.1,
    drifted,
    subsystem_percent_drifted: null,
    subsystem_alert: false,
  };
}

const CHANNEL_SUBSYSTEMS: Record<string, string> = {
  channel_1: "subsystem_1",
  channel_2: "subsystem_1",
  channel_3: "subsystem_2",
};

describe("useSubsystemRollup", () => {
  beforeEach(() => {
    telemetryStore.clear();
    driftStore.clear();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("returns a rollup covering all subsystems and channels", () => {
    const { result } = renderHook(() =>
      useSubsystemRollup(CHANNEL_SUBSYSTEMS),
    );
    expect(result.current.has("subsystem_1")).toBe(true);
    expect(result.current.has("subsystem_2")).toBe(true);
    expect(result.current.get("subsystem_1")?.has("channel_1")).toBe(true);
    expect(result.current.get("subsystem_1")?.has("channel_2")).toBe(true);
    expect(result.current.get("subsystem_2")?.has("channel_3")).toBe(true);
  });

  it("channels start with anomaly=false and drifted=false", () => {
    const { result } = renderHook(() =>
      useSubsystemRollup(CHANNEL_SUBSYSTEMS),
    );
    const status = result.current.get("subsystem_1")?.get("channel_1");
    expect(status?.anomaly).toBe(false);
    expect(status?.drifted).toBe(false);
    expect(status?.lastUpdated).toBeNull();
  });

  it("sets anomaly=true immediately when telemetryStore receives an anomaly event", () => {
    const { result } = renderHook(() =>
      useSubsystemRollup(CHANNEL_SUBSYSTEMS),
    );

    act(() => {
      telemetryStore.push(makeTelemetryEvent("channel_1", true));
      telemetryStore.flushForTest();
      vi.advanceTimersByTime(250); // drain trailing-edge throttle
    });

    expect(
      result.current.get("subsystem_1")?.get("channel_1")?.anomaly,
    ).toBe(true);
  });

  it("normal event does not set anomaly", () => {
    const { result } = renderHook(() =>
      useSubsystemRollup(CHANNEL_SUBSYSTEMS),
    );

    act(() => {
      telemetryStore.push(makeTelemetryEvent("channel_1", false));
      telemetryStore.flushForTest();
    });

    expect(
      result.current.get("subsystem_1")?.get("channel_1")?.anomaly,
    ).toBe(false);
  });

  it("anomaly badge is still active at 59 s", () => {
    const { result } = renderHook(() =>
      useSubsystemRollup(CHANNEL_SUBSYSTEMS),
    );

    act(() => {
      telemetryStore.push(makeTelemetryEvent("channel_1", true));
      telemetryStore.flushForTest();
    });

    act(() => {
      vi.advanceTimersByTime(59_000);
    });

    expect(
      result.current.get("subsystem_1")?.get("channel_1")?.anomaly,
    ).toBe(true);
  });

  it("anomaly badge clears after the 60 s TTL", () => {
    const { result } = renderHook(() =>
      useSubsystemRollup(CHANNEL_SUBSYSTEMS),
    );

    act(() => {
      telemetryStore.push(makeTelemetryEvent("channel_1", true));
      telemetryStore.flushForTest();
    });

    act(() => {
      vi.advanceTimersByTime(61_000);
    });

    expect(
      result.current.get("subsystem_1")?.get("channel_1")?.anomaly,
    ).toBe(false);
  });

  it("sets drifted=true when driftStore has a drifted event", () => {
    const { result } = renderHook(() =>
      useSubsystemRollup(CHANNEL_SUBSYSTEMS),
    );

    act(() => {
      driftStore.push(makeDriftEvent("channel_3", true));
      driftStore.flushForTest();
      vi.advanceTimersByTime(250); // drain trailing-edge throttle
    });

    expect(
      result.current.get("subsystem_2")?.get("channel_3")?.drifted,
    ).toBe(true);
  });

  it("drifted=false when the latest drift event is not drifted", () => {
    const { result } = renderHook(() =>
      useSubsystemRollup(CHANNEL_SUBSYSTEMS),
    );

    act(() => {
      driftStore.push(makeDriftEvent("channel_3", true));
      driftStore.push(makeDriftEvent("channel_3", false));
      driftStore.flushForTest();
    });

    expect(
      result.current.get("subsystem_2")?.get("channel_3")?.drifted,
    ).toBe(false);
  });

  it("anomaly takes priority: channel with both anomaly and drift reports anomaly", () => {
    const { result } = renderHook(() =>
      useSubsystemRollup(CHANNEL_SUBSYSTEMS),
    );

    act(() => {
      telemetryStore.push(makeTelemetryEvent("channel_1", true));
      telemetryStore.flushForTest();
      driftStore.push(makeDriftEvent("channel_1", true));
      driftStore.flushForTest();
      vi.advanceTimersByTime(250); // drain trailing-edge throttle
    });

    const status = result.current.get("subsystem_1")?.get("channel_1");
    expect(status?.anomaly).toBe(true);
    expect(status?.drifted).toBe(true);
  });

  it("lastUpdated is populated after a telemetry event", () => {
    const { result } = renderHook(() =>
      useSubsystemRollup(CHANNEL_SUBSYSTEMS),
    );

    act(() => {
      telemetryStore.push(makeTelemetryEvent("channel_2", false));
      telemetryStore.flushForTest();
      vi.advanceTimersByTime(250); // drain trailing-edge throttle
    });

    expect(
      result.current.get("subsystem_1")?.get("channel_2")?.lastUpdated,
    ).not.toBeNull();
  });

  it("channels from different subsystems are isolated", () => {
    const { result } = renderHook(() =>
      useSubsystemRollup(CHANNEL_SUBSYSTEMS),
    );

    act(() => {
      telemetryStore.push(makeTelemetryEvent("channel_1", true));
      telemetryStore.flushForTest();
    });

    expect(
      result.current.get("subsystem_2")?.get("channel_3")?.anomaly,
    ).toBe(false);
  });
});
