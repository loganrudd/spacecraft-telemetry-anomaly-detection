import { describe, it, expect, beforeEach } from "vitest";
import { act } from "@testing-library/react";
import { driftStore } from "../state/driftStore";
import type { DriftEvent } from "../api/types";

function makeEvent(channel: string, pct: number, alerted = false): DriftEvent {
  return {
    timestamp: "2000-01-01T00:00:00Z",
    mission: "test",
    channel,
    features: [{ feature: "value_normalized", score: pct, drifted: pct > 0.1 }],
    percent_drifted: pct,
    drifted: pct >= 0.3,
    subsystem_percent_drifted: alerted ? pct : null,
    subsystem_alert: alerted,
  };
}

function push(event: DriftEvent) {
  act(() => {
    driftStore.push(event);
    driftStore.flushForTest();
  });
}

describe("DriftStore", () => {
  beforeEach(() => {
    driftStore.clear();
  });

  it("returns null for unknown channel", () => {
    expect(driftStore.latestForChannel("no-such-channel")).toBeNull();
  });

  it("stores latest event per channel, overwriting previous", () => {
    push(makeEvent("ch-a", 0.1));
    push(makeEvent("ch-a", 0.5));
    expect(driftStore.latestForChannel("ch-a")?.percent_drifted).toBeCloseTo(0.5);
  });

  it("keeps per-channel entries isolated", () => {
    push(makeEvent("ch-a", 0.1));
    push(makeEvent("ch-b", 0.9));
    expect(driftStore.latestForChannel("ch-a")?.percent_drifted).toBeCloseTo(0.1);
    expect(driftStore.latestForChannel("ch-b")?.percent_drifted).toBeCloseTo(0.9);
  });

  it("notifies subscribers on flush", () => {
    let callCount = 0;
    const unsub = driftStore.subscribe(() => {
      callCount++;
    });
    push(makeEvent("ch-a", 0.2));
    expect(callCount).toBeGreaterThanOrEqual(1);
    unsub();
  });

  it("updates subsystem fields only when event carries them", () => {
    push(makeEvent("ch-a", 0.2, false)); // no subsystem data
    expect(driftStore.subsystemPercent()).toBeNull();
    expect(driftStore.isSubsystemAlert()).toBe(false);

    push(makeEvent("ch-a", 0.4, true)); // carries subsystem data
    expect(driftStore.subsystemPercent()).toBeCloseTo(0.4);
    expect(driftStore.isSubsystemAlert()).toBe(true);
  });

  it("clear resets all state", () => {
    push(makeEvent("ch-a", 0.5, true));
    act(() => {
      driftStore.clear();
    });
    expect(driftStore.latestForChannel("ch-a")).toBeNull();
    expect(driftStore.subsystemPercent()).toBeNull();
    expect(driftStore.isSubsystemAlert()).toBe(false);
  });

  it("latest wins when two events pushed before flush", () => {
    act(() => {
      driftStore.push(makeEvent("ch-a", 0.1));
      driftStore.push(makeEvent("ch-a", 0.8)); // should overwrite
      driftStore.flushForTest();
    });
    expect(driftStore.latestForChannel("ch-a")?.percent_drifted).toBeCloseTo(0.8);
  });
});
