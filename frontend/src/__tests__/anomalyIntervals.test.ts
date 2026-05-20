import { describe, it, expect } from "vitest";
import { collapseFlags } from "../utils/anomalyIntervals";
import type { TelemetryEvent } from "../api/types";

function event(
  ts: string,
  is_anomaly: boolean,
  is_anomaly_predicted = false,
): TelemetryEvent {
  return {
    timestamp: ts,
    mission: "test",
    channel: "ch",
    value_normalized: 0,
    prediction: null,
    residual: null,
    smoothed_error: null,
    threshold: null,
    is_anomaly,
    is_anomaly_predicted,
  };
}

describe("collapseFlags — is_anomaly", () => {
  it("returns empty for empty input", () => {
    expect(collapseFlags([], "is_anomaly")).toEqual([]);
  });

  it("returns empty when all flags are false", () => {
    const events = ["t1", "t2", "t3"].map((ts) => event(ts, false));
    expect(collapseFlags(events, "is_anomaly")).toEqual([]);
  });

  it("returns single interval spanning full range when all true", () => {
    const events = ["t1", "t2", "t3"].map((ts) => event(ts, true));
    expect(collapseFlags(events, "is_anomaly")).toEqual([
      { startTs: "t1", endTs: "t3" },
    ]);
  });

  it("collapses three disjoint clusters into three intervals", () => {
    const events = [
      event("t1", false),
      event("t2", true),
      event("t3", true),
      event("t4", false),
      event("t5", true),
      event("t6", false),
      event("t7", true),
      event("t8", true),
      event("t9", true),
    ];
    expect(collapseFlags(events, "is_anomaly")).toEqual([
      { startTs: "t2", endTs: "t3" },
      { startTs: "t5", endTs: "t5" },
      { startTs: "t7", endTs: "t9" },
    ]);
  });

  it("closes an open interval at the end of the buffer", () => {
    const events = [event("t1", false), event("t2", true), event("t3", true)];
    expect(collapseFlags(events, "is_anomaly")).toEqual([
      { startTs: "t2", endTs: "t3" },
    ]);
  });
});

describe("collapseFlags — is_anomaly_predicted", () => {
  it("operates on the predicted flag independently", () => {
    const events = [
      event("t1", false, true),
      event("t2", false, true),
      event("t3", false, false),
    ];
    expect(collapseFlags(events, "is_anomaly_predicted")).toEqual([
      { startTs: "t1", endTs: "t2" },
    ]);
  });
});
