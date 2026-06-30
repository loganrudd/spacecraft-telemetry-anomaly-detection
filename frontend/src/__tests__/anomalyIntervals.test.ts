import { describe, it, expect } from "vitest";
import {
  collapseFlags,
  labeledIntervalsWithDetection,
} from "../utils/anomalyIntervals";
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
    // Closed intervals extend endTs to the first non-flagged event for visible
    // chart width. The last cluster has no closing event so endTs stays at t9.
    expect(collapseFlags(events, "is_anomaly")).toEqual([
      { startTs: "t2", endTs: "t4" },
      { startTs: "t5", endTs: "t6" },
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
    // Interval closes at t3 (first non-flagged), extending endTs for visible width.
    expect(collapseFlags(events, "is_anomaly_predicted")).toEqual([
      { startTs: "t1", endTs: "t3" },
    ]);
  });
});

describe("labeledIntervalsWithDetection", () => {
  it("marks a labeled window detected when any point inside was flagged", () => {
    // Labeled t2..t4; model flagged only t3 (a smaller sub-window).
    const events = [
      event("t1", false, false),
      event("t2", true, false),
      event("t3", true, true),
      event("t4", true, false),
      event("t5", false, false),
    ];
    // Interval closes at t5 (first non-labeled event), extending endTs.
    expect(labeledIntervalsWithDetection(events)).toEqual([
      { startTs: "t2", endTs: "t5", detected: true },
    ]);
  });

  it("marks a labeled window missed when no point inside was flagged", () => {
    const events = [
      event("t1", true, false),
      event("t2", true, false),
    ];
    expect(labeledIntervalsWithDetection(events)).toEqual([
      { startTs: "t1", endTs: "t2", detected: false },
    ]);
  });

  it("classifies multiple windows independently (hit, miss)", () => {
    const events = [
      event("t1", true, true), // hit (single point)
      event("t2", false, false),
      event("t3", true, false), // miss window start
      event("t4", true, false),
    ];
    // First window closes at t2 (first non-labeled event), extending endTs.
    expect(labeledIntervalsWithDetection(events)).toEqual([
      { startTs: "t1", endTs: "t2", detected: true },
      { startTs: "t3", endTs: "t4", detected: false },
    ]);
  });

  it("ignores predicted flags outside labeled windows (false positives)", () => {
    const events = [
      event("t1", false, true), // false positive — no labeled window
      event("t2", false, false),
    ];
    expect(labeledIntervalsWithDetection(events)).toEqual([]);
  });
});
