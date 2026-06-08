import { describe, it, expect, vi } from "vitest";
import { TelemetryJitterBuffer } from "../state/jitterBuffer";
import type { TelemetryEvent } from "../api/types";

function makeEvent(channel: string, idx: number): TelemetryEvent {
  return {
    timestamp: `2000-01-01T00:${String(idx).padStart(2, "0")}:00Z`,
    mission: "test",
    channel,
    value_normalized: idx,
    prediction: null,
    residual: null,
    smoothed_error: null,
    threshold: null,
    is_anomaly_predicted: false,
    is_anomaly: false,
  };
}

/**
 * Test harness with a controllable clock. We feed events at chosen arrival
 * times via `now`, then advance the clock and drive `tick()` manually (autoRaf
 * off), recording the wall-clock time of each release.
 */
function makeHarness(bufferMs: number) {
  let clock = 0;
  const released: { event: TelemetryEvent; at: number }[] = [];
  const buf = new TelemetryJitterBuffer({
    bufferMs,
    autoRaf: false,
    now: () => clock,
    onRelease: (event) => released.push({ event, at: clock }),
  });
  /** Advance to `end` in small steps, ticking each step (mimics the rAF loop). */
  const runTo = (end: number, step = 5) => {
    while (clock < end) {
      clock = Math.min(end, clock + step);
      buf.tick(clock);
    }
  };
  return {
    buf,
    released,
    runTo,
    /** Tick the clock forward to `t` (as rAF would), then enqueue at `t`. */
    enqueueAt: (t: number, event: TelemetryEvent) => {
      runTo(t);
      clock = t;
      buf.enqueue(event);
    },
  };
}

describe("TelemetryJitterBuffer", () => {
  it("holds events for the lead buffer before releasing any", () => {
    const h = makeHarness(500);
    // Two events 50 ms apart establishes a 50 ms interval estimate.
    h.enqueueAt(0, makeEvent("a", 0));
    h.enqueueAt(50, makeEvent("a", 1));

    // Before the 500 ms prime deadline, nothing is released.
    h.runTo(400);
    expect(h.released).toHaveLength(0);
  });

  it("re-spaces a clump into a steady cadence", () => {
    const h = makeHarness(200);
    // Establish a ~50 ms cadence with a few evenly-spaced arrivals...
    for (let i = 0; i < 5; i++) h.enqueueAt(i * 50, makeEvent("a", i));
    // ...then a CLUMP: 5 events all land at once at t=250 (network burst).
    for (let i = 5; i < 10; i++) h.enqueueAt(250, makeEvent("a", i));

    // Play out well past the buffer.
    h.runTo(1200);

    // All 10 events come out, in order.
    expect(h.released.map((r) => r.event.value_normalized)).toEqual([
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    ]);

    // The clumped events (idx >= 5, all arrived at t=250) are released spread
    // out over time, not dumped together: consecutive release gaps stay > 0.
    const clumpReleases = h.released.filter((r) => r.event.value_normalized >= 5);
    for (let i = 1; i < clumpReleases.length; i++) {
      const gap = clumpReleases[i].at - clumpReleases[i - 1].at;
      expect(gap).toBeGreaterThan(10); // re-spaced, not bursted
    }
  });

  it("releases roughly one event per measured interval once flowing", () => {
    const h = makeHarness(200);
    const interval = 40;
    // Arrive evenly at 40 ms so the estimate locks to ~40 ms.
    for (let i = 0; i < 40; i++) h.enqueueAt(i * interval, makeEvent("a", i));

    h.runTo(40 * interval);

    // Released count should track elapsed playback time / interval, minus the
    // lead still held in the buffer — i.e. far fewer than a single-frame dump
    // and more than zero. Allow a generous band around the expected steady rate.
    const n = h.released.length;
    expect(n).toBeGreaterThan(20);
    expect(n).toBeLessThan(40);
  });

  it("keeps channels independent", () => {
    const h = makeHarness(100);
    for (let i = 0; i < 6; i++) h.enqueueAt(i * 50, makeEvent("a", i));
    for (let i = 0; i < 6; i++) h.enqueueAt(i * 50, makeEvent("b", i));

    h.runTo(800);

    const chA = h.released.filter((r) => r.event.channel === "a");
    const chB = h.released.filter((r) => r.event.channel === "b");
    expect(chA.length).toBeGreaterThan(0);
    expect(chB.length).toBeGreaterThan(0);
    // Order preserved within each channel.
    expect(chA.map((r) => r.event.value_normalized)).toEqual(
      [...chA].map((r) => r.event.value_normalized).sort((x, y) => x - y),
    );
  });

  it("reset() drops pending events and stops releasing", () => {
    const h = makeHarness(100);
    for (let i = 0; i < 6; i++) h.enqueueAt(i * 50, makeEvent("a", i));
    expect(h.buf.pendingCount()).toBeGreaterThan(0);

    h.buf.reset();
    expect(h.buf.pendingCount()).toBe(0);

    const before = h.released.length;
    h.runTo(2000);
    expect(h.released.length).toBe(before); // nothing more released
  });

  it("does not start a rAF loop when autoRaf is false", () => {
    const raf = vi.spyOn(globalThis, "requestAnimationFrame");
    const h = makeHarness(100);
    h.enqueueAt(0, makeEvent("a", 0));
    h.enqueueAt(50, makeEvent("a", 1));
    expect(raf).not.toHaveBeenCalled();
    raf.mockRestore();
  });
});
