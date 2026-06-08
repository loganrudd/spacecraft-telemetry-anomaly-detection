import { describe, it, expect, vi } from "vitest";
import { TelemetryJitterBuffer } from "../state/jitterBuffer";
import type { TelemetryEvent } from "../api/types";

const TICK_MS = 50; // matches server default (1.0s / 20x)

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

function makeHarness(bufferMs: number) {
  let clock = 0;
  const released: { event: TelemetryEvent; at: number }[] = [];
  const buf = new TelemetryJitterBuffer({
    bufferMs,
    autoRaf: false,
    now: () => clock,
    onRelease: (event) => released.push({ event, at: clock }),
  });
  buf.setTickIntervalMs(TICK_MS);

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
    for (let i = 0; i < 5; i++) h.enqueueAt(i * TICK_MS, makeEvent("a", i));

    // Before the 500 ms prime deadline, nothing is released.
    h.runTo(400);
    expect(h.released).toHaveLength(0);
  });

  it("re-spaces a clump into a steady cadence", () => {
    const h = makeHarness(200);
    // Establish normal arrivals...
    for (let i = 0; i < 5; i++) h.enqueueAt(i * TICK_MS, makeEvent("a", i));
    // ...then a CLUMP: 5 events all land at once (network burst).
    for (let i = 5; i < 10; i++) h.enqueueAt(250, makeEvent("a", i));

    h.runTo(1200);

    // All 10 events come out, in order.
    expect(h.released.map((r) => r.event.value_normalized)).toEqual([
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    ]);

    // The clumped events (idx >= 5, all arrived at t=250) are released spread
    // out over time — not dumped together.
    const clumpReleases = h.released.filter((r) => r.event.value_normalized >= 5);
    for (let i = 1; i < clumpReleases.length; i++) {
      const gap = clumpReleases[i].at - clumpReleases[i - 1].at;
      expect(gap).toBeGreaterThan(10); // re-spaced, not bursted
    }
  });

  it("releases at roughly the configured tick interval once flowing", () => {
    const h = makeHarness(200);
    for (let i = 0; i < 40; i++) h.enqueueAt(i * TICK_MS, makeEvent("a", i));

    h.runTo(40 * TICK_MS);

    // Events in the buffer lag by ~bufferMs, so released count < 40.
    // But well above 0 — the stream is flowing.
    const n = h.released.length;
    expect(n).toBeGreaterThan(15);
    expect(n).toBeLessThan(40);
  });

  it("is immune to a backlog burst (200 events at once)", () => {
    const h = makeHarness(500);
    // Simulate a 200-event backlog arriving at t=0 (as subscribe_with_backlog sends).
    for (let i = 0; i < 200; i++) h.enqueueAt(0, makeEvent("a", i));

    // Prime phase runs for 500ms. During it, some live events also arrive.
    for (let i = 200; i < 220; i++) h.enqueueAt(i * TICK_MS, makeEvent("a", i));

    // Play out past where bad interval estimation would cause a freeze.
    h.runTo(15_000);

    // All events released, no long freeze: the last release should be well
    // before the end of the run (i.e., the buffer didn't stall for 500ms
    // after the backlog drained).
    expect(h.released).toHaveLength(220);
    // Last batch of events (live arrivals at ~50ms each) should be spaced
    // at roughly TICK_MS, not in a sudden dump followed by a freeze.
    const liveReleases = h.released.slice(-10);
    for (let i = 1; i < liveReleases.length; i++) {
      const gap = liveReleases[i].at - liveReleases[i - 1].at;
      expect(gap).toBeGreaterThan(10);
      expect(gap).toBeLessThan(200);
    }
  });

  it("keeps channels independent", () => {
    const h = makeHarness(100);
    for (let i = 0; i < 6; i++) h.enqueueAt(i * TICK_MS, makeEvent("a", i));
    for (let i = 0; i < 6; i++) h.enqueueAt(i * TICK_MS, makeEvent("b", i));

    h.runTo(800);

    const chA = h.released.filter((r) => r.event.channel === "a");
    const chB = h.released.filter((r) => r.event.channel === "b");
    expect(chA.length).toBeGreaterThan(0);
    expect(chB.length).toBeGreaterThan(0);
    // Order preserved within each channel.
    const valuesA = chA.map((r) => r.event.value_normalized);
    expect(valuesA).toEqual([...valuesA].sort((x, y) => x - y));
  });

  it("tick() is a no-op before setTickIntervalMs is called", () => {
    let clock = 0;
    const released: TelemetryEvent[] = [];
    const buf = new TelemetryJitterBuffer({
      bufferMs: 100,
      autoRaf: false,
      now: () => clock,
      onRelease: (e) => released.push(e),
    });
    // No setTickIntervalMs call.
    buf.enqueue(makeEvent("a", 0));
    clock = 5000;
    buf.tick(clock);
    expect(released).toHaveLength(0);
  });

  it("reset() drops pending events and stops releasing", () => {
    const h = makeHarness(100);
    for (let i = 0; i < 6; i++) h.enqueueAt(i * TICK_MS, makeEvent("a", i));
    expect(h.buf.pendingCount()).toBeGreaterThan(0);

    h.buf.reset();
    expect(h.buf.pendingCount()).toBe(0);

    const before = h.released.length;
    h.runTo(2000);
    expect(h.released.length).toBe(before);
  });

  it("does not start a rAF loop when autoRaf is false", () => {
    const raf = vi.spyOn(globalThis, "requestAnimationFrame");
    const buf = new TelemetryJitterBuffer({
      bufferMs: 100,
      autoRaf: false,
      now: () => 0,
      onRelease: () => {},
    });
    buf.setTickIntervalMs(50);
    buf.enqueue(makeEvent("a", 0));
    expect(raf).not.toHaveBeenCalled();
    raf.mockRestore();
  });
});
