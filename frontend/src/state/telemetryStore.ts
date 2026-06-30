import { useSyncExternalStore } from "react";
import type { RawTelemetryEvent, TelemetryEvent } from "../api/types";

const BUFFER_SIZE = 600;

// Number of events shown in the chart's rolling window.  Alerts are visible
// exactly as long as their triggering event remains in this window.  Must
// match CHART_WINDOW in TelemetryChart.tsx.
export const CHART_WINDOW = 200;

// Shared TTL for anomaly recency — used by the overview badge in
// useSubsystemRollup to keep the badge lit for 60 s after an anomaly.
export const ANOMALY_TTL_MS = 60_000;
const MAX_ALERTS = 50;
let _alertId = 0;

export type StoredAlert = {
  id: number;
  capturedAtCount: number; // per-channel push count at capture — used for chart-window visibility
  timestamp: string;       // telemetry event timestamp (display only)
  channel: string;
  smoothed_error: number | null;
  threshold: number | null;
  ground_truth_match: boolean;
  // False for ISS live events where is_anomaly=false (no injection active — GT
  // unknown). True for ESA (all events are labeled) and ISS during injection.
  gt_available: boolean;
};
// Stable empty reference — useSyncExternalStore requires getSnapshot to return
// the same reference when nothing has changed. A `?? []` literal creates a new
// object each call, triggering an infinite re-render loop for empty channels.
const EMPTY: TelemetryEvent[] = [];

// O(1) ring buffer — avoids the O(n) memmove of Array.shift() on every eviction.
// `head` points to the oldest item; `size` tracks how many slots are valid.
interface RingBuffer {
  items: TelemetryEvent[];
  head: number;
  size: number;
}

function ringPush(ring: RingBuffer, event: TelemetryEvent): void {
  if (ring.size < BUFFER_SIZE) {
    ring.items[(ring.head + ring.size) % BUFFER_SIZE] = event;
    ring.size++;
  } else {
    // Full: overwrite the oldest slot and advance head.
    ring.items[ring.head] = event;
    ring.head = (ring.head + 1) % BUFFER_SIZE;
  }
}

function ringToArray(ring: RingBuffer): TelemetryEvent[] {
  const arr = new Array<TelemetryEvent>(ring.size);
  for (let i = 0; i < ring.size; i++) {
    arr[i] = ring.items[(ring.head + i) % BUFFER_SIZE];
  }
  return arr;
}

export class TelemetryStore {
  private buffers = new Map<string, RingBuffer>();
  // Snapshots hold a fresh ordered array minted by the rAF flush so
  // useSyncExternalStore can detect changes via reference equality.
  private snapshots = new Map<string, TelemetryEvent[]>();
  private listeners = new Set<() => void>();
  // rAF throttle state — tracks which channels need a new snapshot minted.
  private dirty = new Set<string>();
  private rafScheduled = false;
  // Epoch-ms of the most recent anomaly tick per channel. Kept for reference;
  // the overview badge now uses lastAnomalyPushCount instead.
  lastAnomalyAtMs: Record<string, number> = {};
  // Per-channel push count of the most recent predicted-anomaly event. Used by
  // useSubsystemRollup so the overview badge clears when the event scrolls off
  // the chart window (same boundary as the alerts panel).
  lastAnomalyPushCount: Record<string, number> = {};
  // Epoch-ms of the most recent tick per channel. Used by useSubsystemRollup to
  // determine whether a channel is actively streaming without reading the full buffer.
  lastTickAtMs: Record<string, number> = {};
  // Monotonically increasing push count per channel.  Stored in each alert so
  // AnomalyAlerts can tell whether the alert is still within the chart window.
  private _pushCount: Record<string, number> = {};
  // Rising-edge anomaly alerts captured globally (regardless of which channel view
  // is mounted). AnomalyAlerts reads from here and removes alerts whose triggering
  // event has scrolled off the chart window.
  recentAlerts: StoredAlert[] = [];
  private prevPredicted: Record<string, boolean> = {};
  // Last *telemetry*-event timestamp (epoch-ms) per channel. Telemetry events
  // are strictly ascending within a pass (replay ticks, or 30s grid buckets in
  // live mode), so a backwards jump means the shared replay loop wrapped back to
  // the slice start and the store should clear. Tracked SEPARATELY from raw
  // ticks (lastRawTsMs): in live mode the two streams use different time bases —
  // raw = wall-clock now, telemetry = bucket-start (~30s behind) — so a shared
  // guard would false-fire a "wrap" on every bucket close and blank the chart.
  private lastTsMs: Record<string, number> = {};
  // Last *raw*-event timestamp (epoch-ms) per channel. Independent monotonicity
  // guard for the event: raw stream (live pump only). See lastTsMs note above.
  private lastRawTsMs: Record<string, number> = {};

  push(event: TelemetryEvent): void {
    // Detect the shared replay loop wrapping. The SSE connection stays open
    // across the wrap (run_shared_loop never closes it), so onOpen never
    // re-fires — we can't key the clean restart off reconnects. Instead, when
    // a channel's timestamp jumps backwards we clear the whole store so the
    // chart restarts from the loop's start instead of showing a backwards jump
    // and a window that mixes the end of the old pass with the start of the new.
    const tsMs = Date.parse(event.timestamp);
    const prevTsMs = this.lastTsMs[event.channel];
    if (prevTsMs !== undefined && tsMs < prevTsMs) {
      this.clear(); // resets lastTsMs too, so sibling channels don't re-trigger
    }

    let ring = this.buffers.get(event.channel);
    if (!ring) {
      ring = { items: new Array<TelemetryEvent>(BUFFER_SIZE), head: 0, size: 0 };
      this.buffers.set(event.channel, ring);
    }
    ringPush(ring, event);
    const nowMs = Date.now();
    this.lastTickAtMs[event.channel] = nowMs;
    const count = (this._pushCount[event.channel] ?? 0) + 1;
    this._pushCount[event.channel] = count;
    if (event.is_anomaly_predicted) {
      this.lastAnomalyAtMs[event.channel] = nowMs;
      this.lastAnomalyPushCount[event.channel] = count;
    }
    // Rising-edge detection: false → true transition triggers a stored alert.
    const prevPred = this.prevPredicted[event.channel] ?? false;
    if (!prevPred && event.is_anomaly_predicted) {
      // GT is available for ESA (always labeled) and for ISS during injection
      // (is_anomaly=true). For ISS without injection is_anomaly=false means
      // "label unknown" — not a confirmed FP — so gt_available=false.
      const gt_available = event.is_anomaly || !event.mission.startsWith("ISS");
      const alert: StoredAlert = {
        id: ++_alertId,
        capturedAtCount: count,
        timestamp: event.timestamp,
        channel: event.channel,
        smoothed_error: event.smoothed_error,
        threshold: event.threshold,
        ground_truth_match: event.is_anomaly,
        gt_available,
      };
      this.recentAlerts = [alert, ...this.recentAlerts].slice(0, MAX_ALERTS);
    }
    this.prevPredicted[event.channel] = event.is_anomaly_predicted;
    this.lastTsMs[event.channel] = tsMs;
    this.dirty.add(event.channel);
    this._scheduleFlush();
  }

  /**
   * Push a raw tick from the live pump (event: raw) into the ring buffer.
   *
   * Raw events carry only value_normalized — no prediction or anomaly fields.
   * They drive the continuous chart line at native Lightstreamer cadence (1-10s)
   * while the slower TelemetryEvents (event: telemetry, 30s) carry predictions.
   * Anomaly / alert state is NOT updated here.
   */
  pushRaw(event: RawTelemetryEvent): void {
    const tsMs = Date.parse(event.timestamp);
    // Guard the raw stream against its own wrap only (raw-to-raw). Must NOT
    // compare against lastTsMs (telemetry): the live telemetry event for a
    // closed 30s bucket is labeled with the bucket start, ~30s behind these
    // wall-clock raw ticks, so a shared guard would clear the whole store on
    // every bucket close — the "Waiting for data…" flapping.
    const prevTsMs = this.lastRawTsMs[event.channel];
    if (prevTsMs !== undefined && tsMs < prevTsMs) {
      this.clear();
    }
    // Construct a synthetic TelemetryEvent with null prediction/anomaly fields
    // so the existing chart pipeline can render the value_normalized point.
    const synthetic: TelemetryEvent = {
      timestamp: event.timestamp,
      mission: "",
      channel: event.channel,
      value_normalized: event.value_normalized,
      prediction: null,
      residual: null,
      smoothed_error: null,
      threshold: null,
      is_anomaly_predicted: false,
      is_anomaly: false,
    };
    let ring = this.buffers.get(synthetic.channel);
    if (!ring) {
      ring = { items: new Array<TelemetryEvent>(BUFFER_SIZE), head: 0, size: 0 };
      this.buffers.set(synthetic.channel, ring);
    }
    ringPush(ring, synthetic);
    const nowMs = Date.now();
    this.lastTickAtMs[synthetic.channel] = nowMs;
    const count = (this._pushCount[synthetic.channel] ?? 0) + 1;
    this._pushCount[synthetic.channel] = count;
    this.lastRawTsMs[synthetic.channel] = tsMs;
    this.dirty.add(synthetic.channel);
    this._scheduleFlush();
  }

  private _scheduleFlush(): void {
    if (!this.rafScheduled) {
      this.rafScheduled = true;
      requestAnimationFrame(() => {
        this.rafScheduled = false;
        for (const ch of this.dirty) {
          const b = this.buffers.get(ch);
          if (b) this.snapshots.set(ch, ringToArray(b));
        }
        this.dirty.clear();
        this.notify();
      });
    }
  }

  snapshot(channel: string): TelemetryEvent[] {
    return this.snapshots.get(channel) ?? EMPTY;
  }

  channels(): string[] {
    return Array.from(this.buffers.keys());
  }

  pushCount(channel: string): number {
    return this._pushCount[channel] ?? 0;
  }

  clear(): void {
    this.buffers.clear();
    this.snapshots.clear();
    this.dirty.clear();
    this.lastAnomalyAtMs = {};
    this.lastAnomalyPushCount = {};
    this.lastTickAtMs = {};
    this._pushCount = {};
    this.recentAlerts = [];
    this.prevPredicted = {};
    this.lastTsMs = {};
    this.lastRawTsMs = {};
    this.notify();
  }

  /** Force a synchronous snapshot mint. For use in tests only. */
  flushForTest(): void {
    this.rafScheduled = false;
    for (const ch of this.dirty) {
      const b = this.buffers.get(ch);
      if (b) this.snapshots.set(ch, ringToArray(b));
    }
    this.dirty.clear();
    this.notify();
  }

  subscribe(fn: () => void): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  private notify(): void {
    this.listeners.forEach((fn) => fn());
  }
}

export const telemetryStore = new TelemetryStore();

export function useTelemetryChannel(channel: string): TelemetryEvent[] {
  return useSyncExternalStore(
    (cb) => telemetryStore.subscribe(cb),
    () => telemetryStore.snapshot(channel),
    () => EMPTY,
  );
}
