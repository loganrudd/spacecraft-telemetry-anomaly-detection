import { useSyncExternalStore } from "react";
import type { TelemetryEvent } from "../api/types";

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

  push(event: TelemetryEvent): void {
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
      const alert: StoredAlert = {
        id: ++_alertId,
        capturedAtCount: count,
        timestamp: event.timestamp,
        channel: event.channel,
        smoothed_error: event.smoothed_error,
        threshold: event.threshold,
        ground_truth_match: event.is_anomaly,
      };
      this.recentAlerts = [alert, ...this.recentAlerts].slice(0, MAX_ALERTS);
    }
    this.prevPredicted[event.channel] = event.is_anomaly_predicted;
    this.dirty.add(event.channel);

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
