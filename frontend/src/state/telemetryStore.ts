import { useSyncExternalStore } from "react";
import type { TelemetryEvent } from "../api/types";

const BUFFER_SIZE = 600;
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
  // Epoch-ms of the most recent anomaly tick per channel. Used by useSubsystemRollup
  // to implement the 60-second rolling anomaly badge without re-reading history.
  lastAnomalyAtMs: Record<string, number> = {};
  // Epoch-ms of the most recent tick per channel. Used by useSubsystemRollup to
  // determine whether a channel is actively streaming without reading the full buffer.
  lastTickAtMs: Record<string, number> = {};

  push(event: TelemetryEvent): void {
    let ring = this.buffers.get(event.channel);
    if (!ring) {
      ring = { items: new Array<TelemetryEvent>(BUFFER_SIZE), head: 0, size: 0 };
      this.buffers.set(event.channel, ring);
    }
    ringPush(ring, event);
    this.lastTickAtMs[event.channel] = Date.now();
    if (event.is_anomaly_predicted) {
      this.lastAnomalyAtMs[event.channel] = Date.now();
    }
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

  clear(): void {
    this.buffers.clear();
    this.snapshots.clear();
    this.dirty.clear();
    this.lastAnomalyAtMs = {};
    this.lastTickAtMs = {};
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
