import { useSyncExternalStore } from "react";
import type { TelemetryEvent } from "../api/types";

const BUFFER_SIZE = 600;
// Stable empty reference — useSyncExternalStore requires getSnapshot to return
// the same reference when nothing has changed. A `?? []` literal creates a new
// object each call, triggering an infinite re-render loop for empty channels.
const EMPTY: TelemetryEvent[] = [];

export class TelemetryStore {
  private buffers = new Map<string, TelemetryEvent[]>();
  // Snapshots hold a fresh slice minted by the rAF flush so useSyncExternalStore
  // can detect changes via reference equality.  The mutable buffer itself never
  // changes identity, so comparing buffer refs would miss every update.
  private snapshots = new Map<string, TelemetryEvent[]>();
  private listeners = new Set<() => void>();
  // rAF throttle state — tracks which channels need a new snapshot minted.
  private dirty = new Set<string>();
  private rafScheduled = false;
  // Epoch-ms of the most recent anomaly tick per channel. Used by useSubsystemRollup
  // to implement the 60-second rolling anomaly badge without re-reading history.
  lastAnomalyAtMs: Record<string, number> = {};

  push(event: TelemetryEvent): void {
    let buf = this.buffers.get(event.channel);
    if (!buf) {
      buf = [];
      this.buffers.set(event.channel, buf);
    }
    buf.push(event);
    if (buf.length > BUFFER_SIZE) buf.shift();
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
          if (b) this.snapshots.set(ch, b.slice());
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
    this.notify();
  }

  /** Force a synchronous snapshot mint. For use in tests only. */
  flushForTest(): void {
    this.rafScheduled = false;
    for (const ch of this.dirty) {
      const b = this.buffers.get(ch);
      if (b) this.snapshots.set(ch, b.slice());
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
