import { useSyncExternalStore } from "react";
import type { TelemetryEvent } from "../api/types";

const BUFFER_SIZE = 600;
// Stable empty reference — useSyncExternalStore requires getSnapshot to return
// the same reference when nothing has changed. A `?? []` literal creates a new
// object each call, triggering an infinite re-render loop for empty channels.
const EMPTY: TelemetryEvent[] = [];

export class TelemetryStore {
  private buffers = new Map<string, TelemetryEvent[]>();
  // Snapshots hold a fresh slice after each push so useSyncExternalStore can
  // detect changes via reference equality (the mutable buffer itself never
  // changes identity, so comparing buffer refs would miss every update).
  private snapshots = new Map<string, TelemetryEvent[]>();
  private listeners = new Set<() => void>();

  push(event: TelemetryEvent): void {
    let buf = this.buffers.get(event.channel);
    if (!buf) {
      buf = [];
      this.buffers.set(event.channel, buf);
    }
    buf.push(event);
    if (buf.length > BUFFER_SIZE) buf.shift();
    this.snapshots.set(event.channel, buf.slice());
    this.notify();
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
