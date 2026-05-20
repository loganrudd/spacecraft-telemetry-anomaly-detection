import { useSyncExternalStore } from "react";
import type { TelemetryEvent } from "../api/types";

const BUFFER_SIZE = 600;

class TelemetryStore {
  private buffers = new Map<string, TelemetryEvent[]>();
  private listeners = new Set<() => void>();

  push(event: TelemetryEvent): void {
    let buf = this.buffers.get(event.channel);
    if (!buf) {
      buf = [];
      this.buffers.set(event.channel, buf);
    }
    buf.push(event);
    if (buf.length > BUFFER_SIZE) buf.shift();
    this.notify();
  }

  snapshot(channel: string): TelemetryEvent[] {
    return this.buffers.get(channel) ?? [];
  }

  channels(): string[] {
    return Array.from(this.buffers.keys());
  }

  clear(): void {
    this.buffers.clear();
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
    () => [],
  );
}
