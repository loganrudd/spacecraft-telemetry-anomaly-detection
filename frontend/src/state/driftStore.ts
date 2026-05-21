import { useSyncExternalStore } from "react";
import type { DriftEvent } from "../api/types";

// Stable empty sentinel — avoids useSyncExternalStore infinite loop on null returns.
const NULL_EVENT: null = null;

export class DriftStore {
  // Latest DriftEvent per channel (point-in-time snapshot; history lives in MLflow).
  private latest = new Map<string, DriftEvent>();
  // Timestamp of the first event where drifted=true for each channel.
  // Cleared when a drifted=false event arrives so the next drift onset gets a fresh time.
  private driftedSince = new Map<string, string>();
  // Most recent subsystem-level aggregation fields (populated periodically by the server).
  private subsystemPct: number | null = null;
  private subsystemAlert = false;

  private listeners = new Set<() => void>();
  // rAF throttle — drift events arrive infrequently (~1/60 ticks), but keep
  // the same batched-flush pattern as TelemetryStore for consistency.
  private rafScheduled = false;

  push(event: DriftEvent): void {
    this.latest.set(event.channel, event);
    if (event.drifted && !this.driftedSince.has(event.channel)) {
      this.driftedSince.set(event.channel, event.timestamp);
    } else if (!event.drifted) {
      this.driftedSince.delete(event.channel);
    }
    if (event.subsystem_percent_drifted !== null && event.subsystem_alert !== null) {
      this.subsystemPct = event.subsystem_percent_drifted;
      this.subsystemAlert = event.subsystem_alert;
    }
    if (!this.rafScheduled) {
      this.rafScheduled = true;
      requestAnimationFrame(() => {
        this.rafScheduled = false;
        this.notify();
      });
    }
  }

  latestForChannel(channel: string): DriftEvent | null {
    return this.latest.get(channel) ?? NULL_EVENT;
  }

  driftedSinceForChannel(channel: string): string | null {
    return this.driftedSince.get(channel) ?? NULL_EVENT;
  }

  subsystemPercent(): number | null {
    return this.subsystemPct;
  }

  isSubsystemAlert(): boolean {
    return this.subsystemAlert;
  }

  clear(): void {
    this.latest.clear();
    this.driftedSince.clear();
    this.subsystemPct = null;
    this.subsystemAlert = false;
    this.notify();
  }

  /** Force a synchronous update for tests (avoids requestAnimationFrame dependency). */
  flushForTest(): void {
    this.rafScheduled = false;
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

export const driftStore = new DriftStore();

export function useChannelDrift(channel: string): DriftEvent | null {
  return useSyncExternalStore(
    (cb) => driftStore.subscribe(cb),
    () => driftStore.latestForChannel(channel),
    () => NULL_EVENT,
  );
}

export function useDriftedSince(channel: string): string | null {
  return useSyncExternalStore(
    (cb) => driftStore.subscribe(cb),
    () => driftStore.driftedSinceForChannel(channel),
    () => NULL_EVENT,
  );
}

export function useSubsystemDrift(): {
  percent: number | null;
  alert: boolean;
} {
  const percent = useSyncExternalStore(
    (cb) => driftStore.subscribe(cb),
    () => driftStore.subsystemPercent(),
    () => NULL_EVENT,
  );
  const alert = useSyncExternalStore(
    (cb) => driftStore.subscribe(cb),
    () => driftStore.isSubsystemAlert(),
    () => false,
  );
  return { percent, alert };
}
