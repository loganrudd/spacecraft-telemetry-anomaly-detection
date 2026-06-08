import type { TelemetryEvent } from "../api/types";
import { telemetryStore } from "./telemetryStore";

// Client-side jitter buffer for the telemetry SSE stream.
//
// The server emits one event per channel at a steady cadence, but over a long
// network path the GFE/TCP layer delivers them in clumps — several events
// bunched together, then a gap — even though they were sent evenly. Rendering
// on arrival faithfully reproduces that unevenness as a periodic "skip".
//
// This buffer decouples rendering from arrival: incoming events are held in a
// per-channel FIFO and released to the store on a steady local clock, a fixed
// lead time behind live. It's the same technique audio/video players use to
// hide network jitter.
//
// The release rate is driven by `tickIntervalMs`, which the server exposes via
// GET /health as `replay_tick_ms` (= tick_interval_seconds * 1000 / speed).
// App.tsx calls `setTickIntervalMs` after the health response lands — before
// any SSE events arrive, since the stream only opens after health succeeds.
// No arrival-time estimation is needed or attempted, which avoids corruption
// by the server-side backlog burst sent to new subscribers on connect.

const DEFAULT_BUFFER_MS = 500;

const MIN_INTERVAL_MS = 8;
const MAX_INTERVAL_MS = 2000;

interface ChannelState {
  queue: TelemetryEvent[];
  flowing: boolean;
  primeDeadlineMs: number | null;
  nextReleaseMs: number;
}

export interface JitterBufferOptions {
  onRelease: (event: TelemetryEvent) => void;
  bufferMs?: number;
  /** Injectable clock for tests. */
  now?: () => number;
  /** When false, the internal rAF loop is not started — tests drive tick(). */
  autoRaf?: boolean;
}

export class TelemetryJitterBuffer {
  private tickIntervalMs: number | null = null;
  private readonly channels = new Map<string, ChannelState>();
  private readonly onRelease: (event: TelemetryEvent) => void;
  private readonly bufferMs: number;
  private readonly now: () => number;
  private readonly autoRaf: boolean;
  private rafId: number | null = null;

  constructor(opts: JitterBufferOptions) {
    this.onRelease = opts.onRelease;
    this.bufferMs = opts.bufferMs ?? DEFAULT_BUFFER_MS;
    this.now = opts.now ?? (() => performance.now());
    this.autoRaf = opts.autoRaf ?? true;
  }

  /**
   * Set the server's wall-clock tick interval (ms). Called once after the
   * health response arrives. This is the only source of timing — no
   * arrival-time estimation is done.
   */
  setTickIntervalMs(ms: number): void {
    this.tickIntervalMs = Math.min(MAX_INTERVAL_MS, Math.max(MIN_INTERVAL_MS, ms));
  }

  /** Record an arriving event. Does not release it — that happens on tick(). */
  enqueue(event: TelemetryEvent): void {
    const ch = this.channelFor(event.channel);
    ch.queue.push(event);

    if (!ch.flowing && ch.primeDeadlineMs === null) {
      ch.primeDeadlineMs = this.now() + this.bufferMs;
    }

    this.ensureRunning();
  }

  /**
   * Release all events that are due as of `nowMs`. Called by the rAF loop in
   * production; called directly by tests with a controlled clock.
   */
  tick(nowMs: number): void {
    const interval = this.tickIntervalMs;
    if (interval === null) return; // wait until setTickIntervalMs is called

    for (const ch of this.channels.values()) {
      if (ch.queue.length === 0) continue;

      if (!ch.flowing) {
        if (ch.primeDeadlineMs === null || nowMs < ch.primeDeadlineMs) continue;
        ch.flowing = true;
        ch.nextReleaseMs = nowMs;
      }

      // If the backlog has grown past 2× the target lead (e.g., a transient
      // clump exceeded the buffer), gently speed up — at most 2× — to drain
      // back without a visible burst.
      const backlogMs = ch.queue.length * interval;
      const targetMs = this.bufferMs;
      const effInterval =
        backlogMs > targetMs * 2
          ? Math.max(interval * 0.5, interval * (targetMs / backlogMs))
          : interval;

      while (ch.queue.length > 0 && nowMs >= ch.nextReleaseMs) {
        this.onRelease(ch.queue.shift()!);
        ch.nextReleaseMs += effInterval;
      }

      if (ch.queue.length === 0) {
        // Gap exceeded the buffer — re-prime before resuming so we don't fall
        // back to rendering raw jittery arrivals.
        ch.flowing = false;
        ch.primeDeadlineMs = null;
      }
    }
  }

  /** Clear all buffered events and stop the rAF loop. */
  reset(): void {
    if (this.rafId !== null && typeof cancelAnimationFrame !== "undefined") {
      cancelAnimationFrame(this.rafId);
    }
    this.rafId = null;
    this.channels.clear();
  }

  /** Total buffered (not-yet-released) events across all channels. */
  pendingCount(): number {
    let n = 0;
    for (const ch of this.channels.values()) n += ch.queue.length;
    return n;
  }

  private channelFor(channel: string): ChannelState {
    let ch = this.channels.get(channel);
    if (!ch) {
      ch = { queue: [], flowing: false, primeDeadlineMs: null, nextReleaseMs: 0 };
      this.channels.set(channel, ch);
    }
    return ch;
  }

  private ensureRunning(): void {
    if (!this.autoRaf || this.rafId !== null) return;
    if (typeof requestAnimationFrame === "undefined") return;
    const loop = () => {
      this.tick(this.now());
      this.rafId = requestAnimationFrame(loop);
    };
    this.rafId = requestAnimationFrame(loop);
  }
}

// Singleton wired to the telemetry store. App.tsx calls setTickIntervalMs()
// after health loads, then routes SSE events here instead of straight to the store.
export const telemetryJitterBuffer = new TelemetryJitterBuffer({
  onRelease: (event) => telemetryStore.push(event),
});
