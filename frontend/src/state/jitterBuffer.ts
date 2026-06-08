import type { TelemetryEvent } from "../api/types";
import { telemetryStore } from "./telemetryStore";

// Client-side jitter buffer for the telemetry SSE stream.
//
// The server emits one event per channel at a steady cadence (one per replay
// tick), but over a long network path (e.g. a viewer in South America hitting
// us-central1) the GFE/TCP layer delivers them in *clumps* — several events
// bunched together, then a gap — even though they were sent evenly. Rendering
// on arrival faithfully reproduces that unevenness as a periodic "skip".
//
// This buffer decouples rendering from arrival: incoming events are held in a
// per-channel FIFO and released to the store on a steady local clock, a fixed
// lead time behind live. It's the same technique audio/video players use to
// hide network jitter, and it works regardless of where the viewer is.
//
// Pacing is rate-based, not timestamp-based: we measure each channel's average
// wall-clock arrival interval (a cumulative mean, which converges to the true
// cadence regardless of clumping) and release at that interval. This needs no
// knowledge of the server's replay speed and is transparent to replay-loop
// restarts (the cadence is unchanged when the slice loops; only the data is).

// Lead time held before an event becomes visible. Must exceed the largest gap
// between arrival clumps; 500 ms comfortably covers the sub-second clumping
// observed on Cloud Run from a high-latency client. Display lag of 500 ms is
// imperceptible for a telemetry replay.
const DEFAULT_BUFFER_MS = 500;

// Clamp the measured interval so a noisy early estimate or a stalled stream
// can't produce absurd pacing.
const MIN_INTERVAL_MS = 8;
const MAX_INTERVAL_MS = 2000;

interface ChannelState {
  queue: TelemetryEvent[];
  firstArrivalMs: number | null; // for the cumulative-mean interval estimate
  count: number;
  intervalMs: number | null; // measured average wall-clock spacing
  flowing: boolean; // true once primed with a full lead buffer
  primeDeadlineMs: number | null; // wall-clock time at which priming completes
  nextReleaseMs: number; // wall-clock time the next event is due
}

export interface JitterBufferOptions {
  onRelease: (event: TelemetryEvent) => void;
  bufferMs?: number;
  /** Injectable clock for tests. Defaults to performance.now. */
  now?: () => number;
  /** When false, the internal rAF loop is not started — tests drive tick(). */
  autoRaf?: boolean;
}

export class TelemetryJitterBuffer {
  private readonly channels = new Map<string, ChannelState>();
  private readonly onRelease: (event: TelemetryEvent) => void;
  private readonly bufferMs: number;
  private readonly maxLeadMs: number;
  private readonly now: () => number;
  private readonly autoRaf: boolean;
  private rafId: number | null = null;

  constructor(opts: JitterBufferOptions) {
    this.onRelease = opts.onRelease;
    this.bufferMs = opts.bufferMs ?? DEFAULT_BUFFER_MS;
    this.maxLeadMs = this.bufferMs * 2;
    this.now = opts.now ?? (() => performance.now());
    this.autoRaf = opts.autoRaf ?? true;
  }

  /** Record an arriving event. Does not release it — that happens on tick(). */
  enqueue(event: TelemetryEvent): void {
    const now = this.now();
    const ch = this.channelFor(event.channel);
    ch.queue.push(event);
    ch.count += 1;

    if (ch.firstArrivalMs === null) {
      ch.firstArrivalMs = now;
    } else if (ch.count > 1) {
      // Cumulative mean: total span / number of intervals. Converges to the
      // true cadence because clumping changes the gap distribution, not its mean.
      const raw = (now - ch.firstArrivalMs) / (ch.count - 1);
      ch.intervalMs = Math.min(MAX_INTERVAL_MS, Math.max(MIN_INTERVAL_MS, raw));
    }

    // Arm the prime deadline on the first event of a (re)start of flow.
    if (!ch.flowing && ch.primeDeadlineMs === null) {
      ch.primeDeadlineMs = now + this.bufferMs;
    }

    this.ensureRunning();
  }

  /**
   * Release all events that are due as of `nowMs`. Called by the rAF loop in
   * production; called directly by tests with a controlled clock.
   */
  tick(nowMs: number): void {
    for (const ch of this.channels.values()) {
      const interval = ch.intervalMs;
      if (interval === null || ch.queue.length === 0) continue;

      if (!ch.flowing) {
        // Still filling the lead buffer.
        if (ch.primeDeadlineMs === null || nowMs < ch.primeDeadlineMs) continue;
        ch.flowing = true;
        ch.nextReleaseMs = nowMs;
      }

      // If the backlog has grown past the target lead (transient clump, or a
      // slightly-too-large interval estimate), gently speed up — down to 2x —
      // to drain back toward the target, bounding display latency without a
      // visible burst.
      const backlogMs = ch.queue.length * interval;
      const effInterval =
        backlogMs > this.maxLeadMs
          ? Math.max(interval * 0.5, interval * (this.bufferMs / backlogMs))
          : interval;

      while (ch.queue.length > 0 && nowMs >= ch.nextReleaseMs) {
        this.onRelease(ch.queue.shift()!);
        ch.nextReleaseMs += effInterval;
      }

      if (ch.queue.length === 0) {
        // Underrun: the lead is exhausted (a gap exceeded the buffer). Re-prime
        // before resuming so we don't fall back to rendering raw arrivals.
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

  /** Total buffered (not-yet-released) events across channels. For tests/debug. */
  pendingCount(): number {
    let n = 0;
    for (const ch of this.channels.values()) n += ch.queue.length;
    return n;
  }

  private channelFor(channel: string): ChannelState {
    let ch = this.channels.get(channel);
    if (!ch) {
      ch = {
        queue: [],
        firstArrivalMs: null,
        count: 0,
        intervalMs: null,
        flowing: false,
        primeDeadlineMs: null,
        nextReleaseMs: 0,
      };
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

// Singleton wired to the telemetry store. App routes SSE events here instead of
// pushing straight to the store; the buffer releases them on a steady clock.
export const telemetryJitterBuffer = new TelemetryJitterBuffer({
  onRelease: (event) => telemetryStore.push(event),
});
