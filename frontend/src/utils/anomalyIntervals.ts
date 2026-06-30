import type { TelemetryEvent } from "../api/types";

export type Interval = { startTs: string; endTs: string };

// A labeled (ground-truth) anomaly window, plus whether the model flagged ANY
// point inside it. This matches how Telemanom is scored (segment-level /
// "point-adjust", Hundman et al.): a labeled window counts as detected if the
// model fires anywhere within it, not only at the exact flagged timesteps.
export type DetectedInterval = Interval & { detected: boolean };

export function collapseFlags(
  events: TelemetryEvent[],
  flagKey: "is_anomaly" | "is_anomaly_predicted",
): Interval[] {
  const out: Interval[] = [];
  let open: Interval | null = null;

  for (const e of events) {
    if (e[flagKey]) {
      if (!open) {
        open = { startTs: e.timestamp, endTs: e.timestamp };
      } else {
        open.endTs = e.timestamp;
      }
    } else if (open) {
      // Extend to the first non-flagged event so single-bucket anomalies have
      // visible width on the chart (startTs === endTs renders as zero-width).
      open.endTs = e.timestamp;
      out.push(open);
      open = null;
    }
  }

  if (open) out.push(open);
  return out;
}

/**
 * Collapse runs of labeled (`is_anomaly`) events into intervals, marking each
 * interval `detected` if ANY event inside it was flagged by the model
 * (`is_anomaly_predicted`). Used to render a labeled window as "caught" at the
 * segment level and to drive the segment-aware tooltip.
 */
export function labeledIntervalsWithDetection(
  events: TelemetryEvent[],
): DetectedInterval[] {
  const out: DetectedInterval[] = [];
  let open: DetectedInterval | null = null;

  for (const e of events) {
    if (e.is_anomaly) {
      if (!open) {
        open = {
          startTs: e.timestamp,
          endTs: e.timestamp,
          detected: e.is_anomaly_predicted,
        };
      } else {
        open.endTs = e.timestamp;
        open.detected = open.detected || e.is_anomaly_predicted;
      }
    } else if (open) {
      // Extend to the first non-labeled event for visible width (same reason as
      // collapseFlags above).
      open.endTs = e.timestamp;
      out.push(open);
      open = null;
    }
  }

  if (open) out.push(open);
  return out;
}
