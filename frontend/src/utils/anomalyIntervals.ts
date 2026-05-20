import type { TelemetryEvent } from "../api/types";

export type Interval = { startTs: string; endTs: string };

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
      out.push(open);
      open = null;
    }
  }

  if (open) out.push(open);
  return out;
}
