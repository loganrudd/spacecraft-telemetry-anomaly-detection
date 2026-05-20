import { useEffect, useRef, useState } from "react";
import { telemetryStore } from "../state/telemetryStore";
import type { TelemetryEvent } from "../api/types";

const MAX_ALERTS = 50;

type Alert = {
  id: number;
  timestamp: string;
  channel: string;
  smoothed_error: number | null;
  threshold: number | null;
  ground_truth_match: boolean;
};

type Props = {
  channels: string[];
};

let _alertId = 0;

export default function AnomalyAlerts({ channels }: Props) {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  // Track previous is_anomaly_predicted per channel to detect rising edge
  const prevPredicted = useRef<Map<string, boolean>>(new Map());

  useEffect(() => {
    prevPredicted.current.clear();
    setAlerts([]);
  }, [channels]);

  useEffect(() => {
    if (channels.length === 0) return;

    const unsub = telemetryStore.subscribe(() => {
      for (const ch of channels) {
        const buf = telemetryStore.snapshot(ch);
        if (buf.length === 0) continue;
        const latest: TelemetryEvent = buf[buf.length - 1];
        const prev = prevPredicted.current.get(ch) ?? false;

        // Rising edge: false → true
        if (!prev && latest.is_anomaly_predicted) {
          const alert: Alert = {
            id: ++_alertId,
            timestamp: latest.timestamp,
            channel: ch,
            smoothed_error: latest.smoothed_error,
            threshold: latest.threshold,
            ground_truth_match: latest.is_anomaly,
          };
          setAlerts((prev) => {
            const next = [alert, ...prev];
            return next.slice(0, MAX_ALERTS);
          });
        }

        prevPredicted.current.set(ch, latest.is_anomaly_predicted);
      }
    });

    return unsub;
  }, [channels]);

  return (
    <aside className="anomaly-alerts">
      <header className="anomaly-alerts__header">
        <span className="anomaly-alerts__title">Anomaly Alerts</span>
        <span className="anomaly-alerts__count">{alerts.length}</span>
      </header>

      {alerts.length === 0 ? (
        <p className="anomaly-alerts__empty">No alerts yet.</p>
      ) : (
        <table className="anomaly-alerts__table">
          <thead>
            <tr>
              <th>Time</th>
              <th>Channel</th>
              <th>Score</th>
              <th>GT</th>
            </tr>
          </thead>
          <tbody>
            {alerts.map((a) => (
              <tr
                key={a.id}
                className={`anomaly-alerts__row${a.ground_truth_match ? " anomaly-alerts__row--tp" : " anomaly-alerts__row--fp"}`}
              >
                <td className="anomaly-alerts__ts">
                  {a.timestamp.replace("T", " ").replace("Z", "")}
                </td>
                <td>{a.channel}</td>
                <td>
                  {a.smoothed_error !== null
                    ? a.smoothed_error.toFixed(3)
                    : "—"}
                </td>
                <td>{a.ground_truth_match ? "✓" : "✗"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </aside>
  );
}
