import { useEffect, useReducer } from "react";
import { telemetryStore, CHART_WINDOW } from "../state/telemetryStore";
import { formatChannel } from "../utils/formatChannel";

type Props = {
  channels: string[];
};

export default function AnomalyAlerts({ channels }: Props) {
  const [, forceUpdate] = useReducer((n: number) => n + 1, 0);

  useEffect(() => {
    // Store notifies on every rAF flush, so alerts clear automatically when
    // their triggering event scrolls off the chart — no polling needed.
    const unsub = telemetryStore.subscribe(forceUpdate);
    return () => unsub();
  }, []);

  const channelSet = new Set(channels);
  const alerts = telemetryStore.recentAlerts.filter(
    (a) =>
      channelSet.has(a.channel) &&
      telemetryStore.pushCount(a.channel) - a.anchorCount < CHART_WINDOW,
  );

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
                <td>{formatChannel(a.channel)}</td>
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
