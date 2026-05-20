import { useEffect, useRef, useState } from "react";
import { fetchHealth } from "../api/health";

type ConnectionState = "connecting" | "open" | "closed" | "error";

type Props = {
  connectionState: ConnectionState;
  eventsPerSecond: number;
};

const STATE_LABELS: Record<ConnectionState, string> = {
  connecting: "Connecting…",
  open: "Live",
  closed: "Closed",
  error: "Error",
};

const STATE_COLORS: Record<ConnectionState, string> = {
  connecting: "var(--fg-muted)",
  open: "var(--success)",
  closed: "var(--fg-muted)",
  error: "var(--danger)",
};

export default function StatusBar({ connectionState, eventsPerSecond }: Props) {
  const [uptime, setUptime] = useState<number | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    function refresh() {
      fetchHealth()
        .then((h) => setUptime(h.uptime_s))
        .catch(() => {});
    }
    refresh();
    timerRef.current = setInterval(refresh, 30_000);
    return () => {
      if (timerRef.current !== null) clearInterval(timerRef.current);
    };
  }, []);

  return (
    <header className="status-bar">
      <span className="status-bar__title">Spacecraft Telemetry Dashboard</span>
      <span
        className="status-bar__conn"
        style={{ color: STATE_COLORS[connectionState] }}
      >
        ● {STATE_LABELS[connectionState]}
      </span>
      <span className="status-bar__rate">
        {eventsPerSecond.toFixed(1)} ev/s
      </span>
      {uptime !== null && (
        <span className="status-bar__uptime">
          API uptime: {uptime.toFixed(0)}s
        </span>
      )}
    </header>
  );
}
