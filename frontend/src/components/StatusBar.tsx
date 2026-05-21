import { useEffect, useRef, useState } from "react";
import { fetchHealth } from "../api/health";

type ConnectionState = "connecting" | "open" | "closed" | "error";

type Speed = 10 | 100 | 1000;

type Props = {
  connectionState: ConnectionState;
  eventsPerSecond: number;
  speed: Speed;
  onSpeedChange: (speed: Speed) => void;
  mission: string | null;
  subsystem: string | null;
  onBackToOverview: () => void;
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

export default function StatusBar({
  connectionState,
  eventsPerSecond,
  speed,
  onSpeedChange,
  mission,
  subsystem,
  onBackToOverview,
}: Props) {
  const [uptime, setUptime] = useState<number | null>(null);
  const [apiOnline, setApiOnline] = useState(true);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    function refresh() {
      fetchHealth()
        .then((h) => {
          setUptime(h.uptime_s);
          setApiOnline(true);
        })
        .catch(() => setApiOnline(false));
    }
    refresh();
    timerRef.current = setInterval(refresh, 30_000);
    return () => {
      if (timerRef.current !== null) clearInterval(timerRef.current);
    };
  }, []);

  return (
    <header className="status-bar">
      <span className="status-bar__title">
        {mission && subsystem ? (
          <>
            <button
              className="status-bar__back"
              onClick={onBackToOverview}
              aria-label="Back to mission overview"
            >
              ← {mission}
            </button>
            <span className="status-bar__sep"> / </span>
            <span className="status-bar__subsystem">{subsystem}</span>
          </>
        ) : (
          "Spacecraft Telemetry Dashboard"
        )}
      </span>
      <span
        className="status-bar__conn"
        style={{ color: STATE_COLORS[connectionState] }}
      >
        ● {STATE_LABELS[connectionState]}
      </span>
      <span className="status-bar__rate">{eventsPerSecond} ev/s</span>
      <label className="status-bar__speed">
        Speed:&nbsp;
        <select
          value={speed}
          onChange={(e) => onSpeedChange(Number(e.target.value) as Speed)}
        >
          <option value={10}>10×</option>
          <option value={100}>100×</option>
          <option value={1000}>1000×</option>
        </select>
      </label>
      {apiOnline ? (
        uptime !== null && (
          <span className="status-bar__uptime">
            API uptime: {uptime.toFixed(0)}s
          </span>
        )
      ) : (
        <span className="status-bar__uptime" style={{ color: "var(--danger)" }}>
          API offline
        </span>
      )}
    </header>
  );
}
