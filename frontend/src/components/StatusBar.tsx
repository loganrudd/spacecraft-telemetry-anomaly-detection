import { useEffect, useRef, useState } from "react";
import { fetchHealth } from "../api/health";
import MissionSwitcher from "./MissionSwitcher";
import type { MissionLink } from "../api/types";

type ConnectionState = "connecting" | "open" | "closed" | "error";
type LiveStreamStatus = "live" | "los" | "connecting" | "closed";

type Props = {
  connectionState: ConnectionState;
  liveStatus: LiveStreamStatus;
  eventsPerSecond: number;
  mission: string | null;
  subsystem: string | null;
  availableMissions: MissionLink[];
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
  liveStatus,
  eventsPerSecond,
  mission,
  subsystem,
  availableMissions,
  onBackToOverview,
}: Props) {
  // During LOS the SSE transport stays open (replay keeps streaming), so the
  // connection state alone would still read "Live". Surface the pump's actual
  // live/LOS status so the header agrees with the LOS banner.
  const inLos = liveStatus === "los";
  const connLabel = inLos ? "Signal lost · replay" : STATE_LABELS[connectionState];
  const connColor = inLos ? "var(--alert)" : STATE_COLORS[connectionState];
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
      <span className="status-bar__conn" style={{ color: connColor }}>
        ● {connLabel}
      </span>
      <span className="status-bar__rate">{eventsPerSecond} ev/s</span>
      {mission && (
        <MissionSwitcher
          availableMissions={availableMissions}
          currentMissionId={mission}
        />
      )}
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
