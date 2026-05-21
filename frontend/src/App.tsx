import { useEffect, useRef, useState } from "react";
import AnomalyAlerts from "./components/AnomalyAlerts";
import ChannelPicker from "./components/ChannelPicker";
import DriftPanel from "./components/DriftPanel";
import MissionOverview from "./components/MissionOverview";
import StatusBar from "./components/StatusBar";
import TelemetryChart from "./components/TelemetryChart";
import { fetchHealth } from "./api/health";
import { openTelemetryStream } from "./api/telemetryStream";
import { openDriftStream } from "./api/driftStream";
import { telemetryStore } from "./state/telemetryStore";
import { driftStore } from "./state/driftStore";
import type { HealthResponse } from "./api/types";
import type { StreamHandle } from "./api/telemetryStream";
import type { DriftStreamHandle } from "./api/driftStream";

type ConnectionState = "connecting" | "open" | "closed" | "error";
type DensityTier = "comfortable" | "compact" | "dense";

type View =
  | { kind: "overview" }
  | { kind: "subsystem"; subsystem: string; selected: string[] };

const SPEED_OPTIONS = [10, 100, 1000] as const;
type Speed = (typeof SPEED_OPTIONS)[number];

function getDensity(count: number): DensityTier {
  if (count <= 4) return "comfortable";
  if (count <= 12) return "compact";
  return "dense";
}

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [view, setView] = useState<View>({ kind: "overview" });
  const [connState, setConnState] = useState<ConnectionState>("closed");
  const [speed, setSpeed] = useState<Speed>(100);
  const [evPerSec, setEvPerSec] = useState(0);
  const [driftDisabled, setDriftDisabled] = useState(false);
  const streamRef = useRef<StreamHandle | null>(null);
  const driftStreamRef = useRef<DriftStreamHandle | null>(null);
  const tickCountRef = useRef(0);

  // Fetch health once on mount to get channel list + subsystem map.
  useEffect(() => {
    fetchHealth().then(setHealth).catch(console.error);
  }, []);

  // Publish tick rate at 1Hz.
  useEffect(() => {
    const id = setInterval(() => {
      setEvPerSec(tickCountRef.current);
      tickCountRef.current = 0;
    }, 1_000);
    return () => clearInterval(id);
  }, []);

  // Open SSE streams for ALL loaded channels on mount (keeps overview warm).
  // Streams stay open across subsystem navigation — no reconnect on drill-in.
  useEffect(() => {
    if (!health || health.channels_loaded.length === 0) return;

    const allChannels = health.channels_loaded;
    setConnState("connecting");
    telemetryStore.clear();
    driftStore.clear();

    const handle = openTelemetryStream({
      channels: allChannels,
      speed,
      onEvent: (e) => {
        telemetryStore.push(e);
        tickCountRef.current += 1;
      },
      onOpen: () => setConnState("open"),
      onError: () => setConnState("error"),
    });
    streamRef.current = handle;

    const driftHandle = openDriftStream({
      channels: allChannels,
      onEvent: (e) => driftStore.push(e),
      onError: (err) => {
        const es = err.target as EventSource;
        if (es.readyState === EventSource.CLOSED) setDriftDisabled(true);
      },
    });
    setDriftDisabled(false);
    driftStreamRef.current = driftHandle;

    return () => {
      handle.close();
      driftHandle.close();
      streamRef.current = null;
      driftStreamRef.current = null;
    };
  // Re-open only when health loads or speed changes (channels don't change at runtime).
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [health, speed]);

  function enterSubsystem(subsystem: string, channels: string[]) {
    setView({ kind: "subsystem", subsystem, selected: channels });
  }

  function backToOverview() {
    setView({ kind: "overview" });
  }

  const selected = view.kind === "subsystem" ? view.selected : [];
  const subsystem = view.kind === "subsystem" ? view.subsystem : null;
  const density = getDensity(selected.length);

  return (
    <div className="app">
      <StatusBar
        connectionState={connState}
        eventsPerSecond={evPerSec}
        speed={speed}
        onSpeedChange={setSpeed}
        mission={health?.mission ?? null}
        subsystem={subsystem}
        onBackToOverview={backToOverview}
      />

      <div className="app__body">
        {view.kind === "overview" ? (
          <main className="app__main">
            {health ? (
              <MissionOverview
                mission={health.mission}
                channelSubsystems={health.channel_subsystems}
                onEnterSubsystem={enterSubsystem}
              />
            ) : (
              <div className="app__empty">
                <p>Connecting to mission…</p>
              </div>
            )}
          </main>
        ) : (
          <>
            <ChannelPicker
              allChannels={
                health
                  ? health.channels_loaded.filter(
                      (ch) => health.channel_subsystems[ch] === subsystem,
                    )
                  : []
              }
              selected={selected}
              onChange={(chs) =>
                setView({ kind: "subsystem", subsystem: subsystem!, selected: chs })
              }
            />

            <main className="app__main">
              {selected.length === 0 ? (
                <div className="app__empty">
                  <p>Select one or more channels from the left panel to begin.</p>
                </div>
              ) : (
                <div className={`app__charts app__charts--${density}`}>
                  {selected.map((ch) => (
                    <TelemetryChart key={ch} channel={ch} density={density} />
                  ))}
                </div>
              )}
            </main>

            <div className="app__right">
              <AnomalyAlerts channels={selected} />
              <DriftPanel channels={selected} disabled={driftDisabled} />
            </div>
          </>
        )}
      </div>
    </div>
  );
}
