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

// Set to true to skip drift monitoring (e.g. short demo windows where the
// rolling window never fills). Flip back to false to re-enable.
const DRIFT_DISABLED = true;

type ConnectionState = "connecting" | "open" | "closed" | "error";
type DensityTier = "comfortable" | "compact" | "dense";

type View =
  | { kind: "overview" }
  | { kind: "subsystem"; subsystem: string; selected: string[] };


function getDensity(count: number): DensityTier {
  if (count <= 4) return "comfortable";
  if (count <= 12) return "compact";
  return "dense";
}

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [view, setView] = useState<View>({ kind: "overview" });
  const [connState, setConnState] = useState<ConnectionState>("closed");
  const [evPerSec, setEvPerSec] = useState(0);
  const [driftDisabled, setDriftDisabled] = useState(false);
  const streamRef = useRef<StreamHandle | null>(null);
  const driftStreamRef = useRef<DriftStreamHandle | null>(null);
  const tickCountRef = useRef(0);
  const autoNavigatedRef = useRef(false);

  // Fetch health on mount; poll every 2 s while status === "loading".
  useEffect(() => {
    fetchHealth().then(setHealth).catch(console.error);
  }, []);

  useEffect(() => {
    if (!health || health.status !== "loading") return;
    const id = setInterval(() => {
      fetchHealth().then(setHealth).catch(console.error);
    }, 2_000);
    return () => clearInterval(id);
  }, [health]);

  // Publish tick rate at 1Hz.
  useEffect(() => {
    const id = setInterval(() => {
      setEvPerSec(tickCountRef.current);
      tickCountRef.current = 0;
    }, 1_000);
    return () => clearInterval(id);
  }, []);

  // Open SSE streams for ALL loaded channels (keeps overview warm).
  // Streams stay open across subsystem navigation — no reconnect on drill-in.
  //
  // Visibility-gated: an open SSE connection is a long-lived in-flight request,
  // and Cloud Run will not scale a serving instance to zero while any request is
  // active. A left-open/backgrounded tab holds two such connections (telemetry +
  // drift) and would pin an instance indefinitely (billed overnight). We close
  // both streams when the tab is hidden and reopen when it's visible again, so
  // unviewed tabs release the instance and let it scale to zero. A server-side
  // timeout would not help here: EventSource auto-reconnects, so a backgrounded
  // tab would immediately re-pin the instance.
  useEffect(() => {
    if (!health || health.channels_loaded.length === 0) return;

    const allChannels = health.channels_loaded;

    const openStreams = () => {
      if (streamRef.current) return; // already open
      setConnState("connecting");
      telemetryStore.clear();
      driftStore.clear();

      streamRef.current = openTelemetryStream({
        channels: allChannels,
        onEvent: (e) => {
          telemetryStore.push(e);
          tickCountRef.current += 1;
        },
        onOpen: () => setConnState("open"),
        onError: () => setConnState("error"),
      });

      setDriftDisabled(false);
      driftStreamRef.current = openDriftStream({
        channels: allChannels,
        onEvent: (e) => driftStore.push(e),
        onError: (err) => {
          const es = err.target as EventSource;
          if (es.readyState === EventSource.CLOSED) setDriftDisabled(true);
        },
      });
    };

    const closeStreams = () => {
      streamRef.current?.close();
      driftStreamRef.current?.close();
      streamRef.current = null;
      driftStreamRef.current = null;
      setConnState("closed");
    };

    const onVisibility = () => {
      if (document.visibilityState === "visible") openStreams();
      else closeStreams();
    };

    if (document.visibilityState === "visible") openStreams();
    document.addEventListener("visibilitychange", onVisibility);

    return () => {
      document.removeEventListener("visibilitychange", onVisibility);
      closeStreams();
    };
  // Re-open when health loads (channels don't change at runtime; speed is fixed).
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [health]);

  // If only one subsystem is served, skip the overview and go straight to it.
  useEffect(() => {
    if (!health || health.channels_loaded.length === 0 || autoNavigatedRef.current) return;
    const subsystems = [...new Set(Object.values(health.channel_subsystems))];
    if (subsystems.length === 1) {
      const sub = subsystems[0];
      const channels = health.channels_loaded.filter(
        (ch) => health.channel_subsystems[ch] === sub,
      );
      setView({ kind: "subsystem", subsystem: sub, selected: channels });
      autoNavigatedRef.current = true;
    }
  }, [health]);

  function enterSubsystem(subsystem: string, channels: string[]) {
    setView({ kind: "subsystem", subsystem, selected: channels });
  }

  function backToOverview() {
    setView({ kind: "overview" });
  }

  const selected = view.kind === "subsystem" ? view.selected : [];
  const subsystem = view.kind === "subsystem" ? view.subsystem : null;
  const density = getDensity(selected.length);

  // Block the dashboard only while still loading, or when degraded with zero
  // channels ready. A degraded mission with >= 1 loaded channel is usable —
  // fall through to the dashboard (a banner notes the untrained channels).
  const ready = health?.channels_ready ?? 0;
  if (!health || health.status === "loading" || (health.status === "degraded" && ready === 0)) {
    const total = health?.channels_total ?? 0;
    const pct = total > 0 ? Math.round((ready / total) * 100) : 0;
    const mission = health?.mission ?? null;
    const isDegraded = health?.status === "degraded";

    return (
      <div className="app__loading">
        <div className="app__loading-card">
          <p className="app__loading-mission">
            {mission ?? "Spacecraft Telemetry"}
          </p>
          <p className="app__loading-title">
            {isDegraded ? "No models available" : "Loading mission models"}
          </p>
          {isDegraded ? (
            <p className="app__loading-error">
              No channels loaded. Train and register at least one channel
              model before starting the server.
            </p>
          ) : (
            <>
              <div className="app__loading-track">
                <div className="app__loading-fill" style={{ width: `${pct}%` }} />
              </div>
              <p className="app__loading-label">
                {total > 0 ? `${ready} / ${total} channels` : "Initializing…"}
              </p>
            </>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      <StatusBar
        connectionState={connState}
        eventsPerSecond={evPerSec}
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
              <DriftPanel channels={selected} disabled={DRIFT_DISABLED || driftDisabled} />
            </div>
          </>
        )}
      </div>
    </div>
  );
}
