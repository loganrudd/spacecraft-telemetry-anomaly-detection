import { useEffect, useRef, useState, useCallback } from "react";
import AnomalyAlerts from "./components/AnomalyAlerts";
import ChannelPicker from "./components/ChannelPicker";
import StatusBar from "./components/StatusBar";
import TelemetryChart from "./components/TelemetryChart";
import { openTelemetryStream } from "./api/telemetryStream";
import { telemetryStore } from "./state/telemetryStore";
import type { StreamHandle } from "./api/telemetryStream";

type ConnectionState = "connecting" | "open" | "closed" | "error";

export default function App() {
  const [selected, setSelected] = useState<string[]>([]);
  const [connState, setConnState] = useState<ConnectionState>("closed");
  const [evPerSec, setEvPerSec] = useState(0);
  const streamRef = useRef<StreamHandle | null>(null);
  const recentEventsRef = useRef<number[]>([]);

  // Track tick rate over a 5s rolling window
  const recordTick = useCallback(() => {
    const now = Date.now();
    recentEventsRef.current.push(now);
    // Keep only the last 5 seconds
    const cutoff = now - 5_000;
    recentEventsRef.current = recentEventsRef.current.filter((t) => t >= cutoff);
    setEvPerSec(recentEventsRef.current.length / 5);
  }, []);

  // Open/reconnect SSE whenever the selected channel set changes
  useEffect(() => {
    // Close the previous connection
    if (streamRef.current) {
      streamRef.current.close();
      streamRef.current = null;
    }

    if (selected.length === 0) {
      setConnState("closed");
      return;
    }

    setConnState("connecting");
    telemetryStore.clear();

    const handle = openTelemetryStream({
      channels: selected,
      onEvent: (e) => {
        telemetryStore.push(e);
        recordTick();
      },
      onOpen: () => setConnState("open"),
      onError: () => setConnState("error"),
    });

    streamRef.current = handle;

    return () => {
      handle.close();
      streamRef.current = null;
    };
  }, [selected, recordTick]);

  return (
    <div className="app">
      <StatusBar connectionState={connState} eventsPerSecond={evPerSec} />

      <div className="app__body">
        <ChannelPicker selected={selected} onChange={setSelected} />

        <main className="app__main">
          {selected.length === 0 ? (
            <div className="app__empty">
              <p>Select one or more channels from the left panel to begin.</p>
            </div>
          ) : (
            <div className="app__charts">
              {selected.map((ch) => (
                <TelemetryChart key={ch} channel={ch} />
              ))}
            </div>
          )}
        </main>

        <AnomalyAlerts channels={selected} />
      </div>
    </div>
  );
}
