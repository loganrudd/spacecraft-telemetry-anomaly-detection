import { useEffect, useRef, useState } from "react";
import AnomalyAlerts from "./components/AnomalyAlerts";
import ChannelPicker from "./components/ChannelPicker";
import StatusBar from "./components/StatusBar";
import TelemetryChart from "./components/TelemetryChart";
import { openTelemetryStream } from "./api/telemetryStream";
import { telemetryStore } from "./state/telemetryStore";
import type { StreamHandle } from "./api/telemetryStream";

type ConnectionState = "connecting" | "open" | "closed" | "error";

const SPEED_OPTIONS = [10, 100, 1000] as const;
type Speed = (typeof SPEED_OPTIONS)[number];

export default function App() {
  const [selected, setSelected] = useState<string[]>([]);
  const [connState, setConnState] = useState<ConnectionState>("closed");
  const [speed, setSpeed] = useState<Speed>(100);
  const [evPerSec, setEvPerSec] = useState(0);
  const streamRef = useRef<StreamHandle | null>(null);
  // P2: count events in the current 1s bucket; a setInterval drains it at 1Hz.
  const tickCountRef = useRef(0);

  // Publish tick rate at 1Hz instead of on every event to avoid cascading
  // re-renders in AnomalyAlerts and ChannelPicker on every SSE message.
  useEffect(() => {
    const id = setInterval(() => {
      setEvPerSec(tickCountRef.current);
      tickCountRef.current = 0;
    }, 1_000);
    return () => clearInterval(id);
  }, []);

  // Open/reconnect SSE whenever channels or speed changes.
  useEffect(() => {
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
      speed,
      onEvent: (e) => {
        telemetryStore.push(e);
        tickCountRef.current += 1;
      },
      onOpen: () => setConnState("open"),
      onError: () => setConnState("error"),
    });

    streamRef.current = handle;

    return () => {
      handle.close();
      streamRef.current = null;
    };
  }, [selected, speed]);

  return (
    <div className="app">
      <StatusBar
        connectionState={connState}
        eventsPerSecond={evPerSec}
        speed={speed}
        onSpeedChange={setSpeed}
      />

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
