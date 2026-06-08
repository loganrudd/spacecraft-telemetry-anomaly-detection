import { useEffect, useReducer, useRef } from "react";
import { telemetryStore, CHART_WINDOW } from "../state/telemetryStore";
import { driftStore } from "../state/driftStore";

export type ChannelStatus = {
  anomaly: boolean;   // is_anomaly_predicted within the current chart window
  drifted: boolean;   // most recent DriftEvent.drifted
  lastUpdated: number | null; // epoch ms of the latest telemetry tick
};

export type SubsystemRollup = Map<string, Map<string, ChannelStatus>>;
// Throttle rollup rebuilds to ~4 Hz. TelemetryStore notifies at rAF rate (~60 Hz);
// without throttling every tick triggers a full Map rebuild across all channels.
const ROLLUP_THROTTLE_MS = 250;

function computeRollup(
  channelSubsystems: Record<string, string>,
): SubsystemRollup {
  const rollup: SubsystemRollup = new Map();

  for (const [ch, sub] of Object.entries(channelSubsystems)) {
    if (!rollup.has(sub)) rollup.set(sub, new Map());

    // Badge is lit while the most recent predicted-anomaly event is still
    // within the chart window — same boundary as the alerts panel.
    const lastPush = telemetryStore.lastAnomalyPushCount[ch] ?? null;
    const anomaly =
      lastPush !== null &&
      telemetryStore.pushCount(ch) - lastPush < CHART_WINDOW;

    const latestDrift = driftStore.latestForChannel(ch);
    const drifted = latestDrift?.drifted ?? false;

    const lastUpdated = telemetryStore.lastTickAtMs[ch] ?? null;

    rollup.get(sub)!.set(ch, { anomaly, drifted, lastUpdated });
  }

  return rollup;
}

export function useSubsystemRollup(
  channelSubsystems: Record<string, string>,
): SubsystemRollup {
  const [, forceUpdate] = useReducer((n: number) => n + 1, 0);
  const pendingRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    // Trailing-edge throttle: schedule one rebuild per ROLLUP_THROTTLE_MS window.
    // Suppresses the ~60 Hz rAF flush from TelemetryStore down to ~4 Hz.
    const throttled = () => {
      if (pendingRef.current === null) {
        pendingRef.current = setTimeout(() => {
          pendingRef.current = null;
          forceUpdate();
        }, ROLLUP_THROTTLE_MS);
      }
    };

    const unsub1 = telemetryStore.subscribe(throttled);
    const unsub2 = driftStore.subscribe(throttled);
    // 1 Hz tick ensures badges clear in overview mode when no new events push
    // the count forward (e.g. stream paused or very slow).
    const id = setInterval(throttled, 1_000);

    return () => {
      unsub1();
      unsub2();
      clearInterval(id);
      if (pendingRef.current !== null) clearTimeout(pendingRef.current);
    };
  }, []); // forceUpdate and pendingRef are stable across renders

  return computeRollup(channelSubsystems);
}
