import { useEffect, useReducer, useState } from "react";
import { telemetryStore } from "../state/telemetryStore";
import { driftStore } from "../state/driftStore";

export type ChannelStatus = {
  anomaly: boolean;   // is_anomaly_predicted in the last 60 seconds
  drifted: boolean;   // most recent DriftEvent.drifted
  lastUpdated: number | null; // epoch ms of the latest telemetry tick
};

export type SubsystemRollup = Map<string, Map<string, ChannelStatus>>;

const ANOMALY_TTL_MS = 60_000;

function computeRollup(
  channelSubsystems: Record<string, string>,
  nowMs: number,
): SubsystemRollup {
  const rollup: SubsystemRollup = new Map();

  for (const [ch, sub] of Object.entries(channelSubsystems)) {
    if (!rollup.has(sub)) rollup.set(sub, new Map());

    const lastAnomalyAt = telemetryStore.lastAnomalyAtMs[ch] ?? null;
    const anomaly =
      lastAnomalyAt !== null && nowMs - lastAnomalyAt < ANOMALY_TTL_MS;

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
  // Force a re-render whenever either store notifies — useSyncExternalStore
  // requires getSnapshot to return a new reference on every change, which a
  // "__noop__" sentinel key cannot do. useReducer is simpler and correct here.
  const [, forceUpdate] = useReducer((n: number) => n + 1, 0);
  useEffect(() => {
    const unsub1 = telemetryStore.subscribe(forceUpdate);
    const unsub2 = driftStore.subscribe(forceUpdate);
    return () => {
      unsub1();
      unsub2();
    };
  }, []);

  // 1 Hz tick to clear expired anomaly badges without waiting for a new SSE event.
  const [nowMs, setNowMs] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNowMs(Date.now()), 1_000);
    return () => clearInterval(id);
  }, []);

  return computeRollup(channelSubsystems, nowMs);
}
