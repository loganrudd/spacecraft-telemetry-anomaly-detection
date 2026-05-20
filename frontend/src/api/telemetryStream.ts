import { API_BASE, type TelemetryEvent } from "./types";

export type StreamHandle = { close: () => void };

export type OpenStreamArgs = {
  channels: string[];
  speed?: number;
  onEvent: (e: TelemetryEvent) => void;
  onError?: (err: Event) => void;
  onOpen?: () => void;
};

export function openTelemetryStream(args: OpenStreamArgs): StreamHandle {
  const params = new URLSearchParams();
  if (args.channels.length > 0) {
    params.set("channels", args.channels.join(","));
  }
  if (args.speed !== undefined) {
    params.set("speed", String(args.speed));
  }
  const url = `${API_BASE}/api/stream/telemetry?${params.toString()}`;
  const es = new EventSource(url);

  es.addEventListener("telemetry", (raw) => {
    const msg = raw as MessageEvent<string>;
    args.onEvent(JSON.parse(msg.data) as TelemetryEvent);
  });

  if (args.onOpen) es.onopen = args.onOpen;
  if (args.onError) es.onerror = args.onError;

  return { close: () => es.close() };
}
