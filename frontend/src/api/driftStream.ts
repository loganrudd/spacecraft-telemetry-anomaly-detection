import { API_BASE, type DriftEvent } from "./types";

export type DriftStreamHandle = { close: () => void };

export type OpenDriftStreamArgs = {
  channels: string[];
  onEvent: (e: DriftEvent) => void;
  onError?: (err: Event) => void;
  onOpen?: () => void;
};

export function openDriftStream(args: OpenDriftStreamArgs): DriftStreamHandle {
  const params = new URLSearchParams();
  if (args.channels.length > 0) {
    params.set("channels", args.channels.join(","));
  }
  const url = `${API_BASE}/api/stream/drift?${params.toString()}`;
  const es = new EventSource(url);

  es.addEventListener("drift", (raw) => {
    const msg = raw as MessageEvent<string>;
    args.onEvent(JSON.parse(msg.data) as DriftEvent);
  });

  if (args.onOpen) es.onopen = args.onOpen;
  if (args.onError) es.onerror = args.onError;

  return { close: () => es.close() };
}
