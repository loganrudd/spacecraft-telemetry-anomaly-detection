export type TelemetryEvent = {
  timestamp: string;
  mission: string;
  channel: string;
  value_normalized: number;
  prediction: number | null;
  residual: number | null;
  smoothed_error: number | null;
  threshold: number | null;
  is_anomaly_predicted: boolean;
  is_anomaly: boolean;
};

export type HealthResponse = {
  status: "ok" | "degraded" | "loading";
  mission: string;
  subsystems: string[] | null;
  channels_loaded: string[];
  channels_total: number;   // target count; use for progress bar denominator
  channels_ready: number;   // loaded so far; equals len(channels_loaded) when ok
  channel_subsystems: Record<string, string>;
  uptime_s: number;
  mlflow_tracking_uri: string;
  replay_tick_ms: number; // wall-clock ms between ticks — drives the jitter buffer release rate
};

export type DriftFeature = {
  feature: string;
  score: number;
  drifted: boolean;
};

export type DriftEvent = {
  timestamp: string;
  mission: string;
  channel: string;
  features: DriftFeature[];
  percent_drifted: number;
  drifted: boolean;
  subsystem_percent_drifted: number | null;
  subsystem_alert: boolean | null;
};

export const API_BASE: string = import.meta.env.VITE_API_BASE_URL ?? "";
