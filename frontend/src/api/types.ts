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

export type MissionLink = {
  id: string;    // mission identifier, e.g. "ISS"
  label: string; // display name, e.g. "ISS Live"
  url: string;   // root URL of the sibling service
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
  available_missions: MissionLink[];
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

/** Per-tick raw event from the live Lightstreamer pump (event: raw). */
export type RawTelemetryEvent = {
  timestamp: string;
  channel: string;
  value_normalized: number;
};

/** Mission-wide status event emitted on LOS onset and recovery (event: status). */
export type StatusEvent = {
  type: "los" | "resumed";
  mode?: string;
  expected_resume_in_s?: number;
};

export const API_BASE: string = import.meta.env.VITE_API_BASE_URL ?? "";
