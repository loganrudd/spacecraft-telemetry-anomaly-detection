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
  status: "ok" | "degraded";
  mission: string;
  subsystem: string;
  channels_loaded: string[];
  uptime_s: number;
  mlflow_tracking_uri: string;
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
  subsystem_alert: boolean;
};

export const API_BASE: string = import.meta.env.VITE_API_BASE_URL ?? "";
