import { API_BASE, type HealthResponse } from "./types";

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  // Parse body regardless of HTTP status — the server always returns a
  // HealthResponse shape (status="loading"|"ok"|"degraded") even on 503.
  // Let callers inspect the status field rather than catching an exception.
  return res.json() as Promise<HealthResponse>;
}
