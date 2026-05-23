import { describe, it, expect, vi, beforeEach } from "vitest";
import { fetchHealth } from "../api/health";
import type { HealthResponse } from "../api/types";

const SAMPLE_HEALTH: HealthResponse = {
  status: "ok",
  mission: "ESA-Mission1",
  subsystem: "subsystem_6",
  channels_loaded: ["channel_12", "channel_13"],
  channels_total: 2,
  channels_ready: 2,
  channel_subsystems: { channel_12: "subsystem_6", channel_13: "subsystem_6" },
  uptime_s: 42.5,
  mlflow_tracking_uri: "sqlite:///mlflow.db",
};

describe("fetchHealth", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("calls /health and returns parsed JSON", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(JSON.stringify(SAMPLE_HEALTH), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const result = await fetchHealth();

    expect(fetchSpy).toHaveBeenCalledOnce();
    const calledUrl = fetchSpy.mock.calls[0][0] as string;
    expect(calledUrl).toMatch(/\/health$/);
    expect(result).toEqual(SAMPLE_HEALTH);
  });

  it("parses and returns body even on non-200 status (degraded response)", async () => {
    const degraded: HealthResponse = {
      status: "degraded",
      mission: "ESA-Mission1",
      subsystem: null,
      channels_loaded: [],
      channels_total: 0,
      channels_ready: 0,
      channel_subsystems: {},
      uptime_s: 1.2,
      mlflow_tracking_uri: "",
    };
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(JSON.stringify(degraded), {
        status: 503,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const result = await fetchHealth();
    expect(result.status).toBe("degraded");
  });

  it("returned type has channels_loaded as an array", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(JSON.stringify(SAMPLE_HEALTH), { status: 200 }),
    );

    const result = await fetchHealth();
    expect(Array.isArray(result.channels_loaded)).toBe(true);
  });
});
