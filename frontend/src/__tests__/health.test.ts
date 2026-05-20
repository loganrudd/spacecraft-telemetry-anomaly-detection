import { describe, it, expect, vi, beforeEach } from "vitest";
import { fetchHealth } from "../api/health";
import type { HealthResponse } from "../api/types";

const SAMPLE_HEALTH: HealthResponse = {
  status: "ok",
  mission: "ESA-Mission1",
  subsystem: "subsystem_6",
  channels_loaded: ["channel_12", "channel_13"],
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

  it("throws when the server returns a non-200 status", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(null, { status: 503 }),
    );

    await expect(fetchHealth()).rejects.toThrow("503");
  });

  it("returned type has channels_loaded as an array", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(JSON.stringify(SAMPLE_HEALTH), { status: 200 }),
    );

    const result = await fetchHealth();
    expect(Array.isArray(result.channels_loaded)).toBe(true);
  });
});
