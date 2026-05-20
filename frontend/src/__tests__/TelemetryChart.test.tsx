import { describe, it, expect, beforeEach } from "vitest";
import { render, screen, act } from "@testing-library/react";
import TelemetryChart from "../components/TelemetryChart";
import { telemetryStore } from "../state/telemetryStore";
import type { TelemetryEvent } from "../api/types";

function makeEvent(channel: string): TelemetryEvent {
  return {
    timestamp: "2000-01-01T00:00:00Z",
    mission: "test",
    channel,
    value_normalized: 0.5,
    prediction: 0.4,
    residual: 0.1,
    smoothed_error: 0.2,
    threshold: 0.5,
    is_anomaly_predicted: false,
    is_anomaly: false,
  };
}

describe("TelemetryChart", () => {
  beforeEach(() => {
    telemetryStore.clear();
  });

  it("renders the channel name as a heading", () => {
    render(<TelemetryChart channel="sensor-v1" />);
    expect(screen.getByRole("heading", { name: "sensor-v1" })).toBeInTheDocument();
  });

  it("shows waiting message when no data is buffered", () => {
    render(<TelemetryChart channel="ch-a" />);
    expect(screen.getByText(/waiting for data/i)).toBeInTheDocument();
  });

  it("hides waiting message after a push is flushed", () => {
    render(<TelemetryChart channel="ch-a" />);
    expect(screen.getByText(/waiting for data/i)).toBeInTheDocument();

    act(() => {
      telemetryStore.push(makeEvent("ch-a"));
      telemetryStore.flushForTest();
    });

    expect(screen.queryByText(/waiting for data/i)).not.toBeInTheDocument();
  });
});
