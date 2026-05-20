import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, act } from "@testing-library/react";
import AnomalyAlerts from "../components/AnomalyAlerts";
import { telemetryStore } from "../state/telemetryStore";
import type { TelemetryEvent } from "../api/types";

function makeEvent(
  channel: string,
  predicted: boolean,
  labeled = false,
  ts = "2000-01-01T00:00:00Z",
): TelemetryEvent {
  return {
    timestamp: ts,
    mission: "test",
    channel,
    value_normalized: 0,
    prediction: 0,
    residual: 0.1,
    smoothed_error: 0.2,
    threshold: 0.1,
    is_anomaly_predicted: predicted,
    is_anomaly: labeled,
  };
}

function push(event: TelemetryEvent) {
  act(() => {
    telemetryStore.push(event);
    telemetryStore.flushForTest();
  });
}

describe("AnomalyAlerts", () => {
  beforeEach(() => {
    telemetryStore.clear();
    vi.restoreAllMocks();
  });

  it("shows empty state when no alerts", () => {
    render(<AnomalyAlerts channels={["ch-a"]} />);
    expect(screen.getByText(/no alerts/i)).toBeInTheDocument();
  });

  it("records one alert per rising edge, not per anomalous tick", () => {
    render(<AnomalyAlerts channels={["ch-a"]} />);

    // Nominal tick first (false → ...)
    push(makeEvent("ch-a", false, false, "t1"));
    // Rising edge: false → true
    push(makeEvent("ch-a", true, true, "t2"));
    // Still true — should NOT generate a second alert
    push(makeEvent("ch-a", true, true, "t3"));
    push(makeEvent("ch-a", true, true, "t4"));
    // Falls back to false
    push(makeEvent("ch-a", false, false, "t5"));
    // Second rising edge: false → true
    push(makeEvent("ch-a", true, false, "t6"));

    const rows = screen.getAllByRole("row");
    // 1 header row + 2 alert rows
    expect(rows).toHaveLength(3);
  });

  it("marks ground-truth match correctly", () => {
    render(<AnomalyAlerts channels={["ch-a"]} />);

    push(makeEvent("ch-a", false, false, "t1"));
    // True positive (predicted + labeled)
    push(makeEvent("ch-a", true, true, "t2"));
    // Drop back and re-trigger as false positive
    push(makeEvent("ch-a", false, false, "t3"));
    push(makeEvent("ch-a", true, false, "t4"));

    // Alerts are newest-first: FP (t4) is row 0, TP (t2) is row 1
    const cells = screen.getAllByRole("cell");
    const gtCells = cells.filter((c) => c.textContent === "✓" || c.textContent === "✗");
    expect(gtCells[0].textContent).toBe("✗"); // FP (newest)
    expect(gtCells[1].textContent).toBe("✓"); // TP (older)
  });

  it("handles multiple channels independently", () => {
    render(<AnomalyAlerts channels={["ch-a", "ch-b"]} />);

    push(makeEvent("ch-a", false, false, "t1"));
    push(makeEvent("ch-b", false, false, "t1"));
    push(makeEvent("ch-a", true, false, "t2")); // alert on ch-a
    push(makeEvent("ch-b", true, false, "t2")); // alert on ch-b

    const rows = screen.getAllByRole("row");
    expect(rows).toHaveLength(3); // header + 2 alerts
  });

  it("caps alerts at MAX_ALERTS=50, dropping the oldest (T3)", () => {
    render(<AnomalyAlerts channels={["ch-a"]} />);

    // Generate 60 rising edges: false → true, true → false, ...
    push(makeEvent("ch-a", false, false, "seed"));
    for (let i = 0; i < 60; i++) {
      push(makeEvent("ch-a", true, false, `alert-${i}`));
      push(makeEvent("ch-a", false, false, `gap-${i}`));
    }

    // 1 header + 50 alerts (oldest 10 dropped)
    const rows = screen.getAllByRole("row");
    expect(rows).toHaveLength(51);
    // Newest alert is first (alerts are prepended); its timestamp is alert-59
    const firstCell = rows[1].querySelector("td");
    expect(firstCell?.textContent).toContain("alert-59");
  });

  it("clears alerts when the channel set changes (T3)", () => {
    const { rerender } = render(<AnomalyAlerts channels={["ch-a"]} />);

    push(makeEvent("ch-a", false, false, "t1"));
    push(makeEvent("ch-a", true, false, "t2")); // rising edge → 1 alert

    expect(screen.getAllByRole("row")).toHaveLength(2); // header + 1

    // Switch to a different channel set
    act(() => {
      rerender(<AnomalyAlerts channels={["ch-b"]} />);
    });

    expect(screen.getByText(/no alerts/i)).toBeInTheDocument();
  });
});
