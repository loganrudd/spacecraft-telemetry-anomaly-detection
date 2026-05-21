import { describe, it, expect, beforeEach } from "vitest";
import { render, screen, act } from "@testing-library/react";
import DriftPanel from "../components/DriftPanel";
import { driftStore } from "../state/driftStore";
import type { DriftEvent } from "../api/types";

function makeEvent(channel: string, pct: number, subPct?: number): DriftEvent {
  return {
    timestamp: "2000-01-01T00:00:00Z",
    mission: "test",
    channel,
    features: [
      { feature: "value_normalized", score: pct, drifted: pct > 0.1 },
      { feature: "rolling_mean_10", score: 0.0, drifted: false },
    ],
    percent_drifted: pct,
    drifted: pct >= 0.3,
    subsystem_percent_drifted: subPct ?? null,
    subsystem_alert: (subPct ?? 0) >= 0.3,
  };
}

function push(event: DriftEvent) {
  act(() => {
    driftStore.push(event);
    driftStore.flushForTest();
  });
}

describe("DriftPanel", () => {
  beforeEach(() => {
    driftStore.clear();
  });

  it("shows disabled state when disabled=true", () => {
    render(<DriftPanel channels={["ch-a"]} disabled />);
    expect(screen.getByText(/drift monitoring disabled/i)).toBeInTheDocument();
  });

  it("shows waiting state for channels with no event yet", () => {
    render(<DriftPanel channels={["ch-a"]} />);
    expect(screen.getByText("ch-a")).toBeInTheDocument();
    // Both the channel row and the subsystem gauge show "waiting" text;
    // assert at least one element with that text exists.
    expect(screen.getAllByText(/waiting/i).length).toBeGreaterThanOrEqual(1);
  });

  it("shows empty state when no channels selected", () => {
    render(<DriftPanel channels={[]} />);
    expect(screen.getByText(/no channels selected/i)).toBeInTheDocument();
  });

  it("shows DRIFT badge when channel is drifted", () => {
    render(<DriftPanel channels={["ch-a"]} />);
    push(makeEvent("ch-a", 0.5)); // pct >= 0.3 → drifted: true
    expect(screen.getByText("DRIFT")).toBeInTheDocument();
  });

  it("shows nominal badge when channel is not drifted", () => {
    render(<DriftPanel channels={["ch-a"]} />);
    push(makeEvent("ch-a", 0.0)); // pct < 0.3 → drifted: false
    expect(screen.getByText("nominal")).toBeInTheDocument();
  });

  it("subsystem gauge is absent before first subsystem summary", () => {
    render(<DriftPanel channels={["ch-a"]} />);
    push(makeEvent("ch-a", 0.1)); // no subPct
    expect(screen.getByText(/waiting for subsystem/i)).toBeInTheDocument();
  });

  it("subsystem gauge shows percent when summary arrives", () => {
    render(<DriftPanel channels={["ch-a"]} />);
    push(makeEvent("ch-a", 0.4, 0.4));
    expect(screen.getByText(/40% channels drifting/i)).toBeInTheDocument();
  });

  it("gauge has alert class at or above 30% threshold", () => {
    render(<DriftPanel channels={["ch-a"]} />);
    push(makeEvent("ch-a", 0.35, 0.35));
    const gauge = screen.getByLabelText("Subsystem drift gauge");
    expect(gauge.className).toContain("drift-panel__gauge--alert");
  });

  it("gauge does not have alert class below 30% threshold", () => {
    render(<DriftPanel channels={["ch-a"]} />);
    push(makeEvent("ch-a", 0.1, 0.1));
    const gauge = screen.getByLabelText("Subsystem drift gauge");
    expect(gauge.className).not.toContain("drift-panel__gauge--alert");
  });

  it("renders feature bars for each channel", () => {
    render(<DriftPanel channels={["ch-a"]} />);
    push(makeEvent("ch-a", 0.5));
    expect(screen.getByLabelText("Feature drift scores")).toBeInTheDocument();
  });

  it("shows 'since HH:MM:SS' timestamp when channel is drifted", () => {
    render(<DriftPanel channels={["ch-a"]} />);
    push(makeEvent("ch-a", 0.5)); // drifted: true, timestamp: "2000-01-01T00:00:00Z"
    expect(screen.getByText(/since 2000-01-01 00:00:00/)).toBeInTheDocument();
  });

  it("does not update the drift timestamp when subsequent drifted events arrive", () => {
    render(<DriftPanel channels={["ch-a"]} />);
    push(makeEvent("ch-a", 0.5)); // first onset: "2000-01-01T00:00:00Z"
    act(() => {
      driftStore.push({ ...makeEvent("ch-a", 0.6), timestamp: "2000-01-01T00:01:00Z" });
      driftStore.flushForTest();
    });
    // Still shows the first onset time, not the updated event timestamp.
    expect(screen.getByText(/since 2000-01-01 00:00:00/)).toBeInTheDocument();
    expect(screen.queryByText(/since 2000-01-01 00:01:00/)).not.toBeInTheDocument();
  });

  it("shows plain event timestamp when channel is nominal", () => {
    render(<DriftPanel channels={["ch-a"]} />);
    push(makeEvent("ch-a", 0.0)); // drifted: false
    expect(screen.getByText("2000-01-01 00:00:00")).toBeInTheDocument();
  });
});
