import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import MissionOverview from "../components/MissionOverview";
import type { ChannelStatus, SubsystemRollup } from "../hooks/useSubsystemRollup";

// Decouple MissionOverview rendering tests from rollup logic (tested separately).
vi.mock("../hooks/useSubsystemRollup", () => ({
  useSubsystemRollup: vi.fn(),
}));

import { useSubsystemRollup } from "../hooks/useSubsystemRollup";

function makeStatus(overrides: Partial<ChannelStatus> = {}): ChannelStatus {
  return { anomaly: false, drifted: false, lastUpdated: Date.now(), ...overrides };
}

function makeRollup(
  map: Record<string, Record<string, Partial<ChannelStatus>>>,
): SubsystemRollup {
  const rollup: SubsystemRollup = new Map();
  for (const [sub, channels] of Object.entries(map)) {
    const channelMap = new Map<string, ChannelStatus>();
    for (const [ch, status] of Object.entries(channels)) {
      channelMap.set(ch, makeStatus(status));
    }
    rollup.set(sub, channelMap);
  }
  return rollup;
}

const CHANNEL_SUBSYSTEMS: Record<string, string> = {
  channel_1: "subsystem_1",
  channel_2: "subsystem_1",
  channel_3: "subsystem_2",
};

const DEFAULT_ROLLUP = makeRollup({
  subsystem_1: { channel_1: {}, channel_2: {} },
  subsystem_2: { channel_3: {} },
});

describe("MissionOverview", () => {
  beforeEach(() => {
    vi.mocked(useSubsystemRollup).mockReturnValue(DEFAULT_ROLLUP);
  });

  it("renders a card for each subsystem", () => {
    render(
      <MissionOverview
        mission="ESA-Mission1"
        channelSubsystems={CHANNEL_SUBSYSTEMS}
        onEnterSubsystem={() => {}}
      />,
    );
    expect(screen.getByText("subsystem_1")).toBeInTheDocument();
    expect(screen.getByText("subsystem_2")).toBeInTheDocument();
  });

  it("shows the mission name in the header", () => {
    render(
      <MissionOverview
        mission="ESA-Mission1"
        channelSubsystems={CHANNEL_SUBSYSTEMS}
        onEnterSubsystem={() => {}}
      />,
    );
    expect(screen.getByText("ESA-Mission1")).toBeInTheDocument();
  });

  it("shows subsystem and channel counts in the header", () => {
    render(
      <MissionOverview
        mission="ESA-Mission1"
        channelSubsystems={CHANNEL_SUBSYSTEMS}
        onEnterSubsystem={() => {}}
      />,
    );
    // 2 subsystems, 3 channels total
    expect(screen.getByText(/2 subsystems/)).toBeInTheDocument();
    expect(screen.getByText(/3 channels/)).toBeInTheDocument();
  });

  it("calls onEnterSubsystem with all channels when a card is clicked", () => {
    const onEnter = vi.fn();
    render(
      <MissionOverview
        mission="ESA-Mission1"
        channelSubsystems={CHANNEL_SUBSYSTEMS}
        onEnterSubsystem={onEnter}
      />,
    );
    fireEvent.click(
      screen.getByRole("button", { name: /subsystem_2: 1 channels/ }),
    );
    expect(onEnter).toHaveBeenCalledWith("subsystem_2", ["channel_3"]);
  });

  it("dot class is anomaly when channel has anomaly", () => {
    vi.mocked(useSubsystemRollup).mockReturnValue(
      makeRollup({
        subsystem_1: { channel_1: { anomaly: true }, channel_2: {} },
        subsystem_2: { channel_3: {} },
      }),
    );
    render(
      <MissionOverview
        mission="ESA-Mission1"
        channelSubsystems={CHANNEL_SUBSYSTEMS}
        onEnterSubsystem={() => {}}
      />,
    );
    expect(screen.getByLabelText("channel_1").className).toContain(
      "subsystem-card__dot--anomaly",
    );
  });

  it("dot class is drift when channel has drift but no anomaly", () => {
    vi.mocked(useSubsystemRollup).mockReturnValue(
      makeRollup({
        subsystem_1: { channel_1: { drifted: true }, channel_2: {} },
        subsystem_2: { channel_3: {} },
      }),
    );
    render(
      <MissionOverview
        mission="ESA-Mission1"
        channelSubsystems={CHANNEL_SUBSYSTEMS}
        onEnterSubsystem={() => {}}
      />,
    );
    expect(screen.getByLabelText("channel_1").className).toContain(
      "subsystem-card__dot--drift",
    );
  });

  it("dot class is nominal when channel has data and no alerts", () => {
    render(
      <MissionOverview
        mission="ESA-Mission1"
        channelSubsystems={CHANNEL_SUBSYSTEMS}
        onEnterSubsystem={() => {}}
      />,
    );
    expect(screen.getByLabelText("channel_1").className).toContain(
      "subsystem-card__dot--nominal",
    );
  });

  it("dot class is waiting when channel has no data yet", () => {
    vi.mocked(useSubsystemRollup).mockReturnValue(
      makeRollup({
        subsystem_1: { channel_1: { lastUpdated: null }, channel_2: {} },
        subsystem_2: { channel_3: {} },
      }),
    );
    render(
      <MissionOverview
        mission="ESA-Mission1"
        channelSubsystems={CHANNEL_SUBSYSTEMS}
        onEnterSubsystem={() => {}}
      />,
    );
    expect(screen.getByLabelText("channel_1").className).toContain(
      "subsystem-card__dot--waiting",
    );
  });

  it("clicking a dot calls onEnterSubsystem with that channel only", () => {
    const onEnter = vi.fn();
    render(
      <MissionOverview
        mission="ESA-Mission1"
        channelSubsystems={CHANNEL_SUBSYSTEMS}
        onEnterSubsystem={onEnter}
      />,
    );
    fireEvent.click(screen.getByLabelText("channel_1"));
    expect(onEnter).toHaveBeenCalledWith("subsystem_1", ["channel_1"]);
  });

  it("shows anomaly badge count in summary when anomalies present", () => {
    vi.mocked(useSubsystemRollup).mockReturnValue(
      makeRollup({
        subsystem_1: {
          channel_1: { anomaly: true },
          channel_2: { anomaly: true },
        },
        subsystem_2: { channel_3: {} },
      }),
    );
    render(
      <MissionOverview
        mission="ESA-Mission1"
        channelSubsystems={CHANNEL_SUBSYSTEMS}
        onEnterSubsystem={() => {}}
      />,
    );
    expect(screen.getByText(/2 anomalies/)).toBeInTheDocument();
  });

  it("shows drift badge even when a drifted channel also has an anomaly", () => {
    vi.mocked(useSubsystemRollup).mockReturnValue(
      makeRollup({
        subsystem_1: {
          channel_1: { anomaly: true, drifted: true },
          channel_2: {},
        },
        subsystem_2: { channel_3: {} },
      }),
    );
    render(
      <MissionOverview
        mission="ESA-Mission1"
        channelSubsystems={CHANNEL_SUBSYSTEMS}
        onEnterSubsystem={() => {}}
      />,
    );
    // Both badges should be visible — drift is independent of anomaly in the summary.
    expect(screen.getByText(/1 anomaly/)).toBeInTheDocument();
    expect(screen.getByText(/drift on 1/)).toBeInTheDocument();
  });

  it("shows nominal badge when no anomalies or drift", () => {
    render(
      <MissionOverview
        mission="ESA-Mission1"
        channelSubsystems={CHANNEL_SUBSYSTEMS}
        onEnterSubsystem={() => {}}
      />,
    );
    expect(screen.getAllByText("nominal")).toHaveLength(2);
  });
});
