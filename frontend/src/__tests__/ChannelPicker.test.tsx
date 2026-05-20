import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import ChannelPicker from "../components/ChannelPicker";
import type { HealthResponse } from "../api/types";
import { fetchHealth } from "../api/health";

vi.mock("../api/health");

const mockFetchHealth = vi.mocked(fetchHealth);

const HEALTH: HealthResponse = {
  status: "ok",
  mission: "ESA-Mission1",
  subsystem: "subsystem_6",
  channels_loaded: ["channel_12", "channel_13", "channel_14"],
  uptime_s: 10,
  mlflow_tracking_uri: "sqlite:///mlflow.db",
};

describe("ChannelPicker", () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it("shows loading state initially", () => {
    mockFetchHealth.mockReturnValue(new Promise(() => {})); // never resolves
    render(<ChannelPicker selected={[]} onChange={() => {}} />);
    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });

  it("renders channel list after health loads", async () => {
    mockFetchHealth.mockResolvedValueOnce(HEALTH);
    render(<ChannelPicker selected={[]} onChange={() => {}} />);
    await waitFor(() =>
      expect(screen.getByText("channel_12")).toBeInTheDocument(),
    );
    expect(screen.getByText("channel_13")).toBeInTheDocument();
    expect(screen.getByText("channel_14")).toBeInTheDocument();
  });

  it("shows mission and subsystem in header", async () => {
    mockFetchHealth.mockResolvedValueOnce(HEALTH);
    render(<ChannelPicker selected={[]} onChange={() => {}} />);
    await waitFor(() =>
      expect(screen.getByText("ESA-Mission1")).toBeInTheDocument(),
    );
    expect(screen.getByText("subsystem_6")).toBeInTheDocument();
  });

  it("calls onChange when a channel is clicked", async () => {
    mockFetchHealth.mockResolvedValueOnce(HEALTH);
    const onChange = vi.fn();
    render(<ChannelPicker selected={[]} onChange={onChange} />);
    await waitFor(() =>
      expect(screen.getByText("channel_12")).toBeInTheDocument(),
    );
    fireEvent.click(screen.getByText("channel_12"));
    expect(onChange).toHaveBeenCalledWith(["channel_12"]);
  });

  it("shows performance warning when >5 channels selected", async () => {
    mockFetchHealth.mockResolvedValueOnce({
      ...HEALTH,
      channels_loaded: ["ch-1", "ch-2", "ch-3", "ch-4", "ch-5", "ch-6", "ch-7"],
    });
    render(
      <ChannelPicker
        selected={["ch-1", "ch-2", "ch-3", "ch-4", "ch-5", "ch-6"]}
        onChange={() => {}}
      />,
    );
    await waitFor(() =>
      expect(screen.getByRole("alert")).toBeInTheDocument(),
    );
  });

  it("shows error message when fetchHealth fails", async () => {
    mockFetchHealth.mockRejectedValueOnce(new Error("503 Service Unavailable"));
    render(<ChannelPicker selected={[]} onChange={() => {}} />);
    await waitFor(() =>
      expect(screen.getByText(/503/)).toBeInTheDocument(),
    );
  });
});
