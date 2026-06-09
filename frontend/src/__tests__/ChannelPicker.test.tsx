import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import ChannelPicker from "../components/ChannelPicker";
import { formatChannel } from "../utils/formatChannel";

const CHANNELS = ["channel_12", "channel_13", "channel_14"];

describe("ChannelPicker", () => {
  it("renders all channels in the list", () => {
    render(<ChannelPicker allChannels={CHANNELS} selected={[]} onChange={() => {}} />);
    expect(screen.getByText(formatChannel("channel_12"))).toBeInTheDocument();
    expect(screen.getByText(formatChannel("channel_13"))).toBeInTheDocument();
    expect(screen.getByText(formatChannel("channel_14"))).toBeInTheDocument();
  });

  it("marks selected channels as active", () => {
    render(
      <ChannelPicker
        allChannels={CHANNELS}
        selected={["channel_12"]}
        onChange={() => {}}
      />,
    );
    const item = screen.getByText(formatChannel("channel_12")).closest("li");
    expect(item?.className).toContain("channel-picker__item--active");
  });

  it("calls onChange with added channel when an unselected channel is clicked", () => {
    const onChange = vi.fn();
    render(
      <ChannelPicker allChannels={CHANNELS} selected={[]} onChange={onChange} />,
    );
    fireEvent.click(screen.getByText(formatChannel("channel_12")));
    expect(onChange).toHaveBeenCalledWith(["channel_12"]);
  });

  it("calls onChange removing channel when a selected channel is clicked", () => {
    const onChange = vi.fn();
    render(
      <ChannelPicker
        allChannels={CHANNELS}
        selected={["channel_12"]}
        onChange={onChange}
      />,
    );
    fireEvent.click(screen.getByText(formatChannel("channel_12")));
    expect(onChange).toHaveBeenCalledWith([]);
  });

  it("select-all checkbox selects all channels", () => {
    const onChange = vi.fn();
    render(
      <ChannelPicker allChannels={CHANNELS} selected={[]} onChange={onChange} />,
    );
    fireEvent.click(screen.getByLabelText("Select all channels"));
    expect(onChange).toHaveBeenCalledWith(CHANNELS);
  });

  it("select-all checkbox deselects all when all are selected", () => {
    const onChange = vi.fn();
    render(
      <ChannelPicker allChannels={CHANNELS} selected={CHANNELS} onChange={onChange} />,
    );
    fireEvent.click(screen.getByLabelText("Select all channels"));
    expect(onChange).toHaveBeenCalledWith([]);
  });
});
