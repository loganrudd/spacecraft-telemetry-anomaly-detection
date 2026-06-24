import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import MissionSwitcher from "../components/MissionSwitcher";
import type { MissionLink } from "../api/types";

const ESA: MissionLink = { id: "ESA-Mission1", label: "ESA Mission 1", url: "http://localhost:8000" };
const ISS: MissionLink = { id: "ISS", label: "ISS Live", url: "http://localhost:8001" };

describe("MissionSwitcher", () => {
  it("renders nothing when list is empty", () => {
    const { container } = render(
      <MissionSwitcher availableMissions={[]} currentMissionId="ESA-Mission1" />
    );
    expect(container.firstChild).toBeNull();
  });

  it("renders nothing when list has one entry", () => {
    const { container } = render(
      <MissionSwitcher availableMissions={[ESA]} currentMissionId="ESA-Mission1" />
    );
    expect(container.firstChild).toBeNull();
  });

  it("renders both missions when two are configured", () => {
    render(
      <MissionSwitcher availableMissions={[ESA, ISS]} currentMissionId="ESA-Mission1" />
    );
    expect(screen.getByText("ESA Mission 1")).toBeInTheDocument();
    expect(screen.getByText("ISS Live")).toBeInTheDocument();
  });

  it("renders current mission as non-clickable span", () => {
    render(
      <MissionSwitcher availableMissions={[ESA, ISS]} currentMissionId="ESA-Mission1" />
    );
    const current = screen.getByText("ESA Mission 1");
    expect(current.tagName).toBe("SPAN");
    expect(current.className).toContain("--active");
  });

  it("renders sibling mission as a link", () => {
    render(
      <MissionSwitcher availableMissions={[ESA, ISS]} currentMissionId="ESA-Mission1" />
    );
    const link = screen.getByText("ISS Live");
    expect(link.tagName).toBe("A");
    expect(link).toHaveAttribute("href", "http://localhost:8001");
  });

  it("navigates to sibling URL on click", () => {
    const assign = vi.fn();
    Object.defineProperty(window, "location", {
      value: { assign },
      writable: true,
    });

    render(
      <MissionSwitcher availableMissions={[ESA, ISS]} currentMissionId="ESA-Mission1" />
    );
    fireEvent.click(screen.getByText("ISS Live"));
    expect(assign).toHaveBeenCalledWith("http://localhost:8001");
  });
});
