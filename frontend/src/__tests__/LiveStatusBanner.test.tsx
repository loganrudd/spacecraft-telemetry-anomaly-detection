import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import LiveStatusBanner from "../components/LiveStatusBanner";

describe("LiveStatusBanner", () => {
  it("renders LIVE when status=live", () => {
    render(<LiveStatusBanner status="live" />);
    expect(screen.getByText(/LIVE/)).toBeDefined();
  });

  it("renders nothing when status=closed", () => {
    const { container } = render(<LiveStatusBanner status="closed" />);
    expect(container.firstChild).toBeNull();
  });

  it("renders nothing when status=connecting", () => {
    const { container } = render(<LiveStatusBanner status="connecting" />);
    expect(container.firstChild).toBeNull();
  });

  it("renders LOS message when status=los", () => {
    render(<LiveStatusBanner status="los" />);
    expect(screen.getByText(/Signal lost/i)).toBeDefined();
  });

  it("includes ETA estimate when expectedResumeInS is provided", () => {
    render(<LiveStatusBanner status="los" expectedResumeInS={240} />);
    const el = screen.getByText(/Signal lost/i);
    // 240s → ~4 min
    expect(el.textContent).toMatch(/~4 min/);
  });

  it("shows ~1 min when expectedResumeInS <= 60", () => {
    render(<LiveStatusBanner status="los" expectedResumeInS={45} />);
    expect(screen.getByText(/~1 min/i)).toBeDefined();
  });

  it("omits ETA when expectedResumeInS is undefined", () => {
    render(<LiveStatusBanner status="los" />);
    const el = screen.getByText(/Signal lost/i);
    expect(el.textContent).not.toMatch(/min/);
  });

  it("live banner has live-banner--live class", () => {
    render(<LiveStatusBanner status="live" />);
    const el = screen.getByRole("status");
    expect(el.className).toContain("live-banner--live");
  });

  it("los banner has live-banner--los class", () => {
    render(<LiveStatusBanner status="los" />);
    const el = screen.getByRole("alert");
    expect(el.className).toContain("live-banner--los");
  });

  it("labels the replay fallback when mode=replay", () => {
    render(<LiveStatusBanner status="los" mode="replay" />);
    const el = screen.getByText(/Signal lost/i);
    expect(el.textContent).toMatch(/showing recent recorded data/i);
  });

  it("omits the replay label when mode is not replay", () => {
    render(<LiveStatusBanner status="los" />);
    const el = screen.getByText(/Signal lost/i);
    expect(el.textContent).not.toMatch(/recorded data/i);
  });

  it("includes both the replay label and ETA when both are provided", () => {
    render(<LiveStatusBanner status="los" mode="replay" expectedResumeInS={240} />);
    const el = screen.getByText(/Signal lost/i);
    expect(el.textContent).toMatch(/showing recent recorded data/i);
    expect(el.textContent).toMatch(/~4 min/);
  });
});
