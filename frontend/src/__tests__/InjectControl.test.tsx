import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import InjectControl from "../components/InjectControl";

describe("InjectControl", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("renders fault type selector, duration input, and inject button", () => {
    render(<InjectControl />);
    expect(screen.getByRole("button", { name: /inject fault/i })).toBeInTheDocument();
    expect(screen.getByRole("combobox")).toBeInTheDocument();
    expect(screen.getByLabelText(/duration/i)).toBeInTheDocument();
  });

  it("shows magnitude input for spike and hides it for flatline", () => {
    render(<InjectControl />);
    // Spike is default — magnitude visible
    expect(screen.getByLabelText(/magnitude/i)).toBeInTheDocument();

    // Switch to flatline — magnitude hidden
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "flatline" } });
    expect(screen.queryByLabelText(/magnitude/i)).toBeNull();
  });

  it("shows magnitude input for drift", () => {
    render(<InjectControl />);
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "drift" } });
    expect(screen.getByLabelText(/magnitude/i)).toBeInTheDocument();
  });

  it("calls fetch with correct payload on click", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) });
    vi.stubGlobal("fetch", fetchMock);

    render(<InjectControl />);
    fireEvent.click(screen.getByRole("button"));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledOnce());
    const [url, opts] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toContain("/api/inject");
    const body = JSON.parse(opts.body as string);
    expect(body.fault_type).toBe("spike");
    expect(body.magnitude_sigma).toBeGreaterThan(0);
    expect(body.duration_ticks).toBeGreaterThan(0);
  });

  it("disables button while injecting", async () => {
    let resolve!: () => void;
    const pending = new Promise<Response>((r) => {
      resolve = () => r({ ok: true } as Response);
    });
    vi.stubGlobal("fetch", vi.fn().mockReturnValue(pending));

    render(<InjectControl />);
    const btn = screen.getByRole("button");
    fireEvent.click(btn);
    expect(btn).toBeDisabled();
    resolve();
  });

  it("shows done state after successful inject", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true }));

    render(<InjectControl />);
    fireEvent.click(screen.getByRole("button"));

    await waitFor(() =>
      expect(screen.getByRole("button")).toHaveTextContent("✓ Injected")
    );
  });

  it("shows error state on failed inject", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, text: async () => "err" }));

    render(<InjectControl />);
    fireEvent.click(screen.getByRole("button"));

    await waitFor(() =>
      expect(screen.getByRole("button")).toHaveTextContent("✗ Failed")
    );
  });
});
