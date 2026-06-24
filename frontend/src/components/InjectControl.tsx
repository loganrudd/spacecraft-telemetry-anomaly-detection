import { useState } from "react";
import { API_BASE } from "../api/types";

type FaultType = "spike" | "drift" | "flatline";
type InjStatus = "idle" | "injecting" | "done" | "error";

const FAULT_LABELS: Record<FaultType, string> = {
  spike: "Spike",
  drift: "Drift",
  flatline: "Flatline",
};

/**
 * On-demand fault injection control for the ISS live demo.
 * Calls POST /api/inject which queues the fault on the shared replay loop
 * so every connected SSE subscriber sees the same anomaly simultaneously.
 */
export default function InjectControl() {
  const [faultType, setFaultType] = useState<FaultType>("spike");
  const [magnitude, setMagnitude] = useState(3.0);
  const [durationTicks, setDurationTicks] = useState(10);
  const [status, setStatus] = useState<InjStatus>("idle");

  const isFlatline = faultType === "flatline";

  async function handleInject() {
    setStatus("injecting");
    try {
      const res = await fetch(`${API_BASE}/api/inject`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          fault_type: faultType,
          channels: [],
          magnitude_sigma: magnitude,
          duration_ticks: durationTicks,
        }),
      });
      if (!res.ok) {
        const err = await res.text();
        console.error("inject failed", res.status, err);
        setStatus("error");
      } else {
        setStatus("done");
      }
    } catch (e) {
      console.error("inject error", e);
      setStatus("error");
    } finally {
      setTimeout(() => setStatus("idle"), 2_500);
    }
  }

  const btnLabel =
    status === "injecting" ? "Injecting…"
    : status === "done"      ? "✓ Injected"
    : status === "error"     ? "✗ Failed"
    : "Inject Fault";

  return (
    <div className="inject-control">
      <div className="inject-control__header">
        <span className="inject-control__title">Inject Fault</span>
      </div>

      <div className="inject-control__body">
        <div className="inject-control__row">
          <label className="inject-control__label">Type</label>
          <select
            className="inject-control__select"
            value={faultType}
            onChange={(e) => setFaultType(e.target.value as FaultType)}
          >
            {(["spike", "drift", "flatline"] as FaultType[]).map((ft) => (
              <option key={ft} value={ft}>
                {FAULT_LABELS[ft]}
              </option>
            ))}
          </select>
        </div>

        {!isFlatline && (
          <div className="inject-control__row">
            <label className="inject-control__label" htmlFor="inject-magnitude">
              Magnitude (σ)
            </label>
            <input
              id="inject-magnitude"
              className="inject-control__input"
              type="number"
              min={0.1}
              max={10}
              step={0.5}
              value={magnitude}
              onChange={(e) => setMagnitude(Number(e.target.value))}
            />
          </div>
        )}

        <div className="inject-control__row">
          <label className="inject-control__label" htmlFor="inject-duration">
            Duration (ticks)
          </label>
          <input
            id="inject-duration"
            className="inject-control__input"
            type="number"
            min={1}
            max={200}
            step={1}
            value={durationTicks}
            onChange={(e) => setDurationTicks(Number(e.target.value))}
          />
        </div>

        <button
          className={`inject-control__btn inject-control__btn--${status}`}
          onClick={handleInject}
          disabled={status === "injecting"}
        >
          {btnLabel}
        </button>
      </div>
    </div>
  );
}
