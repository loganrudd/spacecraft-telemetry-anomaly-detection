import { useState } from "react";
import { API_BASE } from "../api/types";

type FaultType = "spike" | "drift" | "flatline";
type InjStatus = "idle" | "injecting" | "done" | "error";

const FAULT_LABELS: Record<FaultType, string> = {
  spike: "Spike",
  drift: "Drift",
  flatline: "Flatline",
};

// Per-fault defaults, verified against the post-retune ISS scoring regime
// (threshold_window=250, threshold_z=3, K<=4) on the stationary demo channels
// (thermal loops + power voltage). Magnitudes sit at the top of the 3-5 sigma
// range the injection profiles calibrate on, so the demo shows faults the
// detector was actually tuned for rather than cartoonishly large ones.
// - spike: a short sustained burst. 6 sigma / 8 ticks fires at the K-limited
//   floor (~4th bucket) on every demo channel; higher magnitude or duration
//   buys nothing. A single-sample spike is undetectable by design (K needs
//   >=4 consecutive threshold crossings), so "spike" is a brief burst.
// - drift: 5 sigma ramp over half the duration, then hold. 20 ticks (10 min)
//   detects on all demo channels; slower on power voltage (higher eclipse-
//   transition noise floor), but the ramp is visible on the chart throughout,
//   so late detection still reads well.
// - flatline: WEAK by design and duration cannot fix it. A forecaster tracks a
//   frozen signal (flat window -> flat prediction -> ~0 residual), so flatline
//   does not reliably fire on the stationary demo channels. Kept for
//   completeness; detecting it needs a value-space variance rule (out of
//   scope). Magnitude is unused for flatline.
const DEFAULTS: Record<FaultType, { magnitude: number; durationTicks: number }> = {
  drift:    { magnitude: 5, durationTicks: 20 },
  spike:    { magnitude: 6, durationTicks: 8 },
  flatline: { magnitude: 5, durationTicks: 40 },
};

/**
 * On-demand fault injection control for the ISS live demo.
 * Calls POST /api/inject which queues the fault on the shared replay loop
 * so every connected SSE subscriber sees the same anomaly simultaneously.
 */
export default function InjectControl() {
  const [faultType, setFaultType] = useState<FaultType>("drift");
  const [magnitude, setMagnitude] = useState(DEFAULTS.drift.magnitude);
  const [durationTicks, setDurationTicks] = useState(DEFAULTS.drift.durationTicks);
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
            onChange={(e) => {
              const ft = e.target.value as FaultType;
              setFaultType(ft);
              setMagnitude(DEFAULTS[ft].magnitude);
              setDurationTicks(DEFAULTS[ft].durationTicks);
            }}
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
