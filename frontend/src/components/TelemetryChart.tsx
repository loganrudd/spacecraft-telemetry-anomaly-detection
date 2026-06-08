import { memo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceArea,
  ResponsiveContainer,
  Legend,
} from "recharts";
import type { TooltipProps } from "recharts";
import { useTelemetryChannel, CHART_WINDOW } from "../state/telemetryStore";
import { collapseFlags, labeledIntervalsWithDetection } from "../utils/anomalyIntervals";
import type { DetectedInterval } from "../utils/anomalyIntervals";
import { formatChannel } from "../utils/formatChannel";
import type { TelemetryEvent } from "../api/types";

type DensityTier = "comfortable" | "compact" | "dense";

const CHART_HEIGHTS: Record<DensityTier, number> = {
  comfortable: 220,
  compact: 150,
  dense: 100,
};

// Imported from telemetryStore so AnomalyAlerts uses the same window boundary.

type Props = { channel: string; density?: DensityTier };

type TooltipExtra = { labeledIntervals: DetectedInterval[] };

function CustomTooltip({
  active,
  payload,
  labeledIntervals = [],
}: TooltipProps<number, string> & Partial<TooltipExtra>) {
  if (!active || !payload?.length) return null;
  const ev = payload[0]?.payload as TelemetryEvent | undefined;
  if (!ev) return null;

  // For a labeled point, report detection at the SEGMENT level (was this
  // labeled window flagged anywhere?), not just at this exact timestep — that
  // matches how the model is scored and avoids "not detected" reading inside a
  // window the model actually caught.
  const segment = ev.is_anomaly
    ? labeledIntervals.find(
        (iv) => iv.startTs <= ev.timestamp && ev.timestamp <= iv.endTs,
      )
    : undefined;

  return (
    <div className="chart-tooltip">
      <p className="chart-tooltip__ts">{ev.timestamp}</p>
      <p>value: {ev.value_normalized.toFixed(4)}</p>
      {ev.prediction !== null && (
        <p>prediction: {ev.prediction.toFixed(4)}</p>
      )}
      {ev.residual !== null && <p>residual: {ev.residual.toFixed(4)}</p>}
      {ev.smoothed_error !== null && (
        <p>smoothed_error: {ev.smoothed_error.toFixed(4)}</p>
      )}
      {ev.is_anomaly ? (
        <>
          <p>anomaly (labeled): yes</p>
          <p>
            detected:{" "}
            {segment?.detected ? (
              <span style={{ color: "var(--success)" }}>
                yes — flagged in this segment
              </span>
            ) : (
              "no — missed"
            )}
          </p>
          <p style={{ color: "var(--fg-muted)" }}>
            flagged at this point: {ev.is_anomaly_predicted ? "yes" : "no"}
          </p>
        </>
      ) : (
        <p>
          anomaly (predicted):{" "}
          {ev.is_anomaly_predicted ? "yes (false positive)" : "no"}
        </p>
      )}
    </div>
  );
}

function TelemetryChart({ channel, density = "comfortable" }: Props) {
  const events = useTelemetryChannel(channel);
  // Slice to the most recent CHART_WINDOW events so the X axis scale stays
  // stable as data arrives instead of squeezing all history into view.
  const visible = events.slice(-CHART_WINDOW);

  const labeledIntervals = labeledIntervalsWithDetection(visible);
  const predIntervals = collapseFlags(visible, "is_anomaly_predicted");
  const chartHeight = CHART_HEIGHTS[density];

  if (events.length === 0) {
    return (
      <section className={`telemetry-chart telemetry-chart--${density}`}>
        <h2 className="telemetry-chart__title">{formatChannel(channel)}</h2>
        <p className="telemetry-chart__waiting">Waiting for data…</p>
      </section>
    );
  }

  return (
    <section className={`telemetry-chart telemetry-chart--${density}`}>
      <h2 className="telemetry-chart__title">{channel}</h2>
      <ResponsiveContainer width="100%" height={chartHeight}>
        <LineChart
          data={visible}
          margin={{ top: 4, right: 12, left: 0, bottom: 0 }}
        >
          <XAxis
            dataKey="timestamp"
            tick={false}
            axisLine={{ stroke: "var(--border)" }}
          />
          <YAxis
            width={density === "dense" ? 36 : 52}
            tick={{ fill: "var(--fg-muted)", fontSize: density === "dense" ? 9 : 11 }}
            axisLine={{ stroke: "var(--border)" }}
          />
          {density !== "dense" && (
            <Tooltip content={<CustomTooltip labeledIntervals={labeledIntervals} />} />
          )}
          {density === "comfortable" && (
            <Legend wrapperStyle={{ fontSize: 11, color: "var(--fg-muted)" }} />
          )}

          {/* Ground-truth anomaly bands (labeled, red). A green border marks a
              window the model caught (flagged anywhere inside) — segment-level
              detection; no border means the window was missed. */}
          {labeledIntervals.map((iv, i) => (
            <ReferenceArea
              key={`true-${i}`}
              x1={iv.startTs}
              x2={iv.endTs}
              fill="var(--anomaly-true)"
              stroke={iv.detected ? "var(--success)" : undefined}
              strokeWidth={iv.detected ? 1.5 : 0}
              strokeOpacity={iv.detected ? 0.9 : 0}
            />
          ))}

          {/* Predicted anomaly bands (model, yellow) */}
          {predIntervals.map((iv, i) => (
            <ReferenceArea
              key={`pred-${i}`}
              x1={iv.startTs}
              x2={iv.endTs}
              fill="var(--anomaly-predicted)"
              stroke="var(--anomaly-predicted-stroke)"
              strokeWidth={1.5}
              strokeOpacity={1}
            />
          ))}

          <Line
            type="monotone"
            dataKey="value_normalized"
            name="value"
            stroke="var(--fg)"
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="prediction"
            name="prediction"
            stroke="var(--prediction)"
            strokeWidth={1.5}
            strokeDasharray="4 2"
            dot={false}
            isAnimationActive={false}
            connectNulls={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </section>
  );
}

export default memo(TelemetryChart);
