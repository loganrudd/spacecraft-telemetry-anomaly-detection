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
import { useTelemetryChannel } from "../state/telemetryStore";
import { collapseFlags } from "../utils/anomalyIntervals";
import type { TelemetryEvent } from "../api/types";

type DensityTier = "comfortable" | "compact" | "dense";

const CHART_HEIGHTS: Record<DensityTier, number> = {
  comfortable: 220,
  compact: 150,
  dense: 100,
};

type Props = { channel: string; density?: DensityTier };

function CustomTooltip({ active, payload }: TooltipProps<number, string>) {
  if (!active || !payload?.length) return null;
  const ev = payload[0]?.payload as TelemetryEvent | undefined;
  if (!ev) return null;
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
      <p>anomaly (labeled): {ev.is_anomaly ? "yes" : "no"}</p>
      <p>anomaly (predicted): {ev.is_anomaly_predicted ? "yes" : "no"}</p>
    </div>
  );
}

function TelemetryChart({ channel, density = "comfortable" }: Props) {
  const events = useTelemetryChannel(channel);

  const trueIntervals = collapseFlags(events, "is_anomaly");
  const predIntervals = collapseFlags(events, "is_anomaly_predicted");
  const chartHeight = CHART_HEIGHTS[density];

  if (events.length === 0) {
    return (
      <section className={`telemetry-chart telemetry-chart--${density}`}>
        <h2 className="telemetry-chart__title">{channel}</h2>
        <p className="telemetry-chart__waiting">Waiting for data…</p>
      </section>
    );
  }

  return (
    <section className={`telemetry-chart telemetry-chart--${density}`}>
      <h2 className="telemetry-chart__title">{channel}</h2>
      <ResponsiveContainer width="100%" height={chartHeight}>
        <LineChart
          data={events}
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
          {density !== "dense" && <Tooltip content={<CustomTooltip />} />}
          {density === "comfortable" && (
            <Legend wrapperStyle={{ fontSize: 11, color: "var(--fg-muted)" }} />
          )}

          {/* Ground-truth anomaly bands (labeled, red) */}
          {trueIntervals.map((iv, i) => (
            <ReferenceArea
              key={`true-${i}`}
              x1={iv.startTs}
              x2={iv.endTs}
              fill="var(--anomaly-true)"
              strokeOpacity={0}
            />
          ))}

          {/* Predicted anomaly bands (model, yellow) */}
          {predIntervals.map((iv, i) => (
            <ReferenceArea
              key={`pred-${i}`}
              x1={iv.startTs}
              x2={iv.endTs}
              fill="var(--anomaly-predicted)"
              strokeOpacity={0}
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
