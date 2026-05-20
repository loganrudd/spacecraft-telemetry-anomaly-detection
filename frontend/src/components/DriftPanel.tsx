import DriftFeatureBar from "./DriftFeatureBar";
import { useChannelDrift, useSubsystemDrift } from "../state/driftStore";

const ALERT_THRESHOLD = 0.3;

type Props = {
  channels: string[];
  disabled?: boolean;
};

function SubsystemGauge() {
  const { percent, alert } = useSubsystemDrift();
  const pct = percent !== null ? Math.round(percent * 100) : null;

  return (
    <div
      className={`drift-panel__gauge${alert ? " drift-panel__gauge--alert" : ""}`}
      aria-label="Subsystem drift gauge"
    >
      {pct !== null ? (
        <>
          <span className="drift-panel__gauge-label">
            {pct}% channels drifting
          </span>
          <div className="drift-panel__gauge-bar">
            <div
              className={`drift-panel__gauge-fill${pct / 100 >= ALERT_THRESHOLD ? " drift-panel__gauge-fill--alert" : ""}`}
              style={{ width: `${pct}%` }}
            />
          </div>
        </>
      ) : (
        <span className="drift-panel__gauge-label drift-panel__gauge-label--waiting">
          Waiting for subsystem summary…
        </span>
      )}
    </div>
  );
}

function ChannelDriftRow({ channel }: { channel: string }) {
  const event = useChannelDrift(channel);

  if (!event) {
    return (
      <div className="drift-panel__channel drift-panel__channel--waiting">
        <span className="drift-panel__channel-name">{channel}</span>
        <span className="drift-panel__channel-status">waiting…</span>
      </div>
    );
  }

  return (
    <div
      className={`drift-panel__channel${event.drifted ? " drift-panel__channel--drifted" : ""}`}
    >
      <div className="drift-panel__channel-header">
        <span className="drift-panel__channel-name">{channel}</span>
        <span className="drift-panel__channel-pct">
          {Math.round(event.percent_drifted * 100)}% drifted
        </span>
        <span className="drift-panel__channel-ts">
          {event.timestamp.replace("T", " ").slice(0, 19)}
        </span>
      </div>
      <DriftFeatureBar features={event.features} />
    </div>
  );
}

export default function DriftPanel({ channels, disabled = false }: Props) {
  if (disabled) {
    return (
      <aside className="drift-panel drift-panel--disabled">
        <header className="drift-panel__header">
          <span className="drift-panel__title">Drift Monitoring</span>
        </header>
        <p className="drift-panel__empty">Drift monitoring disabled.</p>
      </aside>
    );
  }

  return (
    <aside className="drift-panel">
      <header className="drift-panel__header">
        <span className="drift-panel__title">Drift Monitoring</span>
      </header>

      <SubsystemGauge />

      <div className="drift-panel__channels">
        {channels.length === 0 ? (
          <p className="drift-panel__empty">No channels selected.</p>
        ) : (
          channels.map((ch) => <ChannelDriftRow key={ch} channel={ch} />)
        )}
      </div>
    </aside>
  );
}
