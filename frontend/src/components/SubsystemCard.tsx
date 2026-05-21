import type { ChannelStatus } from "../hooks/useSubsystemRollup";

type Props = {
  subsystem: string;
  channels: Map<string, ChannelStatus>;
  onEnter: (channels: string[]) => void;
  onEnterChannel: (channel: string) => void;
};

export default function SubsystemCard({
  subsystem,
  channels,
  onEnter,
  onEnterChannel,
}: Props) {
  const channelList = Array.from(channels.entries()).sort(([a], [b]) =>
    a.localeCompare(b),
  );
  const anomalyCount = channelList.filter(([, s]) => s.anomaly).length;
  const driftCount = channelList.filter(([, s]) => s.drifted).length;

  return (
    <div
      className="subsystem-card"
      onClick={() => onEnter(channelList.map(([ch]) => ch))}
      role="button"
      tabIndex={0}
      aria-label={`${subsystem}: ${channelList.length} channels`}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ")
          onEnter(channelList.map(([ch]) => ch));
      }}
    >
      <div className="subsystem-card__header">
        <span className="subsystem-card__name">{subsystem}</span>
        <span className="subsystem-card__count">
          {channelList.length} ch
        </span>
      </div>

      <div className="subsystem-card__dots" aria-label="Channel status dots">
        {channelList.map(([ch, status]) => (
          <button
            key={ch}
            className={`subsystem-card__dot${status.anomaly ? " subsystem-card__dot--anomaly" : status.drifted ? " subsystem-card__dot--drift" : status.lastUpdated !== null ? " subsystem-card__dot--nominal" : " subsystem-card__dot--waiting"}`}
            title={`${ch}: ${status.anomaly ? "anomaly" : status.drifted ? "drift" : status.lastUpdated !== null ? "nominal" : "waiting"}`}
            aria-label={ch}
            onClick={(e) => {
              e.stopPropagation();
              onEnterChannel(ch);
            }}
          />
        ))}
      </div>

      <div className="subsystem-card__summary">
        {anomalyCount > 0 && (
          <span className="subsystem-card__badge subsystem-card__badge--anomaly">
            {anomalyCount} anomal{anomalyCount === 1 ? "y" : "ies"}
          </span>
        )}
        {driftCount > 0 && (
          <span className="subsystem-card__badge subsystem-card__badge--drift">
            drift on {driftCount}
          </span>
        )}
        {anomalyCount === 0 && driftCount === 0 && (
          <span className="subsystem-card__badge subsystem-card__badge--nominal">
            nominal
          </span>
        )}
      </div>
    </div>
  );
}
