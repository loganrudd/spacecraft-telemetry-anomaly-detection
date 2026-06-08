import type { ChannelStatus } from "../hooks/useSubsystemRollup";

type Props = {
  subsystem: string;
  channels: Map<string, ChannelStatus>;
  onEnter: (channels: string[]) => void;
  onEnterChannel: (channel: string) => void;
};

function shortName(ch: string): string {
  return ch.replace(/^channel_?/, "Ch. ");
}

export default function SubsystemCard({
  subsystem,
  channels,
  onEnter,
  onEnterChannel,
}: Props) {
  const channelList = Array.from(channels.entries()).sort(([a], [b]) =>
    a.localeCompare(b),
  );
  const anomalousChannels = channelList.filter(([, s]) => s.anomaly);
  const driftedChannels = channelList.filter(([, s]) => s.drifted);
  const isNominal = anomalousChannels.length === 0 && driftedChannels.length === 0;

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
          {channelList.length} channels
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

      <div className={`subsystem-card__summary${isNominal ? " subsystem-card__summary--nominal" : ""}`}>
        {isNominal ? (
          <span className="subsystem-card__badge subsystem-card__badge--nominal">
            nominal
          </span>
        ) : (
          <>
            <div className="subsystem-card__summary-col">
              {anomalousChannels.map(([ch]) => (
                <button
                  key={ch}
                  className="subsystem-card__badge subsystem-card__badge--anomaly"
                  onClick={(e) => { e.stopPropagation(); onEnterChannel(ch); }}
                >
                  anomaly on {shortName(ch)}
                </button>
              ))}
            </div>
            <div className="subsystem-card__summary-col">
              {driftedChannels.map(([ch]) => (
                <button
                  key={ch}
                  className="subsystem-card__badge subsystem-card__badge--drift"
                  onClick={(e) => { e.stopPropagation(); onEnterChannel(ch); }}
                >
                  drift on {shortName(ch)}
                </button>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
