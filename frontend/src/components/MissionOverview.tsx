import SubsystemCard from "./SubsystemCard";
import { useSubsystemRollup } from "../hooks/useSubsystemRollup";

type Props = {
  mission: string;
  channelSubsystems: Record<string, string>;
  onEnterSubsystem: (subsystem: string, channels: string[]) => void;
};

export default function MissionOverview({
  mission,
  channelSubsystems,
  onEnterSubsystem,
}: Props) {
  const rollup = useSubsystemRollup(channelSubsystems);

  const subsystems = Array.from(rollup.entries()).sort(([a], [b]) =>
    a.localeCompare(b),
  );

  return (
    <div className="mission-overview">
      <header className="mission-overview__header">
        <span className="mission-overview__mission">{mission}</span>
        <span className="mission-overview__subtitle">
          {subsystems.length} subsystems · {Object.keys(channelSubsystems).length} channels
        </span>
      </header>

      <div className="mission-overview__grid">
        {subsystems.map(([sub, channels]) => (
          <SubsystemCard
            key={sub}
            subsystem={sub}
            channels={channels}
            onEnter={(chs) => onEnterSubsystem(sub, chs)}
            onEnterChannel={(ch) => onEnterSubsystem(sub, [ch])}
          />
        ))}
      </div>
    </div>
  );
}
