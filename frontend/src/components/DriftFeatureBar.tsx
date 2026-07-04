import type { DriftFeature } from "../api/types";

type Props = {
  features: DriftFeature[];
  /** Channel-level confirmed drift status (DriftEvent.drifted). */
  channelDrifted: boolean;
};

export default function DriftFeatureBar({ features, channelDrifted }: Props) {
  if (features.length === 0) return null;

  return (
    <div className="drift-feature-bar" aria-label="Feature drift scores">
      {features.map((f) => {
        // A feature's own `drifted` flag is the raw per-run signal (unconfirmed --
        // see RollingDriftMonitor's K-consecutive confirmation) so it can flip on
        // a single noisy run even while the channel is still nominal. Only show
        // red once the channel itself has confirmed drift; otherwise green.
        const flagged = channelDrifted && f.drifted;
        return (
          <div
            key={f.feature}
            className={`drift-feature-bar__segment${flagged ? " drift-feature-bar__segment--drifted" : " drift-feature-bar__segment--nominal"}`}
            title={`${f.feature}: ${f.score.toFixed(3)}${flagged ? " (drifted)" : ""}`}
            style={{ flex: 1 }}
          />
        );
      })}
    </div>
  );
}
