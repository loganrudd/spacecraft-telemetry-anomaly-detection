import type { DriftFeature } from "../api/types";

type Props = {
  features: DriftFeature[];
};

export default function DriftFeatureBar({ features }: Props) {
  if (features.length === 0) return null;

  return (
    <div className="drift-feature-bar" aria-label="Feature drift scores">
      {features.map((f) => (
        <div
          key={f.feature}
          className={`drift-feature-bar__segment${f.drifted ? " drift-feature-bar__segment--drifted" : ""}`}
          title={`${f.feature}: ${f.score.toFixed(3)}${f.drifted ? " (drifted)" : ""}`}
          style={{ flex: 1 }}
        />
      ))}
    </div>
  );
}
