import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import DriftFeatureBar from "../components/DriftFeatureBar";
import type { DriftFeature } from "../api/types";

const features: DriftFeature[] = [
  { feature: "value_normalized", score: 0.5, drifted: true },
  { feature: "rate_of_change", score: 0.05, drifted: false },
];

describe("DriftFeatureBar", () => {
  it("renders nominal (green) segments for every feature when the channel is not drifted", () => {
    // Raw per-feature `drifted` can flip on a single noisy run even while the
    // channel is still nominal (K-consecutive confirmation) -- the bar must
    // not show red until the channel itself has confirmed drift.
    const { container } = render(
      <DriftFeatureBar features={features} channelDrifted={false} />
    );
    const segments = container.querySelectorAll(".drift-feature-bar__segment");
    expect(segments).toHaveLength(2);
    segments.forEach((seg) => {
      expect(seg.className).toContain("drift-feature-bar__segment--nominal");
      expect(seg.className).not.toContain("drift-feature-bar__segment--drifted");
    });
  });

  it("shows red only for features that are individually drifted once the channel is drifted", () => {
    const { container } = render(
      <DriftFeatureBar features={features} channelDrifted={true} />
    );
    const segments = Array.from(
      container.querySelectorAll(".drift-feature-bar__segment")
    );
    expect(segments[0].className).toContain("drift-feature-bar__segment--drifted");
    expect(segments[1].className).toContain("drift-feature-bar__segment--nominal");
  });

  it("renders nothing for an empty feature list", () => {
    const { container } = render(
      <DriftFeatureBar features={[]} channelDrifted={false} />
    );
    expect(container.querySelector(".drift-feature-bar")).toBeNull();
  });
});
