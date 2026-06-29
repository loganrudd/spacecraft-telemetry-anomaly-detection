type LiveStreamStatus = "live" | "los" | "connecting" | "closed";

type Props = {
  status: LiveStreamStatus;
  expectedResumeInS?: number;
};

function formatMinutes(seconds: number): string {
  const m = Math.round(seconds / 60);
  return m <= 1 ? "~1 min" : `~${m} min`;
}

/**
 * Narrow banner indicating whether the ISS live telemetry stream is active.
 *
 * Renders nothing in "closed" or "connecting" states to avoid distracting the
 * user before the stream is known to be live.  On LOS it shows an explanatory
 * message with an optional median-duration estimate so observers understand
 * the gap is a normal orbital comms handover, not a server error.
 */
export default function LiveStatusBanner({ status, expectedResumeInS }: Props) {
  if (status === "closed" || status === "connecting") return null;

  if (status === "live") {
    return (
      <div className="live-banner live-banner--live" role="status" aria-live="polite">
        <span className="live-banner__dot" aria-hidden="true">●</span>
        {" "}LIVE
      </div>
    );
  }

  // LOS
  const eta = expectedResumeInS != null ? formatMinutes(expectedResumeInS) : null;
  return (
    <div className="live-banner live-banner--los" role="alert" aria-live="assertive">
      <span className="live-banner__icon" aria-hidden="true">⚠</span>
      {" "}Signal lost (TDRS handover){eta ? ` — typically restored within ${eta}` : ""}
    </div>
  );
}
