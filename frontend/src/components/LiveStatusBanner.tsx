type LiveStreamStatus = "live" | "los" | "connecting" | "closed";

type Props = {
  status: LiveStreamStatus;
  expectedResumeInS?: number;
  /** Status event's "mode" field. "replay" means the LOS fallback is active. */
  mode?: string;
};

function formatMinutes(seconds: number): string {
  const m = Math.round(seconds / 60);
  return m <= 1 ? "~1 min" : `~${m} min`;
}

/**
 * Narrow banner indicating whether the ISS live telemetry stream is active.
 *
 * Renders nothing in "closed" or "connecting" states to avoid distracting the
 * user before the stream is known to be live.  On LOS the pump falls back to
 * replaying recent collected telemetry so the chart stays alive; the banner
 * always labels this explicitly ("showing recent recorded data") so the
 * viewer is never misled into thinking replayed data is live — honesty comes
 * from the label, not from going silent during the gap.
 */
export default function LiveStatusBanner({ status, expectedResumeInS, mode }: Props) {
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
  const replaying = mode === "replay";
  const suffix = eta ? `, live resumes in ${eta}` : "";
  return (
    <div className="live-banner live-banner--los" role="alert" aria-live="assertive">
      <span className="live-banner__icon" aria-hidden="true">⚠</span>
      {" "}Signal lost (TDRS handover)
      {replaying ? ` — showing recent recorded data${suffix}` : suffix}
    </div>
  );
}
