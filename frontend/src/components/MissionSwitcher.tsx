import type { MissionLink } from "../api/types";

type Props = {
  availableMissions: MissionLink[];
  currentMissionId: string;
};

/**
 * Renders a mission selector when more than one mission is configured.
 * Navigation is full-page: each mission runs as an independent same-origin
 * service, so switching is a browser navigate rather than an in-app state
 * change. Renders nothing when availableMissions.length <= 1.
 */
export default function MissionSwitcher({
  availableMissions,
  currentMissionId,
}: Props) {
  if (availableMissions.length <= 1) return null;

  return (
    <nav className="mission-switcher" aria-label="Mission selector">
      {availableMissions.map((m) => {
        const isCurrent = m.id === currentMissionId;
        return isCurrent ? (
          <span key={m.id} className="mission-switcher__item mission-switcher__item--active">
            {m.label}
          </span>
        ) : (
          <a
            key={m.id}
            className="mission-switcher__item"
            href={m.url}
            onClick={(e) => {
              e.preventDefault();
              window.location.assign(m.url);
            }}
          >
            {m.label}
          </a>
        );
      })}
    </nav>
  );
}
