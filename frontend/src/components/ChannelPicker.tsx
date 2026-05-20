import { useEffect, useState } from "react";
import { fetchHealth } from "../api/health";
import type { HealthResponse } from "../api/types";

const MAX_RECOMMENDED_CHANNELS = 5;

type Props = {
  selected: string[];
  onChange: (channels: string[]) => void;
};

export default function ChannelPicker({ selected, onChange }: Props) {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch((e: unknown) => {
        setError(e instanceof Error ? e.message : "Failed to load channels");
      });
  }, []);

  if (error) {
    return (
      <aside className="channel-picker channel-picker--error">
        <p className="channel-picker__error">⚠ {error}</p>
      </aside>
    );
  }

  if (!health) {
    return (
      <aside className="channel-picker channel-picker--loading">
        <p className="channel-picker__loading">Loading channels…</p>
      </aside>
    );
  }

  const { mission, subsystem, channels_loaded } = health;

  function toggle(channel: string) {
    if (selected.includes(channel)) {
      onChange(selected.filter((c) => c !== channel));
    } else {
      onChange([...selected, channel]);
    }
  }

  return (
    <aside className="channel-picker">
      <header className="channel-picker__header">
        <span className="channel-picker__mission">{mission}</span>
        <span className="channel-picker__sep">/</span>
        <span className="channel-picker__subsystem">{subsystem}</span>
      </header>

      {selected.length > MAX_RECOMMENDED_CHANNELS && (
        <div className="channel-picker__warn" role="alert">
          Rendering {selected.length} live charts may be slow (recommended: ≤
          {MAX_RECOMMENDED_CHANNELS}).
        </div>
      )}

      <ul className="channel-picker__list" role="listbox" aria-multiselectable>
        {channels_loaded.map((ch) => {
          const isSelected = selected.includes(ch);
          return (
            <li
              key={ch}
              role="option"
              aria-selected={isSelected}
              className={`channel-picker__item${isSelected ? " channel-picker__item--active" : ""}`}
              onClick={() => toggle(ch)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") toggle(ch);
              }}
              tabIndex={0}
            >
              {ch}
            </li>
          );
        })}
      </ul>
    </aside>
  );
}
