type Props = {
  allChannels: string[];  // channels in the current subsystem
  selected: string[];
  onChange: (channels: string[]) => void;
};

export default function ChannelPicker({ allChannels, selected, onChange }: Props) {
  const allSelected = allChannels.length > 0 && allChannels.every((ch) => selected.includes(ch));

  function toggle(channel: string) {
    if (selected.includes(channel)) {
      onChange(selected.filter((c) => c !== channel));
    } else {
      onChange([...selected, channel]);
    }
  }

  function toggleAll() {
    if (allSelected) {
      onChange([]);
    } else {
      onChange([...allChannels]);
    }
  }

  return (
    <aside className="channel-picker">
      <header className="channel-picker__header">
        <label className="channel-picker__select-all">
          <input
            type="checkbox"
            checked={allSelected}
            onChange={toggleAll}
            aria-label="Select all channels"
          />
          All channels
        </label>
      </header>

      <ul className="channel-picker__list" role="listbox" aria-multiselectable>
        {allChannels.map((ch) => {
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
