"""Check headroom for increasing replay_warmup_rows.

For each demo channel, report the absolute index of the first labeled anomaly
in the test set. The slice can place the anomaly at slice-row = min(hits[0],
warmup_rows). To clear the detector warmup (window_size + threshold_window),
we need hits[0] >= desired_warmup_rows.
"""

from __future__ import annotations

import numpy as np

from spacecraft_telemetry.core.config import load_settings
from spacecraft_telemetry.model.dataset import load_series_parquet

CHANNELS = ["channel_41", "channel_43", "channel_44", "channel_45"]


def main() -> None:
    s = load_settings("cloud")
    mission = s.api.mission
    ws = s.model.window_size
    tw = s.model.threshold_window
    warm_need = ws + tw
    print(f"config window_size={ws} threshold_window={tw} "
          f"-> detector warm after ~{warm_need} ticks")
    print(f"current replay_warmup_rows={s.api.replay_warmup_rows} "
          f"max_rows={s.api.replay_max_rows}\n")
    for ch in CHANNELS:
        _v, _seg, anom, _ts = load_series_parquet(
            s.preprocess.processed_data_dir, mission, ch, "test"
        )
        hits = np.where(anom)[0]
        h0 = int(hits[0]) if hits.size else -1
        n = len(anom)
        # contiguous length of first run
        run_len = 0
        if hits.size:
            run_len = 1
            for j in range(1, len(hits)):
                if hits[j] == hits[j - 1] + 1:
                    run_len += 1
                else:
                    break
        print(f"{ch}: test_len={n} first_anom_abs_idx={h0} first_run_len={run_len} "
              f"-> max placeable warmup_rows={h0} "
              f"({'OK' if h0 >= warm_need else 'TOO EARLY'} for warm_need={warm_need})")


if __name__ == "__main__":
    main()
