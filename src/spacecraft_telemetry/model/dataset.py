"""Dataset utilities for Telemanom LSTM training.

Reads per-timestep series Parquet (written by the preprocessing pipeline)
via PyArrow.  Constructs LSTM windows on-the-fly in the DataLoader, avoiding
the 250x disk inflation of pre-materialized window arrays.

Public API:
    make_dataloaders(settings, mission, channel)     -> (train_loader, val_loader)
    make_test_dataloader(settings, mission, channel) -> (loader, target_timestamps,
                                                         window_is_anomaly)
    load_window_labels(settings, mission, channel)   -> window_is_anomaly (no torch)
    window_target_timestamps(settings, mission, channel) -> target_timestamps (no torch)

Multivariate (docs/plans/021-multivariate-telemanom.md): every function above
resolves its channel group from ``settings.model.input_channels`` (None ->
``[channel]``, the byte-identical univariate default). A non-None group joins
channels on ``telemetry_timestamp`` via load_multichannel_series_parquet /
load_multichannel_series_metadata and ``channel`` becomes the model's
registry/experiment key rather than a literal channel to load.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as _TorchDataset
from upath import UPath

from spacecraft_telemetry.core.config import Settings
from spacecraft_telemetry.core.logging import get_logger
from spacecraft_telemetry.core.paths import output_path

log = get_logger(__name__)


def _resolve_channel_group(settings: Settings, channel: str) -> list[str]:
    """Return the ordered list of channel_ids to load for one model.

    None (default) -> [channel]: the univariate path, byte-identical to
    pre-021 behaviour. A non-None ``settings.model.input_channels`` makes
    ``channel`` the model's registry/experiment key (e.g. a subsystem name)
    while this list supplies the channels to load and forecast jointly.

    Order is significant and is exactly the order downstream (architecture,
    save/load) assumes — see docs/plans/021-multivariate-telemanom.md.
    """
    group = settings.model.input_channels
    return list(group) if group is not None else [channel]


def check_channel_key_pairing(cfg: Any, channel: str) -> None:
    """Assert ``channel`` is a group KEY, not a group MEMBER, when multivariate.

    ``channel`` carries two meanings (docs/reviews/021, item A2): a real
    channel_id in the univariate path, and a group key — a subsystem name — in
    the multivariate one. The typed fix (distinct Channel / Subsystem value
    types) would touch ~8 modules and rewrite the multivariate boundary right
    before DC-VAE reuses it, so the duality is accepted deliberately; this is
    the cheap guard that catches the one confusion it actually enables.

    Lives here, beside ``_resolve_channel_group``, because that function is
    where the duality is interpreted — the check and the interpretation should
    not drift apart.

    Passing a member channel as the key would train or score a model registered
    under a real channel's name whose weights cover the whole group — polluting
    the univariate registry that cli.py's promote/demote discovery depends on,
    with no shape error anywhere to reveal it.

    Takes a ``ModelConfig``-shaped object (``Any``, matching window_span's
    convention in this module).

    Raises:
        ValueError: If ``channel`` appears in ``cfg.input_channels``.
    """
    if cfg.input_channels and channel in cfg.input_channels:
        raise ValueError(
            f"channel={channel!r} is a MEMBER of input_channels={list(cfg.input_channels)}, "
            "but with input_channels set it is used as the group's registry/experiment "
            "KEY (e.g. a subsystem name), not a channel to load. Pass the subsystem "
            "name instead — see docs/plans/021-multivariate-telemanom.md."
        )


def _read_partition_table(
    processed_dir: Path | UPath | str,
    mission: str,
    channel: str,
    split: Literal["train", "test"],
    columns: list[str],
    variant: str | None = None,
) -> pa.Table:
    """Read + concat + timestamp-sort one channel partition's Parquet files.

    Shared by load_series_parquet and load_series_metadata so both read the
    identical partition-discovery/sort logic and differ only in which
    columns PyArrow actually parses.

    ``variant`` defaults to None, reproducing the pre-variant partition path
    byte for byte (see core/paths.output_path). Callers within this module
    that already hold a Settings object pass ``settings.variant`` explicitly;
    external callers (api/, esa_adb/) that pass a bare processed_dir string
    are unaffected by default.

    Raises:
        FileNotFoundError: If the partition directory doesn't exist or has no
            Parquet files.
    """
    partition_dir = output_path(
        processed_dir, mission, variant,
        split, f"mission_id={mission}", f"channel_id={channel}",
    )
    if not partition_dir.exists():
        raise FileNotFoundError(
            f"No series Parquet found for mission={mission!r} channel={channel!r} "
            f"split={split!r}. Expected directory: {partition_dir}"
        )

    parquet_files = sorted(partition_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"Directory exists but contains no .parquet files: {partition_dir}"
        )

    tables = [pq.read_table(str(f), columns=columns) for f in parquet_files]
    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    return table.sort_by("telemetry_timestamp")


def load_series_parquet(
    processed_dir: Path | UPath | str,
    mission: str,
    channel: str,
    split: Literal["train", "test"],
    variant: str | None = None,
) -> tuple[
    np.ndarray[Any, np.dtype[np.float32]],
    np.ndarray[Any, np.dtype[np.int32]],
    np.ndarray[Any, np.dtype[np.bool_]],
    np.ndarray[Any, Any],
]:
    """Read per-timestep series for one channel partition.

    Reads from the Hive-partitioned layout:
        {processed_dir}/{mission}/[{variant}/]{split}/mission_id={mission}/channel_id={channel}/*.parquet

    ``variant`` defaults to None (today's layout, unchanged) — see
    core/paths.output_path.

    Returns:
        values:      (N,) float32  — normalized values, sorted by timestamp
        segment_ids: (N,) int32    — segment ID per timestep (for boundary detection)
        is_anomaly:  (N,) bool     — per-timestep anomaly flag
        timestamps:  (N,) datetime64[ns] — telemetry timestamp per timestep

    Raises:
        FileNotFoundError: If the partition directory doesn't exist or has no
            Parquet files.
    """
    table = _read_partition_table(
        processed_dir, mission, channel, split,
        columns=["telemetry_timestamp", "value_normalized", "segment_id", "is_anomaly"],
        variant=variant,
    )

    values = table.column("value_normalized").to_numpy(zero_copy_only=False).astype(np.float32)
    segment_ids = table.column("segment_id").to_numpy(zero_copy_only=False).astype(np.int32)
    is_anomaly = table.column("is_anomaly").to_numpy(zero_copy_only=False).astype(bool)
    timestamps = table.column("telemetry_timestamp").to_numpy(zero_copy_only=False)

    return values, segment_ids, is_anomaly, timestamps


def load_series_metadata(
    processed_dir: Path | UPath | str,
    mission: str,
    channel: str,
    split: Literal["train", "test"],
    variant: str | None = None,
) -> tuple[
    np.ndarray[Any, np.dtype[np.int32]],
    np.ndarray[Any, np.dtype[np.bool_]],
    np.ndarray[Any, Any],
]:
    """Read only the small per-timestep columns needed for windowing/timeline math.

    Skips ``value_normalized`` — the largest column on disk and the one
    column none of esa_adb.timeline / esa_adb.report / esa_adb.detections
    ever reads. Same partition as load_series_parquet, just fewer columns
    parsed — use this wherever only segment_ids/is_anomaly/timestamps are
    needed (window-index building, target-timestamp derivation, observed
    timeline construction).

    ``variant`` defaults to None (today's layout, unchanged) — see
    core/paths.output_path.

    Returns:
        segment_ids: (N,) int32    — segment ID per timestep
        is_anomaly:  (N,) bool     — per-timestep anomaly flag
        timestamps:  (N,) datetime64[ns] — telemetry timestamp per timestep

    Raises:
        FileNotFoundError: Same as load_series_parquet.
    """
    table = _read_partition_table(
        processed_dir, mission, channel, split,
        columns=["telemetry_timestamp", "segment_id", "is_anomaly"],
        variant=variant,
    )

    segment_ids = table.column("segment_id").to_numpy(zero_copy_only=False).astype(np.int32)
    is_anomaly = table.column("is_anomaly").to_numpy(zero_copy_only=False).astype(bool)
    timestamps = table.column("telemetry_timestamp").to_numpy(zero_copy_only=False)

    return segment_ids, is_anomaly, timestamps


# Alignment loss above this fraction is logged as a warning, not just info —
# see docs/plans/021-multivariate-telemanom.md Design: "Verify alignment
# loss — if the intersection is materially smaller than any single channel,
# that is a finding, not a detail."
_ALIGNMENT_LOSS_WARN_THRESHOLD = 0.10

# Upper bound on a multivariate group (docs/reviews/021, item A4).
#
# _align_multi_channel materialises ALL C channels densely: peak memory is
# roughly C * N * 17 bytes (float32 values + int32 segment ids + bool flags)
# plus the DatetimeIndex overhead of the join. Measured ~800 MB for 6 channels
# x 7.7 M rows, i.e. ~130 MB per channel at ESA-Mission1 scale — so 32
# channels is already ~4 GB, at the ceiling of a small worker, and passing a
# whole ~100-channel mission as one group would need ~13 GB.
#
# Nothing bounded this before. Subsystem-sized groups (6-30 channels, see
# ray_fanout/tune.py) fit under the limit, so the guard exists to catch the
# pathological call — a whole mission handed in as one group — rather than to
# constrain normal use. Note a subsystem at the TOP of that documented range
# is already near the limit; raise this deliberately (and size the worker to
# match) rather than by reflex if a legitimate group ever exceeds it.
_MAX_MULTIVARIATE_CHANNELS = 32


def _joint_segment_ids(
    per_channel_segment_ids: np.ndarray[Any, np.dtype[np.int32]],
) -> np.ndarray[Any, np.dtype[np.int32]]:
    """Derive one monotonic segment id from per-channel segment ids.

    ``per_channel_segment_ids``: (N, C) int32, already aligned on a common
    timestamp index. Each channel's own segment id was assigned independently
    by preprocessing's per-channel gap detection, so two channels can each be
    internally gap-free over a span while disagreeing about where their own
    boundaries fall. A joint window must not cross *either* channel's
    boundary, so a boundary exists at row t iff ANY channel's segment id
    changed between t-1 and t.

    Returns an (N,) int32 array usable directly by ``_build_window_index``
    (which only checks equality across a span, not the numeric values) —
    this keeps the existing, tested window-index logic untouched for the
    multivariate path.
    """
    n = per_channel_segment_ids.shape[0]
    joint = np.zeros(n, dtype=np.int32)
    if n <= 1:
        return joint
    boundary = (np.diff(per_channel_segment_ids, axis=0) != 0).any(axis=1)
    joint[1:] = np.cumsum(boundary)
    return joint


def _align_multi_channel(
    per_channel: list[
        tuple[
            np.ndarray[Any, np.dtype[np.float32]],
            np.ndarray[Any, np.dtype[np.int32]],
            np.ndarray[Any, np.dtype[np.bool_]],
            np.ndarray[Any, Any],
        ]
    ],
    channels: list[str],
    max_channels: int = _MAX_MULTIVARIATE_CHANNELS,
) -> tuple[
    np.ndarray[Any, np.dtype[np.float32]],
    np.ndarray[Any, np.dtype[np.int32]],
    np.ndarray[Any, np.dtype[np.bool_]],
    np.ndarray[Any, Any],
]:
    """Inner-join per-channel series on ``telemetry_timestamp``.

    Each element of ``per_channel`` is one channel's
    ``(values, segment_ids, is_anomaly, timestamps)`` as returned by
    ``load_series_parquet``, already sorted by timestamp. The intersection of
    all channels' timestamps is the defensible default alignment (Design,
    docs/plans/021-multivariate-telemanom.md) — a row that any channel is
    missing cannot be forecast jointly.

    Returns:
        values:      (N, C) float32 — column i is channels[i]'s normalized value.
        segment_ids: (N,) int32     — joint segment id, see _joint_segment_ids.
        is_anomaly:  (N, C) bool    — per-channel flag, preserved (not OR'd)
                                       so callers can report per-channel recall.
        timestamps:  (N,) datetime64[ns] — the aligned, sorted timestamp index.

    Args:
        max_channels: Refuse groups larger than this — see
            _MAX_MULTIVARIATE_CHANNELS for the memory arithmetic. An OOM on a
            spot worker is an expensive and confusing way to discover the
            limit; this fails immediately with the number that was asked for.

    Raises:
        ValueError: If the group exceeds ``max_channels``, or the intersection
            is empty.
    """
    if len(channels) > max_channels:
        raise ValueError(
            f"Multivariate group has {len(channels)} channels, above the "
            f"max_channels={max_channels} limit. _align_multi_channel "
            "materialises every channel densely (~130 MB per channel at "
            f"ESA-Mission1 scale), so this group would need roughly "
            f"{len(channels) * 130 / 1024:.1f} GB and is likely to OOM the "
            "worker. Group by subsystem rather than passing a whole mission, "
            "or raise max_channels deliberately and size the worker to match."
        )

    indices = [pd.DatetimeIndex(ts) for (_, _, _, ts) in per_channel]
    common = indices[0]
    for idx in indices[1:]:
        common = common.intersection(idx)
    common = common.sort_values()

    min_channel_rows = min(len(idx) for idx in indices)
    n_aligned = len(common)
    if n_aligned == 0:
        raise ValueError(
            f"No overlapping timestamps across channels {channels!r} — cannot "
            "align a multivariate window. Check that all channels were "
            "preprocessed over the same time range."
        )
    loss_frac = 1.0 - n_aligned / min_channel_rows
    log_fn = log.warning if loss_frac > _ALIGNMENT_LOSS_WARN_THRESHOLD else log.info
    log_fn(
        "model.dataset.multichannel_align",
        channels=channels,
        n_aligned=n_aligned,
        min_channel_rows=min_channel_rows,
        loss_frac=round(loss_frac, 4),
    )

    n, c = n_aligned, len(channels)
    values = np.empty((n, c), dtype=np.float32)
    segment_ids_2d = np.empty((n, c), dtype=np.int32)
    is_anomaly = np.empty((n, c), dtype=bool)
    zipped = zip(per_channel, indices, strict=True)
    for i, ((vals, seg_ids, is_anom, _ts), idx) in enumerate(zipped):
        pos = idx.get_indexer(common)
        if (pos < 0).any():
            raise AssertionError(
                f"common index is not a subset of channel {channels[i]!r}'s "
                "timestamps — intersection() invariant violated"
            )
        values[:, i] = vals[pos]
        segment_ids_2d[:, i] = seg_ids[pos]
        is_anomaly[:, i] = is_anom[pos]

    return values, _joint_segment_ids(segment_ids_2d), is_anomaly, common.to_numpy()


# Concurrency for per-channel parquet reads (docs/reviews/021, item D2).
#
# Channel loads are network-bound, not compute-bound: measured ~40-70 s per
# channel against GCS at 0.6-3% CPU. PyArrow releases the GIL while reading, so
# threads (not processes) recover most of that — no pickling of large arrays,
# and no interaction with the Ray worker's process model.
#
# Capped rather than unbounded: each in-flight load holds a full channel's
# arrays in memory, so the cap is a memory bound as much as a concurrency one.
_CHANNEL_LOAD_WORKERS = 6


def _load_channels_ordered(
    load_one: Any,
    channels: list[str],
) -> list[Any]:
    """Load every channel concurrently, returning results in INPUT order.

    Order is load-bearing: ``_align_multi_channel`` indexes ``channels`` by
    position, so a result list ordered by completion would silently attribute
    every channel's values to a different channel — same shapes, no error, all
    metrics wrong. ``ThreadPoolExecutor.map`` is used precisely because it
    yields in submission order regardless of which future finishes first.

    A single channel skips the pool entirely, so the univariate path takes on
    no thread-pool overhead or behaviour change.
    """
    if len(channels) == 1:
        return [load_one(channels[0])]

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(
        max_workers=min(_CHANNEL_LOAD_WORKERS, len(channels))
    ) as pool:
        # list() forces completion inside the context so exceptions surface
        # here, attached to the channel that raised, rather than at teardown.
        return list(pool.map(load_one, channels))


def load_multichannel_series_parquet(
    processed_dir: Path | UPath | str,
    mission: str,
    channels: list[str],
    split: Literal["train", "test"],
    variant: str | None = None,
) -> tuple[
    np.ndarray[Any, np.dtype[np.float32]],
    np.ndarray[Any, np.dtype[np.int32]],
    np.ndarray[Any, np.dtype[np.bool_]],
    np.ndarray[Any, Any],
]:
    """Read + align per-timestep series for a group of channels, jointly.

    Reads each channel via ``load_series_parquet`` (same partition layout,
    same normalization — each channel keeps its own z-score) and inner-joins
    them on ``telemetry_timestamp`` via ``_align_multi_channel``.

    For ``len(channels) == 1`` this still goes through the join path (a
    trivial one-channel join), unlike ``make_dataloaders``/
    ``make_test_dataloader`` which special-case that call site to bypass this
    function entirely — see their docstrings for why that distinction matters.

    Returns:
        values:      (N, C) float32 — column i is channels[i]'s normalized value.
        segment_ids: (N,) int32     — joint segment id (see _joint_segment_ids).
        is_anomaly:  (N, C) bool    — per-channel anomaly flag, not OR'd.
        timestamps:  (N,) datetime64[ns] — aligned, sorted timestamps.

    Raises:
        FileNotFoundError: If any channel's partition is missing (propagated
            from load_series_parquet).
        ValueError: If the channels share no common timestamps.
    """
    per_channel = _load_channels_ordered(
        lambda ch: load_series_parquet(processed_dir, mission, ch, split, variant=variant),
        channels,
    )
    return _align_multi_channel(per_channel, channels)


def load_multichannel_series_metadata(
    processed_dir: Path | UPath | str,
    mission: str,
    channels: list[str],
    split: Literal["train", "test"],
    variant: str | None = None,
) -> tuple[
    np.ndarray[Any, np.dtype[np.int32]],
    np.ndarray[Any, np.dtype[np.bool_]],
    np.ndarray[Any, Any],
]:
    """Metadata-only counterpart to load_multichannel_series_parquet.

    Skips ``value_normalized`` for every channel — see load_series_metadata's
    docstring for the single-channel rationale, which applies identically here.

    Returns:
        segment_ids: (N,) int32  — joint segment id.
        is_anomaly:  (N, C) bool — per-channel anomaly flag, not OR'd.
        timestamps:  (N,) datetime64[ns] — aligned, sorted timestamps.
    """
    per_channel_meta = _load_channels_ordered(
        lambda ch: load_series_metadata(processed_dir, mission, ch, split, variant=variant),
        channels,
    )
    # _align_multi_channel expects a values column; synthesize a dummy one so
    # the exact same alignment code path (and its logging) is reused rather
    # than duplicated for the metadata-only case.
    per_channel = [
        (np.empty(len(seg_ids), dtype=np.float32), seg_ids, is_anom, ts)
        for seg_ids, is_anom, ts in per_channel_meta
    ]
    _values, segment_ids, is_anomaly, timestamps = _align_multi_channel(per_channel, channels)
    return segment_ids, is_anomaly, timestamps


def window_span(cfg: Any) -> int:
    """Total timesteps one window consumes: inputs plus every forecast target.

    ``window_size`` inputs, then ``forecast_steps`` targets beginning
    ``prediction_horizon`` steps after the window's last input:

        span = window_size + prediction_horizon + forecast_steps - 1

    At ``forecast_steps == 1`` this is ``window_size + prediction_horizon``, the
    pre-021.7 formula, exactly. Centralised because the span appears in the
    window index, the target-timestamp map, and the per-window anomaly OR — and
    those three disagreeing by one would misalign errors against labels
    silently rather than raising.

    Takes a ``ModelConfig``-shaped object (``Any`` to avoid importing Settings
    into this module's hot path).
    """
    return int(cfg.window_size) + int(cfg.prediction_horizon) + int(cfg.forecast_steps) - 1


def first_target_offset(cfg: Any) -> int:
    """Offset from a window start to its FIRST forecast target.

    A multi-step window is attributed to the first instant it forecasts, not
    the last: that is the earliest moment the model was wrong, so a detection
    lands nearest fault onset rather than trailing the horizon. At
    ``forecast_steps == 1`` it is the single target, identical to pre-021.7.

    Note the mild consequence when errors are aggregated across the horizon
    (``forecast_error_reduction="mean"``/``"max"``): an error driven by a late
    forecast step is still attributed to the first, so detections can lead the
    fault slightly. That is the intended direction for early warning, and it is
    why "first" reduction exists as the strict-ablation alternative.
    """
    return int(cfg.window_size) + int(cfg.prediction_horizon) - 1


def _build_window_index(
    segment_ids: np.ndarray[Any, np.dtype[np.int32]],
    is_anomaly: np.ndarray[Any, np.dtype[np.bool_]],
    window_size: int,
    prediction_horizon: int,
    skip_anomalous_windows: bool,
    forecast_steps: int = 1,
) -> np.ndarray[Any, Any]:
    """Return int32 array of valid window start indices.

    A start index ``s`` is valid iff:

    1. ``segment_ids[s : s + span]`` is all one value — the window plus every
       forecast target don't span a segment gap.
    2. If ``skip_anomalous_windows``: no timestep in that span is anomalous.

    Segment IDs are assigned in ascending temporal order by the preprocessing
    pipeline, so ``segment_ids[s] == segment_ids[s + span - 1]``
    is a correct and O(N)-vectorisable boundary check.

    Args:
        segment_ids:            (N,) int32 — segment ID per timestep.
        is_anomaly:             (N,) bool  — per-timestep anomaly flag.
        window_size:            Number of input timesteps (W).
        prediction_horizon:     Steps from the window end to the FIRST target.
        skip_anomalous_windows: Skip windows that contain any anomalous step.
        forecast_steps:         Number of consecutive targets forecast per
                                window (021.7). 1 = pre-021.7 single target.
                                A longer horizon consumes more timesteps, so
                                FEWER windows fit in a segment — expect the
                                window count to drop as this rises.

    Returns:
        int32 array of valid start indices, length 0 when none qualify.
    """
    n = len(segment_ids)
    # total positions consumed per window — see window_span()
    span = window_size + prediction_horizon + forecast_steps - 1

    if n < span:
        return np.empty(0, dtype=np.int32)

    starts = np.arange(n - span + 1, dtype=np.int32)

    # Boundary check: first and last positions in span must share a segment.
    no_gap = segment_ids[starts] == segment_ids[starts + span - 1]

    if skip_anomalous_windows:
        # Prefix-sum for O(1) range anomaly queries.
        cumsum = np.empty(n + 1, dtype=np.int64)
        cumsum[0] = 0
        np.cumsum(is_anomaly, out=cumsum[1:])
        window_has_anomaly = (cumsum[starts + span] - cumsum[starts]) > 0
        return starts[no_gap & ~window_has_anomaly].astype(np.int32)  # type: ignore[no-any-return]

    return starts[no_gap].astype(np.int32)  # type: ignore[no-any-return]


class WindowedSequenceDataset(_TorchDataset):  # type: ignore[type-arg]
    """Index-based sliding-window Dataset over a per-timestep series.

    Stores the full values tensor once and slices windows lazily in
    ``__getitem__``, so memory use is O(N) not O(N x W).

    Accepts either a univariate ``(N,)`` array (the pre-021 shape) or a
    multivariate ``(N, C)`` array (docs/plans/021-multivariate-telemanom.md).
    The univariate case is a distinct branch, not C=1 of the general one —
    output shapes match pre-021 exactly (y is a 0-d scalar tensor, not a
    length-1 vector), so every existing caller is untouched byte-for-byte.

    ``forecast_steps`` (F, stage 021.7) likewise keeps F=1 on the pre-021.7
    shapes rather than adding a trailing 1, matching the architecture's output
    rank so loss and error extraction line up without squeezes:

        F=1, univariate    x: (W, 1)   y: ()      scalar at s + W + H - 1
        F=1, multivariate  x: (W, C)   y: (C,)
        F>1, univariate    x: (W, 1)   y: (1, F)
        F>1, multivariate  x: (W, C)   y: (C, F)

    Targets are channel-major for F>1 — ``y[i]`` is channel i's trajectory —
    matching TelemanomLSTM's ``(B, C, H)`` output layout so the MSE is
    elementwise with no transpose.
    """

    def __init__(
        self,
        values: np.ndarray[Any, np.dtype[np.float32]],
        start_indices: np.ndarray[Any, np.dtype[np.int32]],
        window_size: int,
        prediction_horizon: int,
        forecast_steps: int = 1,
    ) -> None:
        super().__init__()
        arr = np.ascontiguousarray(values)
        if arr.ndim not in (1, 2):
            raise ValueError(f"values must be 1-D or 2-D, got shape {arr.shape}")
        self._univariate = arr.ndim == 1
        # Contiguous tensor for fast slice in __getitem__.
        self._values = torch.from_numpy(arr)  # (N,) or (N, C) float32
        self._starts = start_indices           # (M,) int32
        self._W = window_size
        self._H = prediction_horizon
        self._F = forecast_steps

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = int(self._starts[idx])
        t0 = s + self._W + self._H - 1          # first forecast target
        if self._univariate:
            x = self._values[s : s + self._W].unsqueeze(-1)   # (W, 1)
            y = (
                self._values[t0]                              # scalar (pre-021.7)
                if self._F == 1
                else self._values[t0 : t0 + self._F].unsqueeze(0)  # (1, F)
            )
        else:
            x = self._values[s : s + self._W, :]               # (W, C)
            y = (
                self._values[t0, :]                            # (C,) (pre-021.7)
                if self._F == 1
                else self._values[t0 : t0 + self._F, :].transpose(0, 1)  # (C, F)
            )
        return x, y


def make_dataloaders(
    settings: Settings,
    mission: str,
    channel: str,
) -> tuple[
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
]:
    """Build train/val DataLoaders from per-timestep series Parquet.

    Reads the train-split Parquet, builds a window index (skipping windows
    that cross segment boundaries or contain any anomalous timestep), then
    partitions into a temporal train/val split.

    Val split: last ``val_fraction`` of valid windows (contiguous tail).
    Train DataLoader shuffles; val DataLoader does not.
    ``num_workers=0`` on MPS (macOS); cloud.yaml sets ``num_workers=4``.
    ``pin_memory`` is enabled automatically when CUDA is available.

    Multivariate (docs/plans/021-multivariate-telemanom.md): when
    ``settings.model.input_channels`` is set, ``channel`` is the model's
    registry key (e.g. a subsystem name) and the group's channels are loaded
    and aligned jointly via load_multichannel_series_parquet. A window is
    skipped if it crosses a segment boundary or contains an anomalous
    timestep in ANY channel — the Design section's "only continuous nominal
    parts ... without any anomalies in any target channel." A single-channel
    group (the default, and the ``len(group) == 1`` case generally) takes
    the original single-channel path unchanged, so this function is
    byte-identical to pre-021 whenever input_channels is None.

    Args:
        settings: Fully resolved Settings.
        mission:  Mission name, e.g. ``"ESA-Mission1"``.
        channel:  Channel ID, e.g. ``"channel_1"`` (or a subsystem key — see above).
    """
    cfg = settings.model
    group = _resolve_channel_group(settings, channel)
    if len(group) == 1:
        values, segment_ids, is_anomaly, _ = load_series_parquet(
            settings.preprocess.processed_data_dir, mission, group[0], "train",
            variant=settings.variant,
        )
    else:
        values, segment_ids, is_anomaly_2d, _ = load_multichannel_series_parquet(
            settings.preprocess.processed_data_dir, mission, group, "train",
            variant=settings.variant,
        )
        is_anomaly = is_anomaly_2d.any(axis=1)
    all_indices = _build_window_index(
        segment_ids, is_anomaly, cfg.window_size, cfg.prediction_horizon,
        skip_anomalous_windows=True, forecast_steps=cfg.forecast_steps,
    )

    n = len(all_indices)
    if n < 2:
        raise ValueError(
            f"Too few valid windows ({n}) for mission={mission!r} channel={channel!r}. "
            "Need at least 2 (1 train + 1 val). "
            "Increase the time series length or decrease window_size."
        )
    n_val = max(1, int(n * cfg.val_fraction))
    n_train = n - n_val

    train_ds = WindowedSequenceDataset(
        values, all_indices[:n_train], cfg.window_size, cfg.prediction_horizon,
        cfg.forecast_steps,
    )
    val_ds = WindowedSequenceDataset(
        values, all_indices[n_train:], cfg.window_size, cfg.prediction_horizon,
        cfg.forecast_steps,
    )

    _pin = torch.cuda.is_available()
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=_pin,
    )
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=_pin,
    )
    return train_loader, val_loader


def _window_any_anomalous(
    is_anomaly: np.ndarray[Any, Any],
    indices: np.ndarray[Any, Any],
    span: int,
) -> np.ndarray[Any, Any]:
    """Per-window OR of ``is_anomaly`` over ``[s, s+span)`` via a prefix sum.

    Generalizes the pre-021 single-channel prefix-sum trick over an optional
    trailing channel axis: a 1-D ``(N,)`` input returns ``(M,)``; a 2-D
    ``(N, C)`` per-channel input returns ``(M, C)`` with the OR computed
    independently per channel (so callers can report per-channel recall —
    see docs/plans/021-multivariate-telemanom.md risk 3).
    """
    cumsum = np.zeros((is_anomaly.shape[0] + 1, *is_anomaly.shape[1:]), dtype=np.int64)
    np.cumsum(is_anomaly, axis=0, out=cumsum[1:])
    result: np.ndarray[Any, Any] = (cumsum[indices + span] - cumsum[indices]) > 0
    return result


def make_test_dataloader(
    settings: Settings,
    mission: str,
    channel: str,
) -> tuple[
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
    np.ndarray[Any, Any],
    np.ndarray[Any, np.dtype[np.bool_]],
]:
    """Build a DataLoader for the test split, with aligned per-window metadata.

    Unlike the train DataLoader, anomalous windows are **not** skipped —
    evaluation requires all windows, including those overlapping anomalies.

    Multivariate (docs/plans/021-multivariate-telemanom.md): same channel
    group resolution as make_dataloaders. ``window_is_anomaly`` is per
    channel (not OR'd across channels) so the caller can score and report
    each channel separately — see model/scoring.py.

    Returns:
        loader:            DataLoader over all valid (cross-segment-free) windows.
        target_timestamps: (M,) datetime64[ns] — timestamp at each window's
                           target position (index s + W + H - 1).
        window_is_anomaly: (M,) bool, or (M, C) bool for a multivariate group
                           — True iff any step in [s, s+W+H) is anomalous
                           (``any(...)`` semantics; matches Phase 3
                           window-overlap definition for metric continuity).

    Args:
        settings: Fully resolved Settings.
        mission:  Mission name, e.g. ``"ESA-Mission1"``.
        channel:  Channel ID, e.g. ``"channel_1"`` (or a subsystem key — see
                  make_dataloaders).
    """
    cfg = settings.model
    group = _resolve_channel_group(settings, channel)
    if len(group) == 1:
        values, segment_ids, is_anomaly, timestamps = load_series_parquet(
            settings.preprocess.processed_data_dir, mission, group[0], "test",
            variant=settings.variant,
        )
        index_is_anomaly = is_anomaly
    else:
        values, segment_ids, is_anomaly, timestamps = load_multichannel_series_parquet(
            settings.preprocess.processed_data_dir, mission, group, "test",
            variant=settings.variant,
        )
        index_is_anomaly = is_anomaly.any(axis=1)
    indices = _build_window_index(
        segment_ids, index_is_anomaly, cfg.window_size, cfg.prediction_horizon,
        skip_anomalous_windows=False, forecast_steps=cfg.forecast_steps,
    )

    span = window_span(cfg)
    # Attributed to the FIRST forecast target, not the last — see
    # first_target_offset(). Identical to pre-021.7 when forecast_steps == 1.
    target_timestamps = timestamps[indices + first_target_offset(cfg)]
    # ...but the anomaly OR spans the WHOLE window (inputs + every target):
    # a window is anomalous if any timestep it touches is.
    window_is_anomaly = _window_any_anomalous(is_anomaly, indices, span)

    ds = WindowedSequenceDataset(
        values, indices, cfg.window_size, cfg.prediction_horizon, cfg.forecast_steps
    )
    _pin = torch.cuda.is_available()
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        ds,
        batch_size=cfg.inference_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=_pin,
    )
    return loader, target_timestamps, window_is_anomaly


def load_window_labels(
    settings: Settings,
    mission: str,
    channel: str,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    """Return per-window anomaly labels for the test split without building a DataLoader.

    Uses the same windowing logic as make_test_dataloader() but has no torch
    dependency — suitable for import in numpy-only Ray Tune trial functions
    (Phase 5).

    Multivariate (docs/plans/021-multivariate-telemanom.md): same channel
    group resolution as make_dataloaders; returns (M,) for a single channel
    or (M, C) per channel (not OR'd) for a multivariate group.

    Returns:
        bool array — True iff any timestep in the window+horizon span
        is anomalous (any() semantics; matches make_test_dataloader).

    Args:
        settings: Fully resolved Settings.
        mission:  Mission name, e.g. ``"ESA-Mission1"``.
        channel:  Channel ID, e.g. ``"channel_1"`` (or a subsystem key — see
                  make_dataloaders).
    """
    cfg = settings.model
    group = _resolve_channel_group(settings, channel)
    if len(group) == 1:
        _, segment_ids, is_anomaly, _ = load_series_parquet(
            settings.preprocess.processed_data_dir, mission, group[0], "test",
            variant=settings.variant,
        )
        index_is_anomaly = is_anomaly
    else:
        _, segment_ids, is_anomaly, _ = load_multichannel_series_parquet(
            settings.preprocess.processed_data_dir, mission, group, "test",
            variant=settings.variant,
        )
        index_is_anomaly = is_anomaly.any(axis=1)
    indices = _build_window_index(
        segment_ids, index_is_anomaly, cfg.window_size, cfg.prediction_horizon,
        skip_anomalous_windows=False, forecast_steps=cfg.forecast_steps,
    )
    span = window_span(cfg)
    return _window_any_anomalous(is_anomaly, indices, span)


def load_window_labels_from_metadata(
    settings: Settings,
    segment_ids: np.ndarray[Any, np.dtype[np.int32]],
    is_anomaly: np.ndarray[Any, np.dtype[np.bool_]],
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    """Pure computation of load_window_labels given preloaded metadata arrays.

    The labels counterpart of window_target_timestamps_from_metadata, and the
    piece that was missing from the preload API: a caller holding a channel's
    (segment_ids, is_anomaly, timestamps) could already avoid re-reading the
    partition for timeline, cutoff, and target timestamps, but not for labels
    — so it re-read anyway (docs/reviews/021, item D1).

    Uses the same ``_build_window_index`` / ``_window_any_anomalous`` pair as
    load_window_labels, so the two agree by construction rather than by being
    kept in step.

    Note this is the SINGLE-channel shape. A multivariate group's labels come
    from load_window_labels, which resolves the group and joins first; HPO and
    the threshold grid both operate on saved per-channel 1-D error arrays and
    therefore run without input_channels set.
    """
    cfg = settings.model
    indices = _build_window_index(
        segment_ids, is_anomaly, cfg.window_size, cfg.prediction_horizon,
        skip_anomalous_windows=False, forecast_steps=cfg.forecast_steps,
    )
    return _window_any_anomalous(is_anomaly, indices, window_span(cfg))


def window_target_timestamps_from_metadata(
    settings: Settings,
    segment_ids: np.ndarray[Any, np.dtype[np.int32]],
    is_anomaly: np.ndarray[Any, np.dtype[np.bool_]],
    timestamps: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Pure computation of window_target_timestamps given preloaded metadata arrays.

    Split out of window_target_timestamps so a caller that already holds a
    channel's (segment_ids, is_anomaly, timestamps) — loaded once via
    load_series_metadata() — can reuse it across multiple calls (timeline,
    HPO cutoff, untuned + tuned detection reconstruction) instead of
    re-reading the parquet partition every time (docs/plans/019 P2/P3).
    """
    cfg = settings.model
    indices = _build_window_index(
        segment_ids, is_anomaly, cfg.window_size, cfg.prediction_horizon,
        skip_anomalous_windows=False, forecast_steps=cfg.forecast_steps,
    )
    result: np.ndarray[Any, Any] = timestamps[indices + first_target_offset(cfg)]
    return result


def window_target_timestamps(
    settings: Settings,
    mission: str,
    channel: str,
) -> np.ndarray[Any, Any]:
    """Return per-window target timestamps for the test split without a DataLoader.

    Uses the same windowing logic as make_test_dataloader() but has no torch
    dependency — lets esa_adb.detections re-derive detection intervals from a
    scoring run's logged errors.npy/threshold.npy without re-running inference
    (window start indices are deterministic given window_size and
    prediction_horizon, so this reproduces make_test_dataloader's
    target_timestamps exactly).

    Reads only segment_ids/is_anomaly/timestamps (load_series_metadata, not
    load_series_parquet) — this function never uses the values column. A
    caller making several calls for the same channel (e.g. build_report)
    should instead preload once via load_series_metadata() and call
    window_target_timestamps_from_metadata() directly.

    Multivariate (docs/plans/021-multivariate-telemanom.md): same channel
    group resolution as make_dataloaders.

    Returns:
        (M,) datetime64[ns] — timestamp at each window's target position
        (index s + W + H - 1), aligned 1:1 with load_window_labels() and with
        the errors.npy/threshold.npy arrays score_channel() logs.

    Args:
        settings: Fully resolved Settings.
        mission:  Mission name, e.g. ``"ESA-Mission1"``.
        channel:  Channel ID, e.g. ``"channel_1"`` (or a subsystem key — see
                  make_dataloaders).
    """
    group = _resolve_channel_group(settings, channel)
    if len(group) == 1:
        segment_ids, is_anomaly, timestamps = load_series_metadata(
            settings.preprocess.processed_data_dir, mission, group[0], "test",
            variant=settings.variant,
        )
    else:
        segment_ids, is_anomaly, timestamps = load_multichannel_series_metadata(
            settings.preprocess.processed_data_dir, mission, group, "test",
            variant=settings.variant,
        )
    return window_target_timestamps_from_metadata(settings, segment_ids, is_anomaly, timestamps)
