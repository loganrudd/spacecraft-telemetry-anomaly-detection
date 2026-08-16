"""Per-channel and mission-level detection intervals for the ESA-ADB report.

Reconstructs anomaly-flag intervals from an *existing* MLflow scoring run —
no re-inference. score_channel() (model/scoring.py) already logs the full-test
smoothed error array (errors.npy) and the dynamic threshold (threshold.npy);
window start indices are deterministic given window_size and
prediction_horizon, so flag_anomalies() + window_target_timestamps() exactly
reproduce what score_channel() itself computed, without loading the model or
running a forward pass.

mission_detection_intervals() unions per-channel intervals in time — this is
the "logical sum of ... detections across all target channels" the ESA-ADB
paper computes its event-wise metrics against (see esa_adb/metrics.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from spacecraft_telemetry.core.logging import get_logger
from spacecraft_telemetry.esa_adb.intervals import union as _union
from spacecraft_telemetry.mlflow_tracking import configure_mlflow, experiment_name
from spacecraft_telemetry.model.dataset import (
    window_target_timestamps,
    window_target_timestamps_from_metadata,
)
from spacecraft_telemetry.model.io import (
    bytes_to_errors,
    download_artifact_bytes,
    read_artifact_bytes,
)
from spacecraft_telemetry.model.scoring import flag_anomalies

if TYPE_CHECKING:
    from spacecraft_telemetry.core.config import Settings
    from spacecraft_telemetry.esa_adb.intervals import Interval
    from spacecraft_telemetry.esa_adb.offline import OfflineRunMap, RunSpec

log = get_logger(__name__)

# (segment_ids, is_anomaly, timestamps) — load_series_metadata()'s return
# shape, preloaded once per channel by esa_adb.report.build_report so
# per-channel detection reconstruction doesn't re-read the parquet partition
# for every scoring run (untuned + tuned) it processes (docs/plans/019 P2/P3).
SeriesMetadata = tuple[
    np.ndarray[Any, np.dtype[np.int32]], np.ndarray[Any, Any], np.ndarray[Any, Any]
]


def find_scoring_run(
    experiment: str,
    channel: str,
    tracking_uri: str,
    *,
    tuned: bool,
) -> str:
    """Return the run_id of the most recent matching scoring run for a channel.

    ``tuned=False`` selects the Hundman-defaults baseline — score_channel()
    only writes the ``tuned_from_run`` tag when called with a
    ``parent_hpo_run_id`` (see ray_fanout/tune.py), so "no tag" means
    "untuned". ``tuned=True`` selects the most recent run that does carry
    the tag.

    Raises:
        RuntimeError: No experiment, or no run matches — states the exact
            tags searched so a silently-skipped channel never quietly
            weakens the mission-level OR-aggregation this report depends on.
    """
    import mlflow

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        raise RuntimeError(
            f"No MLflow experiment named {experiment!r} at {tracking_uri!r}. "
            "Has `ray score` run for this mission yet?"
        )
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=f"tags.channel_id = '{channel}'",
        order_by=["attributes.start_time DESC"],
    )
    for run in runs:
        has_tag = "tuned_from_run" in run.data.tags
        if has_tag == tuned:
            return str(run.info.run_id)

    kind = "tuned" if tuned else "untuned (Hundman-defaults)"
    flag = "with --tuned-configs" if tuned else "without --tuned-configs"
    raise RuntimeError(
        f"No {kind} scoring run found for channel={channel!r} in "
        f"experiment={experiment!r} (searched tags.channel_id='{channel}', "
        f"{'presence' if tuned else 'absence'} of the tuned_from_run tag). "
        f"Run `spacecraft-telemetry ray score` for this channel {flag}."
    )


def _true_runs(flags: np.ndarray[Any, Any]) -> list[tuple[int, int]]:
    """Return (start, end) half-open index ranges of contiguous True runs."""
    padded = np.concatenate(([False], flags.astype(bool), [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    return list(zip(starts, ends, strict=False))


def _flags_to_intervals(
    flags: np.ndarray[Any, Any],
    target_timestamps: np.ndarray[Any, Any],
) -> list[Interval]:
    """Convert window-index flag runs to half-open time intervals.

    A flagged run [s, e) in window-index space becomes
    [target_timestamps[s], target_timestamps[e-1] + step), where ``step`` is
    the median positive gap between consecutive target timestamps across the
    whole channel. Using the median (not target_timestamps[e], which doesn't
    exist when e is the last window) makes every run's closing edge robust to
    the boundary case without needing a lookahead sample.

    ``target_timestamps`` is a tz-naive numpy datetime64 array (PyArrow's
    to_numpy() drops the tz label from the UTC-typed Parquet column even
    though the instants are UTC) — localized to UTC here so the result is
    comparable to esa_adb.events' tz-aware pd.Timestamp intervals (read_labels
    parses with utc=True).
    """
    runs = _true_runs(flags)
    if not runs:
        return []

    diffs = np.diff(target_timestamps)
    positive = diffs[diffs > np.timedelta64(0, "ns")]
    step = pd.Timedelta(np.median(positive)) if len(positive) else pd.Timedelta(1, "ns")

    intervals: list[Interval] = []
    for s, e in runs:
        start_ts = pd.Timestamp(target_timestamps[s]).tz_localize("UTC")
        end_ts = pd.Timestamp(target_timestamps[e - 1]).tz_localize("UTC") + step
        intervals.append((start_ts, end_ts))
    return intervals


def _intervals_from_arrays(
    settings: Settings,
    mission: str,
    channel: str,
    *,
    smoothed: np.ndarray[Any, Any],
    threshold: np.ndarray[Any, Any],
    min_run_length: int,
    min_error_value: float = 0.0,
    source: str,
    metadata: SeriesMetadata | None = None,
) -> list[Interval]:
    """Re-derive flags from a scoring run's arrays and map them onto timestamps.

    Shared by the MLflow-backed and offline paths so both reconstruct flags
    through the identical flag_anomalies() call score_channel() itself uses.

    ``min_error_value`` MUST match what the scoring run applied, or the
    reconstructed detections silently diverge from the ones score_channel
    produced. It defaults to 0.0 for runs logged before the parameter existed,
    which is exactly the behaviour those runs had.

    Args:
        metadata: When given, reuse this channel's preloaded
            (segment_ids, is_anomaly, timestamps) instead of re-reading the
            parquet partition (see esa_adb.report.build_report, docs/plans/019
            P2/P3) — a channel typically has both an untuned and a tuned
            scoring run reconstructed, and the partition is identical for both.

    Raises:
        ValueError: The scoring run's window count doesn't match the current
            processed test partition — a stale run or a settings.model
            mismatch would silently misalign flags to timestamps otherwise.
    """
    flags = flag_anomalies(smoothed, threshold, min_run_length, min_error_value)

    if metadata is not None:
        segment_ids, is_anomaly, timestamps = metadata
        target_timestamps = window_target_timestamps_from_metadata(
            settings, segment_ids, is_anomaly, timestamps
        )
    else:
        target_timestamps = window_target_timestamps(settings, mission, channel)
    if len(target_timestamps) != len(flags):
        raise ValueError(
            f"Window count mismatch for channel={channel!r}, source={source!r}: "
            f"errors.npy has {len(flags)} windows but the current processed "
            f"test partition yields {len(target_timestamps)}. The scoring "
            "run's settings.model.window_size may not match the current "
            "config, or the processed data changed since scoring. Re-run "
            "`spacecraft-telemetry ray score`."
        )

    return _flags_to_intervals(flags, target_timestamps)


def channel_detection_intervals(
    settings: Settings,
    mission: str,
    channel: str,
    run_id: str,
    *,
    metadata: SeriesMetadata | None = None,
) -> list[Interval]:
    """Reconstruct one channel's flagged-anomaly intervals from a scoring run.

    Downloads errors.npy / threshold.npy and the threshold_min_anomaly_len
    param logged by score_channel() for ``run_id``, re-derives flags via
    flag_anomalies() (the same function score_channel() itself calls), and
    maps them onto the current processed test partition's timestamps.

    Requires a reachable MLflow tracking server; see
    channel_detection_intervals_from_spec for the offline equivalent.

    Args:
        metadata: See _intervals_from_arrays — preloaded (segment_ids,
            is_anomaly, timestamps) to avoid re-reading the parquet partition.
    """
    import mlflow

    tracking_uri = settings.mlflow.tracking_uri
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    min_run_length = int(run.data.params["threshold_min_anomaly_len"])
    # Absent on runs logged before min_error_value existed — 0.0 is exactly the
    # behaviour those runs had, so the default reproduces them faithfully.
    min_error_value = float(run.data.params.get("min_error_value", 0.0))

    smoothed = bytes_to_errors(download_artifact_bytes(run_id, "errors.npy", tracking_uri))
    threshold = bytes_to_errors(download_artifact_bytes(run_id, "threshold.npy", tracking_uri))

    return _intervals_from_arrays(
        settings,
        mission,
        channel,
        smoothed=smoothed,
        threshold=threshold,
        min_run_length=min_run_length,
        min_error_value=min_error_value,
        source=f"run_id={run_id}",
        metadata=metadata,
    )


def channel_detection_intervals_from_spec(
    settings: Settings,
    mission: str,
    channel: str,
    spec: RunSpec,
    *,
    metadata: SeriesMetadata | None = None,
) -> list[Interval]:
    """Offline counterpart of channel_detection_intervals — no tracking server.

    Reads the staged errors.npy / threshold.npy at the paths in ``spec`` and
    takes threshold_min_anomaly_len and min_error_value from the spec (both are
    logged MLflow params, absent from threshold_config.json, so they cannot be
    recovered from the artifacts alone). See esa_adb/offline.py.

    Args:
        metadata: See _intervals_from_arrays — preloaded (segment_ids,
            is_anomaly, timestamps) to avoid re-reading the parquet partition.
    """
    smoothed = bytes_to_errors(read_artifact_bytes(spec.errors_path))
    threshold = bytes_to_errors(read_artifact_bytes(spec.threshold_path))

    return _intervals_from_arrays(
        settings,
        mission,
        channel,
        smoothed=smoothed,
        threshold=threshold,
        min_run_length=spec.threshold_min_anomaly_len,
        min_error_value=spec.min_error_value,
        source=f"offline run_id={spec.run_id}",
        metadata=metadata,
    )


def per_channel_detection_intervals(
    settings: Settings,
    mission: str,
    channels: list[str],
    *,
    tuned: bool,
    run_map: OfflineRunMap | None = None,
    metadata_by_channel: dict[str, SeriesMetadata] | None = None,
) -> dict[str, list[Interval]]:
    """Detection intervals for each channel, kept separate (channel identity preserved).

    esa_adb.metrics.channel_aware needs to know WHICH channel produced a
    detection, so it consumes this dict directly rather than the OR'd
    mission_detection_intervals() union.

    When ``run_map`` is supplied the MLflow tracking server is never contacted:
    runs are resolved from the map and their arrays read from staged paths
    (see esa_adb/offline.py). Otherwise runs are found by MLflow run tags.

    Args:
        metadata_by_channel: When given, reuse each channel's preloaded
            (segment_ids, is_anomaly, timestamps) instead of re-reading the
            parquet partition (see esa_adb.report.build_report, docs/plans/019
            P2/P3) — this function is called once per untuned/tuned variant,
            so without preloading, every channel's partition is read again
            for the second call.

    Raises:
        RuntimeError / KeyError: propagated from find_scoring_run() or
            OfflineRunMap.get() — any channel with no matching scoring run
            fails the whole report rather than being silently dropped, which
            would quietly weaken the OR-aggregation.
    """
    if run_map is None:
        # This is the branch where MLflow IS required — see report.py's
        # matching comment. A failure here must stay visible, not vanish
        # silently; find_scoring_run below still fails loudly if the tracking
        # URI ends up misconfigured, so it's safe to continue.
        try:
            configure_mlflow(settings)
        except Exception as exc:
            log.warning("esa_adb.detections.configure_mlflow_failed", error=str(exc))
        tracking_uri = settings.mlflow.tracking_uri
        exp = experiment_name(settings.model.model_type, "scoring", mission, settings.variant)

    result: dict[str, list[Interval]] = {}
    for channel in channels:
        metadata = metadata_by_channel[channel] if metadata_by_channel is not None else None
        if run_map is None:
            run_id = find_scoring_run(exp, channel, tracking_uri, tuned=tuned)
            channel_intervals = channel_detection_intervals(
                settings, mission, channel, run_id, metadata=metadata
            )
            source = run_id
        else:
            spec = run_map.get(channel, tuned=tuned)
            channel_intervals = channel_detection_intervals_from_spec(
                settings, mission, channel, spec, metadata=metadata
            )
            source = spec.run_id
        result[channel] = channel_intervals
        log.info(
            "esa_adb.detections.channel_done",
            mission=mission,
            channel=channel,
            tuned=tuned,
            run_id=source,
            offline=run_map is not None,
            n_intervals=len(channel_intervals),
        )
    return result


def mission_intervals_from_per_channel(per_channel: dict[str, list[Interval]]) -> list[Interval]:
    """OR-aggregate (time union) an already-fetched per-channel detections dict.

    This is the "logical sum ... across all target channels" the ESA-ADB
    paper computes its event-wise metrics against (§3.2.1) — the single
    biggest reason our per-channel macro-averaged metrics aren't directly
    comparable to the paper's numbers (see docs/plans/019).

    Split out of mission_detection_intervals so callers that already hold a
    per_channel_detection_intervals() result (esa_adb.report.build_report)
    can compose the union without re-fetching every channel's artifacts a
    second time — each fetch downloads errors.npy/threshold.npy and re-reads
    the test partition, so the duplicate call roughly doubled report runtime.
    """
    result: list[Interval] = []
    for intervals in per_channel.values():
        result = _union(result, intervals)
    return result


def mission_detection_intervals(
    settings: Settings,
    mission: str,
    channels: list[str],
    *,
    tuned: bool,
    run_map: OfflineRunMap | None = None,
) -> list[Interval]:
    """OR-aggregate detection intervals across ``channels`` (time union).

    Fetches per-channel detections itself — see mission_intervals_from_per_channel
    for the composable version that reuses an already-fetched dict.
    """
    per_channel = per_channel_detection_intervals(
        settings, mission, channels, tuned=tuned, run_map=run_map
    )
    return mission_intervals_from_per_channel(per_channel)
