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

from contextlib import suppress
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from spacecraft_telemetry.core.logging import get_logger
from spacecraft_telemetry.esa_adb.intervals import union as _union
from spacecraft_telemetry.mlflow_tracking import configure_mlflow, experiment_name
from spacecraft_telemetry.model.dataset import window_target_timestamps
from spacecraft_telemetry.model.io import bytes_to_errors, download_artifact_bytes
from spacecraft_telemetry.model.scoring import flag_anomalies

if TYPE_CHECKING:
    from spacecraft_telemetry.core.config import Settings
    from spacecraft_telemetry.esa_adb.intervals import Interval

log = get_logger(__name__)


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
    """
    runs = _true_runs(flags)
    if not runs:
        return []

    diffs = np.diff(target_timestamps)
    positive = diffs[diffs > np.timedelta64(0, "ns")]
    step = pd.Timedelta(np.median(positive)) if len(positive) else pd.Timedelta(1, "ns")

    intervals: list[Interval] = []
    for s, e in runs:
        start_ts = pd.Timestamp(target_timestamps[s])
        end_ts = pd.Timestamp(target_timestamps[e - 1]) + step
        intervals.append((start_ts, end_ts))
    return intervals


def channel_detection_intervals(
    settings: Settings,
    mission: str,
    channel: str,
    run_id: str,
) -> list[Interval]:
    """Reconstruct one channel's flagged-anomaly intervals from a scoring run.

    Downloads errors.npy / threshold.npy and the threshold_min_anomaly_len
    param logged by score_channel() for ``run_id``, re-derives flags via
    flag_anomalies() (the same function score_channel() itself calls), and
    maps them onto the current processed test partition's timestamps.

    Raises:
        ValueError: The scoring run's window count doesn't match the current
            processed test partition — a stale run or a settings.model
            mismatch would silently misalign flags to timestamps otherwise.
    """
    import mlflow

    tracking_uri = settings.mlflow.tracking_uri
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    min_run_length = int(run.data.params["threshold_min_anomaly_len"])

    smoothed = bytes_to_errors(download_artifact_bytes(run_id, "errors.npy", tracking_uri))
    threshold = bytes_to_errors(download_artifact_bytes(run_id, "threshold.npy", tracking_uri))
    flags = flag_anomalies(smoothed, threshold, min_run_length)

    target_timestamps = window_target_timestamps(settings, mission, channel)
    if len(target_timestamps) != len(flags):
        raise ValueError(
            f"Window count mismatch for channel={channel!r}, run_id={run_id!r}: "
            f"errors.npy has {len(flags)} windows but the current processed "
            f"test partition yields {len(target_timestamps)}. The scoring "
            "run's settings.model.window_size may not match the current "
            "config, or the processed data changed since scoring. Re-run "
            "`spacecraft-telemetry ray score`."
        )

    return _flags_to_intervals(flags, target_timestamps)


def mission_detection_intervals(
    settings: Settings,
    mission: str,
    channels: list[str],
    *,
    tuned: bool,
) -> list[Interval]:
    """OR-aggregate detection intervals across ``channels``.

    This is the "logical sum ... across all target channels" the ESA-ADB
    paper computes its event-wise metrics against (§3.2.1) — the single
    biggest reason our per-channel macro-averaged metrics aren't directly
    comparable to the paper's numbers (see docs/plans/019).

    Raises:
        RuntimeError: propagated from find_scoring_run() — any channel with
            no matching scoring run fails the whole report rather than being
            silently dropped from the union.
    """
    with suppress(Exception):
        configure_mlflow(settings)

    tracking_uri = settings.mlflow.tracking_uri
    exp = experiment_name(settings.model.model_type, "scoring", mission)

    result: list[Interval] = []
    for channel in channels:
        run_id = find_scoring_run(exp, channel, tracking_uri, tuned=tuned)
        channel_intervals = channel_detection_intervals(settings, mission, channel, run_id)
        result = _union(result, channel_intervals)
        log.info(
            "esa_adb.detections.channel_done",
            mission=mission,
            channel=channel,
            tuned=tuned,
            n_intervals=len(channel_intervals),
        )
    return result
