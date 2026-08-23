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
from spacecraft_telemetry.core.metadata import load_channel_group_map
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

# Aliased: channel_detection_intervals takes `errors_artifact` /
# `threshold_artifact` as keyword PARAMETERS (callers pass them by name), so
# importing the model.io helpers under their own names would shadow them
# inside that function.
from spacecraft_telemetry.model.io import errors_artifact as _errors_artifact
from spacecraft_telemetry.model.io import threshold_artifact as _threshold_artifact
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
) -> Any:
    """Return the most recent matching scoring Run (mlflow.entities.Run) for a channel.

    ``tuned=False`` selects the Hundman-defaults baseline, ``tuned=True`` the
    most recent tuned run. "Tuned" means carrying either the ``tuned_from_run``
    tag (Ray Tune) or ``tuned_source`` (another search, e.g. the exhaustive
    grid in scripts/threshold_ceiling.py) — see :func:`_is_tuned_run`.

    Returns the full Run (not just its id) — ``client.search_runs`` already
    returns fully-populated Run objects, tags included, so a caller that
    also needs tags (e.g. :func:`tuned_provenance`) can read them off THIS
    object instead of paying a second ``client.get_run()`` round trip for a
    run this function already had in hand (docs/reviews/022, item A2). Use
    :func:`find_scoring_run_id` when only the id is wanted.

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
        if _is_tuned_run(run) == tuned:
            return run

    kind = "tuned" if tuned else "untuned (Hundman-defaults)"
    flag = "with --tuned-configs" if tuned else "without --tuned-configs"
    raise RuntimeError(
        f"No {kind} scoring run found for channel={channel!r} in "
        f"experiment={experiment!r} (searched tags.channel_id='{channel}', "
        f"{'presence' if tuned else 'absence'} of the tuned_from_run / "
        f"tuned_source tags). "
        f"Run `spacecraft-telemetry ray score` for this channel {flag}."
    )


def find_scoring_run_id(
    experiment: str,
    channel: str,
    tracking_uri: str,
    *,
    tuned: bool,
) -> str:
    """Thin str-run_id wrapper around :func:`find_scoring_run`.

    For callers that only want the id, not the full Run (e.g. an artifact
    path that gets logged/passed around, not queried for tags/params).
    """
    return str(find_scoring_run(experiment, channel, tracking_uri, tuned=tuned).info.run_id)


def _is_tuned_run(run: Any) -> bool:
    """True when a scoring run used tuned params rather than Hundman defaults.

    Two tags can carry that, and BOTH must count:

    - ``tuned_from_run`` — params came from a Ray Tune trial (the common case).
    - ``tuned_source``   — params came from another search with no MLflow run
      behind it, e.g. the exhaustive grid in scripts/threshold_ceiling.py.

    Checking only the first silently files a grid-tuned run as the *untuned*
    baseline, which is the row this report compares against the paper's
    no-tuning protocol — it would look spectacular and mean nothing.
    """
    tags = run.data.tags
    return "tuned_from_run" in tags or "tuned_source" in tags


def tuned_provenance(
    settings: Settings,
    mission: str,
    channels: list[str],
    *,
    run_map: OfflineRunMap | None = None,
) -> str | None:
    """Human-readable provenance of the tuned scoring runs behind ``channels``.

    docs/plans/022, stage 022.2b: the report used to hardcode "per-subsystem
    Ray Tune HPO scoring" for every tuned row, which was false whenever the
    config actually came from ``scripts/threshold_ceiling.py``'s exhaustive
    grid (as it did for every 021.7 row). This reads what actually produced
    each channel's tuned run from its MLflow tags instead of asserting it:

    - ``tuned_source`` present (a grid-produced config, or any future
      non-Tune search) -> that string verbatim, e.g. "scripts/
      threshold_ceiling.py exhaustive grid (mission)".
    - Only ``tuned_from_run`` present (a Ray Tune trial) -> "per-subsystem
      Ray Tune HPO scoring", the one case where the old hardcoded claim was
      actually true.
    - Neither tag on any channel -> None (report renders an honest
      "provenance unavailable" line rather than falling back to the old
      claim).

    All channels normally share one tuned config — one subsystem's Ray Tune
    sweep, or threshold_ceiling.py's "first channel wins, the rest must
    agree" grid (docs/plans/022 stage 022.2b, Open Question 2) — so every
    channel is read and compared, rather than trusting one. A mismatch (a
    mixed-config arm — should be rare given run_all_sweeps' one-config-per-
    subsystem and threshold_ceiling.py's mixed-config hard-fail, but not
    impossible, e.g. a partial re-tune) renders a "mixed: ..." summary
    naming every distinct source and how many channels used it, and logs the
    full per-channel map at warning level (docs/reviews/022 stage 2.2).
    Refusing to render at all — the previous behaviour — traded the one
    situation where you most want to see the numbers AND the discrepancy for
    no report at all; asserting uniformity without checking it would be the
    misstatement this function exists to fix, but a checked, disclosed
    mismatch is not that.

    Args:
        run_map: Offline mode (esa_adb/offline.py) — MLflow is never
            contacted, and run *tags* live only in the tracking backend, so
            this returns None unconditionally. The caller must render an
            honest "provenance unavailable in offline mode" line, not fall
            back to the old hardcoded claim behind a harder-to-see branch.
    """
    if run_map is not None or not channels:
        return None

    try:
        configure_mlflow(settings)
    except Exception as exc:
        log.warning(
            "esa_adb.detections.tuned_provenance.configure_mlflow_failed", error=str(exc)
        )

    exp = experiment_name(settings.model.model_type, "scoring", mission, settings.variant)

    provenance_by_channel: dict[str, str] = {}
    for channel in channels:
        # Reads tags off the SAME Run object _resolve_scoring_run's search
        # already returned, rather than a second, independently re-resolved
        # client.get_run() call (docs/reviews/022, item A2) — search_runs
        # already returns fully-populated Run objects.
        run, _is_multivariate = _resolve_scoring_run(settings, mission, exp, channel, tuned=True)
        tags = run.data.tags
        source = tags.get("tuned_source")
        provenance_by_channel[channel] = source if source else "per-subsystem Ray Tune HPO scoring"

    distinct = set(provenance_by_channel.values())
    if len(distinct) > 1:
        log.warning(
            "esa_adb.detections.tuned_provenance.mixed_arm",
            provenance_by_channel=provenance_by_channel,
        )
        counts: dict[str, int] = {}
        for source in provenance_by_channel.values():
            counts[source] = counts.get(source, 0) + 1
        parts = [
            f"{n} channel{'' if n == 1 else 's'} {source!r}" for source, n in counts.items()
        ]
        return "mixed: " + ", ".join(parts)
    return next(iter(distinct))


def _find_multivariate_scoring_run(
    settings: Settings,
    mission: str,
    experiment: str,
    channel: str,
    *,
    tuned: bool,
) -> Any | None:
    """Locate the scoring Run for a channel scored inside a multivariate group.

    A multivariate scoring run deliberately carries no ``channel_id`` tag
    (model/scoring.py — the run's key is a group, not a real channel), so
    :func:`find_scoring_run` can never match it. This resolves the channel's
    GROUP KEY, takes the most recent run tagged with it, and — critically —
    confirms the channel is actually listed in that run's ``channels`` tag
    before trusting it: the key alone does not prove this particular channel
    was in the scored group, since a narrower channel subset could reuse it.

    The lookup uses ``load_channel_group_map`` because the run's ``subsystem``
    tag holds whatever key the model was scored under — ``channel if
    is_multivariate`` in model/scoring.py. Since docs/plans/023 stage .4 that
    is the channel group (``subsystem_6_g09``), not the subsystem
    (``subsystem_6``). Querying the subsystem matches nothing, every channel
    resolves to "not scored", and the mission report silently loses its
    multivariate detections. ``load_channel_group_map`` falls back to the
    subsystem map for missions with no measured grouping, so the pre-023 path
    is unchanged.

    Mirrors ray_fanout.tune._find_multivariate_scoring_run, which solves the
    identical lookup problem for HPO.

    Returns the Run (not just its id — see :func:`find_scoring_run`'s
    docstring for why), or None when no matching multivariate run exists.
    """
    import mlflow

    group_key = load_channel_group_map(settings, mission).get(channel)
    if group_key is None:
        return None
    client = mlflow.MlflowClient(tracking_uri=settings.mlflow.tracking_uri)
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        return None
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=f"tags.subsystem = '{group_key}'",
        order_by=["attributes.start_time DESC"],
    )
    for run in runs:
        # Same tuned/untuned convention as find_scoring_run — via the shared
        # predicate, so the multivariate path can't drift from the univariate
        # one on what counts as "tuned".
        if _is_tuned_run(run) != tuned:
            continue
        members = [c for c in (run.data.tags.get("channels") or "").split(",") if c]
        if channel in members:
            return run
    return None


def _resolve_scoring_run(
    settings: Settings,
    mission: str,
    experiment: str,
    channel: str,
    *,
    tuned: bool,
) -> tuple[Any, bool]:
    """Resolve a channel's scoring Run, trying univariate then multivariate.

    Shared by :func:`find_scoring_run_and_artifacts` (needs to know WHICH
    path matched, to pick the right errors/threshold artifact-path shape)
    and :func:`tuned_provenance` (needs only tags) so both read off the SAME
    resolved Run rather than each re-resolving independently — the
    duplication ``mission_intervals_from_per_channel``'s docstring records
    regretting ("the duplicate call roughly doubled report runtime"),
    applied to run lookup instead of artifact fetch (docs/reviews/022, item
    A2). A run logged between two independent resolutions could make
    provenance describe a different run than the one whose intervals were
    actually scored; sharing one resolution closes that race.

    Returns (run, is_multivariate).

    Raises:
        RuntimeError: When neither lookup finds a run — see
            find_scoring_run_and_artifacts for the combined message shape.
    """
    from contextlib import suppress

    try:
        run = find_scoring_run(experiment, channel, settings.mlflow.tracking_uri, tuned=tuned)
    except RuntimeError as univariate_exc:
        # Best-effort: a failing fallback (missing subsystem metadata,
        # unreachable backend) must degrade to "not found" so the original,
        # more precise univariate error is what the caller sees.
        multivariate_run: Any | None = None
        with suppress(Exception):
            multivariate_run = _find_multivariate_scoring_run(
                settings, mission, experiment, channel, tuned=tuned
            )
        if multivariate_run is None:
            raise RuntimeError(
                f"{univariate_exc} Also searched for a multivariate scoring run "
                f"(tags.subsystem = this channel's subsystem, with {channel!r} "
                "listed in the run's `channels` tag) and found none — so this is "
                "not simply a multivariate run being missed."
            ) from univariate_exc
        return multivariate_run, True
    return run, False


def find_scoring_run_and_artifacts(
    settings: Settings,
    mission: str,
    experiment: str,
    channel: str,
    *,
    tuned: bool,
) -> tuple[str, str, str]:
    """Return (run_id, errors_artifact, threshold_artifact) for a channel.

    Tries the univariate per-channel lookup first — unchanged, and the common
    case, where both arrays sit at the run root. Falls back to the
    multivariate subsystem run, whose arrays are per-channel and therefore
    nested (``errors/{channel}.npy``, ``threshold/{channel}.npy`` — see
    model/scoring.py).

    Raises:
        RuntimeError: When neither lookup finds a run. The message names both
            searches, because "no run found" for a multivariate mission would
            otherwise point at the wrong remedy entirely.
    """
    run, is_multivariate = _resolve_scoring_run(
        settings, mission, experiment, channel, tuned=tuned
    )
    run_id = str(run.info.run_id)
    if is_multivariate:
        log.info(
            "esa_adb.detections.multivariate_run_used",
            channel=channel,
            run_id=run_id,
            tuned=tuned,
        )
        return run_id, _errors_artifact(channel), _threshold_artifact(channel)
    return run_id, _errors_artifact(), _threshold_artifact()


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
    errors_artifact: str | None = None,
    threshold_artifact: str | None = None,
) -> list[Interval]:
    """Reconstruct one channel's flagged-anomaly intervals from a scoring run.

    Downloads the errors / threshold arrays and the threshold_min_anomaly_len
    param logged by score_channel() for ``run_id``, re-derives flags via
    flag_anomalies() (the same function score_channel() itself calls), and
    maps them onto the current processed test partition's timestamps.

    Requires a reachable MLflow tracking server; see
    channel_detection_intervals_from_spec for the offline equivalent.

    Args:
        metadata: See _intervals_from_arrays — preloaded (segment_ids,
            is_anomaly, timestamps) to avoid re-reading the parquet partition.
        errors_artifact / threshold_artifact: Artifact paths within the run.
            ``None`` (the default) resolves to the univariate run-root arrays
            via model.io; a multivariate run stores them per channel since one
            run covers a whole subsystem group — see
            find_scoring_run_and_artifacts, which supplies those paths from the
            same model.io helpers the writer uses.
    """
    import mlflow

    errors_path = errors_artifact if errors_artifact is not None else _errors_artifact()
    threshold_path = (
        threshold_artifact if threshold_artifact is not None else _threshold_artifact()
    )

    tracking_uri = settings.mlflow.tracking_uri
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    min_run_length = int(run.data.params["threshold_min_anomaly_len"])
    # Absent on runs logged before min_error_value existed — 0.0 is exactly the
    # behaviour those runs had, so the default reproduces them faithfully.
    min_error_value = float(run.data.params.get("min_error_value", 0.0))

    smoothed = bytes_to_errors(download_artifact_bytes(run_id, errors_path, tracking_uri))
    threshold = bytes_to_errors(
        download_artifact_bytes(run_id, threshold_path, tracking_uri)
    )

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
        # silently; find_scoring_run_and_artifacts below still fails loudly if
        # the tracking URI ends up misconfigured, so it's safe to continue.
        # (It reads the URI from settings itself, so none is bound here.)
        try:
            configure_mlflow(settings)
        except Exception as exc:
            log.warning("esa_adb.detections.configure_mlflow_failed", error=str(exc))
        exp = experiment_name(settings.model.model_type, "scoring", mission, settings.variant)

    result: dict[str, list[Interval]] = {}
    for channel in channels:
        metadata = metadata_by_channel[channel] if metadata_by_channel is not None else None
        if run_map is None:
            run_id, errors_artifact, threshold_artifact = find_scoring_run_and_artifacts(
                settings, mission, exp, channel, tuned=tuned
            )
            channel_intervals = channel_detection_intervals(
                settings, mission, channel, run_id, metadata=metadata,
                errors_artifact=errors_artifact,
                threshold_artifact=threshold_artifact,
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
