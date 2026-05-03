"""Generic MLflow run-logging helpers — model-type-agnostic.

All functions are safe to call when MLflow is unavailable:
  - configure_mlflow: sets the global tracking URI from Settings.
  - open_run: context manager that yields an ActiveRun or None on failure.
    Training / scoring continue normally either way; filesystem artifacts
    (via model/io.py) are always written regardless.
  - log_params / log_metrics_step / log_metrics_final / log_artifact_bytes:
    no-ops when there is no active run.

Nothing in this module knows what Telemanom is.  Callers pass generic
params / metrics dicts and receive run objects they can use for registry ops.
"""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator

import mlflow

from spacecraft_telemetry.core.config import Settings
from spacecraft_telemetry.core.logging import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)


def configure_mlflow(settings: Settings) -> None:
    """Apply tracking_uri and registry_uri from Settings to the global MLflow client.

    Call this once at process startup (CLI entrypoint, test fixture) before any
    open_run() calls.  Individual Ray worker processes call it via _with_abs_paths
    settings passed through ray.put().
    """
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    if settings.mlflow.registry_uri is not None:
        mlflow.set_registry_uri(settings.mlflow.registry_uri)


@contextmanager
def open_run(
    *,
    experiment: str,
    run_name: str,
    tags: dict[str, str],
    nested: bool = False,
) -> Generator[Any, None, None]:
    """Context manager that opens an MLflow run and yields it, or yields None on failure.

    On any MLflow connectivity or configuration error, a structured warning is
    emitted and None is yielded.  The caller's body executes in both cases —
    filesystem artifacts (via model/io.py) are always produced regardless of
    MLflow availability.

    Callers that need the run_id (e.g. for registry ops) must guard with
    ``if run is not None:``.  The log_* helpers in this module are no-ops when
    there is no active run and do not require that guard.

    Args:
        experiment: MLflow experiment name.  Created if it does not exist.
        run_name:   Human-readable name for this run (e.g. channel_id).
        tags:       Tag dict produced by common_tags() from conventions.py.
        nested:     True when this run is a child of an already-active run.

    Yields:
        mlflow.ActiveRun or None.
    """
    _run: Any = None
    try:
        mlflow.set_experiment(experiment)
        _run = mlflow.start_run(run_name=run_name, tags=tags, nested=nested)
    except Exception as exc:
        log.warning("mlflow.run.start_failed", experiment=experiment, error=str(exc))

    try:
        yield _run
    finally:
        if _run is not None:
            try:
                mlflow.end_run()
            except Exception:
                pass


def log_params(params: dict[str, Any]) -> None:
    """Log a params dict to the active run; no-op when there is no active run."""
    if mlflow.active_run() is not None:
        mlflow.log_params(params)


def log_metrics_step(metrics: dict[str, float], step: int) -> None:
    """Log metrics for a training step (e.g. per-epoch) to the active run."""
    if mlflow.active_run() is not None:
        mlflow.log_metrics(metrics, step=step)


def log_metrics_final(metrics: dict[str, float]) -> None:
    """Log summary metrics (no step) to the active run."""
    if mlflow.active_run() is not None:
        mlflow.log_metrics(metrics)


def log_artifact_bytes(data: bytes, artifact_file: str) -> None:
    """Write bytes as a named artifact in the active run.

    The artifact appears at ``artifact_file`` relative to the run's artifact
    root.  No-op when there is no active run.

    Args:
        data:          Raw bytes to write.
        artifact_file: Destination path within the artifact store,
                       e.g. "configs/model_config.json".
    """
    if mlflow.active_run() is None:
        return
    artifact_path_obj = Path(artifact_file)
    parent = str(artifact_path_obj.parent)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = Path(tmp_dir) / artifact_path_obj.name
        tmp_file.write_bytes(data)
        mlflow.log_artifact(str(tmp_file), artifact_path=parent if parent != "." else None)
