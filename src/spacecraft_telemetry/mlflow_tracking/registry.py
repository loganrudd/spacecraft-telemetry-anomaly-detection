"""MLflow Model Registry helpers — model-type-agnostic.

register_pytorch_model: log a PyTorch model artifact and create a ModelVersion.
promote:                transition a version to Staging / Production / Archived.
latest_uri:             return the models:/{name}/{stage} URI for loading.

Stage-based registry (transition_model_version_stage) is used rather than
MLflow 3.x aliases because the CLI and serving layer use the
``models:/{name}/{stage}`` URI format.  Migrate to
``set_registered_model_alias`` when upgrading to MLflow 4.x removes stages.
"""

from __future__ import annotations

from typing import Any

import mlflow
import mlflow.pytorch

from spacecraft_telemetry.core.logging import get_logger

log = get_logger(__name__)


def register_pytorch_model(
    *,
    model: Any,
    name: str,
    run_id: str,
) -> Any:
    """Log a PyTorch model artifact and register it in the Model Registry.

    Logging goes through mlflow.pytorch.log_model (never manual torch.save).
    The new ModelVersion is queryable via ``models:/{name}/{stage}`` after
    promote() assigns a stage.

    Must be called inside an open_run() context so there is an active run for
    the artifact to attach to.

    Args:
        model:   PyTorch model instance to serialize and log.
        name:    Registered model name from registered_model_name() in conventions.
        run_id:  ID of the active MLflow run (run.info.run_id from open_run).

    Returns:
        The first mlflow.entities.model_registry.ModelVersion created for this
        run, or the raw ModelInfo if the version search returns nothing.
    """
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="model",
        registered_model_name=name,
    )
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{name}' and run_id='{run_id}'")
    if versions:
        return versions[0]
    # Fallback: return the latest version if the run_id filter found nothing.
    all_versions = client.search_model_versions(f"name='{name}'")
    return all_versions[0] if all_versions else None


def promote(
    *,
    name: str,
    version: int | None = None,
    stage: str,
) -> None:
    """Transition a registered model version to the target stage.

    Args:
        name:    Registered model name.
        version: Version number to promote.  When None, resolves to the highest
                 non-Archived version.
        stage:   One of "Staging", "Production", "Archived".

    Raises:
        ValueError: If no promotable version exists for ``name``.
    """
    client = mlflow.tracking.MlflowClient()
    if version is None:
        candidates = [
            v for v in client.search_model_versions(f"name='{name}'")
            if v.current_stage != "Archived"
        ]
        if not candidates:
            raise ValueError(
                f"No promotable versions found for registered model {name!r}. "
                "Train at least one channel before promoting."
            )
        version = max(int(v.version) for v in candidates)

    client.transition_model_version_stage(
        name=name,
        version=str(version),
        stage=stage,
    )
    log.info("mlflow.registry.promoted", name=name, version=version, stage=stage)


def latest_uri(name: str, stage: str = "Production") -> str:
    """Return the models:/{name}/{stage} URI for mlflow.pytorch.load_model.

    Args:
        name:  Registered model name.
        stage: Stage to load from (default "Production").
    """
    return f"models:/{name}/{stage}"
