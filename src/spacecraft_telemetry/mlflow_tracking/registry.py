"""MLflow Model Registry helpers — model-type-agnostic.

register_pytorch_model: log a PyTorch model artifact and create a ModelVersion.
promote:                set the @champion alias on a version (MLflow 3.x pattern).
latest_uri:             return the models:/{name}@champion URI for loading.

MLflow 3.x removed stage-based transitions (Staging/Production/Archived) in
favour of aliases.  The @champion alias marks the version that is active in
production.  Load it via models:/{name}@champion or
client.get_model_version_by_alias(name, "champion").
"""

from __future__ import annotations

from typing import Any

import mlflow
import mlflow.pytorch

from spacecraft_telemetry.core.logging import get_logger

log = get_logger(__name__)

CHAMPION_ALIAS = "champion"


def register_pytorch_model(
    *,
    model: Any,
    name: str,
    run_id: str,
    version_tags: dict[str, str] | None = None,
) -> Any:
    """Log a PyTorch model artifact and register it in the Model Registry.

    Logging goes through mlflow.pytorch.log_model (never manual torch.save).
    The new ModelVersion is queryable via models:/{name}@champion after
    promote() sets the alias.

    Must be called inside an open_run() context so there is an active run for
    the artifact to attach to.

    Args:
        model:        PyTorch model instance to serialize and log.
        name:         Registered model name from registered_model_name() in conventions.
        run_id:       ID of the active MLflow run (run.info.run_id from open_run).
        version_tags: Optional key/value tags to set on the new ModelVersion.
                      Use for params that must survive without a run link
                      (e.g. {"window_size": "250"}).

    Returns:
        The first mlflow.entities.model_registry.ModelVersion created for this
        run, or None if the version search returns nothing.
    """
    mlflow.pytorch.log_model(
        pytorch_model=model,
        name="model",
        registered_model_name=name,
    )
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{name}' and run_id='{run_id}'")
    mv = versions[0] if versions else None
    if mv is not None and version_tags:
        for k, v in version_tags.items():
            client.set_model_version_tag(name, mv.version, k, v)
    return mv


def promote(
    *,
    name: str,
    version: int | None = None,
) -> None:
    """Set the @champion alias on a registered model version.

    The @champion alias is the MLflow 3.x replacement for the Production stage.
    Only one version holds @champion at a time — setting it on a new version
    automatically removes it from the previous one.

    Args:
        name:    Registered model name.
        version: Version number to promote.  When None, resolves to the highest
                 registered version.

    Raises:
        ValueError: If no versions exist for ``name``.
    """
    client = mlflow.tracking.MlflowClient()
    if version is None:
        candidates = client.search_model_versions(f"name='{name}'")
        if not candidates:
            raise ValueError(
                f"No versions found for registered model {name!r}. "
                "Train at least one channel before promoting."
            )
        version = max(int(v.version) for v in candidates)

    client.set_registered_model_alias(name, CHAMPION_ALIAS, str(version))
    log.info("mlflow.registry.promoted", name=name, version=version, alias=CHAMPION_ALIAS)


def latest_uri(name: str) -> str:
    """Return the models:/{name}@champion URI for mlflow.pytorch.load_model.

    Args:
        name: Registered model name.
    """
    return f"models:/{name}@{CHAMPION_ALIAS}"
