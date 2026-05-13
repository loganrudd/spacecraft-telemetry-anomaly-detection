"""Tests for model.io — MLflow-backed artifact helpers.

After the A1 pivot, model/io.py is a thin MLflow client wrapper.
Tests cover:
- errors_to_bytes / bytes_to_errors round-trip (serialisation helpers).
- threshold_to_bytes round-trip.
- download_artifact_bytes — fetches a run artifact via MlflowClient.
- find_latest_run_for_channel — returns the most recent run for a channel.
- load_model_for_scoring — loads a registered model + window_size from registry.
- load_scoring_params — reads the four threshold params from the scoring run.
- Discipline check: training.py and scoring.py must not call raw filesystem IO.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import mlflow  # noqa: E402

from spacecraft_telemetry.model.io import (  # noqa: E402
    ScoringParams,
    bytes_to_errors,
    download_artifact_bytes,
    errors_to_bytes,
    find_latest_run_for_channel,
    load_model_for_scoring,
    load_scoring_params,
    threshold_to_bytes,
)

# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def test_errors_round_trip() -> None:
    """errors_to_bytes / bytes_to_errors must round-trip without data loss."""
    original = np.array([0.1, 0.5, 1.2, 0.0, 3.7], dtype=np.float64)
    data = errors_to_bytes(original)
    recovered = bytes_to_errors(data)
    np.testing.assert_array_equal(original, recovered)


def test_threshold_to_bytes_round_trip() -> None:
    """threshold_to_bytes must produce .npy-compatible bytes."""
    original = np.linspace(0.0, 1.0, 20, dtype=np.float64)
    data = threshold_to_bytes(original)
    recovered = np.load(__import__("io").BytesIO(data))
    np.testing.assert_array_equal(original, recovered)


def test_errors_to_bytes_returns_bytes() -> None:
    arr = np.zeros(5, dtype=np.float32)
    assert isinstance(errors_to_bytes(arr), bytes)


# ---------------------------------------------------------------------------
# MLflow helpers — require an isolated tracking backend
# ---------------------------------------------------------------------------


@pytest.fixture()
def _mlflow_uri(tmp_path: Path):
    """Isolated per-test SQLite MLflow backend."""
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(uri)
    yield uri
    if mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.set_tracking_uri("")


def _log_artifact_in_run(tracking_uri: str, channel: str, artifact_name: str, data: bytes) -> str:
    """Helper: open a run tagged with channel_id, log an artifact, return run_id."""
    import tempfile

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("test-experiment")
    with (
        mlflow.start_run(tags={"channel_id": channel}) as run,
        tempfile.TemporaryDirectory() as tmp,
    ):
        p = Path(tmp) / artifact_name
        p.write_bytes(data)
        mlflow.log_artifact(str(p))
    return run.info.run_id


def test_download_artifact_bytes_retrieves_data(_mlflow_uri: str) -> None:
    """download_artifact_bytes must return the exact bytes that were logged."""
    payload = b"hello artifact"
    run_id = _log_artifact_in_run(_mlflow_uri, "channel_1", "test.bin", payload)
    result = download_artifact_bytes(run_id, "test.bin", _mlflow_uri)
    assert result == payload


def test_find_latest_run_for_channel_returns_most_recent(_mlflow_uri: str) -> None:
    """find_latest_run_for_channel must return the last run for a channel."""
    mlflow.set_tracking_uri(_mlflow_uri)
    mlflow.set_experiment("test-scoring")
    with mlflow.start_run(tags={"channel_id": "channel_1"}):
        mlflow.log_metric("step", 1)
    with mlflow.start_run(tags={"channel_id": "channel_1"}) as run2:
        mlflow.log_metric("step", 2)

    found = find_latest_run_for_channel("test-scoring", "channel_1", _mlflow_uri)
    assert found is not None
    assert found.info.run_id == run2.info.run_id


def test_find_latest_run_for_channel_returns_none_for_missing(_mlflow_uri: str) -> None:
    """find_latest_run_for_channel returns None when no run exists for the channel."""
    result = find_latest_run_for_channel("nonexistent-experiment", "channel_1", _mlflow_uri)
    assert result is None


def test_find_latest_run_for_channel_returns_none_when_channel_absent(_mlflow_uri: str) -> None:
    """Returns None when the experiment exists but no run is tagged for the channel."""
    mlflow.set_tracking_uri(_mlflow_uri)
    mlflow.set_experiment("test-scoring-2")
    with mlflow.start_run(tags={"channel_id": "channel_2"}):
        pass

    result = find_latest_run_for_channel("test-scoring-2", "channel_1", _mlflow_uri)
    assert result is None


def test_load_model_for_scoring_returns_model_and_window_size(_mlflow_uri: str) -> None:
    """load_model_for_scoring returns (model, window_size) from the registry."""
    from spacecraft_telemetry.core.config import ModelConfig
    from spacecraft_telemetry.model.architecture import build_model

    cfg = ModelConfig(hidden_dim=8, num_layers=1, dropout=0.0, window_size=20)
    model = build_model(cfg)

    mlflow.set_tracking_uri(_mlflow_uri)
    mlflow.set_experiment("test-training")
    with mlflow.start_run():
        mlflow.log_param("window_size", str(cfg.window_size))
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name="test-model",
        )

    device = torch.device("cpu")
    loaded_model, window_size = load_model_for_scoring("test-model", device, _mlflow_uri)
    assert window_size == 20
    model.eval()
    loaded_model.eval()
    x = torch.zeros(2, cfg.window_size, 1)
    with torch.no_grad():
        np.testing.assert_allclose(
            model(x).numpy(), loaded_model(x).numpy(), rtol=1e-5
        )


def test_load_model_for_scoring_raises_when_no_version(_mlflow_uri: str) -> None:
    """load_model_for_scoring raises RuntimeError when no registered versions exist."""
    mlflow.set_tracking_uri(_mlflow_uri)
    with pytest.raises(RuntimeError, match="No registered versions found"):
        load_model_for_scoring("nonexistent-model", torch.device("cpu"), _mlflow_uri)


# ---------------------------------------------------------------------------
# load_scoring_params — reads threshold hyperparams from the scoring run
# ---------------------------------------------------------------------------

_SCORING_PARAMS = {
    "error_smoothing_window": "12",
    "threshold_window": "50",
    "threshold_z": "2.5",
    "threshold_min_anomaly_len": "4",
}


def _log_scoring_run(
    tracking_uri: str,
    channel: str,
    mission: str,
    params: dict[str, str] | None = None,
) -> None:
    """Log a minimal scoring run tagged with channel_id and the threshold params."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(f"telemanom-scoring-{mission}")
    with mlflow.start_run(tags={"channel_id": channel}):
        mlflow.log_params(params or _SCORING_PARAMS)


def test_load_scoring_params_returns_params_from_run(_mlflow_uri: str) -> None:
    """load_scoring_params returns a ScoringParams with values from the scoring run."""
    _log_scoring_run(_mlflow_uri, "channel_1", "ESA-Mission1")

    result = load_scoring_params("channel_1", "ESA-Mission1", _mlflow_uri)

    assert isinstance(result, ScoringParams)
    assert result.error_smoothing_window == 12
    assert result.threshold_window == 50
    assert result.threshold_z == pytest.approx(2.5)
    assert result.threshold_min_anomaly_len == 4


def test_load_scoring_params_raises_when_no_run(_mlflow_uri: str) -> None:
    """load_scoring_params raises RuntimeError when no scoring run exists."""
    with pytest.raises(RuntimeError, match="No scoring run found"):
        load_scoring_params("channel_99", "ESA-Mission1", _mlflow_uri)


def test_load_scoring_params_raises_when_param_missing(_mlflow_uri: str) -> None:
    """RuntimeError when a required scoring param is absent from the run."""
    incomplete = {k: v for k, v in _SCORING_PARAMS.items() if k != "threshold_z"}
    _log_scoring_run(_mlflow_uri, "channel_2", "ESA-Mission1", params=incomplete)

    with pytest.raises(RuntimeError, match="threshold_z"):
        load_scoring_params("channel_2", "ESA-Mission1", _mlflow_uri)


# ---------------------------------------------------------------------------
# Discipline check: training.py and scoring.py must not call raw IO directly
# ---------------------------------------------------------------------------

_FORBIDDEN_PATTERNS = [
    r"\.write_bytes\(",
    r"torch\.save\(",
    r"np\.save\(",
    r"open\(.*[\"']w[\"']",
]


def _check_no_raw_io(source_path: Path) -> list[str]:
    """Return lines that contain a forbidden raw-IO call.

    Uses the AST to identify string-literal (docstring) lines and comment
    lines so that documentation mentioning these patterns doesn't false-positive.
    """
    import ast

    text = source_path.read_text()

    # Collect line numbers that belong to module/function docstrings.
    docstring_lines: set[int] = set()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            end: int = getattr(node.value, "end_lineno", None) or node.lineno
            for lineno in range(node.lineno, end + 1):
                docstring_lines.add(lineno)

    hits = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if lineno in docstring_lines:
            continue
        if line.lstrip().startswith("#"):
            continue
        for pat in _FORBIDDEN_PATTERNS:
            if re.search(pat, line):
                hits.append(f"{source_path.name}:{line.strip()!r}")
    return hits


def test_no_direct_filesystem_writes_in_training_or_scoring() -> None:
    """training.py and scoring.py must funnel all writes through MLflow APIs."""
    src = Path(__file__).parents[2] / "src" / "spacecraft_telemetry" / "model"
    violations: list[str] = []
    for module in ("training.py", "scoring.py"):
        path = src / module
        if path.exists():
            violations.extend(_check_no_raw_io(path))
    assert not violations, (
        "Raw filesystem writes found — use MLflow logging APIs instead:\n"
        + "\n".join(violations)
    )
