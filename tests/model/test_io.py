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
    ModelNotFoundError,
    ScoringParams,
    bytes_to_errors,
    download_artifact_bytes,
    errors_to_bytes,
    find_latest_run_for_channel,
    load_model_for_scoring,
    load_scoring_params,
    read_artifact_bytes,
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
# read_artifact_bytes — tracking-server-free path (esa_adb offline mode)
# ---------------------------------------------------------------------------


def test_read_artifact_bytes_reads_local_file(tmp_path: Path) -> None:
    path = tmp_path / "errors.npy"
    original = np.array([1.0, 2.0, 3.0])
    path.write_bytes(errors_to_bytes(original))

    data = read_artifact_bytes(str(path))
    np.testing.assert_array_equal(bytes_to_errors(data), original)


def test_read_artifact_bytes_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError, match="Artifact not found"):
        read_artifact_bytes("/nonexistent/path/errors.npy")


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


def test_download_artifact_bytes_reads_directly_from_gs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A gs:// run.info.artifact_uri must be read directly, bypassing the proxy.

    P1: client.download_artifacts() streams bytes through the MLflow tracking
    server (measured 33-49s for a ~60MB artifact); a direct GCS read is the
    fix. The proxy path (download_artifacts) must not be called at all here.
    """
    import mlflow

    class _FakeInfo:
        artifact_uri = "gs://bucket/mlflow/1/run123/artifacts"

    class _FakeRun:
        info = _FakeInfo()

    class _FakeClient:
        def __init__(self, tracking_uri: str) -> None:
            del tracking_uri

        def get_run(self, run_id: str) -> _FakeRun:
            del run_id
            return _FakeRun()

        def download_artifacts(self, *args: object, **kwargs: object) -> str:
            raise AssertionError(
                "proxy download must not be called when artifact_uri is gs://"
            )

    monkeypatch.setattr(mlflow, "MlflowClient", _FakeClient)

    captured: dict[str, str] = {}

    class _FakeUPath:
        def read_bytes(self) -> bytes:
            return b"gcs-bytes"

    def _fake_to_upath(value: str) -> _FakeUPath:
        captured["value"] = value
        return _FakeUPath()

    monkeypatch.setattr("spacecraft_telemetry.core.paths.to_upath", _fake_to_upath)

    result = download_artifact_bytes("run123", "errors.npy", "https://mlflow.example.run.app")

    assert result == b"gcs-bytes"
    assert captured["value"] == "gs://bucket/mlflow/1/run123/artifacts/errors.npy"


def test_download_artifact_bytes_falls_back_to_proxy_for_non_gs_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-gs:// artifact_uri (e.g. local file://) must use the proxy path."""
    import mlflow

    class _FakeInfo:
        artifact_uri = "mlflow-artifacts:/1/run123/artifacts"

    class _FakeRun:
        info = _FakeInfo()

    class _FakeClient:
        def __init__(self, tracking_uri: str) -> None:
            del tracking_uri

        def get_run(self, run_id: str) -> _FakeRun:
            del run_id
            return _FakeRun()

        def download_artifacts(self, run_id: str, artifact_path: str, dst_path: str) -> str:
            del run_id
            local = Path(dst_path) / artifact_path
            local.write_bytes(b"proxied-bytes")
            return str(local)

    monkeypatch.setattr(mlflow, "MlflowClient", _FakeClient)

    result = download_artifact_bytes("run123", "errors.npy", "https://mlflow.example.run.app")

    assert result == b"proxied-bytes"


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


def test_find_latest_run_for_channel_extra_filter_disambiguates_data_source(
    _mlflow_uri: str,
) -> None:
    """extra_filter isolates a nominal-tagged run from a later injected-tagged one.

    Mirrors ray_fanout.tune._load_nominal_errors: without the filter, "latest"
    would return the injected run even though the caller wants the nominal
    baseline for the false-positive-rate penalty.
    """
    mlflow.set_tracking_uri(_mlflow_uri)
    mlflow.set_experiment("test-scoring-3")
    with mlflow.start_run(tags={"channel_id": "channel_1", "data_source": "nominal"}) as nom_run:
        mlflow.log_metric("step", 1)
    with mlflow.start_run(tags={"channel_id": "channel_1", "data_source": "injected"}):
        mlflow.log_metric("step", 2)

    found = find_latest_run_for_channel(
        "test-scoring-3",
        "channel_1",
        _mlflow_uri,
        extra_filter="tags.data_source = 'nominal'",
    )
    assert found is not None
    assert found.info.run_id == nom_run.info.run_id


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

    # Set @champion alias — load_model_for_scoring only serves the champion version.
    client = mlflow.MlflowClient(_mlflow_uri)
    client.set_registered_model_alias("test-model", "champion", "1")

    device = torch.device("cpu")
    loaded_model, window_size = load_model_for_scoring("test-model", device, _mlflow_uri)
    assert window_size == 20
    model.eval()
    loaded_model.eval()
    x = torch.zeros(2, cfg.window_size, 1)
    with torch.no_grad():
        np.testing.assert_allclose(model(x).numpy(), loaded_model(x).numpy(), rtol=1e-5)


def test_load_model_for_scoring_raises_when_no_version(_mlflow_uri: str) -> None:
    """load_model_for_scoring raises ModelNotFoundError when no versions exist.

    Distinct from RuntimeError so the Ray score task can map it to status=skipped
    and the serving layer can skip the channel rather than crash at startup.
    """
    mlflow.set_tracking_uri(_mlflow_uri)
    with pytest.raises(ModelNotFoundError, match="No registered versions found"):
        load_model_for_scoring("nonexistent-model", torch.device("cpu"), _mlflow_uri)


def test_load_model_for_scoring_raises_when_not_promoted(_mlflow_uri: str) -> None:
    """load_model_for_scoring raises RuntimeError when no version is in Production."""
    from spacecraft_telemetry.core.config import ModelConfig
    from spacecraft_telemetry.model.architecture import build_model

    cfg = ModelConfig(hidden_dim=8, num_layers=1, dropout=0.0, window_size=20)
    model = build_model(cfg)

    mlflow.set_tracking_uri(_mlflow_uri)
    mlflow.set_experiment("test-training-unpromoted")
    with mlflow.start_run():
        mlflow.log_param("window_size", str(cfg.window_size))
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name="test-model-unpromoted",
        )
    # Deliberately not promoting — should raise with a helpful message.
    with pytest.raises(RuntimeError, match="No @champion alias"):
        load_model_for_scoring("test-model-unpromoted", torch.device("cpu"), _mlflow_uri)


def test_load_model_for_scoring_without_champion_loads_latest(_mlflow_uri: str) -> None:
    """require_champion=False loads the latest version even without @champion alias."""
    from spacecraft_telemetry.core.config import ModelConfig
    from spacecraft_telemetry.model.architecture import build_model

    cfg = ModelConfig(hidden_dim=8, num_layers=1, dropout=0.0, window_size=20)
    model = build_model(cfg)

    mlflow.set_tracking_uri(_mlflow_uri)
    mlflow.set_experiment("test-training-no-promote")
    with mlflow.start_run():
        mlflow.log_param("window_size", str(cfg.window_size))
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name="test-model-no-promote",
        )
    # No promotion — but require_champion=False should succeed.
    loaded_model, window_size = load_model_for_scoring(
        "test-model-no-promote", torch.device("cpu"), _mlflow_uri, require_champion=False
    )
    assert window_size == 20
    assert loaded_model is not None


# ---------------------------------------------------------------------------
# load_model_contract — what the model was TRAINED with (docs/reviews/021, 2.1)
# ---------------------------------------------------------------------------


def _register_model(
    tracking_uri: str,
    name: str,
    *,
    window_size: int = 20,
    forecast_steps: int | None = None,
    channels: list[str] | None = None,
    version_tags: dict[str, str] | None = None,
) -> None:
    """Register one model version, logging the contract the way training does.

    ``forecast_steps=None`` / ``channels=None`` omit the entries entirely,
    reproducing a pre-021.7 / pre-021 model version rather than writing a
    default — the point of the absent-key tests is that those versions exist
    in the real registry and must keep loading.
    """
    from spacecraft_telemetry.core.config import ModelConfig
    from spacecraft_telemetry.model.architecture import build_model

    cfg = ModelConfig(hidden_dim=8, num_layers=1, dropout=0.0, window_size=window_size)
    model = build_model(cfg)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(f"test-training-{name}")
    run_tags = {"channels": ",".join(channels)} if channels else {}
    with mlflow.start_run(tags=run_tags):
        mlflow.log_param("window_size", str(window_size))
        if forecast_steps is not None:
            mlflow.log_param("forecast_steps", str(forecast_steps))
        mlflow.pytorch.log_model(
            pytorch_model=model, artifact_path="model", registered_model_name=name
        )
    if version_tags:
        client = mlflow.MlflowClient(tracking_uri)
        for k, v in version_tags.items():
            client.set_model_version_tag(name, "1", k, v)


def test_load_model_contract_round_trips_group_and_horizon(_mlflow_uri: str) -> None:
    """A multivariate H=10 model must report back exactly what it trained on."""
    from spacecraft_telemetry.model.io import load_model_contract

    group = ["channel_41", "channel_42", "channel_43"]
    _register_model(
        _mlflow_uri, "contract-mv", window_size=20, forecast_steps=10, channels=group
    )

    contract = load_model_contract("contract-mv", _mlflow_uri)

    assert contract.window_size == 20
    assert contract.forecast_steps == 10
    assert contract.channels == tuple(group)


def test_load_model_contract_preserves_channel_ORDER(_mlflow_uri: str) -> None:
    """Order is the contract, not membership — a reordered group is a
    different model (docs/plans/021: a silent reorder is 'catastrophic')."""
    from spacecraft_telemetry.model.io import load_model_contract

    group = ["channel_43", "channel_41", "channel_42"]
    _register_model(_mlflow_uri, "contract-order", forecast_steps=3, channels=group)

    assert load_model_contract("contract-order", _mlflow_uri).channels == tuple(group)


def test_load_model_contract_defaults_for_pre_021_version(_mlflow_uri: str) -> None:
    """A version with neither key is a genuinely H=1 univariate model — it must
    keep loading, and describe itself accurately."""
    from spacecraft_telemetry.model.io import load_model_contract

    _register_model(_mlflow_uri, "contract-legacy", window_size=250)

    contract = load_model_contract("contract-legacy", _mlflow_uri)

    assert contract.window_size == 250
    assert contract.forecast_steps == 1
    assert contract.channels is None


def test_load_model_contract_falls_back_to_version_tags(_mlflow_uri: str) -> None:
    """Version tags are the second source, for versions whose run link is
    missing — the same dual-sourcing window_size already relies on."""
    from spacecraft_telemetry.model.io import load_model_contract

    _register_model(
        _mlflow_uri,
        "contract-vtags",
        window_size=20,
        version_tags={"forecast_steps": "7", "channels": "channel_1,channel_2"},
    )

    contract = load_model_contract("contract-vtags", _mlflow_uri)

    assert contract.forecast_steps == 7
    assert contract.channels == ("channel_1", "channel_2")


def test_load_model_contract_raises_when_no_version(_mlflow_uri: str) -> None:
    from spacecraft_telemetry.model.io import load_model_contract

    with pytest.raises(ModelNotFoundError, match="No registered versions found"):
        load_model_contract("contract-missing", _mlflow_uri)


def test_load_model_contract_agrees_with_load_model_for_scoring(_mlflow_uri: str) -> None:
    """The two functions must describe the SAME version — they share
    _resolve_model_version precisely so this cannot drift."""
    from spacecraft_telemetry.model.io import load_model_contract

    _register_model(_mlflow_uri, "contract-agree", window_size=20, forecast_steps=4)

    _model, window_size = load_model_for_scoring(
        "contract-agree", torch.device("cpu"), _mlflow_uri, require_champion=False
    )
    assert load_model_contract("contract-agree", _mlflow_uri).window_size == window_size


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
        "Raw filesystem writes found — use MLflow logging APIs instead:\n" + "\n".join(violations)
    )
