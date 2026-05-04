"""Tests for model.scoring — pure-numpy threshold functions (no torch required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spacecraft_telemetry.model.scoring import (
    dynamic_threshold,
    evaluate,
    flag_anomalies,
    smooth_errors,
)

# ---------------------------------------------------------------------------
# smooth_errors
# ---------------------------------------------------------------------------


def test_smooth_errors_is_ewma() -> None:
    """Verify against the recursive EWMA formula with span=2 (alpha=2/3).

    alpha = 2 / (span + 1) = 2/3
    s[0] = 1
    s[1] = 2/3*2 + 1/3*1  = 5/3
    s[2] = 2/3*3 + 1/3*5/3 = 23/9
    s[3] = 2/3*4 + 1/3*23/9 = 95/27
    s[4] = 2/3*5 + 1/3*95/27 = 365/81
    """
    errors = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = smooth_errors(errors, span=2)
    expected = np.array([1, 5 / 3, 23 / 9, 95 / 27, 365 / 81])
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_smooth_errors_uses_absolute_values() -> None:
    """Negative errors should produce the same result as their absolute counterparts."""
    pos = smooth_errors(np.array([1.0, 2.0, 3.0]), span=3)
    neg = smooth_errors(np.array([-1.0, -2.0, -3.0]), span=3)
    np.testing.assert_allclose(pos, neg)


def test_smooth_errors_single_element() -> None:
    result = smooth_errors(np.array([5.0]), span=10)
    assert result.shape == (1,)
    np.testing.assert_allclose(result[0], 5.0)


# ---------------------------------------------------------------------------
# dynamic_threshold
# ---------------------------------------------------------------------------


def test_dynamic_threshold_first_entry_is_inf() -> None:
    """Position 0 has no history — threshold must be inf so it never flags."""
    smoothed = np.ones(10, dtype=np.float64)
    thresh = dynamic_threshold(smoothed, window=5, z=3.0)
    assert thresh[0] == np.inf, f"Expected inf at index 0, got {thresh[0]}"


def test_dynamic_threshold_no_nan_after_warmup() -> None:
    """No NaN values anywhere in the output (warmup uses min_periods=1)."""
    smoothed = np.random.default_rng(0).standard_normal(50)
    thresh = dynamic_threshold(smoothed, window=10, z=3.0)
    assert not np.any(np.isnan(thresh)), "Unexpected NaN in threshold array"


def test_dynamic_threshold_shape_matches_input() -> None:
    smoothed = np.ones(20)
    thresh = dynamic_threshold(smoothed, window=5, z=2.0)
    assert thresh.shape == smoothed.shape


def test_dynamic_threshold_steady_state_uses_previous_window() -> None:
    """After warmup, threshold at t must be based on smoothed[t-window : t], not t."""
    # Constant background then a spike at position 15; threshold at 15 should
    # be computed from positions 5..14 (constant background), so the spike
    # should NOT contaminate its own threshold.
    smoothed = np.zeros(30, dtype=np.float64)
    smoothed[15] = 100.0  # large spike
    thresh = dynamic_threshold(smoothed, window=10, z=3.0)
    # threshold[15] is based on smoothed[5:15] which are all 0 → threshold ≈ 0
    assert thresh[15] < 1.0, (
        f"threshold[15]={thresh[15]:.4f} should reflect background (0), not the spike"
    )


# ---------------------------------------------------------------------------
# flag_anomalies
# ---------------------------------------------------------------------------


def test_flag_anomalies_drops_short_runs() -> None:
    # Run of 1 at index 2; run of 5 at indices 5-9.
    smoothed = np.array([0, 0, 2, 0, 0, 2, 2, 2, 2, 2], dtype=np.float64)
    threshold = np.ones(10, dtype=np.float64)
    flags = flag_anomalies(smoothed, threshold, min_run_length=3)

    assert not flags[2], "Run of length 1 should be dropped"
    assert np.all(flags[5:10]), "Run of length 5 should be kept"
    assert not np.any(flags[:2]), "Background should not be flagged"
    assert not flags[3], "Background should not be flagged"
    assert not flags[4], "Background should not be flagged"


def test_flag_anomalies_keeps_exact_min_length_run() -> None:
    # Run of exactly min_run_length=3 at indices 1-3.
    smoothed = np.array([0, 2, 2, 2, 0], dtype=np.float64)
    threshold = np.ones(5)
    flags = flag_anomalies(smoothed, threshold, min_run_length=3)
    assert np.all(flags[1:4])
    assert not flags[0]
    assert not flags[4]


def test_flag_anomalies_all_below_threshold() -> None:
    smoothed = np.zeros(10)
    threshold = np.ones(10)
    flags = flag_anomalies(smoothed, threshold, min_run_length=1)
    assert not np.any(flags)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_evaluate_known_confusion() -> None:
    """Hand-computed TP/FP/FN: TP=2, FP=1, FN=1 → precision=recall=2/3."""
    true = np.array([True, True, False, False, True, False])
    pred = np.array([True, False, True, False, True, False])
    # TP=2 (idx 0,4), FP=1 (idx 2), FN=1 (idx 1)
    metrics = evaluate(true, pred)

    assert abs(metrics["precision"] - 2 / 3) < 1e-9
    assert abs(metrics["recall"] - 2 / 3) < 1e-9
    assert abs(metrics["f1"] - 2 / 3) < 1e-9

    # F0.5 with beta=0.5, beta²=0.25, p=r=2/3:
    # = 1.25 * (2/3)² / (0.25*(2/3) + 2/3) = 1.25*(4/9) / (5/6) = (5/9)*(6/5) = 2/3
    assert abs(metrics["f0_5"] - 2 / 3) < 1e-9
    # TP=2 (idx 0,4), FP=1 (idx 2) → 3 ground-truth positives, 3 predicted positives
    assert metrics["n_true_positive_labels"] == 3
    assert metrics["n_predicted_positive_labels"] == 3


def test_evaluate_all_correct() -> None:
    true = np.array([True, True, False, False])
    metrics = evaluate(true, true)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["f0_5"] == 1.0
    assert metrics["n_true_positive_labels"] == 2
    assert metrics["n_predicted_positive_labels"] == 2


def test_evaluate_no_positive_predictions() -> None:
    """No predictions → precision=0, recall=0, f-scores=0 (no ZeroDivision)."""
    true = np.array([True, False, True])
    pred = np.array([False, False, False])
    metrics = evaluate(true, pred)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["f0_5"] == 0.0
    assert metrics["n_true_positive_labels"] == 2
    assert metrics["n_predicted_positive_labels"] == 0


def test_evaluate_no_actual_positives() -> None:
    """All nominal ground truth → recall=0 (no ZeroDivision)."""
    true = np.array([False, False, False])
    pred = np.array([True, False, True])
    metrics = evaluate(true, pred)
    assert metrics["recall"] == 0.0
    assert metrics["n_true_positive_labels"] == 0
    assert metrics["n_predicted_positive_labels"] == 2


# ---------------------------------------------------------------------------
# Integration: dynamic_threshold detects an injected spike
# ---------------------------------------------------------------------------


def test_dynamic_threshold_detects_injected_spike() -> None:
    """A 3-point spike of 20x background std must be flagged after warm-up."""
    rng = np.random.default_rng(42)
    n = 200
    background_std = 0.1
    errors = rng.normal(0, background_std, n)

    # Inject spike at indices 100-102 (well past warm-up window of 30).
    errors[100:103] = 20 * background_std

    smoothed = smooth_errors(errors, span=10)
    threshold = dynamic_threshold(smoothed, window=30, z=2.0)
    flags = flag_anomalies(smoothed, threshold, min_run_length=1)

    assert np.any(flags[99:104]), (
        "Injected spike at indices 100-102 was not detected by flag_anomalies"
    )


# ---------------------------------------------------------------------------
# predict() — unit tests (require torch)
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")


def _make_predict_loader(
    n: int,
    window_size: int,
    batch_size: int = 8,
    seed: int = 0,
) -> torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Build a synthetic DataLoader of (x, y) pairs for predict() tests."""
    rng = np.random.default_rng(seed)
    x_np = rng.standard_normal((n, window_size)).astype(np.float32)
    y_np = rng.standard_normal(n).astype(np.float32)
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x_np).unsqueeze(-1),  # (n, W, 1)
        torch.from_numpy(y_np),                 # (n,)
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def test_predict_output_shape() -> None:
    """predict() must return two 1-D arrays (preds, targets) each of length N."""
    from spacecraft_telemetry.core.config import ModelConfig
    from spacecraft_telemetry.model.architecture import build_model
    from spacecraft_telemetry.model.scoring import predict

    cfg = ModelConfig(hidden_dim=8, num_layers=1, dropout=0.0)
    model = build_model(cfg)
    loader = _make_predict_loader(20, window_size=cfg.window_size, batch_size=8)
    device = torch.device("cpu")

    preds, targets = predict(model, loader, device)

    assert preds.shape == (20,), f"Expected preds shape (20,), got {preds.shape}"
    assert targets.shape == (20,), f"Expected targets shape (20,), got {targets.shape}"
    assert preds.dtype == np.float32
    assert targets.dtype == np.float32


def test_predict_dtype_is_float32() -> None:
    """Output dtype must be float32 regardless of batch size."""
    from spacecraft_telemetry.core.config import ModelConfig
    from spacecraft_telemetry.model.architecture import build_model
    from spacecraft_telemetry.model.scoring import predict

    cfg = ModelConfig(hidden_dim=8, num_layers=1, dropout=0.0)
    model = build_model(cfg)
    loader = _make_predict_loader(5, window_size=cfg.window_size, batch_size=5)

    preds, targets = predict(model, loader, torch.device("cpu"))
    assert preds.dtype == np.float32
    assert targets.dtype == np.float32


def test_predict_batched_matches_single_pass() -> None:
    """Batched predict must produce the same values as a single forward pass."""
    from spacecraft_telemetry.core.config import ModelConfig
    from spacecraft_telemetry.model.architecture import build_model
    from spacecraft_telemetry.model.scoring import predict

    cfg = ModelConfig(hidden_dim=8, num_layers=1, dropout=0.0)
    model = build_model(cfg)
    model.eval()
    rng = np.random.default_rng(1)
    n, W = 20, cfg.window_size
    x_np = rng.standard_normal((n, W)).astype(np.float32)
    y_np = rng.standard_normal(n).astype(np.float32)
    device = torch.device("cpu")

    # Reference: single forward pass over all 20 samples at once.
    with torch.no_grad():
        x_all = torch.from_numpy(x_np).unsqueeze(-1)  # (20, W, 1)
        expected = model(x_all).squeeze(1).cpu().numpy()

    # predict() with batch_size=8 splits into batches.
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x_np).unsqueeze(-1),
        torch.from_numpy(y_np),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
    preds, _ = predict(model, loader, device)

    np.testing.assert_allclose(preds, expected, rtol=1e-5)


def test_predict_sets_eval_mode() -> None:
    """predict() must call model.eval() — verified by checking training=False."""
    from spacecraft_telemetry.core.config import ModelConfig
    from spacecraft_telemetry.model.architecture import build_model
    from spacecraft_telemetry.model.scoring import predict

    cfg = ModelConfig(hidden_dim=8, num_layers=1, dropout=0.0)
    model = build_model(cfg)
    model.train()  # force training mode before calling predict
    loader = _make_predict_loader(4, window_size=cfg.window_size, batch_size=4)

    predict(model, loader, torch.device("cpu"))

    assert not model.training, "predict() should leave the model in eval mode"


# ---------------------------------------------------------------------------
# score_channel — integration test (slow)
# ---------------------------------------------------------------------------


def _override_settings_for_scoring(
    base: object,
    processed_dir: Path,
    artifacts_dir: Path,
) -> object:
    """Return a Settings copy pointing at tmp_path directories."""
    return base.model_copy(  # type: ignore[attr-defined]
        update={
            "spark": base.spark.model_copy(  # type: ignore[attr-defined]
                update={"processed_data_dir": processed_dir}
            ),
            "model": base.model.model_copy(  # type: ignore[attr-defined]
                update={"artifacts_dir": artifacts_dir}
            ),
        }
    )


@pytest.mark.slow
def test_score_channel_artifacts_and_metrics(
    tiny_series_parquet: object,
    mlflow_uri: str,
    tmp_path: Path,
) -> None:
    """score_channel must log artifacts to MLflow and return a well-formed metrics dict.

    Mirrors test_train_channel_runs_and_saves_artifacts on the scoring side.
    Trains a tiny model first (the same pattern as the training integration test),
    then calls score_channel and validates:
    - returned dict has all expected keys with float values in [0, 1]
    - support count keys are non-negative integers
    - errors.npy, threshold.npy, threshold_config.json, metrics/ all exist in MLflow run
    """
    from mlflow.tracking import MlflowClient

    from spacecraft_telemetry.core.config import load_settings
    from spacecraft_telemetry.mlflow_tracking.conventions import experiment_name
    from spacecraft_telemetry.model.io import find_latest_run_for_channel
    from spacecraft_telemetry.model.scoring import score_channel
    from spacecraft_telemetry.model.training import train_channel
    from tests.model.conftest import SeriesParquetFixture

    fx: SeriesParquetFixture = tiny_series_parquet  # type: ignore[assignment]
    settings = _override_settings_for_scoring(
        load_settings("test").model_copy(
            update={"mlflow": load_settings("test").mlflow.model_copy(
                update={"tracking_uri": mlflow_uri}
            )}
        ),
        processed_dir=fx.processed_dir,
        artifacts_dir=tmp_path / "models",
    )
    # Train first — score_channel requires model artifacts to exist.
    train_channel(settings, fx.mission, fx.channel)

    metrics = score_channel(settings, fx.mission, fx.channel)

    # --- metrics dict contract ---
    float_keys = {"precision", "recall", "f1", "f0_5"}
    int_keys = {"n_true_positive_labels", "n_predicted_positive_labels"}
    assert float_keys | int_keys == set(metrics.keys()), (
        f"Unexpected keys: {set(metrics.keys())}"
    )
    for k in float_keys:
        assert isinstance(metrics[k], float), f"{k} must be float"
        assert 0.0 <= metrics[k] <= 1.0, f"{k}={metrics[k]} out of [0,1]"
    for k in int_keys:
        assert isinstance(metrics[k], int), f"{k} must be int"
        assert metrics[k] >= 0, f"{k} must be non-negative"

    # --- artifact files in MLflow (A1: filesystem writes removed) ---
    scoring_exp = experiment_name(settings.model.model_type, "scoring", fx.mission)
    run = find_latest_run_for_channel(scoring_exp, fx.channel, mlflow_uri)
    assert run is not None, "No scoring run found in MLflow"
    client = MlflowClient(tracking_uri=mlflow_uri)
    artifact_names = {a.path for a in client.list_artifacts(run.info.run_id)}
    assert "errors.npy" in artifact_names, "errors.npy not logged to MLflow"
    assert "threshold.npy" in artifact_names, "threshold.npy not logged to MLflow"
    assert "threshold_config.json" in artifact_names, "threshold_config.json not logged to MLflow"
    assert "metrics" in artifact_names, "metrics/ dir not logged to MLflow"


@pytest.mark.slow
def test_score_channel_mlflow_run_has_eval_split_tag(
    tiny_series_parquet: object,
    mlflow_uri: str,
    tmp_path: Path,
) -> None:
    """score_channel creates an MLflow run tagged with eval_split and channel metadata."""
    import mlflow

    from spacecraft_telemetry.core.config import load_settings
    from spacecraft_telemetry.mlflow_tracking.conventions import experiment_name
    from spacecraft_telemetry.model.scoring import score_channel
    from spacecraft_telemetry.model.training import train_channel
    from tests.model.conftest import SeriesParquetFixture

    fx: SeriesParquetFixture = tiny_series_parquet  # type: ignore[assignment]
    settings = _override_settings_for_scoring(
        load_settings("test").model_copy(
            update={"mlflow": load_settings("test").mlflow.model_copy(
                update={"tracking_uri": mlflow_uri}
            )}
        ),
        processed_dir=fx.processed_dir,
        artifacts_dir=tmp_path / "models",
    )

    train_channel(settings, fx.mission, fx.channel)
    score_channel(settings, fx.mission, fx.channel, eval_split="final_portion")

    client = mlflow.tracking.MlflowClient()
    exp_name = experiment_name(settings.model.model_type, "scoring", fx.mission)
    exp = client.get_experiment_by_name(exp_name)
    assert exp is not None, f"scoring experiment {exp_name!r} was not created"

    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 1
    tags = runs[0].data.tags
    assert tags["eval_split"] == "final_portion"
    assert tags["model_type"] == settings.model.model_type
    assert tags["mission_id"] == fx.mission
    assert tags["channel_id"] == fx.channel
    assert runs[0].data.params["eval_split"] == "final_portion"


@pytest.mark.slow
def test_score_channel_eval_split_changes_anomaly_count(
    tiny_series_parquet: object,
    mlflow_uri: str,
    tmp_path: Path,
) -> None:
    """eval_split slices the label arrays: hpo_portion sees no anomalies in this fixture.

    tiny_series_parquet places all anomaly rows at the END of the test segment.
    With hpo_eval_fraction=0.6 those rows fall entirely in the final_portion,
    so hpo_portion must report n_true_positive_labels == 0 while full_test > 0.
    """
    from spacecraft_telemetry.core.config import load_settings
    from spacecraft_telemetry.model.scoring import score_channel
    from spacecraft_telemetry.model.training import train_channel
    from tests.model.conftest import SeriesParquetFixture

    fx: SeriesParquetFixture = tiny_series_parquet  # type: ignore[assignment]
    settings = _override_settings_for_scoring(
        load_settings("test").model_copy(
            update={"mlflow": load_settings("test").mlflow.model_copy(
                update={"tracking_uri": mlflow_uri}
            )}
        ),
        processed_dir=fx.processed_dir,
        artifacts_dir=tmp_path / "models",
    )

    train_channel(settings, fx.mission, fx.channel)
    metrics_full = score_channel(settings, fx.mission, fx.channel, eval_split="full_test")
    metrics_hpo = score_channel(settings, fx.mission, fx.channel, eval_split="hpo_portion")

    # tiny_series_parquet puts anomaly windows at the end of the test segment.
    # With hpo_eval_fraction=0.6, they fall past the HPO cutoff.
    assert metrics_full["n_true_positive_labels"] > 0, (
        "full_test should see labeled anomalies from the test fixture"
    )
    assert metrics_hpo["n_true_positive_labels"] == 0, (
        "hpo_portion (first 60%) should see no anomaly labels — "
        "all anomalies are at the end of the test segment"
    )


@pytest.mark.slow
def test_score_channel_errors_npy_unaffected_by_eval_split(
    tiny_series_parquet: object,
    mlflow_uri: str,
    tmp_path: Path,
) -> None:
    """errors.npy is saved from the full smoothed array regardless of eval_split."""
    import numpy as np

    from spacecraft_telemetry.core.config import load_settings
    from spacecraft_telemetry.mlflow_tracking.conventions import experiment_name
    from spacecraft_telemetry.model.io import (
        bytes_to_errors,
        download_artifact_bytes,
        find_latest_run_for_channel,
    )
    from spacecraft_telemetry.model.scoring import score_channel
    from spacecraft_telemetry.model.training import train_channel
    from tests.model.conftest import SeriesParquetFixture

    fx: SeriesParquetFixture = tiny_series_parquet  # type: ignore[assignment]
    settings = _override_settings_for_scoring(
        load_settings("test").model_copy(
            update={"mlflow": load_settings("test").mlflow.model_copy(
                update={"tracking_uri": mlflow_uri}
            )}
        ),
        processed_dir=fx.processed_dir,
        artifacts_dir=tmp_path / "models",
    )

    train_channel(settings, fx.mission, fx.channel)

    scoring_exp = experiment_name(settings.model.model_type, "scoring", fx.mission)

    score_channel(settings, fx.mission, fx.channel, eval_split="full_test")
    run_full = find_latest_run_for_channel(scoring_exp, fx.channel, mlflow_uri)
    assert run_full is not None, "No scoring run for full_test found in MLflow"
    errors_full = bytes_to_errors(
        download_artifact_bytes(run_full.info.run_id, "errors.npy", mlflow_uri)
    )

    score_channel(settings, fx.mission, fx.channel, eval_split="final_portion")
    run_final = find_latest_run_for_channel(scoring_exp, fx.channel, mlflow_uri)
    assert run_final is not None, "No scoring run for final_portion found in MLflow"
    # find_latest_run_for_channel returns the most recent — must be the new run.
    assert run_final.info.run_id != run_full.info.run_id, (
        "Expected a new run to be created for final_portion scoring"
    )
    errors_final = bytes_to_errors(
        download_artifact_bytes(run_final.info.run_id, "errors.npy", mlflow_uri)
    )

    np.testing.assert_array_equal(
        errors_full, errors_final,
        err_msg="errors.npy must be identical regardless of eval_split",
    )
