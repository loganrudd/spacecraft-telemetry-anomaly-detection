#!/usr/bin/env python3
"""M1.5 cold-start profiling scaffold.

Registers N minimal TelemanomLSTM models into a fresh SQLite MLflow backend
so that ``docker run st-api:dev`` can exercise the real lifespan model-loading
path without a full training run.  Artifact sizes are identical to production
(same architecture, random-init weights are the same byte-size as trained ones).

Usage:
    python scripts/profile_startup.py --n-models 10  --output /tmp/sp_profile_10
    python scripts/profile_startup.py --n-models 50  --output /tmp/sp_profile_50
    python scripts/profile_startup.py --n-models 100 --output /tmp/sp_profile_100

Each invocation prints the docker run command to measure that N.
Requires: pip install -e ".[ml,tracking]" in the active venv.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MODEL_TYPE = "telemanom"
DEFAULT_MISSION = "ESA-Mission2"
WINDOW_SIZE = 250


def _build_model():  # type: ignore[return]
    """Construct a default-arch TelemanomLSTM (random init = same byte size as trained)."""
    from spacecraft_telemetry.model.architecture import TelemanomLSTM

    model = TelemanomLSTM(hidden_dim=80, num_layers=2, dropout=0.3)
    model.eval()
    return model


def _get_or_create_experiment(client, name: str, artifact_root: Path) -> str:
    """Return experiment_id, creating with an explicit artifact_location if absent."""
    exp = client.get_experiment_by_name(name)
    if exp is not None:
        return exp.experiment_id
    return client.create_experiment(name, artifact_location=f"file://{artifact_root}")


def register_channel(
    mission: str,
    channel: str,
    artifact_root: Path,
    window_size: int,
) -> None:
    """Register one fake channel: training run + registry entry + scoring run.

    Mirrors exactly what train_channel() + score_channel() write so that
    load_model_for_scoring() and load_scoring_params() in the lifespan can
    read them without modification.
    """
    import mlflow
    import mlflow.pytorch

    from spacecraft_telemetry.mlflow_tracking.conventions import (
        experiment_name as exp_name,
    )
    from spacecraft_telemetry.mlflow_tracking.conventions import (
        registered_model_name as reg_name,
    )

    client = mlflow.tracking.MlflowClient()

    train_exp = exp_name(MODEL_TYPE, "training", mission)
    score_exp = exp_name(MODEL_TYPE, "scoring", mission)

    train_exp_id = _get_or_create_experiment(client, train_exp, artifact_root)
    score_exp_id = _get_or_create_experiment(client, score_exp, artifact_root)

    model = _build_model()
    model_name = reg_name(MODEL_TYPE, mission, channel)

    # Training run — logs the model artifact and params that load_model_for_scoring reads.
    with mlflow.start_run(
        experiment_id=train_exp_id,
        run_name=channel,
        tags={"channel_id": channel, "mission_id": mission, "phase": "training"},
    ) as run:
        mlflow.log_params({
            "model_type": MODEL_TYPE,
            "window_size": window_size,
            "hidden_dim": 80,
            "num_layers": 2,
            "dropout": 0.3,
            "learning_rate": 0.001,
            "batch_size": 64,
            "seed": 42,
            "early_stopping_patience": 5,
            "prediction_horizon": 1,
        })
        # artifact_path="model" matches the runs:/{run_id}/model URI in load_model_for_scoring.
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name=model_name,
        )

    # Scoring run — logs the four params that load_scoring_params reads.
    with mlflow.start_run(
        experiment_id=score_exp_id,
        run_name=channel,
        tags={"channel_id": channel, "mission_id": mission, "phase": "scoring"},
    ):
        mlflow.log_params({
            "threshold_window": WINDOW_SIZE,
            "threshold_z": 3.0,
            "error_smoothing_window": 30,
            "threshold_min_anomaly_len": 3,
        })


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register N fake LSTM models for M1.5 cold-start profiling.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--n-models", type=int, default=10, metavar="N",
        help="Number of channels to register (default: 10)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/sp_profile"),
        help="Output directory — mlflow.db and artifacts/ created here (default: /tmp/sp_profile)",
    )
    parser.add_argument(
        "--mission", default=DEFAULT_MISSION,
        help=f"Mission name used in model registry (default: {DEFAULT_MISSION})",
    )
    parser.add_argument(
        "--window-size", type=int, default=WINDOW_SIZE,
        help=f"window_size logged as a run param (default: {WINDOW_SIZE})",
    )
    args = parser.parse_args()

    output: Path = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    db_path = output / "mlflow.db"
    artifact_root = output / "artifacts"
    artifact_root.mkdir(exist_ok=True)

    tracking_uri = f"sqlite:///{db_path}"

    import mlflow
    mlflow.set_tracking_uri(tracking_uri)

    channels = [f"channel_{i + 1}" for i in range(args.n_models)]

    print(f"Registering {args.n_models} channels → {db_path}")
    for i, channel in enumerate(channels):
        sys.stdout.write(f"\r  [{i + 1:3d}/{args.n_models}] {channel}   ")
        sys.stdout.flush()
        register_channel(args.mission, channel, artifact_root, args.window_size)
    print(f"\r  Done — {args.n_models} models registered.           ")

    # Render the docker commands for this N value.
    channels_json = json.dumps(channels)

    # Render one chained command per resource config so the timer starts before
    # docker run — a gap between the two would give 0s on fast machines.
    base_env = (
        f"-e SPACECRAFT_MLFLOW__TRACKING_URI={tracking_uri} \\\n"
        f"  -e MLFLOW_ARTIFACTS_DESTINATION={artifact_root} \\\n"
        f"  -e SPACECRAFT_API__MISSION={args.mission} \\\n"
        f"  -e 'SPACECRAFT_API__CHANNELS={channels_json}'"
    )
    base_vols = (
        f"-v {db_path}:{db_path} \\\n"
        f"  -v {artifact_root}:{artifact_root}"
    )

    def _cmd(memory: str, cpus: int) -> str:
        return (
            f"docker stop $CID 2>/dev/null; \\\n"
            f"START=$(date +%s) && \\\n"
            f"CID=$(docker run --rm -d \\\n"
            f"  --memory={memory} --cpus={cpus} \\\n"
            f"  -p 8080:8080 \\\n"
            f"  {base_vols} \\\n"
            f"  {base_env} \\\n"
            f"  st-api:dev) && \\\n"
            f"until curl -fsS http://localhost:8080/health 2>/dev/null | "
            f"grep -q '\"status\":.*\"ok\"'; "
            f"do sleep 1; done && \\\n"
            f"echo \"Cold-start: $(( $(date +%s) - START )) s\" && \\\n"
            f"docker stats --no-stream $CID && \\\n"
            f"docker stop $CID"
        )

    print(f"""
{'=' * 60}
Profiling commands for N={args.n_models}
{'=' * 60}
Run each block as a single command (chained &&) so the timer
starts before docker run — a gap between lines gives a false 0s.

# 2 vCPU / 2 GiB (Plan 009b baseline):
{_cmd("2g", 2)}

# 4 vCPU / 4 GiB (fallback tier):
{_cmd("4g", 4)}
""")


if __name__ == "__main__":
    main()
