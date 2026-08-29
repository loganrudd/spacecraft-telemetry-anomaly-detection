"""Root test fixtures — expanded as phases are implemented."""

import os
import sys
from collections.abc import Generator

import pytest


@pytest.fixture
def sample_data_dir(tmp_path):
    """Return a temporary directory for sample test data."""
    return tmp_path / "sample"


@pytest.fixture(scope="session", autouse=True)
def _mlflow_artifact_root(tmp_path_factory: pytest.TempPathFactory) -> Generator[None, None, None]:
    """Redirect MLflow's default artifact root out of the repo for the whole suite.

    docs/reviews/024, stage 6.3: tests set `mlflow.tracking_uri` to a per-test
    tmp_path SQLite DB, but a plain `sqlite:///...` tracking URI with no
    `artifact_location` falls back to `./mlruns` relative to CWD (MLflow's
    `_get_sqlalchemy_store` — see `_MLFLOW_SERVER_ARTIFACT_ROOT` /
    `DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH`) — i.e. the repo root during a
    local test run. That produced 23 fixture `tuned_configs.json` files under
    the repo's `mlruns/3/` that were, at a glance, indistinguishable from real
    sweep output (their objective_values matched literals in test_tune.py,
    which is the only reason they were ever told apart).

    Session-scoped and autouse so it is set before anything — including
    `ray_local`'s local-mode Ray cluster — starts: local-mode Ray workers
    inherit the parent process's environment, so setting this before
    `ray.init()` is what makes it reach MLflow calls made from inside Ray
    tasks (train_channel/score_channel/run_hpo_sweep), not just calls made
    directly in the test process.
    """
    import os

    os.environ["_MLFLOW_SERVER_ARTIFACT_ROOT"] = str(
        tmp_path_factory.mktemp("mlflow_artifacts")
    )
    yield
    del os.environ["_MLFLOW_SERVER_ARTIFACT_ROOT"]


@pytest.fixture(autouse=True)
def isolate_mlflow_globals() -> Generator[None, None, None]:
    """Reset MLflow's process-global client state around each test.

    Many tests use per-test SQLite tracking URIs. MLflow stores tracking and
    registry URIs in module-global state, so one test can otherwise leak a temp
    database into the next and trigger false-positive URI-change warnings.
    """
    try:
        import mlflow
    except ModuleNotFoundError:
        yield
        return

    if mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.set_tracking_uri("")
    mlflow.set_registry_uri("")

    yield

    if mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.set_tracking_uri("")
    mlflow.set_registry_uri("")


@pytest.fixture(scope="session")
def ray_local():
    """Start a Ray local cluster once per session; shut it down on teardown.

    Sets PYTHONPATH in the Ray runtime_env so remote workers can import
    spacecraft_telemetry.  Without this, tasks fail with ImportError and
    silently retry (max_retries=3), causing multi-minute hangs.

    ignore_reinit_error=True is safe here: if another part of the test
    session already initialised Ray (e.g. a test that calls ray.init
    directly), this fixture is a no-op.
    """
    ray = pytest.importorskip("ray")
    pythonpath = os.pathsep.join(p for p in sys.path if p)
    ray.init(
        num_cpus=2,
        ignore_reinit_error=True,
        runtime_env={"env_vars": {"PYTHONPATH": pythonpath}},
    )
    yield
    ray.shutdown()
