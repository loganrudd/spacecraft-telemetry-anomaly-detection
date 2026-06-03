"""Root test fixtures — expanded as phases are implemented."""

import os
import sys
from collections.abc import Generator

import pytest


@pytest.fixture
def sample_data_dir(tmp_path):
    """Return a temporary directory for sample test data."""
    return tmp_path / "sample"


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
