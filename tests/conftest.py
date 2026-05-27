"""Root test fixtures — expanded as phases are implemented."""

import os
import sys

import pytest


@pytest.fixture
def sample_data_dir(tmp_path):
    """Return a temporary directory for sample test data."""
    return tmp_path / "sample"


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
