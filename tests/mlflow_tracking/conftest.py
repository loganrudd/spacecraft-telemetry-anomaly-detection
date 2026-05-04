"""Shared fixtures for mlflow_tracking tests."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import mlflow
import pytest


@pytest.fixture
def mlflow_uri(tmp_path: Path) -> Generator[str, None, None]:
    """Per-test isolated SQLite MLflow store.

    Sets the global tracking URI before each test and resets it after.
    Any active run is ended to prevent state leaking between tests.
    """
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(uri)
    yield uri
    # End any run left open by the test body.
    if mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.set_tracking_uri("")
