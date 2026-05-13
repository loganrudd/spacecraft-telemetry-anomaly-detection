"""Shared fixtures for the api test package."""

from __future__ import annotations

import pytest

from spacecraft_telemetry.core.config import load_settings


@pytest.fixture()
def test_settings(tmp_path):
    """Settings loaded from the test environment (test.yaml)."""
    return load_settings("test")
