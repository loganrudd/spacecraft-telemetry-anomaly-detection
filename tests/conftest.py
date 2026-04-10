"""Root test fixtures — expanded as phases are implemented."""

import pytest


@pytest.fixture
def sample_data_dir(tmp_path):
    """Return a temporary directory for sample test data."""
    return tmp_path / "sample"
