"""Guard: the ray-free serving image must import the API startup path.

The `api` / `api-iss` serving image installs only `[serving,tracking,gcp,
inference]` — NOT `[ml]` — so `ray` is absent (deploy/api/Dockerfile). The API
startup chain reaches a pure-pandas preprocessing transform:

    api.app -> api.live.los_stats -> preprocess.transforms.compute_los_mask

Importing the `preprocess.transforms` submodule runs the `preprocess` package
`__init__`, which must NOT eagerly import the ray-dependent `pipeline` module or
the container crash-loops at startup with `ModuleNotFoundError: No module named
'ray'` before it can bind the port (regression fixed by lazy PEP 562 export of
run_preprocessing / run_iss_preprocessing).

These run in a subprocess because `ray` is already imported in the main test
process (the full [ml] env) and cannot be cleanly removed — a fresh interpreter
with a meta-path blocker is the only faithful simulation of the slim image.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

# Meta-path finder that makes `import ray` raise ModuleNotFoundError, exactly as
# the slim serving image would. Prepended to sys.meta_path so it wins.
_BLOCK_RAY = textwrap.dedent(
    """
    import importlib.abc
    import sys

    class _BlockRay(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name == "ray" or name.startswith("ray."):
                raise ModuleNotFoundError("No module named 'ray'")
            return None

    sys.meta_path.insert(0, _BlockRay())
    """
)


def _run_in_ray_free_subprocess(body: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _BLOCK_RAY + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_api_startup_import_chain_works_without_ray() -> None:
    """The exact chain that crash-looped api-iss must import in a ray-free env."""
    result = _run_in_ray_free_subprocess(
        """
        # Sanity: ray really is blocked in this interpreter.
        try:
            import ray  # noqa: F401
        except ModuleNotFoundError:
            pass
        else:
            raise AssertionError("ray was importable — blocker did not take effect")

        # The startup import that crashed the container.
        from spacecraft_telemetry.api.live.los_stats import compute_los_stats  # noqa: F401
        from spacecraft_telemetry.preprocess.transforms import compute_los_mask  # noqa: F401
        import spacecraft_telemetry.preprocess as pp
        assert hasattr(pp, "compute_los_mask")
        print("SLIM_IMPORT_OK")
        """
    )
    assert result.returncode == 0, (
        f"ray-free serving import failed (stdout={result.stdout!r}, "
        f"stderr={result.stderr!r})"
    )
    assert "SLIM_IMPORT_OK" in result.stdout


def test_ray_dependent_pipeline_exports_stay_lazy() -> None:
    """Accessing run_preprocessing without ray raises ModuleNotFoundError.

    This documents the boundary: the pipeline entrypoints are only reachable in
    the full [ml] image. If someone re-adds an eager `import` of them to the
    package __init__, the previous test breaks; if someone makes them importable
    without ray, this one flags that the lazy contract changed.
    """
    result = _run_in_ray_free_subprocess(
        """
        import spacecraft_telemetry.preprocess as pp
        try:
            pp.run_preprocessing
        except ModuleNotFoundError:
            print("LAZY_BOUNDARY_OK")
        else:
            raise AssertionError("run_preprocessing resolved without ray")
        """
    )
    assert result.returncode == 0, (
        f"unexpected failure (stdout={result.stdout!r}, stderr={result.stderr!r})"
    )
    assert "LAZY_BOUNDARY_OK" in result.stdout
