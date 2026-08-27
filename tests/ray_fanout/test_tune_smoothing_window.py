"""Characterization tests for ``_scoring_run_smoothing_window`` (docs/reviews/024, stage 0.1).

The function pins the ``error_smoothing_window`` that a sweep's saved
``errors.npy`` arrays were produced with — the pin the emitted config carries
and the value stage 2's grid refinement sweeps against (see tune.py's module
docstring and ``_refine_with_grid``). It had zero direct test coverage before
this file; these tests exist to nail its behaviour down before stage 2 (Q1)
changes it to swallow MLflow failures.
"""

from __future__ import annotations

from typing import Any

import pytest

from spacecraft_telemetry.ray_fanout.tune import _scoring_run_smoothing_window


class _FakeRun:
    def __init__(self, params: dict[str, str]) -> None:
        class _Data:
            def __init__(self, p: dict[str, str]) -> None:
                self.params = p

        self.data = _Data(params)


class _FakeMlflowClient:
    """Stand-in for ``mlflow.MlflowClient`` — the pattern established by
    ``tests/ray_fanout/test_tune.py``'s ``_FakeRun``, extended to the
    client itself since this function constructs one directly."""

    def __init__(self, runs: dict[str, _FakeRun], *, raises: bool = False) -> None:
        self._runs = runs
        self._raises = raises

    def __call__(self, *, tracking_uri: str) -> _FakeMlflowClient:
        # mlflow.MlflowClient(tracking_uri=...) — the fake IS the client,
        # constructing it just returns self so instances share the fixture.
        return self

    def get_run(self, run_id: str) -> _FakeRun:
        if self._raises:
            raise RuntimeError("tracking server unreachable")
        return self._runs[run_id]


def _install_fake_client(
    monkeypatch: pytest.MonkeyPatch, runs: dict[str, _FakeRun], **kw: Any
) -> None:
    fake = _FakeMlflowClient(runs, **kw)
    monkeypatch.setattr("mlflow.MlflowClient", fake)


def test_all_scoring_runs_agree_returns_the_window(monkeypatch: pytest.MonkeyPatch) -> None:
    runs = {
        "run-1": _FakeRun({"error_smoothing_window": "31"}),
        "run-2": _FakeRun({"error_smoothing_window": "31"}),
    }
    _install_fake_client(monkeypatch, runs)

    result = _scoring_run_smoothing_window(
        {"channel_1": "run-1", "channel_2": "run-2"}, "sqlite:///mlflow.db"
    )

    assert result == 31


def test_disagreeing_runs_return_none(monkeypatch: pytest.MonkeyPatch) -> None:
    runs = {
        "run-1": _FakeRun({"error_smoothing_window": "31"}),
        "run-2": _FakeRun({"error_smoothing_window": "12"}),
    }
    _install_fake_client(monkeypatch, runs)

    result = _scoring_run_smoothing_window(
        {"channel_1": "run-1", "channel_2": "run-2"}, "sqlite:///mlflow.db"
    )

    assert result is None


def test_a_run_missing_the_param_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    runs = {
        "run-1": _FakeRun({"error_smoothing_window": "31"}),
        "run-2": _FakeRun({}),  # pre-024 run, logged before the param existed
    }
    _install_fake_client(monkeypatch, runs)

    result = _scoring_run_smoothing_window(
        {"channel_1": "run-1", "channel_2": "run-2"}, "sqlite:///mlflow.db"
    )

    assert result is None


@pytest.mark.parametrize(
    "scoring_run_ids",
    [
        {},
        {"channel_1": None, "channel_2": None},
    ],
)
def test_all_none_or_empty_returns_none(
    monkeypatch: pytest.MonkeyPatch, scoring_run_ids: dict[str, str | None]
) -> None:
    _install_fake_client(monkeypatch, {})

    result = _scoring_run_smoothing_window(scoring_run_ids, "sqlite:///mlflow.db")

    assert result is None


def test_a_raising_client_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """The function's docstring promises None on lookup failure — stage 2.2
    (docs/reviews/024, Q1) wraps the MLflow loop in ``suppress(Exception)``
    to make that true."""
    _install_fake_client(
        monkeypatch, {"run-1": _FakeRun({"error_smoothing_window": "31"})}, raises=True
    )

    result = _scoring_run_smoothing_window({"channel_1": "run-1"}, "sqlite:///mlflow.db")

    assert result is None
