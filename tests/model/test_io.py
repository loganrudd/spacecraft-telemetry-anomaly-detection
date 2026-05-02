"""Tests for model.io — save/load round-trip, architecture-from-config, discipline check."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from spacecraft_telemetry.core.config import ModelConfig, Settings  # noqa: E402
from spacecraft_telemetry.model.architecture import build_model  # noqa: E402
from spacecraft_telemetry.model.io import (  # noqa: E402
    ModelArtifactPaths,
    _read_bytes,
    _write_bytes,
    artifact_paths,
    load_model,
    save_model,
    save_train_log,
)

# ---------------------------------------------------------------------------
# _write_bytes / _read_bytes
# ---------------------------------------------------------------------------


def test_write_bytes_creates_parent_dirs(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c" / "data.bin"
    _write_bytes(target, b"hello")
    assert target.read_bytes() == b"hello"


def test_read_bytes_round_trips(tmp_path: Path) -> None:
    p = tmp_path / "file.bin"
    _write_bytes(p, b"\x00\x01\x02")
    assert _read_bytes(p) == b"\x00\x01\x02"


# ---------------------------------------------------------------------------
# artifact_paths
# ---------------------------------------------------------------------------


def test_artifact_paths_layout(tmp_path: Path) -> None:
    settings = Settings(model=ModelConfig(artifacts_dir=tmp_path / "models"))
    paths = artifact_paths(settings, "ESA-Mission1", "channel_1")
    root = tmp_path / "models" / "ESA-Mission1" / "channel_1"
    assert Path(paths.model) == root / "model.pt"
    assert Path(paths.config) == root / "model_config.json"
    assert Path(paths.threshold) == root / "threshold.npy"
    assert Path(paths.threshold_config) == root / "threshold_config.json"
    assert Path(paths.train_log) == root / "train_log.json"


# ---------------------------------------------------------------------------
# save_model / load_model round-trip
# ---------------------------------------------------------------------------


def test_save_then_load_round_trip(tmp_path: Path) -> None:
    cfg = ModelConfig(hidden_dim=16, num_layers=2)
    model = build_model(cfg)
    model.eval()

    root = tmp_path / "models" / "ESA-Mission1" / "channel_1"
    paths = ModelArtifactPaths(
        root=root,
        model=root / "model.pt",
        config=root / "model_config.json",
        norm=root / "normalization_params.json",
        errors=root / "errors.npy",
        threshold=root / "threshold.npy",
        threshold_config=root / "threshold_config.json",
        metrics=root / "metrics.json",
        train_log=root / "train_log.json",
    )

    save_model(model, paths, cfg, window_size=10)
    assert Path(paths.model).exists()
    assert Path(paths.config).exists()

    loaded_model, loaded_cfg, loaded_window_size = load_model(paths, torch.device("cpu"))
    loaded_model.eval()

    x = torch.zeros(2, 10, 1)
    with torch.no_grad():
        assert torch.equal(model(x), loaded_model(x))

    assert loaded_cfg.hidden_dim == cfg.hidden_dim
    assert loaded_cfg.num_layers == cfg.num_layers
    assert loaded_window_size == 10


def test_load_model_uses_saved_architecture_not_current_settings(
    tmp_path: Path,
) -> None:
    """Model trained at hidden_dim=16 must reload as hidden_dim=16 even if
    the current ModelConfig default is different."""
    saved_cfg = ModelConfig(hidden_dim=16, num_layers=1, dropout=0.0)
    model = build_model(saved_cfg)

    root = tmp_path / "arts"
    paths = ModelArtifactPaths(
        root=root,
        model=root / "model.pt",
        config=root / "model_config.json",
        norm=root / "normalization_params.json",
        errors=root / "errors.npy",
        threshold=root / "threshold.npy",
        threshold_config=root / "threshold_config.json",
        metrics=root / "metrics.json",
        train_log=root / "train_log.json",
    )
    save_model(model, paths, saved_cfg, window_size=250)

    # Reload — current ModelConfig default has hidden_dim=80 (the class default).
    loaded_model, loaded_cfg, loaded_window_size = load_model(paths, torch.device("cpu"))
    assert loaded_model.hidden_dim == 16
    assert loaded_cfg.hidden_dim == 16
    assert loaded_window_size == 250


# ---------------------------------------------------------------------------
# save_train_log
# ---------------------------------------------------------------------------


def test_save_train_log_writes_json(tmp_path: Path) -> None:
    import json

    root = tmp_path / "arts"
    paths = ModelArtifactPaths(
        root=root,
        model=root / "model.pt",
        config=root / "model_config.json",
        norm=root / "normalization_params.json",
        errors=root / "errors.npy",
        threshold=root / "threshold.npy",
        threshold_config=root / "threshold_config.json",
        metrics=root / "metrics.json",
        train_log=root / "train_log.json",
    )
    entries = [{"epoch": 0, "train_loss": 0.5, "val_loss": 0.4}]
    save_train_log(paths, entries)

    written = json.loads(Path(paths.train_log).read_bytes())
    assert written == entries


# ---------------------------------------------------------------------------
# Discipline check: training.py and scoring.py must not call raw IO directly
# ---------------------------------------------------------------------------

_FORBIDDEN_PATTERNS = [
    r"\.write_bytes\(",
    r"torch\.save\(",
    r"np\.save\(",
    r"open\(.*[\"']w[\"']",
]


def _check_no_raw_io(source_path: Path) -> list[str]:
    """Return lines that contain a forbidden raw-IO call.

    Uses the AST to identify string-literal (docstring) lines and comment
    lines so that documentation mentioning these patterns doesn't false-positive.
    """
    import ast

    text = source_path.read_text()

    # Collect line numbers that belong to module/function docstrings.
    docstring_lines: set[int] = set()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            end: int = getattr(node.value, "end_lineno", None) or node.lineno
            for lineno in range(node.lineno, end + 1):
                docstring_lines.add(lineno)

    hits = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if lineno in docstring_lines:
            continue
        if line.lstrip().startswith("#"):
            continue
        for pat in _FORBIDDEN_PATTERNS:
            if re.search(pat, line):
                hits.append(f"{source_path.name}:{line.strip()!r}")
    return hits


def test_no_direct_filesystem_writes_in_training_or_scoring() -> None:
    """training.py and scoring.py must funnel all writes through model.io."""
    src = Path(__file__).parents[2] / "src" / "spacecraft_telemetry" / "model"
    violations: list[str] = []
    for module in ("training.py", "scoring.py"):
        path = src / module
        if path.exists():
            violations.extend(_check_no_raw_io(path))
    assert not violations, (
        "Raw filesystem writes found — use model.io helpers instead:\n"
        + "\n".join(violations)
    )
