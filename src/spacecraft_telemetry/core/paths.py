"""Path helpers for the local-fs ↔ gs:// duality across the data layer.

UPath (universal_pathlib) wraps fsspec so the same `Path`-style operations
work against local disk, `gs://`, `s3://`, etc. Use these helpers anywhere
a Settings field may hold either a local path or a cloud URI.
"""

from __future__ import annotations

from pathlib import Path

from upath import UPath


def to_upath(value: str | Path | UPath) -> UPath:
    """Normalize a Settings path-like value to a UPath.

    Strings, Paths, and UPaths all stringify to a usable URI; UPath's
    constructor picks the right backend based on the scheme.
    """
    return UPath(str(value))


def absolutize_if_local(value: str | Path | UPath) -> UPath:
    """Resolve relative *local* paths to absolute; pass cloud URIs through.

    Ray workers run from Ray's temp session dir, so relative local paths
    would break. ``Path.resolve()`` on a `gs://` URI produces a bogus
    `gs:/...` joined to CWD — so we only resolve when the protocol is
    local (empty or ``file``).
    """
    up = to_upath(value)
    if up.protocol in ("", "file"):
        return UPath(Path(str(up)).resolve())
    return up


def output_path(
    root: str | Path | UPath,
    mission: str,
    variant: str | None,
    *parts: str,
) -> UPath:
    """Compose a mission + optional-variant OUTPUT path.

    ``variant=None`` (the default everywhere today) reproduces the
    pre-variant layout byte-for-byte::

        {root}/{mission}/{*parts}

    A non-null variant inserts one path segment between the mission and the
    rest, giving an experiment/ablation config its own output namespace
    without duplicating the mission's raw input data::

        {root}/{mission}/{variant}/{*parts}

    Use this for every WRITE and READ site keyed by
    ``(preprocess.processed_data_dir, mission)`` or ``(model.artifacts_dir,
    mission)`` — see docs/plans/020-experiment-variant-axis.md.

    Do NOT use this for raw/sample INPUT reads (``data.raw_data_dir``,
    ``data.sample_data_dir``, ``collect.raw_ticks_dir``) — those are keyed on
    mission only. One copy of raw data serves every variant of a mission;
    that is the duplication this axis exists to kill.
    """
    base = to_upath(root) / mission
    if variant:
        base = base / variant
    for part in parts:
        base = base / part
    return base
