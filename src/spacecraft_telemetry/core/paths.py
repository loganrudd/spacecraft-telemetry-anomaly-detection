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
