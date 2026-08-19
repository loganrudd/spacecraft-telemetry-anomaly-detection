"""Guardrail for the --multivariate flag inside cluster_train.yaml's entrypoint.

`--multivariate` is a flag that is either present or absent, and envsubst has
no conditional-expansion syntax, so scripts/cloud_train.sh pre-resolves it into
MULTIVARIATE_ARG on the host (same pattern as CHANNELS_ARG / VARIANT_SEG).

Two failure modes this pins:

1. **The flag silently vanishes.** If ${MULTIVARIATE_ARG} were dropped from the
   entrypoint, `--multivariate` would submit a job that trains the *univariate*
   per-channel models instead — producing a plausible-looking result set that
   answers the wrong question (docs/plans/021-multivariate-telemanom.md).
2. **The null default drifts.** With MULTIVARIATE_ARG unset the parsed
   entrypoint must be byte-identical to what it was before the flag existed —
   the same invariant plan 020 held for variant=None.

Asserts on the *parsed* YAML entrypoint, not the rendered text: the flag sits
in a `>-` folded block scalar, where an empty substitution leaves a
whitespace-only line. Whether that folds away cleanly is exactly the thing
worth testing, and only a YAML parse can tell us.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

_CLUSTER_TRAIN = Path(__file__).parent.parent.parent / "deploy" / "ray" / "cluster_train.yaml"

# Only the variables the entrypoint actually interpolates need real values;
# expandvars leaves anything else literal, which is harmless here.
_ENV = {
    "PROJECT_ID": "test-proj",
    "MISSION": "ESA-Mission1",
    "VARIANT": "adb-84m",
    "VARIANT_SEG": "/adb-84m",
    "CHANNELS_ARG": "--channels channel_41,channel_42",
    "NUM_GPUS": "0.16",
    "WINDOW_SIZE_OVERRIDE": "250",
    "MLFLOW_URL": "http://mlflow.invalid",
    "REGION": "us-central1",
}


def _entrypoint(multivariate_arg: str) -> str:
    """Render cluster_train.yaml and return its parsed spec.entrypoint."""
    env = {**_ENV, "MULTIVARIATE_ARG": multivariate_arg}
    original = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        rendered = os.path.expandvars(_CLUSTER_TRAIN.read_text())
    finally:
        for k, v in original.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    entrypoint: str = yaml.safe_load(rendered)["spec"]["entrypoint"]
    return entrypoint


class TestMultivariateFlagThreading:
    def test_flag_present_when_set(self) -> None:
        assert "--multivariate" in _entrypoint("--multivariate")

    def test_flag_absent_by_default(self) -> None:
        assert "--multivariate" not in _entrypoint("")

    def test_null_default_has_no_stray_whitespace(self) -> None:
        """The empty substitution sits on its own line in a folded scalar —
        it must fold away entirely, leaving no double space or trailing space
        that would reach the shell as a stray empty argv entry."""
        entrypoint = _entrypoint("")
        assert entrypoint == entrypoint.strip(), f"stray edge whitespace: {entrypoint!r}"
        assert "  " not in entrypoint, f"double space: {entrypoint!r}"

    def test_flag_appended_after_channel_selection(self) -> None:
        """Order matters only for readability, but a flag landing *inside* the
        channel CSV would silently corrupt the channel list."""
        entrypoint = _entrypoint("--multivariate")
        assert entrypoint.endswith("--multivariate"), entrypoint
        assert "channel_41,channel_42 --multivariate" in entrypoint, entrypoint

    def test_default_entrypoint_matches_pre_flag_form(self) -> None:
        """The null-default invariant, pinned as a literal rather than
        recomputed — a recomputation would move in lockstep with a regression."""
        assert _entrypoint("") == (
            "spacecraft-telemetry --env cloud ray train "
            "--mission ESA-Mission1 --channels channel_41,channel_42"
        )
