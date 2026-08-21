"""Guardrail: every cloud-* Make target forwards the variables its script reads.

The failure mode is silent, not loud. `make cloud-train SUBSYSTEM=subsystem_5`
looked like it scoped the run to one subsystem; the target never forwarded
SUBSYSTEM, so the flag evaporated and the job would have trained every channel
in channels.txt while reporting success. The same gap existed for VARIANT on
cloud-tune and cloud-drift — there, a dropped VARIANT silently reads and writes
the BASE mission tree instead of the experiment variant's, which in the worst
case means a tune job overwriting production's tuned_configs.json.

Rather than pin a hand-maintained list (which would rot exactly like the
targets did), this derives the expectation: if scripts/cloud_X.sh reads
`${VAR:-}`, then the cloud-X target must pass VAR=$(VAR).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent.parent
_MAKEFILE = _ROOT / "Makefile"
_SCRIPTS = _ROOT / "scripts"

# Variables that are genuinely per-invocation knobs a user would set on the
# make command line. Deliberately excludes environment/plumbing the target
# supplies itself (PROJECT_ID, REGION, MLFLOW_URL) and script-internal
# derivations (VARIANT_SEG, *_ARG).
_USER_KNOBS = frozenset({"VARIANT", "SUBSYSTEM", "MULTIVARIATE", "CHANNELS", "INJECTED", "TUNED"})


def _target_body(name: str) -> str | None:
    """Return the recipe body for a Make target, or None if absent."""
    text = _MAKEFILE.read_text()
    m = re.search(rf"^{re.escape(name)}:.*?$\n((?:\t.*\n|#.*\n)*)", text, re.MULTILINE)
    return m.group(1) if m else None


def _script_reads(script: Path) -> set[str]:
    """Variables the script takes from the environment via ${VAR:-default}."""
    text = script.read_text()
    return {v for v in re.findall(r'^\s*(\w+)="\$\{\1:-', text, re.MULTILINE) if v in _USER_KNOBS}


def _cloud_pairs() -> list[tuple[str, Path]]:
    pairs = []
    for script in sorted(_SCRIPTS.glob("cloud_*.sh")):
        target = script.stem.replace("cloud_", "cloud-")
        if _target_body(target) is not None:
            pairs.append((target, script))
    return pairs


@pytest.mark.parametrize("target,script", _cloud_pairs(), ids=lambda x: getattr(x, "name", x))
def test_target_forwards_every_knob_its_script_reads(target: str, script: Path) -> None:
    body = _target_body(target)
    assert body is not None, f"{target} not found in Makefile"
    missing = {
        var
        for var in _script_reads(script)
        if not re.search(rf"\b{var}=\$\({var}\)", body)
    }
    assert not missing, (
        f"Make target {target!r} does not forward {sorted(missing)}, but "
        f"{script.name} reads them from the environment. A user passing "
        f"`make {target} {sorted(missing)[0]}=...` would have it silently ignored."
    )


def test_helper_detects_a_real_knob() -> None:
    """Guard the guard: if the ${VAR:-} regex ever stops matching, every test
    above would vacuously pass. Pin one known-present variable."""
    assert "VARIANT" in _script_reads(_SCRIPTS / "cloud_train.sh")
