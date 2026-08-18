"""Guardrail for the variant axis inside the RayJob YAML entrypoints.

deploy/ray/cluster_tune.yaml, cluster_score.yaml, and cluster_score_cpu.yaml
compose gs://.../{MISSION}{VARIANT_SEG}/tuned_configs.json paths on the host,
before kubectl ever sees the manifest (envsubst has no conditional-expansion
syntax, so scripts/cloud_*.sh pre-resolves VARIANT_SEG — see those scripts'
comments). A missed occurrence silently reads/writes the base variant's
tuned_configs.json instead of the intended one — this happened once (plan 020
review, item A1: cluster_tune.yaml's upload and cluster_score{,_cpu}.yaml's
fetch used a bare ${MISSION} path) and this test exists so it can't happen
again unnoticed.

Rendered with os.path.expandvars rather than shelling out to `envsubst` (a
GNU gettext binary not guaranteed present on a dev Mac) — this only needs
${VAR} substitution for the three variables under test; any other unresolved
placeholder is left literal by expandvars, which is harmless here since
nothing below inspects it.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

_DEPLOY_DIR = Path(__file__).parent.parent.parent / "deploy" / "ray"
_ALL_CLUSTER_YAMLS = sorted(_DEPLOY_DIR.glob("cluster_*.yaml"))

# Files that compose a gs://.../tuned_configs.json or models/.../tuned_configs.json
# path directly in the YAML. cluster_train*.yaml and cluster_preprocess.yaml
# are deliberately excluded: their channel-selection paths are fully resolved
# on the host into CHANNELS_ARG before envsubst ever sees the YAML, so they
# contain no literal path composition of their own to guard here.
_TUNED_CONFIGS_YAMLS = [
    _DEPLOY_DIR / "cluster_tune.yaml",
    _DEPLOY_DIR / "cluster_score.yaml",
    _DEPLOY_DIR / "cluster_score_cpu.yaml",
]

# Matches a mission/variant-scoped tuned_configs.json path: gs://bucket/... or
# models/... — NOT the unrelated local /tmp/tuned_configs.json fetch target,
# which is deliberately mission/variant-agnostic and must NOT vary.
_COMPOSED_PATH_RE = re.compile(r"(?:gs://|models/)\S*?tuned_configs\.json")

_ENV = {"PROJECT_ID": "test-proj", "MISSION": "ESA-Mission1"}


def _render(path: Path, variant_seg: str) -> str:
    text = path.read_text()
    env = {**_ENV, "VARIANT_SEG": variant_seg}
    original = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        return os.path.expandvars(text)
    finally:
        for k, v in original.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestTunedConfigsPathVariantThreading:
    """The class of bug this file exists to catch: a composed path that
    resolves ${MISSION} but forgets ${VARIANT_SEG}."""

    @pytest.mark.parametrize("yaml_path", _TUNED_CONFIGS_YAMLS, ids=lambda p: p.name)
    def test_null_default_produces_unscoped_path(self, yaml_path: Path) -> None:
        """VARIANT_SEG="" (today's default) must reproduce the pre-variant
        layout byte for byte — the plan's central invariant, applied here."""
        rendered = _render(yaml_path, variant_seg="")
        matches = _COMPOSED_PATH_RE.findall(rendered)
        assert matches, (
            f"{yaml_path.name}: expected at least one composed tuned_configs.json path"
        )
        for m in matches:
            assert m.endswith("/ESA-Mission1/tuned_configs.json"), (
                f"{yaml_path.name}: {m!r} should end .../ESA-Mission1/tuned_configs.json "
                "with VARIANT_SEG empty (null-default invariant)"
            )

    @pytest.mark.parametrize("yaml_path", _TUNED_CONFIGS_YAMLS, ids=lambda p: p.name)
    def test_variant_set_scopes_every_composed_path(self, yaml_path: Path) -> None:
        rendered = _render(yaml_path, variant_seg="/adb-24m")
        matches = _COMPOSED_PATH_RE.findall(rendered)
        assert matches, (
            f"{yaml_path.name}: expected at least one composed tuned_configs.json path"
        )
        # Assert on every match, not just the first — the regression this
        # guards against was exactly one missed occurrence among several.
        for m in matches:
            assert m.endswith("/ESA-Mission1/adb-24m/tuned_configs.json"), (
                f"{yaml_path.name}: {m!r} is missing the variant segment"
            )

    @pytest.mark.parametrize("yaml_path", _TUNED_CONFIGS_YAMLS, ids=lambda p: p.name)
    def test_no_double_slash_or_unresolved_placeholder(self, yaml_path: Path) -> None:
        rendered = _render(yaml_path, variant_seg="/adb-24m")
        for m in _COMPOSED_PATH_RE.findall(rendered):
            assert "//" not in m.removeprefix("gs://"), f"double slash in {m!r}"
            assert "${" not in m, f"unresolved placeholder in {m!r}"

    def test_local_temp_fetch_target_never_scoped(self) -> None:
        """cluster_score{,_cpu}.yaml's fs.get(src, "/tmp/tuned_configs.json")
        target is a local scratch file, not a mission/variant path — it must
        NOT pick up the variant segment (that would be a different bug)."""
        for yaml_path in (
            _DEPLOY_DIR / "cluster_score.yaml",
            _DEPLOY_DIR / "cluster_score_cpu.yaml",
        ):
            rendered = _render(yaml_path, variant_seg="/adb-24m")
            assert "/tmp/tuned_configs.json" in rendered, yaml_path.name


class TestUnaffectedYamlsUnchanged:
    """cluster_train*.yaml and cluster_preprocess.yaml resolve their channel
    paths entirely on the host (CHANNELS_ARG), so they must contain no
    literal tuned_configs.json path composition — if one appears, it means a
    new call site was added directly in the YAML and this test file's
    coverage needs to grow with it."""

    @pytest.mark.parametrize(
        "yaml_path",
        [p for p in _ALL_CLUSTER_YAMLS if p not in _TUNED_CONFIGS_YAMLS],
        ids=lambda p: p.name,
    )
    def test_no_composed_tuned_configs_path(self, yaml_path: Path) -> None:
        rendered = _render(yaml_path, variant_seg="/adb-24m")
        assert not _COMPOSED_PATH_RE.findall(rendered)
