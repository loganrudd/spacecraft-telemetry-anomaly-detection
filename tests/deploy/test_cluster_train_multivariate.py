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

This file also holds the general head/worker env-parity invariant
(TestEnvParityAcrossHeadAndWorkers), which covers every cluster_*.yaml rather
than the two flags this module started with — see that class for why the
per-variable version of the same check was retired.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar

import pytest
import yaml

_DEPLOY_DIR = Path(__file__).parent.parent.parent / "deploy" / "ray"
_CLUSTER_TRAIN = _DEPLOY_DIR / "cluster_train.yaml"
# Both score YAMLs branch on TUNED, so each spells the `ray score` invocation
# TWICE. Plan 020 review item A1 was exactly this shape of bug — a flag added
# to one branch and missed in the other — so these are asserted per-occurrence,
# never on "the first match".
_CLUSTER_SCORES = [_DEPLOY_DIR / "cluster_score.yaml", _DEPLOY_DIR / "cluster_score_cpu.yaml"]
# Every RayJob manifest, discovered rather than enumerated: the head/worker env
# parity invariant (TestEnvParityAcrossHeadAndWorkers) applies to all of them,
# and a list would only cover the ones someone remembered to add — which is the
# exact failure mode that invariant exists to retire.
_ALL_CLUSTER_YAMLS = sorted(_DEPLOY_DIR.glob("cluster_*.yaml"))
# Every cluster whose entrypoint can group channels by subsystem, and therefore
# needs the channels.csv fallback reachable — see TestEnvValueContracts.
_SUBSYSTEM_AWARE_YAMLS = [
    _DEPLOY_DIR / "cluster_train.yaml",
    _DEPLOY_DIR / "cluster_tune.yaml",
    *_CLUSTER_SCORES,
]
# Every cluster that trains or scores a forecaster, and therefore needs the
# 021.7 horizon present at all. cluster_tune.yaml is absent deliberately: HPO
# searches thresholds over saved error arrays, so it never builds a forecast
# head. (Parity for the key is covered generally; this list is about PRESENCE.)
_FORECAST_AWARE_YAMLS = [
    _DEPLOY_DIR / "cluster_train.yaml",
    _DEPLOY_DIR / "cluster_train_cpu.yaml",
    *_CLUSTER_SCORES,
]

# Only the variables the entrypoint actually interpolates need real values;
# expandvars leaves anything else literal, which is harmless here.
_ENV = {
    "PROJECT_ID": "test-proj",
    "MISSION": "ESA-Mission1",
    "VARIANT": "adb-84m",
    "VARIANT_SEG": "/adb-84m",
    "CHANNELS_ARG": "--channels channel_41,channel_42",
    "SUBSYSTEM_ARG": "",
    "NUM_GPUS": "0.16",
    "WINDOW_SIZE_OVERRIDE": "250",
    "MLFLOW_URL": "http://mlflow.invalid",
    "REGION": "us-central1",
    "EVAL_SPLIT": "final_portion",
    "INJECTED_FLAG": "",
    "TUNED": "",
    "PROCESSED_DATA_DIR": "gs://test-proj-processed-data",
    "FORECAST_STEPS": "10",
    "FORECAST_ERROR_REDUCTION": "mean",
    # Interpolated only by cluster_preprocess.yaml, which the glob-discovered
    # parity tests also render. Kept in one dict so every manifest renders
    # fully — a partially-rendered manifest would fail the placeholder check
    # for a reason that has nothing to do with the manifest.
    "TRAIN_FRACTION": "0.8",
    "TRAIN_LOOKBACK": "730D",
    "RAY_IMAGE_TAG": "latest",
    "MULTIVARIATE_ARG": "",
}


def _render(path: Path, **overrides: str) -> str:
    env = {**_ENV, **overrides}
    original = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        return os.path.expandvars(path.read_text())
    finally:
        for k, v in original.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _entrypoint(multivariate_arg: str, **overrides: str) -> str:
    """Render cluster_train.yaml and return its parsed spec.entrypoint."""
    rendered = _render(_CLUSTER_TRAIN, MULTIVARIATE_ARG=multivariate_arg, **overrides)
    entrypoint: str = yaml.safe_load(rendered)["spec"]["entrypoint"]
    return entrypoint


class TestMultivariateFlagThreading:
    def test_flag_present_when_set(self) -> None:
        assert "--multivariate" in _entrypoint("--multivariate")

    def test_flag_absent_by_default(self) -> None:
        assert "--multivariate" not in _entrypoint("")

    @pytest.mark.parametrize("mv", ["", "--multivariate"])
    @pytest.mark.parametrize("sub", ["", "--subsystem subsystem_5"])
    def test_entrypoint_is_a_single_shell_line(self, mv: str, sub: str) -> None:
        """The load-bearing property, for every combination of the optional
        flags being present or absent.

        An empty substitution on its OWN line inside a `>-` folded scalar folds
        to a literal newline, not to nothing — and a newline mid-command is a
        shell COMMAND SEPARATOR, so the tail (`--multivariate`) would be run as
        its own command and fail. This is only survivable when the empty var is
        the last line, where `-` chomping strips it; putting the optional flags
        on one shared line makes it survivable in every position.

        Consecutive and trailing spaces are deliberately NOT asserted against —
        the shell collapses them during word splitting, so they are cosmetic.
        The newline is the only whitespace that changes execution.
        """
        entrypoint = _entrypoint(mv, SUBSYSTEM_ARG=sub)
        assert "\n" not in entrypoint, f"newline splits the command: {entrypoint!r}"
        # Sanity: the command still parses into the expected argv shape.
        argv = entrypoint.split()
        assert argv[:5] == ["spacecraft-telemetry", "--env", "cloud", "ray", "train"], argv

    def test_flag_appended_after_channel_selection(self) -> None:
        """A flag landing *inside* the channel CSV would silently corrupt the
        channel list, so pin that it lands after the whole selector."""
        entrypoint = _entrypoint("--multivariate")
        assert entrypoint.endswith("--multivariate"), entrypoint
        assert entrypoint.index("channel_41,channel_42") < entrypoint.index("--multivariate")

    def test_default_entrypoint_matches_pre_flag_form(self) -> None:
        """The null-default invariant, pinned as a literal rather than
        recomputed — a recomputation would move in lockstep with a regression.
        Split on whitespace so the cosmetic double space left by the two empty
        substitutions doesn't make this a whitespace-formatting test."""
        assert _entrypoint("").split() == [
            "spacecraft-telemetry", "--env", "cloud", "ray", "train",
            "--mission", "ESA-Mission1",
            "--channels", "channel_41,channel_42",
        ]

    def test_subsystem_arg_threaded(self) -> None:
        assert "--subsystem subsystem_5" in _entrypoint(
            "", SUBSYSTEM_ARG="--subsystem subsystem_5"
        )


class TestEnvParityAcrossHeadAndWorkers:
    """EVERY ``SPACECRAFT_*`` key on a head container must reach every worker
    container of the same manifest, with an identical value.

    This is a general invariant, not a list of keys, because the per-variable
    version of it failed twice: ``SPACECRAFT_DATA__SAMPLE_DATA_DIR`` was
    head-only in cluster_train.yaml (021.5-prep — the first ``--multivariate``
    submission aborted with 'cannot resolve --subsystem'), and
    ``SPACECRAFT_MODEL__FORECAST_STEPS`` was missing from workers in 021.7,
    which silently trained H=1 while reporting success. A guard that only
    covers the variables someone remembered to add is the same defect one
    level up.

    Why parity is the right rule: the ``@ray.remote`` fan-out does its real
    work inside WORKER tasks — make_dataloaders, the (C, H) output head,
    collapse_forecast_errors, the subsystem grouping all run there — while the
    RayJob entrypoint runs on the head. A head-only value therefore configures
    the driver and nothing that computes (plan 019 B2, the rule
    SPACECRAFT_VARIANT already follows).

    Manifests are globbed, not enumerated, so a NEW cluster YAML is covered the
    day it lands rather than when someone remembers to add it here.
    """

    # Keys that are deliberately head-only. Each entry needs a reason: the
    # setting must be read by the DRIVER and never inside a @ray.remote task,
    # which makes a worker copy dead config rather than a safety net.
    _HEAD_ONLY: ClassVar[dict[str, str]] = {
        # run_all_sweeps reads this in the driver to decide how many subsystem
        # sweeps to launch concurrently; trial functions never read it.
        "SPACECRAFT_TUNE__MAX_PARALLEL_SUBSYSTEMS": "driver-only sweep concurrency",
    }

    @staticmethod
    def _container_envs(template: dict) -> dict[str, str]:
        containers = template["spec"]["containers"]
        return {e["name"]: e.get("value") for e in containers[0].get("env", [])}

    @classmethod
    def _spacecraft_envs(cls, template: dict) -> dict[str, str]:
        return {
            k: v
            for k, v in cls._container_envs(template).items()
            if k.startswith("SPACECRAFT_")
        }

    @pytest.mark.parametrize("yaml_path", _ALL_CLUSTER_YAMLS, ids=lambda p: p.name)
    def test_every_spacecraft_key_reaches_every_worker(self, yaml_path: Path) -> None:
        spec = yaml.safe_load(_render(yaml_path))["spec"]["rayClusterSpec"]
        head = self._spacecraft_envs(spec["headGroupSpec"]["template"])

        # Vacuous-pass guard: a parse that silently yielded {} would make every
        # assertion below trivially true, which is how this class would stop
        # protecting anything without failing.
        assert head, f"{yaml_path.name}: parsed zero SPACECRAFT_* keys on the head"

        # cluster_score_cpu.yaml and cluster_train_cpu.yaml are deliberately
        # single-node (no workerGroupSpecs), so head-only is correct there.
        for i, worker in enumerate(spec.get("workerGroupSpecs") or []):
            wenv = self._spacecraft_envs(worker["template"])
            group = worker.get("groupName", i)
            for key, head_value in head.items():
                if key in self._HEAD_ONLY:
                    assert key not in wenv, (
                        f"{yaml_path.name}: worker[{group}] sets {key}, which is "
                        f"documented head-only ({self._HEAD_ONLY[key]}). Either the "
                        "setting is now read inside a @ray.remote task — in which "
                        "case remove it from _HEAD_ONLY — or this is dead config."
                    )
                    continue
                assert key in wenv, (
                    f"{yaml_path.name}: worker[{group}] is missing {key}, which the "
                    f"head sets to {head_value!r}. The @ray.remote fan-out reads "
                    "settings inside worker tasks, so the workers would run a "
                    "DIFFERENT configuration than the one submitted."
                )
                assert wenv[key] == head_value, (
                    f"{yaml_path.name}: worker[{group}] {key}={wenv[key]!r} diverges "
                    f"from head {head_value!r}"
                )

    @pytest.mark.parametrize("yaml_path", _ALL_CLUSTER_YAMLS, ids=lambda p: p.name)
    def test_no_worker_sets_a_spacecraft_key_the_head_lacks(self, yaml_path: Path) -> None:
        """The reverse direction. A worker-only key is the same divergence
        viewed from the other side — the driver's own reads (channel discovery,
        experiment naming) would use a different value than the tasks."""
        spec = yaml.safe_load(_render(yaml_path))["spec"]["rayClusterSpec"]
        head = self._spacecraft_envs(spec["headGroupSpec"]["template"])
        for i, worker in enumerate(spec.get("workerGroupSpecs") or []):
            wenv = self._spacecraft_envs(worker["template"])
            extra = sorted(set(wenv) - set(head))
            assert not extra, (
                f"{yaml_path.name}: worker[{worker.get('groupName', i)}] sets "
                f"{extra} which the head does not"
            )

    @pytest.mark.parametrize("yaml_path", _ALL_CLUSTER_YAMLS, ids=lambda p: p.name)
    def test_no_unresolved_placeholders_in_spacecraft_values(self, yaml_path: Path) -> None:
        """Generalises the old per-key 'did FORECAST_STEPS interpolate?' check:
        a ``${...}`` surviving into a rendered value means the submitting
        script never exported it, and the pod would take the config default
        while the manifest looks correct."""
        spec = yaml.safe_load(_render(yaml_path))["spec"]["rayClusterSpec"]
        templates = [spec["headGroupSpec"]["template"]] + [
            w["template"] for w in (spec.get("workerGroupSpecs") or [])
        ]
        for template in templates:
            for key, value in self._spacecraft_envs(template).items():
                assert "${" not in str(value), (
                    f"{yaml_path.name}: {key}={value!r} kept an unresolved placeholder"
                )


class TestEnvValueContracts:
    """Value-shape assertions that parity alone cannot make.

    Parity says head and workers agree; it says nothing about whether the
    agreed value is usable. These two are the ones with a known failure
    behind them.
    """

    _ENVS = TestEnvParityAcrossHeadAndWorkers._spacecraft_envs

    @pytest.mark.parametrize("yaml_path", _SUBSYSTEM_AWARE_YAMLS, ids=lambda p: p.name)
    def test_sample_data_dir_is_a_gcs_uri(self, yaml_path: Path) -> None:
        """`--subsystem` / `--multivariate` group channels by subsystem, which
        for an ESA mission resolves ONLY through the channels.csv fallback in
        core/metadata.py — the channel_subsystems.json branch is written by the
        ISS pipeline alone and never exists for ESA. configs/cloud.yaml points
        data.sample_data_dir at a LOCAL path absent from the image, so every
        subsystem-aware manifest must override it to the GCS bucket."""
        key = "SPACECRAFT_DATA__SAMPLE_DATA_DIR"
        spec = yaml.safe_load(_render(yaml_path))["spec"]["rayClusterSpec"]
        head = self._ENVS(spec["headGroupSpec"]["template"])
        assert key in head, f"{yaml_path.name}: head is missing {key}"
        assert head[key].startswith("gs://"), (
            f"{yaml_path.name}: head {key}={head[key]!r} is not a GCS URI; "
            "the image has no local data/ tree"
        )

    @pytest.mark.parametrize("yaml_path", _FORECAST_AWARE_YAMLS, ids=lambda p: p.name)
    def test_forecast_steps_present_and_interpolated(self, yaml_path: Path) -> None:
        """Presence on the head is not implied by parity — a key absent
        everywhere is perfectly consistent, and would train H=1 silently."""
        key = "SPACECRAFT_MODEL__FORECAST_STEPS"
        spec = yaml.safe_load(_render(yaml_path))["spec"]["rayClusterSpec"]
        head = self._ENVS(spec["headGroupSpec"]["template"])
        assert head.get(key) == "10", (
            f"{yaml_path.name}: head {key}={head.get(key)!r} did not interpolate "
            "FORECAST_STEPS"
        )

    @pytest.mark.parametrize("yaml_path", _CLUSTER_SCORES, ids=lambda p: p.name)
    def test_error_reduction_present_wherever_scoring_happens(self, yaml_path: Path) -> None:
        """The reduction is scoring-time only, so it belongs on the score
        YAMLs alone — collapse_forecast_errors reads it inside the worker."""
        key = "SPACECRAFT_MODEL__FORECAST_ERROR_REDUCTION"
        spec = yaml.safe_load(_render(yaml_path))["spec"]["rayClusterSpec"]
        head = self._ENVS(spec["headGroupSpec"]["template"])
        assert head.get(key) == "mean", f"{yaml_path.name}: head {key}={head.get(key)!r}"


class TestScoreYamlsFlagBothBranches:
    """cluster_score{,_cpu}.yaml each spell `ray score` twice (TUNED and
    untuned). A flag reaching only one branch is plan-020-review item A1 all
    over again — so every occurrence is asserted, not just the first."""

    @pytest.mark.parametrize("yaml_path", _CLUSTER_SCORES, ids=lambda p: p.name)
    def test_both_branches_carry_multivariate(self, yaml_path: Path) -> None:
        rendered = _render(yaml_path, MULTIVARIATE_ARG="--multivariate")
        n_invocations = rendered.count("ray score")
        assert n_invocations == 2, f"expected 2 branches, found {n_invocations}"
        assert rendered.count("--multivariate") == n_invocations, (
            f"{yaml_path.name}: --multivariate reached "
            f"{rendered.count('--multivariate')} of {n_invocations} branches"
        )

    @pytest.mark.parametrize("yaml_path", _CLUSTER_SCORES, ids=lambda p: p.name)
    def test_both_branches_carry_subsystem(self, yaml_path: Path) -> None:
        rendered = _render(yaml_path, SUBSYSTEM_ARG="--subsystem subsystem_5")
        assert rendered.count("--subsystem subsystem_5") == rendered.count("ray score")

    @pytest.mark.parametrize("yaml_path", _CLUSTER_SCORES, ids=lambda p: p.name)
    def test_no_flags_by_default(self, yaml_path: Path) -> None:
        rendered = _render(yaml_path, MULTIVARIATE_ARG="", SUBSYSTEM_ARG="")
        assert "--multivariate" not in rendered
        assert "--subsystem" not in rendered
        assert "${" not in rendered, "unresolved placeholder in rendered score YAML"


