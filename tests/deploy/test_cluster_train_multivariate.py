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

import pytest
import yaml

_DEPLOY_DIR = Path(__file__).parent.parent.parent / "deploy" / "ray"
_CLUSTER_TRAIN = _DEPLOY_DIR / "cluster_train.yaml"
# Both score YAMLs branch on TUNED, so each spells the `ray score` invocation
# TWICE. Plan 020 review item A1 was exactly this shape of bug — a flag added
# to one branch and missed in the other — so these are asserted per-occurrence,
# never on "the first match".
_CLUSTER_SCORES = [_DEPLOY_DIR / "cluster_score.yaml", _DEPLOY_DIR / "cluster_score_cpu.yaml"]
# Every cluster whose entrypoint can group channels by subsystem, and therefore
# needs the channels.csv fallback reachable — see TestSubsystemMapEnvParity.
_SUBSYSTEM_AWARE_YAMLS = [
    _DEPLOY_DIR / "cluster_train.yaml",
    _DEPLOY_DIR / "cluster_tune.yaml",
    *_CLUSTER_SCORES,
]
# Every cluster that trains or scores a forecaster, and therefore needs the
# 021.7 horizon — see TestForecastStepsEnvParity. cluster_tune.yaml is absent
# deliberately: HPO searches thresholds over saved error arrays, so it never
# builds a forecast head.
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


class TestSubsystemMapEnvParity:
    """`--subsystem` / `--multivariate` group channels by subsystem, which for
    an ESA mission resolves ONLY through the channels.csv fallback in
    core/metadata.py — the channel_subsystems.json branch is written by the ISS
    pipeline alone and never exists for ESA. configs/cloud.yaml points
    data.sample_data_dir at a LOCAL path absent from the image, so every cluster
    YAML whose entrypoint can group by subsystem must override it to the GCS
    bucket. cluster_tune.yaml always did; cluster_train.yaml did not, which is
    what made the first --multivariate submission abort with
    'cannot resolve --subsystem'.

    Parity between head and worker is asserted separately because the
    @ray.remote fan-out splits inside worker tasks — a head-only override
    diverges silently (plan 019 B2, the same rule SPACECRAFT_VARIANT follows).
    """

    _KEY = "SPACECRAFT_DATA__SAMPLE_DATA_DIR"

    @staticmethod
    def _container_envs(template: dict) -> dict[str, str]:
        containers = template["spec"]["containers"]
        return {e["name"]: e.get("value") for e in containers[0].get("env", [])}

    @pytest.mark.parametrize("yaml_path", _SUBSYSTEM_AWARE_YAMLS, ids=lambda p: p.name)
    def test_sample_data_dir_set_on_head_and_every_worker(self, yaml_path: Path) -> None:
        spec = yaml.safe_load(_render(yaml_path))["spec"]["rayClusterSpec"]

        head = self._container_envs(spec["headGroupSpec"]["template"])
        assert self._KEY in head, f"{yaml_path.name}: head is missing {self._KEY}"
        assert head[self._KEY].startswith("gs://"), (
            f"{yaml_path.name}: head {self._KEY}={head[self._KEY]!r} is not a GCS URI; "
            "the image has no local data/ tree"
        )

        # cluster_score_cpu.yaml is deliberately single-node (no workerGroupSpecs),
        # so head-only is correct there — assert parity only where workers exist.
        for i, worker in enumerate(spec.get("workerGroupSpecs") or []):
            wenv = self._container_envs(worker["template"])
            assert wenv.get(self._KEY) == head[self._KEY], (
                f"{yaml_path.name}: worker[{i}] {self._KEY}={wenv.get(self._KEY)!r} "
                f"diverges from head {head[self._KEY]!r}"
            )


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


class TestForecastStepsEnvParity:
    """``SPACECRAFT_MODEL__FORECAST_STEPS`` must reach head AND every worker.

    Plan 021.7's horizon has no CLI flag — the only knob is this env var, and
    the RayJob YAMLs enumerate env vars one by one, so exporting it in the
    shell that runs `make cloud-train` reaches the submitting script and NOT
    the pods. The @ray.remote fan-out builds its dataloaders and its (C, H)
    output head inside the WORKER, so a missing or head-only value trains
    H=1 and reports success — a wrong answer that looks like a right one,
    which is the worst failure mode this experiment has.

    Same rule and same rationale as TestSubsystemMapEnvParity above; kept as
    a separate class because the acceptable values differ (an int, not a URI).
    """

    _KEY = "SPACECRAFT_MODEL__FORECAST_STEPS"

    @pytest.mark.parametrize("yaml_path", _FORECAST_AWARE_YAMLS, ids=lambda p: p.name)
    def test_forecast_steps_set_on_head_and_every_worker(self, yaml_path: Path) -> None:
        spec = yaml.safe_load(_render(yaml_path))["spec"]["rayClusterSpec"]
        envs = TestSubsystemMapEnvParity._container_envs

        head = envs(spec["headGroupSpec"]["template"])
        assert self._KEY in head, f"{yaml_path.name}: head is missing {self._KEY}"
        assert head[self._KEY] == "10", (
            f"{yaml_path.name}: head {self._KEY}={head[self._KEY]!r} did not "
            "interpolate FORECAST_STEPS"
        )

        for i, worker in enumerate(spec.get("workerGroupSpecs") or []):
            wenv = envs(worker["template"])
            assert wenv.get(self._KEY) == head[self._KEY], (
                f"{yaml_path.name}: worker[{i}] {self._KEY}={wenv.get(self._KEY)!r} "
                f"diverges from head {head[self._KEY]!r} — the fan-out would "
                "train a different horizon than requested"
            )

    @pytest.mark.parametrize("yaml_path", _CLUSTER_SCORES, ids=lambda p: p.name)
    def test_error_reduction_set_wherever_scoring_happens(self, yaml_path: Path) -> None:
        """The reduction is scoring-time only, so it belongs on the score
        YAMLs alone — but there it has the same head/worker parity rule:
        collapse_forecast_errors runs inside the worker task."""
        key = "SPACECRAFT_MODEL__FORECAST_ERROR_REDUCTION"
        spec = yaml.safe_load(_render(yaml_path))["spec"]["rayClusterSpec"]
        envs = TestSubsystemMapEnvParity._container_envs

        head = envs(spec["headGroupSpec"]["template"])
        assert head.get(key) == "mean", f"{yaml_path.name}: head {key}={head.get(key)!r}"
        for i, worker in enumerate(spec.get("workerGroupSpecs") or []):
            assert envs(worker["template"]).get(key) == head[key], (
                f"{yaml_path.name}: worker[{i}] {key} diverges from head"
            )
