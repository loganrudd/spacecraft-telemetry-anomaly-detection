"""Integration test for esa_adb.report.build_report — wires events, timeline,
detections, and metrics together end-to-end against a real SQLite MLflow
backend and a tiny hand-built series/labels fixture.

This does not re-verify the metric math (test_metrics.py already does that
exhaustively) — it proves the pipeline assembles correctly: the right scoring
runs get found, the right timeline gets clipped for the tuned row, and the
report's shape (rows, scopes, paper-reference values) is what
scripts/esa_adb_report.py expects to render.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from spacecraft_telemetry.core.config import Settings, load_settings
from spacecraft_telemetry.esa_adb.offline import OfflineRunMap, RunSpec
from spacecraft_telemetry.esa_adb.report import _PAPER_REFERENCE, build_report
from spacecraft_telemetry.mlflow_tracking import (
    common_tags,
    experiment_name,
    log_artifact_bytes,
    log_params,
    open_run,
)
from spacecraft_telemetry.model.io import errors_to_bytes, threshold_to_bytes

_MISSION = "ESA-Mission1-ReportTest"
_CHANNEL = "channel_41"
_CHANNEL_2 = "channel_42"
_WINDOW_SIZE = 3
_PREDICTION_HORIZON = 1
_FREQ_S = 90
_N_ROWS = 20  # -> M = 20 - (3+1) + 1 = 17 windows

_SERIES_SCHEMA = pa.schema(
    [
        pa.field("telemetry_timestamp", pa.timestamp("us", tz="UTC")),
        pa.field("value_normalized", pa.float32()),
        pa.field("segment_id", pa.int32()),
        pa.field("is_anomaly", pa.bool_()),
    ]
)


def _write_series(processed_dir: Path, channel: str = _CHANNEL) -> None:
    base = datetime(2000, 1, 1, tzinfo=UTC)
    timestamps = [
        pa.scalar(base.timestamp() + i * _FREQ_S, type=pa.timestamp("s", tz="UTC")).cast(
            pa.timestamp("us", tz="UTC")
        )
        for i in range(_N_ROWS)
    ]
    table = pa.table(
        {
            "telemetry_timestamp": pa.array(timestamps, type=pa.timestamp("us", tz="UTC")),
            "value_normalized": pa.array([0.0] * _N_ROWS, type=pa.float32()),
            "segment_id": pa.array([0] * _N_ROWS, type=pa.int32()),
            "is_anomaly": pa.array([False] * _N_ROWS, type=pa.bool_()),
        },
        schema=_SERIES_SCHEMA,
    )
    partition_dir = (
        processed_dir / _MISSION / "test" / f"mission_id={_MISSION}" / f"channel_id={channel}"
    )
    partition_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, partition_dir / "part.parquet")


def _write_labels(sample_dir: Path) -> None:
    mission_dir = sample_dir / _MISSION
    mission_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "ID": "id_1",
                "Channel": _CHANNEL,
                # Window index 14's target timestamp is 2000-01-01T00:25:30Z
                # (base + (14+span-1)*90s, span=4) — placed inside this window.
                "StartTime": "2000-01-01T00:25:00Z",
                "EndTime": "2000-01-01T00:26:00Z",
            }
        ]
    ).to_csv(mission_dir / "labels.csv", index=False)
    pd.DataFrame(
        [
            {
                "ID": "id_1",
                "Class": "class_1",
                "Subclass": "subclass_1",
                "Category": "Anomaly",
                "Dimensionality": "Univariate",
                "Locality": "Local",
                "Length": "Subsequence",
            }
        ]
    ).to_csv(mission_dir / "anomaly_types.csv", index=False)


def _write_two_category_labels(sample_dir: Path) -> None:
    """Like _write_labels, but with a second event of category Rare Event.

    Used by TestScopeInvariants: Rare Event is excluded from the
    anomalies_only scope but not all_events, so this fixture is what makes
    the anomalies_only-n_events-never-exceeds-all_events invariant test a
    strict inequality rather than a trivially-equal one.
    """
    mission_dir = sample_dir / _MISSION
    mission_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "ID": "id_1",
                "Channel": _CHANNEL,
                # Same window-14-overlapping interval as _write_labels.
                "StartTime": "2000-01-01T00:25:00Z",
                "EndTime": "2000-01-01T00:26:00Z",
            },
            {
                "ID": "id_2",
                "Channel": _CHANNEL,
                # Elsewhere in the window — undetected either way, but still
                # a distinct annotated event that must count in all_events.
                "StartTime": "2000-01-01T00:01:00Z",
                "EndTime": "2000-01-01T00:02:00Z",
            },
        ]
    ).to_csv(mission_dir / "labels.csv", index=False)
    pd.DataFrame(
        [
            {
                "ID": "id_1",
                "Class": "class_1",
                "Subclass": "subclass_1",
                "Category": "Anomaly",
                "Dimensionality": "Univariate",
                "Locality": "Local",
                "Length": "Subsequence",
            },
            {
                "ID": "id_2",
                "Class": "class_2",
                "Subclass": "subclass_2",
                "Category": "Rare Event",
                "Dimensionality": "Univariate",
                "Locality": "Local",
                "Length": "Subsequence",
            },
        ]
    ).to_csv(mission_dir / "anomaly_types.csv", index=False)


def _settings(processed_dir: Path, sample_dir: Path, mlflow_uri: str) -> Settings:
    base_settings = load_settings("test")
    return base_settings.model_copy(
        update={
            "preprocess": base_settings.preprocess.model_copy(
                update={"processed_data_dir": str(processed_dir)}
            ),
            "data": base_settings.data.model_copy(update={"sample_data_dir": str(sample_dir)}),
            "model": base_settings.model.model_copy(
                update={"window_size": _WINDOW_SIZE, "prediction_horizon": _PREDICTION_HORIZON}
            ),
            "mlflow": base_settings.mlflow.model_copy(update={"tracking_uri": mlflow_uri}),
        }
    )


def _log_scoring_run(
    settings: Settings,
    *,
    tuned: bool,
    tuned_source: str | None = None,
    channel: str = _CHANNEL,
    also_tuned_from_run: bool = False,
) -> None:
    """Flag window index 14 alone — well after the hpo_eval_fraction=0.6 cutoff (idx 10).

    ``tuned_source`` sets the ``tuned_source`` tag instead of ``tuned_from_run``
    — i.e. simulates a grid-produced (scripts/threshold_ceiling.py) config
    rather than a Ray Tune trial. Ignored when ``tuned`` is False.

    ``also_tuned_from_run``: when ``tuned_source`` is given, ALSO set
    ``tuned_from_run`` — this is the real shape a post-022.2 Ray Tune scoring
    run carries (``_tuned_meta`` reads ``_meta.source`` into the
    ``tuned_source`` tag unconditionally, alongside ``tuned_from_run`` from
    ``_meta.run_id``), as opposed to the grid writer's config, which has no
    HPO run behind it and therefore only ever carries ``tuned_source`` alone.
    """
    smoothed = np.zeros(17, dtype=np.float64)
    smoothed[14] = 5.0
    threshold = np.ones(17, dtype=np.float64)

    exp = experiment_name(settings.model.model_type, "scoring", _MISSION)
    extra = {"eval_split": "final_portion" if tuned else "full_test"}
    if tuned:
        if tuned_source is not None:
            extra["tuned_source"] = tuned_source
            if also_tuned_from_run:
                extra["tuned_from_run"] = "fake-hpo-run"
        else:
            extra["tuned_from_run"] = "fake-hpo-run"
    tags = common_tags(
        model_type=settings.model.model_type,
        mission=_MISSION,
        phase="scoring",
        channel=channel,
        extra=extra,
    )
    with open_run(experiment=exp, run_name=channel, tags=tags) as run:
        assert run is not None
        log_params({"threshold_min_anomaly_len": 1})
        log_artifact_bytes(errors_to_bytes(smoothed), "errors.npy")
        log_artifact_bytes(threshold_to_bytes(threshold), "threshold.npy")


class TestBuildReport:
    def test_report_structure(self, tmp_path: Path, mlflow_uri: str) -> None:
        processed_dir = tmp_path / "processed"
        sample_dir = tmp_path / "sample"
        _write_series(processed_dir)
        _write_labels(sample_dir)
        settings = _settings(processed_dir, sample_dir, mlflow_uri)

        _log_scoring_run(settings, tuned=False)
        _log_scoring_run(settings, tuned=True)

        report = build_report(settings, _MISSION, channels=[_CHANNEL])

        assert report["mission"] == _MISSION
        assert report["channels"] == [_CHANNEL]
        assert len(report["footnotes"]) > 0

        scopes = {row["scope"] for row in report["rows"]}
        assert scopes == {"all_events", "anomalies_only"}

        # 2 scopes x (untuned + tuned + 2 paper rows) = 8 rows.
        assert len(report["rows"]) == 8

    def test_include_tuned_false_omits_tuned_rows(self, tmp_path: Path, mlflow_uri: str) -> None:
        """include_tuned=False drops the tuned row and never queries a tuned run.

        Only a BASELINE run is logged here — with include_tuned=True this would
        raise (no tuned run found). Succeeding proves the tuned lookup is skipped
        entirely rather than merely dropped from the output.
        """
        processed_dir = tmp_path / "processed"
        sample_dir = tmp_path / "sample"
        _write_series(processed_dir)
        _write_labels(sample_dir)
        settings = _settings(processed_dir, sample_dir, mlflow_uri)

        _log_scoring_run(settings, tuned=False)

        report = build_report(settings, _MISSION, channels=[_CHANNEL], include_tuned=False)

        labels = [r["label"] for r in report["rows"]]
        assert not any(label == "ours (tuned)" for label in labels)
        # 2 scopes x (untuned + 2 paper rows) = 6 rows.
        assert len(report["rows"]) == 6
        assert any("OMITTED" in note for note in report["footnotes"]), (
            "omission of the tuned row must be disclosed in the footnotes"
        )

    def test_include_tuned_true_requires_tuned_run(self, tmp_path: Path, mlflow_uri: str) -> None:
        """The default must still fail loudly when no tuned run exists."""
        processed_dir = tmp_path / "processed"
        sample_dir = tmp_path / "sample"
        _write_series(processed_dir)
        _write_labels(sample_dir)
        settings = _settings(processed_dir, sample_dir, mlflow_uri)

        _log_scoring_run(settings, tuned=False)

        with pytest.raises(RuntimeError, match="tuned scoring run"):
            build_report(settings, _MISSION, channels=[_CHANNEL])


    def test_configure_mlflow_failure_is_logged_not_silently_swallowed(
        self,
        tmp_path: Path,
        mlflow_uri: str,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A configure_mlflow failure (e.g. GCP ID-token fetch) must be visible.

        Regression for the observed 2026-08-15 case: fetch_id_token_failed
        passed silently and the run only worked because ambient gcloud
        credentials happened to cover it. build_report's run_map=None branch
        is exactly where MLflow is required, so the failure must not vanish —
        it still fails loudly downstream rather than being masked.

        Checks capsys rather than structlog.testing.capture_logs() — see
        test_detections.py's matching test for why (cache_logger_on_first_use
        breaks capture_logs() for already-realized module loggers).
        """
        processed_dir = tmp_path / "processed"
        sample_dir = tmp_path / "sample"
        _write_series(processed_dir)
        _write_labels(sample_dir)
        settings = _settings(processed_dir, sample_dir, mlflow_uri)

        def _raise(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("fetch_id_token_failed")

        monkeypatch.setattr("spacecraft_telemetry.esa_adb.report.configure_mlflow", _raise)

        with pytest.raises(RuntimeError, match="No MLflow experiment"):
            build_report(settings, _MISSION, channels=[_CHANNEL], include_tuned=False)

        stdout = capsys.readouterr().out
        assert "esa_adb.report.configure_mlflow_failed" in stdout
        assert "warning" in stdout
        assert "fetch_id_token_failed" in stdout

    def test_fetches_detection_intervals_once_per_variant(
        self, tmp_path: Path, mlflow_uri: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression for A1: build_report must not re-fetch per-channel detections.

        Before A1, build_report called per_channel_detection_intervals() AND
        mission_detection_intervals() (which calls it again internally) for
        each of the untuned/tuned variants — every artifact downloaded and
        every parquet re-read twice per scope. It must now call
        per_channel_detection_intervals() exactly once per variant (untuned,
        tuned) — twice total, not four times.
        """
        import spacecraft_telemetry.esa_adb.report as report_module

        processed_dir = tmp_path / "processed"
        sample_dir = tmp_path / "sample"
        _write_series(processed_dir)
        _write_labels(sample_dir)
        settings = _settings(processed_dir, sample_dir, mlflow_uri)

        _log_scoring_run(settings, tuned=False)
        _log_scoring_run(settings, tuned=True)

        real = report_module.per_channel_detection_intervals
        calls: list[bool] = []

        def _counting(*args: object, **kwargs: object) -> object:
            calls.append(True)
            return real(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(report_module, "per_channel_detection_intervals", _counting)

        build_report(settings, _MISSION, channels=[_CHANNEL])

        assert len(calls) == 2, (
            f"expected 2 calls (untuned + tuned), got {len(calls)} — "
            "a per-channel detection fetch is being repeated"
        )

    def test_preloads_series_metadata_once_per_channel(
        self, tmp_path: Path, mlflow_uri: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression for P2/P3: the test partition must be read once per channel.

        Before P2/P3, mission_timeline, hpo_cutoff, and per-channel detection
        reconstruction (untuned + tuned) each independently re-read the full
        parquet partition — up to 4 reads per channel. build_report now
        preloads (segment_ids, is_anomaly, timestamps) once per channel via
        load_series_metadata() and threads it through all of those.

        Patches every fallback disk-read entry point the downstream functions
        would fall back to if a caller forgot to pass metadata_by_channel
        through (esa_adb.timeline's own load_series_metadata reference, and
        the window_target_timestamps disk-reading path in both report.py's
        hpo_cutoff and detections.py's _intervals_from_arrays) — a single
        counter shared across all of them, so a regression in any one of
        those call sites is caught, not just the top-level preload.
        """
        import spacecraft_telemetry.esa_adb.detections as detections_module
        import spacecraft_telemetry.esa_adb.report as report_module
        import spacecraft_telemetry.esa_adb.timeline as timeline_module

        processed_dir = tmp_path / "processed"
        sample_dir = tmp_path / "sample"
        _write_series(processed_dir)
        _write_labels(sample_dir)
        settings = _settings(processed_dir, sample_dir, mlflow_uri)

        _log_scoring_run(settings, tuned=False)
        _log_scoring_run(settings, tuned=True)

        calls: list[str] = []

        def _counting_metadata_call(real: object):
            def _wrapped(*args: object, **kwargs: object) -> object:
                calls.append("load_series_metadata")
                return real(*args, **kwargs)  # type: ignore[operator]

            return _wrapped

        def _disk_read_fallback_call(*_args: object, **_kwargs: object) -> None:
            raise AssertionError(
                "window_target_timestamps (full disk-read fallback) must not be "
                "called when build_report has preloaded metadata_by_channel"
            )

        monkeypatch.setattr(
            report_module, "load_series_metadata",
            _counting_metadata_call(report_module.load_series_metadata),
        )
        monkeypatch.setattr(
            timeline_module, "load_series_metadata",
            _counting_metadata_call(timeline_module.load_series_metadata),
        )
        monkeypatch.setattr(report_module, "window_target_timestamps", _disk_read_fallback_call)
        monkeypatch.setattr(
            detections_module, "window_target_timestamps", _disk_read_fallback_call
        )

        build_report(settings, _MISSION, channels=[_CHANNEL])

        assert len(calls) == 1, (
            f"expected 1 total load_series_metadata call for 1 channel, got {len(calls)} "
            f"({calls}) — the test partition is being re-read somewhere downstream"
        )

    def test_ours_rows_detect_the_event(self, tmp_path: Path, mlflow_uri: str) -> None:
        """The single event overlaps the flagged window in both untuned and tuned runs."""
        processed_dir = tmp_path / "processed"
        sample_dir = tmp_path / "sample"
        _write_series(processed_dir)
        _write_labels(sample_dir)
        settings = _settings(processed_dir, sample_dir, mlflow_uri)

        _log_scoring_run(settings, tuned=False)
        _log_scoring_run(settings, tuned=True)

        report = build_report(settings, _MISSION, channels=[_CHANNEL])
        ours_rows = [
            r
            for r in report["rows"]
            if r["scope"] == "anomalies_only" and r["label"].startswith("ours")
        ]
        assert len(ours_rows) == 2
        for row in ours_rows:
            assert row["n_events"] == 1
            assert row["recall"] == 1.0, f"{row['label']}: event should have been detected"
            assert 0.0 <= row["precision"] <= 1.0
            assert row["channel_aware_f0_5"] is not None

    def test_paper_rows_match_hardcoded_reference(self, tmp_path: Path, mlflow_uri: str) -> None:
        processed_dir = tmp_path / "processed"
        sample_dir = tmp_path / "sample"
        _write_series(processed_dir)
        _write_labels(sample_dir)
        settings = _settings(processed_dir, sample_dir, mlflow_uri)

        _log_scoring_run(settings, tuned=False)
        _log_scoring_run(settings, tuned=True)

        report = build_report(settings, _MISSION, channels=[_CHANNEL])
        paper_rows = {
            (r["scope"], r["label"]): r for r in report["rows"] if r["label"].startswith("paper")
        }
        for scope, ref_by_label in _PAPER_REFERENCE.items():
            for label, ref in ref_by_label.items():
                row = paper_rows[(scope, label)]
                assert row["precision"] == ref["precision"]
                assert row["recall"] == ref["recall"]
                assert row["f0_5"] == ref["f0_5"]
                assert row["channel_aware_f0_5"] is None
                assert row["n_events"] is None


class TestTunedRowProvenance:
    """docs/plans/022 stage 022.2b: the report renders each row's ACTUAL
    provenance rather than an asserted "per-subsystem Ray Tune HPO" claim."""

    def test_grid_provenance_renders_in_footnote_and_params(
        self, tmp_path: Path, mlflow_uri: str
    ) -> None:
        processed_dir = tmp_path / "processed"
        sample_dir = tmp_path / "sample"
        _write_series(processed_dir)
        _write_labels(sample_dir)
        settings = _settings(processed_dir, sample_dir, mlflow_uri)

        _log_scoring_run(settings, tuned=False)
        _log_scoring_run(
            settings,
            tuned=True,
            tuned_source="scripts/threshold_ceiling.py exhaustive grid (mission)",
        )

        report = build_report(settings, _MISSION, channels=[_CHANNEL])

        tuned_rows = [r for r in report["rows"] if r["label"] == "ours (tuned)"]
        assert tuned_rows
        for row in tuned_rows:
            assert row["params"] == "scripts/threshold_ceiling.py exhaustive grid (mission)"
        assert any(
            "scripts/threshold_ceiling.py exhaustive grid (mission)" in note
            for note in report["footnotes"]
        )
        assert not any("Ray Tune" in note for note in report["footnotes"])

    def test_ray_tune_provenance_renders_in_footnote_and_params(
        self, tmp_path: Path, mlflow_uri: str
    ) -> None:
        """Legacy tag shape: tuned_from_run only, no tuned_source tag.

        A scoring run tagged this way predates 022.2's _meta.source tagging,
        or was produced by a Ray Tune run whose tuned_configs.json entry had
        no `_meta.source` for some other reason. tuned_provenance's hardcoded
        fallback string is the only source of truth here — this is the ONE
        case where the pre-022.2b hardcoded claim was actually true.
        """
        processed_dir = tmp_path / "processed"
        sample_dir = tmp_path / "sample"
        _write_series(processed_dir)
        _write_labels(sample_dir)
        settings = _settings(processed_dir, sample_dir, mlflow_uri)

        _log_scoring_run(settings, tuned=False)
        _log_scoring_run(settings, tuned=True)  # tuned_from_run only -> Ray Tune

        report = build_report(settings, _MISSION, channels=[_CHANNEL])

        tuned_rows = [r for r in report["rows"] if r["label"] == "ours (tuned)"]
        assert tuned_rows
        for row in tuned_rows:
            assert row["params"] == "per-subsystem Ray Tune HPO scoring"
        assert any(
            "per-subsystem Ray Tune HPO scoring" in note for note in report["footnotes"]
        )
        # The row-description footnote is spliced in at FOOTNOTES[:2] + this
        # + FOOTNOTES[2:] — pin the INDEX, not just presence, so a splice-order
        # regression is caught (build_report.py's footnotes assembly).
        assert report["footnotes"][2].startswith(
            "The 'ours (protocol-matched)' row"
        )

    def test_ray_tune_source_tag_renders_prose_not_a_file_path(
        self, tmp_path: Path, mlflow_uri: str
    ) -> None:
        """Real post-022.2 tag shape: BOTH tuned_from_run AND tuned_source are
        set (runner._tuned_meta reads _meta.source unconditionally, alongside
        _meta.run_id). tuned_provenance prefers tuned_source when present, so
        whatever ray_fanout.tune's _to_entry writes as `_meta.source` lands
        verbatim here — this is the C4 regression (docs/reviews/022): today
        that string is the literal module path `ray_fanout/tune.py Ray Tune
        HPO sweep`, a user-visible output bug. Stage 2.1 changes the SOURCE
        string to prose; this test is pinned to TODAY's actual (wrong)
        behaviour and must be flipped in the same commit as that fix.
        """
        processed_dir = tmp_path / "processed"
        sample_dir = tmp_path / "sample"
        _write_series(processed_dir)
        _write_labels(sample_dir)
        settings = _settings(processed_dir, sample_dir, mlflow_uri)

        _log_scoring_run(settings, tuned=False)
        _log_scoring_run(
            settings,
            tuned=True,
            tuned_source="ray_fanout/tune.py Ray Tune HPO sweep",
            also_tuned_from_run=True,
        )

        report = build_report(settings, _MISSION, channels=[_CHANNEL])

        tuned_rows = [r for r in report["rows"] if r["label"] == "ours (tuned)"]
        assert tuned_rows
        for row in tuned_rows:
            assert row["params"] == "ray_fanout/tune.py Ray Tune HPO sweep"

    def test_run_map_mode_renders_neither_claim(self, tmp_path: Path, mlflow_uri: str) -> None:
        """--run-map mode never contacts MLflow, so tags are unreadable — the
        report must degrade to an honest "provenance unavailable" line rather
        than falling back to either hardcoded claim (the old bug, reintroduced
        behind a harder-to-see branch, per docs/plans/022 stage 022.2b)."""
        processed_dir = tmp_path / "processed"
        sample_dir = tmp_path / "sample"
        _write_series(processed_dir)
        _write_labels(sample_dir)
        settings = _settings(processed_dir, sample_dir, mlflow_uri)

        smoothed = np.zeros(17, dtype=np.float64)
        smoothed[14] = 5.0
        threshold = np.ones(17, dtype=np.float64)
        runs_dir = tmp_path / "offline_runs"
        for run_id in ("baseline-run", "tuned-run"):
            run_dir = runs_dir / run_id
            run_dir.mkdir(parents=True)
            (run_dir / "errors.npy").write_bytes(errors_to_bytes(smoothed))
            (run_dir / "threshold.npy").write_bytes(threshold_to_bytes(threshold))

        run_map = OfflineRunMap(
            mission=_MISSION,
            baseline={
                _CHANNEL: RunSpec(
                    run_id="baseline-run",
                    errors_path=str(runs_dir / "baseline-run" / "errors.npy"),
                    threshold_path=str(runs_dir / "baseline-run" / "threshold.npy"),
                    threshold_min_anomaly_len=1,
                )
            },
            tuned={
                _CHANNEL: RunSpec(
                    run_id="tuned-run",
                    errors_path=str(runs_dir / "tuned-run" / "errors.npy"),
                    threshold_path=str(runs_dir / "tuned-run" / "threshold.npy"),
                    threshold_min_anomaly_len=1,
                )
            },
        )

        report = build_report(settings, _MISSION, channels=[_CHANNEL], run_map=run_map)

        tuned_rows = [r for r in report["rows"] if r["label"] == "ours (tuned)"]
        assert tuned_rows
        for row in tuned_rows:
            assert "provenance unavailable in offline mode" in row["params"]
        assert not any("Ray Tune" in note for note in report["footnotes"])
        assert not any("threshold_ceiling.py" in note for note in report["footnotes"])
        assert any(
            "provenance unavailable in offline mode" in note for note in report["footnotes"]
        )


class TestTunedProvenanceMultiChannel:
    """docs/reviews/022 stage 1.3 (T4): tuned_provenance's uniformity check
    was covered only by single-channel fixtures — a two-channel arm that
    genuinely agrees, or disagrees, on provenance was never exercised."""

    def test_two_channels_agreeing_on_provenance_renders_once(
        self, tmp_path: Path, mlflow_uri: str
    ) -> None:
        processed_dir = tmp_path / "processed"
        sample_dir = tmp_path / "sample"
        _write_series(processed_dir, channel=_CHANNEL)
        _write_series(processed_dir, channel=_CHANNEL_2)
        _write_labels(sample_dir)
        settings = _settings(processed_dir, sample_dir, mlflow_uri)

        for channel in (_CHANNEL, _CHANNEL_2):
            _log_scoring_run(settings, tuned=False, channel=channel)
            _log_scoring_run(
                settings,
                tuned=True,
                tuned_source="scripts/threshold_ceiling.py exhaustive grid (mission)",
                channel=channel,
            )

        report = build_report(settings, _MISSION, channels=[_CHANNEL, _CHANNEL_2])

        tuned_rows = [r for r in report["rows"] if r["label"] == "ours (tuned)"]
        assert tuned_rows
        for row in tuned_rows:
            assert row["params"] == "scripts/threshold_ceiling.py exhaustive grid (mission)"

    def test_two_channels_disagreeing_on_provenance_raises(
        self, tmp_path: Path, mlflow_uri: str
    ) -> None:
        """Today's behaviour (C3): a mixed-config arm raises and the WHOLE
        report fails to build, even though the untuned rows and one channel's
        tuned data were perfectly renderable. docs/reviews/022 stage 2.2
        changes this to a rendered "mixed: ..." summary instead — flip this
        test in that same commit.
        """
        processed_dir = tmp_path / "processed"
        sample_dir = tmp_path / "sample"
        _write_series(processed_dir, channel=_CHANNEL)
        _write_series(processed_dir, channel=_CHANNEL_2)
        _write_labels(sample_dir)
        settings = _settings(processed_dir, sample_dir, mlflow_uri)

        _log_scoring_run(settings, tuned=False, channel=_CHANNEL)
        _log_scoring_run(
            settings,
            tuned=True,
            tuned_source="scripts/threshold_ceiling.py exhaustive grid (mission)",
            channel=_CHANNEL,
        )
        _log_scoring_run(settings, tuned=False, channel=_CHANNEL_2)
        _log_scoring_run(settings, tuned=True, channel=_CHANNEL_2)  # tuned_from_run only

        with pytest.raises(RuntimeError, match="disagree on tuned-scoring provenance"):
            build_report(settings, _MISSION, channels=[_CHANNEL, _CHANNEL_2])


class TestScopeInvariants:
    """Two invariants that hold on every row of both the arm-A and production
    reports (see docs/reviews/019-esa-adb-comparable-eval.md's correction to
    the original review). A third, `anomalies_only recall >= all_events
    recall`, was proposed by the review and is FALSE — production data
    disproved it (all_events recall 0.417 vs anomalies_only 0.286). Excluding
    Rare Event changes both numerator and denominator, in either direction.
    Do not add that one here.
    """

    def test_n_detections_identical_across_scopes_for_same_label(
        self, tmp_path: Path, mlflow_uri: str
    ) -> None:
        """Detections are computed once and only scoped at metric time."""
        processed_dir = tmp_path / "processed"
        sample_dir = tmp_path / "sample"
        _write_series(processed_dir)
        _write_two_category_labels(sample_dir)
        settings = _settings(processed_dir, sample_dir, mlflow_uri)

        _log_scoring_run(settings, tuned=False)
        _log_scoring_run(settings, tuned=True)

        report = build_report(settings, _MISSION, channels=[_CHANNEL])
        by_label: dict[str, set[int]] = {}
        for row in report["rows"]:
            if row["n_detections"] is None:  # paper reference rows carry no detections
                continue
            by_label.setdefault(row["label"], set()).add(row["n_detections"])

        assert by_label, "fixture should produce at least one non-paper row"
        for label, values in by_label.items():
            assert len(values) == 1, f"{label}: n_detections differs across scopes: {values}"

    def test_anomalies_only_n_events_never_exceeds_all_events(
        self, tmp_path: Path, mlflow_uri: str
    ) -> None:
        """Excluding Rare Event only removes events — it never adds them."""
        processed_dir = tmp_path / "processed"
        sample_dir = tmp_path / "sample"
        _write_series(processed_dir)
        _write_two_category_labels(sample_dir)
        settings = _settings(processed_dir, sample_dir, mlflow_uri)

        _log_scoring_run(settings, tuned=False)
        _log_scoring_run(settings, tuned=True)

        report = build_report(settings, _MISSION, channels=[_CHANNEL])
        by_label_scope: dict[tuple[str, str], int] = {}
        for row in report["rows"]:
            if row["n_events"] is None:  # paper reference rows carry no n_events
                continue
            by_label_scope[(row["label"], row["scope"])] = row["n_events"]

        labels = {label for label, _scope in by_label_scope}
        assert labels, "fixture should produce at least one non-paper row"
        for label in labels:
            all_events_n = by_label_scope[(label, "all_events")]
            anomalies_only_n = by_label_scope[(label, "anomalies_only")]
            assert anomalies_only_n <= all_events_n, (
                f"{label}: anomalies_only n_events ({anomalies_only_n}) exceeds "
                f"all_events ({all_events_n})"
            )
        # Sanity: the fixture's Rare Event event must actually exercise the
        # inequality on at least one row, or this test would pass vacuously.
        assert any(
            by_label_scope[(label, "anomalies_only")] < by_label_scope[(label, "all_events")]
            for label in labels
        ), "fixture must produce a strict inequality on at least one row"
