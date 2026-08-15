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


def _write_series(processed_dir: Path) -> None:
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
        processed_dir / _MISSION / "test" / f"mission_id={_MISSION}" / f"channel_id={_CHANNEL}"
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


def _log_scoring_run(settings: Settings, *, tuned: bool) -> None:
    """Flag window index 14 alone — well after the hpo_eval_fraction=0.6 cutoff (idx 10)."""
    smoothed = np.zeros(17, dtype=np.float64)
    smoothed[14] = 5.0
    threshold = np.ones(17, dtype=np.float64)

    exp = experiment_name(settings.model.model_type, "scoring", _MISSION)
    extra = {"eval_split": "final_portion" if tuned else "full_test"}
    if tuned:
        extra["tuned_from_run"] = "fake-hpo-run"
    tags = common_tags(
        model_type=settings.model.model_type,
        mission=_MISSION,
        phase="scoring",
        channel=_CHANNEL,
        extra=extra,
    )
    with open_run(experiment=exp, run_name=_CHANNEL, tags=tags) as run:
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
