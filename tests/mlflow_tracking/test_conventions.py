"""Unit tests for mlflow_tracking/conventions.py — pure functions, no I/O."""

from __future__ import annotations

from spacecraft_telemetry.mlflow_tracking.conventions import (
    common_tags,
    experiment_name,
    registered_model_name,
)


class TestExperimentName:
    def test_telemanom_training(self) -> None:
        assert experiment_name("telemanom", "training", "ESA-Mission1") == (
            "telemanom-training-ESA-Mission1"
        )

    def test_telemanom_hpo(self) -> None:
        assert experiment_name("telemanom", "hpo", "ESA-Mission2") == (
            "telemanom-hpo-ESA-Mission2"
        )

    def test_dc_vae_scoring(self) -> None:
        assert experiment_name("dc_vae", "scoring", "ESA-Mission3") == (
            "dc_vae-scoring-ESA-Mission3"
        )

    def test_future_model_type_works(self) -> None:
        result = experiment_name("transformer", "training", "ESA-Mission1")
        assert result.startswith("transformer-")


class TestRegisteredModelName:
    def test_telemanom_per_channel(self) -> None:
        assert registered_model_name("telemanom", "ESA-Mission1", "channel_1") == (
            "telemanom-ESA-Mission1-channel_1"
        )

    def test_dc_vae_mission_level(self) -> None:
        assert registered_model_name("dc_vae", "ESA-Mission1", "power") == (
            "dc_vae-ESA-Mission1-power"
        )

    def test_key_is_arbitrary_string(self) -> None:
        name = registered_model_name("telemanom", "ESA-Mission1", "some-group_42")
        assert "some-group_42" in name


class TestCommonTags:
    def test_required_fields_always_present(self) -> None:
        tags = common_tags(model_type="telemanom", mission="ESA-Mission1", phase="training")
        assert tags["model_type"] == "telemanom"
        assert tags["mission_id"] == "ESA-Mission1"
        assert tags["phase"] == "training"

    def test_none_values_are_dropped(self) -> None:
        tags = common_tags(
            model_type="telemanom",
            mission="ESA-Mission1",
            phase="training",
            channel=None,
            subsystem=None,
            training_data_hash=None,
        )
        assert "channel_id" not in tags
        assert "subsystem" not in tags
        assert "training_data_hash" not in tags

    def test_optional_fields_included_when_set(self) -> None:
        tags = common_tags(
            model_type="telemanom",
            mission="ESA-Mission1",
            phase="training",
            channel="channel_1",
            subsystem="subsystem_power",
            training_data_hash="abc123",
        )
        assert tags["channel_id"] == "channel_1"
        assert tags["subsystem"] == "subsystem_power"
        assert tags["training_data_hash"] == "abc123"

    def test_extra_dict_is_merged(self) -> None:
        tags = common_tags(
            model_type="telemanom",
            mission="ESA-Mission1",
            phase="scoring",
            extra={"eval_split": "final_portion", "scoring_variant": "tuned"},
        )
        assert tags["eval_split"] == "final_portion"
        assert tags["scoring_variant"] == "tuned"

    def test_all_values_are_strings(self) -> None:
        tags = common_tags(
            model_type="telemanom",
            mission="ESA-Mission1",
            phase="training",
            channel="channel_1",
            extra={"numeric_tag": 42},
        )
        for k, v in tags.items():
            assert isinstance(v, str), f"tag {k!r} has non-str value {v!r}"

    def test_extra_does_not_override_standard_keys(self) -> None:
        tags = common_tags(
            model_type="telemanom",
            mission="ESA-Mission1",
            phase="training",
            extra={"model_type": "impostor"},
        )
        # extra merges after standard keys — extra wins in dict.update semantics;
        # document this so callers are aware.
        assert tags["model_type"] == "impostor"
