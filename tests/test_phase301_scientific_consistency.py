"""Phase 3.0.1 scientific-consistency regression tests."""

from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

import src.experiment.bootstrap as bootstrap_module
from src.experiment.bootstrap import (
    bootstrap_metric,
    bootstrap_paired_difference,
    draw_class_stratified_cluster_ids,
    draw_stratified_cluster_indices,
)
from src.experiment.config import BootstrapConfig, ExperimentConfig
from src.experiment.constants import (
    BOOTSTRAP_INFERENCE_SCOPE,
    BOOTSTRAP_POLICY_VERSION,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    BRUSHLESS_MOTOR_ACQUISITION_SESSION_CONFOUND_ID,
    PRIMARY_BOOTSTRAP_STRATA,
    PRIMARY_REPORTING_SLICES,
    ROBOTIC_ARM_ADAPTATION_PARENT_IDS,
    ROBOTIC_ARM_ADAPTATION_PARENT_LIMITATION_ID,
    ROBOTIC_ARM_VALIDATION_PARENT_IDS,
    ROBOTIC_ARM_VALIDATION_PARENT_LIMITATION_ID,
)


V3_PATH = "reports/phase2_2/protocol_b_few_shot_adaptation_v3.csv"


@pytest.fixture(scope="module")
def ra_v3() -> pd.DataFrame:
    frame = pd.read_csv(V3_PATH, dtype=str)
    return frame[(frame.machine == "RoboticArm") & (frame.regime == "T")]


def test_ra_validation_and_adaptation_parent_cohorts_are_exact_and_not_interchangeable(ra_v3):
    validation = ra_v3[
        (ra_v3.data_pool == "source_validation") & (ra_v3["split"] == "validation")
    ]
    adaptation = ra_v3[
        (ra_v3.data_pool == "target_adaptation") & (ra_v3["split"] == "train")
    ]
    validation_parents = tuple(sorted(validation.parent_recording.unique()))
    adaptation_parents = tuple(sorted(adaptation.parent_recording.unique()))

    assert (len(validation), len(validation_parents)) == (186, 6)
    assert (len(adaptation), len(adaptation_parents)) == (27, 18)
    assert validation_parents == ROBOTIC_ARM_VALIDATION_PARENT_IDS
    assert adaptation_parents == ROBOTIC_ARM_ADAPTATION_PARENT_IDS
    assert set(validation_parents).isdisjoint(adaptation_parents)
    assert len(validation_parents) != len(adaptation_parents)


def test_ra_blocked_siblings_are_the_exact_adaptation_parent_closure(ra_v3):
    adaptation = ra_v3[ra_v3.data_pool == "target_adaptation"]
    blocked = ra_v3[ra_v3.data_pool == "adaptation_sibling_blocked"]
    adaptation_parents = set(adaptation.parent_recording)
    blocked_parents = set(blocked.parent_recording)

    assert len(adaptation) == 27
    assert len(blocked) == 96
    assert len(pd.concat([adaptation, blocked])) == 123
    assert adaptation_parents == blocked_parents == set(ROBOTIC_ARM_ADAPTATION_PARENT_IDS)
    assert set(adaptation.segment_id).isdisjoint(blocked.segment_id)


def _config(machine: str = "RoboticArm") -> ExperimentConfig:
    return ExperimentConfig(
        timestamp_utc="2026-09-01T00:00:00+00:00",
        machine=machine,
        protocol="mechasight_experimental_protocol",
        regime="T",
        model_family="synthetic",
        model_hyperparameters={},
        model_seed=0,
        preprocessing_config_id="synthetic",
        preprocessing_config_hash="hash",
        split_version="protocol_b_few_shot_adaptation_v3",
        split_manifest_hash="manifest-hash",
        dataset_release_identity="synthetic",
        feature_representation="synthetic",
        feature_version="synthetic",
        window_length_s=0.1,
        stride_s=0.1,
        normalization_strategy="none",
        code_git_sha="abc",
        dependency_environment={"python": "test"},
    )


def test_machine_readable_limitations_distinguish_ra_cohorts_and_freeze_bm_confound():
    ra = _config()
    assert ra.scientific_limitation_ids == (
        ROBOTIC_ARM_VALIDATION_PARENT_LIMITATION_ID,
        ROBOTIC_ARM_ADAPTATION_PARENT_LIMITATION_ID,
    )
    assert ra.cohort_provenance["source_validation"]["parent_count"] == 6
    assert ra.cohort_provenance["target_adaptation"]["parent_count"] == 18

    bm = _config("BrushlessMotor")
    assert bm.scientific_limitation_ids == (
        BRUSHLESS_MOTOR_ACQUISITION_SESSION_CONFOUND_ID,
    )
    assert "mechanical-fault effects cannot be completely separated" in bm.scientific_limitations[0]
    assert "cannot deconfound fault from session" in bm.scientific_limitations[0]


def test_phase3_scaffolds_carry_corrected_machine_readable_policy():
    with open("configs/phase3/baseline_ladder_v1.json", encoding="utf-8") as handle:
        scaffold = json.load(handle)
    primary = scaffold["primary_bootstrap"]
    assert primary["policy_version"] == BOOTSTRAP_POLICY_VERSION
    assert primary["strata"] == ["is_anomaly"]
    assert primary["automatic_operating_condition_strata"] is False
    assert scaffold["scientific_limitation_ids"]["BrushlessMotor"] == [
        BRUSHLESS_MOTOR_ACQUISITION_SESSION_CONFOUND_ID
    ]


def _ra_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "segment_id": ["n1a", "n1b", "n2", "a1a", "a1b", "a2"],
        "parent_recording": ["np1", "np1", "np2", "ap1", "ap1", "ap2"],
        "machine": ["RoboticArm"] * 6,
        "domain": ["target"] * 6,
        "is_anomaly": [0, 0, 0, 1, 1, 1],
        "operating_condition": ["c1", "c2", "c1", "c3", "c4", "c3"],
        "score": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
        "score_a": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
        "score_b": [0.9, 0.8, 0.7, 0.3, 0.2, 0.1],
    })


def test_primary_class_strata_sample_clusters_and_preserve_cluster_counts():
    frame = _ra_frame()
    draws = draw_class_stratified_cluster_ids(frame, rng=np.random.default_rng(1995))
    expected = frame[["parent_recording", "is_anomaly"]].drop_duplicates().groupby(
        "is_anomaly"
    ).size().to_dict()
    assert {label: len(ids) for label, ids in draws.items()} == expected


class _FirstParentRng:
    def choice(self, ids, *, size, replace):
        assert replace is True
        return np.repeat(ids[0], size)


def test_duplicate_parent_draw_duplicates_its_complete_ra_sibling_set():
    frame = _ra_frame()
    indices = draw_stratified_cluster_indices(frame, rng=_FirstParentRng())
    sampled = frame.loc[indices]
    assert sampled.parent_recording.value_counts().to_dict() == {"np1": 4, "ap1": 4}
    assert sampled.segment_id.value_counts().to_dict() == {
        "n1a": 2, "n1b": 2, "a1a": 2, "a1b": 2,
    }


def test_operating_condition_does_not_create_hidden_primary_strata():
    frame = _ra_frame()
    changed = frame.copy()
    changed["operating_condition"] = [f"unique-{i}" for i in range(len(changed))]
    first = draw_class_stratified_cluster_ids(frame, rng=np.random.default_rng(1995))
    second = draw_class_stratified_cluster_ids(changed, rng=np.random.default_rng(1995))
    assert first == second


def test_domain_and_machine_slicing_is_required_before_primary_bootstrap():
    mixed_domain = _ra_frame()
    mixed_domain.loc[0, "domain"] = "source"
    with pytest.raises(ValueError, match="sliced by domain"):
        bootstrap_metric(mixed_domain, metric="auroc", replicates=2)

    mixed_machine = _ra_frame()
    mixed_machine.loc[0, "machine"] = "BrushlessMotor"
    with pytest.raises(ValueError, match="sliced by machine"):
        bootstrap_metric(mixed_machine, metric="auroc", replicates=2)


def test_paired_comparison_draws_clusters_once_per_replicate(monkeypatch):
    original = bootstrap_module.draw_stratified_cluster_indices
    draws: list[np.ndarray] = []

    def recording_draw(*args, **kwargs):
        result = original(*args, **kwargs)
        draws.append(result.copy())
        return result

    monkeypatch.setattr(bootstrap_module, "draw_stratified_cluster_indices", recording_draw)
    result = bootstrap_paired_difference(_ra_frame(), metric="auroc", replicates=7, seed=1995)
    assert len(draws) == 7
    assert result.number_valid == 7
    assert result.resampling_unit == "physical_parent_recording"


def test_primary_bootstrap_defaults_and_conditional_interpretation_are_frozen():
    config = BootstrapConfig()
    assert config.replicates == BOOTSTRAP_REPLICATES == 10_000
    assert config.seed == BOOTSTRAP_SEED == 1995
    assert config.strata == PRIMARY_BOOTSTRAP_STRATA == ("is_anomaly",)
    assert config.reporting_slices == PRIMARY_REPORTING_SLICES == ("machine", "domain")
    assert config.inference_scope == BOOTSTRAP_INFERENCE_SCOPE
    with pytest.raises(ValueError, match="primary bootstrap design is frozen"):
        replace(config, strata=("domain", "is_anomaly", "operating_condition"))


def test_bootstrap_result_never_claims_independent_segments_or_population_wide_uncertainty():
    first = bootstrap_metric(_ra_frame(), metric="auroc", replicates=20, seed=1995)
    second = bootstrap_metric(_ra_frame(), metric="auroc", replicates=20, seed=1995)
    assert first == second
    assert first.design == BOOTSTRAP_POLICY_VERSION
    assert first.inference_scope == "conditional_on_observed_evaluation_cohort"
    assert first.resampling_unit == "physical_parent_recording"
    assert first.reporting_slice == (("machine", "RoboticArm"), ("domain", "target"))


def test_bm_primary_bootstrap_enforces_audited_one_segment_per_recording_structure():
    bm = _ra_frame().assign(machine="BrushlessMotor")
    with pytest.raises(ValueError, match="one-segment-per-recording"):
        bootstrap_metric(bm, metric="auroc", replicates=2)

    one_to_one = bm.assign(parent_recording=[f"bm-{i}" for i in range(len(bm))])
    result = bootstrap_metric(one_to_one, metric="auroc", replicates=2)
    assert result.resampling_unit == "released_recording_segment_one_to_one"
