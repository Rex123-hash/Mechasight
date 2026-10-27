"""Phase-3.1 classical baselines and pre-test safety boundary."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.experiment.config import ExperimentConfig
from src.experiment.data import (
    SourceValidationCohortContract,
    TrainingCohortContract,
    load_estimator_training_data,
    load_source_validation_data,
)
from src.experiment.phase31 import (
    aggregate_windows_by_segment,
    assert_phase31_role,
    phase31_feature_columns,
    seed_plan,
)
from src.experiment.pretest import PretestResultStore
from src.experiment.provenance import ordered_id_hash
from src.experiment.scores import ThresholdSelection
from src.models.classical import ClassicalAnomalyBaseline, primary_spec
from src.preprocessing.config import load_config


ASSIGNMENTS = Path("reports/phase2_2/protocol_b_few_shot_adaptation_v3.csv")
CONFIG_S = Path("configs/preprocessing/mechasight_few_shot_regime_s.json")


def _cloud(seed: int = 8) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.25, size=(80, 4)), np.full((5, 4), 8.0)


@pytest.mark.parametrize("family", ["statistical_reference", "isolation_forest", "one_class_svm"])
def test_all_baselines_return_finite_higher_is_more_anomalous_scores(family):
    train, far = _cloud()
    model = ClassicalAnomalyBaseline(primary_spec(family), seed=0).fit(train)
    central = model.score(np.zeros((5, 4)))
    distant = model.score(far)
    assert np.isfinite(central).all() and np.isfinite(distant).all()
    assert float(np.median(distant)) > float(np.median(central))


def test_isolation_forest_is_reproducible_per_seed_and_seed_plan_is_exact():
    train, far = _cloud()
    first = ClassicalAnomalyBaseline(primary_spec("isolation_forest"), seed=3).fit(train)
    second = ClassicalAnomalyBaseline(primary_spec("isolation_forest"), seed=3).fit(train)
    np.testing.assert_array_equal(first.score(far), second.score(far))
    assert seed_plan("isolation_forest") == (0, 1, 2, 3, 4)
    assert seed_plan("one_class_svm") == (0,)
    assert seed_plan("statistical_reference") == (0,)


def test_model_rejects_bad_seed_empty_nonfinite_and_schema_mismatch():
    with pytest.raises(ValueError, match="seed"):
        ClassicalAnomalyBaseline(primary_spec("isolation_forest"), seed=5)
    model = ClassicalAnomalyBaseline(primary_spec("statistical_reference"), seed=0)
    with pytest.raises(ValueError, match="non-empty"):
        model.fit(np.empty((0, 2)))
    with pytest.raises(ValueError, match="finite"):
        model.fit(np.array([[1.0, np.nan]]))
    model.fit(np.ones((2, 2)))
    with pytest.raises(ValueError, match="schema"):
        model.score(np.ones((2, 3)))


@pytest.mark.parametrize("family", ["statistical_reference", "isolation_forest", "one_class_svm"])
def test_fitted_model_serialization_round_trip(tmp_path, family):
    train, far = _cloud()
    model = ClassicalAnomalyBaseline(primary_spec(family), seed=0).fit(train)
    path = model.save(tmp_path / f"{family}.joblib")
    restored = ClassicalAnomalyBaseline.load(path)
    np.testing.assert_allclose(model.score(far), restored.score(far), rtol=0, atol=0)
    assert restored.model_id == model.model_id


def _frame_from_assignments(machine: str, regime: str, role: str) -> tuple[pd.DataFrame, object]:
    assignments = pd.read_csv(ASSIGNMENTS)
    config = load_config(CONFIG_S)
    assert regime == "S"
    if role == "training":
        contract = TrainingCohortContract.from_frozen_assignments(
            assignments, config, machine=machine
        )
        part = assignments[
            assignments["machine"].eq(machine)
            & assignments["regime"].eq(regime)
            & assignments["split"].eq("train")
            & assignments["data_pool"].eq("source_training")
        ].copy()
    else:
        contract = SourceValidationCohortContract.from_frozen_assignments(
            assignments, config, machine=machine
        )
        part = assignments[
            assignments["machine"].eq(machine)
            & assignments["regime"].eq(regime)
            & assignments["split"].eq("validation")
            & assignments["data_pool"].eq("source_validation")
        ].copy()
    part["preprocessing_config_id"] = config.config_id
    part["domain"] = part["file_domain"]
    part["sensor__feature"] = np.arange(len(part), dtype=float)
    return part, contract


def test_complete_frozen_training_and_validation_contracts_load_label_safe_matrices():
    train, train_contract = _frame_from_assignments("RoboticArm", "S", "training")
    validation, validation_contract = _frame_from_assignments(
        "RoboticArm", "S", "source_validation"
    )
    loaded_train = load_estimator_training_data(
        train, train_contract, feature_columns=("sensor__feature",)
    )
    loaded_validation = load_source_validation_data(
        validation, validation_contract, feature_columns=("sensor__feature",)
    )
    assert loaded_train.X.shape == (1528, 1)
    assert loaded_validation.X.shape == (186, 1)
    assert len(set(loaded_validation.parent_recordings)) == 6


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("split", "test", "validation rows only"),
        ("data_pool", "common_evaluation", "source_validation role only"),
        ("domain", "target", "source-domain only"),
        ("anomaly_label", "anomaly", "normal-only"),
    ],
)
def test_source_validation_loader_rejects_test_target_anomaly_or_wrong_role(column, value, message):
    frame, contract = _frame_from_assignments("BrushlessMotor", "S", "source_validation")
    frame[column] = value
    with pytest.raises(ValueError, match=message):
        load_source_validation_data(frame, contract, feature_columns=("sensor__feature",))


def test_source_validation_loader_requires_the_complete_frozen_cohort():
    frame, contract = _frame_from_assignments("BrushlessMotor", "S", "source_validation")
    with pytest.raises(ValueError, match="complete frozen cohort"):
        load_source_validation_data(
            frame.iloc[:-1], contract, feature_columns=("sensor__feature",)
        )


def test_role_sentinel_rejects_test_before_touching_the_window_iterator():
    touched = False

    def forbidden_iterator():
        nonlocal touched
        touched = True
        yield ({}, {}, {})

    with pytest.raises(ValueError, match="before materialisation/scoring"):
        aggregate_windows_by_segment(forbidden_iterator(), role="common_evaluation")
    assert touched is False
    with pytest.raises(ValueError):
        assert_phase31_role("test")


def test_segment_aggregation_uses_median_and_preserves_provenance():
    base = {
        "machine": "RoboticArm", "protocol": "mechasight_experimental_protocol",
        "regime": "S", "split": "validation", "data_pool": "source_validation",
        "domain": "source", "parent_recording": "p1", "segment_id": "s1",
        "anomaly_label": "normal", "preprocessing_config_id": "c1",
        "preprocessing_version": "phase2.2-v3",
        "split_version": "protocol_b_few_shot_adaptation_v3",
    }
    windows = [
        (base, {}, {"mic__rms": 1.0, "acc__rms": 4.0}),
        (base, {}, {"mic__rms": 9.0, "acc__rms": 2.0}),
        (base, {}, {"mic__rms": 3.0, "acc__rms": 3.0}),
    ]
    frame = aggregate_windows_by_segment(windows, role="source_validation")
    assert frame.loc[0, "mic__rms"] == 3.0
    assert frame.loc[0, "acc__rms"] == 3.0
    assert frame.loc[0, "window_count"] == 3
    assert phase31_feature_columns(frame) == ("acc__rms", "mic__rms")


def _experiment() -> ExperimentConfig:
    return ExperimentConfig(
        timestamp_utc="2026-09-03T00:00:00+00:00",
        machine="RoboticArm", protocol="mechasight_experimental_protocol", regime="S",
        model_family="statistical_reference", model_hyperparameters={"distance": "rms"},
        model_seed=0, preprocessing_config_id="cfg", preprocessing_config_hash="hash",
        split_version="protocol_b_few_shot_adaptation_v3", split_manifest_hash="manifest",
        dataset_release_identity="synthetic", feature_representation="segment_median",
        feature_version="v1", window_length_s=0.1, stride_s=0.1,
        normalization_strategy="training_zscore", code_git_sha="abc",
        dependency_environment={"python": "test"},
    )


def test_pretest_store_is_validation_only_immutable_and_contains_no_test_metrics(tmp_path):
    config = _experiment()
    scores = pd.DataFrame({
        "segment_id": ["s1", "s2"], "parent_recording": ["p1", "p2"],
        "cohort_role": ["source_validation"] * 2, "domain": ["source"] * 2,
        "anomaly_label": ["normal"] * 2, "score": [0.1, 0.2],
    })
    cohort_hash = ordered_id_hash(("s1", "s2"))
    threshold = ThresholdSelection(
        value=0.2, policy_version="q90", quantile=0.9, quantile_method="higher",
        score_direction="higher_is_more_anomalous", source_experiment_id=config.experiment_id,
        source_cohort_hash=cohort_hash, source_segment_ids=("s1", "s2"),
        source_role="source_validation",
    )
    kwargs = dict(
        config=config,
        provenance={
            "evaluation_stage": "validation_only_pretest", "test_evaluated": False,
            "score_direction": "higher_is_more_anomalous", "training_cohort_hash": "train",
            "validation_cohort_hash": cohort_hash,
        },
        summary={"score_median": 0.15}, threshold=threshold, validation_scores=scores,
    )
    bundle = PretestResultStore(tmp_path).write_bundle(**kwargs)
    persisted = json.loads((bundle / "validation_summary.json").read_text())
    assert "auroc" not in persisted and "test_scores" not in persisted
    with pytest.raises(FileExistsError):
        PretestResultStore(tmp_path).write_bundle(**kwargs)
    with pytest.raises(ValueError, match="performance metrics"):
        PretestResultStore(tmp_path / "bad").write_bundle(
            **{**kwargs, "summary": {"auroc": 0.99}}
        )


def test_phase31_primary_configs_are_frozen_and_do_not_authorize_test_or_later_models():
    preprocessing = json.loads(
        Path("configs/phase3/preprocessing_primary_v1.json").read_text()
    )
    ladder = json.loads(Path("configs/phase3/baseline_ladder_v1.json").read_text())
    assert preprocessing["status"] == "PRIMARY_PRE_REGISTERED_CONFIGURATION"
    assert preprocessing["window_length_s"] == preprocessing["stride_s"] == 0.1
    assert preprocessing["model_input_normalization"]["test_statistics_used"] is False
    assert [entry["family"] for entry in ladder["entries"][:3]] == [
        "statistical_reference", "isolation_forest", "one_class_svm"
    ]
    assert all(entry["status"] == "NOT_AUTHORIZED_IN_PHASE3_1" for entry in ladder["entries"][3:])


def test_scientific_model_change_changes_experiment_identity():
    config = _experiment()
    changed = replace(config, model_hyperparameters={"distance": "different"})
    assert changed.experiment_id != config.experiment_id
