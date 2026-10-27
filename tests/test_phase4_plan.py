"""Single-Modality Benchmarking: frozen experiment plan, seeds and inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.experiment.phase4 import (
    EXPECTED_EXPERIMENT_COUNT,
    FAMILIES,
    LEDGER_EXPECTED_START,
    MODALITIES,
    STOCHASTIC_FAMILIES,
    aggregate_window_scores_by_segment,
    expected_seeds,
    experiment_plan,
    modality_feature_columns,
)


def test_family_set_is_exactly_the_five_approved_families():
    assert FAMILIES == (
        "statistical_reference", "isolation_forest", "one_class_svm",
        "mahalanobis_shrinkage", "dense_autoencoder",
    )
    assert MODALITIES == ("mic", "acc", "gyro")


def test_seed_registry():
    assert expected_seeds("isolation_forest") == (0, 1, 2, 3, 4)
    assert expected_seeds("dense_autoencoder") == (0, 1, 2, 3, 4)
    for family in ("statistical_reference", "one_class_svm", "mahalanobis_shrinkage"):
        assert expected_seeds(family) == (0,)
    assert STOCHASTIC_FAMILIES == frozenset({"isolation_forest", "dense_autoencoder"})
    with pytest.raises(ValueError, match="family"):
        expected_seeds("xgboost")


def test_experiment_plan_count_is_156_and_unique():
    plan = experiment_plan()
    assert len(plan) == EXPECTED_EXPERIMENT_COUNT == 156
    # 2 machines x 2 regimes x 3 modalities x (1 + 5 + 1 + 1 + 5) seeds
    assert len(set(plan)) == 156
    per_cell = 1 + 5 + 1 + 1 + 5
    assert 2 * 2 * 3 * per_cell == 156
    assert {item.machine for item in plan} == {"RoboticArm", "BrushlessMotor"}
    assert {item.regime for item in plan} == {"S", "T"}
    assert {item.modality for item in plan} == set(MODALITIES)
    assert LEDGER_EXPECTED_START == 40


def test_modality_feature_columns_select_only_that_sensor_prefix():
    columns = (
        "acc__x__rms", "acc__y__rms", "gyro__x__rms", "mic__mic__rms",
        "mic__mic__kurtosis_excess", "segment_id", "anomaly_label",
    )
    assert modality_feature_columns(columns, "mic") == (
        "mic__mic__kurtosis_excess", "mic__mic__rms",
    )
    assert modality_feature_columns(columns, "acc") == ("acc__x__rms", "acc__y__rms")
    assert modality_feature_columns(columns, "gyro") == ("gyro__x__rms",)
    with pytest.raises(ValueError, match="modality"):
        modality_feature_columns(columns, "all")


def test_modality_feature_columns_match_frozen_counts_on_real_artifact():
    frame = pd.read_parquet("data/interim/phase31_primary/RoboticArm/S/training.parquet")
    columns = tuple(frame.columns)
    assert len(modality_feature_columns(columns, "mic")) == 14
    assert len(modality_feature_columns(columns, "acc")) == 42
    assert len(modality_feature_columns(columns, "gyro")) == 42


def test_segment_aggregation_is_median_of_window_scores():
    meta = pd.DataFrame({
        "segment_id": ["b", "a", "a", "b", "a"],
        "parent_recording": ["pb", "pa", "pa", "pb", "pa"],
        "domain": ["source", "target", "target", "source", "target"],
        "anomaly_label": ["normal"] * 5,
    })
    scores = np.array([1.0, 5.0, 1.0, 3.0, 9.0])
    table = aggregate_window_scores_by_segment(meta, scores)
    assert list(table["segment_id"]) == ["a", "b"]
    assert list(table["score"]) == [5.0, 2.0]
    assert list(table["window_count"]) == [3, 2]
    assert list(table["parent_recording"]) == ["pa", "pb"]
    assert list(table["domain"]) == ["target", "source"]


def test_segment_aggregation_refuses_inconsistent_segment_provenance():
    meta = pd.DataFrame({
        "segment_id": ["a", "a"],
        "parent_recording": ["p1", "p2"],
        "domain": ["source", "source"],
        "anomaly_label": ["normal", "normal"],
    })
    with pytest.raises(ValueError, match="provenance"):
        aggregate_window_scores_by_segment(meta, np.array([1.0, 2.0]))
