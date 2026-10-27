from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.evaluation.policy import (
    PairedExperimentSpec,
    aggregate_segment_scores,
    assert_paired_experiment_specs,
    assert_preprocessing_pair,
    normal_quantile_threshold,
    draw_parent_cluster_sample,
)
from src.preprocessing.config import load_config
from src.preprocessing.pipeline import merge_protocol_b_assignments
from src.preprocessing.splits import (
    assert_matched_regime_cohorts,
    attach_parent_recording,
    load_audit_catalog,
    matched_common_test_assignments,
)


SPLIT_PATH = "reports/phase2_1/protocol_b_common_test_v2.csv"


def _assignments() -> pd.DataFrame:
    return pd.read_csv(SPLIT_PATH)


def test_real_s_t_test_segment_and_parent_ids_are_identical():
    assignments = _assignments()
    assert_matched_regime_cohorts(assignments)
    for _, part in assignments.groupby("machine"):
        s = part[(part.regime == "S") & (part.split == "test")]
        t = part[(part.regime == "T") & (part.split == "test")]
        assert set(s.segment_id) == set(t.segment_id)
        assert set(s.parent_recording) == set(t.parent_recording)


def test_real_adaptation_pool_expansion_is_frozen():
    assignments = _assignments()
    unique = assignments[assignments.regime == "T"]
    counts = unique[unique.data_pool == "target_adaptation"].groupby("machine").agg(
        parents=("parent_recording", "nunique"), segments=("segment_id", "size")
    )
    assert tuple(counts.loc["RoboticArm"]) == (18, 123)
    assert tuple(counts.loc["BrushlessMotor"]) == (18, 18)


def test_split_rebuild_is_deterministic():
    catalog = attach_parent_recording(
        load_audit_catalog("reports/dataset_audit/segment_manifest.csv")
    )
    rebuilt = pd.concat([
        matched_common_test_assignments(catalog, seed=1995, regime=regime)
        for regime in ("S", "T")
    ])
    frozen = _assignments()
    key = ["machine", "regime", "segment_id"]
    left = rebuilt.set_index(key)["split"].sort_index()
    right = frozen.set_index(key)["split"].sort_index()
    pd.testing.assert_series_equal(left, right, check_names=False, check_dtype=False)


def test_common_test_preprocessing_configs_differ_only_by_regime():
    s = load_config("configs/preprocessing/mechasight_common_test_regime_s.json")
    t = load_config("configs/preprocessing/mechasight_common_test_regime_t.json")
    assert_preprocessing_pair(s, t)
    assert s.config_id != t.config_id
    old = load_config("configs/preprocessing/mechasight_regime_t.json")
    assert s.config_id != old.config_id and t.config_id != old.config_id


def test_v2_config_rejects_historical_v1_split_manifest():
    catalog = pd.DataFrame({"machine": [], "segment_id": []})
    old = pd.read_csv("reports/phase2/protocol_b_split_v1.csv")
    with pytest.raises(ValueError, match="incompatible"):
        merge_protocol_b_assignments(
            catalog, old, "T", split_version="protocol_b_common_test_v2"
        )


def _spec(regime: str) -> PairedExperimentSpec:
    return PairedExperimentSpec(
        regime=regime,
        training_pool_id="source" if regime == "S" else "source+adaptation",
        adaptation_access=regime == "T",
        test_segment_ids=frozenset({"a", "b"}),
        test_parent_recordings=frozenset({"p1", "p2"}),
        anomaly_test_segment_ids=frozenset({"b"}),
        preprocessing_signature="shared",
        window_configuration="100ms/100ms",
        feature_configuration="f1",
        model_configuration="m1",
        score_aggregation="median",
        metric_implementation="standard-v1",
    )


def test_paired_experiment_invariants_accept_only_adaptation_difference():
    assert_paired_experiment_specs(_spec("S"), _spec("T"))


@pytest.mark.parametrize("field", [
    "test_segment_ids", "test_parent_recordings", "anomaly_test_segment_ids",
    "preprocessing_signature", "window_configuration", "feature_configuration",
    "model_configuration", "score_aggregation", "metric_implementation",
])
def test_paired_experiment_invariants_reject_other_differences(field):
    value = frozenset({"changed"}) if field.endswith("ids") or field == "test_parent_recordings" else "changed"
    with pytest.raises(AssertionError, match=field):
        assert_paired_experiment_specs(_spec("S"), replace(_spec("T"), **{field: value}))


def test_threshold_is_higher_method_normal_quantile():
    scores = np.arange(10, dtype=float)
    assert normal_quantile_threshold(scores, quantile=0.9) == 9.0


def test_segment_aggregation_is_frozen_median_with_window_links():
    result = aggregate_segment_scores(["w1", "w2", "w3"], np.array([1.0, 9.0, 3.0]))
    assert result["segment_score"] == 3.0
    assert result["contributing_window_ids"] == ("w1", "w2", "w3")
    with pytest.raises(ValueError, match="median"):
        aggregate_segment_scores(["w"], np.array([1.0]), method="mean")


def test_parent_cluster_bootstrap_keeps_all_sibling_segments_together():
    frame = pd.DataFrame({
        "parent_recording": ["p1", "p1", "p2"],
        "segment_id": ["a", "b", "c"],
        "domain": ["source"] * 3,
        "is_anomaly": [False] * 3,
        "operating_condition": ["w0"] * 3,
    })
    sampled = draw_parent_cluster_sample(frame, seed=1995)
    assert sampled["bootstrap_cluster_instance"].nunique() == 2
    original_sizes = frame.groupby("parent_recording").size().to_dict()
    for _, cluster in sampled.groupby("bootstrap_cluster_instance"):
        parent = cluster["bootstrap_parent_recording"].iloc[0]
        assert len(cluster) == original_sizes[parent]
