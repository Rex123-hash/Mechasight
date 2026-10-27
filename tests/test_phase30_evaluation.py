"""Phase 3.0 score, metric, aggregation, and bootstrap contracts."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.experiment.aggregation import aggregate_segment_windows
from src.experiment.bootstrap import (
    bootstrap_metric,
    bootstrap_paired_difference,
    draw_stratified_cluster_indices,
)
from src.experiment.metrics import (
    protocol_b_ranking_metrics,
    protocol_b_threshold_metrics,
)
from src.experiment.provenance import ordered_id_hash
from src.experiment.scores import ScoreBatch, ScoreProvenance, select_protocol_b_threshold


def _provenance() -> ScoreProvenance:
    return ScoreProvenance(
        experiment_id="p3-synthetic",
        machine="RoboticArm",
        protocol="mechasight_experimental_protocol",
        regime="S",
        model_id="statistical-v1",
        model_family="statistical_reference",
        model_seed=0,
        preprocessing_config_id="phase2.2-v3-test",
        preprocessing_config_hash="config-hash",
        split_version="protocol_b_few_shot_adaptation_v3",
        score_direction="higher_is_more_anomalous",
        data_provenance={
            "manifest_hash": "manifest-hash",
            "source_validation_cohort_hash": ordered_id_hash(
                tuple(f"s{i}" for i in range(10))
            ),
        },
        config_provenance={"git_sha": "abc"},
    )


def _batch(
    *,
    role: str = "source_validation",
    labels: tuple[str, ...] = ("normal",) * 10,
    provenance: ScoreProvenance | None = None,
) -> ScoreBatch:
    n = len(labels)
    return ScoreBatch(
        provenance=provenance or _provenance(),
        scores=np.arange(n, dtype=float),
        segment_ids=tuple(f"s{i}" for i in range(n)),
        parent_recording_ids=tuple(f"p{i}" for i in range(n)),
        cohort_roles=(role,) * n,
        domains=("source",) * n,
        anomaly_labels=labels,
    )


def test_score_direction_is_frozen_to_higher_more_anomalous():
    with pytest.raises(ValueError, match="score direction"):
        replace(_provenance(), score_direction="lower_is_more_anomalous")


def test_score_batch_requires_unique_segment_ids():
    batch = _batch()
    with pytest.raises(ValueError, match="unique"):
        replace(batch, segment_ids=("same",) * len(batch.segment_ids))


def test_validation_threshold_is_q90_higher_with_provenance():
    selected = select_protocol_b_threshold(_batch(), expected_provenance=_provenance())
    assert selected.value == 9.0
    assert selected.quantile == 0.9
    assert selected.quantile_method == "higher"
    assert selected.source_role == "source_validation"


@pytest.mark.parametrize("role", ["common_evaluation", "target_adaptation", "source_training"])
def test_threshold_rejects_non_validation_roles(role):
    with pytest.raises(ValueError, match="source_validation"):
        select_protocol_b_threshold(_batch(role=role), expected_provenance=_provenance())


def test_threshold_rejects_mixed_roles():
    batch = _batch()
    mixed = replace(batch, cohort_roles=("source_validation",) * 9 + ("common_evaluation",))
    with pytest.raises(ValueError, match="pure source_validation"):
        select_protocol_b_threshold(mixed, expected_provenance=_provenance())


def test_threshold_rejects_anomaly_scores():
    labels = ("normal",) * 9 + ("fault",)
    with pytest.raises(ValueError, match="normal scores only"):
        select_protocol_b_threshold(_batch(labels=labels), expected_provenance=_provenance())


def test_threshold_rejects_target_domain_validation():
    batch = replace(_batch(), domains=("target",) * 10)
    with pytest.raises(ValueError, match="source-domain"):
        select_protocol_b_threshold(batch, expected_provenance=_provenance())


def test_threshold_rejects_stale_experiment_provenance():
    expected = replace(_provenance(), experiment_id="p3-other")
    with pytest.raises(ValueError, match="stale or wrong"):
        select_protocol_b_threshold(_batch(), expected_provenance=expected)


def test_threshold_rejects_ids_outside_frozen_validation_cohort():
    batch = replace(_batch(), segment_ids=tuple(f"different-{i}" for i in range(10)))
    with pytest.raises(ValueError, match="frozen source-validation cohort"):
        select_protocol_b_threshold(batch, expected_provenance=_provenance())


@pytest.mark.parametrize(
    ("scores", "expected"),
    [([1.0], 1.0), ([1.0, 5.0, 3.0], 3.0), ([1.0, 2.0, 9.0, 10.0], 5.5)],
)
def test_median_aggregation_one_odd_even(scores, expected):
    n = len(scores)
    result = aggregate_segment_windows(
        segment_ids=["s"] * n,
        parent_recording_ids=["p"] * n,
        window_ids=[f"w{i}" for i in range(n)],
        window_scores=np.asarray(scores),
    )
    assert result.segment_score == expected
    assert result.n_windows == n


def test_median_aggregation_is_order_independent():
    first = aggregate_segment_windows(
        segment_ids=["s"] * 3, parent_recording_ids=["p"] * 3,
        window_ids=["w1", "w2", "w3"], window_scores=np.array([1.0, 9.0, 3.0]),
    )
    second = aggregate_segment_windows(
        segment_ids=["s"] * 3, parent_recording_ids=["p"] * 3,
        window_ids=["w3", "w1", "w2"], window_scores=np.array([3.0, 1.0, 9.0]),
    )
    assert first == second


def test_aggregation_rejects_duplicate_missing_and_mixed_ids():
    common = dict(parent_recording_ids=["p", "p"], window_scores=np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_segment_windows(segment_ids=["s", "s"], window_ids=["w", "w"], **common)
    with pytest.raises(ValueError, match="missing"):
        aggregate_segment_windows(segment_ids=["s", "s"], window_ids=["w", ""], **common)
    with pytest.raises(ValueError, match="mixed segment"):
        aggregate_segment_windows(segment_ids=["s1", "s2"], window_ids=["w1", "w2"], **common)


def test_protocol_b_metrics_match_known_perfect_example():
    result = protocol_b_ranking_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert result["auroc"] == 1.0
    assert result["average_precision"] == 1.0
    assert result["standardized_pauroc_fpr_0_10"] == 1.0


def test_protocol_b_metrics_match_known_reversed_example():
    result = protocol_b_ranking_metrics(np.array([0, 0, 1, 1]), np.array([0.9, 0.8, 0.2, 0.1]))
    assert result["auroc"] == 0.0
    assert np.isclose(result["average_precision"], 5 / 12)
    assert np.isclose(result["standardized_pauroc_fpr_0_10"], 9 / 19)


def test_threshold_metrics_use_strict_greater_than_rule():
    result = protocol_b_threshold_metrics(
        np.array([0, 0, 1, 1]), np.array([0.5, 0.6, 0.5, 0.9]), threshold=0.5
    )
    assert (result.true_negative, result.false_positive) == (1, 1)
    assert (result.false_negative, result.true_positive) == (1, 1)
    assert result.f1 == 0.5


def _bootstrap_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "segment_id": ["a", "b", "c", "d", "e", "f"],
        "parent_recording": ["p1", "p1", "p2", "p3", "p3", "p4"],
        "machine": ["RoboticArm"] * 6,
        "domain": ["source"] * 6,
        "is_anomaly": [0, 0, 0, 1, 1, 1],
        "operating_condition": ["c"] * 6,
        "score": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
        "score_a": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
        "score_b": [0.9, 0.8, 0.7, 0.3, 0.2, 0.1],
    })


def test_cluster_draw_keeps_every_parent_child_together():
    frame = _bootstrap_frame()
    indices = draw_stratified_cluster_indices(frame, rng=np.random.default_rng(1995))
    sampled = frame.loc[indices]
    original_sizes = frame.groupby("parent_recording").size().to_dict()
    for parent, count in sampled.groupby("parent_recording").size().items():
        assert count % original_sizes[parent] == 0


def test_bootstrap_is_reproducible():
    first = bootstrap_metric(_bootstrap_frame(), metric="auroc", replicates=100, seed=1995)
    second = bootstrap_metric(_bootstrap_frame(), metric="auroc", replicates=100, seed=1995)
    assert first == second
    assert first.number_valid == 100 and first.number_invalid == 0


def test_paired_bootstrap_uses_one_shared_draw_and_is_reproducible():
    first = bootstrap_paired_difference(
        _bootstrap_frame(), metric="auroc", replicates=100, seed=1995
    )
    second = bootstrap_paired_difference(
        _bootstrap_frame(), metric="auroc", replicates=100, seed=1995
    )
    assert first == second
    assert first.point_estimate == 1.0
    assert first.number_valid == 100


def test_degenerate_bootstrap_reports_invalid_replicates_without_fabrication():
    frame = _bootstrap_frame().iloc[:3].copy()
    result = bootstrap_metric(frame, metric="auroc", replicates=25, seed=1995)
    assert result.point_estimate is None
    assert result.number_requested == 25
    assert result.number_valid == 0
    assert result.number_invalid == 25
    assert result.ci_lower is None and result.ci_upper is None
