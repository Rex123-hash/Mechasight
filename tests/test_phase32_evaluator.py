"""Synthetic-only tests for the locked Phase-3.2 evaluator."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.experiment.ledger import TestLookLedger
from src.experiment.phase32 import (
    EstimatorTestData,
    REPORTING_POLICY_ID,
    TestBatchAuthorization,
    authorize_first_test_batch,
    evaluate_authorized_experiment,
    family_bootstrap_distribution,
    invoke_authorized_scorer,
    make_common_bootstrap_draws,
    paired_distribution_difference,
    ranking_metrics_or_undefined,
    weighted_bootstrap_metric,
)
from src.experiment.metrics import RANKING_METRICS, ranking_metric_value


def _fake_reproduced(count: int = 28) -> dict:
    output = {}
    for index in range(count):
        experiment_id = f"p3-synthetic-{index:02d}"
        output[experiment_id] = SimpleNamespace(
            config=SimpleNamespace(
                experiment_id=experiment_id,
                machine="RoboticArm" if index % 2 == 0 else "BrushlessMotor",
                protocol="mechasight_experimental_protocol",
                regime="S" if index % 4 < 2 else "T",
            ),
            model_id=f"model-{index}",
        )
    return output


def test_exact_frozen_28_id_batch_is_written_atomically_before_any_score(tmp_path):
    ledger = TestLookLedger(tmp_path / "ledger.jsonl")
    reproduced = _fake_reproduced()
    receipt = authorize_first_test_batch(
        ledger=ledger,
        reproduced=reproduced,
        expected_experiment_ids=frozenset(reproduced),
        test_cohort_hashes_by_machine={"RoboticArm": "ra", "BrushlessMotor": "bm"},
        test_evaluation_code_sha="locked-code-sha",
    )
    assert receipt.ledger_start_count == 0
    assert receipt.ledger_end_count == 28
    assert len(ledger.read()) == 28
    assert {row["git_sha"] for row in ledger.read()} == {"locked-code-sha"}
    assert all(REPORTING_POLICY_ID in row["reason"] for row in ledger.read())

    observed = []

    def scorer(X):
        observed.append(len(ledger.read()))
        return np.zeros(len(X))

    first_id = sorted(reproduced)[0]
    invoke_authorized_scorer(
        experiment_id=first_id,
        test_cohort_hash=receipt.experiment_test_cohort_hashes[first_id],
        authorization=receipt,
        scorer=scorer,
        standardized_test_X=np.ones((3, 2)),
    )
    assert observed == [28]


def test_unknown_or_incomplete_experiment_set_is_rejected_without_ledger_write(tmp_path):
    ledger = TestLookLedger(tmp_path / "ledger.jsonl")
    reproduced = _fake_reproduced(27)
    expected = frozenset({*reproduced, "unknown-frozen-id"})
    with pytest.raises(ValueError, match="exactly the 28 frozen"):
        authorize_first_test_batch(
            ledger=ledger,
            reproduced=reproduced,
            expected_experiment_ids=expected,
            test_cohort_hashes_by_machine={"RoboticArm": "ra", "BrushlessMotor": "bm"},
            test_evaluation_code_sha="locked-code-sha",
        )
    assert ledger.read() == [] and not ledger.path.exists()


def test_cohort_hash_mismatch_fails_before_scorer_call():
    called = False

    def scorer(X):
        nonlocal called
        called = True
        return np.zeros(len(X))

    receipt = TestBatchAuthorization(
        batch_id="batch", test_evaluation_code_sha="sha",
        reporting_policy_id=REPORTING_POLICY_ID,
        experiment_ids=frozenset({"e"}),
        experiment_test_cohort_hashes={"e": "expected"},
        ledger_start_count=0, ledger_end_count=28,
    )
    with pytest.raises(ValueError, match="cohort hashes differ"):
        invoke_authorized_scorer(
            experiment_id="e", test_cohort_hash="wrong", authorization=receipt,
            scorer=scorer, standardized_test_X=np.ones((2, 1)),
        )
    assert called is False


class _IdentityScaler:
    @staticmethod
    def transform(X):
        return np.asarray(X)


class _FirstColumnModel:
    def __init__(self):
        self.called = False

    def score(self, X):
        self.called = True
        return np.asarray(X)[:, 0]


def _synthetic_test_data() -> EstimatorTestData:
    return EstimatorTestData(
        X=np.array([[0.1], [0.8], [0.4], [0.9]]),
        feature_names=("sensor__feature",),
        segment_ids=("s1", "s2", "s3", "s4"),
        parent_recordings=("p1", "p2", "p3", "p4"),
        domains=("source", "source", "target", "target"),
        anomaly_labels=("normal", "fault", "normal", "fault"),
        operating_conditions=("a", "a", "b", "b"),
        background_conditions=("a", "a", "b", "b"),
        cohort_hash="cohort", machine="RoboticArm", regime="S",
    )


def _receipt() -> TestBatchAuthorization:
    return TestBatchAuthorization(
        batch_id="batch", test_evaluation_code_sha="sha",
        reporting_policy_id=REPORTING_POLICY_ID,
        experiment_ids=frozenset({"e"}),
        experiment_test_cohort_hashes={"e": "cohort"},
        ledger_start_count=0, ledger_end_count=28,
    )


def test_threshold_identity_mismatch_fails_before_test_model_scoring():
    model = _FirstColumnModel()
    reproduced = SimpleNamespace(
        config=SimpleNamespace(
            experiment_id="e", machine="RoboticArm", regime="S",
            model_family="statistical_reference", model_seed=0,
        ),
        model=model, standardizer=_IdentityScaler(), feature_names=("sensor__feature",),
        threshold=SimpleNamespace(source_experiment_id="other", value=0.5),
    )
    with pytest.raises(ValueError, match="another experiment"):
        evaluate_authorized_experiment(reproduced, _synthetic_test_data(), _receipt())
    assert model.called is False


def test_threshold_confusion_counts_and_domain_one_class_status_are_explicit():
    model = _FirstColumnModel()
    reproduced = SimpleNamespace(
        config=SimpleNamespace(
            experiment_id="e", machine="RoboticArm", regime="S",
            model_family="statistical_reference", model_seed=0,
        ),
        model=model, standardizer=_IdentityScaler(), feature_names=("sensor__feature",),
        threshold=SimpleNamespace(source_experiment_id="e", value=0.5),
    )
    result = evaluate_authorized_experiment(
        reproduced, _synthetic_test_data(), _receipt()
    )
    assert result.complete_metrics["true_negative"] == 2
    assert result.complete_metrics["true_positive"] == 2
    assert result.complete_metrics["false_positive"] == 0
    assert result.complete_metrics["false_negative"] == 0
    assert result.healthy_target == {
        "healthy_target_sample_count": 1,
        "false_positive": 0,
        "false_positive_rate": 0.0,
    }
    undefined = ranking_metrics_or_undefined(np.zeros(3), np.array([0.1, 0.2, 0.3]))
    assert undefined["status"] == "UNDEFINED — ONE CLASS PRESENT"
    assert undefined["auroc"] is None


def _bootstrap_frame(scores) -> pd.DataFrame:
    return pd.DataFrame({
        "segment_id": ["n1", "n2", "n3", "a1", "a2", "a3"],
        "parent_recording": ["np1", "np2", "np3", "ap1", "ap2", "ap3"],
        "machine": ["RoboticArm"] * 6,
        "domain": ["source", "target", "target", "source", "target", "target"],
        "is_anomaly": [0, 0, 0, 1, 1, 1],
        "score": scores,
    })


def test_deterministic_bootstrap_and_paired_s_t_difference_reproduce():
    frame_s = _bootstrap_frame([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    frame_t = _bootstrap_frame([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    draws_a = make_common_bootstrap_draws(frame_s, replicates=30, seed=1995)
    draws_b = make_common_bootstrap_draws(frame_s, replicates=30, seed=1995)
    np.testing.assert_array_equal(draws_a.sample_weights, draws_b.sample_weights)
    assert draws_a.identity == draws_b.identity
    dist_s = family_bootstrap_distribution(
        "statistical_reference", {0: frame_s}, metric="auroc", draws=draws_a
    )
    dist_t = family_bootstrap_distribution(
        "statistical_reference", {0: frame_t}, metric="auroc", draws=draws_a
    )
    repeated = family_bootstrap_distribution(
        "statistical_reference", {0: frame_s}, metric="auroc", draws=draws_b
    )
    np.testing.assert_array_equal(dist_s.values, repeated.values)
    delta = paired_distribution_difference(dist_t, dist_s, orientation="T_minus_S")
    assert delta["orientation"] == "T_minus_S"
    assert delta["valid_replicates"] == 30


def test_if_bootstrap_family_uses_all_five_seed_vectors_with_shared_draws():
    frames = {
        seed: _bootstrap_frame(
            [0.1, 0.2 + seed * 0.005, 0.4, 0.6, 0.8 - seed * 0.005, 0.9]
        )
        for seed in range(5)
    }
    draws = make_common_bootstrap_draws(frames[0], replicates=20, seed=1995)
    result = family_bootstrap_distribution(
        "isolation_forest", frames, metric="average_precision", draws=draws
    )
    assert result.replicates_requested == result.valid_replicates == 20
    with pytest.raises(ValueError, match="frozen seed set"):
        family_bootstrap_distribution(
            "isolation_forest", {4: frames[4]}, metric="average_precision", draws=draws
        )


def test_vectorized_weighted_bootstrap_matches_official_metrics_on_expanded_draws():
    frame = _bootstrap_frame([0.1, 0.1, 0.4, 0.6, 0.8, 0.8])
    draws = make_common_bootstrap_draws(frame, replicates=20, seed=1995)
    labels = frame["is_anomaly"].to_numpy()
    scores = frame["score"].to_numpy()
    for metric in RANKING_METRICS:
        optimized = weighted_bootstrap_metric(
            labels, scores, draws.sample_weights, metric=metric
        )
        reference = []
        for weights in draws.sample_weights:
            expanded = np.repeat(np.arange(len(frame)), weights.astype(int))
            reference.append(ranking_metric_value(metric, labels[expanded], scores[expanded]))
        np.testing.assert_allclose(optimized, reference, rtol=0, atol=1e-15)


def test_paired_model_comparison_orientation_is_stable():
    first = family_bootstrap_distribution(
        "statistical_reference", {0: _bootstrap_frame([0, 0.1, 0.2, 0.8, 0.9, 1])},
        metric="auroc", draws=make_common_bootstrap_draws(_bootstrap_frame([0, 0.1, 0.2, 0.8, 0.9, 1]), replicates=10),
    )
    second = family_bootstrap_distribution(
        "one_class_svm", {0: _bootstrap_frame([1, 0.9, 0.8, 0.2, 0.1, 0])},
        metric="auroc", draws=make_common_bootstrap_draws(_bootstrap_frame([0, 0.1, 0.2, 0.8, 0.9, 1]), replicates=10),
    )
    delta = paired_distribution_difference(
        first, second, orientation="statistical_reference_minus_one_class_svm"
    )
    assert delta["point_delta"] > 0
    assert delta["orientation"] == "statistical_reference_minus_one_class_svm"


def test_atomic_ledger_batch_refuses_wrong_existing_count(tmp_path):
    ledger = TestLookLedger(tmp_path / "ledger.jsonl")
    reproduced = _fake_reproduced()
    authorize_first_test_batch(
        ledger=ledger, reproduced=reproduced,
        expected_experiment_ids=frozenset(reproduced),
        test_cohort_hashes_by_machine={"RoboticArm": "ra", "BrushlessMotor": "bm"},
        test_evaluation_code_sha="sha",
    )
    with pytest.raises(RuntimeError, match="starting count"):
        authorize_first_test_batch(
            ledger=ledger, reproduced=reproduced,
            expected_experiment_ids=frozenset(reproduced),
            test_cohort_hashes_by_machine={"RoboticArm": "ra", "BrushlessMotor": "bm"},
            test_evaluation_code_sha="sha",
        )
