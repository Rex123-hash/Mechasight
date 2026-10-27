"""Synthetic/planted checks for the Phase-3.1.1 reporting freeze."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.reporting import (
    FrozenThresholdBinding,
    aggregate_if_bootstrap_replicate,
    assert_phase311_activity,
    reporting_policy_identity,
    require_identical_test_cohort_identity,
    require_machine_specific_result,
    require_new_version_after_test_look,
    summarize_seed_metrics,
    validate_primary_metric_freeze,
    validate_threshold_bindings,
)
from src.experiment.ledger import TestLookLedger
from src.experiment.phase31 import phase31_source_hash
from scripts.run_phase31_baselines import SOURCE_IDENTITY_PATHS


POLICY_PATH = Path("configs/phase3/first_test_reporting_policy_v1.json")


def _policy() -> dict:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def test_five_if_seeds_are_required_and_cannot_be_reduced_to_best_seed():
    values = {0: 0.50, 1: 0.55, 2: 0.40, 3: 0.60, 4: 0.45}
    summary = summarize_seed_metrics("isolation_forest", values)
    assert summary.seed_values == values
    assert summary.arithmetic_mean == pytest.approx(0.50)
    assert summary.population_standard_deviation == pytest.approx(np.std(list(values.values())))
    assert summary.minimum == 0.40
    assert summary.maximum == 0.60
    assert summary.arithmetic_mean != summary.maximum
    with pytest.raises(ValueError, match="exactly seeds"):
        summarize_seed_metrics("isolation_forest", {4: 0.60})


@pytest.mark.parametrize("family", ["statistical_reference", "one_class_svm"])
def test_deterministic_models_remain_single_run(family):
    summary = summarize_seed_metrics(family, {0: 0.7})
    assert summary.arithmetic_mean == 0.7
    assert summary.population_standard_deviation == 0.0
    with pytest.raises(ValueError, match="exactly seeds"):
        summarize_seed_metrics(family, {seed: 0.7 for seed in range(5)})


def test_if_bootstrap_averages_five_seed_metrics_after_one_identical_draw():
    values = {seed: float(seed + 1) for seed in range(5)}
    draw_identities = {seed: "replicate-17-clusters-aabbcc" for seed in range(5)}
    assert aggregate_if_bootstrap_replicate(values, draw_identities) == 3.0


def test_if_bootstrap_rejects_seed_specific_cluster_draws():
    values = {seed: float(seed) for seed in range(5)}
    identities = {seed: f"different-{seed}" for seed in range(5)}
    with pytest.raises(ValueError, match="identical cluster draw"):
        aggregate_if_bootstrap_replicate(values, identities)


def test_s_vs_t_requires_identical_common_test_cohort_identity():
    assert require_identical_test_cohort_identity("same-hash", "same-hash") == "same-hash"
    with pytest.raises(ValueError, match="identical test cohort"):
        require_identical_test_cohort_identity("s-hash", "t-hash")


def test_machine_results_cannot_be_silently_pooled():
    assert require_machine_specific_result(["RoboticArm", "RoboticArm"]) == "RoboticArm"
    with pytest.raises(ValueError, match="cannot pool"):
        require_machine_specific_result(["RoboticArm", "BrushlessMotor"])


def test_primary_metric_change_alters_identity_and_requires_post_test_version_change():
    original = _policy()
    validate_primary_metric_freeze(original)
    assert reporting_policy_identity(original) == "p311-reporting-5e76ab7dabd5689c7b0ced68"
    changed = deepcopy(original)
    changed["primary_ranking_metrics"].append("f1")
    assert reporting_policy_identity(changed) != reporting_policy_identity(original)
    with pytest.raises(ValueError, match="new schema version"):
        require_new_version_after_test_look(original, changed, test_look_occurred=True)
    with pytest.raises(ValueError, match="frozen to AUROC and AP"):
        validate_primary_metric_freeze(changed)


def _if_thresholds() -> dict[int, FrozenThresholdBinding]:
    return {
        seed: FrozenThresholdBinding(
            model_seed=seed,
            source_experiment_id=f"p3-seed-{seed}",
            source_validation_cohort_hash="validation-hash",
            threshold_value=0.5 + seed / 100,
        )
        for seed in range(5)
    }


def test_threshold_metrics_bind_each_if_seed_to_its_own_frozen_q90():
    bindings = _if_thresholds()
    validate_threshold_bindings("isolation_forest", bindings)
    assert len({item.threshold_value for item in bindings.values()}) == 5
    swapped = dict(bindings)
    swapped[0] = bindings[1]
    with pytest.raises(ValueError, match="threshold seed"):
        validate_threshold_bindings("isolation_forest", swapped)


def test_phase311_authorizes_audit_policy_and_synthetic_work_but_no_test_scoring(tmp_path):
    ledger = TestLookLedger(tmp_path / "test_look_ledger.jsonl")
    before = ledger.read()
    for activity in ("bundle_identity_audit", "policy_validation", "synthetic_test"):
        assert_phase311_activity(activity)
    for forbidden in ("materialize_common_test_features", "score_common_test", "run_real_bootstrap"):
        with pytest.raises(ValueError, match="no-test-scoring freeze"):
            assert_phase311_activity(forbidden)
    assert ledger.read() == before == []
    assert not ledger.path.exists()


def test_policy_freezes_required_slices_seed_aggregation_and_bootstrap():
    policy = _policy()
    assert policy["stochastic_seed_aggregation"]["isolation_forest_seeds"] == [0, 1, 2, 3, 4]
    assert policy["stochastic_seed_aggregation"]["no_best_seed_rule"] is True
    assert policy["paired_bootstrap"]["replicates"] == 10_000
    assert policy["paired_bootstrap"]["seed"] == 1995
    assert policy["regime_comparison"]["contrast"] == "T_minus_S"
    assert policy["model_family_comparisons"]["cross_machine_pooling"] is False
    assert policy["phase3_1_1_test_access"] == {
        "common_test_features_materialized": False,
        "common_test_segments_scored": False,
        "test_performance_metrics_computed": False,
        "real_bootstrap_run": False,
    }


def test_all_28_immutable_bundles_share_the_verified_phase31_code_identity():
    report = json.loads(
        Path("reports/phase3_1/phase31_pretest_freeze_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(report["experiments"]) == 28
    expected_source = "93c33580054e9e45f228e385ba726bb43ad0dfb3531f83bb9f3c9c5fb003e229"
    assert phase31_source_hash(SOURCE_IDENTITY_PATHS) == expected_source
    for experiment in report["experiments"]:
        bundle = Path(experiment["bundle"])
        config = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
        provenance = json.loads((bundle / "provenance.json").read_text(encoding="utf-8"))
        assert config["code_git_sha"] == "3bf046dd8edb29d99f1d246b33f88674a5b53a2d"
        assert config["model_hyperparameters"]["phase31_source_hash"] == expected_source
        assert (
            provenance["score_provenance"]["config_provenance"]["phase31_source_hash"]
            == expected_source
        )
        assert provenance["test_evaluated"] is False


def test_real_frozen_if_bundles_bind_all_five_seeds_to_own_q90_per_machine_regime():
    report = json.loads(
        Path("reports/phase3_1/phase31_pretest_freeze_report.json").read_text(
            encoding="utf-8"
        )
    )
    for machine in ("RoboticArm", "BrushlessMotor"):
        for regime in ("S", "T"):
            selected = [
                item for item in report["experiments"]
                if item["machine"] == machine
                and item["regime"] == regime
                and item["family"] == "isolation_forest"
            ]
            bindings = {}
            for item in selected:
                threshold = json.loads(
                    (Path(item["bundle"]) / "threshold.json").read_text(encoding="utf-8")
                )
                bindings[item["seed"]] = FrozenThresholdBinding(
                    model_seed=item["seed"],
                    source_experiment_id=threshold["source_experiment_id"],
                    source_validation_cohort_hash=threshold["source_cohort_hash"],
                    threshold_value=threshold["value"],
                    quantile=threshold["quantile"],
                    source_role=threshold["source_role"],
                )
            validate_threshold_bindings("isolation_forest", bindings)
