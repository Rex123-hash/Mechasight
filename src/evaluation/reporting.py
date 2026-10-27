"""Pre-registered first-test reporting rules; contains no test-scoring path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from src.experiment.provenance import content_sha256


ISOLATION_FOREST_SEEDS = (0, 1, 2, 3, 4)
DETERMINISTIC_MODEL_FAMILIES = frozenset({"statistical_reference", "one_class_svm"})
PRIMARY_TEST_METRICS = ("auroc", "average_precision")
SECONDARY_TEST_METRICS = ("standardized_pauroc_fpr_0_10",)
THRESHOLD_DEPENDENT_METRICS = (
    "true_negative", "false_positive", "false_negative", "true_positive",
    "false_positive_rate", "recall", "precision", "f1",
)
ALLOWED_PHASE311_ACTIVITIES = frozenset({
    "bundle_identity_audit", "policy_validation", "synthetic_test",
})


@dataclass(frozen=True)
class FamilyMetricSummary:
    family: str
    seed_values: Mapping[int, float]
    arithmetic_mean: float
    population_standard_deviation: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class FrozenThresholdBinding:
    model_seed: int
    source_experiment_id: str
    source_validation_cohort_hash: str
    threshold_value: float
    quantile: float = 0.90
    source_role: str = "source_validation"


def expected_seeds(family: str) -> tuple[int, ...]:
    if family == "isolation_forest":
        return ISOLATION_FOREST_SEEDS
    if family in DETERMINISTIC_MODEL_FAMILIES:
        return (0,)
    raise ValueError(f"model family is not frozen for first-test reporting: {family!r}")


def summarize_seed_metrics(
    family: str, seed_values: Mapping[int, float]
) -> FamilyMetricSummary:
    """Summarize every required seed; no best-seed reduction is available."""

    expected = expected_seeds(family)
    if set(seed_values) != set(expected):
        raise ValueError(f"{family} reporting requires exactly seeds {list(expected)}")
    ordered = np.asarray([seed_values[seed] for seed in expected], dtype=np.float64)
    if not np.all(np.isfinite(ordered)):
        raise ValueError("seed metrics must all be finite")
    return FamilyMetricSummary(
        family=family,
        seed_values={seed: float(seed_values[seed]) for seed in expected},
        arithmetic_mean=float(np.mean(ordered)),
        population_standard_deviation=float(np.std(ordered, ddof=0)),
        minimum=float(np.min(ordered)),
        maximum=float(np.max(ordered)),
    )


def aggregate_if_bootstrap_replicate(
    seed_metric_values: Mapping[int, float],
    seed_resample_identities: Mapping[int, str],
) -> float:
    """Average five IF metrics after proving the replicate draw is shared."""

    summary = summarize_seed_metrics("isolation_forest", seed_metric_values)
    if set(seed_resample_identities) != set(ISOLATION_FOREST_SEEDS):
        raise ValueError("every Isolation-Forest seed requires a resample identity")
    identities = {str(seed_resample_identities[seed]) for seed in ISOLATION_FOREST_SEEDS}
    if "" in identities or len(identities) != 1:
        raise ValueError("all Isolation-Forest seeds must use the identical cluster draw")
    return summary.arithmetic_mean


def validate_threshold_bindings(
    family: str, bindings: Mapping[int, FrozenThresholdBinding]
) -> None:
    """Bind each reported seed to its own already-frozen validation q90."""

    expected = expected_seeds(family)
    if set(bindings) != set(expected):
        raise ValueError(f"{family} requires one frozen threshold for every seed")
    experiment_ids: set[str] = set()
    for seed in expected:
        binding = bindings[seed]
        if binding.model_seed != seed:
            raise ValueError("threshold seed and reported model seed differ")
        if binding.source_role != "source_validation" or binding.quantile != 0.90:
            raise ValueError("threshold must be the frozen source-validation q90")
        if not binding.source_experiment_id or not binding.source_validation_cohort_hash:
            raise ValueError("threshold provenance is incomplete")
        if not np.isfinite(float(binding.threshold_value)):
            raise ValueError("threshold must be finite")
        if binding.source_experiment_id in experiment_ids:
            raise ValueError("different seeds cannot silently share an experiment threshold")
        experiment_ids.add(binding.source_experiment_id)


def require_identical_test_cohort_identity(
    regime_s_cohort_hash: str, regime_t_cohort_hash: str
) -> str:
    if not regime_s_cohort_hash or regime_s_cohort_hash != regime_t_cohort_hash:
        raise ValueError("paired S-vs-T reporting requires identical test cohort identity")
    return regime_s_cohort_hash


def require_machine_specific_result(machine_values: Iterable[str]) -> str:
    machines = {str(value) for value in machine_values}
    if len(machines) != 1 or not next(iter(machines), ""):
        raise ValueError("first-test results cannot pool machine cohorts")
    return next(iter(machines))


def reporting_policy_identity(policy: Mapping) -> str:
    if not str(policy.get("schema_version", "")):
        raise ValueError("reporting policy requires a schema version")
    return f"p311-reporting-{content_sha256(dict(policy))[:24]}"


def validate_primary_metric_freeze(policy: Mapping) -> None:
    if tuple(policy.get("primary_ranking_metrics", ())) != PRIMARY_TEST_METRICS:
        raise ValueError("primary first-test metrics are frozen to AUROC and AP")
    if tuple(policy.get("secondary_ranking_metrics", ())) != SECONDARY_TEST_METRICS:
        raise ValueError("secondary first-test metric is frozen to standardized pAUROC")
    if tuple(policy.get("threshold_dependent_metrics", ())) != THRESHOLD_DEPENDENT_METRICS:
        raise ValueError("threshold-dependent first-test metrics changed")


def require_new_version_after_test_look(
    old_policy: Mapping, new_policy: Mapping, *, test_look_occurred: bool
) -> None:
    if reporting_policy_identity(old_policy) == reporting_policy_identity(new_policy):
        return
    if test_look_occurred and old_policy.get("schema_version") == new_policy.get("schema_version"):
        raise ValueError("post-test reporting-policy changes require a new schema version")


def assert_phase311_activity(activity: str) -> None:
    if activity not in ALLOWED_PHASE311_ACTIVITIES:
        raise ValueError(
            "Phase 3.1.1 is a no-test-scoring freeze; activity is not authorized"
        )
