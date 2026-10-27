"""Phase-3.2.1 post-test reporting correction.

This module corrects how already-recorded Phase-3.2 evidence is *presented*.
It performs no experiment.

Architectural guarantee: nothing here imports an estimator, a scorer, a
preprocessing pipeline or the Phase-3.2 common-test runner.  Every corrected
value is arithmetic over confusion counts and cohort metadata that were already
persisted and hashed at Phase 3.2.  ``assert_no_model_dependency`` makes that
guarantee testable rather than merely stated.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import sys
from typing import Mapping

from src.experiment.provenance import content_sha256


REPORTING_CORRECTION_SCHEMA = "phase3.2.1-reporting-correction-v1"
PARENT_POLICY_ID = "p311-reporting-5e76ab7dabd5689c7b0ced68"

PRECISION_UNDEFINED = "undefined - no positive predictions"
NO_ALARM_ANNOTATION = (
    "No positive alarms were generated at the frozen operating point; therefore "
    "zero false positives coexist with zero fault detections."
)

#: Modules whose presence in a correction call stack would mean a model was touched.
FORBIDDEN_MODEL_MODULES = (
    "src.models.classical",
    "src.experiment.phase32",
    "src.experiment.phase31",
    "sklearn.ensemble",
    "sklearn.svm",
)

ISOLATION_FOREST_SEEDS = (0, 1, 2, 3, 4)


def reporting_correction_identity(policy: Mapping) -> str:
    """Deterministic identity for one correction policy document."""

    if str(policy.get("schema_version", "")) != REPORTING_CORRECTION_SCHEMA:
        raise ValueError(
            f"correction policy schema must be {REPORTING_CORRECTION_SCHEMA!r}"
        )
    if policy.get("parent_policy", {}).get("policy_id") != PARENT_POLICY_ID:
        raise ValueError("correction policy must declare the historical parent policy")
    return f"p321-correction-{content_sha256(dict(policy))[:24]}"


def assert_no_model_dependency() -> None:
    """Fail if any estimator/scorer module has been imported into this process.

    The correction is defined as arithmetic on persisted evidence.  If a model
    module is loaded while a correction is being produced, that definition is no
    longer structurally guaranteed, so the build refuses to continue.
    """

    loaded = sorted(m for m in FORBIDDEN_MODEL_MODULES if m in sys.modules)
    if loaded:
        raise RuntimeError(
            "Phase-3.2.1 corrections must be derived without model code; "
            f"forbidden modules already imported: {loaded}"
        )


# --------------------------------------------------------------------- metrics
def count_identity_f1(
    true_positive: int, false_positive: int, false_negative: int
) -> float | None:
    """F1 = 2TP / (2TP + FP + FN); undefined only when the denominator is zero.

    Equivalent to the harmonic mean of precision and recall wherever that form
    is itself defined, and additionally defined when TP = 0 while anomalies or
    positive predictions exist -- the case the Phase-3.2 implementation reported
    as null.
    """

    tp, fp, fn = int(true_positive), int(false_positive), int(false_negative)
    if min(tp, fp, fn) < 0:
        raise ValueError("confusion counts cannot be negative")
    denominator = 2 * tp + fp + fn
    if denominator == 0:
        return None
    return 2.0 * tp / denominator


def precision_or_undefined(
    true_positive: int, false_positive: int
) -> float | None:
    """TP/(TP+FP), or None when no positive prediction was made at all."""

    tp, fp = int(true_positive), int(false_positive)
    if min(tp, fp) < 0:
        raise ValueError("confusion counts cannot be negative")
    if tp + fp == 0:
        return None
    return tp / (tp + fp)


def recall_or_undefined(true_positive: int, false_negative: int) -> float | None:
    tp, fn = int(true_positive), int(false_negative)
    if min(tp, fn) < 0:
        raise ValueError("confusion counts cannot be negative")
    if tp + fn == 0:
        return None
    return tp / (tp + fn)


def false_positive_rate_or_undefined(
    false_positive: int, true_negative: int
) -> float | None:
    fp, tn = int(false_positive), int(true_negative)
    if min(fp, tn) < 0:
        raise ValueError("confusion counts cannot be negative")
    if fp + tn == 0:
        return None
    return fp / (fp + tn)


def produced_no_alarms(true_positive: int, false_positive: int) -> bool:
    """True when the configuration flagged nothing at all on this slice."""

    return int(true_positive) == 0 and int(false_positive) == 0


@dataclass(frozen=True)
class CorrectedThresholdMetrics:
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int
    false_positive_rate: float | None
    recall: float | None
    precision: float | None
    precision_status: str
    phase321_count_identity_f1: float | None
    f1_convention: str
    produced_no_positive_predictions: bool
    threshold_transfer_annotation: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def correct_threshold_metrics(
    *, true_negative: int, false_positive: int, false_negative: int, true_positive: int
) -> CorrectedThresholdMetrics:
    """Recompute one row's threshold metrics under the corrected conventions."""

    precision = precision_or_undefined(true_positive, false_positive)
    no_alarms = produced_no_alarms(true_positive, false_positive)
    return CorrectedThresholdMetrics(
        true_negative=int(true_negative),
        false_positive=int(false_positive),
        false_negative=int(false_negative),
        true_positive=int(true_positive),
        false_positive_rate=false_positive_rate_or_undefined(false_positive, true_negative),
        recall=recall_or_undefined(true_positive, false_negative),
        precision=precision,
        precision_status="defined" if precision is not None else PRECISION_UNDEFINED,
        phase321_count_identity_f1=count_identity_f1(
            true_positive, false_positive, false_negative
        ),
        f1_convention="2*TP / (2*TP + FP + FN)",
        produced_no_positive_predictions=no_alarms,
        threshold_transfer_annotation=NO_ALARM_ANNOTATION if no_alarms else None,
    )


# ---------------------------------------------------------------------- family
@dataclass(frozen=True)
class CorrectedFamilySummary:
    model_family: str
    seed_f1: dict
    family_f1: float | None
    seed_precision: dict
    family_precision: float | None
    family_precision_status: str
    defined_precision_seed_count: int
    expected_seed_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def correct_family_threshold_metrics(
    family: str, seed_counts: Mapping[int, Mapping[str, int]]
) -> CorrectedFamilySummary:
    """Average per-seed corrected F1; never average scores, never pick a seed."""

    expected = ISOLATION_FOREST_SEEDS if family == "isolation_forest" else (0,)
    if set(seed_counts) != set(expected):
        raise ValueError(f"{family} correction requires exactly seeds {list(expected)}")

    seed_f1: dict[int, float | None] = {}
    seed_precision: dict[int, float | None] = {}
    for seed in expected:
        counts = seed_counts[seed]
        seed_f1[seed] = count_identity_f1(
            counts["true_positive"], counts["false_positive"], counts["false_negative"]
        )
        seed_precision[seed] = precision_or_undefined(
            counts["true_positive"], counts["false_positive"]
        )

    f1_values = [v for v in seed_f1.values() if v is not None]
    family_f1 = sum(f1_values) / len(expected) if len(f1_values) == len(expected) else None

    defined_precision = [v for v in seed_precision.values() if v is not None]
    # No imputation is invented: one undefined seed makes the family value undefined.
    family_precision = (
        sum(defined_precision) / len(expected)
        if len(defined_precision) == len(expected)
        else None
    )
    return CorrectedFamilySummary(
        model_family=family,
        seed_f1=dict(seed_f1),
        family_f1=family_f1,
        seed_precision=dict(seed_precision),
        family_precision=family_precision,
        family_precision_status=(
            "defined" if family_precision is not None
            else (f"{PRECISION_UNDEFINED}; precision defined for only "
                  f"{len(defined_precision)} of {len(expected)} seeds")
        ),
        defined_precision_seed_count=len(defined_precision),
        expected_seed_count=len(expected),
    )


# ------------------------------------------------------------------ prevalence
@dataclass(frozen=True)
class PrevalenceContext:
    total_count: int
    anomaly_count: int
    normal_count: int
    prevalence: float
    average_precision: float
    ap_minus_prevalence: float
    status: str = "CONTEXT_ONLY"
    note: str = (
        "Average Precision has a prevalence-dependent floor. This difference is "
        "descriptive context, not a chance-adjusted metric."
    )

    def to_dict(self) -> dict:
        return asdict(self)


def prevalence_context(
    *, anomaly_count: int, normal_count: int, average_precision: float
) -> PrevalenceContext:
    """Attach the evaluated slice's anomaly prevalence to an AP value."""

    anomaly, normal = int(anomaly_count), int(normal_count)
    if min(anomaly, normal) < 0:
        raise ValueError("slice class counts cannot be negative")
    total = anomaly + normal
    if total == 0:
        raise ValueError("prevalence requires a non-empty slice")
    prevalence = anomaly / total
    return PrevalenceContext(
        total_count=total,
        anomaly_count=anomaly,
        normal_count=normal,
        prevalence=prevalence,
        average_precision=float(average_precision),
        ap_minus_prevalence=float(average_precision) - prevalence,
    )
