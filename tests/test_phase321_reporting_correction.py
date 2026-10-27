"""Phase-3.2.1: corrected reporting conventions and original-evidence immutability."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.experiment.ledger_history import verify_ledger_history
from src.experiment.reporting_correction import (
    ISOLATION_FOREST_SEEDS,
    NO_ALARM_ANNOTATION,
    PRECISION_UNDEFINED,
    assert_no_model_dependency,
    correct_family_threshold_metrics,
    correct_threshold_metrics,
    count_identity_f1,
    precision_or_undefined,
    prevalence_context,
    produced_no_alarms,
    recall_or_undefined,
    reporting_correction_identity,
)

PHASE32 = Path("reports/phase3_2")
PHASE321 = Path("reports/phase3_2_1")
POLICY = Path("configs/phase3/phase321_reporting_correction_policy_v1.json")
LEDGER = Path("reports/experiment_results/test_look_ledger.jsonl")

LEDGER_SHA256 = "db4bde80099a870ae90686fa463400f02f93434fc11dfce6d8c20f2eac0b0db0"
PARENT_POLICY_SHA256 = "a1f695d31a6042619bad3e4fa9103a62b32c41a699c54ce7d9ae35ce36b7da2c"
FROZEN_SPLIT_SHA256 = "68e5406d63e96cf5ee418ae58808f5c3e7790889ba1b8497420642d1f0d8409e"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------- A. count-identity F1
def test_f1_is_zero_when_nothing_is_flagged_and_anomalies_exist():
    """TP=0, FP=0, FN>0 -- the nine Phase-3.2 rows that reported null."""

    assert count_identity_f1(0, 0, 232) == 0.0


def test_f1_is_zero_when_only_false_alarms_are_raised():
    """TP=0, FP>0, FN>0 -- the RoboticArm/T seed-0 case, precision defined at 0.0."""

    assert count_identity_f1(0, 2, 232) == 0.0
    assert precision_or_undefined(0, 2) == 0.0
    assert recall_or_undefined(0, 232) == 0.0


def test_f1_matches_the_harmonic_definition_wherever_that_is_defined():
    for tp, fp, fn in [(129, 13, 103), (10, 1, 222), (148, 73, 8), (1, 1, 1), (7, 0, 3)]:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        harmonic = 2 * precision * recall / (precision + recall)
        assert count_identity_f1(tp, fp, fn) == pytest.approx(harmonic, abs=1e-12)


def test_f1_is_undefined_only_when_the_denominator_vanishes():
    assert count_identity_f1(0, 0, 0) is None


def test_negative_counts_are_rejected():
    for bad in ((-1, 0, 0), (0, -1, 0), (0, 0, -1)):
        with pytest.raises(ValueError):
            count_identity_f1(*bad)


# ------------------------------------------------------------- B. precision
def test_precision_is_undefined_when_no_positive_prediction_is_made():
    assert precision_or_undefined(0, 0) is None
    corrected = correct_threshold_metrics(
        true_negative=234, false_positive=0, false_negative=232, true_positive=0)
    assert corrected.precision is None
    assert corrected.precision_status == PRECISION_UNDEFINED
    # F1 is still defined and zero in exactly this situation.
    assert corrected.phase321_count_identity_f1 == 0.0


def test_undefined_precision_is_never_silently_replaced_by_zero():
    corrected = correct_threshold_metrics(
        true_negative=234, false_positive=0, false_negative=232, true_positive=0)
    # The failure guarded against is a float 0.0 standing in for "undefined",
    # so assert on type and value. `is not 0.0` compared object identity between
    # unrelated float objects, which CPython does not guarantee either way, so it
    # would not have caught the substitution it was written to catch.
    assert not (isinstance(corrected.precision, float) and corrected.precision == 0.0)
    assert corrected.precision is None


def test_no_alarm_rows_carry_the_threshold_transfer_annotation():
    silent = correct_threshold_metrics(
        true_negative=234, false_positive=0, false_negative=232, true_positive=0)
    assert silent.produced_no_positive_predictions is True
    assert silent.threshold_transfer_annotation == NO_ALARM_ANNOTATION
    assert silent.false_positive_rate == 0.0

    noisy = correct_threshold_metrics(
        true_negative=221, false_positive=13, false_negative=103, true_positive=129)
    assert noisy.produced_no_positive_predictions is False
    assert noisy.threshold_transfer_annotation is None
    assert produced_no_alarms(129, 13) is False


# -------------------------------------------------- E. family aggregation
def test_family_f1_is_the_arithmetic_mean_of_corrected_seed_values():
    counts = {s: {"true_positive": 129 + s, "false_positive": 81,
                  "false_negative": 27 - s, "true_negative": 75}
              for s in ISOLATION_FOREST_SEEDS}
    summary = correct_family_threshold_metrics("isolation_forest", counts)
    expected = sum(
        count_identity_f1(counts[s]["true_positive"], counts[s]["false_positive"],
                          counts[s]["false_negative"]) for s in ISOLATION_FOREST_SEEDS) / 5
    assert summary.family_f1 == pytest.approx(expected, abs=1e-12)
    assert summary.defined_precision_seed_count == 5


def test_family_f1_is_zero_when_every_seed_flagged_nothing():
    counts = {s: {"true_positive": 0, "false_positive": 0,
                  "false_negative": 232, "true_negative": 234}
              for s in ISOLATION_FOREST_SEEDS}
    summary = correct_family_threshold_metrics("isolation_forest", counts)
    assert summary.family_f1 == 0.0
    assert summary.family_precision is None
    assert summary.defined_precision_seed_count == 0
    assert "defined for only 0 of 5 seeds" in summary.family_precision_status


def test_family_precision_is_undefined_when_any_seed_is_undefined():
    """One defined seed must not be quietly promoted to the family value."""

    counts = {s: {"true_positive": 0, "false_positive": 0,
                  "false_negative": 232, "true_negative": 234}
              for s in ISOLATION_FOREST_SEEDS}
    counts[0] = {"true_positive": 0, "false_positive": 2,
                 "false_negative": 232, "true_negative": 232}
    summary = correct_family_threshold_metrics("isolation_forest", counts)
    assert summary.seed_precision[0] == 0.0
    assert summary.family_precision is None
    assert summary.defined_precision_seed_count == 1
    assert "defined for only 1 of 5 seeds" in summary.family_precision_status
    assert summary.family_f1 == 0.0


def test_family_correction_requires_the_complete_frozen_seed_set():
    counts = {s: {"true_positive": 1, "false_positive": 1,
                  "false_negative": 1, "true_negative": 1} for s in (0, 1, 2)}
    with pytest.raises(ValueError, match="seeds"):
        correct_family_threshold_metrics("isolation_forest", counts)


# ------------------------------------------------------------ D. prevalence
def test_prevalence_is_computed_from_the_exact_slice_class_counts():
    ctx = prevalence_context(anomaly_count=116, normal_count=20, average_precision=0.821318)
    assert ctx.total_count == 136
    assert ctx.prevalence == pytest.approx(116 / 136, abs=1e-12)
    assert ctx.prevalence == pytest.approx(0.8529411764705882, abs=1e-12)
    assert ctx.ap_minus_prevalence < 0
    assert ctx.status == "CONTEXT_ONLY"


def test_prevalence_context_rejects_an_empty_slice():
    with pytest.raises(ValueError):
        prevalence_context(anomaly_count=0, normal_count=0, average_precision=0.5)


# ------------------------------------------- structural no-model guarantee
def test_correction_module_declares_no_model_dependency():
    import src.experiment.reporting_correction as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    for forbidden in ("sklearn", "score_samples", "decision_function",
                      "ClassicalAnomalyBaseline", "src.models"):
        assert forbidden not in source.split("FORBIDDEN_MODEL_MODULES")[0], forbidden


def test_assert_no_model_dependency_raises_once_a_model_module_is_loaded():
    import src.models.classical  # noqa: F401 - deliberately imported to trip the guard

    with pytest.raises(RuntimeError, match="without model code"):
        assert_no_model_dependency()


# --------------------------------------------- C/G. immutability of evidence
def test_test_look_ledger_keeps_the_28_phase32_looks_as_an_immutable_prefix():
    """The 28 Phase-3.2 test looks must be immutable.

    The ledger is append-only, so the guarantee is that the original 28
    entries survive as a byte-identical *prefix* — not that the file never
    grows. A whole-file hash would conflate "Phase-3.2 evidence was edited"
    with "a later phase recorded a look", and only the first is a violation.

    Stage note: Single-Modality Benchmarking also records Protocol-B looks, so
    the Phase-3.2 looks are identified by their batch and evaluator SHA, not by
    the ``protocol`` value.
    """
    history = verify_ledger_history(LEDGER)
    assert hashlib.sha256(b"".join(history.lines[:28])).hexdigest() == LEDGER_SHA256

    batch = history.segment("phase3_2_first_common_test")
    assert (batch.start, batch.end) == (0, 28)
    assert batch.batch_id == "p32-test-batch-8f7d48324fd10df97ae29100"
    phase32 = [e for e in history.rows if e["git_sha"] == batch.git_sha]
    assert phase32 == list(history.rows[:28])
    assert all(e["protocol"] == "mechasight_experimental_protocol" for e in phase32)


def test_every_original_phase32_artifact_hash_still_matches_its_manifest():
    manifest = json.loads((PHASE32 / "artifact_manifest.json").read_text(encoding="utf-8"))
    checked = 0
    for name, expected in manifest["files"].items():
        path = Path(name) if Path(name).exists() else PHASE32 / name
        assert path.exists(), name
        assert _sha256(path) == expected, name
        checked += 1
    assert checked == 155


def test_frozen_policy_and_split_manifest_are_untouched():
    assert _sha256(Path("configs/phase3/first_test_reporting_policy_v1.json")) == \
        PARENT_POLICY_SHA256
    assert _sha256(Path("reports/phase2_2/protocol_b_few_shot_adaptation_v3.csv")) == \
        FROZEN_SPLIT_SHA256


def test_correction_writes_nothing_into_the_phase32_directory():
    corrected = {p.name for p in PHASE321.iterdir() if p.is_file()}
    original = {p.name for p in PHASE32.iterdir() if p.is_file()}
    assert corrected and original
    assert PHASE321.resolve() != PHASE32.resolve()
    # the corrected layer must not shadow an original filename in place
    assert not (PHASE32 / "primary_baselines_corrected.csv").exists()


# ------------------------------------------ F. traceability of corrected rows
def test_corrected_rows_preserve_original_experiment_identities():
    original = pd.read_csv(PHASE32 / "individual_experiment_metrics.csv")
    corrected = pd.read_csv(PHASE321 / "threshold_metrics_corrected_individual.csv")
    assert len(corrected) == len(original) == 28
    assert set(corrected.experiment_id) == set(original.experiment_id)
    merged = original.merge(corrected, on="experiment_id", suffixes=("_o", "_c"))
    for column in ("true_negative", "false_positive", "false_negative", "true_positive"):
        assert (merged[f"{column}_o"] == merged[f"{column}_c"]).all(), column
    assert merged.threshold_o.equals(merged.threshold_c)
    for column in ("reporting_correction_id", "test_cohort_hash",
                   "threshold_source_experiment_id", "source_phase32_artifact"):
        assert corrected[column].notna().all()
        assert (corrected[column].astype(str) != "").all()


def test_corrected_f1_equals_zero_for_every_previously_null_row():
    corrected = pd.read_csv(PHASE321 / "threshold_metrics_corrected_individual.csv")
    affected = corrected[corrected.f1_corrected_from_null]
    assert len(affected) == 10
    assert set(affected.machine) == {"RoboticArm"}
    assert set(affected.model_family) == {"isolation_forest"}
    assert (affected.phase321_count_identity_f1 == 0.0).all()
    assert affected.original_phase32_f1.isna().all()
    assert (affected.true_positive == 0).all()


def test_previously_defined_f1_values_are_left_exactly_as_published():
    corrected = pd.read_csv(PHASE321 / "threshold_metrics_corrected_individual.csv")
    unchanged = corrected[~corrected.f1_corrected_from_null]
    assert len(unchanged) == 18
    assert (unchanged.original_phase32_f1 -
            unchanged.phase321_count_identity_f1).abs().max() < 1e-12


def test_every_average_precision_row_carries_its_slice_prevalence():
    slices = pd.read_csv(PHASE321 / "domain_slice_metrics_with_prevalence.csv")
    assert slices.anomaly_prevalence.notna().all()
    assert (slices.normal_count + slices.anomaly_count == slices.sample_count).all()
    import numpy as np
    assert np.allclose(slices.ap_minus_prevalence.to_numpy(),
                       (slices.average_precision - slices.anomaly_prevalence).to_numpy(),
                       atol=1e-12)
    ra_target = slices[(slices.machine == "RoboticArm") & (slices.domain == "target")]
    assert (ra_target.normal_count == 20).all()
    assert (ra_target.anomaly_count == 116).all()
    assert ra_target.anomaly_prevalence.round(4).eq(0.8529).all()
    primary = pd.read_csv(PHASE321 / "primary_baselines_corrected.csv")
    assert primary.anomaly_prevalence.notna().all()


def test_healthy_target_context_exposes_cluster_counts_without_new_intervals():
    ht = pd.read_csv(PHASE321 / "healthy_target_false_positives_context.csv")
    assert (~ht.confidence_interval_reported).all()
    assert ht.values_unchanged_from_phase32.all()
    ra = ht[ht.machine == "RoboticArm"]
    assert (ra.healthy_target_sample_count == 20).all()
    assert (ra.healthy_target_parent_recordings == 6).all()
    bm = ht[ht.machine == "BrushlessMotor"]
    assert (bm.healthy_target_sample_count == 78).all()
    assert (bm.healthy_target_parent_recordings == 78).all()
    assert ht.uncertainty_warning.str.len().gt(0).all()
    original = pd.read_csv(PHASE32 / "healthy_target_false_positives_individual.csv")
    ind = ht[ht.level == "individual"].merge(original, on="experiment_id",
                                             suffixes=("_c", "_o"))
    assert (ind.false_positive_c == ind.false_positive_o).all()


def test_correction_policy_identity_is_deterministic_and_declares_its_parent():
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    identity = reporting_correction_identity(policy)
    assert identity == reporting_correction_identity(policy)
    assert identity.startswith("p321-correction-")
    assert policy["parent_policy"]["policy_id"] == "p311-reporting-5e76ab7dabd5689c7b0ced68"
    assert policy["immutability"]["test_look_ledger_entries_before"] == 28
    assert policy["immutability"]["test_look_ledger_entries_after"] == 28
    assert policy["immutability"]["new_test_look"] is False
    manifest = json.loads((PHASE321 / "correction_manifest.json").read_text(encoding="utf-8"))
    assert manifest["reporting_correction_id"] == identity
    assert manifest["no_model_trained_refit_or_rescored"] is True
    assert manifest["no_new_test_look"] is True


def test_correction_policy_rejects_a_foreign_parent_or_schema():
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    wrong_parent = {**policy, "parent_policy": {"policy_id": "p311-reporting-somethingelse"}}
    with pytest.raises(ValueError, match="parent policy"):
        reporting_correction_identity(wrong_parent)
    with pytest.raises(ValueError, match="schema"):
        reporting_correction_identity({**policy, "schema_version": "made-up"})
