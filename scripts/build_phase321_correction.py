"""Build the Phase-3.2.1 corrected reporting layer.

Derives every corrected value from already-persisted Phase-3.2 tables, frozen
experiment provenance and the frozen v3 split manifest.  It never fits, scores
or loads a model, and it never writes into ``reports/phase3_2/``.

    python scripts/build_phase321_correction.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment.provenance import canonical_json, file_sha256  # noqa: E402
from src.experiment.reporting_correction import (  # noqa: E402
    ISOLATION_FOREST_SEEDS,
    NO_ALARM_ANNOTATION,
    PRECISION_UNDEFINED,
    assert_no_model_dependency,
    correct_family_threshold_metrics,
    correct_threshold_metrics,
    prevalence_context,
    reporting_correction_identity,
)

PHASE32 = Path("reports/phase3_2")
SPLIT_MANIFEST = Path("reports/phase2_2/protocol_b_few_shot_adaptation_v3.csv")
POLICY_PATH = Path("configs/phase3/phase321_reporting_correction_policy_v1.json")

DETERMINISTIC_FAMILIES = ("statistical_reference", "one_class_svm")


def _limitation(machine: str) -> str:
    return ("robotic_arm_parent_clustered_inference_v1" if machine == "RoboticArm"
            else "brushless_motor_acquisition_time_class_confound_v1")


def _cohort_counts(manifest: pd.DataFrame) -> dict:
    """Segment and independent-parent counts for every reported cohort."""

    out: dict[tuple, dict] = {}
    for (machine, regime), part in manifest.groupby(["machine", "regime"]):
        test = part[part.data_pool == "common_evaluation"].copy()
        test["is_anomaly"] = test.anomaly_label.astype(str).str.lower() != "normal"
        val = part[part.data_pool == "source_validation"]
        adapt = part[part.data_pool == "target_adaptation"]
        blocked = part[part.data_pool == "adaptation_sibling_blocked"]
        healthy = test[(test.file_domain == "target") & (~test.is_anomaly)]
        entry = {
            "common_evaluation_segments": int(len(test)),
            "common_evaluation_parent_recordings": int(test.parent_recording.nunique()),
            "source_validation_segments": int(len(val)),
            "source_validation_parent_recordings": int(val.parent_recording.nunique()),
            "target_adaptation_segments": int(len(adapt)),
            "target_adaptation_parent_recordings": int(adapt.parent_recording.nunique()),
            "adaptation_sibling_blocked_segments": int(len(blocked)),
            "healthy_target_segments": int(len(healthy)),
            "healthy_target_parent_recordings": int(healthy.parent_recording.nunique()),
        }
        for dom in ("source", "target"):
            sl = test[test.file_domain == dom]
            entry[f"{dom}_slice_segments"] = int(len(sl))
            entry[f"{dom}_slice_anomaly"] = int(sl.is_anomaly.sum())
            entry[f"{dom}_slice_normal"] = int((~sl.is_anomaly).sum())
            entry[f"{dom}_slice_parent_recordings"] = int(sl.parent_recording.nunique())
        entry["complete_anomaly"] = int(test.is_anomaly.sum())
        entry["complete_normal"] = int((~test.is_anomaly).sum())
        out[(machine, regime)] = entry
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("reports/phase3_2_1"))
    args = parser.parse_args()

    # Structural guarantee, checked before any artifact is produced.
    assert_no_model_dependency()

    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    correction_id = reporting_correction_identity(policy)
    policy_sha = file_sha256(POLICY_PATH)

    individual = pd.read_csv(PHASE32 / "individual_experiment_metrics.csv")
    primary = pd.read_csv(PHASE32 / "primary_baselines.csv")
    slice_ind = pd.read_csv(PHASE32 / "domain_slice_metrics_individual.csv")
    slice_fam = pd.read_csv(PHASE32 / "domain_slice_metrics_family.csv")
    healthy_ind = pd.read_csv(PHASE32 / "healthy_target_false_positives_individual.csv")
    healthy_fam = pd.read_csv(PHASE32 / "healthy_target_false_positives_family.csv")
    manifest = pd.read_csv(SPLIT_MANIFEST)
    cohorts = _cohort_counts(manifest)

    provenance = {}
    for d in sorted((PHASE32 / "experiments").iterdir()):
        if not d.is_dir():
            continue
        prov = json.loads((d / "provenance.json").read_text(encoding="utf-8"))
        provenance[prov["experiment_id"]] = prov

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------- corrected individual rows
    rows = []
    for _, r in individual.iterrows():
        prov = provenance[r.experiment_id]
        tp_ = prov["threshold_provenance"]
        c = correct_threshold_metrics(
            true_negative=r.true_negative, false_positive=r.false_positive,
            false_negative=r.false_negative, true_positive=r.true_positive,
        )
        rows.append({
            "reporting_correction_id": correction_id,
            "experiment_id": r.experiment_id,
            "machine": r.machine, "regime": r.regime,
            "model_family": r.model_family, "model_seed": r.model_seed,
            "scientific_limitation_id": r.scientific_limitation_id,
            "source_phase32_artifact": "reports/phase3_2/individual_experiment_metrics.csv",
            "test_cohort_hash": prov["test_cohort_hash"],
            "threshold": r.threshold,
            "threshold_source_experiment_id": tp_["source_experiment_id"],
            "true_negative": c.true_negative, "false_positive": c.false_positive,
            "false_negative": c.false_negative, "true_positive": c.true_positive,
            "false_positive_rate": c.false_positive_rate,
            "recall": c.recall,
            "precision": c.precision,
            "precision_status": c.precision_status,
            "original_phase32_f1": r.f1,
            "phase321_count_identity_f1": c.phase321_count_identity_f1,
            "f1_convention": c.f1_convention,
            "f1_corrected_from_null": bool(pd.isna(r.f1)
                                           and c.phase321_count_identity_f1 is not None),
            "produced_no_positive_predictions": c.produced_no_positive_predictions,
            "threshold_transfer_annotation": c.threshold_transfer_annotation,
        })
    corrected_individual = pd.DataFrame(rows).sort_values(
        ["machine", "regime", "model_family", "model_seed"]).reset_index(drop=True)
    corrected_individual.to_csv(out / "threshold_metrics_corrected_individual.csv", index=False)

    # ------------------------------------------------- corrected family rows
    fam_rows = []
    for (machine, regime, family), part in individual.groupby(
            ["machine", "regime", "model_family"]):
        seed_counts = {
            int(r.model_seed): {
                "true_positive": int(r.true_positive),
                "false_positive": int(r.false_positive),
                "false_negative": int(r.false_negative),
                "true_negative": int(r.true_negative),
            } for _, r in part.iterrows()
        }
        summary = correct_family_threshold_metrics(family, seed_counts)
        orig = primary[(primary.machine == machine) & (primary.regime == regime) &
                       (primary.model_family == family)].iloc[0]
        cc = cohorts[(machine, regime)]
        no_alarm = all(
            seed_counts[s]["true_positive"] == 0 and seed_counts[s]["false_positive"] == 0
            for s in seed_counts)
        fam_rows.append({
            "reporting_correction_id": correction_id,
            "machine": machine, "regime": regime, "model_family": family,
            "scientific_limitation_id": _limitation(machine),
            "source_phase32_artifact": "reports/phase3_2/primary_baselines.csv",
            "seed_count": summary.expected_seed_count,
            "mean_true_negative": orig.true_negative, "mean_false_positive": orig.false_positive,
            "mean_false_negative": orig.false_negative, "mean_true_positive": orig.true_positive,
            "false_positive_rate": orig.false_positive_rate, "recall": orig.recall,
            "original_phase32_precision": orig.precision,
            "family_precision": summary.family_precision,
            "family_precision_status": summary.family_precision_status,
            "defined_precision_seed_count": summary.defined_precision_seed_count,
            "original_phase32_f1": orig.f1,
            "phase321_count_identity_f1": summary.family_f1,
            "f1_convention": "arithmetic mean of the per-seed 2*TP/(2*TP+FP+FN) values",
            "f1_corrected_from_null": bool(pd.isna(orig.f1) and summary.family_f1 is not None),
            "per_seed_f1": canonical_json({str(k): v for k, v in summary.seed_f1.items()}),
            "per_seed_precision": canonical_json(
                {str(k): v for k, v in summary.seed_precision.items()}),
            "all_seeds_produced_no_positive_predictions": no_alarm,
            "threshold_transfer_annotation": NO_ALARM_ANNOTATION if no_alarm else None,
            "common_evaluation_segments": cc["common_evaluation_segments"],
            "common_evaluation_parent_recordings": cc["common_evaluation_parent_recordings"],
        })
    corrected_family = pd.DataFrame(fam_rows).sort_values(
        ["machine", "regime", "model_family"]).reset_index(drop=True)
    corrected_family.to_csv(
        out / "isolation_forest_family_threshold_metrics_corrected.csv", index=False)

    # ------------------------------------------------- corrected primary table
    prim_rows = []
    for _, r in primary.iterrows():
        cf = corrected_family[(corrected_family.machine == r.machine) &
                              (corrected_family.regime == r.regime) &
                              (corrected_family.model_family == r.model_family)].iloc[0]
        cc = cohorts[(r.machine, r.regime)]
        pc = prevalence_context(anomaly_count=cc["complete_anomaly"],
                                normal_count=cc["complete_normal"],
                                average_precision=r.average_precision)
        prim_rows.append({
            "reporting_correction_id": correction_id,
            "machine": r.machine, "regime": r.regime, "model_family": r.model_family,
            "scientific_limitation_id": r.scientific_limitation_id,
            "source_phase32_artifact": "reports/phase3_2/primary_baselines.csv",
            "auroc": r.auroc, "auroc_ci_lower": r.auroc_ci_lower,
            "auroc_ci_upper": r.auroc_ci_upper,
            "average_precision": r.average_precision,
            "average_precision_ci_lower": r.average_precision_ci_lower,
            "average_precision_ci_upper": r.average_precision_ci_upper,
            "anomaly_prevalence": pc.prevalence,
            "ap_minus_prevalence": pc.ap_minus_prevalence,
            "ap_prevalence_status": pc.status,
            "standardized_pauroc_fpr_0_10": r.standardized_pauroc_fpr_0_10,
            "standardized_pauroc_fpr_0_10_ci_lower": r.standardized_pauroc_fpr_0_10_ci_lower,
            "standardized_pauroc_fpr_0_10_ci_upper": r.standardized_pauroc_fpr_0_10_ci_upper,
            "true_negative": r.true_negative, "false_positive": r.false_positive,
            "false_negative": r.false_negative, "true_positive": r.true_positive,
            "false_positive_rate": r.false_positive_rate, "recall": r.recall,
            "original_phase32_precision": r.precision,
            "precision": cf.family_precision,
            "precision_status": cf.family_precision_status,
            "defined_precision_seed_count": cf.defined_precision_seed_count,
            "original_phase32_f1": r.f1,
            "phase321_count_identity_f1": cf.phase321_count_identity_f1,
            "f1_corrected_from_null": cf.f1_corrected_from_null,
            "produced_no_positive_predictions": cf.all_seeds_produced_no_positive_predictions,
            "threshold_transfer_annotation": cf.threshold_transfer_annotation,
            "common_evaluation_segments": cc["common_evaluation_segments"],
            "common_evaluation_parent_recordings": cc["common_evaluation_parent_recordings"],
            "source_validation_segments": cc["source_validation_segments"],
            "source_validation_parent_recordings": cc["source_validation_parent_recordings"],
        })
    corrected_primary = pd.DataFrame(prim_rows)
    corrected_primary.to_csv(out / "primary_baselines_corrected.csv", index=False)

    # --------------------------------------------- slice metrics + prevalence
    slice_rows = []
    for source_table, level in ((slice_fam, "family"), (slice_ind, "individual")):
        for _, r in source_table.iterrows():
            cc = cohorts[(r.machine, r.regime)]
            pc = prevalence_context(
                anomaly_count=cc[f"{r.domain}_slice_anomaly"],
                normal_count=cc[f"{r.domain}_slice_normal"],
                average_precision=r.average_precision)
            if pc.total_count != int(r.sample_count):
                raise ValueError(
                    f"slice composition disagrees with the recorded sample_count for "
                    f"{r.machine}/{r.regime}/{r.domain}")
            slice_rows.append({
                "reporting_correction_id": correction_id,
                "level": level,
                "experiment_id": getattr(r, "experiment_id", None),
                "machine": r.machine, "regime": r.regime,
                "model_family": r.model_family,
                "model_seed": getattr(r, "model_seed", None),
                "domain": r.domain,
                "scientific_limitation_id": r.scientific_limitation_id,
                "source_phase32_artifact": (
                    f"reports/phase3_2/domain_slice_metrics_{level}.csv"),
                "sample_count": pc.total_count,
                "normal_count": pc.normal_count, "anomaly_count": pc.anomaly_count,
                "parent_recordings": cc[f"{r.domain}_slice_parent_recordings"],
                "auroc": r.auroc,
                "average_precision": r.average_precision,
                "anomaly_prevalence": pc.prevalence,
                "ap_minus_prevalence": pc.ap_minus_prevalence,
                "ap_at_or_below_prevalence": pc.ap_minus_prevalence <= 0,
                "ap_prevalence_status": pc.status,
                "standardized_pauroc_fpr_0_10": r.standardized_pauroc_fpr_0_10,
            })
    slices_corrected = pd.DataFrame(slice_rows)
    slices_corrected.to_csv(out / "domain_slice_metrics_with_prevalence.csv", index=False)

    # ------------------------------------------------ healthy-target context
    ht_rows = []
    for source_table, level in ((healthy_fam, "family"), (healthy_ind, "individual")):
        for _, r in source_table.iterrows():
            cc = cohorts[(r.machine, r.regime)]
            ht_rows.append({
                "reporting_correction_id": correction_id,
                "level": level,
                "experiment_id": getattr(r, "experiment_id", None),
                "machine": r.machine, "regime": r.regime,
                "model_family": r.model_family,
                "model_seed": getattr(r, "model_seed", None),
                "scientific_limitation_id": r.scientific_limitation_id,
                "source_phase32_artifact": (
                    f"reports/phase3_2/healthy_target_false_positives_{level}.csv"),
                "healthy_target_sample_count": r.healthy_target_sample_count,
                "healthy_target_parent_recordings": cc["healthy_target_parent_recordings"],
                "false_positive": r.false_positive,
                "false_positive_rate": r.false_positive_rate,
                "values_unchanged_from_phase32": True,
                "confidence_interval_reported": False,
                "uncertainty_warning": (
                    f"{int(r.healthy_target_sample_count)} healthy target-domain segments "
                    f"come from only {cc['healthy_target_parent_recordings']} independent "
                    "physical parent recordings. No interval is reported for this rate under "
                    "the frozen policy; treat the direction as informative and the magnitude "
                    "as imprecise."),
            })
    healthy_corrected = pd.DataFrame(ht_rows)
    healthy_corrected.to_csv(out / "healthy_target_false_positives_context.csv", index=False)

    # ------------------------------------------------------------- manifest
    corrected_files = sorted(p.name for p in out.iterdir() if p.is_file())
    source_files = sorted(
        str(p).replace("\\", "/") for p in PHASE32.iterdir() if p.is_file())
    manifest_doc = {
        "schema_version": "phase3.2.1-reporting-correction-v1",
        "reporting_correction_id": correction_id,
        "reporting_correction_policy": str(POLICY_PATH).replace("\\", "/"),
        "reporting_correction_policy_sha256": policy_sha,
        "parent_reporting_policy_id": policy["parent_policy"]["policy_id"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "no_model_trained_refit_or_rescored": True,
        "no_new_experiment_id": True,
        "no_new_test_look": True,
        "test_look_ledger_entries": 28,
        "source_phase32_files": {
            f: file_sha256(f) for f in source_files},
        "frozen_split_manifest": {
            "path": str(SPLIT_MANIFEST).replace("\\", "/"),
            "sha256": file_sha256(SPLIT_MANIFEST)},
        "corrected_files": {
            f: file_sha256(out / f) for f in corrected_files
            if f != "correction_manifest.json"},
        "cohort_counts": {f"{m}|{r}": v for (m, r), v in cohorts.items()},
    }
    (out / "correction_manifest.json").write_text(
        json.dumps(manifest_doc, indent=2, sort_keys=True), encoding="utf-8")

    summary = {
        "reporting_correction_id": correction_id,
        "rows_with_f1_corrected_from_null": int(
            corrected_individual.f1_corrected_from_null.sum()),
        "family_rows_with_f1_corrected_from_null": int(
            corrected_family.f1_corrected_from_null.sum()),
        "rows_with_undefined_precision": int(
            (corrected_individual.precision_status != "defined").sum()),
        "rows_with_no_positive_predictions": int(
            corrected_individual.produced_no_positive_predictions.sum()),
        "slice_rows_at_or_below_prevalence": int(
            slices_corrected.ap_at_or_below_prevalence.sum()),
    }
    print(json.dumps(summary, indent=2))
    print("\nwrote", out)
    for f in corrected_files:
        print("  ", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
