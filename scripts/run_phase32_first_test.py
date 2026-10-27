"""Run the first locked Protocol-B common-test evaluation exactly once."""

from __future__ import annotations

from dataclasses import asdict
import argparse
from importlib.metadata import version as package_version
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.reporting import (  # noqa: E402
    reporting_policy_identity,
    summarize_seed_metrics,
)
from src.experiment.phase32 import (  # noqa: E402
    PHASE32_SCHEMA_VERSION,
    REPORTING_POLICY_ID,
    CommonTestCohortContract,
    authorize_first_test_batch,
    evaluate_authorized_experiment,
    family_bootstrap_distribution,
    load_common_test_artifact,
    make_common_bootstrap_draws,
    paired_distribution_difference,
    prepare_common_test_artifact,
    reproduce_frozen_experiment,
    score_frame,
)
from src.experiment.ledger import TestLookLedger  # noqa: E402
from src.experiment.metrics import RANKING_METRICS  # noqa: E402
from src.experiment.provenance import canonical_json, file_sha256  # noqa: E402
from src.experiment.store import ResultStore  # noqa: E402
from src.preprocessing.config import load_config  # noqa: E402
from src.preprocessing.protected_access import authorized_protected_read  # noqa: E402


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": package_version("numpy"),
        "pandas": package_version("pandas"),
        "scikit-learn": package_version("scikit-learn"),
        "scipy": package_version("scipy"),
        "joblib": package_version("joblib"),
    }


def _family_groups(evaluated: dict) -> dict[tuple[str, str, str], dict[int, pd.DataFrame]]:
    groups: dict[tuple[str, str, str], dict[int, pd.DataFrame]] = {}
    for item in evaluated.values():
        config = item.reproduced.config
        groups.setdefault((config.machine, config.regime, config.model_family), {})[
            config.model_seed
        ] = score_frame(item)
    return groups


def _value_summary(family: str, values: dict[int, float | int | None]) -> dict[str, Any]:
    if any(value is None for value in values.values()):
        return {
            "status": "undefined_for_at_least_one_seed",
            "seed_values": values,
            "mean": None, "population_std": None, "minimum": None, "maximum": None,
        }
    summary = summarize_seed_metrics(family, {seed: float(value) for seed, value in values.items()})
    return {
        "status": "defined",
        "seed_values": dict(summary.seed_values),
        "mean": summary.arithmetic_mean,
        "population_std": summary.population_standard_deviation,
        "minimum": summary.minimum,
        "maximum": summary.maximum,
    }


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")


def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "UNDEFINED"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def _limitation_id(machine: str) -> str:
    return (
        "brushless_motor_acquisition_time_class_confound_v1"
        if machine == "BrushlessMotor"
        else "robotic_arm_parent_clustered_inference_v1"
    )


def _markdown_table(
    title: str,
    rows: list[dict[str, Any]],
    columns: tuple[str, ...],
) -> list[str]:
    return [
        f"## {title}",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
        *(
            "| " + " | ".join(_fmt(row.get(column)) for column in columns) + " |"
            for row in rows
        ),
    ]


def _primary_markdown(rows: list[dict], machine: str) -> list[str]:
    table_letter = "A" if machine == "RoboticArm" else "B"
    lines = [
        f"## Table {table_letter} — {machine} primary baselines",
        "",
        "| Regime | Model | AUROC (95% CI) | AP (95% CI) | pAUROC@0.10 (95% CI) | FPR | Recall | Precision | F1 |",
        "|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["machine"] != machine:
            continue
        lines.append(
            f"| {row['regime']} | {row['model_family']} | "
            f"{_fmt(row['auroc'])} [{_fmt(row['auroc_ci_lower'])}, {_fmt(row['auroc_ci_upper'])}] | "
            f"{_fmt(row['average_precision'])} [{_fmt(row['average_precision_ci_lower'])}, {_fmt(row['average_precision_ci_upper'])}] | "
            f"{_fmt(row['standardized_pauroc_fpr_0_10'])} [{_fmt(row['standardized_pauroc_fpr_0_10_ci_lower'])}, {_fmt(row['standardized_pauroc_fpr_0_10_ci_upper'])}] | "
            f"{_fmt(row['false_positive_rate'])} | {_fmt(row['recall'])} | "
            f"{_fmt(row['precision'])} | {_fmt(row['f1'])} |"
        )
    return lines


def _delta_markdown(rows: list[dict[str, Any]]) -> list[str]:
    indexed = {
        (row["machine"], row["model_family"], row["metric"]): row
        for row in rows
    }
    lines = [
        "## Table C — S vs T deltas (T minus S)",
        "",
        "| Machine | Model | ΔAUROC (95% CI) | ΔAP (95% CI) | ΔpAUROC@0.10 (95% CI) |",
        "|---|---|---|---|---|",
    ]
    for machine in ("RoboticArm", "BrushlessMotor"):
        for family in ("statistical_reference", "isolation_forest", "one_class_svm"):
            cells = []
            for metric in RANKING_METRICS:
                row = indexed[(machine, family, metric)]
                cells.append(
                    f"{_fmt(row['point_delta'])} "
                    f"[{_fmt(row['ci_lower'])}, {_fmt(row['ci_upper'])}]"
                )
            lines.append(f"| {machine} | {family} | " + " | ".join(cells) + " |")
    return lines


def _comparison_markdown(rows: list[dict[str, Any]]) -> list[str]:
    indexed = {
        (row["machine"], row["regime"], row["comparison"], row["metric"]): row
        for row in rows
    }
    lines = [
        "## Table E — Paired model-family differences (first minus second)",
        "",
        "| Machine | Regime | Comparison | ΔAUROC (95% CI) | ΔAP (95% CI) | ΔpAUROC@0.10 (95% CI) |",
        "|---|---|---|---|---|---|",
    ]
    comparisons = (
        "statistical_reference_minus_isolation_forest",
        "statistical_reference_minus_one_class_svm",
        "isolation_forest_minus_one_class_svm",
    )
    for machine in ("RoboticArm", "BrushlessMotor"):
        for regime in ("S", "T"):
            for comparison in comparisons:
                cells = []
                for metric in RANKING_METRICS:
                    row = indexed[(machine, regime, comparison, metric)]
                    cells.append(
                        f"{_fmt(row['point_delta'])} "
                        f"[{_fmt(row['ci_lower'])}, {_fmt(row['ci_upper'])}]"
                    )
                lines.append(
                    f"| {machine} | {regime} | {comparison} | "
                    + " | ".join(cells) + " |"
                )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/extracted"))
    parser.add_argument(
        "--phase31-features", type=Path, default=Path("data/interim/phase31_primary")
    )
    parser.add_argument(
        "--test-features", type=Path, default=Path("data/interim/phase32_common_test")
    )
    parser.add_argument(
        "--split-manifest", type=Path,
        default=Path("reports/phase2_2/protocol_b_few_shot_adaptation_v3.csv"),
    )
    parser.add_argument(
        "--phase31-report", type=Path,
        default=Path("reports/phase3_1/phase31_pretest_freeze_report.json"),
    )
    parser.add_argument(
        "--reporting-policy", type=Path,
        default=Path("configs/phase3/first_test_reporting_policy_v1.json"),
    )
    parser.add_argument("--out", type=Path, default=Path("reports/phase3_2"))
    parser.add_argument(
        "--ledger", type=Path,
        default=Path("reports/experiment_results/test_look_ledger.jsonl"),
    )
    args = parser.parse_args()

    code_sha = _git("rev-parse", "HEAD")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Phase-3.2 test evaluation requires a clean tracked source tree")
    if args.out.exists() or args.test_features.exists():
        raise FileExistsError("Phase-3.2 output/test-feature namespace already exists")
    policy = json.loads(args.reporting_policy.read_text(encoding="utf-8"))
    if reporting_policy_identity(policy) != REPORTING_POLICY_ID:
        raise ValueError("frozen first-test reporting policy identity mismatch")
    ledger = TestLookLedger(args.ledger)
    if len(ledger.read()) != 0:
        raise RuntimeError("first common-test evaluation requires ledger count zero")
    phase31 = json.loads(args.phase31_report.read_text(encoding="utf-8"))
    expected_ids = frozenset(item["experiment_id"] for item in phase31["experiments"])
    if len(expected_ids) != 28:
        raise ValueError("Phase-3.1 report does not contain 28 unique experiments")

    args.out.mkdir(parents=True)
    state = {
        "schema_version": PHASE32_SCHEMA_VERSION,
        "state": "pretest_reproduction_started",
        "test_evaluation_code_sha": code_sha,
        "reporting_policy_id": REPORTING_POLICY_ID,
        "expected_experiment_count": 28,
        "expected_ledger_ending_count": 28,
        "common_test_scores_completed": 0,
    }
    _write_json(args.out / "evaluation_state.json", state)

    reproduced = {}
    reproduction_rows = []
    for entry in phase31["experiments"]:
        regime = entry["regime"]
        item = reproduce_frozen_experiment(
            bundle=entry["bundle"],
            feature_root=args.phase31_features,
            split_manifest_path=args.split_manifest,
            preprocessing_config_path=Path(
                f"configs/preprocessing/mechasight_few_shot_regime_{regime.lower()}.json"
            ),
        )
        reproduced[item.config.experiment_id] = item
        reproduction_rows.append({
            "experiment_id": item.config.experiment_id,
            "machine": item.config.machine,
            "regime": item.config.regime,
            "model_family": item.config.model_family,
            "model_seed": item.config.model_seed,
            "validation_exact_match": item.validation_exact_match,
            "training_cohort_hash": item.training_cohort_hash,
            "validation_cohort_hash": item.validation_cohort_hash,
            "normalizer_identity": item.normalizer_identity,
            "threshold": item.threshold.value,
            "scientific_limitation_id": _limitation_id(item.config.machine),
        })
    if frozenset(reproduced) != expected_ids:
        raise RuntimeError("pre-test reproduction did not yield the exact frozen 28-ID set")
    _write_json(args.out / "pretest_reproduction_check.json", {
        "status": "PASS_28_OF_28_EXACT",
        "experiments": reproduction_rows,
    })
    state["state"] = "pretest_reproduction_passed_28_of_28"
    _write_json(args.out / "evaluation_state.json", state)

    # Phase 3.2 opened its common-test features before its ledger batch existed;
    # the authority is the frozen first-test evaluation at this code SHA.
    with authorized_protected_read(
        authority=f"phase3.2-first-locked-test@{code_sha}",
        purpose="first locked test: materialise and verify common-test features",
    ):
        for regime in ("S", "T"):
            config_path = Path(
                f"configs/preprocessing/mechasight_few_shot_regime_{regime.lower()}.json"
            )
            for machine in ("RoboticArm", "BrushlessMotor"):
                metadata = prepare_common_test_artifact(
                    data_root=args.data_root,
                    split_manifest_path=args.split_manifest,
                    preprocessing_config_path=config_path,
                    artifact_root=args.test_features,
                    machine=machine,
                    test_evaluation_code_sha=code_sha,
                )
                print(canonical_json(metadata), flush=True)

        assignments = pd.read_csv(args.split_manifest)
        test_data = {}
        cohort_rows = []
        for regime in ("S", "T"):
            config = load_config(
                f"configs/preprocessing/mechasight_few_shot_regime_{regime.lower()}.json"
            )
            for machine in ("RoboticArm", "BrushlessMotor"):
                contract = CommonTestCohortContract.from_frozen_assignments(
                    assignments, config, machine=machine
                )
                _frame, loaded, metadata = load_common_test_artifact(
                    args.test_features, machine=machine, regime=regime, contract=contract
                )
                test_data[(machine, regime)] = loaded
                cohort_rows.append({
                    "machine": machine, "regime": regime,
                    "segment_count": len(loaded.segment_ids),
                    "cohort_hash": loaded.cohort_hash,
                    "frozen_manifest_contract_hash": contract.frozen_manifest_hash,
                    "feature_artifact_hash": metadata["artifact_sha256"],
                    "scientific_limitation_id": _limitation_id(machine),
                })
    for machine in ("RoboticArm", "BrushlessMotor"):
        if test_data[(machine, "S")].cohort_hash != test_data[(machine, "T")].cohort_hash:
            raise RuntimeError(f"{machine} S/T common-test identities differ")
    _write_json(args.out / "common_test_cohort_verification.json", {
        "status": "PASS", "cohorts": cohort_rows, "s_t_identities_equal": True,
    })
    state["state"] = "common_test_features_verified_no_model_scores"
    _write_json(args.out / "evaluation_state.json", state)

    authorization = authorize_first_test_batch(
        ledger=ledger,
        reproduced=reproduced,
        expected_experiment_ids=expected_ids,
        test_cohort_hashes_by_machine={
            machine: test_data[(machine, "S")].cohort_hash
            for machine in ("RoboticArm", "BrushlessMotor")
        },
        test_evaluation_code_sha=code_sha,
        expected_ledger_start=0,
    )
    state.update({
        "state": "ledger_opened_before_scoring",
        "test_batch_id": authorization.batch_id,
        "ledger_start_count": authorization.ledger_start_count,
        "ledger_end_count": authorization.ledger_end_count,
    })
    _write_json(args.out / "evaluation_state.json", state)

    evaluated = {}
    experiment_store = ResultStore(args.out / "experiments")
    for experiment_id, item in sorted(reproduced.items()):
        data = test_data[(item.config.machine, item.config.regime)]
        result = evaluate_authorized_experiment(item, data, authorization)
        experiment_store.write_bundle(
            config=item.config,
            provenance={
                "test_cohort_hash": data.cohort_hash,
                "score_direction": item.config.anomaly_score_direction,
                "threshold_provenance": asdict(item.threshold),
                "phase32_schema_version": PHASE32_SCHEMA_VERSION,
                "test_evaluation_code_sha": code_sha,
                "reporting_policy_id": REPORTING_POLICY_ID,
                "test_batch_id": authorization.batch_id,
                "ledger_entry_semantics": "one of exactly 28 atomic batch entries written before scoring",
                "training_cohort_hash": item.training_cohort_hash,
                "validation_cohort_hash": item.validation_cohort_hash,
                "normalizer_identity": item.normalizer_identity,
                "scientific_limitation_id": _limitation_id(item.config.machine),
            },
            metrics={
                "complete_common_test": result.complete_metrics,
                "domain_slices": result.domain_metrics,
                "healthy_target_false_positive": result.healthy_target,
            },
            scores=result.score_table,
        )
        evaluated[experiment_id] = result
        state["common_test_scores_completed"] = len(evaluated)
        state["state"] = "test_scoring_in_progress"
        _write_json(args.out / "evaluation_state.json", state)
    if len(evaluated) != 28 or len(ledger.read()) != 28:
        raise RuntimeError("Phase-3.2 individual evaluation/ledger count mismatch")

    individual_rows = []
    domain_rows = []
    healthy_rows = []
    for experiment_id, result in sorted(evaluated.items()):
        config = result.reproduced.config
        individual_rows.append({
            "experiment_id": experiment_id, "machine": config.machine,
            "regime": config.regime, "model_family": config.model_family,
            "model_seed": config.model_seed, "threshold": result.reproduced.threshold.value,
            "scientific_limitation_id": _limitation_id(config.machine),
            **result.complete_metrics,
        })
        for domain, values in result.domain_metrics.items():
            domain_rows.append({
                "experiment_id": experiment_id, "machine": config.machine,
                "regime": config.regime, "model_family": config.model_family,
                "model_seed": config.model_seed, "domain": domain,
                "scientific_limitation_id": _limitation_id(config.machine), **values,
            })
        healthy_rows.append({
            "experiment_id": experiment_id, "machine": config.machine,
            "regime": config.regime, "model_family": config.model_family,
            "model_seed": config.model_seed,
            "scientific_limitation_id": _limitation_id(config.machine),
            **result.healthy_target,
        })
    _csv(args.out / "individual_experiment_metrics.csv", individual_rows)
    _csv(args.out / "domain_slice_metrics_individual.csv", domain_rows)
    _csv(args.out / "healthy_target_false_positives_individual.csv", healthy_rows)

    groups = _family_groups(evaluated)
    family_rows = []
    if_seed_summary_rows = []
    family_domain_rows = []
    family_healthy_rows = []
    scalar_metrics = (
        *RANKING_METRICS, "true_negative", "false_positive", "false_negative",
        "true_positive", "false_positive_rate", "recall", "precision", "f1",
    )
    family_order = ("statistical_reference", "isolation_forest", "one_class_svm")
    for machine in ("RoboticArm", "BrushlessMotor"):
        for regime in ("S", "T"):
            for family in family_order:
                results = {
                    item.reproduced.config.model_seed: item
                    for item in evaluated.values()
                    if (item.reproduced.config.machine, item.reproduced.config.regime,
                        item.reproduced.config.model_family) == (machine, regime, family)
                }
                row = {
                    "machine": machine, "regime": regime, "model_family": family,
                    "scientific_limitation_id": _limitation_id(machine),
                }
                for metric in scalar_metrics:
                    summary = _value_summary(
                        family, {seed: result.complete_metrics[metric] for seed, result in results.items()}
                    )
                    row[metric] = summary["mean"]
                    if family == "isolation_forest":
                        if_seed_summary_rows.append({
                            "machine": machine, "regime": regime, "metric": metric,
                            "scientific_limitation_id": _limitation_id(machine),
                            **{f"seed_{seed}": summary["seed_values"][seed] for seed in range(5)},
                            "mean": summary["mean"], "population_std": summary["population_std"],
                            "minimum": summary["minimum"], "maximum": summary["maximum"],
                            "count_summary_note": (
                                "mean across seeds; fractional values are not a literal confusion matrix"
                                if metric in {"true_negative", "false_positive", "false_negative", "true_positive"}
                                else "scalar family summary"
                            ),
                        })
                family_rows.append(row)
                for domain in ("source", "target"):
                    sample = next(iter(results.values())).domain_metrics[domain]
                    drow = {
                        "machine": machine, "regime": regime, "model_family": family,
                        "domain": domain, "sample_count": sample["sample_count"],
                        "status": sample["status"],
                        "scientific_limitation_id": _limitation_id(machine),
                    }
                    for metric in RANKING_METRICS:
                        values = {seed: result.domain_metrics[domain][metric] for seed, result in results.items()}
                        drow[metric] = _value_summary(family, values)["mean"]
                    family_domain_rows.append(drow)
                hrow = {
                    "machine": machine, "regime": regime, "model_family": family,
                    "scientific_limitation_id": _limitation_id(machine),
                }
                for metric in ("healthy_target_sample_count", "false_positive", "false_positive_rate"):
                    values = {seed: result.healthy_target[metric] for seed, result in results.items()}
                    summary = _value_summary(family, values)
                    hrow[metric] = summary["mean"]
                    hrow[f"{metric}_population_std"] = summary["population_std"]
                hrow["count_summary_note"] = (
                    "IF count is mean across seeds, possibly fractional" if family == "isolation_forest"
                    else "deterministic literal count"
                )
                family_healthy_rows.append(hrow)
    _csv(args.out / "isolation_forest_seed_summaries.csv", if_seed_summary_rows)
    _csv(args.out / "domain_slice_metrics_family.csv", family_domain_rows)
    _csv(args.out / "healthy_target_false_positives_family.csv", family_healthy_rows)

    bootstrap_rows = []
    distributions = {}
    draws_by_machine = {}
    for machine in ("RoboticArm", "BrushlessMotor"):
        reference = groups[(machine, "S", "statistical_reference")][0]
        draws_by_machine[machine] = make_common_bootstrap_draws(reference)
    for key, frames in sorted(groups.items()):
        machine, regime, family = key
        for metric in RANKING_METRICS:
            distribution = family_bootstrap_distribution(
                family, frames, metric=metric, draws=draws_by_machine[machine]
            )
            distributions[(*key, metric)] = distribution
            bootstrap_rows.append({
                "machine": machine, "regime": regime, "model_family": family,
                "metric": metric, "point_estimate": distribution.point_estimate,
                "scientific_limitation_id": _limitation_id(machine),
                "ci_lower": distribution.ci_lower, "ci_upper": distribution.ci_upper,
                "replicates_requested": distribution.replicates_requested,
                "valid_replicates": distribution.valid_replicates,
                "invalid_replicates": distribution.invalid_replicates,
            })
    _csv(args.out / "bootstrap_confidence_intervals.csv", bootstrap_rows)

    for row in family_rows:
        for metric in RANKING_METRICS:
            distribution = distributions[(row["machine"], row["regime"], row["model_family"], metric)]
            row[f"{metric}_ci_lower"] = distribution.ci_lower
            row[f"{metric}_ci_upper"] = distribution.ci_upper
    _csv(args.out / "primary_baselines.csv", family_rows)

    delta_rows = []
    for machine in ("RoboticArm", "BrushlessMotor"):
        for family in ("statistical_reference", "isolation_forest", "one_class_svm"):
            for metric in RANKING_METRICS:
                delta = paired_distribution_difference(
                    distributions[(machine, "T", family, metric)],
                    distributions[(machine, "S", family, metric)],
                    orientation="T_minus_S",
                )
                delta_rows.append({
                    "machine": machine, "model_family": family,
                    "scientific_limitation_id": _limitation_id(machine), **delta,
                })
    _csv(args.out / "s_vs_t_adaptation_deltas.csv", delta_rows)

    comparison_rows = []
    comparisons = (
        ("statistical_reference", "isolation_forest"),
        ("statistical_reference", "one_class_svm"),
        ("isolation_forest", "one_class_svm"),
    )
    for machine in ("RoboticArm", "BrushlessMotor"):
        for regime in ("S", "T"):
            for first, second in comparisons:
                for metric in RANKING_METRICS:
                    delta = paired_distribution_difference(
                        distributions[(machine, regime, first, metric)],
                        distributions[(machine, regime, second, metric)],
                        orientation=f"{first}_minus_{second}",
                    )
                    comparison_rows.append({
                        "machine": machine, "regime": regime,
                        "comparison": f"{first}_minus_{second}", **delta,
                        "scientific_limitation_id": _limitation_id(machine),
                    })
    _csv(args.out / "model_family_paired_differences.csv", comparison_rows)

    report = {
        "schema_version": PHASE32_SCHEMA_VERSION,
        "status": "FIRST_LOCKED_COMMON_TEST_COMPLETE",
        "test_evaluation_code_sha": code_sha,
        "reporting_policy_id": REPORTING_POLICY_ID,
        "test_batch_id": authorization.batch_id,
        "ledger_start_count": 0,
        "ledger_entries_added": 28,
        "ledger_end_count": len(ledger.read()),
        "test_look_ledger": {
            "path": args.ledger.as_posix(),
            "semantics": "one durable entry per experiment in one pre-score batch append",
            "batch_id": authorization.batch_id,
        },
        "pretest_reproduction": "PASS_28_OF_28_EXACT",
        "common_test_cohorts": cohort_rows,
        "experiment_ids": sorted(expected_ids),
        "primary_results": family_rows,
        "isolation_forest_seed_summaries": if_seed_summary_rows,
        "domain_slice_results": family_domain_rows,
        "healthy_target_false_positive_results": family_healthy_rows,
        "bootstrap_confidence_intervals": bootstrap_rows,
        "s_vs_t_deltas": delta_rows,
        "model_family_comparisons": comparison_rows,
        "environment": _environment(),
        "bootstrap_policy": {
            "method_id": "phase3.0.1-class-stratified-cluster-v1",
            "replicates": 10000,
            "seed": 1995,
            "confidence_interval": "95% percentile",
            "scope": "conditional on the observed evaluation cohort",
        },
        "limitations": {
            "BrushlessMotor": {
                "id": "brushless_motor_acquisition_time_class_confound_v1",
                "text": "Fault/class and acquisition session/time cannot be fully separated in the released cohort; results are not pure isolated mechanical-fault evidence."
            },
            "RoboticArm": {
                "validation": "186 segments from 6 physical parents",
                "regime_t_adaptation": "27 released target-normal segments from 18 parents",
                "blocked_adaptation_siblings": 96,
                "test_inference_cluster": "physical parent recording"
            }
        },
        "no_best_seed_selection": True,
        "test_derived_threshold_used": False,
        "post_test_retuning_performed": False,
    }
    _write_json(args.out / "phase32_first_test_report.json", report)

    markdown = [
        "# Phase 3.2 First Locked Common-Test Evaluation",
        "",
        f"Test evaluator commit: `{code_sha}`. Reporting policy: `{REPORTING_POLICY_ID}`.",
        "",
        "The 28 pre-registered configurations reproduced exactly before the test was opened. The ledger received exactly 28 entries atomically before the first model score.",
        "",
        *_primary_markdown(family_rows, "RoboticArm"),
        "",
        *_primary_markdown(family_rows, "BrushlessMotor"),
        "",
        *_markdown_table(
            "Domain-slice ranking metrics",
            family_domain_rows,
            ("machine", "regime", "model_family", "domain", "sample_count", "status", *RANKING_METRICS),
        ),
        "",
        *_markdown_table(
            "Table D — Healthy-target false positives",
            family_healthy_rows,
            ("machine", "regime", "model_family", "healthy_target_sample_count", "false_positive", "false_positive_rate"),
        ),
        "",
        *_delta_markdown(delta_rows),
        "",
        *_comparison_markdown(comparison_rows),
        "",
        *_markdown_table(
            "Isolation-Forest individual-seed appendix",
            if_seed_summary_rows,
            ("machine", "regime", "metric", "seed_0", "seed_1", "seed_2", "seed_3", "seed_4", "mean", "population_std", "minimum", "maximum"),
        ),
        "",
        "Isolation-Forest rows are arithmetic five-seed family summaries. Full individual-seed, domain, healthy-target, bootstrap, S-vs-T, and paired-comparison tables accompany this report.",
        "",
        "No pooled cross-machine result or universal winner is reported.",
        "",
        "BrushlessMotor limitation `brushless_motor_acquisition_time_class_confound_v1`: fault/class and acquisition session/time cannot be fully separated, so these results are not pure isolated mechanical-fault evidence.",
        "",
        "RoboticArm uncertainty clusters by physical parent recording. Source-normal validation was 186 segments from six parents; Regime-T adaptation was 27 released segments from 18 parents, with 96 siblings blocked.",
        "",
    ]
    (args.out / "phase32_first_test_report.md").write_text("\n".join(markdown), encoding="utf-8")

    state.update({"state": "complete", "common_test_scores_completed": 28})
    _write_json(args.out / "evaluation_state.json", state)
    artifact_files = sorted(
        path for path in args.out.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    _write_json(args.out / "artifact_manifest.json", {
        "schema_version": PHASE32_SCHEMA_VERSION,
        "test_batch_id": authorization.batch_id,
        "files": {
            path.relative_to(args.out).as_posix(): file_sha256(path)
            for path in artifact_files
        },
    })
    print(canonical_json({
        "status": "complete", "test_batch_id": authorization.batch_id,
        "experiments": 28, "ledger_count": len(ledger.read()),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
