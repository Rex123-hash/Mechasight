"""Freeze the Phase-2.2 few-shot adaptation-budget split without touching v1/v2.

Phase 2.1 fixed unequal S/T test cohorts.  Phase 2.2 fixes a different problem in
the same freeze: parent closure was used as the Regime-T *training* budget, so
RoboticArm adapted on 123 segments when the release only offers 27.  This script
separates ``target_adaptation`` (consumed) from ``adaptation_sibling_blocked``
(withheld from everything) and proves the common evaluation cohort did not move.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.splits import (  # noqa: E402
    BLOCKED_SIBLING_POOL,
    PROTOCOL_B_V3_PREPROCESSING_VERSION,
    PROTOCOL_B_V3_SPLIT_VERSION,
    NON_CONSUMING_SPLITS,
    adaptation_budget_summary,
    assert_common_test_unchanged,
    assert_few_shot_adaptation_cohorts,
    assert_no_parent_leakage,
    attach_parent_recording,
    few_shot_adaptation_assignments,
    load_audit_catalog,
)


SEED = 1995
V2_PATH = Path("reports/phase2_1/protocol_b_common_test_v2.csv")


def main() -> int:
    out = Path("reports/phase2_2")
    out.mkdir(parents=True, exist_ok=True)
    catalog = attach_parent_recording(
        load_audit_catalog("reports/dataset_audit/segment_manifest.csv")
    )

    assignments = pd.concat([
        few_shot_adaptation_assignments(catalog, seed=SEED, regime=regime)
        for regime in ("S", "T")
    ], ignore_index=True)
    assert_no_parent_leakage(assignments, ignore_splits=NON_CONSUMING_SPLITS)
    assert_few_shot_adaptation_cohorts(assignments)

    previous = pd.read_csv(V2_PATH)
    assert_common_test_unchanged(previous, assignments)

    keep = [
        "machine", "protocol", "regime", "split", "data_pool", "split_seed",
        "split_version", "preprocessing_version", "parent_recording", "segment_id",
        "source_csv", "file_split", "file_domain", "split_label", "anomaly_label",
        "domain_shift_op", "domain_shift_env",
    ]
    frozen = assignments[keep].sort_values(
        ["machine", "regime", "data_pool", "parent_recording", "segment_id"]
    )
    frozen.to_csv(out / "protocol_b_few_shot_adaptation_v3.csv", index=False)

    counts = assignments.groupby(
        ["machine", "regime", "file_domain", "data_pool", "split"], sort=True
    ).agg(
        n_recordings=("parent_recording", "nunique"),
        n_segments=("segment_id", "size"),
    ).reset_index()
    counts.to_csv(out / "protocol_b_few_shot_adaptation_v3_counts.csv", index=False)

    budget = adaptation_budget_summary(assignments)
    budget.to_csv(out / "adaptation_budget.csv", index=False)

    # Per-machine, per-regime protocol role counts in the exact terms the
    # Phase-2.2 checkpoint asks for.
    role_rows = []
    for (machine, regime), part in assignments.groupby(["machine", "regime"], sort=True):
        row = {"machine": machine, "regime": regime}
        for role in ("source_training", "source_validation", "target_adaptation",
                     BLOCKED_SIBLING_POOL, "common_evaluation"):
            selected = part[part["data_pool"].eq(role)]
            row[f"{role}_parents"] = int(selected["parent_recording"].nunique())
            row[f"{role}_segments"] = int(len(selected))
        fitted = part[part["split"].eq("train")]
        row["train_parents"] = int(fitted["parent_recording"].nunique())
        row["train_segments"] = int(len(fitted))
        role_rows.append(row)
    roles = pd.DataFrame(role_rows)
    roles.to_csv(out / "protocol_role_counts.csv", index=False)

    # Proof that blocked siblings appear in no consuming role, and that the
    # common test is untouched relative to the Phase-2.1 freeze.
    proof_rows = []
    for machine, part in assignments.groupby("machine", sort=True):
        blocked = set(part.loc[part["data_pool"].eq(BLOCKED_SIBLING_POOL), "segment_id"])
        entry = {"machine": machine, "blocked_sibling_segments": len(blocked)}
        for regime in ("S", "T"):
            regime_part = part[part["regime"].eq(regime)]
            for split in ("train", "validation", "test"):
                ids = set(regime_part.loc[regime_part["split"].eq(split), "segment_id"])
                entry[f"blocked_in_{regime.lower()}_{split}"] = len(blocked & ids)
        v2_machine = previous[previous["machine"].eq(machine)]
        v2_test_s = set(v2_machine.loc[
            v2_machine["regime"].eq("S") & v2_machine["split"].eq("test"), "segment_id"
        ])
        v3_test_s = set(part.loc[
            part["regime"].eq("S") & part["split"].eq("test"), "segment_id"
        ])
        v3_test_t = set(part.loc[
            part["regime"].eq("T") & part["split"].eq("test"), "segment_id"
        ])
        entry["v2_v3_test_symmetric_difference"] = len(v2_test_s ^ v3_test_s)
        entry["s_t_test_symmetric_difference"] = len(v3_test_s ^ v3_test_t)
        proof_rows.append(entry)
    proof = pd.DataFrame(proof_rows)
    proof.to_csv(out / "blocked_sibling_isolation_proof.csv", index=False)

    summary = {
        "status": "PASS",
        "split_version": PROTOCOL_B_V3_SPLIT_VERSION,
        "preprocessing_version": PROTOCOL_B_V3_PREPROCESSING_VERSION,
        "seed": SEED,
        "supersedes": (
            "reports/phase2_1/protocol_b_common_test_v2.csv for the primary S-vs-T "
            "adaptation-budget definition only"
        ),
        "historical_v1_preserved": True,
        "historical_v2_preserved": True,
        "correction": (
            "Phase 2.1 solved unequal S/T test cohorts by making the whole parent "
            "closure the Regime-T training pool. Phase 2.2 separates 'target examples "
            "actually consumed' from 'same-parent siblings blocked to prevent leakage'. "
            "Parent independence requires siblings to leave evaluation; it does not "
            "require them to become training data."
        ),
        "primary_scientific_question": (
            "Does access to the released few-shot target-normal training allocation "
            "improve anomaly detection under domain shift?"
        ),
        "secondary_regime_documented_not_run": {
            "name": "T-parent-complete",
            "definition": "adaptation on all 123 normal RA segments of the 18 adaptation parents",
            "authorised_now": False,
        },
        "adaptation_budget": budget.to_dict("records"),
        "protocol_role_counts": roles.to_dict("records"),
        "blocked_sibling_isolation": proof.to_dict("records"),
    }
    (out / "protocol_b_few_shot_adaptation_v3_design.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
