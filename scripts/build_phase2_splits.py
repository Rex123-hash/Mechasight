"""Freeze Phase-2 Protocol-B parent-recording assignments and statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.splits import (  # noqa: E402
    assert_no_parent_leakage,
    build_protocol_b_assignments,
    load_audit_catalog,
    robotic_arm_assignments,
    split_statistics,
)


CANDIDATE_SEEDS = (1995, 20240821, 20260821)
SELECTED_SEED = 1995


def candidate_statistics(catalog: pd.DataFrame) -> pd.DataFrame:
    ra = catalog[catalog["machine"].eq("RoboticArm")]
    outputs = []
    for seed in CANDIDATE_SEEDS:
        assignment = robotic_arm_assignments(ra, seed=seed, regime="T")
        grouped = assignment.groupby(
            ["file_domain", "domain_shift_op", "anomaly_label", "split"], sort=True
        ).agg(
            n_recordings=("parent_recording", "nunique"),
            n_segments=("segment_id", "size"),
            n_backgrounds=("domain_shift_env", "nunique"),
        ).reset_index()
        grouped.insert(0, "seed", seed)
        grouped.insert(0, "regime", "T")
        grouped.insert(0, "protocol", "mechasight_experimental_protocol")
        grouped.insert(0, "machine", "RoboticArm")
        grouped["split_version"] = "protocol_b_v1"
        grouped["preprocessing_version"] = "phase2-v1"
        outputs.append(grouped)
    return pd.concat(outputs, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-manifest", type=Path,
        default=Path("reports/dataset_audit/segment_manifest.csv"),
    )
    parser.add_argument("--out", type=Path, default=Path("reports/phase2"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    catalog = load_audit_catalog(args.audit_manifest)
    candidates = candidate_statistics(catalog)
    candidates.to_csv(args.out / "protocol_b_candidate_statistics.csv", index=False)

    assignments = pd.concat([
        build_protocol_b_assignments(catalog, seed=SELECTED_SEED, regime=regime)
        for regime in ("S", "T")
    ], ignore_index=True)
    assignments["split_version"] = "protocol_b_v1"
    assignments["preprocessing_version"] = "phase2-v1"
    assert_no_parent_leakage(assignments)
    keep = [
        "machine", "protocol", "regime", "split", "split_seed",
        "split_version", "preprocessing_version",
        "parent_recording", "segment_id", "source_csv", "file_split",
        "file_domain", "split_label", "anomaly_label", "domain_shift_op",
        "domain_shift_env",
    ]
    assignments[keep].sort_values(
        ["machine", "regime", "split", "parent_recording", "segment_id"]
    ).to_csv(args.out / "protocol_b_split_v1.csv", index=False)
    split_statistics(assignments).to_csv(
        args.out / "protocol_b_split_statistics.csv", index=False
    )

    summary = {
        "split_version": "protocol_b_v1",
        "selected_seed": SELECTED_SEED,
        "candidate_seeds_reported": list(CANDIDATE_SEEDS),
        "selection_rule": (
            "Seed 1995 was adopted a priori to reuse the pinned upstream baseline seed; "
            "no signal values, anomaly scores, or future model results were inspected."
        ),
        "robotic_arm_rule": (
            "All anomaly recordings test; normal recordings stratified by domain and "
            "operating condition, approximately 80/10/10 by recording with minimum one "
            "validation and two test recordings per stratum. Regime S maps every target "
            "normal recording to test."
        ),
        "brushless_motor_rule": (
            "Released test retained; released normal-train recordings split 90/10 "
            "train/validation within domain and operating condition. Regime S maps "
            "released target-train normals to test."
        ),
        "rows": int(len(assignments)),
        "parent_leakage_count": 0,
    }
    (args.out / "protocol_b_split_design.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
