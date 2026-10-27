"""Freeze the Phase-2.1 matched S/T common-test split without touching v1."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.splits import (  # noqa: E402
    assert_matched_regime_cohorts,
    assert_no_parent_leakage,
    attach_parent_recording,
    load_audit_catalog,
    matched_common_test_assignments,
)


SEED = 1995
SPLIT_VERSION = "protocol_b_common_test_v2"


def main() -> int:
    out = Path("reports/phase2_1")
    out.mkdir(parents=True, exist_ok=True)
    catalog = attach_parent_recording(
        load_audit_catalog("reports/dataset_audit/segment_manifest.csv")
    )

    assignments = pd.concat([
        matched_common_test_assignments(catalog, seed=SEED, regime=regime)
        for regime in ("S", "T")
    ], ignore_index=True)
    assert_no_parent_leakage(assignments)
    assert_matched_regime_cohorts(assignments)

    keep = [
        "machine", "protocol", "regime", "split", "data_pool", "split_seed",
        "split_version", "preprocessing_version", "parent_recording", "segment_id",
        "source_csv", "file_split", "file_domain", "split_label", "anomaly_label",
        "domain_shift_op", "domain_shift_env",
    ]
    frozen = assignments[keep].sort_values(
        ["machine", "regime", "data_pool", "parent_recording", "segment_id"]
    )
    frozen.to_csv(out / "protocol_b_common_test_v2.csv", index=False)

    counts = assignments.groupby(
        ["machine", "regime", "file_domain", "data_pool", "split"], sort=True
    ).agg(
        n_recordings=("parent_recording", "nunique"),
        n_segments=("segment_id", "size"),
    ).reset_index()
    counts.to_csv(out / "protocol_b_common_test_v2_counts.csv", index=False)

    released_rows = []
    adaptation_parent_rows = []
    for machine, part in catalog.groupby("machine", sort=True):
        target_normal = part[
            part["file_domain"].eq("target") & part["anomaly_label"].eq("normal")
        ]
        released = target_normal[target_normal["file_split"].eq("train")]
        parents = set(released["parent_recording"])
        expanded = target_normal[target_normal["parent_recording"].isin(parents)]
        released_rows.append({
            "machine": machine,
            "released_target_normal_segments": int(len(released)),
            "released_target_normal_parent_recordings": int(released["parent_recording"].nunique()),
            "leakage_safe_adaptation_parent_recordings": int(len(parents)),
            "leakage_safe_adaptation_segments": int(len(expanded)),
            "additional_sibling_segments_reserved": int(len(expanded) - len(released)),
        })
        parent_summary = expanded.groupby(
            ["parent_recording", "domain_shift_op"], sort=True
        ).agg(
            n_segments=("segment_id", "size"),
            n_released_train_segments=("file_split", lambda values: int((values == "train").sum())),
            n_released_test_siblings=("file_split", lambda values: int((values == "test").sum())),
        ).reset_index()
        parent_summary.insert(0, "machine", machine)
        adaptation_parent_rows.append(parent_summary)
    expansion = pd.DataFrame(released_rows)
    expansion.to_csv(out / "target_adaptation_expansion.csv", index=False)
    pd.concat(adaptation_parent_rows, ignore_index=True).to_csv(
        out / "target_adaptation_parent_groups.csv", index=False
    )

    proof = []
    for machine, part in assignments.groupby("machine", sort=True):
        s = part[part["regime"].eq("S")]
        t = part[part["regime"].eq("T")]
        s_segments = set(s.loc[s["split"].eq("test"), "segment_id"])
        t_segments = set(t.loc[t["split"].eq("test"), "segment_id"])
        s_parents = set(s.loc[s["split"].eq("test"), "parent_recording"])
        t_parents = set(t.loc[t["split"].eq("test"), "parent_recording"])
        s_anomaly = set(s.loc[s["anomaly_label"].ne("normal"), "segment_id"])
        t_anomaly = set(t.loc[t["anomaly_label"].ne("normal"), "segment_id"])
        proof.append({
            "machine": machine,
            "s_test_segments": len(s_segments),
            "t_test_segments": len(t_segments),
            "test_segment_symmetric_difference": len(s_segments ^ t_segments),
            "s_test_parents": len(s_parents),
            "t_test_parents": len(t_parents),
            "test_parent_symmetric_difference": len(s_parents ^ t_parents),
            "anomaly_test_symmetric_difference": len(s_anomaly ^ t_anomaly),
        })
    proof_frame = pd.DataFrame(proof)
    proof_frame.to_csv(out / "matched_test_identity_proof.csv", index=False)

    summary = {
        "status": "PASS",
        "split_version": SPLIT_VERSION,
        "seed": SEED,
        "supersedes": "reports/phase2/protocol_b_split_v1.csv for future S-vs-T comparisons",
        "historical_v1_preserved": True,
        "reason": (
            "Regime S and Regime T must be compared on an identical held-out evaluation "
            "cohort so that performance differences are attributable to target-normal "
            "adaptation rather than test-set composition."
        ),
        "adaptation_expansion": expansion.to_dict("records"),
        "matched_test_proof": proof,
    }
    (out / "protocol_b_common_test_v2_design.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
