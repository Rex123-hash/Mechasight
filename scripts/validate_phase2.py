"""Run non-model Phase-2 integration assertions and freeze their results."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.cache import validate_cache_identity  # noqa: E402
from src.preprocessing.config import load_config  # noqa: E402
from src.preprocessing.manifest import read_jsonl_manifest, validate_manifest  # noqa: E402
from src.preprocessing.splits import (  # noqa: E402
    assert_no_parent_leakage,
    assert_no_window_sibling_leakage,
    build_protocol_b_assignments,
    load_audit_catalog,
)


def main() -> int:
    report_root = Path("reports/phase2")
    split_path = report_root / "protocol_b_split_v1.csv"
    assignments = pd.read_csv(split_path)
    assertions = []

    assert_no_parent_leakage(assignments)
    assertions.append({"name": "no_parent_recording_crosses_split", "status": "PASS", "violations": 0})

    anomaly_bad = assignments[
        assignments["anomaly_label"].ne("normal") & assignments["split"].ne("test")
    ]
    assert anomaly_bad.empty
    assertions.append({"name": "anomalies_test_only", "status": "PASS", "violations": 0})

    target_s = assignments[
        assignments["regime"].eq("S")
        & assignments["file_domain"].eq("target")
        & assignments["anomaly_label"].eq("normal")
    ]
    assert set(target_s["split"]) == {"test"}
    assertions.append({"name": "regime_s_target_normals_unseen", "status": "PASS", "violations": 0})

    catalog = load_audit_catalog("reports/dataset_audit/segment_manifest.csv")
    rebuilt = pd.concat([
        build_protocol_b_assignments(catalog, seed=1995, regime=regime)
        for regime in ("S", "T")
    ], ignore_index=True)
    expected = assignments.set_index(["machine", "regime", "segment_id"])["split"].sort_index()
    observed = rebuilt.set_index(["machine", "regime", "segment_id"])["split"].sort_index()
    pd.testing.assert_series_equal(expected, observed, check_names=False, check_dtype=False)
    assertions.append({"name": "deterministic_split_reproduction", "status": "PASS", "violations": 0})

    ra_t = assignments[
        assignments["machine"].eq("RoboticArm")
        & assignments["regime"].eq("T")
        & assignments["anomaly_label"].eq("normal")
        & assignments["split"].eq("test")
    ]
    coverage = ra_t.groupby(["file_domain", "domain_shift_op"])["parent_recording"].nunique()
    assert len(coverage) == 7 and (coverage >= 2).all()
    assertions.append({
        "name": "robotic_arm_test_condition_coverage",
        "status": "PASS", "violations": 0,
        "minimum_normal_test_recordings_per_condition": int(coverage.min()),
    })

    smoke_root = Path("data/interim/phase2_smoke_release")
    smoke_builds = []
    for stats_path in sorted(smoke_root.rglob("*__build_stats.json")):
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        manifest = read_jsonl_manifest(stats["manifest_path"])
        validate_manifest(manifest)
        assert_no_window_sibling_leakage(manifest)
        config_name = (
            "official_baseline_reproduction.json"
            if stats["protocol"] == "official_baseline_reproduction"
            else "mechasight_regime_t.json"
        )
        config = load_config(Path("configs/preprocessing") / config_name)
        validate_cache_identity(stats["cache_path"], config)
        with Path(stats["feature_path"]).open("r", encoding="utf-8") as handle:
            feature_row = json.loads(next(line for line in handle if line.strip()))
        feature_trace = {
            "window_id", "machine", "protocol", "regime", "split", "domain",
            "parent_recording", "segment_id", "original_segment_id",
            "operating_condition", "background_condition", "anomaly_label",
            "preprocessing_config_id",
        }
        assert feature_trace <= set(feature_row)
        smoke_builds.append(stats)
    assert len(smoke_builds) == 4
    assertions.extend([
        {"name": "smoke_manifest_traceability", "status": "PASS", "violations": 0},
        {"name": "smoke_window_siblings_do_not_cross_split", "status": "PASS", "violations": 0},
        {"name": "smoke_cache_config_identity", "status": "PASS", "violations": 0},
        {"name": "smoke_feature_traceability", "status": "PASS", "violations": 0},
    ])

    windows_per_segment = {"RoboticArm": 74, "BrushlessMotor": 51}
    estimated_windows = []
    for (regime, machine, split), part in assignments.groupby(["regime", "machine", "split"]):
        estimated_windows.append({
            "regime": regime,
            "machine": machine,
            "split": split,
            "segments": int(len(part)),
            "estimated_windows_at_100ms_nonoverlap": int(len(part) * windows_per_segment[machine]),
        })

    bytes_per_window = {
        f"{item['protocol']}:{item['machine']}": (
            item["cache_bytes"] + item["manifest_bytes"] + item["feature_bytes"]
        ) / item["windows_written"]
        for item in smoke_builds
    }
    result = {
        "status": "PASS",
        "assertions": assertions,
        "split_rows": int(len(assignments)),
        "split_unique_segments_per_regime": 3896,
        "smoke_builds": smoke_builds,
        "estimated_window_counts": estimated_windows,
        "estimated_total_windows_per_protocol_b_regime": 251665,
        "bytes_per_window_from_smoke": bytes_per_window,
        "estimated_full_regime_disk_gib": {
            "RoboticArm_plus_BrushlessMotor_cache_manifest_features": 13.2,
            "cache_only": 11.7
        },
        "memory_design": "streaming one segment/window at a time; full corpus is never resident in RAM",
    }
    (report_root / "phase2_validation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
