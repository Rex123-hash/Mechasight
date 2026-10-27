"""Validate the matched common-test protocol and its smoke caches."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.policy import assert_preprocessing_pair, load_evaluation_policy  # noqa: E402
from src.preprocessing.cache import validate_cache_identity  # noqa: E402
from src.preprocessing.config import load_config  # noqa: E402
from src.preprocessing.manifest import read_jsonl_manifest, validate_manifest  # noqa: E402
from src.preprocessing.splits import (  # noqa: E402
    assert_matched_regime_cohorts,
    assert_no_parent_leakage,
)


def main() -> int:
    out = Path("reports/phase2_1")
    assignments = pd.read_csv(out / "protocol_b_common_test_v2.csv")
    assertions = []

    assert_no_parent_leakage(assignments)
    assert_matched_regime_cohorts(assignments)
    assertions.extend([
        {"name": "no_parent_leakage", "status": "PASS", "violations": 0},
        {"name": "identical_s_t_test_segments", "status": "PASS", "violations": 0},
        {"name": "identical_s_t_test_parents", "status": "PASS", "violations": 0},
        {"name": "identical_s_t_anomaly_population", "status": "PASS", "violations": 0},
        {"name": "identical_s_t_source_validation", "status": "PASS", "violations": 0},
    ])

    policy = load_evaluation_policy("configs/evaluation/protocol_b_common_test_v2.json")
    assert policy["metrics"]["primary_threshold_free"] == ["AUROC", "AUPRC"]
    assert policy["threshold"]["test_labels_used"] is False
    assert policy["threshold"]["quantile"] == 0.9
    assert policy["score_aggregation"]["protocol_b_primary"] == "median of window anomaly scores"
    assert policy["uncertainty"]["robotic_arm"]["unit_sampled"].startswith("parent_recording")
    assertions.append({
        "name": "metric_threshold_aggregation_uncertainty_policy_frozen",
        "status": "PASS", "violations": 0,
    })

    config_s = load_config("configs/preprocessing/mechasight_common_test_regime_s.json")
    config_t = load_config("configs/preprocessing/mechasight_common_test_regime_t.json")
    assert_preprocessing_pair(config_s, config_t)
    old_s = load_config("configs/preprocessing/mechasight_regime_s.json")
    old_t = load_config("configs/preprocessing/mechasight_regime_t.json")
    assert config_s.config_id not in {old_s.config_id, old_t.config_id}
    assert config_t.config_id not in {old_s.config_id, old_t.config_id}
    assertions.extend([
        {"name": "paired_preprocessing_differs_only_by_regime", "status": "PASS", "violations": 0},
        {"name": "phase2_cache_identity_invalidated", "status": "PASS", "violations": 0},
    ])

    smoke_root = Path("data/interim/phase21_smoke_release")
    builds = []
    manifests = {}
    for stats_path in sorted(smoke_root.rglob("*__build_stats.json")):
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        config = config_s if stats["regime"] == "S" else config_t
        validate_cache_identity(stats["cache_path"], config)
        manifest = read_jsonl_manifest(stats["manifest_path"])
        validate_manifest(manifest)
        assert set(manifest["data_pool"]) == {"common_evaluation"}
        builds.append(stats)
        manifests[(stats["machine"], stats["regime"])] = manifest
    assert len(builds) == 4
    for machine in ("RoboticArm", "BrushlessMotor"):
        s = manifests[(machine, "S")].sort_values(["segment_id", "window_index"])
        t = manifests[(machine, "T")].sort_values(["segment_id", "window_index"])
        for field in (
            "segment_id", "parent_recording", "window_index", "window_start_s",
            "window_end_s", "domain", "operating_condition", "background_condition",
            "anomaly_label", "raw_file_references",
        ):
            pd.testing.assert_series_equal(
                s[field].reset_index(drop=True), t[field].reset_index(drop=True),
                check_names=False, check_dtype=False,
            )
    assertions.append({
        "name": "smoke_s_t_physical_windows_identical", "status": "PASS", "violations": 0
    })

    proof = pd.read_csv(out / "matched_test_identity_proof.csv").to_dict("records")
    expansion = pd.read_csv(out / "target_adaptation_expansion.csv").to_dict("records")
    result = {
        "status": "PASS",
        "assertions": assertions,
        "matched_test_identity": proof,
        "target_adaptation_expansion": expansion,
        "new_config_ids": {"S": config_s.config_id, "T": config_t.config_id},
        "historical_config_ids": {"S": old_s.config_id, "T": old_t.config_id},
        "smoke_builds": builds,
        "smoke_total_segments": sum(item["segments_read"] for item in builds),
        "smoke_total_windows": sum(item["windows_written"] for item in builds),
    }
    (out / "phase21_validation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
