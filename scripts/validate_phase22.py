"""Validate the Phase-2.2 few-shot adaptation budget and its smoke caches."""

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
from src.preprocessing.pipeline import merge_protocol_b_assignments  # noqa: E402
from src.preprocessing.splits import (  # noqa: E402
    BLOCKED_SIBLING_POOL,
    BLOCKED_SIBLING_SPLIT,
    CONSUMING_SPLITS,
    NON_CONSUMING_SPLITS,
    adaptation_budget_summary,
    assert_common_test_unchanged,
    assert_few_shot_adaptation_cohorts,
    assert_matched_regime_cohorts,
    assert_no_parent_leakage,
)


V3_CONFIGS = (
    "configs/preprocessing/mechasight_few_shot_regime_s.json",
    "configs/preprocessing/mechasight_few_shot_regime_t.json",
)
HISTORICAL_CONFIGS = (
    "configs/preprocessing/mechasight_regime_s.json",
    "configs/preprocessing/mechasight_regime_t.json",
    "configs/preprocessing/mechasight_common_test_regime_s.json",
    "configs/preprocessing/mechasight_common_test_regime_t.json",
)


def main() -> int:
    out = Path("reports/phase2_2")
    assignments = pd.read_csv(out / "protocol_b_few_shot_adaptation_v3.csv")
    previous = pd.read_csv("reports/phase2_1/protocol_b_common_test_v2.csv")
    assertions = []

    def record(name: str) -> None:
        assertions.append({"name": name, "status": "PASS", "violations": 0})

    # --- protocol structure -------------------------------------------------
    assert_no_parent_leakage(assignments, ignore_splits=NON_CONSUMING_SPLITS)
    assert_matched_regime_cohorts(assignments)
    assert_few_shot_adaptation_cohorts(assignments)
    assert_common_test_unchanged(previous, assignments)
    for name in (
        "no_consuming_parent_leakage",
        "identical_s_t_test_segments",
        "identical_s_t_test_parents",
        "identical_s_t_anomaly_population",
        "identical_s_t_source_validation",
        "every_segment_has_exactly_one_protocol_role",
        "blocked_siblings_never_enter_model_fitting",
        "blocked_siblings_never_enter_validation",
        "blocked_siblings_never_enter_evaluation",
        "consumed_adaptation_segments_belong_to_adaptation_parents",
        "only_released_target_normals_are_consumed_in_primary_regime_t",
        "common_test_identical_to_phase_2_1_freeze",
    ):
        record(name)

    # --- explicit numeric budget separation --------------------------------
    budget = adaptation_budget_summary(assignments).set_index("machine")
    expected = {
        "RoboticArm": (27, 18, 96, 123),
        "BrushlessMotor": (18, 18, 0, 18),
    }
    for machine, (observed, parents, blocked, closure) in expected.items():
        row = budget.loc[machine]
        assert int(row["target_adaptation_observed"]) == observed, machine
        assert int(row["target_adaptation_parent_groups"]) == parents, machine
        assert int(row["target_adaptation_blocked_siblings"]) == blocked, machine
        assert int(row["target_adaptation_parent_closure"]) == closure, machine
        assert bool(row["blocked_parents_subset_of_adaptation_parents"]), machine
    record("adaptation_budget_concepts_not_collapsed")

    # RoboticArm Regime T must train on source + 27, never source + 123.
    ra = assignments[assignments["machine"].eq("RoboticArm")]
    ra_source_train = int(len(ra[
        ra["regime"].eq("S") & ra["data_pool"].eq("source_training")
    ]))
    ra_t_train = int(len(ra[ra["regime"].eq("T") & ra["split"].eq("train")]))
    ra_s_train = int(len(ra[ra["regime"].eq("S") & ra["split"].eq("train")]))
    assert ra_s_train == ra_source_train
    assert ra_t_train == ra_source_train + 27
    assert ra_t_train != ra_source_train + 123
    record("robotic_arm_regime_t_trains_on_source_plus_released_27")

    bm = assignments[assignments["machine"].eq("BrushlessMotor")]
    bm_source_train = int(len(bm[
        bm["regime"].eq("S") & bm["data_pool"].eq("source_training")
    ]))
    bm_t_train = int(len(bm[bm["regime"].eq("T") & bm["split"].eq("train")]))
    assert int(len(bm[bm["regime"].eq("S") & bm["split"].eq("train")])) == bm_source_train
    assert bm_t_train == bm_source_train + 18
    record("brushless_motor_regime_t_trains_on_source_plus_released_18")

    # --- evaluation policy: corrected fields, frozen fields untouched -------
    policy = load_evaluation_policy(
        "configs/evaluation/protocol_b_few_shot_adaptation_v3.json"
    )
    frozen = load_evaluation_policy("configs/evaluation/protocol_b_common_test_v2.json")
    for section in ("metrics", "threshold", "score_aggregation", "uncertainty"):
        assert policy[section] == frozen[section], section
    assert policy["validation"]["forbidden"] == frozen["validation"]["forbidden"]
    record("phase_2_1_metric_threshold_aggregation_uncertainty_policy_unchanged")

    assert policy["split_version"] == "protocol_b_few_shot_adaptation_v3"
    assert policy["adaptation_budget"]["RoboticArm"]["target_adaptation_observed"] == 27
    assert policy["adaptation_budget"]["RoboticArm"]["target_adaptation_blocked_siblings"] == 96
    assert policy["adaptation_budget"]["RoboticArm"]["target_adaptation_parent_closure"] == 123
    assert policy["secondary_regimes"]["T-parent-complete"]["authorised_now"] is False
    assert all(
        value is False
        for key, value in policy["adaptation_budget"]["blocked_sibling_use"].items()
        if isinstance(value, bool)
    )
    record("few_shot_budget_policy_frozen_and_secondary_regime_unauthorised")

    # --- configuration / cache versioning ----------------------------------
    config_s, config_t = (load_config(path) for path in V3_CONFIGS)
    assert_preprocessing_pair(config_s, config_t)
    historical = {load_config(path).config_id for path in HISTORICAL_CONFIGS}
    assert config_s.config_id not in historical
    assert config_t.config_id not in historical
    assert config_s.split_version == config_t.split_version == "protocol_b_few_shot_adaptation_v3"
    record("paired_preprocessing_differs_only_by_regime")
    record("phase_2_and_phase_2_1_cache_identities_invalidated")

    # A v3 config must refuse the Phase-2.1 split manifest outright.
    try:
        merge_protocol_b_assignments(
            pd.DataFrame({"machine": [], "segment_id": []}), previous, "T",
            split_version=config_t.split_version,
        )
    except ValueError as error:
        assert "incompatible" in str(error)
    else:  # pragma: no cover - defensive
        raise AssertionError("v3 config accepted the Phase-2.1 split manifest")
    record("v3_config_rejects_phase_2_1_split_manifest")

    # --- smoke builds -------------------------------------------------------
    smoke_root = Path("data/interim/phase22_smoke_release")
    builds = []
    test_manifests = {}
    adaptation_manifests = {}
    released_ids = {
        machine: set(part.loc[
            part["regime"].eq("T") & part["data_pool"].eq("target_adaptation"), "segment_id"
        ])
        for machine, part in assignments.groupby("machine")
    }
    blocked_ids = set(assignments.loc[
        assignments["data_pool"].eq(BLOCKED_SIBLING_POOL), "segment_id"
    ])
    for stats_path in sorted(smoke_root.rglob("*__build_stats.json")):
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        config = config_s if stats["regime"] == "S" else config_t
        validate_cache_identity(stats["cache_path"], config)
        manifest = read_jsonl_manifest(stats["manifest_path"])
        validate_manifest(manifest)
        assert set(manifest["split"]) <= CONSUMING_SPLITS
        assert not set(manifest["segment_id"]) & blocked_ids
        pool = set(manifest["data_pool"])
        assert len(pool) == 1
        role = pool.pop()
        if role == "common_evaluation":
            test_manifests[(stats["machine"], stats["regime"])] = manifest
        elif role == "target_adaptation":
            assert stats["regime"] == "T"
            assert set(manifest["segment_id"]) <= released_ids[stats["machine"]]
            adaptation_manifests[stats["machine"]] = manifest
        else:  # pragma: no cover - defensive
            raise AssertionError(f"unexpected smoke role {role}")
        builds.append(stats)
    assert len(builds) == 6
    assert len(test_manifests) == 4 and len(adaptation_manifests) == 2
    record("no_smoke_artifact_contains_a_blocked_sibling")
    record("adaptation_smoke_builds_contain_only_released_target_normals")

    for machine in ("RoboticArm", "BrushlessMotor"):
        s = test_manifests[(machine, "S")].sort_values(["segment_id", "window_index"])
        t = test_manifests[(machine, "T")].sort_values(["segment_id", "window_index"])
        for field in (
            "segment_id", "parent_recording", "window_index", "window_start_s",
            "window_end_s", "domain", "operating_condition", "background_condition",
            "anomaly_label", "raw_file_references",
        ):
            pd.testing.assert_series_equal(
                s[field].reset_index(drop=True), t[field].reset_index(drop=True),
                check_names=False, check_dtype=False,
            )
    record("smoke_s_t_physical_windows_identical")

    # A blocked sibling can never be materialised, even if asked for directly.
    from src.preprocessing.pipeline import iter_processed_windows

    try:
        next(iter_processed_windows(
            pd.DataFrame(columns=["split", "machine", "regime", "protocol"]),
            config_t, split=BLOCKED_SIBLING_SPLIT,
        ))
    except ValueError as error:
        assert "no computation" in str(error)
    else:  # pragma: no cover - defensive
        raise AssertionError("pipeline materialised a non-consuming split")
    record("pipeline_refuses_to_materialise_non_consuming_splits")

    roles = pd.read_csv(out / "protocol_role_counts.csv")
    proof = pd.read_csv(out / "blocked_sibling_isolation_proof.csv")
    result = {
        "status": "PASS",
        "assertions": assertions,
        "adaptation_budget": adaptation_budget_summary(assignments).to_dict("records"),
        "protocol_role_counts": roles.to_dict("records"),
        "blocked_sibling_isolation": proof.to_dict("records"),
        "regime_train_counts": {
            "RoboticArm": {"S": ra_s_train, "T": ra_t_train},
            "BrushlessMotor": {
                "S": int(len(bm[bm["regime"].eq("S") & bm["split"].eq("train")])),
                "T": bm_t_train,
            },
        },
        "new_config_ids": {"S": config_s.config_id, "T": config_t.config_id},
        "historical_config_ids": {
            Path(path).stem: load_config(path).config_id for path in HISTORICAL_CONFIGS
        },
        "smoke_builds": builds,
        "smoke_total_segments": sum(item["segments_read"] for item in builds),
        "smoke_total_windows": sum(item["windows_written"] for item in builds),
    }
    (out / "phase22_validation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
