"""Phase-2.2: the released adaptation budget is not the parent closure."""

from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.policy import assert_preprocessing_pair, load_evaluation_policy
from src.preprocessing.config import load_config
from src.preprocessing.pipeline import iter_processed_windows, merge_protocol_b_assignments
from src.preprocessing.splits import (
    BLOCKED_SIBLING_POOL,
    BLOCKED_SIBLING_SPLIT,
    CONSUMING_SPLITS,
    NON_CONSUMING_SPLITS,
    PROTOCOL_ROLES,
    adaptation_budget_summary,
    assert_common_test_unchanged,
    assert_few_shot_adaptation_cohorts,
    assert_no_parent_leakage,
    attach_parent_recording,
    few_shot_adaptation_assignments,
    load_audit_catalog,
)


V3_PATH = "reports/phase2_2/protocol_b_few_shot_adaptation_v3.csv"
V2_PATH = "reports/phase2_1/protocol_b_common_test_v2.csv"

EXPECTED_BUDGET = {
    # machine: (observed, parent_groups, blocked_siblings, parent_closure)
    "RoboticArm": (27, 18, 96, 123),
    "BrushlessMotor": (18, 18, 0, 18),
}


@pytest.fixture(scope="module")
def v3() -> pd.DataFrame:
    return pd.read_csv(V3_PATH)


@pytest.fixture(scope="module")
def v2() -> pd.DataFrame:
    return pd.read_csv(V2_PATH)


def test_real_v3_split_satisfies_every_few_shot_invariant(v3):
    assert_few_shot_adaptation_cohorts(v3)


@pytest.mark.parametrize("machine", sorted(EXPECTED_BUDGET))
def test_four_adaptation_quantities_stay_distinct(v3, machine):
    budget = adaptation_budget_summary(v3).set_index("machine").loc[machine]
    observed, parents, blocked, closure = EXPECTED_BUDGET[machine]
    assert int(budget["target_adaptation_observed"]) == observed
    assert int(budget["target_adaptation_parent_groups"]) == parents
    assert int(budget["target_adaptation_blocked_siblings"]) == blocked
    assert int(budget["target_adaptation_parent_closure"]) == closure
    assert observed + blocked == closure
    assert bool(budget["blocked_parents_subset_of_adaptation_parents"])


def test_regime_t_consumes_released_allocation_not_parent_closure(v3):
    for machine, (observed, _, _, closure) in EXPECTED_BUDGET.items():
        part = v3[v3.machine == machine]
        source_train = len(part[(part.regime == "S") & (part.data_pool == "source_training")])
        s_train = len(part[(part.regime == "S") & (part.split == "train")])
        t_train = len(part[(part.regime == "T") & (part.split == "train")])
        assert s_train == source_train
        assert t_train == source_train + observed
        if closure != observed:
            assert t_train != source_train + closure


def test_blocked_siblings_are_consumed_by_nothing(v3):
    blocked = set(v3.loc[v3.data_pool == BLOCKED_SIBLING_POOL, "segment_id"])
    assert len(blocked) == 96
    for regime in ("S", "T"):
        for split in sorted(CONSUMING_SPLITS):
            used = set(v3.loc[(v3.regime == regime) & (v3.split == split), "segment_id"])
            assert not blocked & used, f"blocked sibling reached {regime}/{split}"
    assert set(v3.loc[v3.data_pool == BLOCKED_SIBLING_POOL, "split"]) == {
        BLOCKED_SIBLING_SPLIT
    }


def test_consumed_adaptation_is_exactly_the_released_target_normal_allocation(v3):
    for machine, part in v3.groupby("machine"):
        released = set(part.loc[
            (part.file_domain == "target")
            & (part.file_split == "train")
            & (part.anomaly_label.str.lower() == "normal"),
            "segment_id",
        ])
        consumed = set(part.loc[part.data_pool == "target_adaptation", "segment_id"])
        assert consumed == released
        parents = set(part.loc[part.data_pool == "target_adaptation", "parent_recording"])
        assert len(parents) == EXPECTED_BUDGET[machine][1]
        blocked_parents = set(part.loc[
            part.data_pool == BLOCKED_SIBLING_POOL, "parent_recording"
        ])
        assert blocked_parents <= parents


def test_every_segment_carries_exactly_one_primary_protocol_role(v3):
    assert set(v3.data_pool) <= set(PROTOCOL_ROLES)
    per_segment = v3.groupby(["machine", "regime", "segment_id"])["data_pool"].nunique()
    assert (per_segment == 1).all()


def test_common_test_is_bit_for_bit_the_phase_21_cohort(v3, v2):
    assert_common_test_unchanged(v2, v3)
    for machine, part in v3.groupby("machine"):
        s = part[(part.regime == "S") & (part.split == "test")]
        t = part[(part.regime == "T") & (part.split == "test")]
        assert set(s.segment_id) == set(t.segment_id)
        assert set(s.parent_recording) == set(t.parent_recording)
        old = v2[(v2.machine == machine) & (v2.regime == "T") & (v2.split == "test")]
        assert set(t.segment_id) == set(old.segment_id)
        assert set(t.parent_recording) == set(old.parent_recording)


def test_moving_a_blocked_sibling_into_training_is_rejected(v3):
    """Promoting one sibling in both regimes still fails the budget assertion."""

    tampered = v3.copy()
    victim = tampered.loc[
        tampered.data_pool == BLOCKED_SIBLING_POOL, "segment_id"
    ].iloc[0]
    promoted = tampered.segment_id == victim
    tampered.loc[promoted, "data_pool"] = "target_adaptation"
    tampered.loc[promoted & (tampered.regime == "T"), "split"] = "train"
    tampered.loc[promoted & (tampered.regime == "S"), "split"] = "reserved_adaptation"
    with pytest.raises(AssertionError, match="non-released|released allocation"):
        assert_few_shot_adaptation_cohorts(tampered)


def test_unbalancing_the_adaptation_pool_between_regimes_is_rejected(v3):
    tampered = v3.copy()
    victim = tampered.index[
        (tampered.data_pool == BLOCKED_SIBLING_POOL) & (tampered.regime == "T")
    ][0]
    tampered.loc[victim, ["split", "data_pool"]] = ["train", "target_adaptation"]
    with pytest.raises(AssertionError, match="adaptation pools differ"):
        assert_few_shot_adaptation_cohorts(tampered)


def test_moving_a_blocked_sibling_into_test_is_rejected(v3):
    tampered = v3.copy()
    victim = tampered.index[
        (tampered.data_pool == BLOCKED_SIBLING_POOL) & (tampered.regime == "T")
    ][0]
    tampered.loc[victim, ["split", "data_pool"]] = ["test", "common_evaluation"]
    with pytest.raises(AssertionError):
        assert_few_shot_adaptation_cohorts(tampered)


def test_parent_closure_training_would_now_fail_the_budget_assertion(v3):
    """The Phase-2.1 behaviour must no longer validate as primary Regime T."""

    parent_complete = v3.copy()
    mask = (parent_complete.data_pool == BLOCKED_SIBLING_POOL)
    parent_complete.loc[mask, "data_pool"] = "target_adaptation"
    parent_complete.loc[mask & (parent_complete.regime == "T"), "split"] = "train"
    parent_complete.loc[mask & (parent_complete.regime == "S"), "split"] = (
        "reserved_adaptation"
    )
    with pytest.raises(AssertionError, match="non-released|released allocation"):
        assert_few_shot_adaptation_cohorts(parent_complete)


def test_split_rebuild_is_deterministic(v3):
    catalog = attach_parent_recording(
        load_audit_catalog("reports/dataset_audit/segment_manifest.csv")
    )
    rebuilt = pd.concat([
        few_shot_adaptation_assignments(catalog, seed=1995, regime=regime)
        for regime in ("S", "T")
    ])
    key = ["machine", "regime", "segment_id"]
    left = rebuilt.set_index(key)[["split", "data_pool"]].sort_index()
    right = v3.set_index(key)[["split", "data_pool"]].sort_index()
    pd.testing.assert_frame_equal(left, right, check_dtype=False)


def test_non_consuming_splits_are_excluded_from_the_leakage_check():
    """A blocked sibling shares a parent with training data by design."""

    frame = pd.DataFrame({
        "machine": ["RoboticArm"] * 2,
        "regime": ["T"] * 2,
        "parent_recording": ["p1", "p1"],
        "split": ["train", BLOCKED_SIBLING_SPLIT],
    })
    assert_no_parent_leakage(frame, ignore_splits=NON_CONSUMING_SPLITS)
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_parent_leakage(frame)
    consuming = frame.assign(split=["train", "test"])
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_parent_leakage(consuming, ignore_splits=NON_CONSUMING_SPLITS)


def test_pipeline_refuses_to_materialise_a_blocked_sibling_split():
    config = load_config("configs/preprocessing/mechasight_few_shot_regime_t.json")
    empty = pd.DataFrame(columns=["split", "machine", "regime", "protocol"])
    with pytest.raises(ValueError, match="no computation"):
        next(iter_processed_windows(empty, config, split=BLOCKED_SIBLING_SPLIT))


def test_few_shot_configs_differ_only_by_regime_and_invalidate_earlier_caches():
    s = load_config("configs/preprocessing/mechasight_few_shot_regime_s.json")
    t = load_config("configs/preprocessing/mechasight_few_shot_regime_t.json")
    assert_preprocessing_pair(s, t)
    assert s.config_id != t.config_id
    historical = {
        load_config(f"configs/preprocessing/{name}.json").config_id
        for name in (
            "mechasight_regime_s", "mechasight_regime_t",
            "mechasight_common_test_regime_s", "mechasight_common_test_regime_t",
        )
    }
    assert s.config_id not in historical and t.config_id not in historical
    assert s.split_version == "protocol_b_few_shot_adaptation_v3"


def test_v3_config_rejects_the_phase_21_split_manifest(v2):
    with pytest.raises(ValueError, match="incompatible"):
        merge_protocol_b_assignments(
            pd.DataFrame({"machine": [], "segment_id": []}), v2, "T",
            split_version="protocol_b_few_shot_adaptation_v3",
        )


def test_evaluation_policy_records_the_budget_and_keeps_phase21_freezes():
    policy = load_evaluation_policy(
        "configs/evaluation/protocol_b_few_shot_adaptation_v3.json"
    )
    frozen = load_evaluation_policy("configs/evaluation/protocol_b_common_test_v2.json")
    for section in ("metrics", "threshold", "score_aggregation", "uncertainty"):
        assert policy[section] == frozen[section]
    assert policy["validation"]["forbidden"] == frozen["validation"]["forbidden"]
    for machine, (observed, parents, blocked, closure) in EXPECTED_BUDGET.items():
        entry = policy["adaptation_budget"][machine]
        assert entry["target_adaptation_observed"] == observed
        assert entry["target_adaptation_parent_groups"] == parents
        assert entry["target_adaptation_blocked_siblings"] == blocked
        assert entry["target_adaptation_parent_closure"] == closure
    use = policy["adaptation_budget"]["blocked_sibling_use"]
    assert not any(value for value in use.values() if isinstance(value, bool))
    assert policy["secondary_regimes"]["T-parent-complete"]["authorised_now"] is False


def test_phase21_artifacts_remain_intact(v2):
    from pathlib import Path

    for path in (
        "reports/phase2_1/protocol_b_common_test_v2.csv",
        "reports/phase2_1/protocol_b_common_test_v2_counts.csv",
        "reports/phase2_1/matched_test_identity_proof.csv",
        "reports/phase2_1/phase21_validation.json",
        "reports/phase2_1/phase21_report.md",
        "reports/phase2/protocol_b_split_v1.csv",
        "configs/evaluation/protocol_b_common_test_v2.json",
        "configs/preprocessing/mechasight_common_test_regime_s.json",
        "configs/preprocessing/mechasight_common_test_regime_t.json",
    ):
        assert Path(path).exists(), path
    counts = v2[v2.regime == "T"]
    ra = counts[(counts.machine == "RoboticArm") & (counts.data_pool == "target_adaptation")]
    assert len(ra) == 123, "the historical v2 interpretation must not be rewritten"
