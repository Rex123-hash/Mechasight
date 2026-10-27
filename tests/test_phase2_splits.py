from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessing.splits import (
    assert_no_parent_leakage,
    assert_no_window_sibling_leakage,
    robotic_arm_assignments,
)


def _catalog() -> pd.DataFrame:
    rows = []
    for condition_index, condition in enumerate(("weight00", "weight25")):
        domain = "source" if condition_index == 0 else "target"
        for recording in range(8):
            for segment in range(2):
                rows.append({
                    "machine": "RoboticArm",
                    "segment_id": f"2024010{condition_index + 1}_12_{recording:02d}_00_dataLog_{segment}",
                    "file_domain": domain,
                    "domain_shift_op": condition,
                    "domain_shift_env": "BckgA",
                    "anomaly_label": "normal",
                })
        for recording in range(2):
            rows.append({
                "machine": "RoboticArm",
                "segment_id": f"2024020{condition_index + 1}_13_{recording:02d}_00_dataLog_0",
                "file_domain": domain,
                "domain_shift_op": condition,
                "domain_shift_env": "BckgA",
                "anomaly_label": "fault",
            })
    return pd.DataFrame(rows)


def test_grouped_split_is_deterministic_and_order_independent():
    first = robotic_arm_assignments(_catalog(), seed=1995, regime="T")
    second = robotic_arm_assignments(
        _catalog().sample(frac=1.0, random_state=7), seed=1995, regime="T"
    )
    mapping_a = first.set_index("segment_id")["split"].sort_index()
    mapping_b = second.set_index("segment_id")["split"].sort_index()
    pd.testing.assert_series_equal(mapping_a, mapping_b)


def test_every_condition_has_two_normal_test_parents():
    result = robotic_arm_assignments(_catalog(), seed=1995, regime="T")
    normal_test = result[
        result["anomaly_label"].eq("normal") & result["split"].eq("test")
    ]
    counts = normal_test.groupby(["file_domain", "domain_shift_op"])["parent_recording"].nunique()
    assert (counts >= 2).all()


def test_regime_s_never_uses_target_normals_for_train_or_validation():
    result = robotic_arm_assignments(_catalog(), seed=1995, regime="S")
    target_normal = result[
        result["file_domain"].eq("target") & result["anomaly_label"].eq("normal")
    ]
    assert set(target_normal["split"]) == {"test"}


def test_anomalies_are_test_only():
    result = robotic_arm_assignments(_catalog(), seed=1995, regime="T")
    assert set(result.loc[result["anomaly_label"].ne("normal"), "split"]) == {"test"}


def test_parent_leakage_assertion_fails_loudly():
    bad = pd.DataFrame({
        "machine": ["RoboticArm", "RoboticArm"],
        "regime": ["T", "T"],
        "parent_recording": ["p", "p"],
        "split": ["train", "test"],
    })
    with pytest.raises(AssertionError, match="parent-recording leakage"):
        assert_no_parent_leakage(bad)


def test_window_sibling_assertion_checks_segments_too():
    bad = pd.DataFrame({
        "machine": ["RoboticArm", "RoboticArm"],
        "regime": ["T", "T"],
        "parent_recording": ["p1", "p2"],
        "segment_id": ["same", "same"],
        "split": ["train", "test"],
    })
    with pytest.raises(AssertionError, match="window siblings"):
        assert_no_window_sibling_leakage(bad)
