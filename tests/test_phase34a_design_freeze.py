"""Phase 3.4A — invariants for the frozen (unexecuted) parent-overlap design.

These tests assert design properties only. Nothing here trains, scores, or
touches a checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.experiment.ledger_history import verify_ledger_history

CONFIG = Path("configs/phase3/roboticarm_parent_overlap_sensitivity_v1.json")
DESIGN = Path("reports/phase3_4_design")
ARMS = DESIGN / "proposed_arm_memberships.csv"
LEDGER = Path("reports/experiment_results/test_look_ledger.jsonl")

CONFIG_IDENTITY = "p34a-ra-overlap-13dbc49b2b983d72e957a320"
ARM_SEGMENTS = 214
ARM_WINDOWS = 15836
ARM_PARENTS = 14
REPLICATES = 5


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _arms() -> pd.DataFrame:
    return pd.read_csv(ARMS)


def test_design_is_marked_unexecuted_and_carries_no_performance_value():
    cfg = _config()
    assert cfg["design_status"] == "PREEXECUTION_DESIGN_ONLY"
    assert cfg["model_trained"] is False
    assert cfg["test_score_computed"] is False
    assert cfg["inference_run"] is False
    assert cfg["contains_any_performance_value_from_this_experiment"] is False
    assert cfg["config_identity"] == CONFIG_IDENTITY

    # No numeric performance value from this experiment may appear. AUC words are
    # expected -- the estimand is defined in terms of them -- so assert that every
    # AUC-related field is a *definition* (a string), never a number.
    def walk(node, path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}/{key}")
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{path}[{i}]")
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            leaf = path.rsplit("/", 1)[-1].lower()
            assert not any(t in leaf for t in ("auc", "auroc", "delta")), \
                f"numeric performance value at {path}: {node}"

    walk(_config())


def test_post_test_classification_is_recorded_permanently():
    cfg = _config()
    assert cfg["scientific_classification"] == "POST-TEST CONTROLLED SENSITIVITY ANALYSIS"
    note = cfg["classification_note"].lower()
    assert "after" in note and "post-test" in note


def test_arm_memberships_contain_no_score_column():
    arms = _arms()
    forbidden = {"total_loss", "score", "auc", "auroc", "anomaly_score", "reconstruction_error"}
    assert not (forbidden & {c.lower() for c in arms.columns}), arms.columns.tolist()


def test_every_arm_matches_on_segments_windows_parents_and_strata():
    arms = _arms()
    disjoint = arms[arms.arm == "disjoint"]
    assert len(disjoint) == ARM_SEGMENTS
    assert int(disjoint.windows.sum()) == ARM_WINDOWS
    assert disjoint.parent_id.nunique() == ARM_PARENTS
    assert disjoint.replicate.unique().tolist() == [0]
    assert not disjoint.parent_overlaps_official_test.any()

    reference = disjoint.groupby("strata").size()
    overlap = arms[arms.arm == "overlap"]
    assert sorted(overlap.replicate.unique()) == list(range(1, REPLICATES + 1))

    for replicate, sub in overlap.groupby("replicate"):
        assert len(sub) == ARM_SEGMENTS, replicate
        assert int(sub.windows.sum()) == ARM_WINDOWS, replicate
        assert sub.parent_id.nunique() == ARM_PARENTS, replicate
        assert sub.parent_overlaps_official_test.all(), replicate
        assert sub.groupby("strata").size().reindex(reference.index).equals(reference), replicate


def test_arms_are_disjoint_in_parent_overlap_status():
    """The single factor being varied must actually differ between the arms."""
    arms = _arms()
    disjoint_parents = set(arms[arms.arm == "disjoint"].parent_id)
    overlap_parents = set(arms[arms.arm == "overlap"].parent_id)
    assert not (disjoint_parents & overlap_parents), "a parent appears in both arms"


def test_new_protocol_identity_does_not_reuse_protocol_a():
    cfg = _config()
    policy = cfg["ledger_policy"]
    assert policy["new_protocol"] == "roboticarm_parent_overlap_sensitivity_v1"
    assert policy["must_not_reuse"] == "official_baseline_reproduction"
    assert policy["entries_appended_in_this_phase"] == 0


def test_no_ledger_entry_was_appended_for_this_design():
    """The V1 overlap design was superseded and never authorised: its protocol value
    must never appear. Stage note: ledger was 30 until step D (authorised at c10fdfc appended the ten
    Phase-3.4 looks atomically) and 40 at the Phase-3 close; later authorised stages append after
    that. Current invariant: every row belongs to an authorised stage and none is V1."""
    history = verify_ledger_history(LEDGER)
    assert not any(e["protocol"] == "roboticarm_parent_overlap_sensitivity_v1"
                   for e in history.rows)


def test_claim_boundary_forbids_the_overclaiming_language():
    cfg = _config()
    forbidden = " ".join(cfg["claim_boundary"]["forbidden"]).lower()
    assert "published paper is invalid" in forbidden
    assert "inflated by exactly" in forbidden
    assert "brushlessmotor" in forbidden
    assert cfg["primary_estimand"]["forbidden_name"] == "true leakage inflation"


def test_score_persistence_is_a_binding_requirement():
    cfg = _config()
    req = cfg["score_persistence_requirement"]
    assert req["binding"] is True
    for field in ("segment_id", "parent_id", "overlap_status", "arm", "seed", "total_loss"):
        assert field in req["must_persist_per_test_segment"], field


def test_residual_confounds_are_recorded_and_verdict_is_not_forced_green():
    cfg = _config()
    assert cfg["design_verdict"] == "YELLOW"
    assert len(cfg["residual_confounds"]) >= 3
    joined = " ".join(cfg["residual_confounds"]).lower()
    assert "one possible membership" in joined or "exhausted" in joined
    assert "shorter recordings" in joined
    assert "target" in joined
