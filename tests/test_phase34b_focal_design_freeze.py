"""Phase 3.4B — invariants for the frozen (unexecuted) focal-parent toggle design."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.experiment.ledger_history import verify_ledger_history

V1 = Path("configs/phase3/roboticarm_parent_overlap_sensitivity_v1.json")
V2 = Path("configs/phase3/roboticarm_parent_overlap_sensitivity_v2.json")
DESIGN = Path("reports/phase3_4b_design")
MEMB = DESIGN / "proposed_focal_arm_memberships.csv"
FOLDS = DESIGN / "focal_fold_candidates.csv"
LEDGER = Path("reports/experiment_results/test_look_ledger.jsonl")

V2_IDENTITY = "p34b-ra-focal-2ab15394f78b560f1e8c0880"
V1_IDENTITY = "p34a-ra-overlap-13dbc49b2b983d72e957a320"
K = 5


def _v2() -> dict:
    return json.loads(V2.read_text(encoding="utf-8"))


def test_v1_config_is_untouched_and_still_frozen():
    v1 = json.loads(V1.read_text(encoding="utf-8"))
    assert v1["config_identity"] == V1_IDENTITY
    assert v1["design_status"] == "PREEXECUTION_DESIGN_ONLY"
    assert v1["design_verdict"] == "YELLOW"
    assert v1["model_trained"] is False


def test_v2_is_unexecuted_and_carries_no_performance_value():
    cfg = _v2()
    assert cfg["config_identity"] == V2_IDENTITY
    assert cfg["design_status"] == "PREEXECUTION_DESIGN_ONLY"
    assert cfg["model_trained"] is False
    assert cfg["test_score_computed"] is False
    assert cfg["inference_run"] is False
    assert cfg["contains_any_performance_value_from_this_experiment"] is False
    assert cfg["v1_config_edited"] is False
    assert cfg["supersedes_as_primary"] == V1_IDENTITY

    def walk(node, path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}/{key}")
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{path}[{i}]")
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            leaf = path.rsplit("/", 1)[-1].lower()
            assert not any(t in leaf for t in ("auc", "auroc", "delta", "score_shift")), \
                f"numeric performance value at {path}: {node}"

    walk(cfg)


def test_post_test_classification_preserved():
    cfg = _v2()
    assert cfg["scientific_classification"] == "POST-TEST CONTROLLED SENSITIVITY ANALYSIS"


def test_every_fold_matches_focal_and_donor_exactly():
    folds = pd.read_csv(FOLDS)
    assert len(folds) == K
    assert folds.size_match_exact.all()
    assert folds.window_match_exact.all()
    assert folds.strata_match_exact.all()
    assert folds.donor_parents_disjoint_from_focal.all()
    assert (folds.focal_train_segments == folds.donor_train_segments).all()
    # target-domain contributions are replaced one for one
    assert (folds.target_domain_focal_segments == folds.target_domain_donor_segments).all()
    # the whole point of v2: near-official training volume, unlike v1's 11.6%
    assert (folds.pct_of_official_training_volume > 75).all()


def test_memberships_carry_no_scores_and_partition_focal_parents():
    m = pd.read_csv(MEMB)
    forbidden = {"total_loss", "score", "auc", "auroc", "anomaly_score", "reconstruction_error"}
    assert not (forbidden & {c.lower() for c in m.columns})
    assert set(m.role) == {"focal_contribution", "donor_replacement"}

    focal_by_fold = {k: set(sub[sub.role == "focal_contribution"].parent_id)
                     for k, sub in m.groupby("fold")}
    # folds are parent-disjoint from each other
    seen: set[str] = set()
    for parents in focal_by_fold.values():
        assert not (seen & parents), "a focal parent appears in two folds"
        seen |= parents
    # donors are never drawn from that fold's focal parents
    for k, sub in m.groupby("fold"):
        donors = set(sub[sub.role == "donor_replacement"].parent_id)
        assert not (donors & focal_by_fold[k]), k


def test_new_protocol_identity_is_distinct_from_both_earlier_protocols():
    policy = _v2()["ledger_policy"]
    assert policy["new_protocol"] == "roboticarm_focal_parent_sibling_toggle_v1"
    assert "official_baseline_reproduction" in policy["must_not_reuse"]
    assert "roboticarm_parent_overlap_sensitivity_v1" in policy["must_not_reuse"]
    assert policy["entries_appended_in_this_phase"] == 0


def test_only_the_authorised_phase_34_looks_were_appended():
    """Stage note: ledger was 30 until step D (authorised at c10fdfc appended the ten
    Phase-3.4 looks atomically); 40 = 28 B + 2 A + 10 Phase-3.4 at the Phase-3 close,
    with later authorised stages appended after. In the whole ledger the superseded V1
    protocol value never appears; the V2/V3 protocol appears exactly ten times, once
    per frozen experiment, all appended at step D as rows 30..40."""
    history = verify_ledger_history(LEDGER)
    entries = list(history.rows)
    protocols = {e.get("protocol") for e in entries}
    assert "roboticarm_parent_overlap_sensitivity_v1" not in protocols
    p34 = [e for e in entries if e.get("protocol") == "roboticarm_focal_parent_sibling_toggle_v1"]
    assert len(p34) == 10 and len({e["experiment_id"] for e in p34}) == 10
    assert {e["git_sha"] for e in p34} == {"c10fdfcc890f97a595c132d87dc8b4bb04d7161e"}
    assert p34 == entries[30:40]


def test_claim_boundary_keeps_the_estimand_narrow():
    cfg = _v2()
    boundary = cfg["claim_boundary"]
    assert "SAME-PARENT FAMILIARITY" in boundary["measures"]
    assert "not GLOBAL SPLIT LEAKAGE" in boundary["measures"]
    forbidden = " ".join(boundary["forbidden"]).lower()
    assert "global split-leakage penalty" in forbidden
    assert "brushlessmotor" in forbidden
    assert cfg["causal_language"]["either_design_is_causal"] is False


def test_v1_review_findings_are_recorded_with_severity():
    review = _v2()["v1_review"]
    assert review["parent_contribution_finding"]["severity"] == "MATERIAL"
    assert review["estimand_dependence_finding"]["severity"] == "MATERIAL"
    assert review["exact_contribution_matching_feasible"] is False
    assert review["estimand_dependence_finding"]["preferable_to_v1_definition"] is True
