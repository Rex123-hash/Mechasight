"""Phase 3.4D — invariants for the frozen near-parent-balanced V3 execution design."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.experiment.focal_concordance import parent_concordance_delta
from src.experiment.ledger_history import verify_ledger_history

V2_EXEC = Path("configs/phase3/roboticarm_focal_toggle_execution_v1.json")
V3_EXEC = Path("configs/phase3/roboticarm_focal_toggle_execution_v2.json")
GATE = Path("reports/phase3_4d_release_gate")
MEMB = GATE / "final_arm_memberships.csv"
PLAN = GATE / "planned_experiments_v3.csv"
OLD_MEMB = Path("reports/phase3_4b_design/proposed_focal_arm_memberships.csv")
LEDGER = Path("reports/experiment_results/test_look_ledger.jsonl")

V3_IDENTITY = "p34d-exec-e5470b025229f59a06f4c772"
FOCAL_PARENTS, FOCAL_SEGMENTS, PLANNED_RUNS = 60, 203, 10


def _v3() -> dict:
    return json.loads(V3_EXEC.read_text(encoding="utf-8"))


def _memb() -> pd.DataFrame:
    return pd.read_csv(MEMB)


# ------------------------------------------------------- membership invariants
def test_focal_cohort_is_identical_to_v2_so_the_estimand_is_unchanged():
    new = set(_memb().query("role == 'focal_contribution'").parent_id)
    old = set(pd.read_csv(OLD_MEMB).query("role == 'focal_contribution'").parent_id)
    assert new == old
    assert len(new) == FOCAL_PARENTS


def test_each_focal_parent_appears_in_exactly_one_fold():
    focal = _memb().query("role == 'focal_contribution'")
    assert (focal.groupby("parent_id").fold.nunique() == 1).all()
    assert focal.groupby("fold").parent_id.nunique().tolist() == [12] * 5


def test_donor_and_focal_parents_are_disjoint_within_each_fold():
    m = _memb()
    for fold, sub in m.groupby("fold"):
        focal = set(sub.query("role == 'focal_contribution'").parent_id)
        donor = set(sub.query("role == 'donor_replacement'").parent_id)
        assert not (focal & donor), fold


def test_arms_match_exactly_on_segments_windows_and_all_25_strata():
    m = _memb()
    for fold, sub in m.groupby("fold"):
        focal = sub.query("role == 'focal_contribution'")
        donor = sub.query("role == 'donor_replacement'")
        assert len(focal) == len(donor), fold
        assert int(focal.windows.sum()) == int(donor.windows.sum()), fold
        assert focal.groupby("strata").size().equals(donor.groupby("strata").size()), fold


def test_target_domain_contribution_is_replaced_one_for_one():
    m = _memb()
    for fold, sub in m.groupby("fold"):
        f = sub.query("role == 'focal_contribution' and split_label == 'Normal_Target_Train'")
        d = sub.query("role == 'donor_replacement' and split_label == 'Normal_Target_Train'")
        assert len(f) == len(d), fold


def test_parent_exclusive_background_is_common_and_arms_stay_separate():
    """B* excludes BOTH focal and donor parents; each arm's parents are exclusive to it."""
    tr_parents = set(_memb().parent_id)
    m = _memb()
    for fold, sub in m.groupby("fold"):
        focal_p = set(sub.query("role == 'focal_contribution'").parent_id)
        donor_p = set(sub.query("role == 'donor_replacement'").parent_id)
        # no segment may appear in both arms
        fs = set(sub.query("role == 'focal_contribution'").segment_id)
        ds = set(sub.query("role == 'donor_replacement'").segment_id)
        assert not (fs & ds), fold
        assert focal_p.issubset(tr_parents) and donor_p.issubset(tr_parents)


def test_residual_parent_gap_is_small_and_recorded():
    balance = _v3()["parent_balance"]
    gaps = [int(v) for v in balance["residual_parent_gap"].values()]
    assert max(abs(g) for g in gaps) <= 3
    assert balance["folds_exactly_feasible_at_12_slots"] == 2
    assert sum(1 for v in balance["exact_12_slot_feasible_per_fold"].values() if not v) == 3


# --------------------------------------------------------------- frozen policy
def test_v3_supersedes_v2_and_v2_config_is_untouched():
    v3 = _v3()
    assert v3["config_identity"] == V3_IDENTITY
    assert v3["supersedes_execution_config"] == "p34c-exec-0eecc19f81e200f5c301ca25"
    v2 = json.loads(V2_EXEC.read_text(encoding="utf-8"))
    assert v2["config_identity"] == "p34c-exec-0eecc19f81e200f5c301ca25"
    assert v2["status"] == "PRETEST_EXECUTION_FROZEN"


def test_primary_estimand_and_seed_policy_are_unchanged():
    v3 = _v3()
    assert v3["primary_estimand"]["parent_aggregation"] == "MEAN"
    assert "parent-equal-weighted" in v3["primary_estimand"]["name"]
    assert v3["seed_policy"]["chosen"] == "POLICY A - ONE-SEED PRIMARY"
    assert v3["seed_policy"]["seed"] == 1995
    assert "NOT part" in v3["seed_policy"]["second_seed_status"]


def test_environment_and_run_counts_are_unchanged():
    v3 = _v3()
    assert v3["execution_environment"]["execution_environment_id"] == "p34c-env-b7a5de2af4c1d9614e49"
    assert v3["execution_environment"]["device"] == "cpu"
    assert v3["planned_training_runs"] == PLANNED_RUNS
    assert v3["planned_test_looks"] == PLANNED_RUNS
    plan = pd.read_csv(PLAN)
    assert len(plan) == PLANNED_RUNS
    assert (plan.seed == 1995).all()
    assert (plan.status == "PLANNED_NOT_EXECUTED").all()
    assert len(set(plan.experiment_id)) == PLANNED_RUNS


def test_bootstrap_is_frozen_as_fold_stratified():
    unc = _v3()["uncertainty"]
    assert "FOLD-STRATIFIED" in unc["frozen"]
    assert unc["replicates"] == 10000
    assert unc["seed"] == 27182
    assert unc["explicitly_conditional_on"] == "the ten fitted models"
    assert "N=5 folds is never an inferential sample" in unc["prohibition"]


def test_fold_stratified_bootstrap_holds_fold_composition_fixed():
    """Stratifying must not change the point estimate, and must be reproducible."""
    rng = np.random.default_rng(3)
    anomaly = rng.normal(3.0, 1.0, 232)
    rows = []
    for arm in ("PRESENT", "ABSENT"):
        for j, a in enumerate(anomaly):
            rows.append({"segment_id": f"a{j}", "parent_id": f"ap{j}", "arm": arm,
                         "anomaly_label": "anomaly", "focal_fold": np.nan, "total_loss": a})
        for fold in range(1, 6):
            for i in range(12):
                for s, base in enumerate(rng.normal(3.0, 1.0, 3)):
                    rows.append({"segment_id": f"f{fold}p{i}s{s}", "parent_id": f"f{fold}p{i}",
                                 "arm": arm, "anomaly_label": "normal", "focal_fold": fold,
                                 "total_loss": base - (0.4 if arm == "PRESENT" else 0.0)})
    scores = pd.DataFrame(rows)
    a = parent_concordance_delta(scores, replicates=1000, seed=27182, stratify_by_fold=True)
    b = parent_concordance_delta(scores, replicates=1000, seed=27182, stratify_by_fold=True)
    free = parent_concordance_delta(scores, replicates=1000, seed=27182, stratify_by_fold=False)

    assert a.n_parents == FOCAL_PARENTS and len(a.fold_deltas) == 5
    assert a.bootstrap_stratified_by_fold is True and free.bootstrap_stratified_by_fold is False
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)          # reproducible
    assert a.point_estimate == pytest.approx(free.point_estimate)  # estimator unchanged
    assert a.interval_is_conditional_on_trained_models is True


def test_no_segment_pseudo_replication_in_the_primary_unit():
    """The estimator must average parents, never the 203 segments."""
    rng = np.random.default_rng(4)
    anomaly = rng.normal(3.0, 1.0, 232)
    rows = []
    layout = {"big": 8, "small": 1}
    base = {p: rng.normal(3.0, 1.0, n) for p, n in layout.items()}
    for arm in ("PRESENT", "ABSENT"):
        for j, a in enumerate(anomaly):
            rows.append({"segment_id": f"a{j}", "parent_id": f"ap{j}", "arm": arm,
                         "anomaly_label": "anomaly", "focal_fold": np.nan, "total_loss": a})
        for parent, values in base.items():
            shift = 1.5 if (arm == "PRESENT" and parent == "big") else 0.0
            for s, v in enumerate(values):
                rows.append({"segment_id": f"{parent}_{s}", "parent_id": parent, "arm": arm,
                             "anomaly_label": "normal", "focal_fold": 1, "total_loss": v - shift})
    result = parent_concordance_delta(pd.DataFrame(rows), bootstrap=False)
    assert result.point_estimate == pytest.approx(result.parent_deltas.mean(), abs=1e-15)
    assert result.point_estimate != pytest.approx(result.segment_deltas.mean(), abs=1e-6)


def test_no_performance_value_is_present_anywhere_in_v3():
    def walk(node, path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}/{key}")
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{path}[{i}]")
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            leaf = path.rsplit("/", 1)[-1].lower()
            assert not any(t in leaf for t in ("auc", "auroc", "concordance_value",
                                               "point_estimate", "delta_value")), \
                f"performance value at {path}: {node}"

    v3 = _v3()
    walk(v3)
    assert v3["model_trained"] is False
    assert v3["test_score_computed"] is False
    assert v3["readiness_verdict"] == "YELLOW"


def test_ledger_holds_exactly_ten_phase34_looks_bound_to_v3():
    """Stage note: ledger was 30 until step D (authorised at c10fdfc appended the ten
    Phase-3.4 looks atomically) and 40 = 28 B + 2 A + 10 Phase-3.4 at the Phase-3
    close; later authorised stages append after. Every Phase-3.4 look in the whole
    ledger binds the V3 config identity and the frozen seed."""
    history = verify_ledger_history(LEDGER)
    p34 = [e for e in history.rows if e["protocol"] == "roboticarm_focal_parent_sibling_toggle_v1"]
    assert p34 == list(history.rows_of("phase3_4_focal_parent_sibling_toggle"))
    assert len(p34) == 10
    plan = pd.read_csv(PLAN)
    assert {e["experiment_id"] for e in p34} == set(plan.experiment_id)
    for e in p34:
        assert e["model_config_identity"] == V3_IDENTITY
        assert e["regime"] == "T"
        assert "seed=1995;" in e["reason"]


def test_no_stray_score_artifact_exists():
    """Stage note: 'no checkpoint' was the pre-training invariant (step B, e93886c);
    'no score' was the pre-step-E invariant (step E1, ac71988). Lasting invariant: no
    per-segment score table outside reports/phase3_4_results/scores (+ the E1 combined)."""
    for path in Path("reports").glob("phase3_4*/**/*.csv"):
        # combined raw tables (E1, fold-1 recovery) are hashed deliverables of scoring stages
        if path.parent == Path("reports/phase3_4_results/scores") or path.name.endswith("_combined_raw_scores.csv"):
            continue
        columns = {c.lower() for c in pd.read_csv(path, nrows=0).columns}
        assert "total_loss" not in columns, path
