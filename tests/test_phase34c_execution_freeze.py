"""Phase 3.4C — invariants for the frozen (unexecuted) execution policy."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.experiment.ledger_history import verify_ledger_history
from src.experiment.focal_concordance import (
    concordance_per_segment,
    parent_concordance_delta,
    segment_weighted_auroc,
)

EXEC = Path("configs/phase3/roboticarm_focal_toggle_execution_v1.json")
FREEZE = Path("reports/phase3_4c_freeze")
MEMB = Path("reports/phase3_4b_design/proposed_focal_arm_memberships.csv")
LEDGER = Path("reports/experiment_results/test_look_ledger.jsonl")

EXEC_IDENTITY = "p34c-exec-0eecc19f81e200f5c301ca25"
FOCAL_PARENTS = 60
FOCAL_SEGMENTS = 203
ANOMALY_REFERENCE = 232
PLANNED_RUNS = 10


def _cfg() -> dict:
    return json.loads(EXEC.read_text(encoding="utf-8"))


# ------------------------------------------------------------- cohort/folds
def test_focal_cohort_is_exactly_60_parents_and_203_segments():
    cohort = pd.read_csv(FREEZE / "focal_cohort_manifest.csv")
    assert len(cohort) == FOCAL_SEGMENTS
    assert cohort.parent_id.nunique() == FOCAL_PARENTS
    assert (cohort.anomaly_label == "normal").all()
    assert sorted(cohort.focal_fold.unique()) == [1, 2, 3, 4, 5]
    # a parent belongs to exactly one fold
    assert (cohort.groupby("parent_id").focal_fold.nunique() == 1).all()


def test_excluded_focal_parents_account_for_exactly_nine_segments():
    ex = pd.read_csv(FREEZE / "excluded_focal_parents.csv")
    assert len(ex) == 2
    assert int(ex.normal_test_segments_excluded.sum()) == 9
    assert (ex.donor_shortfall_segments == 1).all()
    assert (ex.status == "EXCLUDED_FROM_PRIMARY_FOCAL_COHORT").all()


def test_folds_and_donor_matching_are_still_exact():
    memb = pd.read_csv(MEMB)
    focal = memb[memb.role == "focal_contribution"]
    donor = memb[memb.role == "donor_replacement"]
    assert focal.parent_id.nunique() == FOCAL_PARENTS
    for fold in sorted(memb.fold.unique()):
        f = focal[focal.fold == fold]
        d = donor[donor.fold == fold]
        assert len(f) == len(d), fold
        assert int(f.windows.sum()) == int(d.windows.sum()), fold
        assert f.groupby("strata").size().equals(d.groupby("strata").size()), fold
        assert not (set(d.parent_id) & set(f.parent_id)), fold


# --------------------------------------------------------------- estimator
def test_segment_weighted_mean_of_C_equals_strict_auroc():
    rng = np.random.default_rng(11)
    for _ in range(5):
        normal = rng.normal(0, 1, 23).round(2)     # rounding injects ties
        anomaly = rng.normal(0.5, 1, ANOMALY_REFERENCE).round(2)
        assert concordance_per_segment(normal, anomaly).mean() == pytest.approx(
            segment_weighted_auroc(normal, anomaly), abs=1e-15)


def test_concordance_is_invariant_to_monotone_rescaling_of_that_arm():
    rng = np.random.default_rng(12)
    normal, anomaly = rng.normal(0, 1, 30), rng.normal(0.5, 1, ANOMALY_REFERENCE)
    base = concordance_per_segment(normal, anomaly)
    for transform in (lambda x: x * 1000.0, lambda x: x + 7.0, np.exp):
        assert np.array_equal(base, concordance_per_segment(transform(normal),
                                                            transform(anomaly)))


def test_exact_ties_score_zero_like_upstream():
    normal = np.array([1.0, 1.0])
    anomaly = np.array([1.0, 2.0])
    # first anomaly ties both normals -> zero credit; second beats both.
    assert np.array_equal(concordance_per_segment(normal, anomaly), np.array([0.5, 0.5]))


def _synthetic_scores(shift_by_parent: dict[str, float], global_shift: float = 0.0):
    """Two arms over the same segments; ABSENT is globally rescaled.

    Normals overlap the anomaly reference so C_i is unsaturated -- with normals
    far below every anomaly, C_i is 1.0 in both arms and every delta is
    trivially zero, which would make these tests vacuous.
    """
    rng = np.random.default_rng(99)
    anomaly = rng.normal(3.0, 1.0, ANOMALY_REFERENCE)
    normal_base = {parent: rng.normal(3.0, 1.0, 3) for parent in shift_by_parent}
    rows = []
    for arm in ("PRESENT", "ABSENT"):
        offset = global_shift if arm == "ABSENT" else 0.0
        for j, a in enumerate(anomaly):
            rows.append({"segment_id": f"anom_{j}", "parent_id": f"apar_{j}", "arm": arm,
                         "anomaly_label": "anomaly", "focal_fold": np.nan,
                         "total_loss": a + offset})
        for parent, shift in shift_by_parent.items():
            for s, base in enumerate(normal_base[parent]):
                value = base - (shift if arm == "PRESENT" else 0.0)
                rows.append({"segment_id": f"{parent}_{s}", "parent_id": parent, "arm": arm,
                             "anomaly_label": "normal", "focal_fold": 1,
                             "total_loss": value + offset})
    return pd.DataFrame(rows)


def test_estimator_is_unmoved_by_a_pure_global_scale_shift():
    """The property that disqualified the raw-score difference must not bite here."""
    parents = {f"p{i}": 0.0 for i in range(6)}
    plain = parent_concordance_delta(_synthetic_scores(parents), bootstrap=False)
    shifted = parent_concordance_delta(
        _synthetic_scores(parents, global_shift=25.0), bootstrap=False)
    assert plain.point_estimate == pytest.approx(shifted.point_estimate, abs=1e-12)


def test_parents_are_weighted_equally_not_by_segment_count():
    rng = np.random.default_rng(7)
    anomaly = rng.normal(3.0, 1.0, ANOMALY_REFERENCE)
    # "big" contributes 8 segments, "small" contributes 1; only big improves.
    # Normals overlap the anomalies so the concordance is not saturated.
    layout = {"big": 8, "small": 1}
    base = {p: rng.normal(3.0, 1.0, n) for p, n in layout.items()}
    rows = []
    for arm in ("PRESENT", "ABSENT"):
        for j, a in enumerate(anomaly):
            rows.append({"segment_id": f"anom_{j}", "parent_id": f"apar_{j}", "arm": arm,
                         "anomaly_label": "anomaly", "total_loss": a})
        for parent, values in base.items():
            improve = 1.5 if (arm == "PRESENT" and parent == "big") else 0.0
            for s, value in enumerate(values):
                rows.append({"segment_id": f"{parent}_{s}", "parent_id": parent, "arm": arm,
                             "anomaly_label": "normal", "total_loss": value - improve})
    scores = pd.DataFrame(rows)
    result = parent_concordance_delta(scores, bootstrap=False)

    assert result.n_parents == 2 and result.n_segments == 9
    # parent-equal weighting: the mean of the two parent deltas, NOT of 9 segments
    assert result.point_estimate == pytest.approx(result.parent_deltas.mean(), abs=1e-15)
    segment_weighted = result.segment_deltas.mean()
    assert result.point_estimate != pytest.approx(segment_weighted, abs=1e-6), \
        "parent weighting must differ from segment weighting on an unbalanced cohort"


def test_bootstrap_resamples_parents_and_is_reproducible():
    parents = {f"p{i}": 0.3 for i in range(12)}
    scores = _synthetic_scores(parents)
    a = parent_concordance_delta(scores, replicates=500, seed=27182)
    b = parent_concordance_delta(scores, replicates=500, seed=27182)
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)
    assert a.ci_low <= a.point_estimate <= a.ci_high
    assert a.bootstrap_seed == 27182
    assert a.interval_is_conditional_on_trained_models is True


# ------------------------------------------------------------ frozen policy
def test_seed_policy_is_resolved_to_a_single_seed():
    cfg = _cfg()
    policy = cfg["seed_policy"]
    assert policy["chosen"] == "POLICY A - ONE-SEED PRIMARY"
    assert policy["seed"] == 1995
    assert policy["models"] == PLANNED_RUNS
    assert "NOT part" in policy["second_seed_status"]
    assert "optional" in policy["binding_condition"]


def test_planned_run_and_test_look_counts_agree_everywhere():
    cfg = _cfg()
    plan = pd.read_csv(FREEZE / "planned_experiments.csv")
    assert len(plan) == PLANNED_RUNS
    assert cfg["planned_training_runs"] == PLANNED_RUNS
    assert cfg["planned_test_looks"] == PLANNED_RUNS
    assert cfg["ledger_policy"]["entries_to_authorise"] == PLANNED_RUNS
    assert len(set(plan.experiment_id)) == PLANNED_RUNS
    assert sorted(plan.arm.unique()) == ["ABSENT", "PRESENT"]
    assert (plan.seed == 1995).all()
    assert (plan.status == "PLANNED_NOT_EXECUTED").all()
    # counterbalanced order: the leading arm alternates by fold parity
    lead = plan.sort_values("execution_position").groupby("fold").first().arm
    assert lead.loc[1] == "PRESENT" and lead.loc[2] == "ABSENT"


def test_score_schema_covers_every_field_needed_for_later_re_derivation():
    schema = json.loads((FREEZE / "score_schema.json").read_text(encoding="utf-8"))
    assert schema["binding"] is True
    names = {c["name"] for c in schema["columns"]}
    for field in ("experiment_id", "fold", "arm", "seed", "segment_id", "parent_id",
                  "focal_parent_status", "focal_fold", "split_label", "anomaly_label",
                  "domain_shift_op", "domain_shift_env", "total_loss", "checkpoint_sha256",
                  "evaluator_git_sha", "test_cohort_hash", "execution_environment_id"):
        assert field in names, field
    assert schema["rows_per_experiment"] == 464
    assert schema["total_rows_planned"] == 464 * PLANNED_RUNS


def test_environment_is_frozen_and_device_is_cpu():
    env = json.loads((FREEZE / "execution_environment.json").read_text(encoding="utf-8"))
    assert env["device"] == "cpu"
    assert env["cuda_available"] is False
    assert env["execution_environment_id"].startswith("p34c-env-")
    assert "no CPU/CUDA switch" in env["no_device_switch_rule"]


def test_status_is_pretest_and_no_performance_value_is_present():
    cfg = _cfg()
    assert cfg["status"] == "PRETEST_EXECUTION_FROZEN"
    assert cfg["model_trained"] is False
    assert cfg["test_score_computed"] is False
    assert cfg["inference_run"] is False
    assert cfg["contains_any_performance_value_from_this_experiment"] is False
    assert cfg["readiness_verdict"] == "YELLOW"

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

    walk(cfg)


def test_ledger_holds_exactly_the_ten_authorised_phase34_looks():
    """Stage note: ledger was 30 until step D (authorised at c10fdfc appended the ten
    Phase-3.4 looks atomically). Historical scope: 40 = 28 B + 2 A + 10 Phase-3.4 at the
    Phase-3 close; the 40-entry prefix is append-only and byte-stable, and later
    authorised stages append after it. The ten looks are rows 30..40 and nowhere else."""
    history = verify_ledger_history(LEDGER)
    entries = list(history.rows)
    protocols = {e.get("protocol") for e in entries[:40]}
    assert protocols == {"mechasight_experimental_protocol", "official_baseline_reproduction",
                         "roboticarm_focal_parent_sibling_toggle_v1"}
    assert [e.get("protocol") for e in entries[30:40]] == \
        ["roboticarm_focal_parent_sibling_toggle_v1"] * 10
    assert not any(e.get("protocol") == "roboticarm_focal_parent_sibling_toggle_v1"
                   for e in entries[:30] + entries[40:])


def test_score_artifacts_exist_only_in_the_authorised_scores_directory():
    """Stage note: 'no checkpoint' was the pre-training invariant (step B, e93886c);
    'no score' was the pre-step-E invariant (step E1, ac71988). The lasting invariant
    is that per-segment scores exist ONLY under reports/phase3_4_results/scores and
    ONLY for ledger-authorised experiment ids; no other Phase-3.4 directory holds one."""
    import json
    ledger = Path("reports/experiment_results/test_look_ledger.jsonl")
    authorised = {json.loads(l)["experiment_id"] for l in ledger.read_text(encoding="utf-8").splitlines()
                  if l.strip() and "roboticarm_focal_parent_sibling_toggle_v1" in l}
    for p in Path("reports/phase3_4_results").glob("scores/*.csv"):
        assert p.stem.split("__")[0] in authorised, p
    for path in Path("reports").glob("phase3_4*/**/*.csv"):
        # combined raw tables (E1, fold-1 recovery) are hashed deliverables of scoring stages
        if path.parent == Path("reports/phase3_4_results/scores") or path.name.endswith("_combined_raw_scores.csv"):
            continue
        columns = {c.lower() for c in pd.read_csv(path, nrows=0).columns}
        assert "total_loss" not in columns, path
