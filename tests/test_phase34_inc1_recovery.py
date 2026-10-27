"""Phase 3.4 INC-1 — recovery freeze invariants, gates and harness-guard proofs.

Nothing here reads protected test data. Every scorer invocation runs against a
temporary results root with a fake checkpoint; the real-state checks are read-only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from src.experiment import focal_toggle_execution as fx
from src.experiment.ledger import TestLookLedger, TestLookRecord
from src.experiment.ledger_history import PHASE3_CLOSE_ROWS, verify_ledger_history
from src.experiment.provenance import canonical_sha256, file_sha256

REC = Path("configs/phase3/roboticarm_focal_toggle_inc1_recovery_v1.json")
RESULTS = Path("reports/phase3_4_results")
LEDGER = Path("reports/experiment_results/test_look_ledger.jsonl")
MEMB = Path("reports/phase3_4d_release_gate/final_arm_memberships.csv")
PLAN = Path("reports/phase3_4d_release_gate/planned_experiments_v3.csv")

INC1 = "p34d-92fd40fc93c8c8d2e819"
F1A = "p34d-e56529a2f56f98fd4a7a"
RETAINED_ORDER = ["2A", "2P", "3P", "3A", "4A", "4P", "5P", "5A"]
RETAINED_IDS = ["p34d-fd6e3bfd4c091e155d99", "p34d-a4e8316f62fa982ec52d",
                "p34d-88b93ddbacbc5829f326", "p34d-2af84983abd170013b73",
                "p34d-652bc1e3f9d29a51a460", "p34d-406b59f08e154b284159",
                "p34d-66656cf0901b180cf037", "p34d-ed8467570acf122c36cd"]


def _rec() -> dict:
    return json.loads(REC.read_text(encoding="utf-8"))


def _ledger() -> list[dict]:
    return [json.loads(l) for l in LEDGER.read_text(encoding="utf-8").splitlines() if l.strip()]


# ------------------------------------------------------------ frozen config
def test_recovery_config_identity_recomputes_from_its_payload():
    import hashlib
    c = _rec()
    payload = {
        "incident": "INC-1",
        "supersedes_analysis_of": c["parent_config_identity"],
        "retained_folds": c["salvage_analysis"]["retained_folds"],
        "retained_experiment_ids": c["salvage_analysis"]["retained_experiment_ids"],
        "excluded_experiment_ids": [c["frozen_decisions"]["fold_1_present"]["experiment_id"],
                                    c["frozen_decisions"]["fold_1_absent"]["experiment_id"]],
        "bootstrap": ["fold-stratified", c["salvage_analysis"]["bootstrap"]["replicates"],
                      c["salvage_analysis"]["bootstrap"]["seed"]],
        "seed": 1995,
        "estimator": "parent-equal-weighted focal concordance delta, unchanged definition",
    }
    expected = "p34r-inc1-" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
    assert c["config_identity"] == expected
    assert c["parent_config_identity"] == "p34d-exec-e5470b025229f59a06f4c772"
    assert c["parent_config_document_sha256"] == canonical_sha256(
        Path("configs/phase3/roboticarm_focal_toggle_execution_v2.json"))


def test_frozen_decisions_no_retry_and_fold_1_absent_unused():
    c = _rec()
    fp, fa = c["frozen_decisions"]["fold_1_present"], c["frozen_decisions"]["fold_1_absent"]
    assert fp["experiment_id"] == INC1 and fp["decision"] == "NO RETRY"
    assert "CONSUMED / INCOMPLETE" in fp["state"]
    assert fa["experiment_id"] == F1A and fa["decision"] == "DO NOT SCORE"
    assert "AUTHORIZED / UNUSED" in fa["state"]
    assert c["original_primary_analysis"]["status"] == "NOT FULLY ESTIMABLE DUE TO INC-1"
    assert c["salvage_analysis"]["label"] == "INCIDENT-CONTINGENT SECONDARY SENSITIVITY ANALYSIS"
    assert "confirmatory" in c["salvage_analysis"]["not_to_be_called"]


def test_retained_cohort_is_exactly_the_eight_intact_looks_in_frozen_order():
    c = _rec()
    plan = pd.read_csv(PLAN).sort_values("execution_position")
    assert c["scoring_plan"]["frozen_order"] == RETAINED_ORDER
    assert c["scoring_plan"]["frozen_order_experiment_ids"] == RETAINED_IDS
    assert c["salvage_analysis"]["retained_experiment_ids"] == RETAINED_IDS
    # derived from the execution matrix with fold 1 removed, order preserved
    assert plan[plan.fold != 1].experiment_id.tolist() == RETAINED_IDS
    assert [f"{r.fold}{r.arm[0]}" for r in plan[plan.fold != 1].itertuples()] == RETAINED_ORDER
    by_fold = plan[plan.fold != 1].groupby("fold").arm.apply(set)
    assert all(v == {"PRESENT", "ABSENT"} for v in by_fold) and list(by_fold.index) == [2, 3, 4, 5]
    assert set(RETAINED_IDS) | {INC1, F1A} == set(plan.experiment_id)
    assert [e.experiment_id for e in fx.retained_experiments()] == RETAINED_IDS


def test_retained_parent_cohort_is_48_from_frozen_memberships():
    c = _rec()
    focal = pd.read_csv(MEMB).query("role == 'focal_contribution'")
    per_fold = focal.groupby("fold").parent_id.nunique().to_dict()
    assert per_fold == {1: 12, 2: 12, 3: 12, 4: 12, 5: 12}
    retained = focal[focal.fold.isin([2, 3, 4, 5])]
    assert retained.parent_id.nunique() == 48 == c["salvage_analysis"]["expected_focal_parents"]
    assert c["salvage_analysis"]["focal_parents_per_retained_fold"] == {"2": 12, "3": 12, "4": 12, "5": 12}
    assert c["salvage_analysis"]["expected_persisted_score_rows"] == 464 * 8


def test_salvage_bootstrap_is_the_original_restricted_to_four_folds():
    b = _rec()["salvage_analysis"]["bootstrap"]
    parent = json.loads(Path("configs/phase3/roboticarm_focal_toggle_execution_v2.json")
                        .read_text(encoding="utf-8"))["uncertainty"]
    assert b["folds"] == [2, 3, 4, 5] and b["parents_per_fold"] == 12 and b["resampled_parents"] == 48
    assert b["replicates"] == parent["replicates"] == 10000
    assert b["seed"] == parent["seed"] == 27182
    assert b["interval"] == "percentile 95 percent"
    assert "eight fitted models and the four retained folds" in b["conditional_on"]
    assert "uncertainty from the missing fold-1 pair" in b["does_not_include"]


def test_recovery_config_carries_incident_evidence_and_no_performance_value():
    c = _rec()
    inc = c["incident"]
    claim = json.loads((RESULTS / "score_claims" / f"{INC1}.claim").read_text(encoding="utf-8"))
    auth = next(e for e in _ledger() if e.get("experiment_id") == INC1)
    assert inc["claim_timestamp_utc"] == claim["timestamp_utc"]
    assert inc["authorization_timestamp_utc"] == auth["timestamp_utc"]
    assert inc["checkpoint_sha256"] == claim["checkpoint_sha256"]
    assert inc["attempt_history"] == ["SCORE_STARTED"]
    assert inc["protected_forward_passes"] == 0
    assert inc["score_completed_exists"] is False and inc["score_artifact_exists"] is False
    assert c["contains_any_performance_value"] is False
    assert c["inference_run"] is False and c["test_score_computed"] is False

    def walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}/{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            leaf = path.rsplit("/", 1)[-1].lower()
            assert not any(t in leaf for t in ("auc", "auroc", "concordance_value", "point_estimate",
                                               "delta_value", "ci95")), path
    walk(c)


# ------------------------------------------------- real state (read-only)
def test_inc1_is_preserved_exactly_and_the_original_namespace_matches_e1():
    """Stage note: 'everything else unclaimed' held until step E1 (ac71988). INC-1's
    original record must stay byte-identical forever; the ORIGINAL namespace (no
    '__' recovery suffix) now holds exactly INC-1 + the eight completed retained looks."""
    orig = lambda p: "__" not in p.stem
    claims = sorted(p.stem for p in RESULTS.glob("score_claims/*.claim") if orig(p))
    attempts = sorted(p.stem for p in RESULTS.glob("score_attempts/*.jsonl") if orig(p))
    assert claims == attempts == sorted(RETAINED_IDS + [INC1])
    history = [json.loads(l)["status"] for l in
               (RESULTS / "score_attempts" / f"{INC1}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert history == [fx.SCORE_STARTED]
    assert not (RESULTS / "scores" / f"{INC1}.csv").exists()
    assert not (RESULTS / "score_claims" / f"{F1A}.claim").exists()
    assert sorted(p.stem for p in RESULTS.glob("scores/*.csv") if orig(p)) == sorted(RETAINED_IDS)
    # the committed INC-1 evidence is byte-identical to 576b8a1
    import subprocess
    for rel in (f"reports/phase3_4_results/score_claims/{INC1}.claim",
                f"reports/phase3_4_results/score_attempts/{INC1}.jsonl"):
        blob = subprocess.run(["git", "show", f"576b8a1:{rel}"], capture_output=True).stdout
        assert blob and blob == Path(rel).read_bytes(), rel


def test_ledger_still_holds_all_ten_step_d_authorisations_untouched():
    """Stage note: the ledger has grown by authorised appends since 576b8a1, so
    the step-D ledger must survive as the byte-identical 40-row prefix rather
    than equal the whole file."""
    history = verify_ledger_history(LEDGER)
    p34 = [e for e in history.rows if e["protocol"] == fx.PROTOCOL]
    assert {e["experiment_id"] for e in p34} == set(RETAINED_IDS) | {INC1, F1A}
    assert len({e["timestamp_utc"] for e in p34}) == 1            # one atomic append
    import subprocess
    blob = subprocess.run(["git", "show", "576b8a1:reports/experiment_results/test_look_ledger.jsonl"],
                          capture_output=True).stdout
    assert blob.replace(b"\r\n", b"\n") == b"".join(history.lines[:PHASE3_CLOSE_ROWS])


# ------------------------------------------------- gates (isolated root)
@pytest.fixture
def iso(tmp_path, monkeypatch):
    """Isolated results root + ledger with all ten looks authorised on fake checkpoints."""
    results = tmp_path / "results"
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(fx, "RESULTS_ROOT", results)
    monkeypatch.setattr(fx, "LEDGER", ledger)
    monkeypatch.setattr(fx, "git_head", lambda: "evaluator-sha-for-tests")
    monkeypatch.setattr(fx, "_run_protected_inference",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("forward ran")))
    for exp in fx.planned_experiments():
        ck = tmp_path / f"{exp.experiment_id}.pth"
        ck.write_bytes(exp.experiment_id.encode())
        rec = {"checkpoint_path": str(ck), "checkpoint_sha256": file_sha256(ck)}
        p = fx.training_record_path(exp); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rec), encoding="utf-8")
        TestLookLedger(ledger).append(TestLookRecord(
            timestamp_utc="2026-09-11T00:00:00+00:00", experiment_id=exp.experiment_id,
            git_sha="evaluator-sha-for-tests", machine="RoboticArm", protocol=fx.PROTOCOL,
            regime="T", model_config_identity="p34d-exec-e5470b025229f59a06f4c772",
            test_cohort_hash=fx.TEST_COHORT_HASH,
            reason=f"iso; checkpoint_sha256={rec['checkpoint_sha256']}", metrics_requested=("x",)))
    return {e.experiment_id: e for e in fx.planned_experiments()}


def test_gate_refuses_fold_1_present_and_absent_before_any_claim(iso):
    with pytest.raises(SystemExit, match="NO RETRY"):
        fx.score_experiment(iso[INC1])
    with pytest.raises(SystemExit, match="AUTHORIZED / UNUSED"):
        fx.score_experiment(iso[F1A])
    assert not (fx.RESULTS_ROOT / "score_claims").exists()
    assert not (fx.RESULTS_ROOT / "score_attempts").exists()


def test_gate_enforces_the_frozen_eight_look_order_without_skipping(iso):
    # anything but 2A is refused while 2A is unscored; nothing is claimed
    for eid in RETAINED_IDS[1:]:
        with pytest.raises(SystemExit, match="next look in the frozen order is " + RETAINED_IDS[0]):
            fx.score_experiment(iso[eid])
    assert not (fx.RESULTS_ROOT / "score_claims").exists()
    # a claimed-but-incomplete retained look blocks the whole phase
    fx.acquire_score_claim(iso[RETAINED_IDS[0]], {"checkpoint_sha256":
                           json.loads(fx.training_record_path(iso[RETAINED_IDS[0]]).read_text())["checkpoint_sha256"]})
    fx._append_durable(fx.score_attempt_path(iso[RETAINED_IDS[0]]), {"status": fx.SCORE_STARTED})
    with pytest.raises(SystemExit, match="next look in the frozen order is " + RETAINED_IDS[0]):
        fx.score_experiment(iso[RETAINED_IDS[1]])
    with pytest.raises(SystemExit, match="claim is already held"):
        fx.score_experiment(iso[RETAINED_IDS[0]])
    assert sorted(p.stem for p in fx.RESULTS_ROOT.glob("score_claims/*")) == [RETAINED_IDS[0]]


# ------------------------------------------------- harness guard proofs
def test_guard_refuses_real_claim_before_any_os_open_with_o_excl(monkeypatch):
    """The guard must fire BEFORE the O_EXCL create. exp0's claim already exists, so
    even a failed guard could not create anything - but a failed guard would reach
    os.open, which this test records."""
    opened = []
    real_open = os.open

    def spy(path, flags, *a, **k):
        if flags & os.O_EXCL:
            opened.append(str(path))
        return real_open(path, flags, *a, **k)
    monkeypatch.setattr(os, "open", spy)
    assert fx.RESULTS_ROOT == Path("reports/phase3_4_results")
    exp = next(e for e in fx.planned_experiments() if e.experiment_id == INC1)
    with pytest.raises(AssertionError, match="TEST HARNESS GUARD"):
        fx.acquire_score_claim(exp, {"status": "CLAIMED"})
    assert opened == []
    # and for an UNCLAIMED real id the guard still fires first; nothing is created
    unclaimed = next(e for e in fx.planned_experiments() if e.experiment_id == F1A)
    with pytest.raises(AssertionError, match="TEST HARNESS GUARD"):
        fx.acquire_score_claim(unclaimed, {"status": "CLAIMED"})
    assert opened == []
    assert not fx.score_claim_path(unclaimed).exists()


def test_guard_refuses_real_score_started_completed_and_ledger_appends():
    exp = next(e for e in fx.planned_experiments() if e.experiment_id == F1A)
    before = fx.score_attempt_path(exp).exists()
    for status in (fx.SCORE_STARTED, fx.SCORE_COMPLETED):
        with pytest.raises(AssertionError, match="TEST HARNESS GUARD"):
            fx._append_durable(fx.score_attempt_path(exp), {"status": status})
    assert fx.score_attempt_path(exp).exists() == before is False
    raw = LEDGER.read_bytes()
    with pytest.raises(AssertionError, match="TEST HARNESS GUARD"):
        TestLookLedger(fx.LEDGER).append_many_atomic([], expected_existing_count=40)
    with pytest.raises(AssertionError, match="TEST HARNESS GUARD"):
        TestLookLedger(fx.LEDGER).append(TestLookRecord(
            timestamp_utc="2026-09-11T00:00:00+00:00", experiment_id="x", git_sha="x", machine="RoboticArm",
            protocol=fx.PROTOCOL, regime="T", model_config_identity="x",
            test_cohort_hash=fx.TEST_COHORT_HASH, reason="x", metrics_requested=("x",)))
    assert LEDGER.read_bytes() == raw


def test_guard_passes_through_for_isolated_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(fx, "RESULTS_ROOT", tmp_path / "r")
    exp = fx.retained_experiments()[0]
    fx.acquire_score_claim(exp, {"status": "CLAIMED"})
    assert fx.score_claim_path(exp).exists()
    fx._append_durable(fx.score_attempt_path(exp), {"status": fx.SCORE_STARTED})
    assert [a["status"] for a in fx.score_attempts(exp)] == [fx.SCORE_STARTED]


def test_no_test_function_scores_against_the_real_results_root():
    """Any function that points RESULTS_ROOT at the real root must not call
    score_experiment (build_authorisations is read-only and may)."""
    real = 'RESULTS_ROOT", Path("reports/phase3_4_results")'
    call = "score_experiment" + "("
    for path in Path("tests").glob("test_*.py"):
        for block in path.read_text(encoding="utf-8").split("\ndef "):
            if real in block and not block.startswith("test_no_test_function_scores"):
                assert call not in block, (path, block.split("(", 1)[0])


# ------------------------------------------ salvage analysis on synthetic data
def test_salvage_analysis_uses_exactly_the_eight_retained_artifacts_and_48_parents(tmp_path, monkeypatch):
    """Synthetic artifacts only. Proves the frozen analysis path: eight retained
    artifacts, hash-verified, 48 parents, folds 2-5, stratified bootstrap, label and
    incident status carried in the output. No real score exists or is read."""
    import numpy as np
    monkeypatch.setattr(fx, "RESULTS_ROOT", tmp_path)
    rng = np.random.default_rng(0)
    focal = pd.read_csv(MEMB).query("role == 'focal_contribution'")
    parents_by_fold = {f: sorted(set(sub.parent_id)) for f, sub in focal.groupby("fold")}
    for exp in fx.planned_experiments():
        rows = []
        for j in range(232):
            rows.append({"segment_id": f"anom{j}", "parent_id": f"ap{j}", "arm": exp.arm,
                         "anomaly_label": "anomaly", "focal_parent_status": "non_focal",
                         "fold": exp.fold, "total_loss": rng.normal(3, 1)})
        k = 0
        for p in parents_by_fold[exp.fold]:
            for s in range(2):
                rows.append({"segment_id": f"f{exp.fold}p{p}s{s}", "parent_id": p, "arm": exp.arm,
                             "anomaly_label": "normal", "focal_parent_status": "focal_this_fold",
                             "fold": exp.fold, "total_loss": rng.normal(3, 1) - (0.3 if exp.arm == "PRESENT" else 0)})
                k += 1
        while len(rows) < 464:
            rows.append({"segment_id": f"other{len(rows)}", "parent_id": "o", "arm": exp.arm,
                         "anomaly_label": "normal", "focal_parent_status": "non_focal",
                         "fold": exp.fold, "total_loss": rng.normal(3, 1)})
        p = fx.score_path(exp)
        fx.atomic_write_text(p, pd.DataFrame(rows).to_csv(index=False))
        fx.atomic_write_text(p.with_suffix(".sha256"), canonical_sha256(p) + "\n")
    # fold-1 artifacts exist in this synthetic root but must never be read
    read = []
    real_read_csv = pd.read_csv
    monkeypatch.setattr(pd, "read_csv", lambda path, *a, **k: (read.append(str(path)), real_read_csv(path, *a, **k))[1])
    out = fx.analyse_from_persisted_scores()
    assert not any(INC1 in r or F1A in r for r in read)
    assert out["analysis_label"] == "INCIDENT-CONTINGENT SECONDARY SENSITIVITY ANALYSIS"
    assert out["original_primary_analysis"]["status"] == "NOT FULLY ESTIMABLE DUE TO INC-1"
    assert out["incident"]["protected_forward_passes"] == 0
    assert out["retained_experiment_ids"] == RETAINED_IDS and out["retained_folds"] == [2, 3, 4, 5]
    assert out["salvage"]["n_focal_parents"] == 48
    assert sorted(out["salvage"]["fold_level_deltas_descriptive"]) == [2, 3, 4, 5]
    b = out["salvage"]["bootstrap"]
    assert b["replicates"] == 10000 and b["seed"] == 27182 and b["fold_stratified"] is True
    assert "four retained folds" in b["conditional_on"]
    assert len(out["parent_deltas"]) == 48
    assert out["mandatory_report_statements"] == _rec()["mandatory_report_statements"]


def test_inc1_recovery_manifest_is_a_faithful_historical_record_of_64694e6():
    """Stage note: became history at the fold-1 recovery amendment (control code and
    its own test file are listed). Each value must equal the SHA-256 of the raw bytes
    stored in Git at 64694e6 (LF form)."""
    import hashlib, subprocess
    m = json.loads((RESULTS / "inc1_recovery_manifest.json").read_text(encoding="utf-8"))
    for name, recorded in m["files"].items():
        blob = subprocess.run(["git", "show", f"64694e6:{name}"], capture_output=True).stdout
        assert blob, name
        assert hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest() == recorded, name
    assert m["score_completed"] == 0 and m["score_artifacts"] == 0
    assert m["protected_test_inference_run"] is False
    assert m["retained_experiment_ids"] == RETAINED_IDS
    assert m["excluded_experiment_ids"] == [INC1, F1A]
