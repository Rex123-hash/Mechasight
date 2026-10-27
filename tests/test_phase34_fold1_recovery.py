"""Phase 3.4 — fold-1 zero-exposure technical recovery amendment: invariants and gates.

Every scorer invocation here runs against a temporary results root with fake
checkpoints and an inference sentinel; real-state checks are read-only. The
suite-wide guard in conftest.py additionally refuses any real claim.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from src.experiment import focal_toggle_execution as fx
from src.experiment.ledger import TestLookLedger, TestLookRecord
from src.experiment.provenance import canonical_sha256, file_sha256

AMEND = Path("configs/phase3/roboticarm_focal_toggle_fold1_recovery_amendment_v1.json")
RESULTS = Path("reports/phase3_4_results")
P_ID, A_ID = "p34d-92fd40fc93c8c8d2e819", "p34d-e56529a2f56f98fd4a7a"
FOUR_FOLD_SHA = "6441ed66b88074417ee0772e2618af0842050d2533d48029bb07a831ec1384bd"


def _amend() -> dict:
    return json.loads(AMEND.read_text(encoding="utf-8"))


# ------------------------------------------------------------- amendment
def test_amendment_is_frozen_and_names_the_exact_frozen_objects():
    a = _amend()
    assert a["status"] == "FROZEN_BEFORE_RECOVERY_SCORING"
    recP = json.loads((RESULTS / "training" / f"{P_ID}.json").read_text(encoding="utf-8"))
    recA = json.loads((RESULTS / "training" / f"{A_ID}.json").read_text(encoding="utf-8"))
    assert a["frozen_objects"]["fold_1_present"]["checkpoint_sha256"] == recP["checkpoint_sha256"]
    assert a["frozen_objects"]["fold_1_absent"]["checkpoint_sha256"] == recA["checkpoint_sha256"]
    assert recP["checkpoint_sha256"].startswith("85386f2df404")
    assert recA["checkpoint_sha256"].startswith("4b4d848b74d8")
    ex = sorted(a["recovery_executions"], key=lambda r: r["order"])
    assert [(r["experiment_id"], r["arm"]) for r in ex] == [(P_ID, "PRESENT"), (A_ID, "ABSENT")]
    assert all(r["permitted_protected_evaluations"] == 1 for r in ex)
    assert ex[0]["original_incident_claim"] == P_ID and ex[1]["original_incident_claim"] is None
    assert len({r["recovery_id"] for r in ex}) == 2 and all(r["recovery_id"].startswith("p34r-f1rec-") for r in ex)


def test_amendment_records_the_decision_context_honestly():
    a = _amend()
    d = a["decision_context"]
    assert d["made_after_four_fold_result_was_known"] is True
    assert d["recovery_justified_by_direction_of_result"] is False
    k = d["four_fold_result_known_at_decision"]
    assert k["commit"] == "a8b3d6c" and k["analysis_sha256"] == FOUR_FOLD_SHA
    assert a["acceptance"]["both_recovered_fold_1_results_accepted_regardless_of_value"] is True
    assert a["acceptance"]["no_selection_between_four_fold_and_five_fold"] is True
    assert a["inc1_original_record"]["protected_forward_passes"] == 0
    assert a["inc1_original_record"]["attempt_history"] == ["SCORE_STARTED"]
    assert "NEVER edited, deleted, completed or reused" in a["inc1_original_record"]["preservation"]
    assert a["five_fold_analysis"]["bootstrap"] == {
        "frozen": "fold-stratified parent bootstrap, folds 1-5, 12 parents with replacement per fold, 60 per replicate",
        "replicates": 10000, "seed": 27182, "interval": "percentile 95 percent"}
    assert a["five_fold_analysis"]["focal_parents"] == 60 and a["five_fold_analysis"]["folds"] == [1, 2, 3, 4, 5]


def test_recovery_ids_recompute_from_their_payload():
    import hashlib
    a = _amend()
    claim = json.loads((RESULTS / "score_claims" / f"{P_ID}.claim").read_text(encoding="utf-8"))
    ex = {r["arm"]: r for r in a["recovery_executions"]}
    pl = {"amendment": "fold1-technical-recovery-v1", "experiment_id": P_ID, "arm": "PRESENT",
          "checkpoint_sha256": ex["PRESENT"]["checkpoint_sha256"], "original_inc1_claim_timestamp_utc": claim["timestamp_utc"]}
    assert ex["PRESENT"]["recovery_id"] == "p34r-f1rec-" + hashlib.sha256(json.dumps(pl, sort_keys=True).encode()).hexdigest()[:20]
    al = {"amendment": "fold1-technical-recovery-v1", "experiment_id": A_ID, "arm": "ABSENT",
          "checkpoint_sha256": ex["ABSENT"]["checkpoint_sha256"], "original_state": "AUTHORIZED / UNUSED, never claimed"}
    assert ex["ABSENT"]["recovery_id"] == "p34r-f1rec-" + hashlib.sha256(json.dumps(al, sort_keys=True).encode()).hexdigest()[:20]
    assert [e.recovery_id for e in fx.recovery_experiments()] == [ex["PRESENT"]["recovery_id"], ex["ABSENT"]["recovery_id"]]


# ------------------------------------------------ four-fold result preserved
def test_four_fold_result_and_tag_are_preserved():
    assert canonical_sha256(RESULTS / "phase3_4_inc1_salvage_analysis.json") == FOUR_FOLD_SHA
    a = json.loads((RESULTS / "phase3_4_inc1_salvage_analysis.json").read_text(encoding="utf-8"))
    assert a["salvage"]["point_estimate"] == pytest.approx(0.008441599745142315, abs=1e-15)
    assert (a["salvage"]["ci95_low"], a["salvage"]["ci95_high"]) == (-0.0014088851686507907, 0.02038057407656849)
    tag = subprocess.run(["git", "rev-parse", "phase3-pre-fold1-recovery^{commit}"], capture_output=True, text=True).stdout.strip()
    assert tag.startswith("a8b3d6c")
    blob = subprocess.run(["git", "show", "phase3-pre-fold1-recovery:reports/phase3_4_results/phase3_4_inc1_salvage_analysis.json"],
                          capture_output=True).stdout
    import hashlib
    assert hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest() == FOUR_FOLD_SHA


def test_four_fold_analysis_function_is_unchanged_from_64694e6():
    import ast
    src = Path("src/experiment/focal_toggle_execution.py").read_text(encoding="utf-8")
    old = subprocess.run(["git", "show", "64694e6:src/experiment/focal_toggle_execution.py"], capture_output=True).stdout.decode("utf-8")

    def seg(s, name):
        t = ast.parse(s)
        return next(ast.get_source_segment(s, n) for n in ast.walk(t) if isinstance(n, ast.FunctionDef) and n.name == name)
    for name in ("analyse_from_persisted_scores", "load_persisted_scores", "_run_protected_inference",
                 "acquire_score_claim", "_append_durable", "atomic_write_bytes"):
        assert seg(src, name) == seg(old, name), name


# ------------------------------------------------------- namespace helpers
def test_recovery_namespace_never_collides_with_the_original():
    rp, ra = fx.recovery_experiments()
    orig = next(e for e in fx.planned_experiments() if e.experiment_id == P_ID)
    assert fx.score_claim_path(orig).name == f"{P_ID}.claim"
    assert fx.score_claim_path(rp).name == f"{P_ID}__{rp.recovery_id}.claim"
    assert fx.score_attempt_path(rp).name == f"{P_ID}__{rp.recovery_id}.jsonl"
    assert fx.score_path(rp).name == f"{P_ID}__{rp.recovery_id}.csv"
    assert fx.training_record_path(rp) == fx.training_record_path(orig)     # same frozen record
    assert (rp.fold, rp.arm, rp.seed, rp.training_membership_sha256) == (orig.fold, orig.arm, orig.seed, orig.training_membership_sha256)
    assert fx._recovery_binding(orig) == {}
    b = fx._recovery_binding(rp)
    assert b["recovery_id"] == rp.recovery_id and b["original_incident_claim"]["attempt_history"] == ["SCORE_STARTED"]
    assert fx._recovery_binding(ra)["original_incident_claim"] is None


# ---------------------------------------------------------------- gates
@pytest.fixture
def iso(tmp_path, monkeypatch):
    """Isolated root replicating the documented original state of fold 1 exactly
    (INC-1 claim + [SCORE_STARTED]; ABSENT unclaimed), fake checkpoints for both, and an
    inference sentinel. The amendment 'committed at HEAD' check is satisfied by
    feeding the working-copy bytes, so the gate logic is tested independently of git."""
    results = tmp_path / "results"
    monkeypatch.setattr(fx, "RESULTS_ROOT", results)
    monkeypatch.setattr(fx, "LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(fx, "git_head", lambda: "amendment-head-for-tests")
    forwards = {"n": 0}

    def sentinel(*a, **k):
        forwards["n"] += 1
        raise AssertionError("forward ran")
    monkeypatch.setattr(fx, "_run_protected_inference", sentinel)
    real_run = subprocess.run

    def fake_run(cmd, *a, **k):
        if cmd[:2] == ["git", "show"] and cmd[2].startswith("HEAD:"):
            class R: stdout = AMEND.read_bytes()
            return R()
        return real_run(cmd, *a, **k)
    monkeypatch.setattr(subprocess, "run", fake_run)
    amend = _amend()
    inc = amend["inc1_original_record"]
    by_id = {e.experiment_id: e for e in fx.planned_experiments()}
    for eid in (P_ID, A_ID):
        e = by_id[eid]
        ck = tmp_path / f"{eid}.pth"; ck.write_bytes(eid.encode())
        real = json.loads((RESULTS / "training" / f"{eid}.json").read_text(encoding="utf-8"))
        rec = {"checkpoint_path": str(ck), "checkpoint_sha256": file_sha256(ck)}
        p = fx.training_record_path(e); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rec), encoding="utf-8")
        TestLookLedger(fx.LEDGER).append(TestLookRecord(
            timestamp_utc="2026-09-11T19:49:04.480058+00:00", experiment_id=eid, git_sha="c10fdfc",
            machine="RoboticArm", protocol=fx.PROTOCOL, regime="T",
            model_config_identity="p34d-exec-e5470b025229f59a06f4c772", test_cohort_hash=fx.TEST_COHORT_HASH,
            reason=f"iso; checkpoint_sha256={rec['checkpoint_sha256']}", metrics_requested=("x",)))
    # original INC-1 state in the original namespace
    orig = by_id[P_ID]
    fx.acquire_score_claim(orig, {"status": "CLAIMED", "timestamp_utc": inc["claim_timestamp_utc"],
                                  "checkpoint_sha256": file_sha256(tmp_path / f"{P_ID}.pth")})
    fx._append_durable(fx.score_attempt_path(orig), {"status": fx.SCORE_STARTED})
    # the amendment's checkpoint hashes are the real ones; point the amendment copy at the fakes
    fake = json.loads(AMEND.read_text(encoding="utf-8"))
    for r in fake["recovery_executions"]:
        r["checkpoint_sha256"] = file_sha256(tmp_path / f"{r['experiment_id']}.pth")
    fake_path = tmp_path / "amendment.json"
    fake_path.write_text(json.dumps(fake, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setattr(fx, "FOLD1_AMENDMENT_PATH", fake_path)

    def fake_run2(cmd, *a, **k):
        if cmd[:2] == ["git", "show"] and cmd[2].startswith("HEAD:"):
            class R: stdout = fake_path.read_bytes()
            return R()
        return real_run(cmd, *a, **k)
    monkeypatch.setattr(subprocess, "run", fake_run2)
    rp, ra = fx.recovery_experiments()
    return {"rp": rp, "ra": ra, "orig": orig, "forwards": forwards, "results": results, "tmp": tmp_path}


def test_original_inc1_namespace_still_refuses_and_is_never_touched(iso):
    before = (fx.score_claim_path(iso["orig"]).read_bytes(), fx.score_attempt_path(iso["orig"]).read_bytes())
    with pytest.raises(SystemExit, match="NO RETRY"):
        fx.score_experiment(iso["orig"])
    assert (fx.score_claim_path(iso["orig"]).read_bytes(), fx.score_attempt_path(iso["orig"]).read_bytes()) == before
    assert iso["forwards"]["n"] == 0


def test_recovery_gate_enforces_order_1p_then_1a_and_refuses_when_original_state_differs(iso):
    rp, ra = iso["rp"], iso["ra"]
    with pytest.raises(SystemExit, match="next in the frozen recovery order is " + rp.recovery_id):
        fx.score_experiment(ra)
    assert not fx.score_claim_path(ra).exists() and iso["forwards"]["n"] == 0
    # tamper: append a fake COMPLETED to the original INC-1 history -> documented state differs -> refuse
    fx._append_durable(fx.score_attempt_path(iso["orig"]), {"status": fx.SCORE_COMPLETED})
    with pytest.raises(SystemExit, match="original INC-1 record is not in its documented state"):
        fx.score_experiment(rp)
    assert not fx.score_claim_path(rp).exists() and iso["forwards"]["n"] == 0


def test_recovery_gate_refuses_if_amendment_not_committed_byte_for_byte(iso, monkeypatch):
    real_run = subprocess.run

    def differs(cmd, *a, **k):
        if cmd[:2] == ["git", "show"] and cmd[2].startswith("HEAD:"):
            class R: stdout = b"{}"
            return R()
        return real_run(cmd, *a, **k)
    monkeypatch.setattr(subprocess, "run", differs)
    with pytest.raises(SystemExit, match="not committed at HEAD byte-for-byte"):
        fx.score_experiment(iso["rp"])
    assert not fx.score_claim_path(iso["rp"]).exists() and iso["forwards"]["n"] == 0


def test_recovery_claim_is_acquired_before_forward_and_is_single_use(iso):
    rp = iso["rp"]
    with pytest.raises(AssertionError, match="forward ran"):
        fx.score_experiment(rp)
    assert iso["forwards"]["n"] == 1
    claim = fx.read_score_claim(rp)
    assert claim["recovery_id"] == rp.recovery_id and claim["experiment_id"] == P_ID
    assert claim["original_incident_claim"]["attempt_history"] == ["SCORE_STARTED"]
    assert claim["recovery_amendment_commit"] == "amendment-head-for-tests"
    assert [a["status"] for a in fx.score_attempts(rp)] == [fx.SCORE_STARTED]
    assert fx.score_attempts(rp)[0]["recovery_id"] == rp.recovery_id
    # original namespace untouched
    assert [a["status"] for a in fx.score_attempts(iso["orig"])] == [fx.SCORE_STARTED]
    # consumed: second attempt refuses, no further forward; 1A blocked while 1P incomplete
    with pytest.raises(SystemExit, match="next in the frozen recovery order is " + rp.recovery_id):
        fx.score_experiment(iso["ra"])
    with pytest.raises(SystemExit, match="claim is already held"):
        fx.score_experiment(rp)
    assert iso["forwards"]["n"] == 1


def test_recovery_1a_proceeds_only_after_1p_completed_and_output_uses_frozen_schema(iso, monkeypatch):
    import numpy as np
    from tests.test_phase34_score_exposure_gate import _fake_grouped
    rp, ra = iso["rp"], iso["ra"]
    monkeypatch.setattr(fx, "_run_protected_inference", lambda r, d: _fake_grouped())
    out = fx.score_experiment(rp)
    assert out["rows"] == 464 and [a["status"] for a in fx.score_attempts(rp)] == [fx.SCORE_STARTED, fx.SCORE_COMPLETED]
    t = pd.read_csv(fx.score_path(rp), dtype=str)
    schema = [c["name"] for c in json.loads(Path("reports/phase3_4c_freeze/score_schema.json").read_text(encoding="utf-8"))["columns"]]
    assert list(t.columns) == schema and set(t.experiment_id) == {P_ID}
    out2 = fx.score_experiment(ra)
    assert out2["rows"] == 464
    with pytest.raises(SystemExit, match="already has SCORE_COMPLETED"):
        fx.score_experiment(ra)


# --------------------------------------------------- real state (read-only)
def test_real_fold1_checkpoints_and_backups_match_commit_b():
    for eid, tag in ((P_ID, "PRESENT_p34d-92fd40fc93c8c8d2e819_85386f2d"), (A_ID, "ABSENT_p34d-e56529a2f56f98fd4a7a_4b4d848b")):
        rec = json.loads((RESULTS / "training" / f"{eid}.json").read_text(encoding="utf-8"))
        assert file_sha256(Path(rec["checkpoint_path"])) == rec["checkpoint_sha256"]
        backup = Path("data/interim/phase3_4_recovery_backup/fold1") / f"{tag}.pth"
        if backup.exists():
            assert file_sha256(backup) == rec["checkpoint_sha256"]


def test_real_inc1_original_record_is_byte_identical_to_576b8a1():
    for rel in (f"reports/phase3_4_results/score_claims/{P_ID}.claim", f"reports/phase3_4_results/score_attempts/{P_ID}.jsonl"):
        blob = subprocess.run(["git", "show", f"576b8a1:{rel}"], capture_output=True).stdout
        assert blob and blob == Path(rel).read_bytes(), rel
    assert not (RESULTS / "scores" / f"{P_ID}.csv").exists()
    assert not (RESULTS / "score_claims" / f"{A_ID}.claim").exists()


# ------------------------------------------------ test-isolation audit
def test_no_test_can_reach_protected_scoring_outside_an_isolated_root():
    """Static audit of the whole suite (added before the fold-1 recovery scoring).

    (1) No test file invokes the runner's scoring stage function, passes the
        scoring stage to the runner script, or uses a shell call or a raw process spawn.
    (2) Every test function that calls the scorer redirects RESULTS_ROOT to a
        temporary root itself or through an isolating fixture, and never names
        the real results root.
    (3) The only child-process scorer (_race_worker) refuses real report paths.
    Forbidden tokens are assembled at runtime so this file does not contain them.
    """
    forbidden = ["stage_" + "score", "--" + "stage", "os." + "system", "shell" + "=True",
                 "subprocess." + "Popen", "run_focal_toggle_experiment.py\", \"" + "score"]
    real = 'Path("reports/' + 'phase3_4_results")'
    call = "score_experiment" + "("
    me = Path(__file__).name
    for path in sorted(Path("tests").glob("*.py")):
        src = path.read_text(encoding="utf-8")
        for tok in forbidden:
            assert tok not in src, (path, tok)
        for block in src.split("\ndef "):
            if call in block and not (path.name == me and block.startswith("test_no_test_can_reach")):
                isolated = ('RESULTS_ROOT", ' in block or "RESULTS_ROOT = " in block
                            or any(f in block for f in ("harness", "iso", "_prepare(", "tmp_path")))
                assert isolated, (path, block[:60])
                assert real not in block, (path, block[:60])
    worker = Path("tests/_race_worker.py").read_text(encoding="utf-8")
    assert "refuses real protected path" in worker


def test_race_worker_refuses_the_real_results_root(tmp_path):
    import multiprocessing as mp
    from tests import _race_worker
    with pytest.raises(RuntimeError, match="refuses real protected path"):
        _race_worker.main({"repo_root": str(Path(".").resolve()),
                           "results_root": str(Path("reports/phase3_4_results").resolve()),
                           "ledger": str(tmp_path / "l.jsonl"), "experiment_id": P_ID,
                           "forward_log": str(tmp_path / "f.jsonl"), "name": "X"},
                          mp.get_context("spawn").Barrier(1), mp.get_context("spawn").Queue())
