"""Phase 3.4F.2 — atomic single-owner protected-score claim (finding F3).

The race test spawns two independent OS processes that both pass every
pre-check, synchronise on a cross-process barrier immediately before the
O_CREAT|O_EXCL claim attempt, and then race it. All inference is a fake; the
real protected cohort is never opened.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path

import pytest

from src.experiment import focal_toggle_execution as fx
from src.experiment.ledger import TestLookLedger, TestLookRecord
from src.experiment.provenance import file_sha256

from tests import _race_worker
from tests.test_phase34_score_exposure_gate import _fake_grouped  # noqa: F401

REPO = str(Path(__file__).resolve().parents[1])


def _prepare(tmp_path):
    """Training record, checkpoint and a valid authorisation, all in tmp."""
    exp = fx.retained_experiments()[0]  # INC-1 stage note: fold-1 PRESENT is permanently refused; use the first retained look
    results = tmp_path / "results"
    ledger = tmp_path / "ledger.jsonl"
    ckpt = tmp_path / "ckpt.pth"
    ckpt.write_bytes(b"fake-checkpoint-bytes")
    record = {"checkpoint_path": str(ckpt), "checkpoint_sha256": file_sha256(ckpt),
              "initialization_sha256": "x", "initial_state_sha256": "y"}
    rec = results / "training" / f"{exp.experiment_id}.json"
    rec.parent.mkdir(parents=True)
    rec.write_text(json.dumps(record), encoding="utf-8")
    TestLookLedger(ledger).append(TestLookRecord(
        timestamp_utc="2026-09-11T00:00:00+00:00", experiment_id=exp.experiment_id,
        git_sha="evaluator-sha-for-race-test", machine="RoboticArm", protocol=fx.PROTOCOL,
        regime="T", model_config_identity="p34d-exec-e5470b025229f59a06f4c772",
        test_cohort_hash=fx.TEST_COHORT_HASH,
        reason=f"race test; checkpoint_sha256={record['checkpoint_sha256']}",
        metrics_requested=("x",)))
    return exp, results, ledger, record


# ------------------------------------------------------------ the real race
def test_two_processes_race_the_claim_and_exactly_one_scores(tmp_path):
    exp, results, ledger, _ = _prepare(tmp_path)
    forward_log = tmp_path / "forwards.jsonl"
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2)            # inherited by both children at spawn, not pickled
    queue = ctx.Queue()
    common = {"repo_root": REPO, "results_root": str(results), "ledger": str(ledger),
              "experiment_id": exp.experiment_id, "forward_log": str(forward_log)}
    procs = [ctx.Process(target=_race_worker.main, args=({**common, "name": n}, barrier, queue))
             for n in ("A", "B")]
    for pr in procs:
        pr.start()
    results_ = [queue.get(timeout=120) for _ in procs]
    for pr in procs:
        pr.join(timeout=30)
    assert all(pr.exitcode == 0 for pr in procs), [pr.exitcode for pr in procs]

    outcomes = sorted(r["outcome"] for r in results_)
    assert outcomes == ["REFUSED", "SCORED"], results_
    refused = next(r for r in results_ if r["outcome"] == "REFUSED")
    assert "claim is already held" in refused["message"]

    # exactly one protected forward happened, and the claim was on disk when it did
    forwards = [json.loads(l) for l in forward_log.read_text(encoding="utf-8").splitlines()]
    assert len(forwards) == 1
    assert forwards[0]["claim_on_disk"] is True

    # exactly one claim, exactly one STARTED, one COMPLETED; nothing reset
    fx.RESULTS_ROOT = results
    try:
        claim = fx.read_score_claim(exp)
        history = [a["status"] for a in fx.score_attempts(exp)]
    finally:
        fx.RESULTS_ROOT = Path("reports/phase3_4_results")
    assert claim is not None and claim["experiment_id"] == exp.experiment_id
    assert history == [fx.SCORE_STARTED, fx.SCORE_COMPLETED]
    assert list((results / "score_claims").glob("*")) == [results / "score_claims" / f"{exp.experiment_id}.claim"]
    assert not list((results / "score_claims").glob("*.tmp"))


def test_claim_acquisition_is_a_single_os_primitive(tmp_path, monkeypatch):
    """The create IS the check: a pre-existing file makes os.open(O_EXCL) fail."""
    exp = fx.retained_experiments()[0]  # INC-1 stage note: fold-1 PRESENT is permanently refused; use the first retained look
    monkeypatch.setattr(fx, "RESULTS_ROOT", tmp_path)
    claim = {"experiment_id": exp.experiment_id, "checkpoint_sha256": "abc"}
    fx.acquire_score_claim(exp, claim)
    assert fx.read_score_claim(exp) == claim
    with pytest.raises(fx.ClaimAlreadyHeld):
        fx.acquire_score_claim(exp, {**claim, "checkpoint_sha256": "different"})
    assert fx.read_score_claim(exp) == claim              # first writer's content survives
    # the file was created read-only: normal writes must not silently replace it
    assert not os.access(fx.score_claim_path(exp), os.W_OK) or os.name == "nt"


# ------------------------------------------------------------ tamper tests
@pytest.fixture
def harness(tmp_path, monkeypatch):
    exp, results, ledger, record = _prepare(tmp_path)
    monkeypatch.setattr(fx, "RESULTS_ROOT", results)
    monkeypatch.setattr(fx, "LEDGER", ledger)
    monkeypatch.setattr(fx, "git_head", lambda: "evaluator-sha-for-race-test")
    calls = {"forward": 0}

    def fake(record_, device):
        calls["forward"] += 1
        return _fake_grouped()

    monkeypatch.setattr(fx, "_run_protected_inference", fake)
    return {"exp": exp, "record": record, "calls": calls, "results": results}


def _write_claim(exp, checkpoint_sha):
    p = fx.score_claim_path(exp)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"status": "CLAIMED", "experiment_id": exp.experiment_id,
                             "checkpoint_sha256": checkpoint_sha}) + "\n", encoding="utf-8")


def test_existing_claim_without_started_refuses_as_incomplete(harness):
    _write_claim(harness["exp"], harness["record"]["checkpoint_sha256"])
    with pytest.raises(SystemExit, match="claim only, no SCORE_STARTED.*INCOMPLETE"):
        fx.score_experiment(harness["exp"])
    assert harness["calls"]["forward"] == 0


def test_existing_claim_with_started_refuses(harness):
    _write_claim(harness["exp"], harness["record"]["checkpoint_sha256"])
    fx._append_durable(fx.score_attempt_path(harness["exp"]),
                       {"status": fx.SCORE_STARTED, "experiment_id": harness["exp"].experiment_id})
    with pytest.raises(SystemExit, match=r"claim is already held.*SCORE_STARTED"):
        fx.score_experiment(harness["exp"])
    assert harness["calls"]["forward"] == 0


def test_existing_claim_with_completed_refuses(harness):
    fx.score_experiment(harness["exp"])                      # legitimate first and only look
    assert harness["calls"]["forward"] == 1
    # INC-1 stage note: after SCORE_COMPLETED the frozen-order gate now refuses first
    # (the next look is a different id); the claim gate remains behind it.
    with pytest.raises(SystemExit, match=r"claim is already held.*SCORE_COMPLETED|next look in the frozen order"):
        fx.score_experiment(harness["exp"])
    assert harness["calls"]["forward"] == 1


def test_claim_checkpoint_mismatch_is_a_hard_failure(harness):
    _write_claim(harness["exp"], "0" * 64)                   # names a different checkpoint
    with pytest.raises(SystemExit, match="HARD FAILURE.*Claims are never replaced"):
        fx.score_experiment(harness["exp"])
    assert harness["calls"]["forward"] == 0
    assert fx.read_score_claim(harness["exp"])["checkpoint_sha256"] == "0" * 64   # untouched


def test_claim_cannot_be_replaced_or_reset_by_scorer_commands(harness):
    exp = harness["exp"]
    fx.score_experiment(exp)
    claim_path = fx.score_claim_path(exp)
    before = claim_path.read_bytes()
    history_before = fx.score_attempt_path(exp).read_bytes()
    for _ in range(3):
        with pytest.raises(SystemExit):
            fx.score_experiment(exp)
    assert claim_path.read_bytes() == before
    assert fx.score_attempt_path(exp).read_bytes() == history_before
    # no scorer command exists that deletes or resets a claim
    import inspect
    src = inspect.getsource(fx)
    assert "score_claim_path(exp).unlink" not in src and ".claim').unlink" not in src


def test_crash_before_started_still_consumes_the_look(harness, monkeypatch):
    """Claim acquired, then a crash BEFORE SCORE_STARTED is appended -> look consumed."""
    exp = harness["exp"]

    def crash_after_claim():
        raise RuntimeError("simulated crash between claim and SCORE_STARTED")

    # the hook runs before the claim; make the crash happen right after acquisition instead
    original = fx.acquire_score_claim

    def acquire_then_crash(e, c):
        original(e, c)
        raise RuntimeError("simulated crash between claim and SCORE_STARTED")

    monkeypatch.setattr(fx, "acquire_score_claim", acquire_then_crash)
    with pytest.raises(RuntimeError):
        fx.score_experiment(exp)
    assert fx.read_score_claim(exp) is not None
    assert fx.score_attempts(exp) == []                      # no STARTED was ever written
    assert harness["calls"]["forward"] == 0

    monkeypatch.setattr(fx, "acquire_score_claim", original)
    with pytest.raises(SystemExit, match="claim only, no SCORE_STARTED.*INCOMPLETE"):
        fx.score_experiment(exp)
    assert harness["calls"]["forward"] == 0
