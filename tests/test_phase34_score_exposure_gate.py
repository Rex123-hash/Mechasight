"""Phase 3.4F.1 — protected-score exposure state machine (finding F1) and F2 hashes.

Every test here runs against a temporary results root, a temporary ledger and a
FAKE ``_run_protected_inference``. The real official test cohort is never
opened and no real model is ever run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.experiment import focal_toggle_execution as fx
from src.experiment.ledger import TestLookLedger, TestLookRecord
from src.experiment.provenance import (
    file_sha256, initial_state_hash, initialization_hash,
)
from src.reproduction.protocol_a import OFFICIAL_PARAMS, SENSORS, build_official_autoencoder

EXPECTED_INIT = "d889e90224988ac27bceba086f695ac5d7f8f34ca1e2ad1debc16c6f61384d55"
EXPECTED_STATE = "01d68f698cdf32628604536d282e218106ac48f51fb14d2a74bf667c2c980048"


# ------------------------------------------------------------ fake harness
def _fake_grouped(n: int = 464) -> pd.DataFrame:
    """A synthetic segment-median table with the shape the real scorer produces."""

    rng = np.random.default_rng(0)
    seg = [f"20240101_00_00_00_dataLog_{i}" for i in range(n)]
    frame = pd.DataFrame({
        "segment_id": seg,
        "split_label": ["Normal_Source_Test" if i % 2 == 0 else "Anomaly_Source_Test" for i in range(n)],
        "anomaly_label": ["normal" if i % 2 == 0 else "anomaly" for i in range(n)],
        "domain_shift_op": "weight00", "domain_shift_env": "BckgA",
    })
    for col in fx.SCORE_COLUMNS:
        frame[col] = rng.random(n)
    return frame


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """Isolated results root, ledger, training record and checkpoint for experiment 1."""

    exp = fx.retained_experiments()[0]  # INC-1 stage note: fold-1 PRESENT is permanently refused; use the first retained look
    results = tmp_path / "results"
    ledger = tmp_path / "ledger.jsonl"
    ckpt = tmp_path / f"fold{exp.fold}" / exp.arm / "focal_toggle_seed1995.pth"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"fake-checkpoint-bytes")
    monkeypatch.setattr(fx, "RESULTS_ROOT", results)
    monkeypatch.setattr(fx, "LEDGER", ledger)
    monkeypatch.setattr(fx, "git_head", lambda: "evaluator-sha-for-tests")

    record = {"checkpoint_path": str(ckpt), "checkpoint_sha256": file_sha256(ckpt),
              "initialization_sha256": EXPECTED_INIT, "initial_state_sha256": EXPECTED_STATE}
    p = fx.training_record_path(exp)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record), encoding="utf-8")

    def authorise(checkpoint_sha=record["checkpoint_sha256"]):
        TestLookLedger(ledger).append(TestLookRecord(
            timestamp_utc="2026-09-11T00:00:00+00:00", experiment_id=exp.experiment_id,
            git_sha="evaluator-sha-for-tests", machine="RoboticArm", protocol=fx.PROTOCOL,
            regime="T", model_config_identity="p34d-exec-e5470b025229f59a06f4c772",
            test_cohort_hash=fx.TEST_COHORT_HASH,
            reason=f"test authorisation; checkpoint_sha256={checkpoint_sha}",
            metrics_requested=("x",)))

    calls = {"forward": 0}

    def fake_inference(record_, device):
        calls["forward"] += 1
        # C: SCORE_STARTED must already be durable on disk when the first forward happens
        attempts = fx.score_attempts(exp)
        assert attempts and attempts[0]["status"] == fx.SCORE_STARTED, \
            "test forward ran before SCORE_STARTED was persisted"
        return _fake_grouped()

    monkeypatch.setattr(fx, "_run_protected_inference", fake_inference)
    return {"exp": exp, "authorise": authorise, "calls": calls, "record": record,
            "results": results, "ckpt": ckpt}


# ------------------------------------------------------------------ gates
def test_A_score_refuses_without_ledger_authorisation(harness):
    with pytest.raises(SystemExit, match="no ledger entry authorises"):
        fx.score_experiment(harness["exp"])
    assert harness["calls"]["forward"] == 0
    assert fx.score_attempts(harness["exp"]) == []


def test_B_score_refuses_when_checkpoint_hash_differs_from_authorised(harness):
    harness["authorise"](checkpoint_sha="0" * 64)          # authorises a different checkpoint
    with pytest.raises(SystemExit, match="no ledger entry authorises this exact checkpoint"):
        fx.score_experiment(harness["exp"])
    assert harness["calls"]["forward"] == 0


def test_B2_score_refuses_when_checkpoint_on_disk_was_altered(harness):
    harness["authorise"]()
    harness["ckpt"].write_bytes(b"tampered")
    with pytest.raises(SystemExit, match="does not match the frozen hash"):
        fx.score_experiment(harness["exp"])
    assert harness["calls"]["forward"] == 0


def test_C_score_started_is_durable_before_the_first_forward_and_completes(harness):
    harness["authorise"]()
    out = fx.score_experiment(harness["exp"])
    assert harness["calls"]["forward"] == 1                 # the fake asserted START existed
    history = [a["status"] for a in fx.score_attempts(harness["exp"])]
    assert history == [fx.SCORE_STARTED, fx.SCORE_COMPLETED]
    started = fx.score_attempts(harness["exp"])[0]
    for field in ("experiment_id", "fold", "arm", "seed", "checkpoint_sha256",
                  "evaluator_git_sha", "config_identity", "training_membership_sha256",
                  "test_cohort_hash", "ledger_authorization", "timestamp_utc"):
        assert field in started, field
    completed = fx.score_attempts(harness["exp"])[1]
    assert completed["rows"] == 464 and completed["expected_rows"] == 464
    assert completed["score_artifact_sha256"] == out["score_artifact_sha256"]
    assert Path(out["score_artifact"]).exists()
    assert out["aggregates_computed_here"] is False


def test_D_crash_after_started_before_any_file_refuses_a_second_attempt(harness, monkeypatch):
    harness["authorise"]()

    def crash(record_, device):
        harness["calls"]["forward"] += 1
        raise RuntimeError("simulated crash during inference")

    monkeypatch.setattr(fx, "_run_protected_inference", crash)
    with pytest.raises(RuntimeError):
        fx.score_experiment(harness["exp"])
    assert not fx.score_path(harness["exp"]).exists()       # no artifact was ever produced
    assert [a["status"] for a in fx.score_attempts(harness["exp"])] == [fx.SCORE_STARTED]

    # a naive retry, even with the inference restored, must refuse
    monkeypatch.setattr(fx, "_run_protected_inference", lambda r, d: _fake_grouped())
    with pytest.raises(SystemExit, match="claim is already held"):
        fx.score_experiment(harness["exp"])
    assert harness["calls"]["forward"] == 1                  # never ran again


def test_E_crash_after_partial_inference_before_atomic_finalisation_refuses(harness, monkeypatch):
    harness["authorise"]()
    exp = harness["exp"]

    def partial_then_crash(record_, device):
        harness["calls"]["forward"] += 1
        # simulate a partial, non-atomic write that a crash could leave behind
        tmp = fx.score_path(exp).with_name(fx.score_path(exp).name + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text("partial,rows\n1,2\n", encoding="utf-8")
        raise RuntimeError("simulated crash mid-write")

    monkeypatch.setattr(fx, "_run_protected_inference", partial_then_crash)
    with pytest.raises(RuntimeError):
        fx.score_experiment(exp)
    assert not fx.score_path(exp).exists()                   # only the .tmp fragment exists
    assert [a["status"] for a in fx.score_attempts(exp)] == [fx.SCORE_STARTED]

    monkeypatch.setattr(fx, "_run_protected_inference", lambda r, d: _fake_grouped())
    with pytest.raises(SystemExit, match="INCOMPLETE"):
        fx.score_experiment(exp)
    assert harness["calls"]["forward"] == 1


def test_F_existing_completed_record_refuses_a_second_attempt(harness, monkeypatch):
    harness["authorise"]()
    fx.score_experiment(harness["exp"])
    monkeypatch.setattr(fx, "_run_protected_inference", lambda r, d: _fake_grouped())
    # INC-1 stage note: the frozen-order gate refuses a completed look before the claim gate
    with pytest.raises(SystemExit, match="claim is already held|next look in the frozen order"):
        fx.score_experiment(harness["exp"])
    assert harness["calls"]["forward"] == 1


def test_F2_exposure_history_is_append_only_and_never_reset(harness):
    harness["authorise"]()
    fx.score_experiment(harness["exp"])
    p = fx.score_attempt_path(harness["exp"])
    before = p.read_bytes()
    with pytest.raises(SystemExit):
        fx.score_experiment(harness["exp"])
    assert p.read_bytes() == before                          # refusal writes nothing, deletes nothing


def test_G_no_aggregate_metric_is_computed_during_protected_scoring(harness, monkeypatch):
    harness["authorise"]()
    called = {"auc": 0, "delta": 0}
    monkeypatch.setattr(fx, "official_single_auc",
                        lambda *a, **k: called.__setitem__("auc", called["auc"] + 1))
    import src.experiment.focal_concordance as fc
    monkeypatch.setattr(fc, "parent_concordance_delta",
                        lambda *a, **k: called.__setitem__("delta", called["delta"] + 1))
    out = fx.score_experiment(harness["exp"])
    assert called == {"auc": 0, "delta": 0}
    assert out["aggregates_computed_here"] is False
    assert not any(k for k in out if "auc" in k.lower() or "delta" in k.lower())


def test_H_analysis_reads_only_hash_verified_complete_score_collection(harness, monkeypatch):
    # only one of the eight retained experiments has an artifact -> analysis must refuse
    harness["authorise"]()
    fx.score_experiment(harness["exp"])
    with pytest.raises(SystemExit, match="score artifact missing"):
        fx.analyse_from_persisted_scores()
    # a tampered artifact must also be refused, even when all eight exist
    for exp in fx.retained_experiments():
        p = fx.score_path(exp)
        if not p.exists():
            fx.atomic_write_text(p, _fake_grouped().assign(
                experiment_id=exp.experiment_id, fold=exp.fold, arm=exp.arm, seed=exp.seed,
                parent_id="x", focal_parent_status="non_focal").to_csv(index=False))
            fx.atomic_write_text(p.with_suffix(".sha256"), "0" * 64 + "\n")
    with pytest.raises(SystemExit, match="hash mismatch"):
        fx.analyse_from_persisted_scores()


# ---------------------------------------------------------- F2 state hashes
def _build(seed: int = 1995):
    torch.manual_seed(seed)
    return build_official_autoencoder([1600, 706, 706], [1, 3, 3], dict(OFFICIAL_PARAMS))


def test_initial_state_hash_covers_buffers_and_matches_pre_registration():
    model = _build()
    keys = list(model.state_dict().keys())
    assert len(keys) == 46
    assert sum(1 for k in keys if "running_mean" in k) == 6
    assert sum(1 for k in keys if "num_batches_tracked" in k) == 6
    assert initialization_hash(model) == EXPECTED_INIT           # unchanged from COMMIT A
    assert initial_state_hash(model) == EXPECTED_STATE
    cfg = json.loads(Path("configs/phase3/roboticarm_focal_toggle_execution_v2.json")
                     .read_text(encoding="utf-8"))["initialization_policy"]
    assert cfg["expected_initialization_sha256"] == EXPECTED_INIT
    assert cfg["expected_initial_state_sha256"] == EXPECTED_STATE


def test_initial_state_hash_sees_a_buffer_change_the_parameter_hash_cannot():
    model = _build()
    with torch.no_grad():
        model.encoder[1].running_mean.add_(1e-3)
    assert initialization_hash(model) == EXPECTED_INIT        # blind to buffers, by design
    assert initial_state_hash(model) != EXPECTED_STATE          # sees it


def test_paired_arms_reproduce_both_hashes_naturally():
    """Equality must arise from the frozen seed and architecture, not from any adjustment."""
    hashes = {(initialization_hash(m), initial_state_hash(m)) for m in (_build(), _build())}
    assert hashes == {(EXPECTED_INIT, EXPECTED_STATE)}


def test_freeze_refuses_unequal_or_off_registration_state_hashes(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("_runner", "scripts/run_focal_toggle_experiment.py")
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    monkeypatch.setattr(fx, "RESULTS_ROOT", tmp_path)
    ck = tmp_path / "c.pth"; ck.write_bytes(b"x")

    def write_all(state_for):
        for exp in fx.planned_experiments():
            rec = {"checkpoint_sha256": file_sha256(ck), "checkpoint_path": str(ck),
                   "initialization_sha256": EXPECTED_INIT,
                   "initial_state_sha256": state_for(exp), "epochs_run": 1,
                   "final_epoch_actually_scored_one_indexed": 1}
            p = fx.training_record_path(exp); p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(rec), encoding="utf-8")

    write_all(lambda e: "other" if (e.fold == 2 and e.arm == "ABSENT") else EXPECTED_STATE)
    with pytest.raises(SystemExit, match="paired initial-state hashes differ"):
        runner.stage_freeze()
    write_all(lambda e: "not-the-registered-value")     # paired, but off-registration
    with pytest.raises(SystemExit, match="pre-registered"):
        runner.stage_freeze()


# ------------------------------------------------- config identity payload
def test_config_identity_recomputes_from_scientific_fields_only():
    import hashlib
    c = json.loads(Path("configs/phase3/roboticarm_focal_toggle_execution_v2.json")
                   .read_text(encoding="utf-8"))
    slots = json.loads(c["parent_balance"]["chosen_resolution"].split(": ", 1)[1])
    payload = {"parent": c["parent_design_config"], "family": "parent-exclusive",
               "seed": c["seed_policy"]["seed"], "experiments": c["planned_experiment_ids"],
               "env": c["execution_environment"]["execution_environment_id"],
               "slots": {str(k): int(slots[str(k)]) for k in range(1, 6)},
               "bootstrap": ["fold-stratified", c["uncertainty"]["replicates"], c["uncertainty"]["seed"]]}
    rid = "p34d-exec-" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
    assert rid == c["config_identity"] == "p34d-exec-e5470b025229f59a06f4c772"
    # the excluded provenance fields are documented and really are outside the payload
    excluded = set(c["config_identity_definition"]["excluded_fields"])
    assert {"claim_boundary", "initialization_policy", "execution_machinery",
            "provenance_hash_policy"} <= excluded
    assert not (excluded & set(payload))
