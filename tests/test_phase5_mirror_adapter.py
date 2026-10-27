"""Phase 5.6C: protected-style input adapter (development mirror) and JSONL batch interface.

Permitted development source-validation inputs and SYNTHETIC labels only, in
temporary sandboxes. No protected recording, cache, feature artifact, label or
sealed score is opened; the real ledger is only read (copied) and never written.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile

import pytest

from src.experiment import phase5_input_manifest as mi
from src.experiment.ledger_history import verify_ledger_history
from src.experiment.phase5_execution import Executor, FrozenPackage, authorize_fixture
from src.experiment.phase5_fixture import create_fixture
from src.experiment.phase5_inference import FrozenL1Adapter
from src.experiment.phase5_inputs import (
    MIRROR_SCHEMA, PROTECTED_SCHEMA, SYNTHETIC, Access, Refused, Sandbox, atomic_json,
)
from src.experiment.phase5_ledger import append_protected_batch, look_budget, seed_jsonl_ledger
from src.experiment.phase5_mirror_fixture import create_mirror_fixture
from src.experiment.provenance import file_sha256

os.environ.setdefault("MECHASIGHT_TORCH_THREADS", "4")
REAL_LEDGER = Path("reports/experiment_results/test_look_ledger.jsonl")
REAL_LEDGER_SHA256 = "f75584611c18ff387bee8c06f1fcc999c3046def876ba13dea1c744fce64de5f"


@pytest.fixture(scope="module")
def frozen():
    return FrozenPackage.load()


@pytest.fixture(scope="module")
def template(frozen):
    root = Path(tempfile.mkdtemp(prefix="mechasight-SYNTHETIC-5-6c-template-"))
    create_mirror_fixture(Sandbox(root), frozen)
    yield root
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def mirror(template, frozen, tmp_path):
    root = tmp_path / "SYNTHETIC-mirror"
    shutil.copytree(template, root)
    box = Sandbox(root)
    return box, Executor(box, frozen, predictor=FrozenL1Adapter(frozen))


@pytest.fixture
def jsonl(tmp_path, frozen):
    """Synthetic-stub executor over a temporary copy of the closed JSONL ledger."""
    box = Sandbox(tmp_path / "SYNTHETIC-jsonl")
    create_fixture(box, frozen)
    fixture = box.json("fixture.json")
    fixture["ledger"] = {**seed_jsonl_ledger(box), "code_git_sha": "0" * 40}
    atomic_json(box.path("fixture.json"), fixture, replace=True)
    return box, Executor(box, frozen)


def permit(box, package, action="score", **kw):
    return authorize_fixture(box, package, action, **kw)


def _history(box):
    return verify_ledger_history(box.path("ledger/test_look_ledger.jsonl"),
                                 authorization_root=box.path("ledger/authorizations"))


def _count_reads(monkeypatch):
    reads = []
    real_parquet, real_h5 = mi.pd.read_parquet, mi.h5py.File
    monkeypatch.setattr(mi.pd, "read_parquet", lambda *a, **k: reads.append("parquet") or real_parquet(*a, **k))
    monkeypatch.setattr(mi.h5py, "File", lambda *a, **k: reads.append("hdf5") or real_h5(*a, **k))
    return reads


# ------------------------------------------------------------------ A. protected-style adapter
def test_A_mirror_fixture_names_no_input_path_only_trusted_manifest_identities(template, frozen):
    fixture = json.loads((template / "fixture.json").read_text(encoding="utf-8"))
    assert fixture["schema"] == MIRROR_SCHEMA and fixture["scope"] == SYNTHETIC
    for cell, entry in fixture["cells"].items():
        assert set(entry) == {"input_manifest_identity"}
        machine, regime = cell.split("|")
        assert entry["input_manifest_identity"] == mi.development_manifest(frozen, machine=machine, regime=regime).identity
    assert {m: len(c) for m, c in fixture["cohorts"].items()} == {"RoboticArm": 186, "BrushlessMotor": 126}
    assert not any("source_validation" in name for name in fixture["files"])


def test_A_forged_capability_reads_nothing(mirror, monkeypatch):
    box, executor = mirror
    reads = _count_reads(monkeypatch)
    forged = Access("forged", box.identity, "SYNTHETIC-forged", "score", "0" * 64)
    for call in (lambda: executor.inputs.windows(forged, "RoboticArm|S"),
                 lambda: executor.inputs.features(forged, "BrushlessMotor|T")):
        with pytest.raises(Refused):
            call()
    assert reads == [] and box.events == []


def test_A_unexposed_but_issued_batch_is_refused_before_any_read(mirror, monkeypatch):
    box, executor = mirror
    reads = _count_reads(monkeypatch)
    access = executor._issue("SYNTHETIC-never-exposed", "score")          # issued, but no durable exposure
    with pytest.raises(Refused, match="authorization/exposure records missing"):
        executor.inputs.windows(access, "RoboticArm|S")
    assert reads == []


@pytest.mark.parametrize("tamper", ["identity", "cohort"])
def test_A_tampered_mirror_binding_stops_after_exposure_without_artifact_reads(mirror, frozen, monkeypatch, tamper):
    box, _ = mirror
    fixture = box.json("fixture.json")
    if tamper == "identity":
        fixture["cells"]["RoboticArm|S"]["input_manifest_identity"] = "0" * 64
    else:
        fixture["cohorts"]["RoboticArm"] = fixture["cohorts"]["RoboticArm"][1:]
    atomic_json(box.path("fixture.json"), fixture, replace=True)
    executor = Executor(box, frozen, predictor=FrozenL1Adapter(frozen))
    reads = _count_reads(monkeypatch)
    real_trusted = type(executor.inputs)._trusted

    def first_mirror_read(self, access, cell, role):
        # Fail closed at the first protected-style read: nothing of that cell is opened.
        before = len(reads)
        try:
            return real_trusted(self, access, cell, role)
        finally:
            assert len(reads) == before

    monkeypatch.setattr(type(executor.inputs), "_trusted", first_mirror_read)
    with pytest.raises(Refused, match="trusted input manifest|trusted cohort|alignment"):
        executor.score_all(permit(box, frozen))
    state = box.json("state.json")
    assert state["status"] == "FAILED_TECHNICAL" and len(_history(box).rows) == 428


def test_A_protected_schema_and_entry_points_stay_disabled(tmp_path, frozen):
    box = Sandbox(tmp_path / "SYNTHETIC-protected")
    atomic_json(box.path("fixture.json"), {"scope": SYNTHETIC, "schema": PROTECTED_SCHEMA, "files": {}, "cells": {},
                                           "cohorts": {}, "historical": {}, "labels": {}})
    with pytest.raises(Refused, match="unknown fixture input schema"):
        Executor(box, frozen)
    with pytest.raises(Refused, match="disabled"):
        mi.read_protected_inputs()
    with pytest.raises(Refused, match="disabled"):
        append_protected_batch()
    with pytest.raises(Refused, match="protected input access is disabled"):
        mi.read_features(mi.protected_manifest_template(frozen, machine="BrushlessMotor", regime="T"))


# ------------------------------------------------------------------ B. JSONL batch interface
def test_B_scoring_batch_is_exactly_24_l1_plus_208_l2_and_196_to_428(jsonl, frozen):
    box, executor = jsonl
    executor.score_all(permit(box, frozen))
    history = _history(box)
    assert len(history.rows) == 428 and REAL_LEDGER.read_bytes().count(b"\n") == 196
    operations = [r["reason"].rsplit("operation=", 1)[1] for r in history.rows[196:]]
    assert operations.count("new_L1_inference") == 24 and operations.count("derived_L2_scoring") == 208
    reused = {u["L1"]["experiment_id"] for u in frozen.registry["units"] if u["L1"]["source"] != "new_fit_phase5"}
    assert not reused & {r["experiment_id"] for r in history.rows[196:]}       # comparators consume no new look
    assert len(executor.ledger.verify()) == 428                                # validate_batches green


@pytest.mark.parametrize("edit", ["scope", "extra_key", "reservation", "code_sha"])
def test_B_altered_batch_authorization_is_refused(jsonl, frozen, edit):
    box, executor = jsonl
    state = executor.score_all(permit(box, frozen))
    path = box.path(f"ledger/authorizations/{state['batch_id']}/batch_authorization.json")
    record = json.loads(path.read_text(encoding="utf-8"))
    if edit == "scope":
        record["scope"] = "PROTECTED"
    elif edit == "extra_key":
        record["approved_by"] = "self"
    elif edit == "reservation":
        record["reservation_sha256"] = "0" * 64
    else:
        record["code_git_sha"] = "1" * 40
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(Refused):
        executor.ledger.verify()


def test_B_reservation_change_after_exposure_is_refused(jsonl, frozen):
    box, executor = jsonl
    state = executor.score_all(permit(box, frozen))
    reservation = box.path(f"reservations/{state['batch_id']}.json")
    reservation.write_text(reservation.read_text(encoding="utf-8").replace('"ledger_end":428', '"ledger_end":429'),
                           encoding="utf-8")
    with pytest.raises(Refused, match="not bound to its durable reservation"):
        executor.ledger.verify()


def test_B_crash_after_exposure_retries_with_unique_ids_and_reused_batch_is_refused(jsonl, frozen):
    box, executor = jsonl

    def crash(stage, unit):
        if stage == "after_read":
            raise OSError("SYNTHETIC crash after the first read")

    executor.fault = crash
    with pytest.raises(OSError):
        executor.score_all(permit(box, frozen))
    executor.fault = None
    assert box.json("state.json")["status"] == "FAILED_TECHNICAL" and len(_history(box).rows) == 428
    state = executor.recover(permit(box, frozen, "recovery", recovery_from=box.json("state.json")["batch_id"]))
    rows = executor.ledger.verify()
    assert len(rows) == 660 == 196 + 232 + 232 and len(state["tables"]) == 232
    assert all(r["experiment_id"] == r["original_experiment_id"] + "-r1" for r in rows[428:])
    assert len({r["experiment_id"] for r in rows}) == 660 == look_budget(frozen)["max_total_rows"]
    trusted = executor.ledger.budget
    with pytest.raises(Refused, match="cap"):
        executor.ledger.validate_batches(rows + rows[196:197])
    executor.ledger.budget = {**trusted, "max_total_rows": 10_000}       # isolate reuse detection from the cap
    with pytest.raises(Refused, match="reused"):
        executor.ledger.validate_batches(rows + rows[196:197])
    executor.ledger.budget = trusted


def test_B_crash_before_ledger_append_leaves_no_look_and_recovers_to_428(jsonl, frozen):
    box, executor = jsonl

    def crash(stage, unit):
        if stage == "after_batch_authorization":
            raise OSError("SYNTHETIC crash before the ledger append")

    executor.fault = crash
    with pytest.raises(OSError):
        executor.score_all(permit(box, frozen))
    executor.fault = None
    assert len(_history(box).rows) == 196
    executor.recover(permit(box, frozen, "recovery", recovery_from=box.json("state.json")["batch_id"]))
    assert len(executor.ledger.verify()) == 428


def test_B_scoring_capability_cannot_read_labels_and_analysis_is_separate(jsonl, frozen):
    box, executor = jsonl
    state = executor.score_all(permit(box, frozen))
    score_access = executor._access(state["batch_id"], "score")
    with pytest.raises(Refused, match="label access requires analysis capability"):
        executor.inputs.labels(score_access, "RoboticArm")
    executor._revoke()
    with pytest.raises(Refused, match="analysis"):
        executor._access(state["batch_id"], "analysis")                     # completion gate not claimed
    executor.analyse_once(permit(box, frozen, "analysis"))
    with pytest.raises(Refused, match="already claimed"):
        executor.analyse_once(permit(box, frozen, "analysis"))
    assert len(executor.ledger.verify()) == 428


def test_B_real_ledger_is_never_a_backend_and_stays_closed(jsonl, frozen):
    box, executor = jsonl
    fixture = box.json("fixture.json")
    fixture["ledger"]["path"] = str(REAL_LEDGER.resolve())
    atomic_json(box.path("fixture.json"), fixture, replace=True)
    with pytest.raises(Refused):
        Executor(box, frozen)
    assert file_sha256(REAL_LEDGER) == REAL_LEDGER_SHA256


# ------------------------------------------------------------------ C. look budget checked before any mutation
def _crash_at(stage):
    def fail(where, unit):
        if where == stage:
            raise OSError("SYNTHETIC crash at " + stage)
    return fail


def _at_660(box, executor, frozen):
    """196 -> 428 (crash after exposure) -> 660 (recovery crashes after exposure)."""
    executor.fault = _crash_at("after_reservation")
    with pytest.raises(OSError):
        executor.score_all(permit(box, frozen))
    with pytest.raises(OSError):
        executor.recover(permit(box, frozen, "recovery", recovery_from=box.json("state.json")["batch_id"]))
    executor.fault = None
    assert len(_history(box).rows) == 660
    return box.json("state.json")["batch_id"]


def test_C_budget_is_trusted_total_660_or_464_additional_and_baseline_cannot_be_moved(jsonl, frozen):
    box, executor = jsonl
    assert look_budget(frozen) == {"baseline_rows": 196, "max_total_rows": 660, "max_additional_looks": 464}
    assert executor.ledger.budget == look_budget(frozen)
    fixture = box.json("fixture.json")
    fixture["ledger"]["base_rows"] = 100
    atomic_json(box.path("fixture.json"), fixture, replace=True)
    with pytest.raises(Refused, match="baseline"):
        Executor(box, frozen)


def test_C_660_plus_232_is_refused_before_any_mutation_or_read(jsonl, frozen, monkeypatch):
    """The original 660 -> 892 counterexample, through recovery."""
    box, executor = jsonl
    failed = _at_660(box, executor, frozen)
    ledger_bytes = box.path("ledger/test_look_ledger.jsonl").read_bytes()
    authorizations = sorted(p.name for p in box.path("ledger/authorizations").iterdir())
    reservations = sorted(p.name for p in box.path("reservations").iterdir())
    exposures = sorted(p.name for p in box.path("exposures").iterdir())
    events = len(box.events)
    exposed = [r for r in executor._audit_records() if r["state"] == "EXPOSURE_RECORDED"]
    monkeypatch.setattr(type(executor.inputs), "_trusted", lambda *a, **k: pytest.fail("protected-style reader invoked"))
    recovery = permit(box, frozen, "recovery", recovery_from=failed)
    with pytest.raises(Refused, match=r"look budget exceeded before any mutation: 660 existing \+ 232 pending"):
        executor.recover(recovery)
    assert box.path("ledger/test_look_ledger.jsonl").read_bytes() == ledger_bytes and len(_history(box).rows) == 660
    assert sorted(p.name for p in box.path("ledger/authorizations").iterdir()) == authorizations
    assert not box.path(f"ledger/authorizations/{recovery.batch_id}").exists()
    assert sorted(p.name for p in box.path("reservations").iterdir()) == reservations
    assert sorted(p.name for p in box.path("exposures").iterdir()) == exposures
    assert [r for r in executor._audit_records() if r["state"] == "EXPOSURE_RECORDED"] == exposed
    assert len(box.events) == events
    assert box.json("state.json")["status"] == "REFUSED_PRE_EXPOSURE"
    assert file_sha256(REAL_LEDGER) == REAL_LEDGER_SHA256


def test_C_append_rechecks_the_budget_itself(jsonl, frozen):
    box, executor = jsonl
    _at_660(box, executor, frozen)
    before = sorted(p.name for p in box.path("ledger/authorizations").iterdir())
    pending = sorted(executor.operations)
    with pytest.raises(Refused, match="look budget exceeded"):
        executor.ledger.append(permit(box, frozen, "recovery"), pending, {e: e + "-r9" for e in pending}, 660)
    assert len(_history(box).rows) == 660
    assert sorted(p.name for p in box.path("ledger/authorizations").iterdir()) == before


def test_C_exact_budget_passes_and_one_more_look_fails(jsonl, frozen):
    box, executor = jsonl
    executor.fault = _crash_at("after_reservation")
    with pytest.raises(OSError):
        executor.score_all(permit(box, frozen))
    executor.fault = None
    rows = executor.ledger.verify()
    retries = {e: e + "-r1" for e in executor.operations}
    executor.ledger.precheck(rows, retries)                               # 428 + 232 - 196 = 464: on budget
    with pytest.raises(Refused, match="465 > 464"):
        executor.ledger.precheck(rows + [{"experiment_id": "SYNTHETIC-extra"}], retries)
    state = executor.recover(permit(box, frozen, "recovery", recovery_from=box.json("state.json")["batch_id"]))
    assert len(state["tables"]) == 232 and len(_history(box).rows) == 660   # exactly on budget succeeds


def test_C_duplicate_and_malformed_identities_still_fail(jsonl, frozen):
    box, executor = jsonl
    executor.score_all(permit(box, frozen))
    rows = executor.ledger.verify()
    eid, other = sorted(executor.operations)[:2]
    with pytest.raises(Refused, match="duplicate evaluation attempt identity"):
        executor.ledger.precheck(rows, {eid: eid})
    with pytest.raises(Refused, match="duplicate pending identities"):
        executor.ledger.precheck(rows, {eid: eid + "-r1", other: eid + "-r1"})
    with pytest.raises(Refused, match="well-formed retry"):
        executor.ledger.precheck(rows, {eid: eid + "-retry"})


def test_C_pre_append_crash_consumes_no_budget_and_exposed_retries_do(jsonl, frozen):
    box, executor = jsonl
    executor.fault = _crash_at("after_batch_authorization")             # authorization written, no look appended
    with pytest.raises(OSError):
        executor.score_all(permit(box, frozen))
    executor.fault = _crash_at("after_reservation")                      # recovery exposes 232 then crashes
    with pytest.raises(OSError):
        executor.recover(permit(box, frozen, "recovery", recovery_from=box.json("state.json")["batch_id"]))
    executor.fault = None
    assert len(_history(box).rows) == 428                                # the orphan authorization counted nothing
    state = executor.recover(permit(box, frozen, "recovery", recovery_from=box.json("state.json")["batch_id"]))
    assert len(_history(box).rows) == 660 and all(e["attempt_id"].endswith("-r1") for e in state["tables"].values())


def test_C_concurrent_recoveries_cannot_jointly_exceed_the_budget(jsonl, frozen):
    box, a = jsonl
    a.fault = _crash_at("after_reservation")
    with pytest.raises(OSError):
        a.score_all(permit(box, frozen))
    failed = box.json("state.json")["batch_id"]
    b = Executor(box, frozen)
    outcome = []

    def contend(stage, unit):
        if stage == "before_reservation" and not outcome:               # A has passed its precheck
            try:
                b.recover(permit(box, frozen, "recovery", recovery_from=failed))
                outcome.append("B accepted")
            except Refused as exc:
                outcome.append(str(exc))

    a.fault = contend
    a.recover(permit(box, frozen, "recovery", recovery_from=failed))
    assert len(outcome) == 1 and outcome[0] != "B accepted"
    assert len(_history(box).rows) == 660


def test_C_ledger_moved_after_reservation_publishes_no_authorization(jsonl, frozen):
    box, executor = jsonl

    def move(stage, unit):
        if stage == "after_authorization":                                # reservation written; precheck now stale
            with box.path("ledger/test_look_ledger.jsonl").open("ab") as out:
                out.write(b'{"experiment_id": "SYNTHETIC-foreign"}\n')

    executor.fault = move
    with pytest.raises(Refused):
        executor.score_all(permit(box, frozen))
    batch = box.json("state.json")["batch_id"]
    assert not box.path(f"ledger/authorizations/{batch}").exists()
    assert not box.path(f"exposures/{batch}.json").exists()
