"""SYNTHETIC-only security, crash, identity, mapping and statistical integration.

No production reader/scope, real model inference, real ledger write or replay.
"""
from copy import deepcopy
from pathlib import Path
import json
import threading

import h5py
import numpy as np
import pandas as pd
import pytest

from src.experiment.phase5_execution import (
    Executor, FixturePermit, FrozenPackage, Refused, Sandbox, authorize_fixture,
    execute_real_protected,
)
from src.experiment.phase5_fixture import ANOMALY, NORMAL, create_fixture
from src.experiment.phase5_inputs import SCORE_COLUMNS, SYNTHETIC, atomic_json
from src.experiment.provenance import canonical_sha256, content_sha256, file_sha256


@pytest.fixture(scope="session")
def frozen():
    return FrozenPackage.load()


@pytest.fixture
def fixture(tmp_path, frozen):
    sandbox = Sandbox(tmp_path / "SYNTHETIC")
    create_fixture(sandbox, frozen)
    return sandbox, Executor(sandbox, frozen)


def permit(sandbox, package, action="score", **kw):
    return authorize_fixture(sandbox, package, action, **kw)


def test_trusted_original_artifacts(frozen):
    assert len(frozen.registry["units"]) == 52
    assert len(frozen.registry["calibrations"]) == 208
    frozen.verify_memory()
    report = json.loads(Path("reports/phase5_development/l1/l1_completion_and_reproduction.json").read_text())
    assert report["input_hash_forms"] == {"CRLF": 14, "LF": 10}


@pytest.mark.parametrize("field", ["invented", "threshold", "model", "source", "split", "cohort", "subset", "L3", "missing", "duplicate"])
def test_mutated_identity_refused_before_read(fixture, frozen, field):
    sandbox, executor = fixture
    auth = permit(sandbox, frozen)
    package = deepcopy(frozen)
    u = next(u for u in package.registry["units"] if u["L1"]["source"] == "new_fit_phase5")
    eid = u["L1"]["experiment_id"]
    rec = package.registry["detectors"][eid]
    if field == "invented":
        u["L1"]["experiment_id"] = "p3-" + "0" * 24
        package.registry["detectors"][u["L1"]["experiment_id"]] = package.registry["detectors"].pop(eid)
    elif field == "threshold":
        package.registry["calibrations"][next(iter(package.registry["calibrations"]))]["threshold"]["value"] += .001
    elif field == "model":
        rec["model_manifest"]["model_id"] = "p4-model-forged"
    elif field == "source":
        rec["code_git_sha"] = "0" * 40
    elif field == "split":
        rec["split_manifest_hash"] = "f" * 64
    elif field == "cohort":
        rec["cohort_contracts"]["training_contract_hash"] = "f" * 64
    elif field == "subset":
        package.registry["calibrations"][next(iter(package.registry["calibrations"]))]["subset"] = ["mic"]
    elif field == "L3":
        u["L3"]["all_value_invalid"] = "DECISION"
    elif field == "missing":
        package.registry["units"].pop()
    else:
        package.registry["units"][-1] = deepcopy(u)
    executor.package = package
    with pytest.raises(Refused, match="in-memory registry"):
        executor.score_all(auth)
    assert sandbox.events == [] and len(sandbox.json("ledger.json")) == 196


def test_rehashed_forgery_cannot_replace_git_anchor(tmp_path):
    root = tmp_path / "forged"
    root.mkdir()
    manifest = json.loads(Path("reports/phase5_freeze_candidate/candidate_manifest.json").read_text(encoding="utf-8"))
    manifest["files"]["scientific_registry.json"] = "0" * 64
    atomic_json(root / "candidate_manifest.json", manifest)
    with pytest.raises(Refused, match="trusted commit"):
        FrozenPackage.load(root)


@pytest.mark.parametrize("kind", ["missing", "changed"])
def test_model_files_rechecked_after_load(fixture, frozen, monkeypatch, kind):
    sandbox, executor = fixture
    auth = permit(sandbox, frozen)
    import src.experiment.phase5_execution as module
    original = module.canonical_sha256

    def altered(path):
        if str(path).replace("\\", "/").startswith("data/interim/phase5/models/"):
            if kind == "missing":
                raise FileNotFoundError("SYNTHETIC missing model simulation")
            return "0" * 64
        return original(path)

    monkeypatch.setattr(module, "canonical_sha256", altered)
    with pytest.raises((Refused, FileNotFoundError)):
        executor.score_all(auth)
    assert not sandbox.events and len(sandbox.json("ledger.json")) == 196


def test_frozen_statistical_source_change_refuses_before_read(fixture, frozen, monkeypatch):
    sandbox, executor = fixture
    auth = permit(sandbox, frozen)
    import src.experiment.phase5_execution as module
    original = module.canonical_bytes

    def changed(path):
        return b"SYNTHETIC changed scientific source" if str(path) == "src/fusion/statistics.py" else original(path)

    monkeypatch.setattr(module, "canonical_bytes", changed)
    with pytest.raises(Refused, match="scientific source"):
        executor.score_all(auth)
    assert not sandbox.events and len(sandbox.json("ledger.json")) == 196


def test_no_authorization_and_real_execution_refuse(fixture):
    sandbox, executor = fixture
    with pytest.raises(Refused, match="authorization"):
        executor.score_all()
    with pytest.raises(Refused, match="disabled"):
        execute_real_protected()
    assert not sandbox.events


def test_foreign_permit_and_mutated_fixture_refuse(fixture, frozen):
    sandbox, executor = fixture
    auth = permit(sandbox, frozen)
    from dataclasses import replace
    with pytest.raises(Refused):
        executor.score_all(replace(auth, sandbox_identity="SYNTHETIC-foreign"))
    fixture_data = sandbox.json("fixture.json")
    fixture_data["recipe"] = "tampered"
    atomic_json(sandbox.path("fixture.json"), fixture_data, replace=True)
    with pytest.raises(Refused, match="identity mismatch"):
        executor.score_all(auth)
    assert not sandbox.events and len(sandbox.json("ledger.json")) == 196


def test_real_and_escaping_paths_refuse(tmp_path):
    with pytest.raises(Refused):
        Sandbox(Path.cwd())
    sandbox = Sandbox(tmp_path / "SYNTHETIC")
    for path in (Path.cwd() / "data", "../escape", Path.cwd()):
        with pytest.raises(Refused):
            sandbox.path(path)


def test_projection_and_durable_pre_access(fixture, frozen, monkeypatch):
    sandbox, executor = fixture
    parquet = pd.read_parquet
    dataset = h5py.Group.__getitem__
    requested = []

    def projected(path, *args, **kw):
        columns = kw.get("columns")
        assert columns and "fake_label" not in columns
        assert sandbox.json("state.json")["status"] == "EXPOSED"
        assert len(sandbox.json("ledger.json")) == 428
        reservations = list(sandbox.path("reservations").glob("*.json"))
        assert len(reservations) == 1 and sandbox.path("exposures").is_dir()
        requested.append(tuple(columns))
        frame = parquet(path, *args, **kw)
        assert NORMAL not in repr(frame.to_dict()) and ANOMALY not in repr(frame.to_dict())
        return frame

    def numeric_only(self, key):
        assert key in {"mic", "acc", "gyro"}
        return dataset(self, key)

    monkeypatch.setattr(pd, "read_parquet", projected)
    monkeypatch.setattr(h5py.Group, "__getitem__", numeric_only)
    state = executor.score_all(permit(sandbox, frozen))
    assert len(state["tables"]) == 232 and requested
    assert not any(phase == "analysis" for phase, _ in sandbox.events)
    for entry in state["tables"].values():
        payload = sandbox.json(entry["file"])
        assert NORMAL not in repr(payload) and ANOMALY not in repr(payload)
        assert tuple(pd.DataFrame(payload["rows"]).columns) != ()
        assert set(payload["rows"][0]) == set(SCORE_COLUMNS)
    # score capability cannot open labels or historical reused-L1 comparators
    score_access = executor._access(state["batch_id"], "score")
    with pytest.raises(Refused, match="analysis capability"):
        executor.inputs.labels(score_access, "RoboticArm")
    sandbox.path("attempts.jsonl").write_bytes(sandbox.path("attempts.jsonl").read_bytes() + b"SYNTHETIC incomplete log")
    with pytest.raises(Refused, match="partial/corrupt"):
        executor.verify_complete()


@pytest.mark.parametrize("stage,count", [
    ("before_reservation", 196), ("after_authorization", 196),
    ("after_ledger", 428), ("after_reservation", 428), ("before_read", 428),
    ("after_read", 428), ("partial_write", 428), ("after_publication", 428),
])
def test_injected_crashes_preserve_accounting(fixture, frozen, stage, count):
    sandbox, executor = fixture

    def fail(where, unit):
        if where == stage:
            raise OSError("SYNTHETIC crash at " + stage)

    executor.fault = fail
    with pytest.raises(OSError):
        executor.score_all(permit(sandbox, frozen))
    assert len(sandbox.json("ledger.json")) == count
    assert sandbox.path("attempt.lock").exists()
    if stage in {"before_reservation", "after_authorization", "after_ledger", "after_reservation", "before_read"}:
        assert sandbox.events == []
    with pytest.raises(Refused):
        executor.score_all(permit(sandbox, frozen))
    with pytest.raises(Refused):
        executor.analyse_once(permit(sandbox, frozen, "analysis"))
    if stage in {"partial_write", "after_publication"}:
        state = sandbox.json("state.json")
        recovery = permit(sandbox, frozen, "recovery", recovery_from=state["batch_id"])
        with pytest.raises(Refused, match="interrupted write"):
            executor.recover(recovery)


def test_recovery_retains_persisted_unit_and_uses_new_ids(fixture, frozen):
    sandbox, executor = fixture
    count = 0

    def fail(stage, unit):
        nonlocal count
        if stage == "before_read":
            count += 1
            if count == 2:
                raise OSError("SYNTHETIC interruption after one persisted unit")

    executor.fault = fail
    with pytest.raises(OSError):
        executor.score_all(permit(sandbox, frozen))
    first = sandbox.json("state.json")
    assert len(first["tables"]) == 1
    eid, entry = next(iter(first["tables"].items()))
    before = file_sha256(sandbox.path(entry["file"]))
    executor.fault = None
    recovery = permit(sandbox, frozen, "recovery", recovery_from=first["batch_id"])
    state = executor.recover(recovery)
    assert len(state["tables"]) == 232 and len(sandbox.json("ledger.json")) == 659
    assert file_sha256(sandbox.path(entry["file"])) == before
    assert sum(e["attempt_id"].endswith("-r1") for e in state["tables"].values()) == 231
    assert state["tables"][eid]["attempt_id"] == eid
    with pytest.raises(Refused):
        executor.recover(recovery)


def test_hard_interruption_recovery_never_steals_live_lock(fixture, frozen):
    sandbox, executor = fixture

    def fail(stage, unit):
        if stage == "after_reservation":
            raise OSError("SYNTHETIC interrupted process simulation")

    executor.fault = fail
    with pytest.raises(OSError):
        executor.score_all(permit(sandbox, frozen))
    state = sandbox.json("state.json")
    state["status"] = "EXPOSED"  # emulate missing exception-handler state write
    atomic_json(sandbox.path("state.json"), state, replace=True)
    # A hard exit also leaves the lock as the live owner wrote it (no failure mark).
    import os
    import psutil
    atomic_json(sandbox.path("attempt.lock"), {"scope": SYNTHETIC, "pid": os.getpid(),
                "process_created": psutil.Process().create_time(), "state": "ACTIVE"}, replace=True)
    auth = permit(sandbox, frozen, "recovery", recovery_from=state["batch_id"])
    with pytest.raises(Refused, match="live/concurrent"):
        executor.recover(auth)
    # Synthetic dead-PID lock record emulates a killed owner; no real process
    # is killed and no OS signal is sent by this test or implementation.
    atomic_json(sandbox.path("attempt.lock"), {"scope": SYNTHETIC, "pid": 2147483647, "process_created": 0}, replace=True)
    executor.fault = None
    result = executor.recover(auth)
    assert len(result["tables"]) == 232 and len(sandbox.json("ledger.json")) == 660
    assert all(entry["attempt_id"].endswith("-r1") for entry in result["tables"].values())


def test_pre_exposure_explicit_recovery_consumes_first_looks_only(fixture, frozen):
    sandbox, executor = fixture

    def fail(stage, unit):
        if stage == "before_reservation":
            raise OSError("SYNTHETIC pre-exposure refusal")

    executor.fault = fail
    with pytest.raises(OSError):
        executor.score_all(permit(sandbox, frozen))
    state = sandbox.json("state.json")
    assert state["status"] == "REFUSED_PRE_EXPOSURE" and len(sandbox.json("ledger.json")) == 196
    executor.fault = None
    result = executor.recover(permit(sandbox, frozen, "recovery", recovery_from=state["batch_id"]))
    assert len(sandbox.json("ledger.json")) == 428
    assert all(eid == entry["attempt_id"] for eid, entry in result["tables"].items())


def test_concurrent_attempt_refuses(fixture, frozen):
    sandbox, executor = fixture
    auth = permit(sandbox, frozen)
    entered, release = threading.Event(), threading.Event()
    errors = []

    def pause(stage, unit):
        if stage == "before_reservation":
            entered.set()
            release.wait(20)
            raise OSError("SYNTHETIC stop concurrent exercise")

    executor.fault = pause

    def worker():
        try:
            executor.score_all(auth)
        except OSError as exc:
            errors.append(str(exc))

    thread = threading.Thread(target=worker)
    thread.start()
    assert entered.wait(10)
    try:
        with pytest.raises(Refused):
            Executor(sandbox, frozen).score_all(auth)
        assert not sandbox.events
    finally:
        release.set()
        thread.join(20)
    assert not thread.is_alive() and errors


def test_atomic_partial_never_becomes_completed_result(tmp_path):
    path = tmp_path / "SYNTHETIC.json"

    def fail(*args):
        raise OSError("SYNTHETIC interrupted write")

    with pytest.raises(OSError):
        atomic_json(path, {"score": .4}, fault=fail)
    assert not path.exists() and path.with_name(path.name + ".partial").exists()
    with pytest.raises(FileExistsError):
        atomic_json(path, {"score": .4})


def test_input_integrity_failure_is_after_exposure(fixture, frozen):
    sandbox, executor = fixture
    auth = permit(sandbox, frozen)
    name = next(iter(executor.fixture["historical"].values()))
    sandbox.path(name).write_bytes(b"SYNTHETIC corruption")
    with pytest.raises(Refused, match="input hash mismatch"):
        executor.score_all(auth)
    assert len(sandbox.json("ledger.json")) == 428


def test_complete_mapping_analysis_and_once_semantics(fixture, frozen):
    sandbox, executor = fixture
    state = executor.score_all(permit(sandbox, frozen))
    assert len(state["tables"]) == 232
    for u in frozen.registry["units"]:
        assert u["L3"]["two_value_invalid_survivor"] == u["L0"]
        for mod in ("mic", "acc", "gyro"):
            pair = "+".join(m for m in ("mic", "acc", "gyro") if m != mod)
            assert u["L3"]["one_value_invalid"][mod] == u["L2"][pair]
        for cid in u["L2"].values():
            calibration = frozen.registry["calibrations"][cid]
            expected = np.zeros(12)
            for mod in ("mic", "acc", "gyro"):
                if mod in calibration["subset"]:
                    source = pd.read_parquet(sandbox.path(executor.fixture["historical"][u["L0"][mod]]), columns=["score"])["score"].to_numpy()
                    c = calibration["modality_calibration"][mod]
                    expected += (source - c["median"]) / c["iqr"]
            actual = np.array([r["score"] for r in sandbox.json(state["tables"][cid]["file"])["rows"]])
            np.testing.assert_array_equal(actual, expected / len(calibration["subset"]))
    with pytest.raises(Refused):
        executor.analyse_once()
    analysis = executor.analyse_once(permit(sandbox, frozen, "analysis"))
    assert analysis["scope"] == SYNTHETIC and analysis["not_real_performance"]
    assert analysis["replicates"] == 10_000 and analysis["seed"] == 1995
    assert analysis["primary_intervals"] == 420 and analysis["primary_contrasts"] == 140
    assert analysis["protected_L3_executions"] == analysis["missing_scores"] == 0
    primary = [r for r in analysis["intervals"] if r["contrast"].startswith("P") and r["metric"] in {"auroc", "recall", "healthy_target_fpr"}]
    assert len(primary) == 420
    assert any(r["valid_replicates"] < 9000 and r["category"] == "INDETERMINATE" for r in primary)
    assert any(r["contrast"] == "S6:T-S" for r in analysis["intervals"])
    assert any(phase == "analysis" and "labels" in name for phase, name in sandbox.events)
    with pytest.raises(Refused, match="already claimed"):
        executor.analyse_once(permit(sandbox, frozen, "analysis"))
    with pytest.raises(Refused):
        executor.score_all(permit(sandbox, frozen))
    # A completed analysis has its own final integrity check.
    sandbox.path("analysis.json").write_bytes(b"SYNTHETIC corrupted final analysis")
    with pytest.raises(Refused, match="analysis integrity"):
        executor.verify_complete()


@pytest.mark.parametrize("kind", ["missing", "nonfinite", "parent", "labels", "duplicate"])
def test_missing_or_misaligned_scores_never_become_abstentions(fixture, frozen, kind):
    sandbox, executor = fixture
    frame = pd.DataFrame(executor.fixture["cohorts"]["RoboticArm"])
    frame["score"], frame["window_count"] = .4, 2
    if kind == "missing":
        frame = frame.iloc[:-1]
    elif kind == "nonfinite":
        frame.loc[0, "score"] = np.nan
    elif kind == "parent":
        frame.loc[0, "parent_recording"] = "SYNTHETIC-wrong"
    elif kind == "labels":
        frame["fake_label"] = ANOMALY
    else:
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    with pytest.raises(Refused):
        executor._aligned(frame, "RoboticArm")


def test_abstention_retention_and_strict_threshold_ties():
    from src.experiment.phase5_analysis import abstention_accounting
    from src.experiment.metrics import protocol_b_threshold_metrics
    assert abstention_accounting([True, False, None, None]) == {"total": 4, "decisions": 2, "abstentions": 2, "dropped": 0}
    result = protocol_b_threshold_metrics(np.array([0, 1]), np.array([.9, .9]), threshold=.9)
    assert result.false_positive == result.true_positive == 0


def test_completion_detects_partial_and_changed_output(fixture, frozen):
    sandbox, executor = fixture
    state = executor.score_all(permit(sandbox, frozen))
    eid = next(iter(state["tables"]))
    path = sandbox.path(state["tables"][eid]["file"])
    path.write_bytes(b"SYNTHETIC altered output")
    with pytest.raises(Refused, match="hash/path"):
        executor.verify_complete()


# =====================================================================================
# Phase 5.5C: N1 read-capability forgery, N3 interruption/recovery, N4 candidate identity
# =====================================================================================
import dataclasses
import os
import shutil

import psutil

from src.experiment import phase5_execution as execution
from src.experiment.phase5_inputs import Access


def _complete(sandbox, executor, frozen):
    return executor.score_all(permit(sandbox, frozen))


def test_N1_access_cannot_carry_its_own_verifier(fixture):
    with pytest.raises(TypeError):
        Access(sandbox=fixture[0], nonce=object(), batch_id="x", phase="analysis", check=lambda access: None)
    assert not any(callable(getattr(Access, f.name, None)) for f in dataclasses.fields(Access))


def test_N1_constructed_or_copied_capabilities_are_refused(fixture, frozen):
    sandbox, executor = fixture
    state = _complete(sandbox, executor, frozen)
    batch = state["batch_id"]
    forged = Access(executor._identity, sandbox.identity, batch, "analysis", "f" * 64)
    for read in (lambda a: executor.inputs.labels(a, "RoboticArm"), lambda a: executor.inputs.features(a, "RoboticArm|S")):
        with pytest.raises(Refused, match="unissued"):
            read(forged)
    legit = executor._access(batch, "score")
    # Copying a real token into an object with escalated phase or other fields is refused.
    for altered in (dataclasses.replace(legit, phase="analysis"), dataclasses.replace(legit, batch_id="SYNTHETIC-other"),
                    dataclasses.replace(legit, executor_identity="0" * 32)):
        with pytest.raises(Refused, match="unissued"):
            executor.inputs.labels(altered, "RoboticArm")
    with pytest.raises(Refused, match="analysis capability"):
        executor.inputs.labels(legit, "RoboticArm")
    assert not any(phase == "analysis" for phase, _ in sandbox.events)


def test_N1_capabilities_are_revoked_when_their_attempt_ends(fixture, frozen):
    sandbox, executor = fixture
    captured = []

    def capture(stage, unit):
        if stage == "before_read" and not captured:
            captured.extend(executor._capabilities.values())

    executor.fault = capture
    _complete(sandbox, executor, frozen)
    assert captured and not executor._capabilities
    with pytest.raises(Refused, match="revoked or stale"):
        executor.inputs.features(captured[0], "RoboticArm|S")


def test_N1_capability_of_another_executor_or_sandbox_is_refused(fixture, frozen, tmp_path):
    sandbox, executor = fixture
    state = _complete(sandbox, executor, frozen)
    legit = executor._access(state["batch_id"], "score")
    other_executor = Executor(sandbox, frozen)
    with pytest.raises(Refused, match="unissued"):
        other_executor.inputs.features(legit, "RoboticArm|S")
    second = Sandbox(tmp_path / "SYNTHETIC-second")
    create_fixture(second, frozen)
    with pytest.raises(Refused, match="unissued"):
        Executor(second, frozen).inputs.features(legit, "RoboticArm|S")


def test_N1_no_capability_without_durable_exposure_records(fixture, frozen):
    sandbox, executor = fixture
    with pytest.raises(Refused, match="missing or unreadable"):
        executor._access("SYNTHETIC-never-reserved", "score")
    state = _complete(sandbox, executor, frozen)
    legit = executor._access(state["batch_id"], "score")
    exposure = sandbox.path(f"exposures/{state['batch_id']}.json")
    held = exposure.read_bytes()
    exposure.unlink()
    with pytest.raises(Refused, match="missing or unreadable"):
        executor.inputs.features(legit, "RoboticArm|S")
    exposure.write_bytes(held)
    ledger = sandbox.json("ledger.json")
    atomic_json(sandbox.path("ledger.json"), ledger[:300], replace=True)
    with pytest.raises(Refused, match="exposure not durably recorded"):
        executor.inputs.features(legit, "RoboticArm|S")


def test_N1_analysis_before_completion_and_permit_confusion_are_refused(fixture, frozen):
    sandbox, executor = fixture
    with pytest.raises(Refused, match="no scored batch"):
        executor.analyse_once(permit(sandbox, frozen, "analysis"))
    with pytest.raises(Refused, match="wrong authorization"):
        executor.score_all(permit(sandbox, frozen, "recovery", recovery_from="x"))
    with pytest.raises(Refused, match="wrong authorization"):
        executor.score_all(permit(sandbox, frozen, "analysis"))
    with pytest.raises(Refused, match="wrong authorization"):
        executor.recover(permit(sandbox, frozen, "incident"))
    assert not sandbox.events and len(sandbox.json("ledger.json")) == 196
    first, second = permit(sandbox, frozen, "recovery", recovery_from="B"), permit(sandbox, frozen, "recovery", recovery_from="B")
    assert first.batch_id != second.batch_id       # every authorization is a distinct identity


def _nth_control_write(n):
    seen = []

    def fault(stage, unit):
        if stage == "control_partial_write":
            seen.append(unit)
            if len(seen) == n:
                raise OSError(f"SYNTHETIC crash during control write {n} ({unit})")
    return fault


# n -> ledger rows after the crash, ledger rows after audited resolution + recovery
CONTROL_WRITES = {1: (196, 428), 2: (196, 428), 3: (196, 428), 4: (428, 660), 5: (428, 660), 6: (428, 660)}


@pytest.mark.parametrize("n", sorted(CONTROL_WRITES))
def test_N3_interrupted_control_writes_resolve_only_through_audited_actions(fixture, frozen, n):
    sandbox, executor = fixture
    after_crash, after_recovery = CONTROL_WRITES[n]
    executor.fault = _nth_control_write(n)
    with pytest.raises(OSError):
        executor.score_all(permit(sandbox, frozen))
    executor.fault = None
    assert len(sandbox.json("ledger.json")) == after_crash
    if after_crash == 196:
        assert sandbox.events == []                 # no read before durable exposure
    report = executor.diagnose()
    assert report["classification"] == "INTERRUPTED_WRITE" and report["partials"]
    # Nothing proceeds while the partial is present.
    with pytest.raises(Refused):
        executor.score_all(permit(sandbox, frozen))
    if sandbox.path("state.json").exists():
        with pytest.raises(Refused, match="interrupted write"):
            executor.recover(permit(sandbox, frozen, "recovery", recovery_from=sandbox.json("state.json")["batch_id"]))
    record = executor.resolve_interrupted_writes(permit(sandbox, frozen, "incident"))
    for item in record["items"]:
        quarantined = sandbox.path(f"incidents/{record['batch_id']}/{item['path']}")
        assert quarantined.is_file() and file_sha256(quarantined) == item["sha256"]
    if sandbox.path("state.json").exists():
        state = sandbox.json("state.json")
        result = executor.recover(permit(sandbox, frozen, "recovery", recovery_from=state["batch_id"]))
    else:
        result = executor.score_all(permit(sandbox, frozen))
    assert len(result["tables"]) == 232 and len(sandbox.json("ledger.json")) == after_recovery
    renamed = sum(e["attempt_id"].endswith("-r1") for e in result["tables"].values())
    assert renamed == (232 if after_crash == 428 else 0)
    executor.verify_complete()
    events = [r["state"] for r in executor._audit_records()]
    assert "INTERRUPTED_WRITES_QUARANTINED" in events


def test_N3_hard_crash_snapshot_requires_dead_owner_and_audited_resolution(fixture, frozen, tmp_path):
    sandbox, executor = fixture
    hard = tmp_path / "SYNTHETIC-hard-crash"
    seen = []

    def snapshot_then_crash(stage, unit):
        if stage == "control_partial_write":
            seen.append(unit)
            if len(seen) == 5:       # the EXPOSED state write, after the 232 exposures
                shutil.copytree(sandbox.root, hard)   # exact on-disk state; no handler has run
                raise OSError("SYNTHETIC hard crash")

    executor.fault = snapshot_then_crash
    with pytest.raises(OSError):
        executor.score_all(permit(sandbox, frozen))
    box = Sandbox(hard)
    crashed = Executor(box, frozen)
    assert box.json("attempt.lock")["state"] == "ACTIVE" and len(box.json("ledger.json")) == 428
    with pytest.raises(Refused, match="may be live"):
        crashed.resolve_interrupted_writes(permit(box, frozen, "incident"))
    atomic_json(box.path("attempt.lock"), {"scope": SYNTHETIC, "pid": 2147483647, "process_created": 0,
                                          "state": "ACTIVE"}, replace=True)
    crashed.resolve_interrupted_writes(permit(box, frozen, "incident"))
    state = box.json("state.json")
    result = crashed.recover(permit(box, frozen, "recovery", recovery_from=state["batch_id"]))
    assert len(result["tables"]) == 232 and len(box.json("ledger.json")) == 660


def test_N3_interrupted_recovery_gets_new_unique_identities(fixture, frozen):
    sandbox, executor = fixture
    reads = []

    def first(stage, unit):
        if stage == "before_read":
            reads.append(unit)
            if len(reads) == 2:
                raise OSError("SYNTHETIC first interruption")

    executor.fault = first
    with pytest.raises(OSError):
        executor.score_all(permit(sandbox, frozen))
    original = sandbox.json("state.json")
    kept, kept_entry = next(iter(original["tables"].items()))

    def during_recovery(stage, unit):
        if stage == "after_reservation":
            raise OSError("SYNTHETIC interruption during recovery")

    executor.fault = during_recovery
    first_recovery = permit(sandbox, frozen, "recovery", recovery_from=original["batch_id"])
    with pytest.raises(OSError):
        executor.recover(first_recovery)
    interrupted = sandbox.json("state.json")
    assert interrupted["batch_id"] == first_recovery.batch_id and interrupted["status"] == "FAILED_TECHNICAL"
    executor.fault = None
    with pytest.raises(Refused, match="wrong recovery authority"):
        executor.recover(permit(sandbox, frozen, "recovery", recovery_from=original["batch_id"]))
    second_recovery = permit(sandbox, frozen, "recovery", recovery_from=first_recovery.batch_id)
    result = executor.recover(second_recovery)
    assert second_recovery.batch_id != first_recovery.batch_id
    assert len(sandbox.json("ledger.json")) == 428 + 231 + 231
    assert sum(e["attempt_id"].endswith("-r2") for e in result["tables"].values()) == 231
    assert result["tables"][kept] == kept_entry                    # valid result retained, never overwritten
    ledger_ids = [r["experiment_id"] for r in sandbox.json("ledger.json")]
    assert len(ledger_ids) == len(set(ledger_ids))
    authorized = [r for r in executor._audit_records() if r["state"] == "RECOVERY_AUTHORIZED"]
    assert [r["batch_id"] for r in authorized] == [first_recovery.batch_id, second_recovery.batch_id]


def test_N3_recovery_refuses_a_changed_retained_result(fixture, frozen):
    sandbox, executor = fixture
    reads = []

    def fail(stage, unit):
        if stage == "before_read":
            reads.append(unit)
            if len(reads) == 3:
                raise OSError("SYNTHETIC interruption")

    executor.fault = fail
    with pytest.raises(OSError):
        executor.score_all(permit(sandbox, frozen))
    state = sandbox.json("state.json")
    entry = next(iter(state["tables"].values()))
    sandbox.path(entry["file"]).write_bytes(b"SYNTHETIC tampered retained result")
    executor.fault = None
    with pytest.raises(Refused, match="retained output changed"):
        executor.recover(permit(sandbox, frozen, "recovery", recovery_from=state["batch_id"]))


@pytest.mark.parametrize("stage", ["after_analysis_claim", "after_analysis_compute", "control_partial_write"])
def test_N3_interrupted_analysis_is_never_repeated_automatically(fixture, frozen, stage):
    sandbox, executor = fixture
    _complete(sandbox, executor, frozen)

    def fail(where, unit):
        if where == stage and (stage != "control_partial_write" or unit == "analysis.json"):
            raise OSError("SYNTHETIC analysis interruption at " + stage)

    executor.fault = fail
    with pytest.raises(OSError):
        executor.analyse_once(permit(sandbox, frozen, "analysis"))
    executor.fault = None
    if stage == "after_analysis_claim":
        assert not any(phase == "analysis" for phase, _ in sandbox.events)   # claim logged, no label read yet
    assert executor.diagnose()["classification"] == "INTERRUPTED_ANALYSIS"
    assert sandbox.json("state.json")["status"] == "SCORED"
    with pytest.raises(Refused):
        executor.analyse_once(permit(sandbox, frozen, "analysis"))
    with pytest.raises(Refused):
        executor.resolve_interrupted_writes(permit(sandbox, frozen, "incident"))
    with pytest.raises(Refused):
        executor.recover(permit(sandbox, frozen, "recovery", recovery_from=sandbox.json("state.json")["batch_id"]))
    # Deleting the claim file does not grant a fresh claim: the append-only log remembers it.
    sandbox.path("analysis.claim.json").unlink()
    for partial in sandbox.root.glob("analysis*.partial"):
        partial.unlink()
    with pytest.raises(Refused, match="previously claimed"):
        executor.analyse_once(permit(sandbox, frozen, "analysis"))


def test_N3_analysis_claim_is_durably_logged_before_any_label_access(fixture, frozen, monkeypatch):
    sandbox, executor = fixture
    _complete(sandbox, executor, frozen)
    original = type(executor.inputs).labels
    checked = []

    def labels(self, access, machine):
        events = [r["state"] for r in executor._audit_records()]
        assert "ANALYSIS_CLAIMED" in events and sandbox.path("analysis.claim.json").is_file()
        checked.append(machine)
        return original(self, access, machine)

    monkeypatch.setattr(type(executor.inputs), "labels", labels)
    executor.analyse_once(permit(sandbox, frozen, "analysis"))
    assert checked == ["RoboticArm", "BrushlessMotor"]


# ------------------------------------------------------------------ N4 candidate identity
def _candidate(tmp_path, frozen, *, mutate_source=None):
    """A simulated reviewed commit: manifest + evidence built from the current sources."""
    from hashlib import sha256
    root = tmp_path / "candidate"
    root.mkdir(parents=True)
    sources = {p: canonical_sha256(p) for p in execution.CANDIDATE_SOURCES}
    if mutate_source:
        sources[mutate_source] = "0" * 64
    identity = content_sha256({p: sources[p] for p in execution.EXECUTOR_SOURCES})
    evidence = (json.dumps({"executor_source_identity": identity, "scope": SYNTHETIC}) + "\n").encode()
    xml = b"<testsuites/>\n"
    integration = b'{"scope": "SYNTHETIC_ONLY"}\n'
    (root / "synthetic_evidence.json").write_bytes(evidence)
    (root / "integration_evidence.json").write_bytes(integration)
    (root / "regression_tests.xml").write_bytes(xml)
    (root / "development_sources.json").write_bytes(integration)
    (root / "focused_tests.xml").write_bytes(xml)
    manifest = {
        "status": execution.CANDIDATE_STATUS, "stage": "5.5C", "review_status": "UNREVIEWED",
        "real_execution_enabled": False, "authorization_issued": False, "independently_approved": False,
        "trusted_package_commit": execution.TRUSTED_COMMIT, "trusted_package_manifest_sha256": execution.TRUSTED_MANIFEST,
        "original_scientific_registry_sha256": frozen.proposed["identity"]["registry_sha256"],
        "executor_source_identity": identity, "sources": sources,
        "files": {"synthetic_evidence.json": sha256(evidence).hexdigest(), "focused_tests.xml": sha256(xml).hexdigest(),
                  "integration_evidence.json": sha256(integration).hexdigest(),
                  "regression_tests.xml": sha256(xml).hexdigest(), "development_sources.json": sha256(integration).hexdigest()},
        "generated_from_head": "0" * 40, "tracked_tree_clean_at_generation": False, "proposed_real_looks": 232,
        "real_ledger_rows": 196, "restrictions": [], "interfaces": {}, "coverage": {}}
    raw = (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    (root / "manifest.json").write_bytes(raw)
    store = {(execution.CANDIDATE_ROOT / n).as_posix(): (root / n).read_bytes()
             for n in ("manifest.json", *execution.CANDIDATE_FILES)}

    def reader(path):
        key = Path(path).as_posix()
        return store[key] if key in store else canonical_bytes_of(key)
    return root, reader, store


def canonical_bytes_of(path):
    from src.experiment.provenance import canonical_bytes
    return canonical_bytes(path)


def test_N4_simulated_reviewed_candidate_verifies(tmp_path, frozen):
    root, reader, _ = _candidate(tmp_path, frozen)
    manifest = execution.verify_executable_candidate("a" * 40, committed_reader=reader, root=root)
    assert manifest["executor_source_identity"] == execution.source_identity()


def test_N4_modified_executor_source_is_refused(tmp_path, frozen, monkeypatch):
    root, reader, _ = _candidate(tmp_path, frozen)
    original = execution.canonical_sha256
    monkeypatch.setattr(execution, "canonical_sha256", lambda p: "0" * 64 if str(p).replace("\\", "/") ==
                        "src/experiment/phase5_inputs.py" else original(p))
    with pytest.raises(Refused, match="modified after review"):
        execution.verify_executable_candidate("a" * 40, committed_reader=reader, root=root)


def test_N4_altered_trust_anchor_is_refused(tmp_path, frozen, monkeypatch):
    root, reader, _ = _candidate(tmp_path, frozen)
    monkeypatch.setattr(execution, "TRUSTED_COMMIT", "f" * 40)
    with pytest.raises(Refused, match="trust anchors"):
        execution.verify_executable_candidate("a" * 40, committed_reader=reader, root=root)


def test_N4_forged_or_restructured_manifest_is_refused(tmp_path, frozen):
    root, reader, store = _candidate(tmp_path, frozen)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["real_execution_enabled"] = True
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(Refused, match="differs from the reviewed commit"):
        execution.verify_executable_candidate("a" * 40, committed_reader=reader, root=root)
    manifest["real_execution_enabled"] = False
    manifest["extra"] = 1
    raw = json.dumps(manifest).encode()
    (root / "manifest.json").write_bytes(raw)
    store[(execution.CANDIDATE_ROOT / "manifest.json").as_posix()] = raw
    with pytest.raises(Refused, match="structure"):
        execution.verify_executable_candidate("a" * 40, committed_reader=reader, root=root)
    with pytest.raises(Refused, match="full reviewed commit"):
        execution.verify_executable_candidate("HEAD", committed_reader=reader, root=root)


def test_N4_modified_evidence_is_refused(tmp_path, frozen):
    root, reader, _ = _candidate(tmp_path, frozen)
    (root / "synthetic_evidence.json").write_bytes(b'{"executor_source_identity": "forged"}\n')
    with pytest.raises(Refused, match="evidence modified after review"):
        execution.verify_executable_candidate("a" * 40, committed_reader=reader, root=root)


def test_N4_internally_consistent_but_unreviewed_metadata_is_refused(tmp_path, frozen):
    # The working candidate is self-consistent for a changed source, but the reviewed
    # commit holds the original: a self-approved hash never passes.
    reviewed_root, reviewed_reader, _ = _candidate(tmp_path / "reviewed", frozen)
    forged_root, _, _ = _candidate(tmp_path / "forged", frozen, mutate_source="src/experiment/phase5_analysis.py")
    with pytest.raises(Refused, match="differs from the reviewed commit"):
        execution.verify_executable_candidate("a" * 40, committed_reader=reviewed_reader, root=forged_root)
    # And a "reviewed" commit that itself records a hash the working code does not have fails too.
    _, forged_reader, _ = _candidate(tmp_path / "forged2", frozen, mutate_source="src/experiment/phase5_analysis.py")
    forged2 = tmp_path / "forged2" / "candidate"
    with pytest.raises(Refused):
        execution.verify_executable_candidate("a" * 40, committed_reader=forged_reader, root=forged2)


# =====================================================================================
# Phase 5.5D: reader-binding bypass (N1), caller-controlled comparator class (N2),
# recovery takeover race (N3). Independent-review counterexamples reproduced as regressions.
# =====================================================================================
from src.experiment import phase5_inputs as inputs_module
from src.experiment.phase5_inputs import ProjectedInputs


def _record_raw_reads(monkeypatch):
    """Trap every parquet open and every input hash in the reader module."""
    calls = []
    real_read, real_hash = inputs_module.pd.read_parquet, inputs_module.file_sha256
    monkeypatch.setattr(inputs_module.pd, "read_parquet", lambda *a, **k: (calls.append(("read", a)), real_read(*a, **k))[1])
    monkeypatch.setattr(inputs_module, "file_sha256", lambda p: (calls.append(("hash", p)), real_hash(p))[1])
    return calls


def _reused_l1(frozen):
    return next(u["L1"]["experiment_id"] for u in frozen.registry["units"] if u["L1"]["source"] != "new_fit_phase5")


def test_D_N1_reviewer_counterexample_noop_verifier_reader_returns_no_label_rows(fixture, frozen, monkeypatch):
    sandbox, executor = fixture
    calls = _record_raw_reads(monkeypatch)
    manifest = sandbox.json("fixture.json")
    invented = Access("0" * 32, sandbox.identity, "SYNTHETIC-invented-no-exposure", "analysis", "0" * 64)
    returned = []
    # 5.5C counterexample: a separately constructed reader with a no-op verifier.
    with pytest.raises(Refused, match="bound to its issuing Executor"):
        returned.append(ProjectedInputs(sandbox, manifest, lambda access, phase: None))

    class Lenient(Executor):
        def _verify_capability(self, access, phase):
            return None
    with pytest.raises(Refused, match="bound to its issuing Executor"):
        ProjectedInputs(sandbox, manifest, Lenient.__new__(Lenient))
    # A genuine but different executor as owner: the reader is not that executor's reader.
    stray = ProjectedInputs(sandbox, executor.fixture, executor)
    with pytest.raises(Refused, match="not bound"):
        returned.append(stray.labels(invented, "RoboticArm"))
    # The legitimate reader with an invented batch: refused, even if a token were issued for it.
    with pytest.raises(Refused, match="unissued"):
        returned.append(executor.inputs.labels(invented, "RoboticArm"))
    with pytest.raises(Refused, match="missing or unreadable"):
        returned.append(executor.inputs.labels(executor._issue("SYNTHETIC-invented-no-exposure", "analysis"), "RoboticArm"))
    assert returned == [] and calls == [] and sandbox.events == []
    assert len(sandbox.json("ledger.json")) == 196


def test_D_N1_corrupted_reservation_invalidates_live_access(fixture, frozen):
    sandbox, executor = fixture
    state = _complete(sandbox, executor, frozen)
    legit = executor._access(state["batch_id"], "score")
    reservation = sandbox.path(f"reservations/{state['batch_id']}.json")
    record = json.loads(reservation.read_text(encoding="utf-8"))
    record["attempt_ids"] = {}
    reservation.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(Refused, match="authorization not durable"):
        executor.inputs.features(legit, "RoboticArm|S")


def test_D_N2_analysis_only_comparator_never_readable_while_scoring(fixture, frozen, monkeypatch):
    sandbox, executor = fixture
    state = _complete(sandbox, executor, frozen)
    score_access = executor._access(state["batch_id"], "score")
    l1 = _reused_l1(frozen)
    calls = _record_raw_reads(monkeypatch)
    for call in (lambda: executor.inputs.historical(score_access, l1),
                 lambda: executor.inputs.historical(access=score_access, eid=l1)):
        with pytest.raises(Refused, match="analysis capability"):
            call()
    for flag in (True, False):
        with pytest.raises(TypeError):                         # the caller flag no longer exists
            executor.inputs.historical(score_access, l1, reused_l1=flag)
    with pytest.raises(TypeError):
        executor.inputs.historical(score_access, l1, False)
    assert calls == []                                          # refused before any file was opened or hashed


def test_D_N2_identity_substitution_is_refused_before_reading(fixture, frozen, monkeypatch):
    sandbox, executor = fixture
    state = _complete(sandbox, executor, frozen)
    score_access = executor._access(state["batch_id"], "score")
    l0 = frozen.registry["units"][0]["L0"]["mic"]
    l1 = _reused_l1(frozen)
    calls = _record_raw_reads(monkeypatch)
    for bogus in ("p3-not-a-registered-identity", "../historical/" + l1, None, l1.upper()):
        with pytest.raises(Refused, match="not a frozen historical comparator identity"):
            executor.inputs.historical(score_access, bogus)
    # Aliasing a permitted identity to the analysis-only file is refused too.
    executor.fixture["historical"][l0] = f"historical/{l1}.parquet"
    with pytest.raises(Refused, match="does not match its trusted identity"):
        executor.inputs.historical(score_access, l0)
    assert calls == []


def test_D_N2_permitted_comparators_still_work(fixture, frozen):
    sandbox, executor = fixture
    state = _complete(sandbox, executor, frozen)
    l0 = frozen.registry["units"][0]["L0"]["acc"]
    frame = executor.inputs.historical(executor._access(state["batch_id"], "score"), l0)
    assert tuple(frame.columns) == SCORE_COLUMNS and len(frame) == 12
    executor.analyse_once(permit(sandbox, frozen, "analysis"))
    l1_reads = [(phase, name) for phase, name in sandbox.events if name == f"historical/{_reused_l1(frozen)}.parquet"]
    assert l1_reads and all(phase == "analysis" for phase, _ in l1_reads)
    assert not any(name.startswith("historical/") and name[len("historical/"):-len(".parquet")]
                   in {u["L1"]["experiment_id"] for u in frozen.registry["units"] if u["L1"]["source"] != "new_fit_phase5"}
                   and phase == "score" for phase, name in sandbox.events)


def _failed_after_one_unit(sandbox, executor, frozen):
    reads = []

    def fail(stage, unit):
        if stage == "before_read":
            reads.append(unit)
            if len(reads) == 2:
                raise OSError("SYNTHETIC interruption after one persisted unit")

    executor.fault = fail
    with pytest.raises(OSError):
        executor.score_all(permit(sandbox, frozen))
    executor.fault = None
    return sandbox.json("state.json")


def _recovery_events(executor):
    return [r for r in executor._audit_records() if r["state"] == "RECOVERY_AUTHORIZED"]


def test_D_N3A_reviewer_two_recovery_interleaving_admits_exactly_one(fixture, frozen):
    sandbox, a = fixture
    state = _failed_after_one_unit(sandbox, a, frozen)
    b = Executor(sandbox, frozen)
    b_stages, outcomes = [], []
    b.fault = lambda stage, unit: b_stages.append(stage)
    permit_a = permit(sandbox, frozen, "recovery", recovery_from=state["batch_id"])
    permit_b = permit(sandbox, frozen, "recovery", recovery_from=state["batch_id"])

    def interleave(stage, unit):
        # A holds admission and is between its owner check and its takeover:
        # exactly the window the 5.5C race used. B must not get in.
        if stage == "recovery_admitted":
            try:
                b.recover(permit_b)
                outcomes.append("B accepted")
            except Refused as exc:
                outcomes.append(str(exc))

    a.fault = interleave
    result = a.recover(permit_a)
    assert len(outcomes) == 1 and "coordinator is held" in outcomes[0]
    assert "recovery_admitted" not in b_stages                 # B never reached the owner check
    assert [r["batch_id"] for r in _recovery_events(a)] == [permit_a.batch_id]
    assert not sandbox.path(f"recoveries/{permit_b.batch_id}.json").exists()
    assert len(result["tables"]) == 232 and len(sandbox.json("ledger.json")) == 428 + 231
    assert not sandbox.path("recovery.admission").exists() and not sandbox.path("attempt.lock").exists()
    # G: a second recovery after the successful claim is refused.
    with pytest.raises(Refused, match="wrong recovery authority"):
        b.recover(permit(sandbox, frozen, "recovery", recovery_from=state["batch_id"]))


def test_D_N3G_recovery_while_a_recovered_owner_is_live_is_refused(fixture, frozen):
    sandbox, a = fixture
    state = _failed_after_one_unit(sandbox, a, frozen)
    b = Executor(sandbox, frozen)
    outcomes = []

    def contend(stage, unit):
        if stage == "before_read" and not outcomes:
            try:
                b.recover(permit(sandbox, frozen, "recovery", recovery_from=sandbox.json("state.json")["batch_id"]))
                outcomes.append("B accepted")
            except Refused as exc:
                outcomes.append(str(exc))
            assert not sandbox.path("recovery.admission").exists()   # loser releases its admission
            assert sandbox.json("attempt.lock")["executor"] == a._identity

    a.fault = contend
    a.recover(permit(sandbox, frozen, "recovery", recovery_from=state["batch_id"]))
    assert outcomes and "live/concurrent" in outcomes[0]
    assert len(_recovery_events(a)) == 1


def test_D_N3C_recovery_while_the_original_owner_is_live_is_refused(fixture, frozen):
    sandbox, executor = fixture
    state = _failed_after_one_unit(sandbox, executor, frozen)
    atomic_json(sandbox.path("attempt.lock"), {"scope": SYNTHETIC, "pid": os.getpid(),
                "process_created": psutil.Process().create_time(), "state": "ACTIVE"}, replace=True)
    with pytest.raises(Refused, match="live/concurrent"):
        executor.recover(permit(sandbox, frozen, "recovery", recovery_from=state["batch_id"]))
    assert not sandbox.path("recovery.admission").exists() and not _recovery_events(executor)
    assert sandbox.json("attempt.lock")["pid"] == os.getpid()       # the live owner's lock is untouched


@pytest.mark.parametrize("stage", ["recovery_admitted", "after_lock_takeover"])
def test_D_N3EF_recovery_interrupted_before_or_after_takeover_fails_closed_and_retries_uniquely(fixture, frozen, stage):
    sandbox, executor = fixture
    state = _failed_after_one_unit(sandbox, executor, frozen)
    ledger_before = len(sandbox.json("ledger.json"))

    def fail(where, unit):
        if where == stage:
            raise OSError("SYNTHETIC interruption at " + stage)

    executor.fault = fail
    interrupted = permit(sandbox, frozen, "recovery", recovery_from=state["batch_id"])
    with pytest.raises(OSError):
        executor.recover(interrupted)
    executor.fault = None
    assert not sandbox.path("recovery.admission").exists()
    assert len(sandbox.json("ledger.json")) == ledger_before        # no exposure without a takeover + reservation
    assert sandbox.json("state.json")["batch_id"] == state["batch_id"]
    lock = sandbox.json("attempt.lock")
    if stage == "after_lock_takeover":
        assert lock["executor"] == executor._identity and lock["state"] == "FAILED_HANDLER_RAN"
        assert [r["batch_id"] for r in _recovery_events(executor)] == [interrupted.batch_id]
    else:
        assert not _recovery_events(executor)
    with pytest.raises(Refused):
        executor.score_all(permit(sandbox, frozen))                  # a failed recovery never enables scoring
    with pytest.raises(Refused):
        executor.analyse_once(permit(sandbox, frozen, "analysis"))   # ... or analysis
    retry = permit(sandbox, frozen, "recovery", recovery_from=state["batch_id"])
    result = executor.recover(retry)
    assert retry.batch_id != interrupted.batch_id
    assert len(result["tables"]) == 232 and len(sandbox.json("ledger.json")) == ledger_before + 231
    ids = [r["experiment_id"] for r in sandbox.json("ledger.json")]
    assert len(ids) == len(set(ids))


def test_D_N3E_hard_exit_inside_admission_fails_closed_until_audited_resolution(fixture, frozen):
    sandbox, executor = fixture
    state = _failed_after_one_unit(sandbox, executor, frozen)
    live = {"scope": SYNTHETIC, "pid": os.getpid(), "process_created": psutil.Process().create_time(),
            "state": "ACTIVE", "batch_id": "SYNTHETIC-crashed-recovery"}
    sandbox.path("recovery.admission").write_text(json.dumps(live), encoding="utf-8")
    with pytest.raises(Refused, match="recovery admission is present"):
        executor.recover(permit(sandbox, frozen, "recovery", recovery_from=state["batch_id"]))
    assert executor.diagnose()["classification"] == "INTERRUPTED_RECOVERY_ADMISSION"
    with pytest.raises(Refused, match="live process holds recovery admission"):
        executor.resolve_interrupted_writes(permit(sandbox, frozen, "incident"))
    sandbox.path("recovery.admission").write_text(json.dumps({**live, "pid": 2147483647, "process_created": 0}),
                                                  encoding="utf-8")
    record = executor.resolve_interrupted_writes(permit(sandbox, frozen, "incident"))
    assert [i["kind"] for i in record["items"]] == ["stale_recovery_admission"]
    result = executor.recover(permit(sandbox, frozen, "recovery", recovery_from=state["batch_id"]))
    assert len(result["tables"]) == 232


def test_D_N3B_two_processes_racing_for_recovery_admit_exactly_one(fixture, frozen):
    import multiprocessing as mp
    from tests import _recovery_race_worker
    sandbox, executor = fixture
    _failed_after_one_unit(sandbox, executor, frozen)
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2)            # inherited by both children at spawn
    queue = ctx.Queue()
    common = {"repo_root": str(Path(".").resolve()), "root": str(sandbox.root), "hold_s": 4}
    procs = [ctx.Process(target=_recovery_race_worker.main, args=({**common, "name": n}, barrier, queue))
             for n in ("one", "two")]
    for p in procs:
        p.start()
    try:
        results = [queue.get(timeout=420) for _ in procs]
    finally:
        for p in procs:
            p.join(60)
            if p.is_alive():
                p.terminate()
    verdicts = sorted(r["result"] for r in results)
    assert verdicts == ["accepted", "refused"], results
    loser = next(r for r in results if r["result"] == "refused")
    assert "coordinator is held" in loser["message"] or "admission is present" in loser["message"]         or "live/concurrent" in loser["message"]         or "wrong recovery authority" in loser["message"], loser
    assert len(_recovery_events(executor)) == 1
    assert len(sandbox.json("ledger.json")) == 428 + 231
    assert sandbox.json("state.json")["status"] == "SCORED"
    assert not sandbox.path("recovery.admission").exists() and not sandbox.path("attempt.lock").exists()


# =====================================================================================
# Phase 5.5E: stale-admission cleanup race (delayed cleaner acting on a reused pathname)
# =====================================================================================
def _dead_admission(sandbox, admission_id="SYNTHETIC-old-dead-admission"):
    record = {"scope": SYNTHETIC, "pid": 2147483647, "process_created": 0, "state": "ACTIVE",
              "executor": "0" * 32, "batch_id": "SYNTHETIC-crashed-recovery", "admission_id": admission_id}
    sandbox.path("recovery.admission").write_text(json.dumps(record), encoding="utf-8")
    return file_sha256(sandbox.path("recovery.admission"))


def _quarantined_admissions(sandbox):
    return sorted(p for p in sandbox.path("incidents").rglob("recovery.admission")) if sandbox.path("incidents").exists() else []


def test_E_reviewer_delayed_cleaner_counterexample_is_rejected(fixture, frozen):
    """A validates old admission -> B tries -> A quarantines -> R admits -> B resumes -> S tries."""
    sandbox, a = fixture
    state = _failed_after_one_unit(sandbox, a, frozen)
    old_hash = _dead_admission(sandbox)
    b, r, s = (Executor(sandbox, frozen) for _ in range(3))
    log = []

    def try_(label, call):
        try:
            call()
            log.append((label, "ACCEPTED"))
        except Refused as exc:
            log.append((label, str(exc)))

    def a_hook(stage, unit):
        if stage == "incident_recorded":   # A has validated the dead admission and holds the coordinator
            try_("B validates", lambda: b.resolve_interrupted_writes(permit(sandbox, frozen, "incident")))

    a.fault = a_hook
    a.resolve_interrupted_writes(permit(sandbox, frozen, "incident"))
    a.fault = None
    permit_r = permit(sandbox, frozen, "recovery", recovery_from=state["batch_id"])

    def r_hook(stage, unit):
        if stage == "recovery_admitted":   # R holds a NEW admission at the reused pathname
            current = sandbox.json("recovery.admission")
            log.append(("R admission", current["batch_id"] == permit_r.batch_id and current["pid"] == os.getpid()))
            try_("B resumes", lambda: b.resolve_interrupted_writes(permit(sandbox, frozen, "incident")))
            try_("S acquires", lambda: s.recover(permit(sandbox, frozen, "recovery", recovery_from=state["batch_id"])))
            assert sandbox.json("recovery.admission")["admission_id"] == current["admission_id"]   # R keeps it

    r.fault = r_hook
    result = r.recover(permit_r)
    try_("B after R finished", lambda: b.resolve_interrupted_writes(permit(sandbox, frozen, "incident")))
    labels = dict(log)
    assert "coordinator is held" in labels["B validates"]
    assert labels["R admission"] is True
    assert "coordinator is held" in labels["B resumes"] and "coordinator is held" in labels["S acquires"]
    assert "no interrupted write" in labels["B after R finished"]
    quarantined = _quarantined_admissions(sandbox)
    assert len(quarantined) == 1 and file_sha256(quarantined[0]) == old_hash    # only the dead instance
    assert len(_recovery_events(a)) == 1 and len(result["tables"]) == 232
    assert len(sandbox.json("ledger.json")) == 428 + 231
    events = [r_["state"] for r_ in a._audit_records()]
    assert events.count("INTERRUPTED_WRITES_QUARANTINED") == 1


def test_E_B_cleanup_cannot_race_a_fresh_admission(fixture, frozen):
    sandbox, cleaner = fixture
    state = _failed_after_one_unit(sandbox, cleaner, frozen)
    _dead_admission(sandbox)
    recoverer = Executor(sandbox, frozen)
    seen = []

    def hold(stage, unit):
        if stage == "incident_recorded":
            try:
                recoverer.recover(permit(sandbox, frozen, "recovery", recovery_from=state["batch_id"]))
                seen.append("recovery admitted during cleanup")
            except Refused as exc:
                seen.append(str(exc))

    cleaner.fault = hold
    cleaner.resolve_interrupted_writes(permit(sandbox, frozen, "incident"))
    assert len(seen) == 1 and "coordinator is held" in seen[0]
    assert len(recoverer.recover(permit(sandbox, frozen, "recovery", recovery_from=state["batch_id"]))["tables"]) == 232


def test_E_C_live_admission_owner_is_never_displaced(fixture, frozen):
    sandbox, executor = fixture
    _failed_after_one_unit(sandbox, executor, frozen)
    live = {"scope": SYNTHETIC, "pid": os.getpid(), "process_created": psutil.Process().create_time(),
            "state": "ACTIVE", "batch_id": "SYNTHETIC-live", "admission_id": "SYNTHETIC-live-admission"}
    sandbox.path("recovery.admission").write_text(json.dumps(live), encoding="utf-8")
    before = file_sha256(sandbox.path("recovery.admission"))
    with pytest.raises(Refused, match="live process holds recovery admission"):
        executor.resolve_interrupted_writes(permit(sandbox, frozen, "incident"))
    assert file_sha256(sandbox.path("recovery.admission")) == before and not _quarantined_admissions(sandbox)


def test_E_G_exact_instance_checks_protect_a_replacement_admission(fixture, frozen):
    sandbox, executor = fixture
    state = _failed_after_one_unit(sandbox, executor, frozen)
    # Cleaner: the admission is replaced between validation and move -> nothing moved.
    _dead_admission(sandbox)
    replacement = {"scope": SYNTHETIC, "pid": os.getpid(), "process_created": psutil.Process().create_time(),
                   "state": "ACTIVE", "batch_id": "SYNTHETIC-new", "admission_id": "SYNTHETIC-new-admission"}

    def swap(stage, unit):
        if stage == "incident_recorded":
            sandbox.path("recovery.admission").write_text(json.dumps(replacement), encoding="utf-8")

    executor.fault = swap
    with pytest.raises(Refused, match="changed after validation"):
        executor.resolve_interrupted_writes(permit(sandbox, frozen, "incident"))
    executor.fault = None
    assert sandbox.json("recovery.admission") == replacement and not _quarantined_admissions(sandbox)
    # Release: a recovery whose admission was replaced never deletes the replacement.
    sandbox.path("recovery.admission").unlink()
    foreign = {**replacement, "admission_id": "SYNTHETIC-foreign-admission"}

    def foreign_swap(stage, unit):
        if stage == "recovery_admitted":
            sandbox.path("recovery.admission").write_text(json.dumps(foreign), encoding="utf-8")

    executor.fault = foreign_swap
    ledger_before = len(sandbox.json("ledger.json"))
    with pytest.raises(Refused, match="not ours"):
        executor.recover(permit(sandbox, frozen, "recovery", recovery_from=state["batch_id"]))
    assert sandbox.json("recovery.admission") == foreign                          # left untouched
    assert sandbox.json("attempt.lock")["state"] == "FAILED_HANDLER_RAN"          # no silent owner
    assert len(sandbox.json("ledger.json")) == ledger_before                      # nothing exposed
    with pytest.raises(Refused):
        executor.score_all(permit(sandbox, frozen))


def _spawn(sandbox, specs, timeout=420):
    import multiprocessing as mp
    from tests import _recovery_race_worker
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(len(specs))
    queue = ctx.Queue()
    common = {"repo_root": str(Path(".").resolve()), "root": str(sandbox.root)}
    procs = [ctx.Process(target=_recovery_race_worker.main, args=({**common, **spec}, barrier, queue)) for spec in specs]
    for p in procs:
        p.start()
    results = []
    try:
        expected = sum(1 for spec in specs if "die_stage" not in spec)
        results = [queue.get(timeout=timeout) for _ in range(expected)]
    finally:
        for p in procs:
            p.join(60)
            if p.is_alive():
                p.terminate()
    return results, [p.exitcode for p in procs]


def test_E_A_two_cleaner_processes_quarantine_a_dead_admission_exactly_once(fixture, frozen):
    sandbox, executor = fixture
    state = _failed_after_one_unit(sandbox, executor, frozen)
    old_hash = _dead_admission(sandbox)
    results, codes = _spawn(sandbox, [{"name": n, "mode": "resolve", "hold_stage": "incident_recorded", "hold_s": 3}
                                       for n in ("one", "two")])
    assert sorted(r["result"] for r in results) == ["accepted", "refused"], results
    loser = next(r for r in results if r["result"] == "refused")
    assert "coordinator is held" in loser["message"] or "no interrupted write" in loser["message"], loser
    quarantined = _quarantined_admissions(sandbox)
    assert len(quarantined) == 1 and file_sha256(quarantined[0]) == old_hash
    assert [r["state"] for r in executor._audit_records()].count("INTERRUPTED_WRITES_QUARANTINED") == 1
    assert len(executor.recover(permit(sandbox, frozen, "recovery", recovery_from=state["batch_id"]))["tables"]) == 232


def test_E_cleaner_process_and_recovery_process_never_both_act_on_the_admission(fixture, frozen):
    sandbox, executor = fixture
    state = _failed_after_one_unit(sandbox, executor, frozen)
    _dead_admission(sandbox)
    results, _ = _spawn(sandbox, [{"name": "cleaner", "mode": "resolve", "hold_stage": "incident_recorded", "hold_s": 3},
                                  {"name": "recoverer", "mode": "recover"}])
    by = {r["name"]: r for r in results}
    # Either may reach the coordinator first. The recoverer can never act on the stale
    # admission (it is present, or cleanup holds the coordinator); the cleaner either
    # quarantines it or was refused only because the recoverer briefly held the coordinator.
    assert by["recoverer"]["result"] == "refused" and ("coordinator is held" in by["recoverer"]["message"]
                                                      or "admission is present" in by["recoverer"]["message"]), results
    assert by["cleaner"]["result"] == "accepted" or "coordinator is held" in by["cleaner"]["message"], results
    assert not _recovery_events(executor)                       # never both acted; no recovery admitted
    if by["cleaner"]["result"] != "accepted":
        assert sandbox.path("recovery.admission").exists()       # untouched, still fails closed
        executor.resolve_interrupted_writes(permit(sandbox, frozen, "incident"))
    assert len(_quarantined_admissions(sandbox)) == 1 and not sandbox.path("recovery.admission").exists()
    assert len(executor.recover(permit(sandbox, frozen, "recovery", recovery_from=state["batch_id"]))["tables"]) == 232


def test_E_E_H_crash_during_stale_admission_resolution_fails_closed_then_resolves(fixture, frozen):
    sandbox, executor = fixture
    state = _failed_after_one_unit(sandbox, executor, frozen)
    old_hash = _dead_admission(sandbox)
    results, codes = _spawn(sandbox, [{"name": "dies", "mode": "resolve", "die_stage": "incident_recorded"}])
    assert codes == [3] and results == []
    # The OS released the coordinator; the admission was recorded but never moved.
    assert file_sha256(sandbox.path("recovery.admission")) == old_hash and not _quarantined_admissions(sandbox)
    assert len(list(sandbox.path("incidents").glob("*/incident.json"))) == 1          # evidence kept
    assert executor.diagnose()["classification"] == "INTERRUPTED_RECOVERY_ADMISSION"
    with pytest.raises(Refused, match="admission is present"):
        executor.recover(permit(sandbox, frozen, "recovery", recovery_from=state["batch_id"]))
    executor.resolve_interrupted_writes(permit(sandbox, frozen, "incident"))
    assert len(list(sandbox.path("incidents").glob("*/incident.json"))) == 2
    assert len(_quarantined_admissions(sandbox)) == 1
    result = executor.recover(permit(sandbox, frozen, "recovery", recovery_from=state["batch_id"]))
    assert len(result["tables"]) == 232 and len(sandbox.json("ledger.json")) == 428 + 231


def test_E_F_admission_owner_exits_unexpectedly(fixture, frozen):
    sandbox, executor = fixture
    state = _failed_after_one_unit(sandbox, executor, frozen)
    kept, kept_entry = next(iter(state["tables"].items()))
    results, codes = _spawn(sandbox, [{"name": "dies", "mode": "recover", "die_stage": "recovery_admitted"}])
    assert codes == [3]
    left = sandbox.json("recovery.admission")
    assert executor._owner_dead(left)                    # a genuinely dead process, not a forged PID
    assert len(sandbox.json("ledger.json")) == 428       # it died before any takeover or exposure
    with pytest.raises(Refused, match="admission is present"):
        executor.recover(permit(sandbox, frozen, "recovery", recovery_from=state["batch_id"]))
    record = executor.resolve_interrupted_writes(permit(sandbox, frozen, "incident"))
    assert [i["kind"] for i in record["items"]] == ["stale_recovery_admission"]
    result = executor.recover(permit(sandbox, frozen, "recovery", recovery_from=state["batch_id"]))
    assert result["tables"][kept] == kept_entry          # J: completed result never overwritten
    assert sum(e["attempt_id"].endswith("-r1") for e in result["tables"].values()) == 231   # I: unique identities
    ids = [r["experiment_id"] for r in sandbox.json("ledger.json")]
    assert len(ids) == len(set(ids)) == 428 + 231
