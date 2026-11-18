"""Phase 5.6C-r4 (B4): one stable cross-process lock around every ledger check and append.

Every writer of a ledger file takes ``ledger_lock(<ledger>)``, an OS byte-range lock on
``<ledger>.lock``. Under it the writer re-reads the ledger, re-checks the look budget on the fresh
rows, re-verifies history and batches, checks its start count, publishes its batch authorization
and appends. The mandatory counterexample is 428 -> 660 -> 892: two writers that both prechecked at
428 can never jointly reach 892; the second excessive operation fails before any mutation.

Cross-process writers run as separate ``spawn`` processes (``tests/_ledger_lock_worker.py``) over a
SYNTHETIC temporary copy of the closed ledger. Nothing here touches the real ledger.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

import pytest

from src.experiment import phase5_ledger as ledger_module
from src.experiment.ledger_history import verify_ledger_history
from src.experiment.phase5_execution import Executor, FrozenPackage, authorize_fixture, source_identity
from src.experiment.phase5_fixture import create_fixture
from src.experiment.phase5_inputs import SYNTHETIC, Refused, Sandbox, atomic_json
from src.experiment.phase5_ledger import (
    PENDING_TAG, ledger_lock, ledger_lock_held, ledger_lock_path, ledger_sealed, seal_ledger, seed_jsonl_ledger,
)
from src.experiment.provenance import file_sha256

REAL_LEDGER = Path("reports/experiment_results/test_look_ledger.jsonl")


@pytest.fixture(autouse=True)
def _ledger_primitives_never_touch_the_real_ledger(monkeypatch):
    """Harness guard: the frozen conftest guards TestLookLedger; this guards the replace and seal primitives."""
    from src.experiment import phase5_ledger as _ledger_module
    real_ledger = Path("reports/experiment_results/test_look_ledger.jsonl").resolve()
    real_replace, real_seal = _ledger_module._replace_sealed, _ledger_module.seal_ledger

    def guarded_replace(source, target):
        assert Path(target).resolve() != real_ledger, "TEST HARNESS GUARD: refusing to replace the real ledger"
        return real_replace(source, target)

    def guarded_seal(path):
        assert Path(path).resolve() != real_ledger, "TEST HARNESS GUARD: refusing to seal the real ledger"
        return real_seal(path)
    monkeypatch.setattr(_ledger_module, "_replace_sealed", guarded_replace)
    monkeypatch.setattr(_ledger_module, "seal_ledger", guarded_seal)

@pytest.fixture(scope="module")
def frozen():
    return FrozenPackage.load()


def _crash_at(stage):
    def fail(where, unit):
        if where == stage:
            raise OSError("SYNTHETIC crash at " + stage)
    return fail


@pytest.fixture
def at_428(tmp_path, frozen):
    """A SEALED SYNTHETIC JSONL ledger copy at 428: the 232-look batch exposed, then the attempt crashed."""
    box = Sandbox(tmp_path / "SYNTHETIC-ledger-lock")
    create_fixture(box, frozen)
    fixture = box.json("fixture.json")
    fixture["ledger"] = {**seed_jsonl_ledger(box, sealed=True), "code_git_sha": "0" * 40}
    atomic_json(box.path("fixture.json"), fixture, replace=True)
    executor = Executor(box, frozen)
    executor.fault = _crash_at("after_reservation")
    with pytest.raises(OSError):
        executor.score_all(authorize_fixture(box, frozen, "score"))
    executor.fault = None
    assert _rows(box) == 428 and ledger_sealed(_ledger(box))
    return box, executor


def _ledger(box):
    return box.path("ledger/test_look_ledger.jsonl")


def _rows(box):
    return len(verify_ledger_history(_ledger(box), authorization_root=box.path("ledger/authorizations")).rows)


def _reserve(box, frozen, executor, batch_id, suffix, start=428):
    """The durable reservation an executor writes before its append (here for an independent writer)."""
    attempt_ids = {eid: f"{eid}-{suffix}" for eid in sorted(executor.operations)}
    atomic_json(box.path(f"reservations/{batch_id}.json"), {
        "scope": SYNTHETIC, "batch_id": batch_id, "package": frozen.identity, "fixture": file_sha256(box.path("fixture.json")),
        "attempt_ids": attempt_ids, "recovery_from": None, "code": source_identity(), "ledger_start": start,
        "ledger_end": start + len(attempt_ids)})
    return attempt_ids


def _spawn(box, jobs, timeout=420):
    from tests import _ledger_lock_worker
    ctx = mp.get_context("spawn")
    barrier, queue = ctx.Barrier(len(jobs)), ctx.Queue()
    common = {"repo_root": str(Path(".").resolve()), "root": str(box.root), "start": 428}
    procs = [ctx.Process(target=_ledger_lock_worker.main, args=({**common, **job}, barrier, queue)) for job in jobs]
    for p in procs:
        p.start()
    results = []
    try:
        expected = sum(1 for job in jobs if not job.get("die_after_authorization") and not job.get("die_at"))
        results = [queue.get(timeout=timeout) for _ in range(expected)]
    finally:
        for p in procs:
            p.join(60)
            if p.is_alive():
                p.terminate()
    return {r["name"]: r for r in results}, [p.exitcode for p in procs]


class _Permit:
    def __init__(self, batch_id):
        self.batch_id = batch_id


# ------------------------------------------------------------------ cross-process: 428 -> 660, never 892
@pytest.mark.parametrize("hold_s", [0.0, 2.0])
def test_two_processes_prechecked_at_428_cannot_reach_892(at_428, frozen, hold_s):
    box, executor = at_428
    jobs = [{"name": name, "batch_id": f"SYNTHETIC-lock-{name}", "hold_s": hold_s,
             "attempt_ids": _reserve(box, frozen, executor, f"SYNTHETIC-lock-{name}", suffix)}
            for name, suffix in (("A", "r1"), ("B", "r2"))]
    # Both writers' own prechecks pass at 428 (the stale state each would act on).
    for job in jobs:
        executor.ledger.precheck(executor.ledger.verify(), job["attempt_ids"])
    results, codes = _spawn(box, jobs)
    assert codes == [0, 0] and sorted(r["result"] for r in results.values()) == ["appended", "refused"], results
    winner = next(r for r in results.values() if r["result"] == "appended")
    loser = next(r for r in results.values() if r["result"] == "refused")
    assert winner["rows"] == 660
    assert "look budget exceeded before any mutation" in loser["message"] and "892" not in loser["message"]
    assert _rows(box) == 660                                                  # never 892
    assert (box.path("ledger/authorizations") / f"SYNTHETIC-lock-{winner['name']}" / "batch_authorization.json").is_file()
    assert not (box.path("ledger/authorizations") / f"SYNTHETIC-lock-{loser['name']}").exists()   # no partial authorization
    if hold_s:
        assert loser["ended"] >= winner["started"] + hold_s - 0.5            # the loser waited for the holder's release


def test_a_writer_killed_inside_the_lock_releases_it_and_the_cap_still_holds(at_428, frozen):
    box, executor = at_428
    a = {"name": "A", "batch_id": "SYNTHETIC-lock-A", "die_after_authorization": True,
         "attempt_ids": _reserve(box, frozen, executor, "SYNTHETIC-lock-A", "r1")}
    results, codes = _spawn(box, [a])
    assert results == {} and codes == [3]                                     # died holding the lock, after its authorization
    assert _rows(box) == 428                                                  # the dangling authorization counts nothing
    assert (box.path("ledger/authorizations/SYNTHETIC-lock-A") / "batch_authorization.json").is_file()
    b = {"name": "B", "batch_id": "SYNTHETIC-lock-B", "attempt_ids": _reserve(box, frozen, executor, "SYNTHETIC-lock-B", "r2")}
    results, codes = _spawn(box, [b])
    assert codes == [0] and results["B"]["result"] == "appended" and _rows(box) == 660   # the OS released the dead lock
    c = {"name": "C", "batch_id": "SYNTHETIC-lock-C", "attempt_ids": _reserve(box, frozen, executor, "SYNTHETIC-lock-C", "r3")}
    before = file_sha256(_ledger(box))
    results, _ = _spawn(box, [c])
    assert results["C"]["result"] == "refused" and "look budget exceeded" in results["C"]["message"]
    assert _rows(box) == 660 and file_sha256(_ledger(box)) == before
    with pytest.raises(Refused, match="look budget exceeded|already exists"):   # the dead batch cannot be re-published
        executor.ledger.append(_Permit("SYNTHETIC-lock-A"), sorted(a["attempt_ids"]), a["attempt_ids"], 428)


def test_a_lock_held_by_another_process_refuses_the_writer_before_any_mutation(at_428, frozen):
    box, executor = at_428
    job = {"name": "W", "batch_id": "SYNTHETIC-lock-W", "lock_timeout": 1.0,
           "attempt_ids": _reserve(box, frozen, executor, "SYNTHETIC-lock-W", "r1")}
    before = file_sha256(_ledger(box))
    with ledger_lock(_ledger(box)):                                           # this process holds the OS lock
        results, codes = _spawn(box, [job])
    assert codes == [0] and results["W"]["result"] == "refused" and "ledger lock is held" in results["W"]["message"]
    assert file_sha256(_ledger(box)) == before and not box.path("ledger/authorizations/SYNTHETIC-lock-W").exists()


# ------------------------------------------------------------------ in-process: stale precheck, failures, threads
def test_a_stale_precheck_cannot_exceed_the_cap(at_428, frozen):
    """Recovery A prechecks at 428; before its append another writer takes the ledger to 660."""
    box, executor = at_428
    failed = box.json("state.json")["batch_id"]
    other = _reserve(box, frozen, executor, "SYNTHETIC-competing", "r2")
    seen = []

    def compete(stage, unit):
        if stage == "after_authorization" and not seen:          # A's reservation is written; its precheck is now stale
            seen.append(executor.ledger.append(_Permit("SYNTHETIC-competing"), sorted(other), other, 428))
    executor.fault = compete
    with pytest.raises(Refused, match="look budget exceeded before any mutation"):
        executor.recover(authorize_fixture(box, frozen, "recovery", recovery_from=failed))
    executor.fault = None
    assert len(seen[0]) == 660 and _rows(box) == 660
    state = box.json("state.json")
    assert state["status"] == "REFUSED_PRE_EXPOSURE"
    assert not (box.path("ledger/authorizations") / state["batch_id"]).exists()
    assert not box.path(f"exposures/{state['batch_id']}.json").exists()


def test_a_failed_append_withdraws_its_authorization_and_leaves_recovery_possible(tmp_path, frozen, monkeypatch):
    box = Sandbox(tmp_path / "SYNTHETIC-withdraw")
    create_fixture(box, frozen)
    fixture = box.json("fixture.json")
    fixture["ledger"] = {**seed_jsonl_ledger(box), "code_git_sha": "0" * 40}
    atomic_json(box.path("fixture.json"), fixture, replace=True)
    executor = Executor(box, frozen)
    real = ledger_module._replace_bounded

    def fail(source, target):
        raise OSError("SYNTHETIC failure before the ledger write")
    monkeypatch.setattr(ledger_module, "_replace_bounded", fail)
    with pytest.raises(OSError):
        executor.score_all(authorize_fixture(box, frozen, "score"))
    batch = box.json("state.json")["batch_id"]
    folder = box.path("ledger/authorizations") / batch
    assert not (folder / "batch_authorization.json").exists()
    assert [p.name for p in folder.iterdir()] and all(re.fullmatch(r"batch_authorization\.aborted-[0-9a-f]{12}\.json", p.name)
                                                      for p in folder.iterdir())
    assert _rows(box) == 196 and box.json("state.json")["status"] == "REFUSED_PRE_EXPOSURE"
    monkeypatch.setattr(ledger_module, "_replace_bounded", real)
    assert not _siblings(_ledger(box), PENDING_TAG)                         # the failed pending file was removed
    state = executor.recover(authorize_fixture(box, frozen, "recovery", recovery_from=batch))
    assert state["status"] == "SCORED" and _rows(box) == 428


def test_threads_of_one_process_also_serialize_on_the_os_lock(tmp_path):
    ledger = tmp_path / "SYNTHETIC.jsonl"
    ledger.write_bytes(b"")
    spans = []

    def hold():
        with ledger_lock(ledger, timeout=30):
            start = time.monotonic()
            time.sleep(0.3)
            spans.append((start, time.monotonic()))
    threads = [threading.Thread(target=hold) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    spans.sort()
    assert len(spans) == 3 and all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
    with ledger_lock(ledger):
        assert ledger_lock_held(ledger)
        with ledger_lock(ledger):                       # re-entrant in the holding thread only
            assert ledger_lock_held(ledger)
    assert not ledger_lock_held(ledger) and ledger_lock_path(ledger).is_file()


def test_every_non_frozen_ledger_writer_is_the_locked_phase5_backend():
    """Writers of the test-look ledger: frozen legacy closed-stage code (unchanged since d7ea8ac) or phase5_ledger."""
    pattern = re.compile(r"append_many_atomic\(|TestLookLedger\(")
    writers = sorted(str(p.as_posix()) for root in ("src", "scripts") for p in Path(root).rglob("*.py")
                     if pattern.search(p.read_text(encoding="utf-8")))
    for name in writers:
        if name == "src/experiment/phase5_ledger.py":
            continue
        frozen_blob = subprocess.run(["git", "rev-parse", f"d7ea8ac3022acd755a847b77118375fb719ef6e3:{name}"],
                                     capture_output=True, text=True)
        current = subprocess.run(["git", "hash-object", name], capture_output=True, text=True)
        assert frozen_blob.returncode == 0 and frozen_blob.stdout.strip() == current.stdout.strip(), name
    source = Path("src/experiment/phase5_ledger.py").read_text(encoding="utf-8")
    body = source[source.index("    def append(self, permit"):source.index("    def _write_authorization")]
    assert "with ledger_lock(self.path):" in body and body.index("with ledger_lock") < body.index("_replace_bounded(")
    commit = body[body.index("    def _commit"):]                             # the only byte writer checks first
    assert commit.index("self._locked_checks(permit)") < commit.index("self.precheck(rows, batch)") < commit.index(
        "_replace_bounded(")
    assert not hasattr(ledger_module, "sealed_append") and not hasattr(ledger_module, "settle_interrupted_append")
    assert "append_many_atomic" not in body and "TestLookLedger(" not in source


# ------------------------------------------------------------------ 5.6C-r5: legacy writers and the sealed single writer
def _legacy_record(tag):
    from datetime import datetime, timezone
    from src.experiment.ledger import TestLookRecord
    return TestLookRecord(timestamp_utc=datetime.now(timezone.utc).isoformat(), experiment_id=f"LEGACY-{tag}",
                          git_sha="0" * 40, machine="RoboticArm", protocol="legacy", regime="S",
                          model_config_identity="legacy", test_cohort_hash="legacy", reason="SYNTHETIC legacy writer",
                          metrics_requested=("auroc",))


def _legacy_writers(path):
    """Every way frozen code writes a test-look ledger (gate.py, phase32.py, phase4.py and the Phase 3/4 scripts all
    reduce to the two public TestLookLedger methods), plus plain file writes."""
    from src.experiment.ledger import TestLookLedger
    ledger = TestLookLedger(path)
    return {
        "TestLookLedger.append": lambda: ledger.append(_legacy_record("append")),
        "TestLookLedger.append_many_atomic": lambda: ledger.append_many_atomic(
            [_legacy_record(f"many-{i}") for i in range(233)], expected_existing_count=len(ledger.read())),
        "open(path, 'ab')": lambda: open(path, "ab").close(),
        "os.open(O_APPEND | O_WRONLY)": lambda: os.close(os.open(path, os.O_APPEND | os.O_WRONLY | getattr(os, "O_BINARY", 0))),
        "Path.write_bytes": lambda: Path(path).write_bytes(b""),
    }


def _assert_legacy_writers_refused(path):
    for name, call in _legacy_writers(path).items():
        before = file_sha256(path)
        with pytest.raises(PermissionError):
            call()
        assert file_sha256(path) == before and ledger_sealed(path), name


def _legacy_rows(path):
    return [line for line in Path(path).read_bytes().splitlines() if b"LEGACY-" in line]


def _siblings(path, tag):
    return list(path.parent.glob(path.name + tag + "*"))


def test_direct_legacy_helpers_are_refused_by_the_sealed_ledger_at_428_and_at_660(at_428, frozen):
    box, executor = at_428
    _assert_legacy_writers_refused(_ledger(box))                              # 428 + 233 would be 661 > 660
    assert _rows(box) == 428
    other = _reserve(box, frozen, executor, "SYNTHETIC-to-660", "r1")
    assert len(executor.ledger.append(_Permit("SYNTHETIC-to-660"), sorted(other), other, 428)) == 660
    assert ledger_sealed(_ledger(box))                                        # the single writer keeps the seal
    before = file_sha256(_ledger(box))
    _assert_legacy_writers_refused(_ledger(box))                              # every legacy write past 660: refused
    extra = _reserve(box, frozen, executor, "SYNTHETIC-past-660", "r2", start=660)
    with pytest.raises(Refused, match="look budget exceeded before any mutation"):
        executor.ledger.append(_Permit("SYNTHETIC-past-660"), sorted(extra), extra, 660)
    assert _rows(box) == 660 and file_sha256(_ledger(box)) == before and not _legacy_rows(_ledger(box))
    assert not (box.path("ledger/authorizations") / "SYNTHETIC-past-660").exists()


def test_legacy_writers_are_refused_at_every_instant_of_a_sealed_append(at_428, frozen):
    """Before the atomic replace the path is the sealed old ledger, after it the sealed new one; it is never
    missing or writable, so there is no window in which a legacy writer could create or extend it."""
    box, executor = at_428
    path = _ledger(box)
    seen = []

    def legacy_now(stage, unit=""):
        if stage in {"after_batch_authorization", "before_ledger_swap", "after_ledger_swap"}:
            assert path.is_file() and ledger_sealed(path), stage
            _assert_legacy_writers_refused(path)
            seen.append((stage, len(ledger_module.ledger_lines(path))))
    executor.fault = legacy_now
    ids = _reserve(box, frozen, executor, "SYNTHETIC-instants", "r1")
    assert len(executor.ledger.append(_Permit("SYNTHETIC-instants"), sorted(ids), ids, 428)) == 660
    executor.fault = None
    assert seen == [("after_batch_authorization", 428), ("before_ledger_swap", 428), ("after_ledger_swap", 660)]
    assert ledger_sealed(path) and _rows(box) == 660 and not _legacy_rows(path) and not _siblings(path, PENDING_TAG)


@pytest.mark.parametrize("stage, rows_after", [
    ("before_ledger_swap", 428),     # sealed next ledger written, not yet the ledger
    ("after_ledger_swap", 660),      # atomically replaced, not yet verified by the writer
])
def test_a_writer_killed_around_the_atomic_replace_leaves_a_consistent_ledger(at_428, frozen, stage, rows_after):
    box, executor = at_428
    path = _ledger(box)
    a = {"name": "A", "batch_id": "SYNTHETIC-swap-A", "die_at": stage,
         "attempt_ids": _reserve(box, frozen, executor, "SYNTHETIC-swap-A", "r1")}
    results, codes = _spawn(box, [a])
    assert results == {} and codes == [3]
    assert path.is_file() and ledger_sealed(path)                            # the ledger path always names a sealed file
    assert bool(_siblings(path, PENDING_TAG)) == (stage == "before_ledger_swap")   # a leftover is never the ledger
    assert _rows(box) == rows_after
    if rows_after == 428:
        b = {"name": "B", "batch_id": "SYNTHETIC-swap-B", "attempt_ids": _reserve(box, frozen, executor, "SYNTHETIC-swap-B", "r2")}
        results, codes = _spawn(box, [b])
        assert codes == [0] and results["B"]["result"] == "appended" and _rows(box) == 660
    assert not _siblings(path, PENDING_TAG)                                   # the next writer removed the leftover
    before = file_sha256(path)
    c = {"name": "C", "batch_id": "SYNTHETIC-swap-C", "attempt_ids": _reserve(box, frozen, executor, "SYNTHETIC-swap-C", "r3")}
    results, _ = _spawn(box, [c])
    assert results["C"]["result"] == "refused" and "look budget exceeded" in results["C"]["message"]
    assert _rows(box) == 660 and file_sha256(path) == before and ledger_sealed(path)


@pytest.mark.parametrize("hold_s", [0.0, 2.0])
def test_concurrent_legacy_and_locked_writers_never_merge_a_legacy_row(at_428, frozen, hold_s, record_property):
    """A legacy process hammers the frozen helpers while two locked writers race from 428 (one may reach 660)."""
    box, executor = at_428
    path = _ledger(box)
    jobs = [{"name": name, "batch_id": f"SYNTHETIC-mixed-{name}", "hold_s": hold_s,
             "attempt_ids": _reserve(box, frozen, executor, f"SYNTHETIC-mixed-{name}", suffix)}
            for name, suffix in (("A", "r1"), ("B", "r2"))]
    jobs.append({"name": "L", "mode": "legacy", "ledger": str(path), "duration_s": 4.0 + 2 * hold_s})
    results, codes = _spawn(box, jobs)
    assert codes == [0, 0, 0], results
    legacy = results["L"]["outcomes"]
    assert results["L"]["attempts"] > 0 and sum(legacy.values()) == 2 * results["L"]["attempts"]
    locked = sorted(results[n]["result"] for n in ("A", "B"))
    record_property("legacy_attempts", results["L"]["attempts"])
    record_property("legacy_outcomes", json.dumps(legacy, sort_keys=True))
    record_property("locked_writers", ",".join(locked))
    record_property("final_rows", len(ledger_module.ledger_lines(path)))
    # Every legacy write is refused: by the OS (PermissionError), or by its own count check when the ledger was
    # replaced between its two reads (RuntimeError). Nothing else, and never a write.
    assert set(legacy) <= {"append:PermissionError", "append_many_atomic:PermissionError",
                           "append_many_atomic:RuntimeError"}, legacy
    assert legacy.get("append:PermissionError") == results["L"]["attempts"]
    assert not _legacy_rows(path) and locked == ["appended", "refused"]
    assert _rows(box) == 660 and ledger_sealed(path) and not _siblings(path, PENDING_TAG)
    loser = next(results[n] for n in ("A", "B") if results[n]["result"] == "refused")
    assert "look budget exceeded before any mutation" in loser["message"]


@pytest.mark.skipif(os.name != "nt", reason="Windows sharing semantics")
def test_a_reader_holding_the_ledger_open_delays_the_swap_boundedly_then_refuses(at_428, frozen, monkeypatch):
    box, executor = at_428
    path = _ledger(box)
    monkeypatch.setattr(ledger_module, "SWAP_RETRY_SECONDS", 0.5)
    ids = _reserve(box, frozen, executor, "SYNTHETIC-reader", "r1")
    before = file_sha256(path)
    with open(path, "rb"):                                                    # no delete sharing: the rename must wait
        with pytest.raises(PermissionError):
            executor.ledger.append(_Permit("SYNTHETIC-reader"), sorted(ids), ids, 428)
    assert file_sha256(path) == before and ledger_sealed(path) and _rows(box) == 428 and not _siblings(path, PENDING_TAG)
    assert not (box.path("ledger/authorizations") / "SYNTHETIC-reader" / "batch_authorization.json").exists()
    ids = _reserve(box, frozen, executor, "SYNTHETIC-reader-2", "r2")
    assert len(executor.ledger.append(_Permit("SYNTHETIC-reader-2"), sorted(ids), ids, 428)) == 660


def test_seal_preserves_content_and_the_human_seal_script_refuses_without_confirmation(tmp_path):
    ledger = tmp_path / "SYNTHETIC.jsonl"
    ledger.write_bytes(b"".join(ledger_module.ledger_lines(REAL_LEDGER)))
    before = file_sha256(ledger)
    seal_ledger(ledger)
    assert ledger_sealed(ledger) and file_sha256(ledger) == before
    _assert_legacy_writers_refused(ledger)
    script = subprocess.run([sys.executable, "scripts/seal_phase5_ledger.py", "--confirm", "yes"], capture_output=True, text=True)
    assert script.returncode != 0 and "must be SEAL-THE-REAL-LEDGER" in (script.stderr + script.stdout)
    assert not ledger_sealed(REAL_LEDGER)                                     # never sealed by a test or by this pass
    assert file_sha256(REAL_LEDGER) == "f75584611c18ff387bee8c06f1fcc999c3046def876ba13dea1c744fce64de5f"


# ------------------------------------------------------------------ the independent audit (detection, synthetic copies)
def _audit(box, committed=None):
    from src.experiment.phase5_ledger_audit import audit_ledger
    return audit_ledger(_ledger(box), authorization_root=box.path("ledger/authorizations"), baseline_rows=196,
                        baseline_sha256="f75584611c18ff387bee8c06f1fcc999c3046def876ba13dea1c744fce64de5f",
                        max_total_rows=660, scope_marker="SYNTHETIC", committed_bytes=committed)


def _rewrite(path, data):
    """A same-user rewrite of the sealed file: the seal does not prevent it."""
    os.chmod(path, 0o644)
    Path(path).write_bytes(data)
    seal_ledger(path)


def test_every_batch_authorization_records_the_ledger_digests_before_and_after(at_428, frozen):
    box, executor = at_428
    ids = _reserve(box, frozen, executor, "SYNTHETIC-digests", "r1")
    executor.ledger.append(_Permit("SYNTHETIC-digests"), sorted(ids), ids, 428)
    lines = ledger_module.ledger_lines(_ledger(box))
    for batch, (start, end) in {box.json("state.json")["batch_id"]: (196, 428), "SYNTHETIC-digests": (428, 660)}.items():
        record = json.loads((box.path("ledger/authorizations") / batch / "batch_authorization.json").read_text(encoding="utf-8"))
        assert record["ledger_sha256_before"] == ledger_module.ledger_digest(lines, start)
        assert record["ledger_sha256_after"] == ledger_module.ledger_digest(lines, end)
    report = _audit(box)
    assert report["rows"] == 660 and [(b["start"], b["end"]) for b in report["batches"]] == [(196, 428), (428, 660)]


@pytest.mark.parametrize("tamper, match", [
    ("edit_an_appended_row", "differ from the ledger bytes its authorization recorded"),
    ("remove_a_middle_row", "unauthorized history|differ from the ledger bytes"),
    ("edit_a_historical_row", "closed 196-row history|unauthorized history"),
    ("truncate_the_batch_against_the_committed_copy", "does not extend the committed ledger"),
])
def test_the_audit_and_the_writer_detect_rewritten_ledger_bytes(at_428, frozen, tamper, match):
    box, executor = at_428
    path = _ledger(box)
    lines = ledger_module.ledger_lines(path)
    committed = b"".join(lines)                                               # as if committed after the operation
    if tamper == "edit_an_appended_row":
        data = b"".join(lines[:300] + (lines[300].replace(b'"machine":"', b'"machine":"X'),) + lines[301:])
    elif tamper == "remove_a_middle_row":
        data = b"".join(lines[:300] + lines[301:])
    elif tamper == "edit_a_historical_row":
        data = b"".join(lines[:10] + (lines[10].replace(b'"reason":"', b'"reason":"edited '),) + lines[11:])
    else:
        data = b"".join(lines[:196])
    _rewrite(path, data)
    before = file_sha256(path)
    with pytest.raises(Refused, match=match):
        _audit(box, committed=committed if tamper.startswith("truncate") else None)   # each detector on its own
    if tamper != "truncate_the_batch_against_the_committed_copy":           # the writer's own verification refuses too
        ids = _reserve(box, frozen, executor, "SYNTHETIC-after-tamper", "r1")
        with pytest.raises(Refused):
            executor.ledger.append(_Permit("SYNTHETIC-after-tamper"), sorted(ids), ids, 428)
    assert file_sha256(path) == before
