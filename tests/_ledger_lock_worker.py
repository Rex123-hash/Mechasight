"""Separate-process writer for the Phase 5.6C-r4 cross-process ledger-lock tests (B4).

Runs in its own OS process (multiprocessing ``spawn``) and shares no interpreter state with the
test or the other writers. It opens a SYNTHETIC sandbox (``Sandbox`` refuses anything outside a
dedicated OS temp directory), builds the real ``Executor`` and its real ``JsonlLedger`` backend,
waits on a cross-process barrier and then calls the real ``JsonlLedger.append`` for its own batch,
so every writer contends for the same ``<ledger>.lock``.

``hold_s`` pauses the writer inside the locked critical section (after its batch authorization is
published, before its rows are appended), so the other writer lands in the contended window.
``die_after_authorization`` ends the process with ``os._exit`` at that point: the OS, not Python,
releases the lock, and no in-process clean-up runs. ``die_at`` does the same at any other hook
stage, e.g. around the atomic sealed replace (``before_ledger_swap``, ``after_ledger_swap``).

``mode="legacy"`` (5.6C-r5) is a writer that never takes the lock: for ``duration_s`` seconds it
calls the frozen public helpers ``TestLookLedger.append`` and ``append_many_atomic`` on the same
ledger as fast as it can and reports what the operating system and the helpers did. No protected
data exists in the sandbox and the real ledger is never touched.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time


def main(args: dict, barrier, queue) -> None:
    try:
        queue.put(run(args, barrier))
    except Exception as exc:  # noqa: BLE001 - any unexpected error is reported, never swallowed
        queue.put({"name": args["name"], "result": "error", "message": f"{type(exc).__name__}: {exc}"})


class _Permit:
    def __init__(self, batch_id):
        self.batch_id = batch_id


def legacy(args: dict, barrier) -> dict:
    """A writer outside the lock: the frozen helpers, hammering the ledger for ``duration_s``."""
    sys.path.insert(0, args["repo_root"])
    from datetime import datetime, timezone
    from src.experiment.ledger import TestLookLedger, TestLookRecord

    path = Path(args["ledger"]).resolve()
    real = (Path(args["repo_root"]) / "reports").resolve()
    if real in path.parents:
        raise RuntimeError(f"legacy worker refuses real protected path: {path}")
    outcomes, n = {}, 0
    barrier.wait(timeout=300)
    deadline = time.time() + args.get("duration_s", 5.0)
    while time.time() < deadline:
        n += 1
        record = TestLookRecord(timestamp_utc=datetime.now(timezone.utc).isoformat(), experiment_id=f"LEGACY-{os.getpid()}-{n}",
                                git_sha="0" * 40, machine="RoboticArm", protocol="legacy", regime="S",
                                model_config_identity="legacy", test_cohort_hash="legacy", reason="SYNTHETIC legacy writer",
                                metrics_requested=("auroc",))
        ledger = TestLookLedger(path)
        for name, call in (("append", lambda: ledger.append(record)),
                           ("append_many_atomic", lambda: ledger.append_many_atomic(
                               [record], expected_existing_count=len(ledger.read())))):
            try:
                call()
                key = f"{name}:WROTE"
            except PermissionError:
                key = f"{name}:PermissionError"
            except (OSError, RuntimeError, ValueError) as exc:
                key = f"{name}:{type(exc).__name__}"
            outcomes[key] = outcomes.get(key, 0) + 1
    return {"name": args["name"], "result": "legacy", "outcomes": outcomes, "attempts": n}


def run(args: dict, barrier) -> dict:
    if args.get("mode") == "legacy":
        return legacy(args, barrier)
    sys.path.insert(0, args["repo_root"])
    from src.experiment import phase5_ledger as ledger_module
    from src.experiment.phase5_execution import Executor, FrozenPackage
    from src.experiment.phase5_inputs import Refused, Sandbox

    real_reports = (Path(args["repo_root"]) / "reports").resolve()
    root = Path(args["root"]).resolve()
    if root == real_reports or real_reports in root.parents:
        raise RuntimeError(f"ledger lock worker refuses real protected path: {root}")
    if args.get("lock_timeout") is not None:
        ledger_module.LEDGER_LOCK_TIMEOUT_SECONDS = float(args["lock_timeout"])
    executor = Executor(Sandbox(root), FrozenPackage.load())

    def fault(stage, unit=""):
        if stage == args.get("die_at"):
            os._exit(3)
        if stage == "after_batch_authorization":
            if args.get("die_after_authorization"):
                os._exit(3)
            time.sleep(args.get("hold_s", 0))

    executor.fault = fault
    attempt_ids = args["attempt_ids"]
    barrier.wait(timeout=300)
    started = time.time()
    try:
        rows = executor.ledger.append(_Permit(args["batch_id"]), sorted(attempt_ids), attempt_ids, args["start"])
        return {"name": args["name"], "result": "appended", "rows": len(rows), "started": started, "ended": time.time()}
    except Refused as exc:
        return {"name": args["name"], "result": "refused", "message": str(exc), "started": started, "ended": time.time()}
