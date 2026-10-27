"""Separate-process worker for the Phase 5.5D/5.5E recovery-admission race tests.

Runs in its own OS process (multiprocessing ``spawn``), so it shares no
interpreter state with the test or the other worker. It loads the frozen
package, opens the given SYNTHETIC sandbox (``Sandbox`` itself refuses anything
outside a dedicated OS temp directory), waits on a cross-process barrier placed
immediately before the action, and reports the outcome.

``mode`` is ``recover`` or ``resolve`` (incident cleanup). ``hold_stage`` /
``hold_s`` pause the worker inside its critical section so the other worker's
attempt lands in the contended window (test-only widening, not synchronization).
``die_stage`` ends the process abruptly with ``os._exit`` at that stage, so the
OS — not Python — releases whatever the worker held. No protected data exists in
the sandbox and the real ledger is never touched.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys


def main(args: dict, barrier, queue) -> None:
    """Process entry point: run the worker and report its outcome on ``queue``."""
    try:
        queue.put(run(args, barrier))
    except Exception as exc:  # noqa: BLE001 - any unexpected error is reported, never swallowed
        queue.put({"name": args["name"], "result": "error", "message": f"{type(exc).__name__}: {exc}"})


def run(args: dict, barrier) -> dict:
    sys.path.insert(0, args["repo_root"])
    import time

    from src.experiment.phase5_execution import Executor, FrozenPackage, authorize_fixture
    from src.experiment.phase5_inputs import Refused, Sandbox

    real_reports = (Path(args["repo_root"]) / "reports").resolve()
    root = Path(args["root"]).resolve()
    if root == real_reports or real_reports in root.parents:
        raise RuntimeError(f"recovery race worker refuses real protected path: {root}")
    package = FrozenPackage.load()
    box = Sandbox(root)
    executor = Executor(box, package)

    def fault(stage, unit):
        if stage == args.get("die_stage"):
            os._exit(3)
        if stage == args.get("hold_stage", "recovery_admitted"):
            time.sleep(args.get("hold_s", 0))

    executor.fault = fault
    mode = args.get("mode", "recover")
    if mode == "recover":
        grant = authorize_fixture(box, package, "recovery", recovery_from=box.json("state.json")["batch_id"])
        action = lambda: executor.recover(grant)  # noqa: E731
    else:
        grant = authorize_fixture(box, package, "incident")
        action = lambda: executor.resolve_interrupted_writes(grant)  # noqa: E731
    barrier.wait(timeout=300)
    try:
        action()
        return {"name": args["name"], "result": "accepted", "mode": mode}
    except Refused as exc:
        return {"name": args["name"], "result": "refused", "mode": mode, "message": str(exc)}
