"""Subprocess worker for the Phase 3.4F.2 protected-score claim race test.

Runs in a *separate* OS process (multiprocessing ``spawn``), so nothing here
shares interpreter state with the test or with the other worker. The worker
binds a fake inference, redirects the results root and ledger to the test's
temporary directory, waits on a cross-process barrier placed immediately before
the atomic claim attempt, then calls the real ``score_experiment``. The real
protected test cohort is never opened.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys


def main(args: dict, barrier, queue) -> None:
    """Process entry point: run the worker and report its outcome on ``queue``."""
    queue.put(run(args, barrier))


def run(args: dict, barrier) -> dict:
    sys.path.insert(0, args["repo_root"])
    import numpy as np
    import pandas as pd

    from src.experiment import focal_toggle_execution as fx

    # Child processes are outside the conftest guard: refuse anything but an isolated root.
    real_reports = (Path(args["repo_root"]) / "reports").resolve()
    for key in ("results_root", "ledger"):
        target = Path(args[key]).resolve()
        if target == real_reports or real_reports in target.parents:
            raise RuntimeError(f"race worker refuses real protected path for {key}: {target}")
    fx.RESULTS_ROOT = Path(args["results_root"])
    fx.LEDGER = Path(args["ledger"])
    fx.git_head = lambda: "evaluator-sha-for-race-test"

    forward_log = Path(args["forward_log"])
    exp = next(e for e in fx.planned_experiments() if e.experiment_id == args["experiment_id"])

    def fake_inference(record, device):
        # each real forward is recorded durably so the parent can count them
        with forward_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"worker": args["name"], "claim_on_disk":
                                 fx.score_claim_path(exp).exists()}) + "\n")
        rng = np.random.default_rng(0)
        n = 464
        frame = pd.DataFrame({
            "segment_id": [f"20240101_00_00_00_dataLog_{i}" for i in range(n)],
            "split_label": ["Normal_Source_Test" if i % 2 == 0 else "Anomaly_Source_Test"
                            for i in range(n)],
            "anomaly_label": ["normal" if i % 2 == 0 else "anomaly" for i in range(n)],
            "domain_shift_op": "weight00", "domain_shift_env": "BckgA"})
        for col in fx.SCORE_COLUMNS:
            frame[col] = rng.random(n)
        return frame

    fx._run_protected_inference = fake_inference
    # both workers pass every pre-check, then meet here, then race the O_EXCL create
    fx._pre_claim_hook = lambda: barrier.wait(timeout=60)

    try:
        out = fx.score_experiment(exp)
        return {"worker": args["name"], "outcome": "SCORED",
                "history": out["exposure_history"]}
    except SystemExit as exc:
        return {"worker": args["name"], "outcome": "REFUSED", "message": str(exc)}
