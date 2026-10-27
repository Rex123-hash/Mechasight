"""Phase 3.4 RoboticArm focal-parent sibling-toggle (V3) — staged execution CLI.

The audited sequence, one stage per step so each can be committed separately:

    plan        verify memberships, hashes and the frozen matrix   (safe any time)
    train       train the NEXT experiment in the frozen order      (step B)
    freeze      summarise the ten training records for COMMIT B    (step C)
    authorize   append all ten test-look entries atomically        (step D)
    score       score ONE authorised, frozen checkpoint             (step E/F)
    analyze     frozen statistics from persisted scores only       (step G)

Every stage refuses to run out of order. ``train`` refuses any experiment that
is not next in the frozen order; ``authorize`` refuses unless all ten
checkpoints are frozen and the ledger holds exactly the 30 historical entries;
``score`` refuses unless a ledger entry names the exact checkpoint hash and NO
exposure record exists - it writes a durable SCORE_STARTED record before the
first protected forward, so a crash at any later point leaves the look consumed
and the experiment INCOMPLETE rather than silently re-scorable; ``analyze``
refuses unless all 4,640 rows exist and hash-verify. Nothing here inspects test
performance before ``analyze``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment import focal_toggle_execution as fx  # noqa: E402
from src.experiment.provenance import canonical_sha256  # noqa: E402


def stage_plan() -> dict:
    cfg = fx.load_config()
    rows = []
    for exp in fx.planned_experiments():
        ids = fx.training_segment_ids(exp.fold, exp.arm)
        rows.append({"execution_position": exp.execution_position,
                     "experiment_id": exp.experiment_id, "fold": exp.fold, "arm": exp.arm,
                     "seed": exp.seed, "training_segments": len(ids),
                     "membership_hash_matches_plan":
                         fx.membership_hash(ids) == exp.training_membership_sha256,
                     "training_record_exists": fx.training_record_path(exp).exists(),
                     "score_artifact_exists": fx.score_path(exp).exists()})
    return {"config_identity": cfg["config_identity"],
            "execution_environment_id": cfg["execution_environment"]["execution_environment_id"],
            "all_memberships_match": all(r["membership_hash_matches_plan"] for r in rows),
            "next_in_frozen_order": (fx.next_in_frozen_order() or fx.PlannedExperiment(
                "none (all trained)", 0, 0, "", 0, "")).experiment_id,
            "experiments": rows}


def stage_train(experiment_id: str | None, device: str, restart_count: int) -> dict:
    nxt = fx.next_in_frozen_order()
    if nxt is None:
        raise SystemExit("all ten experiments already have training records")
    if experiment_id and experiment_id != nxt.experiment_id:
        raise SystemExit(f"refusing: next in frozen order is {nxt.experiment_id}, not {experiment_id}")
    record = fx.train_experiment(nxt, device=device, restart_count=restart_count)
    return {k: v for k, v in fx.asdict(record).items()
            if k not in ("training_losses", "valid_losses")}


def stage_freeze() -> dict:
    """COMMIT B summary: every checkpoint hash, init hash and pairing check. No performance."""

    records = {}
    for exp in fx.planned_experiments():
        p = fx.training_record_path(exp)
        if not p.exists():
            raise SystemExit(f"cannot freeze: {exp.experiment_id} is not trained")
        records[exp.experiment_id] = json.loads(p.read_text(encoding="utf-8"))
    cfg = fx.load_config()
    policy = cfg["initialization_policy"]
    pairs, state_pairs = {}, {}
    for exp in fx.planned_experiments():
        rec = records[exp.experiment_id]
        pairs.setdefault(exp.fold, {})[exp.arm] = rec["initialization_sha256"]
        state_pairs.setdefault(exp.fold, {})[exp.arm] = rec["initial_state_sha256"]
    pairing = {fold: arms["PRESENT"] == arms["ABSENT"] for fold, arms in pairs.items()}
    state_pairing = {fold: arms["PRESENT"] == arms["ABSENT"] for fold, arms in state_pairs.items()}
    if not all(pairing.values()):
        raise SystemExit(f"paired initialisation hashes differ: {pairing}")
    if not all(state_pairing.values()):
        raise SystemExit(f"paired initial-state hashes differ: {state_pairing}")
    # every model must also reproduce the PRE-REGISTERED values (F2/A4)
    expected_init = policy["expected_initialization_sha256"]
    expected_state = policy["expected_initial_state_sha256"]
    off = {eid: (r["initialization_sha256"], r["initial_state_sha256"]) for eid, r in records.items()
           if r["initialization_sha256"] != expected_init
           or r["initial_state_sha256"] != expected_state}
    if off:
        raise SystemExit(f"initial state does not match the pre-registered hashes: {sorted(off)}")
    summary = {
        "schema_version": "phase3.4-commit-b-freeze-v2",
        "config_identity": cfg["config_identity"],
        "checkpoints": {eid: {"checkpoint_sha256": r["checkpoint_sha256"],
                              "initialization_sha256": r["initialization_sha256"],
                              "initial_state_sha256": r["initial_state_sha256"],
                              "checkpoint_on_disk_matches":
                                  fx.file_sha256(Path(r["checkpoint_path"])) == r["checkpoint_sha256"],
                              "epochs_run": r["epochs_run"],
                              "final_epoch_actually_scored": r["final_epoch_actually_scored_one_indexed"]}
                        for eid, r in records.items()},
        "paired_initialization_hashes_equal_by_fold": pairing,
        "paired_initial_state_hashes_equal_by_fold": state_pairing,
        "all_models_match_pre_registered_initial_hashes": True,
        "contains_test_performance": False,
    }
    out = fx.RESULTS_ROOT / "commit_b_checkpoint_freeze.json"
    fx.atomic_write_text(out, json.dumps(summary, indent=2))
    summary["written_to"] = str(out).replace("\\", "/")
    return summary


def stage_authorize(evaluator_sha: str) -> dict:
    from src.experiment.ledger import TestLookLedger

    ledger = TestLookLedger(fx.LEDGER)
    before = len(ledger.read())
    records = fx.build_authorisations(evaluator_sha)
    ledger.append_many_atomic(records, expected_existing_count=before)
    after = len(ledger.read())
    out = fx.RESULTS_ROOT / "test_look_authorization.json"
    payload = {"ledger_before": before, "entries_added": len(records), "ledger_after": after,
               "protocol": fx.PROTOCOL, "evaluator_git_sha": evaluator_sha,
               "entries": [{"experiment_id": r.experiment_id, "reason": r.reason} for r in records]}
    fx.atomic_write_text(out, json.dumps(payload, indent=2))
    return payload


def stage_score(experiment_id: str, device: str, recovery_id: str | None = None) -> dict:
    if recovery_id:
        exp = next((e for e in fx.recovery_experiments()
                    if e.experiment_id == experiment_id and e.recovery_id == recovery_id), None)
        if exp is None:
            raise SystemExit(f"unknown recovery execution {experiment_id} / {recovery_id}")
        return fx.score_experiment(exp, device=device)
    exp = next((e for e in fx.planned_experiments() if e.experiment_id == experiment_id), None)
    if exp is None:
        raise SystemExit(f"unknown experiment id {experiment_id}")
    return fx.score_experiment(exp, device=device)


def stage_analyze(five_fold_recovered: bool = False, corrected_folds: str | None = None) -> dict:
    if corrected_folds:
        folds = [int(f) for f in corrected_folds.split(",")]
        result = fx.analyse_corrected(folds)
        out = fx.RESULTS_ROOT / f"phase3_4_impl1_corrected_{len(folds)}fold_analysis.json"
        fx.atomic_write_text(out, json.dumps(result, indent=2))
        result["written_to"] = str(out).replace("\\", "/")
        result["analysis_sha256"] = canonical_sha256(out)
        return result
    if five_fold_recovered:
        result = fx.analyse_five_fold_recovered()
        out = fx.RESULTS_ROOT / Path(fx.load_fold1_amendment()["five_fold_analysis"]["output"].split(" ")[0]).name
        fx.atomic_write_text(out, json.dumps(result, indent=2))
        result["written_to"] = str(out).replace("\\", "/")
        result["analysis_sha256"] = canonical_sha256(out)
        return result
    result = fx.analyse_from_persisted_scores()
    out = fx.RESULTS_ROOT / Path(fx.load_recovery_config()["analysis_output"]["path"]).name
    fx.atomic_write_text(out, json.dumps(result, indent=2))
    result["written_to"] = str(out).replace("\\", "/")
    result["analysis_sha256"] = canonical_sha256(out)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("plan", "train", "freeze", "authorize", "score", "analyze"))
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--evaluator-sha", default=None)
    parser.add_argument("--restart-count", type=int, default=0)
    parser.add_argument("--recovery-id", default=None,
                        help="fold-1 technical-recovery execution identity (amendment v1)")
    parser.add_argument("--five-fold-recovered", action="store_true",
                        help="analyze: compute the recovered five-fold analysis instead of the four-fold")
    parser.add_argument("--corrected-folds", default=None,
                        help="analyze: IMPL-1 corrected analysis over these folds, e.g. 2,3,4,5 or 1,2,3,4,5")
    args = parser.parse_args()

    if args.stage == "plan":
        out = stage_plan()
    elif args.stage == "train":
        out = stage_train(args.experiment_id, args.device, args.restart_count)
    elif args.stage == "freeze":
        out = stage_freeze()
    elif args.stage == "authorize":
        if not args.evaluator_sha:
            raise SystemExit("--evaluator-sha is required")
        out = stage_authorize(args.evaluator_sha)
    elif args.stage == "score":
        if not args.experiment_id:
            raise SystemExit("--experiment-id is required")
        out = stage_score(args.experiment_id, args.device, args.recovery_id)
    else:
        out = stage_analyze(args.five_fold_recovered, args.corrected_folds)
    print(json.dumps(out, indent=2, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
