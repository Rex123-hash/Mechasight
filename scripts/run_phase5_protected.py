"""Phase 5.7/5.8 protected-evaluation driver (A1-A6; refused unless a human has activated it).

Commands:
  bindings-proposal       write the metadata-only protected artifact-binding proposal
                          (from the frozen d7ea8ac inventory; no protected file opened or statted)
  authorization-template  write the unsigned human batch-authorization template (all H1-H7 UNRESOLVED)
  score-all               protected score-all (232 looks, 196 -> 428); refused unless activated
  analyse-once            protected analyse-once under the separate analysis token; refused unless activated
  recover                 H4 recovery batch under its own committed recovery record and token; refused unless activated
  rehearse                the identical chain on the development mirror: rehearsal bindings ->
                          rehearsal authorization with synthetic tokens -> preflight -> score-all
                          (temporary ledger 196 -> 428) -> analyse-once (synthetic labels)

Tokens are read from the environment (MECHASIGHT_SCORING_TOKEN / MECHASIGHT_ANALYSIS_TOKEN /
MECHASIGHT_RECOVERY_TOKEN) and never written; the records store only their SHA-256.

There is no reviewed-commit argument: the reviewed commit is taken only from the committed human
records, and the running source must equal that commit's Git objects (``verified_activation``).
"""
import argparse
import json
import os
from pathlib import Path
import secrets
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment import phase5_protected as pp
from src.experiment.phase5_execution import FrozenPackage
from src.experiment.phase5_inputs import Refused
from src.experiment.provenance import canonical_json, file_sha256

ROOT = pp.PROTECTED_ROOT


def _write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {path.as_posix()}")


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rehearse(package):
    """Full rehearsal; returns evidence. Development data, temporary ledger, synthetic labels only."""
    real_before = file_sha256(pp.REAL_LEDGER)
    bindings = pp.rehearsal_bindings(package)
    bindings_sha256 = pp.validate_bindings(bindings, package)
    scoring, analysis = secrets.token_hex(32), secrets.token_hex(32)
    record = pp.rehearsal_authorization(package, bindings_sha256, scoring, analysis)
    refusals = {}
    for name, call in {
        "analysis_token_for_scoring": lambda: pp.check_token(record, analysis, "score"),
        "scoring_token_for_analysis": lambda: pp.check_token(record, scoring, "analysis"),
        "protected_score_all": lambda: pp.score_all(package, scope=pp.PROTECTED, bindings=bindings,
                                                    authorization=record, token=scoring),
        "protected_bindings": lambda: pp.validate_bindings(pp.protected_bindings_proposal(package), package),
    }.items():
        try:
            call()
            refusals[name] = False
        except Refused:
            refusals[name] = True
    box, executor, state, checks = pp.score_all(package, scope=pp.REHEARSAL, bindings=bindings, authorization=record,
                                                token=scoring, workdir=tempfile.mkdtemp(prefix="mechasight-SYNTHETIC-5-6c-r2-rehearsal-"))
    report = pp.analyse_once(package, box, executor, scope=pp.REHEARSAL, bindings=bindings, authorization=record, token=analysis)
    from src.experiment.ledger_history import verify_ledger_history
    rows = len(verify_ledger_history(box.path("ledger/test_look_ledger.jsonl"),
                                     authorization_root=box.path("ledger/authorizations")).rows)
    binding_checks = sum(1 for _, n in box.events if n.startswith("trusted:"))
    require = pp.require
    require(file_sha256(pp.REAL_LEDGER) == real_before == pp.REAL_LEDGER_SHA256, "real ledger changed")
    return {"scope": pp.REHEARSAL, "not_protected_performance": True, "protected_execution_enabled": pp.protected_execution_enabled(),
            "bindings_sha256": bindings_sha256, "authorization_sha256": checks["authorization_sha256"],
            "units_scored": len(state["tables"]), "temporary_ledger_rows_before_after": [196, rows],
            "binding_checks_at_read": binding_checks, "analysis_primary_contrasts": report["primary_contrasts"],
            "analysis_primary_intervals": report["primary_intervals"], "analysis_scope": report["scope"],
            "refusals": refusals, "real_ledger_sha256_unchanged": True, "fixture_root": "<temp>"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("bindings-proposal")
    sub.add_parser("authorization-template")
    for name in ("score-all", "analyse-once", "recover"):
        p = sub.add_parser(name)
        if name != "score-all":
            p.add_argument("--staging", required=True, help="the scoring batch's staging directory (printed by score-all)")
        if name == "recover":
            p.add_argument("--recovery-record", required=True,
                           help="committed record under reports/phase5_protected/recoveries/")
    r = sub.add_parser("rehearse")
    r.add_argument("--write-evidence", action="store_true")
    args = parser.parse_args()
    package = FrozenPackage.load()
    if args.command == "bindings-proposal":
        _write(ROOT / "artifact_bindings_proposal.json", pp.protected_bindings_proposal(package))
    elif args.command == "authorization-template":
        proposal = pp.protected_bindings_proposal(package)
        from src.experiment.provenance import content_sha256
        _write(ROOT / "batch_authorization_template.json",
               pp.authorization_template(package, content_sha256(proposal), "<reviewed activation commit>"))
    elif args.command in {"score-all", "analyse-once", "recover"}:
        env = {"score-all": "MECHASIGHT_SCORING_TOKEN", "analyse-once": "MECHASIGHT_ANALYSIS_TOKEN",
               "recover": "MECHASIGHT_RECOVERY_TOKEN"}[args.command]
        token = os.environ.get(env, "")
        try:
            pp.load_protected_records()           # fails closed unless the committed records verify
            if args.command == "score-all":
                box, _, state, checks = pp.score_all(package, scope=pp.PROTECTED, token=token)
                print(canonical_json({"status": state["status"], "tables": len(state["tables"]), "staging": str(box.root),
                                      "authorization_sha256": checks["authorization_sha256"]}))
            else:
                from src.experiment.phase5_inputs import Sandbox
                box = Sandbox(args.staging)
                if args.command == "analyse-once":
                    report = pp.analyse_once(package, box, None, scope=pp.PROTECTED, token=token)
                    print(canonical_json({"scope": report["scope"], "primary_contrasts": report["primary_contrasts"],
                                          "primary_intervals": report["primary_intervals"]}))
                else:
                    _, state = pp.recover(package, box, recovery_record=args.recovery_record, token=token)
                    print(canonical_json({"status": state["status"], "tables": len(state["tables"])}))
        except Refused as exc:
            raise SystemExit(f"REFUSED: {exc}")
    else:
        evidence = rehearse(package)
        if args.write_evidence:
            _write(ROOT / "rehearsal_evidence.json", evidence)
        print(canonical_json({k: evidence[k] for k in ("units_scored", "temporary_ledger_rows_before_after",
                                                       "binding_checks_at_read", "refusals")}))


if __name__ == "__main__":
    main()
