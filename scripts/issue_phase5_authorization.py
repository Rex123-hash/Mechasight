"""Human-only: record the H1-H7 decisions, approve the bindings, issue tokens and activate.

Run by the principal researcher, never by automation. It records exactly the decisions given
on the command line. If any decision is REJECT, it writes only a decision record and
activates nothing. If all seven are APPROVE, it writes:
- reports/phase5_protected/artifact_bindings.json: the frozen-inventory proposal, marked
  APPROVED by you; content hashes are still null (H3);
- reports/phase5_protected/batch_authorization_record.json: bound to the reviewed commit,
  with the SHA-256 of two new tokens;
- reports/phase5_protected/ACTIVATION.json: the switch, naming the reviewed commit and the content
  identities (SHA-256) of the authorization record and the bindings.

The three files must then be committed together in ONE commit on top of the reviewed commit, and
never edited afterwards. The executor verifies this on every protected operation, together with a
clean tree and a running source identical to the reviewed commit; any later edit, an uncommitted
record or a record pointing at another commit disables protected execution.

The two tokens (scoring, analysis) are printed ONCE and never stored. Before writing anything,
it checks the protected artifacts' stat identities (size, mtime) against the frozen inventory, so
looks are never spent on files that changed. It opens and hashes no protected file.

Example:

```
python scripts/issue_phase5_authorization.py --decided-by "Amaan Khan" --role "principal researcher" \\
  --decision H1=APPROVE H2=APPROVE H3=APPROVE H4=APPROVE H5=APPROVE H6=APPROVE H7=APPROVE \\
  --reviewer "Independent review" --reviewed-commit <40-hex SHA> --review-verdict GREEN
```

Then commit the three files together (one commit, nothing else in it). The source must stay
identical to the reviewed commit.
"""
import argparse
from datetime import date
import json
from pathlib import Path
import secrets
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment import phase5_protected as pp
from src.experiment.phase5_execution import FrozenPackage
from src.experiment.provenance import content_sha256


def _write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path.as_posix()}: a new batch needs a new, versioned record")
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--decided-by", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--decision", nargs=7, required=True, metavar="Hk=APPROVE|REJECT")
    parser.add_argument("--reviewer", required=True, help="who reviewed the activation commit")
    parser.add_argument("--reviewed-commit", required=True)
    parser.add_argument("--review-verdict", required=True, choices=["GREEN", "YELLOW", "RED"])
    args = parser.parse_args()
    decisions = dict(item.split("=", 1) for item in args.decision)
    if set(decisions) != set(pp.HUMAN_DECISIONS) or not set(decisions.values()) <= {"APPROVE", "REJECT"}:
        raise SystemExit("give exactly H1..H7, each =APPROVE or =REJECT")
    today = date.today().isoformat()
    record = {h: {"decision": "APPROVED" if v == "APPROVE" else "REJECTED", "by": f"{args.decided_by} ({args.role})", "on": today}
              for h, v in sorted(decisions.items())}
    if "REJECT" in decisions.values() or args.review_verdict != "GREEN":
        _write(pp.PROTECTED_ROOT / f"decision_record_{today}.json",
               {"decisions": record, "reviewer": args.reviewer, "reviewed_commit": args.reviewed_commit,
                "review_verdict": args.review_verdict, "outcome": "NOT ACTIVATED"})
        raise SystemExit("not activated: a decision was REJECTED or the review is not GREEN (decision record written)")
    if len(args.reviewed_commit) != 40 or any(c not in "0123456789abcdef" for c in args.reviewed_commit):
        raise SystemExit("the reviewed commit must be a full 40-hex SHA")
    package = FrozenPackage.load()
    proposal = pp.protected_bindings_proposal(package)
    # Pre-exposure integrity per H3: stat identities only (no open, no hash).
    changed = []
    for cell, entry in proposal["cells"].items():
        for role, artifact in entry["artifacts"].items():
            stat = (pp.ROOT / artifact["path"]).stat()
            if {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns} != artifact["stat"]:
                changed.append(artifact["path"])
    if changed:
        raise SystemExit(f"not activated: protected artifacts differ from the frozen inventory: {changed}")
    bindings = {**proposal, "status": pp.APPROVED, "approval": {"by": f"{args.decided_by} ({args.role})", "on": today}}
    scoring, analysis = secrets.token_hex(32), secrets.token_hex(32)
    authorization = pp.authorization_template(package, content_sha256(bindings), args.reviewed_commit)
    authorization.update(human_decisions=record,
                         independent_reviewer={"name": args.reviewer, "reviewed_commit": args.reviewed_commit,
                                               "verdict": args.review_verdict},
                         scoring_token_sha256=pp.token_sha256(scoring), analysis_token_sha256=pp.token_sha256(analysis),
                         issued_by=f"{args.decided_by} ({args.role})", issued_on=today)
    _write(pp.APPROVED_BINDINGS_PATH, bindings)
    _write(pp.AUTHORIZATION_PATH, authorization)
    _write(pp.ACTIVATION_PATH, {"schema": pp.ACTIVATION_SCHEMA, "enabled": True,
                                "activated_by": f"{args.decided_by} ({args.role})", "activated_on": today,
                                "reviewed_commit": args.reviewed_commit,
                                "authorization_sha256": content_sha256(authorization),
                                "bindings_sha256": content_sha256(bindings)})
    print("Written: artifact_bindings.json, batch_authorization_record.json, ACTIVATION.json.")
    print("Commit all three together in one commit on top of the reviewed commit; never edit them afterwards.")
    print("Seal the real ledger read-only before scoring: "
          "python scripts/seal_phase5_ledger.py --confirm SEAL-THE-REAL-LEDGER")
    print("Keep these two tokens private; they are shown once and are not stored anywhere:")
    print(f"  MECHASIGHT_SCORING_TOKEN={scoring}")
    print(f"  MECHASIGHT_ANALYSIS_TOKEN={analysis}")


if __name__ == "__main__":
    main()
