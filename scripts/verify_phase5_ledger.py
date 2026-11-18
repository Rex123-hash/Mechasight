"""Read-only integrity audit of the real test-look ledger (detection only; it changes nothing).

Recomputes the ledger's integrity from its bytes against anchors outside the file: the reviewed SHA-256
of the closed 196-row history, the frozen history verifier and batch authorizations, each batch's recorded
before/after digests, and the ledger committed at Git HEAD. Prints a JSON report; exits 2 on any mismatch.

    python scripts/verify_phase5_ledger.py

Run it after every protected operation (and before committing the ledger), and before reporting results.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment import phase5_protected as pp  # noqa: E402
from src.experiment.phase5_execution import FrozenPackage  # noqa: E402
from src.experiment.phase5_inputs import Refused  # noqa: E402
from src.experiment.phase5_ledger import look_budget  # noqa: E402
from src.experiment.phase5_ledger_audit import audit_ledger  # noqa: E402
from src.experiment.provenance import canonical_json  # noqa: E402


def main():
    budget = look_budget(FrozenPackage.load())
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    committed = subprocess.run(["git", "-C", str(ROOT), "show", f"{head}:{pp.REAL_LEDGER.as_posix()}"], capture_output=True)
    try:
        report = audit_ledger(ROOT / pp.REAL_LEDGER, authorization_root=ROOT / pp.BATCH_AUTH_ROOT,
                              baseline_rows=budget["baseline_rows"], baseline_sha256=pp.REAL_LEDGER_SHA256,
                              max_total_rows=budget["max_total_rows"], scope_marker=pp.PROTECTED,
                              committed_bytes=committed.stdout if committed.returncode == 0 else None,
                              require_committed_anchor=True)
    except Refused as exc:
        print(f"LEDGER INTEGRITY FAILURE: {exc}", file=sys.stderr)
        return 2
    print(canonical_json({**report, "git_head": head}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
