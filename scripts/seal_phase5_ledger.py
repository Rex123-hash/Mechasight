"""Phase 5.6C-r5: seal the real test-look ledger read-only (a HUMAN activation step).

The frozen ``src/experiment/ledger.py`` helpers (``TestLookLedger.append`` and
``append_many_atomic``) take no lock and enforce no look budget. Sealing the ledger read-only makes
the operating system refuse every such writer before a byte is written. After sealing, the only
writer is ``phase5_ledger.sealed_append`` under the ledger lock, which never unseals the file.

The ledger content is not changed: this script checks the closed 196-row identity, sets the
read-only attribute and checks the identity again. It does not activate anything, issue tokens or
read protected data. Protected execution refuses an unsealed real ledger.

Usage (the principal researcher, at activation, with the records):

    python scripts/seal_phase5_ledger.py --confirm SEAL-THE-REAL-LEDGER
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment import phase5_protected as pp  # noqa: E402
from src.experiment.ledger_history import ledger_lines  # noqa: E402
from src.experiment.phase5_ledger import ledger_lock, ledger_sealed, seal_ledger  # noqa: E402
from src.experiment.provenance import file_sha256  # noqa: E402

CONFIRMATION = "SEAL-THE-REAL-LEDGER"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--confirm", required=True, help=f"must be {CONFIRMATION}")
    args = parser.parse_args(argv)
    if args.confirm != CONFIRMATION:
        raise SystemExit(f"refused: --confirm must be {CONFIRMATION}")
    ledger = ROOT / pp.REAL_LEDGER
    with ledger_lock(ledger):
        before = file_sha256(ledger)
        if before != pp.REAL_LEDGER_SHA256 or len(ledger_lines(ledger)) != 196:
            raise SystemExit("refused: the real ledger is not at its closed 196-row identity; human incident review")
        if ledger_sealed(ledger):
            print("already sealed; content unchanged")
            return 0
        seal_ledger(ledger)
        if file_sha256(ledger) != before:
            raise SystemExit("refused: ledger content changed while sealing; human incident review")
    print(f"sealed read-only: {pp.REAL_LEDGER} (196 rows, SHA-256 {before})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
