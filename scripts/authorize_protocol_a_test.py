"""Write the Protocol-A test-look ledger entries BEFORE any official-test scoring.

Conservative extension: the existing ledger schema already carries a ``protocol``
field, so Protocol-A records are appended with
``protocol = "official_baseline_reproduction"`` and remain fully distinguishable
from the 28 Protocol-B records without any schema change.

    python scripts/authorize_protocol_a_test.py --evaluator-sha <sha>
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment.ledger import TestLookLedger, TestLookRecord  # noqa: E402
from src.experiment.provenance import content_sha256, ordered_id_hash  # noqa: E402

LEDGER = Path("reports/experiment_results/test_look_ledger.jsonl")
CONFIG = Path("configs/phase3/protocol_a_official_reproduction_v1.json")
WORK = Path("data/interim/protocol_a")

PROTOCOL = "official_baseline_reproduction"
REASON = "first locked Protocol-A official-baseline reproduction"
METRICS = ("official_pairwise_auc_overall", "official_pairwise_auc_source",
           "official_pairwise_auc_target")


def official_test_cohort_hash(machine: str) -> tuple[str, int]:
    path = WORK / machine / "test_dataset_window_0.100s.h5"
    with h5py.File(path, "r") as f:
        ids = [v.decode("utf-8") for v in f["segment_id"][:].flatten()]
    unique = sorted(set(ids))
    return ordered_id_hash(unique), len(unique)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator-sha", required=True)
    args = parser.parse_args()

    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    identity = cfg["config_identity"]
    ledger = TestLookLedger(LEDGER)
    existing = ledger.read()
    before = len(existing)

    protocol_b = [e for e in existing if e.get("protocol") == "mechasight_experimental_protocol"]
    if len(protocol_b) != 28:
        raise SystemExit(f"expected the 28 Protocol-B entries to be intact, found {len(protocol_b)}")
    if any(e.get("protocol") == PROTOCOL for e in existing):
        raise SystemExit("Protocol-A entries already exist; refusing to duplicate a test look")

    timestamp = datetime.now(timezone.utc).isoformat()
    records, detail = [], []
    for machine in ("RoboticArm", "BrushlessMotor"):
        cohort_hash, segments = official_test_cohort_hash(machine)
        experiment_id = "p33a-" + content_sha256(
            {"protocol": PROTOCOL, "machine": machine, "config": identity})[:20]
        records.append(TestLookRecord(
            timestamp_utc=timestamp,
            experiment_id=experiment_id,
            git_sha=args.evaluator_sha,
            machine=machine,
            protocol=PROTOCOL,
            regime="T",
            model_config_identity=identity,
            test_cohort_hash=cohort_hash,
            reason=REASON,
            metrics_requested=METRICS,
        ))
        detail.append({"machine": machine, "experiment_id": experiment_id,
                       "test_cohort_hash": cohort_hash, "official_test_segments": segments})

    ledger.append_many_atomic(records, expected_existing_count=before)
    after = len(ledger.read())

    summary = {"ledger_before": before, "entries_added": len(records), "ledger_after": after,
               "protocol": PROTOCOL, "evaluator_git_sha": args.evaluator_sha,
               "config_identity": identity, "timestamp_utc": timestamp,
               "protocol_b_entries_preserved": 28, "entries": detail}
    out = Path("reports/phase3_3_protocol_a/protocol_a_ledger_authorization.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
