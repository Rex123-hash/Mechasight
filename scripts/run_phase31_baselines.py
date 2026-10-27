"""Fit the frozen Phase-3.1 baselines and score source validation only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment.ledger import TestLookLedger  # noqa: E402
from src.experiment.phase31 import (  # noqa: E402
    ACCEPTED_PHASE31_HEAD,
    phase31_source_hash,
    run_one_baseline,
    seed_plan,
)
from src.models.classical import PRIMARY_BASELINE_SPECS  # noqa: E402


SOURCE_IDENTITY_PATHS = (
    Path("src/models/classical.py"),
    Path("src/experiment/phase31.py"),
    Path("src/experiment/data.py"),
    Path("src/experiment/pretest.py"),
    Path("configs/phase3/preprocessing_primary_v1.json"),
    Path("configs/phase3/baseline_ladder_v1.json"),
)


def _current_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _markdown_report(records: list[dict], ledger_count: int, source_hash: str) -> str:
    lines = [
        "# Phase 3.1 Pre-Test Freeze Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Status: validation-only pre-test baseline fitting complete.",
        "",
        "Common-test evaluation: **NO**. No test cohort was materialized or scored.",
        "",
        f"Test-look ledger entries before/after: {ledger_count}/{ledger_count}.",
        "",
        f"Accepted starting HEAD: `{ACCEPTED_PHASE31_HEAD}`.",
        "",
        f"Phase-3.1 source snapshot hash: `{source_hash}`.",
        "",
        "| Machine | Regime | Model | Seed | Train segments | Validation segments | Features | q90 threshold |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['machine']} | {row['regime']} | {row['family']} | {row['seed']} | "
            f"{row['training_segments']} | {row['validation_segments']} | "
            f"{row['feature_count']} | {row['threshold_q90']:.12g} |"
        )
    lines.extend([
        "",
        "Thresholds are descriptive source-normal q90 calibration values, not test results and not a model ranking.",
        "",
        "The BrushlessMotor acquisition-time/class confound remains explicitly identified as `brushless_motor_acquisition_time_class_confound_v1`.",
        "",
        "The RoboticArm source-validation cohort contains 186 correlated segments from six physical parent recordings; the Regime-T adaptation allocation remains 27 released segments from 18 parents, with 96 siblings blocked.",
        "",
        "Protocol A, deep models, fusion, corruptions, bootstrap inference, test metrics, and final model selection were not run.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=Path("data/interim/phase31_primary")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("reports/phase3_1")
    )
    parser.add_argument(
        "--split-manifest", type=Path,
        default=Path("reports/phase2_2/protocol_b_few_shot_adaptation_v3.csv"),
    )
    parser.add_argument(
        "--test-look-ledger", type=Path,
        default=Path("reports/experiment_results/test_look_ledger.jsonl"),
    )
    args = parser.parse_args()

    if _current_head() != ACCEPTED_PHASE31_HEAD:
        raise RuntimeError("Phase-3.1 runner requires the accepted starting HEAD")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite Phase-3.1 report directory: {args.out}")
    ledger = TestLookLedger(args.test_look_ledger)
    ledger_before = ledger.read()
    ledger_bytes_before = args.test_look_ledger.read_bytes() if args.test_look_ledger.exists() else None
    source_hash = phase31_source_hash(SOURCE_IDENTITY_PATHS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".phase3_1.", dir=args.out.parent))
    try:
        records = []
        result_root = stage / "validation_only"
        for machine in ("RoboticArm", "BrushlessMotor"):
            for regime in ("S", "T"):
                config = Path(
                    f"configs/preprocessing/mechasight_few_shot_regime_{regime.lower()}.json"
                )
                for spec in PRIMARY_BASELINE_SPECS:
                    for seed in seed_plan(spec.family):
                        record = run_one_baseline(
                            machine=machine,
                            regime=regime,
                            family=spec.family,
                            seed=seed,
                            feature_root=args.features,
                            result_root=result_root,
                            split_manifest_path=args.split_manifest,
                            preprocessing_config_path=config,
                            code_source_hash=source_hash,
                        )
                        record["bundle"] = (
                            f"reports/phase3_1/validation_only/{record['experiment_id']}"
                        )
                        records.append(record)
                        print(json.dumps(record, sort_keys=True), flush=True)
        ledger_after = ledger.read()
        ledger_bytes_after = args.test_look_ledger.read_bytes() if args.test_look_ledger.exists() else None
        if ledger_after != ledger_before or ledger_bytes_after != ledger_bytes_before:
            raise RuntimeError("test-look ledger changed during Phase-3.1; refusing results")
        report = {
            "schema_version": "phase3.1-pretest-freeze-report-v1",
            "evaluation_stage": "validation_only_pretest",
            "test_evaluated": False,
            "accepted_starting_head": ACCEPTED_PHASE31_HEAD,
            "phase31_source_hash": source_hash,
            "test_look_ledger_entries_before": len(ledger_before),
            "test_look_ledger_entries_after": len(ledger_after),
            "experiment_count": len(records),
            "experiments": records,
        }
        (stage / "phase31_pretest_freeze_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (stage / "phase31_pretest_freeze_report.md").write_text(
            _markdown_report(records, len(ledger_before), source_hash), encoding="utf-8"
        )
        stage.rename(args.out)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    print(json.dumps({"phase31_complete": True, "experiments": len(records), "test_looks": len(ledger_before)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
