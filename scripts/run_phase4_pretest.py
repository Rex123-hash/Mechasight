"""Fit all 156 frozen Single-Modality Benchmarking configurations; validation only.

No common-test cohort is materialised or scored.  Bundles land under
``reports/phase4_pretest/experiments`` and hash-manifested model artifacts
under ``data/interim/phase4/models``.  The proposed protected batch identity
is written for review; the ledger is not touched.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment.ledger import TestLookLedger  # noqa: E402
from src.experiment.phase4 import (  # noqa: E402
    EXPECTED_EXPERIMENT_COUNT,
    LEDGER_EXPECTED_START,
    PHASE4_SCHEMA_VERSION,
    experiment_plan,
    load_frozen_bundles,
    plan_protected_batch,
    run_one_pretest_experiment,
)
from src.experiment.phase32 import common_test_artifact_paths  # noqa: E402
from src.experiment.provenance import canonical_json, canonical_sha256, content_sha256, file_sha256  # noqa: E402


SOURCE_IDENTITY_PATHS = (
    Path("src/models/classical.py"),
    Path("src/models/mahalanobis.py"),
    Path("src/models/dense_autoencoder.py"),
    Path("src/experiment/phase4.py"),
    Path("src/experiment/phase4_analysis.py"),
    Path("src/experiment/data.py"),
    Path("src/experiment/pretest.py"),
    Path("src/preprocessing/spectral_windows.py"),
    Path("src/preprocessing/spectral_cache.py"),
    Path("configs/phase4/single_modality_benchmark_v1.json"),
)
LEDGER_PATH = Path("reports/experiment_results/test_look_ledger.jsonl")


def _head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def _source_hash() -> str:
    return content_sha256({str(path): canonical_sha256(path) for path in SOURCE_IDENTITY_PATHS})


def _frozen_test_cohort_hashes(common_test_root: Path) -> dict[str, str]:
    """Cohort identity hashes only (segment-id hashes recorded at Phase 3.2)."""

    hashes = {}
    for machine in ("RoboticArm", "BrushlessMotor"):
        values = set()
        for regime in ("S", "T"):
            _artifact, metadata_path = common_test_artifact_paths(common_test_root, machine=machine, regime=regime)
            values.add(json.loads(metadata_path.read_text(encoding="utf-8"))["cohort_hash"])
        if len(values) != 1:
            raise ValueError(f"{machine} common-test cohort hash differs between regimes")
        hashes[machine] = values.pop()
    return hashes


def _markdown(rows: list[dict], *, head: str, source_hash: str, ledger_count: int, planned: dict) -> str:
    lines = [
        "# Single-Modality Model Benchmarking — Pre-Test Freeze Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Status: validation-only pre-test fitting of all 156 frozen configurations complete.",
        "",
        "Common-test evaluation: **NO**. No test cohort was materialised, scored or metric-evaluated.",
        "",
        f"Test-look ledger entries before/after: {ledger_count}/{ledger_count} (expected start for the protected batch: {LEDGER_EXPECTED_START}).",
        "",
        f"Code HEAD at fit time: `{head}`. Source snapshot hash: `{source_hash}`.",
        "",
        f"Proposed protected batch identity (not authorised): `{planned['batch_id']}`.",
        "",
        "| Machine | Regime | Modality | Family | Seed | Features/bins | q90 threshold | Epochs | Best epoch | Fit s |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['machine']} | {row['regime']} | {row['modality']} | {row['family']} | {row['seed']} | "
            f"{row['feature_count']} | {row['threshold_q90']:.6g} | {row['epochs_run'] or ''} | "
            f"{row['best_epoch'] or ''} | {row['fit_seconds']:.1f} |"
        )
    lines += [
        "",
        "Thresholds are descriptive source-normal q90 calibration values, not test results and not a model ranking.",
        "",
        "BrushlessMotor carries `brushless_motor_acquisition_time_class_confound_v1`; RoboticArm carries the six-parent validation and eighteen-parent adaptation limitations.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, default=Path("data/interim/phase31_primary"))
    parser.add_argument("--spectral-root", type=Path, default=Path("data/interim/phase4/spectral"))
    parser.add_argument("--model-root", type=Path, default=Path("data/interim/phase4/models"))
    parser.add_argument("--out", type=Path, default=Path("reports/phase4_pretest"))
    parser.add_argument("--common-test-root", type=Path, default=Path("data/interim/phase32_common_test"))
    parser.add_argument(
        "--split-manifest", type=Path,
        default=Path("reports/phase2_2/protocol_b_few_shot_adaptation_v3.csv"),
    )
    parser.add_argument("--only-family", default=None)
    parser.add_argument("--machines", nargs="+", default=None)
    parser.add_argument("--regimes", nargs="+", default=None)
    parser.add_argument("--log-name", default="pretest_run_log.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    head = _head()
    source_hash = _source_hash()
    ledger_before = len(TestLookLedger(LEDGER_PATH).read())
    result_root = args.out / "experiments"
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / args.log_name
    rows: list[dict] = []
    existing = load_frozen_bundles(result_root) if (args.skip_existing and result_root.exists()) else {}
    existing_keys = {
        (b.config.machine, b.config.regime, b.modality, b.config.model_family, b.config.model_seed)
        for b in existing.values()
    }
    plan = [
        item for item in experiment_plan()
        if (args.only_family is None or item.family == args.only_family)
        and (args.machines is None or item.machine in args.machines)
        and (args.regimes is None or item.regime in args.regimes)
        and (item.machine, item.regime, item.modality, item.family, item.seed) not in existing_keys
    ]
    if args.limit is not None:
        plan = plan[: args.limit]
    for index, item in enumerate(plan, 1):
        config = Path(f"configs/preprocessing/mechasight_few_shot_regime_{item.regime.lower()}.json")
        if args.skip_existing and result_root.exists():
            # Re-check just in time so parallel workers never fit the same object twice.
            fitted = {
                (b.config.machine, b.config.regime, b.modality, b.config.model_family, b.config.model_seed)
                for b in load_frozen_bundles(result_root).values()
            }
            if (item.machine, item.regime, item.modality, item.family, item.seed) in fitted:
                continue
        row = run_one_pretest_experiment(
            machine=item.machine, regime=item.regime, modality=item.modality,
            family=item.family, seed=item.seed,
            feature_root=args.feature_root, spectral_cache_root=args.spectral_root,
            model_root=args.model_root, result_root=result_root,
            split_manifest_path=args.split_manifest, preprocessing_config_path=config,
            code_git_sha=head, code_source_hash=source_hash,
        )
        row["order"] = index
        rows.append(row)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(row) + "\n")
        print(canonical_json({k: row[k] for k in ("order", "machine", "regime", "modality", "family", "seed", "fit_seconds", "epochs_run", "threshold_q90")}), flush=True)

    ledger_after = len(TestLookLedger(LEDGER_PATH).read())
    if ledger_after != ledger_before:
        raise RuntimeError("the test-look ledger changed during pre-test fitting")
    bundles = load_frozen_bundles(result_root)
    summary = {
        "schema_version": PHASE4_SCHEMA_VERSION,
        "code_git_sha": head,
        "source_hash": source_hash,
        "bundles_present": len(bundles),
        "bundles_expected": EXPECTED_EXPERIMENT_COUNT,
        "ledger_count_before": ledger_before,
        "ledger_count_after": ledger_after,
        "test_evaluated": False,
        "run_rows": len(rows),
    }
    if len(bundles) == EXPECTED_EXPERIMENT_COUNT:
        planned = plan_protected_batch(
            bundles,
            test_cohort_hashes_by_machine=_frozen_test_cohort_hashes(args.common_test_root),
            code_git_sha=head,
        ).to_dict()
        planned["status"] = "PROPOSED_NOT_AUTHORIZED"
        (args.out / "proposed_protected_batch.json").write_text(
            json.dumps(planned, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        summary["proposed_batch_id"] = planned["batch_id"]
        all_rows = []
        for experiment_id, bundle in bundles.items():
            summary_payload = json.loads((bundle.bundle_path / "validation_summary.json").read_text(encoding="utf-8"))
            history = summary_payload.get("training_history") or {}
            all_rows.append({
                "machine": bundle.config.machine, "regime": bundle.config.regime,
                "modality": bundle.modality, "family": bundle.config.model_family,
                "seed": bundle.config.model_seed, "feature_count": summary_payload["feature_count"],
                "threshold_q90": bundle.threshold.value, "epochs_run": history.get("epochs_run"),
                "best_epoch": history.get("best_epoch"), "fit_seconds": summary_payload["fit_seconds"],
            })
        all_rows.sort(key=lambda r: (r["machine"], r["regime"], r["modality"], r["family"], r["seed"]))
        (args.out / "phase4_pretest_freeze_report.md").write_text(
            _markdown(all_rows, head=head, source_hash=source_hash, ledger_count=ledger_after, planned=planned),
            encoding="utf-8",
        )
        (args.out / "bundle_manifest.json").write_text(
            json.dumps({
                "schema_version": PHASE4_SCHEMA_VERSION,
                "experiment_ids": sorted(bundles),
                "bundle_manifest_sha256": {
                    experiment_id: file_sha256(bundle.bundle_path / "bundle_manifest.json")
                    for experiment_id, bundle in sorted(bundles.items())
                },
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (args.out / "phase4_pretest_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
