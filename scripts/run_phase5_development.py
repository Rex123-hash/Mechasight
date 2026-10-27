"""Multi-Sensor Fusion Development (Phase 5.3): development-only L1 fits and L2 calibration.

Training and source-validation data only.  No protected input or sealed test
score is read, no protected inference runs and the test-look ledger is never
written (it is checked to be unchanged before and after every command).

Commands
  l1-fit         fit the 24 planned L1 units in registry order, one fresh
                 subprocess per fit; completed units are skipped
  l1-fit-one     fit one unit (used by l1-fit)
  l1-verify      completion check + validation-score reproduction of all 24 fits
  l2-calibrate   compute and persist the 208 L2 calibration records
  l2-verify      recompute the 208 records and compare with the persisted ones

Outputs land in ``reports/phase5_development`` (committed) and
``data/interim/phase5/models`` (git-ignored, hash-manifested).  Real outputs
require a clean tracked tree, so every artifact names committed code.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment.ledger_history import verify_ledger_history  # noqa: E402
from src.experiment.provenance import canonical_json, canonical_sha256, content_sha256  # noqa: E402
from src.fusion.assembly import load_design_registry  # noqa: E402
from src.fusion.identity import bundle_identity, unit_binding  # noqa: E402
from src.fusion.l1 import L1FitError, l1_units, load_l1_bundle, reproduce_l1_bundle, run_l1_fit  # noqa: E402
from src.fusion.assembly import PREPROCESSING_CONFIGS  # noqa: E402
from src.fusion.l2 import calibrate_all, verify_calibration_artifacts, write_calibration_artifacts  # noqa: E402

DEV_ROOT = Path("reports/phase5_development")
L1_RESULT_ROOT = DEV_ROOT / "l1" / "experiments"
L1_MODEL_ROOT = Path("data/interim/phase5/models")
L1_RUN_LOG = DEV_ROOT / "l1" / "fit_run_log.jsonl"
L2_ROOT = DEV_ROOT / "l2"
FEATURE_ROOT = Path("data/interim/phase31_primary")
SPECTRAL_ROOT = Path("data/interim/phase4/spectral")
PHASE4_PRETEST_ROOT = Path("reports/phase4_pretest/experiments")
PHASE4_MODEL_ROOT = Path("data/interim/phase4/models")
SPLIT_MANIFEST = Path("reports/phase2_2/protocol_b_few_shot_adaptation_v3.csv")
LEDGER_ROWS = 196

# Reproducibility policy inherited from Phase 4: every single-modality AE fit ran
# with 4 torch intra-op threads (recorded in all 60 histories); the thread count
# changes floating-point summation order, so Phase-5 fits pin the same value.
TORCH_THREADS = "4"
# Pre-flight memory gate: measured peak working set of one fit (Phase-5.3 profiles)
# plus a fixed margin must be physically available before a fit process starts.
MEASURED_PEAK_GB = {("dense_autoencoder", "RoboticArm"): 3.25, ("dense_autoencoder", "BrushlessMotor"): 1.81,
                    ("mahalanobis_shrinkage", "RoboticArm"): 0.5, ("mahalanobis_shrinkage", "BrushlessMotor"): 0.5}
MEMORY_MARGIN_GB = 1.0

SOURCE_IDENTITY_PATHS = (
    Path("src/fusion/assembly.py"), Path("src/fusion/l1.py"), Path("src/fusion/l2.py"),
    Path("src/fusion/statistics.py"), Path("src/models/mahalanobis.py"), Path("src/models/dense_autoencoder.py"),
    Path("src/experiment/phase4.py"), Path("src/experiment/data.py"), Path("src/experiment/pretest.py"),
    Path("src/experiment/scores.py"), Path("src/experiment/gate.py"), Path("src/preprocessing/normalization.py"),
    Path("src/preprocessing/spectral_cache.py"), Path("src/preprocessing/spectral_windows.py"),
    Path("configs/phase5/fusion_design_v2.json"), Path("scripts/run_phase5_development.py"),
)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def _identity() -> tuple[str, str]:
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing: the tracked tree must be clean so artifacts name committed code")
    return _git("rev-parse", "HEAD"), content_sha256({str(p): canonical_sha256(p) for p in SOURCE_IDENTITY_PATHS})


def _ledger_rows() -> int:
    return len(verify_ledger_history().rows)


def _log(record: dict) -> None:
    L1_RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with L1_RUN_LOG.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json({"utc": datetime.now(timezone.utc).isoformat(), **record}) + "\n")


def _unit_name(unit) -> str:
    return f"{unit['machine']}|{unit['regime']}|{unit['family']}|{unit['seed']}"


def completed_l1_bundles() -> dict[str, dict]:
    """Unit name -> loaded complete bundle; a unit with two complete bundles is an error."""

    found: dict[str, dict] = {}
    if not L1_RESULT_ROOT.is_dir():
        return found
    for path in sorted(p for p in L1_RESULT_ROOT.iterdir() if p.is_dir() and not p.name.startswith(".")):
        bundle = load_l1_bundle(path)
        name = _unit_name(bundle["provenance"]["fusion_unit"])
        if name in found:
            raise SystemExit(f"refusing: unit {name} has more than one complete L1 bundle")
        found[name] = bundle
    return found


def _memory_gate(unit) -> tuple[bool, float, float]:
    need = MEASURED_PEAK_GB[(unit["family"], unit["machine"])] + MEMORY_MARGIN_GB
    available = psutil.virtual_memory().available / 2**30
    return available >= need, available, need


def cmd_l1_fit_one(args) -> int:
    head, source_hash = _identity()
    if os.environ.get("MECHASIGHT_TORCH_THREADS") != TORCH_THREADS:
        raise SystemExit(f"refusing: MECHASIGHT_TORCH_THREADS must be {TORCH_THREADS} (Phase-4 reproducibility policy)")
    unit = l1_units(load_design_registry())[args.unit_index]
    _log({"unit": _unit_name(unit), "status": "STARTED", "code_git_sha": head, "phase5_source_hash": source_hash})
    try:
        result = run_l1_fit(
            unit=unit, feature_root=FEATURE_ROOT, spectral_cache_root=SPECTRAL_ROOT, model_root=L1_MODEL_ROOT,
            result_root=L1_RESULT_ROOT, split_manifest_path=SPLIT_MANIFEST, code_git_sha=head,
            code_source_hash=source_hash, phase4_pretest_root=PHASE4_PRETEST_ROOT, phase4_model_root=PHASE4_MODEL_ROOT)
    except Exception as error:  # recorded, then re-raised: a failure is never silent
        _log({"unit": _unit_name(unit), "status": "FAILED", "error": repr(error),
              "traceback": traceback.format_exc(limit=8)})
        raise
    _log({"unit": _unit_name(unit), "status": "COMPLETED", "experiment_id": result["experiment_id"],
          "summary": {k: result[k] for k in ("threshold_q90", "epochs_run", "best_epoch",
                                              "best_validation_window_mse", "stopped_early", "fit_seconds")}})
    print(canonical_json(result), flush=True)
    return 0


def cmd_l1_fit(args) -> int:
    _identity()
    before = _ledger_rows()
    units = l1_units(load_design_registry())
    done = completed_l1_bundles()
    for index, unit in enumerate(units):
        if args.families and unit["family"] not in args.families:
            continue
        if args.machines and unit["machine"] not in args.machines:
            continue
        if _unit_name(unit) in done:
            continue
        ok, available, need = _memory_gate(unit)
        if not ok:
            _log({"unit": _unit_name(unit), "status": "BLOCKED_RESOURCE",
                  "available_gb": round(available, 2), "required_gb": need})
            print(f"BLOCKED_RESOURCE {_unit_name(unit)}: {available:.2f} GB available < {need:.2f} GB required", flush=True)
            continue
        orphan = [p for p in L1_MODEL_ROOT.glob("*") if p.is_dir() and not (L1_RESULT_ROOT / p.name).exists()] \
            if L1_MODEL_ROOT.is_dir() else []
        if orphan:
            raise SystemExit(f"refusing: model artifacts without a completed bundle (incomplete fit): {orphan}")
        completed = subprocess.run([sys.executable, __file__, "l1-fit-one", "--unit-index", str(index)],
                                   env={**os.environ, "MECHASIGHT_TORCH_THREADS": TORCH_THREADS})
        if completed.returncode != 0:
            raise SystemExit(f"L1 fit of {_unit_name(unit)} failed (see {L1_RUN_LOG}); stopping")
    if _ledger_rows() != before or before != LEDGER_ROWS:
        raise SystemExit("ledger changed during development fitting")
    print(json.dumps({"completed_units": len(completed_l1_bundles()), "planned_units": len(units)}))
    return 0


def cmd_l1_verify(args) -> int:
    head, _ = _identity()
    units = l1_units(load_design_registry())
    done = completed_l1_bundles()
    names = {_unit_name(u) for u in units}
    if set(done) - names:
        raise SystemExit(f"unplanned L1 bundles: {sorted(set(done) - names)}")
    missing = sorted(names - set(done))
    rows = []
    for unit in units:
        if _unit_name(unit) not in done:
            continue
        bundle = done[_unit_name(unit)]
        check = reproduce_l1_bundle(bundle["path"], model_root=L1_MODEL_ROOT, feature_root=FEATURE_ROOT,
                                    spectral_cache_root=SPECTRAL_ROOT, split_manifest_path=SPLIT_MANIFEST)
        # Per-bundle identity: each bundle's own commit, source hash and raw input-hash serialisation.
        identity = bundle_identity(bundle, source_paths=SOURCE_IDENTITY_PATHS,
                                   preprocessing_configs=PREPROCESSING_CONFIGS, split_manifest=SPLIT_MANIFEST)
        # Scientific-unit binding: labels, recipe, representation and model identity equal the
        # registry unit this bundle is counted for (independent audit finding A, Phase 5.4).
        binding = unit_binding(bundle, unit, model_root=L1_MODEL_ROOT)
        rows.append({"unit": _unit_name(unit), **check, "identity": identity, "binding": binding,
                     "code_git_sha": bundle["config"]["code_git_sha"],
                     "torch_threads": json.loads((bundle["path"] / "validation_summary.json").read_text(
                         encoding="utf-8"))["training_history"].get("torch_threads"),
                     "standardizer_equal_to_phase4": bundle["provenance"]["score_provenance"]["data_provenance"]
                     ["standardizer_equivalence"]["equal_to_phase4_standardizers"]})
    report = {"verified_at_head": head, "planned_units": len(units), "completed_units": len(done),
              "status": "COMPLETE" if not missing else "PARTIAL", "missing_units": missing,
              "all_reproduced": all(r["reproduced"] for r in rows),
              "all_standardizers_equal_to_phase4": all(r["standardizer_equal_to_phase4"] for r in rows),
              "all_identities_valid": all(r["identity"]["identity_valid"] for r in rows),
              "all_units_bound": all(r["binding"]["binding_valid"] for r in rows),
              "input_hash_forms": {form: sum(1 for r in rows if r["identity"]["preprocessing_config_hash_form"] == form
                                             and r["identity"]["split_manifest_hash_form"] == form)
                                   for form in ("CRLF", "LF")},
              "duplicate_experiment_ids": sorted({r["experiment_id"] for r in rows
                                                  if sum(x["experiment_id"] == r["experiment_id"] for x in rows) > 1}),
              "ledger_rows": _ledger_rows(), "test_evaluated": False, "units": rows}
    verified = (report["all_reproduced"] and report["all_standardizers_equal_to_phase4"]
                and report["all_identities_valid"] and report["all_units_bound"]
                and not report["duplicate_experiment_ids"])
    # Completion is never inferred from the bundle count alone.
    if not verified:
        report["status"] = "FAILED_VERIFICATION"
    out = DEV_ROOT / "l1" / "l1_completion_and_reproduction.json"
    out.write_bytes((canonical_json(report) + "\n").encode("utf-8"))
    print(json.dumps({k: v for k, v in report.items() if k != "units"}, indent=1))
    if not verified:
        return 1
    return 0 if not missing else 2


def cmd_l2_calibrate(args) -> int:
    head, source_hash = _identity()
    before = _ledger_rows()
    records, fused = calibrate_all(PHASE4_PRETEST_ROOT)
    manifest = write_calibration_artifacts(L2_ROOT, records, fused, code_git_sha=head, code_source_hash=source_hash)
    if _ledger_rows() != before:
        raise SystemExit("ledger changed during L2 calibration")
    print(json.dumps(manifest, indent=1))
    return 0


def cmd_l2_verify(args) -> int:
    print(json.dumps(verify_calibration_artifacts(L2_ROOT, PHASE4_PRETEST_ROOT), indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("l1-fit")
    fit.add_argument("--families", nargs="+", default=None)
    fit.add_argument("--machines", nargs="+", default=None)
    one = sub.add_parser("l1-fit-one")
    one.add_argument("--unit-index", type=int, required=True)
    sub.add_parser("l1-verify")
    sub.add_parser("l2-calibrate")
    sub.add_parser("l2-verify")
    args = parser.parse_args()
    return {"l1-fit": cmd_l1_fit, "l1-fit-one": cmd_l1_fit_one, "l1-verify": cmd_l1_verify,
            "l2-calibrate": cmd_l2_calibrate, "l2-verify": cmd_l2_verify}[args.command](args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except L1FitError as error:
        raise SystemExit(f"L1 fit refused: {error}")
