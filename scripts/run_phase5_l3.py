"""Multi-Sensor Fusion Development (Phase 5.4): L3 integrity gate, clean scan and falsification suite.

Development-only.  Raw training and source-validation recordings are the only
inputs (``l3.development_segment_rows`` refuses every other segment), corrupted
copies go to a development directory, no protected input or sealed test score
is read, no protected inference runs and the test-look ledger is checked to be
unchanged before and after every command.

Commands
  clean-scan   G1-G3 on every clean training and source-validation
               segment-channel; per-channel flatline distribution; clean-training
               channel statistics used by the corruption generator
  falsify      the 1,992-run end-to-end suite (12 segments x 83 scenarios x 2
               machines), assertions E0-E6 / B1-B5, saturation characterisation
               and the descriptive ungated demonstration

Official outputs (``reports/phase5_development/l3``) require a clean tracked
tree so every artifact names committed code; ``--out`` redirects a development
dry run elsewhere (``--segments`` limits it) and records it as unofficial.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("MECHASIGHT_TORCH_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.experiment.ledger_history import verify_ledger_history  # noqa: E402
from src.experiment.provenance import canonical_json, canonical_sha256, content_sha256  # noqa: E402
from src.fusion import l3, l3_falsification as falsification  # noqa: E402
from src.fusion.assembly import DESIGN_CONFIG, DESIGN_CONFIG_SHA256, PREPROCESSING_CONFIGS, load_design_registry  # noqa: E402
from src.preprocessing.pipeline import load_raw_catalog  # noqa: E402

DATA_ROOT = Path("data/raw/extracted")
SPLIT_MANIFEST = Path("reports/phase2_2/protocol_b_few_shot_adaptation_v3.csv")
PHASE4_PRETEST_ROOT = Path("reports/phase4_pretest/experiments")
PHASE4_MODEL_ROOT = Path("data/interim/phase4/models")
L2_ROOT = Path("reports/phase5_development/l2")
L3_ROOT = Path("reports/phase5_development/l3")
WORK_ROOT = Path("data/interim/phase5/l3_falsification")
LEDGER_ROWS = 196
TORCH_THREADS = "4"

SOURCE_IDENTITY_PATHS = (
    Path("src/fusion/l3.py"), Path("src/fusion/l3_falsification.py"), Path("src/fusion/l2.py"),
    Path("src/fusion/assembly.py"), Path("src/experiment/phase4.py"), Path("src/preprocessing/signals.py"),
    Path("src/preprocessing/pipeline.py"), Path("src/preprocessing/spectral_windows.py"),
    Path("src/features/extraction.py"), Path("src/models/dense_autoencoder.py"), Path("src/models/classical.py"),
    Path("src/models/mahalanobis.py"), Path("configs/phase5/fusion_design_v2.json"),
    Path("configs/preprocessing/mechasight_few_shot_regime_s.json"),
    Path("configs/preprocessing/mechasight_few_shot_regime_t.json"), Path("scripts/run_phase5_l3.py"),
)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def _identity(official: bool) -> dict:
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    if official and dirty:
        raise SystemExit("refusing: the tracked tree must be clean so official artifacts name committed code")
    return {"code_git_sha": _git("rev-parse", "HEAD"), "tracked_tree_clean": not dirty, "official": official,
            "l3_source_hash": content_sha256({str(p): canonical_sha256(p) for p in SOURCE_IDENTITY_PATHS}),
            "design_config": str(DESIGN_CONFIG), "design_config_sha256": canonical_sha256(DESIGN_CONFIG),
            "gate_version": l3.GATE_VERSION, "runner_version": l3.L3_RUNNER_VERSION,
            "corruption_version": falsification.CORRUPTION_VERSION,
            "split_manifest_canonical_sha256": canonical_sha256(SPLIT_MANIFEST),
            "preprocessing_configs_canonical_sha256": {r: canonical_sha256(p) for r, p in PREPROCESSING_CONFIGS.items()},
            "l2_records_sha256": canonical_sha256(L2_ROOT / "l2_calibration_records.json"),
            "torch_threads": os.environ.get("MECHASIGHT_TORCH_THREADS")}


def _ledger_rows() -> int:
    return len(verify_ledger_history().rows)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8", newline="\n")


def _write_csv(path: Path, frame: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    return canonical_sha256(path)


def _development_rows() -> pd.DataFrame:
    return l3.development_segment_rows(load_raw_catalog(DATA_ROOT), pd.read_csv(SPLIT_MANIFEST))


def _preflight(args) -> tuple[dict, Path]:
    if os.environ.get("MECHASIGHT_TORCH_THREADS") != TORCH_THREADS:
        raise SystemExit(f"refusing: MECHASIGHT_TORCH_THREADS must be {TORCH_THREADS} (Phase-4 reproducibility policy)")
    if canonical_sha256(DESIGN_CONFIG) != DESIGN_CONFIG_SHA256:
        raise SystemExit("refusing: the frozen design v2 config changed")
    official = args.out is None
    out = L3_ROOT if official else Path(args.out)
    if _ledger_rows() != LEDGER_ROWS:
        raise SystemExit("refusing: the test-look ledger is not at its 196-row state")
    return _identity(official), out


def cmd_clean_scan(args) -> int:
    identity, out = _preflight(args)
    started = datetime.now(timezone.utc).isoformat()
    rows = _development_rows()
    if args.segments:
        rows = rows.groupby("machine", group_keys=False).head(args.segments)
    config = l3.load_regime_configs(PREPROCESSING_CONFIGS)
    t0 = time.perf_counter()
    table, stats = falsification.clean_scan(rows, config)
    summary = falsification.clean_scan_summary(table)
    files = {"segment_channels.csv": _write_csv(out / "clean_scan" / "segment_channels.csv", table),
             "training_channel_stats.csv": _write_csv(out / "clean_scan" / "training_channel_stats.csv", stats)}
    if _ledger_rows() != LEDGER_ROWS:
        raise SystemExit("ledger changed during the clean scan")
    summary.update({"identity": identity, "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
                    "elapsed_s": time.perf_counter() - t0, "files": files, "test_evaluated": False,
                    "ledger_rows_before_after": [LEDGER_ROWS, _ledger_rows()],
                    "completion": "COMPLETE" if summary["accepted"] else "STOP_CLEAN_DATA_TRIGGERS"})
    _write_json(out / "clean_scan" / "clean_scan_summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("segments", "segment_channels", "triggers", "accepted",
                                              "low_margin_sensors", "completion")}, indent=2))
    return 0 if summary["accepted"] else 2


def cmd_falsify(args) -> int:
    identity, out = _preflight(args)
    summary_path = out / "clean_scan" / "clean_scan_summary.json"
    if not summary_path.is_file() or not json.loads(summary_path.read_text(encoding="utf-8"))["accepted"]:
        raise SystemExit("refusing: the clean scan must exist and be accepted before the falsification suite")
    stats = pd.read_csv(out / "clean_scan" / "training_channel_stats.csv", float_precision="round_trip")
    work_root = WORK_ROOT / ("official" if identity["official"] else "dry_run") / "corrupted"
    if work_root.exists() and any(work_root.iterdir()):
        raise SystemExit(f"refusing: {work_root} already holds corrupted copies; remove them before a fresh run")
    started = datetime.now(timezone.utc).isoformat()
    rows = _development_rows()
    config = l3.load_regime_configs(PREPROCESSING_CONFIGS)
    units = load_design_registry()
    records = l3.load_l2_records(L2_ROOT / "l2_calibration_records.json")
    persisted_fused = pd.read_csv(L2_ROOT / "l2_fused_validation_scores.csv", dtype={"segment_id": str},
                                  float_precision="round_trip")
    output = falsification.SuiteOutput()
    banks = {}
    t0 = time.perf_counter()
    for machine in args.machines:
        banks[machine] = l3.DetectorBank.load(machine, PHASE4_PRETEST_ROOT, PHASE4_MODEL_ROOT)
        counter = {"n": 0}

        def progress(entry, machine=machine):
            counter["n"] += 1
            if counter["n"] % 83 == 0 or entry.get("exception"):
                print(f"{machine}: {counter['n']} runs, {time.perf_counter() - t0:.0f} s"
                      + (f"  EXCEPTION {entry['run_id']}: {entry['exception']}" if entry.get("exception") else ""),
                      flush=True)

        falsification.run_machine_suite(machine, rows=rows, config=config, units=units, bank=banks[machine],
                                        records=records, stats=stats, work_root=work_root, output=output,
                                        progress=progress, segment_limit=args.segments)
    elapsed = time.perf_counter() - t0
    runs = pd.DataFrame(output.runs)
    decisions = pd.DataFrame(output.decisions)
    scores = pd.DataFrame(output.detector_scores)
    ungated = pd.DataFrame(output.ungated)
    expected_runs = 12 * 83 * len(args.machines) if not args.segments else None
    saturation = falsification.saturation_characterisation(runs, decisions)
    demo = (falsification.ungated_demonstration(ungated, banks=banks,
                                                units=[u for u in units if u["machine"] in args.machines],
                                                records=records, persisted_fused=persisted_fused)
            if not ungated.empty else [])
    base = out / "falsification"
    runs_out = runs.assign(corruption=runs["corruption"].map(lambda c: canonical_json(c) if isinstance(c, dict) else None))
    files = {
        "runs.csv": _write_csv(base / "runs.csv", runs_out),
        "unit_decisions.csv": _write_csv(base / "unit_decisions.csv", decisions),
        "detector_scores.csv": _write_csv(base / "detector_scores.csv", scores),
        "ungated_dead_input_scores.csv": _write_csv(base / "ungated_dead_input_scores.csv", ungated),
    }
    # Raw outputs are persisted before the assertions run, so a defect in the checks never loses a run.
    assertions = falsification.evaluate_assertions(
        runs, decisions, scores, banks=banks, units=[u for u in units if u["machine"] in args.machines],
        records=records, persisted_fused=persisted_fused) if not args.segments else {"skipped": "partial dry run"}
    _write_json(base / "assertions.json", assertions)
    _write_json(base / "saturation_characterisation.json", saturation)
    _write_json(base / "ungated_demonstration.json", demo)
    if _ledger_rows() != LEDGER_ROWS:
        raise SystemExit("ledger changed during the falsification suite")
    complete = (expected_runs is not None and len(runs) == expected_runs
                and assertions.get("all_pass_criteria_passed") is True)
    manifest = {
        "identity": identity, "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": elapsed, "machines": list(args.machines), "runs": int(len(runs)),
        "expected_runs": expected_runs, "unit_decision_rows": int(len(decisions)),
        "detector_score_rows": int(len(scores)),
        "selected_segments": {m: list(l3.selected_validation_segments(rows, m)) for m in args.machines},
        "scenarios": [{"scenario_id": s.scenario_id, "code": s.code, "targets": list(s.targets),
                       "expected_status": s.expected_status, "expected_verdicts": s.expected_verdicts}
                      for s in falsification.scenario_catalogue()],
        "corruption_parameters": falsification.CORRUPTION_PARAMETERS,
        "gate_parameters": {"flatline_min_run_s": l3.FLATLINE_MIN_RUN_S,
                            "flatline_invalid_fraction": l3.FLATLINE_INVALID_FRACTION,
                            "coverage_min_fraction": l3.COVERAGE_MIN_FRACTION,
                            "nominal_duration_s": l3.NOMINAL_DURATION_S},
        "corrupted_copies_root": work_root.as_posix(),
        "clean_scan_summary_sha256": canonical_sha256(summary_path),
        "files": files, "test_evaluated": False, "protected_inputs_read": False,
        "ledger_rows_before_after": [LEDGER_ROWS, _ledger_rows()],
        "completion": "COMPLETE" if complete else ("DRY_RUN" if not identity["official"] else "FAILED"),
        "limitations": [
            "synthetic corruptions are negative controls for gate behaviour, not physical fault evidence",
            "B5 (surviving sensors unaffected by the other failure) is an assumption the suite cannot test",
            "the ungated demonstration is in-sample on the validation cohort that set the thresholds",
            "partial flatlines below 0.50, gain error, drift, noise and timing jitter are not detected by design",
        ],
    }
    _write_json(base / "falsification_manifest.json", manifest)
    print(json.dumps({"runs": len(runs), "elapsed_s": round(elapsed), "completion": manifest["completion"],
                      **{k: assertions[k]["passed"] for k in ("E0", "E1", "E2", "E3", "E4", "E5", "E6")
                         if k in assertions}}, indent=2))
    return 0 if manifest["completion"] in ("COMPLETE", "DRY_RUN") else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("clean-scan", "falsify"):
        command = sub.add_parser(name)
        command.add_argument("--out", default=None, help="development dry-run output root (unofficial)")
        command.add_argument("--segments", type=int, default=None, help="dry run: limit segments per machine")
        if name == "falsify":
            command.add_argument("--machines", nargs="+", default=["RoboticArm", "BrushlessMotor"])
    args = parser.parse_args()
    if args.segments and args.out is None:
        raise SystemExit("refusing: --segments is a development dry-run option and needs --out")
    return {"clean-scan": cmd_clean_scan, "falsify": cmd_falsify}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
