"""Run the 232-unit synthetic demonstration and seal its candidate evidence.

Real execution is deliberately unavailable. Temporary fixture inputs/outputs
remain marked SYNTHETIC; committed evidence contains hashes and counts only.

``--verify-candidate <commit>`` (Phase 5.5C) checks the committed candidate
against an externally supplied reviewed commit and writes nothing.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment.phase5_execution import (
    CANDIDATE_SOURCES, CANDIDATE_STATUS, TRUSTED_COMMIT, Executor, FrozenPackage, authorize_fixture,
    source_identity, verify_executable_candidate,
)
from src.experiment.ledger_history import verify_ledger_history
from src.experiment.phase5_fixture import create_fixture
from src.experiment.phase5_inputs import SYNTHETIC, Sandbox
from src.experiment.provenance import canonical_sha256, file_sha256

OUTPUT = Path("reports/phase5_executable_freeze_candidate")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic-demo", action="store_true")
    mode.add_argument("--verify-candidate", metavar="REVIEWED_COMMIT",
                      help="read-only: verify the candidate against this externally reviewed commit")
    parser.add_argument("--test-report", type=Path, help="successful focused pytest JUnit XML in the OS temp directory")
    parser.add_argument("--regression-report", type=Path, help="successful bounded regression JUnit XML in the OS temp directory")
    args = parser.parse_args()
    if args.verify_candidate:
        manifest = verify_executable_candidate(args.verify_candidate)
        print(json.dumps({"verified_against": args.verify_candidate, "status": manifest["status"],
                          "executor_source_identity": manifest["executor_source_identity"],
                          "review_status_recorded": manifest["review_status"]}, ensure_ascii=False))
        return
    if args.test_report is None:
        parser.error("--synthetic-demo requires --test-report")
    test_path = args.test_report.resolve()
    if not test_path.is_relative_to(Path(tempfile.gettempdir()).resolve()):
        parser.error("test evidence must come from a temporary test-report path")
    test_xml = test_path.read_bytes()
    suites = list(ET.fromstring(test_xml).iter("testsuite"))
    outcomes = {key: sum(int(s.attrib.get(key, 0)) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
    if not outcomes["tests"] or outcomes["failures"] or outcomes["errors"]:
        parser.error("focused test report must contain successful tests and no failures/errors")
    if args.regression_report is None or not args.regression_report.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve()):
        parser.error("--synthetic-demo requires a temporary --regression-report")
    regression_xml = args.regression_report.read_bytes()
    regression = {k: sum(int(s.attrib.get(k, 0)) for s in ET.fromstring(regression_xml).iter("testsuite"))
                  for k in ("tests", "failures", "errors", "skipped")}
    if not regression["tests"] or regression["failures"] or regression["errors"]:
        parser.error("regression report must contain successful tests and no failures/errors")
    package = FrozenPackage.load()
    ledger_path = Path("reports/experiment_results/test_look_ledger.jsonl")
    ledger_before = file_sha256(ledger_path)
    expected_ledger = "f75584611c18ff387bee8c06f1fcc999c3046def876ba13dea1c744fce64de5f"
    if ledger_before != expected_ledger or len(verify_ledger_history().rows) != 196:
        parser.error("real ledger differs from the closed 196-row checkpoint")
    original_inventory = json.loads((Path("reports/phase5_freeze_candidate") / "integrity_inventory.json").read_text(encoding="utf-8"))
    for name, expected in original_inventory["all_ignored_data_and_reports_stat_only"].items():
        path = Path(name)
        stat = path.stat()
        if {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns} != expected:
            parser.error("ignored data/report stat identity changed")
    # tempfile creates a unique owned directory. No repository dataset path is
    # accepted. Keep the fixture for independent inspection; never delete a
    # computed directory through a second shell.
    sandbox = Sandbox(tempfile.mkdtemp(prefix="mechasight-SYNTHETIC-5-5b-"))
    fixture = create_fixture(sandbox, package)
    executor = Executor(sandbox, package)
    state = executor.score_all(authorize_fixture(sandbox, package, "score"))
    analysis = executor.analyse_once(authorize_fixture(sandbox, package, "analysis"))
    executor.verify_complete()
    integration = _integration_run(package)
    assert file_sha256(ledger_path) == ledger_before
    counts = Counter(p["operation"] for p in package.proposed["planned_units"])
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    clean = not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    primary = [r for r in analysis["intervals"] if r["contrast"].startswith("P") and r["metric"] in ("auroc", "recall", "healthy_target_fpr")]
    zero_width = sum(1 for r in primary if r["ci_lower"] is not None and r["ci_lower"] == r["ci_upper"])
    auroc_categories = Counter(r["category"] for r in primary if r["metric"] == "auroc")
    evidence = {"scope": SYNTHETIC, "not_real_performance": True, "fixture_root": str(sandbox.root),
                "executor_source_identity": source_identity(), "generated_from_head": head,
                "tracked_tree_clean_at_generation": clean,
                "provenance_note": ("generated before the 5.5G commit from the working tree whose executor source "
                                    "identity is recorded here; a source identity is not an independent review"),
                "statistics_non_degeneracy": {"primary_intervals": len(primary), "zero_width_primary_intervals": zero_width,
                                              "auroc_primary_categories": dict(auroc_categories)},
                "fixture_manifest_sha256": file_sha256(sandbox.path("fixture.json")), "recipe": fixture["recipe"],
                "units": len(state["tables"]), "operations": dict(counts), "synthetic_ledger_before_after": [196, len(sandbox.json("ledger.json"))],
                "analysis_primary_contrasts": analysis["primary_contrasts"], "analysis_primary_intervals": analysis["primary_intervals"],
                "analysis_all_intervals": len(analysis["intervals"]), "replicates": analysis["replicates"], "bootstrap_seed": analysis["seed"],
                "analysis_sha256": file_sha256(sandbox.path("analysis.json")),
                "temporary_artifacts_sha256": {p.relative_to(sandbox.root).as_posix(): file_sha256(p)
                                               for p in sandbox.root.rglob("*") if p.is_file()},
                "real_ledger_sha256": ledger_before, "real_ledger_rows": len(ledger_path.read_bytes().splitlines()),
                "real_execution_enabled": False, "original_candidate_rewritten": False,
                "focused_test_results": outcomes, "focused_test_report_sha256": file_sha256(test_path),
                "development_artifact_hashes_verified": len(original_inventory["development_and_model_canonical_sha256"]),
                "ignored_stat_identities_verified": len(original_inventory["all_ignored_data_and_reports_stat_only"]),
                "environment": {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "h5py", "psutil", "pytest")}}}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    evidence_path = OUTPUT / "synthetic_evidence.json"
    (OUTPUT / "focused_tests.xml").write_bytes(test_xml)
    (OUTPUT / "regression_tests.xml").write_bytes(regression_xml)
    evidence["regression_tests"] = regression
    evidence["predictor_provenance_coverage"] = _provenance_coverage(sandbox, state)
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    integration_path = OUTPUT / "integration_evidence.json"
    integration_path.write_text(json.dumps(integration, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    manifest = {"status": CANDIDATE_STATUS, "stage": "5.5G",
                "review_status": "UNREVIEWED: becomes trusted only when an independent review names the commit holding this manifest",
                "real_execution_enabled": False, "authorization_issued": False, "independently_approved": False,
                "trusted_package_commit": TRUSTED_COMMIT, "trusted_package_manifest_sha256": package.identity,
                "original_scientific_registry_sha256": package.proposed["identity"]["registry_sha256"],
                "executor_source_identity": source_identity(),
                "sources": {p: canonical_sha256(p) for p in CANDIDATE_SOURCES},
                "files": {"synthetic_evidence.json": canonical_sha256(evidence_path),
                          "integration_evidence.json": canonical_sha256(integration_path),
                          "regression_tests.xml": canonical_sha256(OUTPUT / "regression_tests.xml"),
                          "development_sources.json": canonical_sha256(OUTPUT / "development_sources.json"),
                          "focused_tests.xml": canonical_sha256(OUTPUT / "focused_tests.xml")},
                "generated_from_head": head, "tracked_tree_clean_at_generation": clean,
                "proposed_real_looks": 232, "real_ledger_rows": 196,
                "interfaces": integration["interfaces"], "coverage": integration["coverage"],
                "restrictions": ["SYNTHETIC/development temporary sandboxes only; no real scope, token or protected input",
                    "Read capabilities are issued and verified only by the owning executor; caller-built Access is refused",
                    "Projected parquet columns and numeric HDF5 only; label-bearing cache provenance is never deserialized",
                    "Label-free window index implemented and verified on development caches; protected index construction DISABLED",
                    "Frozen L1 inference adapter verified on permitted source-validation inputs only",
                    "JSONL ledger adapter verified on a temporary copy only; the real ledger is never an executor backend",
                    "Interrupted writes resolve only via the audited incident action; interrupted analysis needs a human incident",
                    "Retries add separately accounted looks: 196 -> 428 holds only for an uninterrupted evaluation"]}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                                          encoding="utf-8", newline="\n")
    print(json.dumps({k: evidence[k] for k in ("scope", "units", "operations", "synthetic_ledger_before_after", "analysis_primary_intervals", "real_execution_enabled")}))


def _integration_run(package):
    """Phase 5.5F: real frozen models + label-free index + JSONL ledger on permitted development inputs."""
    import pandas as pd
    from src.experiment.ledger_history import verify_ledger_history
    from src.experiment.phase5_dev_fixture import DEV_RECIPE, create_dev_fixture
    from src.experiment.phase5_inference import ADAPTER_VERSION, FrozenL1Adapter
    from src.experiment.phase5_inputs import DEV_SCHEMA
    from src.experiment.phase5_window_index import INDEX_COLUMNS, INDEX_SCHEMA_VERSION

    box = Sandbox(tempfile.mkdtemp(prefix="mechasight-SYNTHETIC-5-5f-integration-"))
    fixture = create_dev_fixture(box, package)
    adapter = FrozenL1Adapter(package)
    executor = Executor(box, package, predictor=adapter)
    state = executor.score_all(authorize_fixture(box, package, "score"))
    analysis = executor.analyse_once(authorize_fixture(box, package, "analysis"))
    executor.verify_complete()
    l1_diff = 0.0
    for call in adapter.calls:
        got = pd.DataFrame(box.json(state["tables"][call["experiment_id"]]["file"])["rows"]).set_index("segment_id")["score"]
        ref = pd.read_csv(Path(package.registry["detectors"][call["experiment_id"]]["bundle_path"]) / "validation_scores.csv",
                          dtype={"segment_id": str}, float_precision="round_trip").set_index("segment_id")["score"]
        l1_diff = max(l1_diff, float((got - ref.loc[got.index]).abs().max()))
    fused = pd.read_csv("reports/phase5_development/l2/l2_fused_validation_scores.csv", dtype={"segment_id": str},
                        float_precision="round_trip").set_index(["calibration_id", "segment_id"])["fused_score"]
    l2_diff, l2_rows = 0.0, 0
    for eid, op in executor.operations.items():
        if op["operation"] == "derived_L2_scoring":
            for row in box.json(state["tables"][eid]["file"])["rows"]:
                l2_diff, l2_rows = max(l2_diff, abs(row["score"] - fused[(eid, row["segment_id"])])), l2_rows + 1
    ledger_path, auth = box.path("ledger/test_look_ledger.jsonl"), box.path("ledger/authorizations")
    history = verify_ledger_history(ledger_path, authorization_root=auth)
    families = Counter(c["family"] for c in adapter.calls)
    return {"scope": SYNTHETIC, "not_real_performance": True, "fixture_root": str(box.root), "recipe": DEV_RECIPE,
            "data_origin": fixture["data_origin"], "fixture_manifest_sha256": file_sha256(box.path("fixture.json")),
            "cohort_segments": {m: len(c) for m, c in fixture["cohorts"].items()},
            "units": len(state["tables"]),
            "ledger": {"backend": "jsonl (TestLookLedger.append_many_atomic + batch_authorization.json)",
                       "rows_before_after": [196, len(history.rows)], "verified_with": "verify_ledger_history (unmodified)",
                       "appended_segments": [s.stage for s in history.segments if not s.closed]},
            "l1_real_model_units": len(adapter.calls), "l1_max_abs_diff_vs_frozen_bundle_validation_scores": l1_diff,
            "l2_rows": l2_rows, "l2_max_abs_diff_vs_persisted_l2_fused_validation": l2_diff,
            "analysis_primary_contrasts": analysis["primary_contrasts"], "analysis_primary_intervals": analysis["primary_intervals"],
            "analysis_sha256": file_sha256(box.path("analysis.json")),
            "interfaces": {"input_schema": DEV_SCHEMA, "window_index_schema": INDEX_SCHEMA_VERSION,
                           "window_index_columns": list(INDEX_COLUMNS), "inference_adapter": ADAPTER_VERSION},
            "coverage": {"l1_new_units_real_frozen_model": len(adapter.calls), "l1_new_units_stub": 0,
                         "l1_families": dict(families),
                         "l2_derived_from_real_phase4_validation_scores": 208,
                         "reused_l1_comparators_real_phase3_validation_scores_analysis_only": 28,
                         "labels": "SYNTHETIC fake labels (development validation has no faults)",
                         "synthetic_stub_demo_units": 232},
            "predictor_provenance_coverage": _provenance_coverage(box, state),
            "index_bypass_rejection": _index_rejection_evidence()}


def _provenance_coverage(box, state):
    records = [box.json(t["file"])["computation"] for t in state["tables"].values()]
    return {"kinds": dict(Counter(r["kind"] for r in records)),
            "implementations": dict(Counter(r["implementation"] for r in records)),
            "l2_component_origins": dict(Counter(r["component_origin"] for r in records if "component_origin" in r))}


def _index_rejection_evidence():
    """Repeat the original counterexample with sensor I/O tripwires, not protected files."""
    from unittest.mock import patch
    from src.experiment import phase5_window_index as wi
    from src.fusion.assembly import PREPROCESSING_CONFIGS
    from src.preprocessing.config import load_config
    source = wi.SegmentSource("RoboticArm", "SYNTHETIC-unregistered-test-segment", "SYNTHETIC-parent", "target",
                              {m: f"SYNTHETIC-UNOPENED/{m}.parquet" for m in ("mic", "acc", "gyro")})
    config = load_config(PREPROCESSING_CONFIGS["S"])
    rejected = []
    with patch.object(wi.pq, "read_schema", side_effect=AssertionError("unauthorized Parquet open")) as schema, \
         patch.object(wi.pd, "read_parquet", side_effect=AssertionError("unauthorized timestamp read")) as read:
        for name, call in (("generic", lambda: wi.build_window_index([source], config, regime="S")),
                           ("segment", lambda: wi.segment_windows(source, config)),
                           ("timestamps", lambda: wi._timestamps(source.paths["mic"], "mic", source=source, config=config))):
            try:
                call()
            except wi.IndexRefused:
                rejected.append(name)
            else:
                raise AssertionError("original index bypass remains reproducible")
        return {"counterexample": "unregistered target-domain SegmentSource", "rejected_entry_points": rejected,
                "parquet_schema_opens": schema.call_count, "timestamp_reads": read.call_count,
                "source_content_hashes": 0, "source_hash_note": "index construction never content-hashes raw sensors",
                "development_manifest_sha256": wi.DEVELOPMENT_SOURCES_SHA256}


if __name__ == "__main__":
    main()
