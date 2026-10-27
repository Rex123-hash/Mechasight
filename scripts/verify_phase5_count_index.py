"""5.6B development evidence and additive candidate binding; never protected execution."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.experiment.phase5_execution import CANDIDATE_SOURCES, EXECUTOR_SOURCES, FrozenPackage
from src.experiment.phase5_input_manifest import (
    ROOT, MODALITIES, development_manifest, protected_manifest_template, read_features, read_spectral,
)
from src.experiment.phase5_count_index import build_count_derived_index, inference_index
from src.experiment.phase5_inference import FrozenL1Adapter
from src.experiment.phase5_inputs import require
from src.experiment.provenance import canonical_bytes, canonical_sha256, content_sha256, file_sha256

BASELINE = "5da28f82b94cdafafe33ac235dcea08d16e31692"
START = "fc22821de81954c46ff503b244fd8b086564603b"
OUTPUT = Path("reports/phase5_6b_candidate")
NEW_SOURCES = ("src/experiment/phase5_input_manifest.py", "src/experiment/phase5_count_index.py",
               "scripts/verify_phase5_count_index.py", "tests/test_phase5_count_index.py",
               "docs/phase5_6b_count_derived_index.md", "docs/phase5_6_protected_evaluation_readiness.md")
SOURCES = tuple(sorted(set(CANDIDATE_SOURCES + NEW_SOURCES)))
FILES = ("evidence.json", "focused_tests.xml", "regression_tests.xml")


def component_identity():
    return content_sha256({p: canonical_sha256(p) for p in NEW_SOURCES[:3]})


def development_evidence(package):
    """Validation only: original timestamp builder on permitted development recordings.

    The count builder itself never invokes the timestamp builder or raw readers.
    Numeric model scoring below uses the unchanged adapter, without fitting/metrics.
    """
    from src.experiment import phase5_window_index as old
    from src.fusion.assembly import PREPROCESSING_CONFIGS
    from src.preprocessing.config import load_config
    before = component_identity()
    cells, models, windows = {}, [], 0
    adapter = FrozenL1Adapter(package)
    allowed = old._development_contract()["sources"]
    for machine in ("RoboticArm", "BrushlessMotor"):
        for regime in ("S", "T"):
            manifest = development_manifest(package, machine=machine, regime=regime)
            bundle = build_count_derived_index(manifest)
            source_rows = [r for r in allowed if (r["machine"], r["regime"]) == (machine, regime)]
            sources = [old.SegmentSource(machine, r["segment_id"], r["parent_recording"], r["domain"],
                         {m: str(ROOT / p) for m, p in r["paths"].items()}) for r in source_rows]
            timestamp = old.build_window_index(sources, load_config(PREPROCESSING_CONFIGS[regime]), regime=regime)
            columns = ["machine", "regime", "segment_id", "parent_recording", "domain", "window_index", "row"]
            require(bundle.rows[columns].equals(timestamp[columns]), "count/timestamp alignment discrepancy")
            for _, part in timestamp.groupby("segment_id", sort=False):
                counted = bundle.rows[bundle.rows.segment_id.eq(part.segment_id.iloc[0])]
                require((counted.segment_row_offset == int(part.row.iloc[0])).all()
                        and (counted.window_count == len(part)).all(), "count-derived cumulative offset discrepancy")
            classical = read_features(manifest, classical=True)
            layout, arrays = read_spectral(manifest, values=True)
            index = inference_index(bundle, manifest)
            require(all(layout["modalities"][m]["rows"] == len(index) for m in MODALITIES), "feature/spectral mismatch")
            doc = manifest.snapshot()
            for eid in doc["inference_experiment_ids"]:
                detector = package.registry["detectors"][eid]
                got = adapter(detector, arrays, index) if detector["family"] == "dense_autoencoder" else adapter(detector, classical)
                ref = pd.read_csv(Path(detector["bundle_path"]) / "validation_scores.csv",
                                  usecols=["segment_id", "score", "window_count"], dtype={"segment_id": str}, float_precision="round_trip")
                got, ref = got.set_index("segment_id"), ref.set_index("segment_id")
                require(got.index.is_unique and set(got.index) == set(ref.index), "model output segment mismatch")
                difference = float(np.max(np.abs(got.score - ref.loc[got.index, "score"])))
                require(difference == 0. and (got.window_count == ref.loc[got.index, "window_count"]).all(),
                        "frozen-model development reproduction discrepancy")
                models.append({"experiment_id": eid, "family": detector["family"], "seed": detector["seed"],
                               "machine": machine, "regime": regime, "segments": len(got), "max_abs_score_difference": difference})
            cells[f"{machine}|{regime}"] = {"windows": len(bundle.rows), "segments": bundle.descriptor["segments"],
                "exact_segment_parent_domain_ordinal_row_offset_alignment": True,
                "all_modality_row_counts_match": True, "index_content_sha256": bundle.descriptor["content_sha256"],
                "manifest_identity": manifest.identity, "physical_times_reconstructed": False}
            windows += len(bundle.rows)
            del arrays, timestamp, bundle, classical, index
    require(windows == 40380 and len(models) == len(adapter.calls) == 24, "development coverage mismatch")
    require(component_identity() == before, "component source changed during evidence generation")
    return {"scope": "PERMITTED_DEVELOPMENT_ONLY", "not_protected_performance": True, "cells": cells,
            "windows": windows, "real_frozen_models": models, "stub_model_calls": 0,
            "max_abs_score_difference": max(r["max_abs_score_difference"] for r in models),
            "model_input_segments": sum(r["segments"] for r in models),
            "protected_artifact_bindings": "UNRESOLVED; no protected artifact metadata/content inspected",
            "protected_reader_enabled": False, "human_decisions_H1_H7": "UNRESOLVED",
            "real_ledger_modified": False, "physical_timestamps_reconstructed": False,
            "component_source_identity": before}


def _junit(path):
    require(path.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve()), "test report must be temporary")
    tree = ET.fromstring(path.read_bytes())
    counts = {k: sum(int(s.attrib.get(k, 0)) for s in tree.iter("testsuite")) for k in ("tests", "failures", "errors", "skipped")}
    require(counts["tests"] and not counts["failures"] and not counts["errors"], "test evidence has failures/errors")
    raw = ET.tostring(tree, encoding="utf-8", xml_declaration=False)
    require(b"\r" not in raw and b"\n" not in raw, "JUnit must use portable single-line form")
    return raw, counts


def unauthorized_evidence(package):
    """Actual refusal paths with fake I/O hooks; no protected-like contents used."""
    from unittest.mock import patch
    from src.experiment import phase5_input_manifest as mi
    from src.experiment.phase5_count_index import build_protected_count_index, _expand_count_rows
    from src.experiment.phase5_inputs import Refused
    protected = protected_manifest_template(package, machine="RoboticArm", regime="S")
    development = development_manifest(package, machine="RoboticArm", regime="S")
    fake = protected.snapshot()
    fake["scope"], fake["activation"] = mi.DEVELOPMENT, "DEVELOPMENT_ONLY"
    fake_development = mi._seal(fake)
    calls = {"protected_builder": lambda: build_protected_count_index(protected, authorized=True),
             "generic_builder": lambda: build_count_derived_index(protected),
             "feature_reader": lambda: mi.read_features(protected),
             "spectral_reader": lambda: mi.read_spectral(protected),
             "lower_path": lambda: mi._checked_path(protected, "features"),
             "lower_draft": lambda: _expand_count_rows(protected, pd.DataFrame(), {}),
             "fake_development": lambda: build_count_derived_index(fake_development),
             "substituted_path": lambda: mi.read_features(development, path=ROOT / "SYNTHETIC-UNOPENED/protected-like.parquet"),
             "unregistered_metadata": lambda: mi._frozen_bytes("reports/SYNTHETIC-protected-like.csv")}
    rejected = []
    with patch.object(mi.pd, "read_parquet", side_effect=AssertionError("unauthorized Parquet read")) as parquet, \
         patch.object(mi.h5py, "File", side_effect=AssertionError("unauthorized HDF5 open")) as hdf5, \
         patch.object(mi, "file_sha256", side_effect=AssertionError("unauthorized artifact hash")) as hashed:
        for name, call in calls.items():
            try:
                call()
            except Refused:
                rejected.append(name)
            else:
                raise AssertionError("unauthorized request accepted")
        return {"rejected": rejected, "parquet_reads": parquet.call_count, "hdf5_opens": hdf5.call_count,
                "artifact_content_hashes": hashed.call_count, "actual_protected_paths_used": False}


def write_candidate(focused, regression):
    package = FrozenPackage.load()
    tests, test_counts = _junit(focused)
    regressions, regression_counts = _junit(regression)
    ledger = Path("reports/experiment_results/test_look_ledger.jsonl")
    ledger_sha = file_sha256(ledger)
    require(ledger_sha == "f75584611c18ff387bee8c06f1fcc999c3046def876ba13dea1c744fce64de5f"
            and len(ledger.read_bytes().splitlines()) == 196, "real ledger changed")
    evidence = development_evidence(package)
    evidence["unauthorized_refusals"] = unauthorized_evidence(package)
    evidence.update(focused_tests=test_counts, regression_tests=regression_counts, real_ledger_rows=196, real_ledger_sha256=ledger_sha)
    require(file_sha256(ledger) == ledger_sha, "real ledger changed during validation")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / FILES[0]).write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    (OUTPUT / FILES[1]).write_bytes(tests)
    (OUTPUT / FILES[2]).write_bytes(regressions)
    manifest = {"status": "UNREVIEWED — NOT AUTHORIZED FOR REAL PROTECTED EVALUATION.", "stage": "5.6B",
                "accepted_baseline": BASELINE, "starting_checkpoint": START, "real_execution_enabled": False,
                "independently_approved": False, "authorization_issued": False,
                "executor_source_identity": content_sha256({p: canonical_sha256(p) for p in EXECUTOR_SOURCES}),
                "component_source_identity": component_identity(), "scientific_package_identity": package.identity,
                "sources": {p: canonical_sha256(p) for p in SOURCES},
                "files": {p: canonical_sha256(OUTPUT / p) for p in FILES},
                "generated_from_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "tracked_tree_clean_at_generation": False,
                "note": "Additive component candidate. Accepted Phase 5.5 evidence is retained as historical evidence, not regenerated or reapproved."}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"windows": evidence["windows"], "models": len(evidence["real_frozen_models"]),
                      "max_abs_score_difference": evidence["max_abs_score_difference"], "component_identity": component_identity()}))


def verify_candidate(commit):
    import re
    require(isinstance(commit, str) and re.fullmatch("[0-9a-f]{40}", commit), "full external commit SHA required")
    def git(revision, path):
        return subprocess.run(["git", "show", f"{revision}:{path}"], capture_output=True, check=True).stdout.replace(b"\r\n", b"\n")
    trusted = git(commit, OUTPUT.as_posix() + "/manifest.json")
    require(canonical_bytes(OUTPUT / "manifest.json") == trusted, "candidate differs from named commit")
    manifest = json.loads(trusted)
    require(manifest["status"] == "UNREVIEWED — NOT AUTHORIZED FOR REAL PROTECTED EVALUATION."
            and manifest["real_execution_enabled"] is False and manifest["authorization_issued"] is False
            and manifest["independently_approved"] is False and manifest["accepted_baseline"] == BASELINE,
            "candidate status/authority mismatch")
    require(manifest["stage"] == "5.6B" and manifest["starting_checkpoint"] == START, "candidate stage/checkpoint mismatch")
    require(set(manifest["sources"]) == set(SOURCES) and set(manifest["files"]) == set(FILES), "candidate inventory mismatch")
    for path, digest in manifest["sources"].items():
        require(sha256(git(commit, path)).hexdigest() == digest == canonical_sha256(path), "candidate source mismatch")
    for path, digest in manifest["files"].items():
        require(sha256(git(commit, f"{OUTPUT.as_posix()}/{path}")).hexdigest() == digest == canonical_sha256(OUTPUT / path),
                "candidate evidence mismatch")
    baseline = json.loads(git(BASELINE, "reports/phase5_executable_freeze_candidate/manifest.json"))
    require(all(canonical_sha256(p) == baseline["sources"][p] for p in EXECUTOR_SOURCES), "accepted executor code changed")
    require(manifest["executor_source_identity"] == baseline["executor_source_identity"], "accepted executor identity mismatch")
    require(manifest["component_source_identity"] == component_identity(), "component identity mismatch")
    evidence = json.loads((OUTPUT / "evidence.json").read_text(encoding="utf-8"))
    require(evidence["component_source_identity"] == component_identity(), "evidence came from another component")
    require(FrozenPackage.load().identity == manifest["scientific_package_identity"], "scientific package mismatch")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write-candidate", action="store_true")
    mode.add_argument("--verify-candidate")
    parser.add_argument("--focused-report", type=Path)
    parser.add_argument("--regression-report", type=Path)
    args = parser.parse_args()
    if args.verify_candidate:
        m = verify_candidate(args.verify_candidate)
        print(json.dumps({"verified_commit": args.verify_candidate, "status": m["status"], "component_identity": m["component_source_identity"]}))
    else:
        require(args.focused_report is not None and args.regression_report is not None, "focused and regression reports required")
        write_candidate(args.focused_report, args.regression_report)
