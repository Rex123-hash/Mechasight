"""Phase 5.6C: development mirror of the protected-style executor path (SYNTHETIC labels).

The mirror registers the four permitted development source-validation cells as if
they were the protected common-evaluation inputs and runs them through the same
code the protected path would use:

* cells carry only the identity of the trusted 5.6B input manifest; the reader
  rebuilds that manifest from Git-pinned frozen metadata and resolves every path,
  stat and content hash from it (the fixture names no feature or spectral path);
* classical inputs are the trusted 98-feature column projection; spectral inputs
  are numeric modality datasets only, aligned by the count-derived label-free index;
* all of it happens only after the batch's ledger append and exposure record;
* historical L0 tables (and the 28 analysis-only reused L1 comparators) are the
  frozen bundles' development validation scores, projected as in Phase 5.5F;
* labels are SYNTHETIC parent-homogeneous fake labels — development validation has
  no faults, so the analysis is wiring evidence, never performance;
* the ledger is a temporary copy of the closed 196-row history.

No protected recording, cache, feature artifact, label or sealed score is opened.
"""
from __future__ import annotations

from pathlib import Path
import subprocess

import pandas as pd

from src.experiment.phase5_dev_fixture import POISON_COLUMN, fake_labels
from src.experiment.phase5_fixture import ANOMALY
from src.experiment.phase5_input_manifest import development_manifest, read_features
from src.experiment.phase5_inputs import IDENTITY_COLUMNS, MIRROR_SCHEMA, SCORE_COLUMNS, SYNTHETIC, atomic_json, require
from src.experiment.phase5_ledger import seed_jsonl_ledger
from src.experiment.provenance import file_sha256

MIRROR_RECIPE = "phase5-protected-style-dev-mirror-v1"


def create_mirror_fixture(sandbox, package):
    """Write the mirror fixture into ``sandbox``; returns its manifest."""
    manifest = {"scope": SYNTHETIC, "schema": MIRROR_SCHEMA, "recipe": MIRROR_RECIPE,
                "data_origin": ("PERMITTED development source-validation cohorts registered as a protected-style mirror; "
                                "SYNTHETIC fake labels"),
                "files": {}, "cells": {}, "cohorts": {}, "historical": {}, "labels": {},
                "package_identity": package.identity, "real_model_inference": True}

    def put(name, frame):
        path = sandbox.path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        manifest["files"][name] = file_sha256(path)

    window_counts = {}
    for machine in ("RoboticArm", "BrushlessMotor"):
        cohort = None
        for regime in ("S", "T"):
            trusted = development_manifest(package, machine=machine, regime=regime)
            segments = pd.DataFrame(trusted.snapshot()["segments"]).loc[:, list(IDENTITY_COLUMNS)]
            segments = segments.sort_values("segment_id").reset_index(drop=True)
            require(cohort is None or segments.equals(cohort), f"{machine}: S/T development cohorts differ")
            cohort = segments
            # Trusted count projection (identities + window_count only), for comparators lacking the column.
            window_counts[(machine, regime)] = read_features(trusted).set_index("segment_id")["window_count"]
            # Metadata only: the identity of a manifest the reader must independently rebuild.
            manifest["cells"][f"{machine}|{regime}"] = {"input_manifest_identity": trusted.identity}
        manifest["cohorts"][machine] = cohort.to_dict(orient="records")
        put(f"inputs/{machine}/labels.parquet", cohort.assign(fake_label=fake_labels(cohort)))
        manifest["labels"][machine] = f"inputs/{machine}/labels.parquet"

    detectors = package.registry["detectors"]
    for unit in package.registry["units"]:
        machine = unit["machine"]
        keep = {r["segment_id"] for r in manifest["cohorts"][machine]}
        reused = unit["L1"]["source"] != "new_fit_phase5"
        for eid in list(unit["L0"].values()) + ([unit["L1"]["experiment_id"]] if reused else []):
            if eid in manifest["historical"]:
                continue
            bundle = Path(detectors[eid]["bundle_path"]) / "validation_scores.csv"
            columns = [c for c in SCORE_COLUMNS if c in pd.read_csv(bundle, nrows=0).columns]
            table = pd.read_csv(bundle, usecols=columns, dtype={"segment_id": str, "parent_recording": str},
                                float_precision="round_trip")
            table = table[table["segment_id"].isin(keep)].sort_values("segment_id").reset_index(drop=True)
            if "window_count" not in table:
                table["window_count"] = table["segment_id"].map(window_counts[(machine, unit["regime"])]).astype(int)
            require(len(table) == len(keep), f"historical table {eid} lacks mirror cohort segments")
            name = f"historical/{eid}.parquet"
            put(name, table.loc[:, list(SCORE_COLUMNS)].assign(**{POISON_COLUMN: ANOMALY}))
            manifest["historical"][eid] = name

    manifest["ledger"] = {**seed_jsonl_ledger(sandbox),
                          "code_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    atomic_json(sandbox.path("fixture.json"), manifest)
    return manifest
