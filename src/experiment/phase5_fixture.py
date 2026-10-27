"""Small SYNTHETIC-only fixture recipe. Never reads any MechaSight data."""
from __future__ import annotations

import h5py
import numpy as np
import pandas as pd

from src.experiment.phase5_inputs import SYNTHETIC, Sandbox, atomic_json
from src.experiment.provenance import content_sha256, file_sha256

NORMAL = "SYNTHETIC_NORMAL_SECRET_DO_NOT_LOAD_DURING_SCORING"
ANOMALY = "SYNTHETIC_FAULT_SECRET_DO_NOT_LOAD_DURING_SCORING"
# v2 (Phase 5.5C): fake scores OVERLAP between the fake classes (v1 was perfectly
# separable, which made every AUROC contrast trivially equivalent). Segments 0-5 are
# fake normals, 6-11 fake faults; two fake faults score below some fake normals.
OVERLAPPING_BASE = (0.10, 0.35, 0.20, 0.55, 0.30, 0.45, 0.40, 0.25, 0.65, 0.50, 0.80, 0.60)


def create_fixture(sandbox: Sandbox, package):
    """All 52 actual unit identities; 12 fake segments per machine; no real models."""
    manifest = {"scope": SYNTHETIC, "recipe": "phase5-fixture-v2-overlapping", "files": {}, "cells": {}, "cohorts": {},
                "historical": {}, "labels": {}, "package_identity": package.identity}

    def parquet(name, frame):
        path = sandbox.path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        manifest["files"][name] = file_sha256(path)

    for machine in ("RoboticArm", "BrushlessMotor"):
        cohort = []
        for i in range(12):
            cohort.append({"segment_id": f"SYNTHETIC-{machine}-{i:02}",
                           "parent_recording": f"SYNTHETIC-parent-{machine}-{i // 2 if machine == 'RoboticArm' else i:02}",
                           "domain": "target" if i in (0, 1, 6, 7) else "source"})
        manifest["cohorts"][machine] = cohort
        labels = pd.DataFrame(cohort)
        labels["fake_label"] = [NORMAL if i < 6 else ANOMALY for i in range(12)]
        name = f"inputs/{machine}/labels.parquet"
        parquet(name, labels)
        manifest["labels"][machine] = name
        for regime in ("S", "T"):
            cell = machine + "|" + regime
            frame = labels.copy()
            x = np.asarray(OVERLAPPING_BASE) + (.01 if regime == "T" else 0)
            frame["x0"], frame["x1"] = x, x + .2
            features = f"inputs/{machine}/{regime}/features.parquet"
            parquet(features, frame)
            index = frame.loc[frame.index.repeat(2)].reset_index(drop=True)
            index["window_index"] = np.tile([0, 1], 12)
            index_name = f"inputs/{machine}/{regime}/window_index.parquet"
            parquet(index_name, index)
            spectral = f"inputs/{machine}/{regime}/numeric.h5"
            with h5py.File(sandbox.path(spectral), "w") as handle:
                for k, mod in enumerate(("mic", "acc", "gyro")):
                    handle.create_dataset(mod, data=np.repeat(np.stack([x, x + .2], axis=1), 2, axis=0) + k * .1)
                # Poison outcome-bearing datasets exist but must NEVER be read.
                handle.create_dataset("provenance_json", data=("[\"" + ANOMALY + "\"]").encode())
                handle.create_dataset("fake_labels", data=np.array([ANOMALY.encode()] * 24))
            manifest["files"][spectral] = file_sha256(sandbox.path(spectral))
            manifest["cells"][cell] = {"features": features, "index": index_name, "spectral": spectral}
    historical = {eid for u in package.registry["units"] for eid in u["L0"].values()}
    historical |= {u["L1"]["experiment_id"] for u in package.registry["units"] if u["L1"]["source"] != "new_fit_phase5"}
    for eid in sorted(historical):
        d = package.registry["detectors"][eid]
        frame = pd.DataFrame(manifest["cohorts"][d["machine"]])
        # Identity-specific but deterministic fake scores, independent of labels. The
        # per-identity jitter gives each fake detector its own ranking, so paired
        # contrasts between arms have genuine (nonzero) bootstrap uncertainty.
        offset = int(content_sha256(eid)[:6], 16) % 11 / 100
        jitter = np.random.default_rng(int(content_sha256(eid)[:12], 16)).normal(0.0, 0.08, size=12)
        frame["score"] = np.asarray(OVERLAPPING_BASE) + offset + jitter
        frame["window_count"] = 2
        frame["fake_label"] = [NORMAL if i < 6 else ANOMALY for i in range(12)]
        name = f"historical/{eid}.parquet"
        parquet(name, frame)
        manifest["historical"][eid] = name
    atomic_json(sandbox.path("fixture.json"), manifest)
    atomic_json(sandbox.path("ledger.json"), [{"scope": SYNTHETIC, "experiment_id": f"SYNTHETIC-HISTORY-{i}"} for i in range(196)])
    return manifest
