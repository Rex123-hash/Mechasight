"""Phase 5.5F: development integration fixture (permitted source-validation data, SYNTHETIC labels).

Builds a temporary sandbox the Phase-5 executor can run end to end with the real
frozen models, using only permitted development inputs:

* cohort: 12 source-validation segments per machine (RoboticArm: 2 per recording
  parent for its 6 parents; BrushlessMotor: the deterministic 12-of-126 rule);
* L1 classical inputs: the Phase-3.1 source-validation feature artifact, written
  with exactly the identity columns, ``window_count`` and the frozen 98 features
  (plus an injected poison label column the scoring projection must never request);
* L1 spectral inputs: the selected windows' numeric float32 rows of the Phase-4
  source-validation spectral cache, located with the label-free window index
  (built from raw timestamps) — the cache's label-bearing ``provenance_json`` is
  never read here; a poison provenance dataset is planted in the sandbox copy;
* historical tables: the Phase-4 single-modality and reused Phase-3.2 L1
  source-validation scores of the same segments (identity, score, window count);
* labels: SYNTHETIC fake labels (parent-homogeneous), because development
  validation contains no faults — analysis results are engineering evidence only;
* ledger: a temporary copy of the closed 196-row JSONL history, extended only by
  the executor's authorized synthetic batches.

No protected input, cache, label or sealed score is opened.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
import pandas as pd

from src.experiment.phase5_fixture import ANOMALY, NORMAL
from src.experiment.phase5_inputs import DEV_SCHEMA, IDENTITY_COLUMNS, SCORE_COLUMNS, SYNTHETIC, atomic_json, require
from src.experiment.phase5_ledger import seed_jsonl_ledger
from src.experiment.phase5_window_index import (
    SegmentSource, build_window_index, development_catalog, index_manifest, label_free_cohort,
)
from src.experiment.provenance import file_sha256
from src.fusion.assembly import PREPROCESSING_CONFIGS, SPECTRAL_DIMS
from src.preprocessing.config import load_config
from src.preprocessing.spectral_cache import spectral_cache_path

DEV_RECIPE = "phase5-dev-integration-v1"
SPLIT_MANIFEST = Path("reports/phase2_2/protocol_b_few_shot_adaptation_v3.csv")
DATA_ROOT = Path("data/raw/extracted")
FEATURE_ROOT = Path("data/interim/phase31_primary")
SPECTRAL_ROOT = Path("data/interim/phase4/spectral")
POISON_COLUMN = "fake_label_poison"


def select_cohort(cohort: pd.DataFrame, machine: str) -> pd.DataFrame:
    ordered = cohort.sort_values(["parent_recording", "segment_id"]).reset_index(drop=True)
    if machine == "RoboticArm":
        picked = ordered.groupby("parent_recording", sort=True).head(2)
    else:
        ids = sorted(ordered["segment_id"])
        chosen = {ids[int(round(k * (len(ids) - 1) / 11))] for k in range(12)}
        picked = ordered[ordered["segment_id"].isin(chosen)]
    require(len(picked) == 12, f"{machine}: development cohort selection must give 12 segments")
    return picked.sort_values("segment_id").reset_index(drop=True)


def fake_labels(cohort: pd.DataFrame) -> list[str]:
    """Parent-homogeneous SYNTHETIC labels: the first half of the parents fake-normal."""
    parents = sorted(cohort["parent_recording"].unique())
    normal = set(parents[: len(parents) // 2])
    return [NORMAL if p in normal else ANOMALY for p in cohort["parent_recording"]]


def _sources(cohort: pd.DataFrame, catalog: pd.DataFrame) -> list[SegmentSource]:
    paths = cohort.merge(catalog, on=["machine", "segment_id"], validate="one_to_one", how="left")
    require(paths[["mic_path", "acc_path", "gyro_path"]].notna().all().all(), "raw paths missing")
    return [SegmentSource(r.machine, r.segment_id, r.parent_recording, r.domain,
                          {"mic": r.mic_path, "acc": r.acc_path, "gyro": r.gyro_path}) for r in paths.itertuples()]


def create_dev_fixture(sandbox, package, *, catalog=None):
    """Write the development integration fixture into ``sandbox``; returns its manifest."""

    if catalog is None:
        catalog = development_catalog()
    manifest = {"scope": SYNTHETIC, "schema": DEV_SCHEMA, "recipe": DEV_RECIPE,
                "data_origin": "PERMITTED development source-validation inputs; SYNTHETIC fake labels",
                "files": {}, "cells": {}, "cohorts": {}, "historical": {}, "labels": {},
                "package_identity": package.identity, "real_model_inference": True}

    def put(name, write):
        path = sandbox.path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        write(path)
        manifest["files"][name] = file_sha256(path)

    detectors = package.registry["detectors"]
    features98 = next(tuple(d["data_provenance"]["feature_columns"]) for d in detectors.values()
                      if d["family"] == "mahalanobis_shrinkage" and d.get("model_path", "").startswith("data/interim/phase5"))
    feature_windows = {}
    for machine in ("RoboticArm", "BrushlessMotor"):
        full = label_free_cohort(SPLIT_MANIFEST, machine=machine, regime="S", split="validation")
        cohort = select_cohort(full, machine)
        manifest["cohorts"][machine] = cohort.loc[:, list(IDENTITY_COLUMNS)].to_dict(orient="records")
        labels = cohort.loc[:, list(IDENTITY_COLUMNS)].assign(fake_label=fake_labels(cohort))
        put(f"inputs/{machine}/labels.parquet", lambda p, f=labels: f.to_parquet(p, index=False))
        manifest["labels"][machine] = f"inputs/{machine}/labels.parquet"
        keep = set(cohort["segment_id"])
        for regime in ("S", "T"):
            cell = f"{machine}|{regime}"
            config = load_config(PREPROCESSING_CONFIGS[regime])
            whole = build_window_index(_sources(full.assign(machine=machine), catalog), config, regime=regime)
            index = whole[whole["segment_id"].isin(keep)].reset_index(drop=True)
            cache = spectral_cache_path(SPECTRAL_ROOT, machine=machine, regime=regime, role="source_validation")
            meta = json.loads(cache.with_suffix(".metadata.json").read_text(encoding="utf-8"))
            require(meta["split"] == "validation" and meta["role"] == "source_validation", "not a development cache")
            rows = index["row"].to_numpy()
            with h5py.File(cache, "r") as source:          # numeric datasets only; provenance_json untouched
                arrays = {m: np.asarray(source[m][rows], dtype=np.float32) for m in ("mic", "acc", "gyro")}

            def write_h5(path, arrays=arrays):
                with h5py.File(path, "w") as out:
                    for m, a in arrays.items():
                        out.create_dataset(m, data=a)
                    out.create_dataset("provenance_json", data=json.dumps([ANOMALY]).encode())   # poison
            put(f"inputs/{cell.replace('|', '/')}/numeric.h5", write_h5)
            put(f"inputs/{cell.replace('|', '/')}/window_index.parquet", lambda p, f=index: f.to_parquet(p, index=False))
            representation = {"source_cache": cache.as_posix(), "source_cache_sha256": meta["cache_sha256"],
                              "input_dims": dict(SPECTRAL_DIMS[machine]), "rows_are_source_cache_rows": True}
            manifest["cells"][cell] = {
                "features": f"inputs/{cell.replace('|', '/')}/features.parquet",
                "index": f"inputs/{cell.replace('|', '/')}/window_index.parquet",
                "spectral": f"inputs/{cell.replace('|', '/')}/numeric.h5",
                "index_manifest": index_manifest(index, representation=representation,
                                                 file_sha256=manifest["files"][f"inputs/{cell.replace('|', '/')}/window_index.parquet"]),
            }
            artifact = FEATURE_ROOT / machine / regime / "source_validation.parquet"
            features = pd.read_parquet(artifact, columns=list(IDENTITY_COLUMNS) + ["window_count", *features98])
            features = features[features["segment_id"].isin(keep)].sort_values("segment_id").reset_index(drop=True)
            feature_windows[cell] = features.set_index("segment_id")["window_count"]
            poisoned = features.assign(**{POISON_COLUMN: ANOMALY})
            put(manifest["cells"][cell]["features"], lambda p, f=poisoned: f.to_parquet(p, index=False))

    for unit in package.registry["units"]:
        machine, cell = unit["machine"], f"{unit['machine']}|{unit['regime']}"
        keep = {r["segment_id"] for r in manifest["cohorts"][machine]}
        reused = unit["L1"]["source"] != "new_fit_phase5"
        for eid in list(unit["L0"].values()) + ([unit["L1"]["experiment_id"]] if reused else []):
            if eid in manifest["historical"]:
                continue
            bundle = Path(detectors[eid]["bundle_path"]) / "validation_scores.csv"
            available = pd.read_csv(bundle, nrows=0).columns
            columns = [c for c in (*IDENTITY_COLUMNS, "score", "window_count") if c in available]
            table = pd.read_csv(bundle, usecols=columns, dtype={"segment_id": str, "parent_recording": str},
                                float_precision="round_trip")
            table = table[table["segment_id"].isin(keep)].sort_values("segment_id").reset_index(drop=True)
            if "window_count" not in table:
                table["window_count"] = table["segment_id"].map(feature_windows[cell]).astype(int)
            table = table.loc[:, list(SCORE_COLUMNS)].assign(**{POISON_COLUMN: ANOMALY})
            require(len(table) == 12, f"historical table {eid} lacks cohort segments")
            name = f"historical/{eid}.parquet"
            put(name, lambda p, f=table: f.to_parquet(p, index=False))
            manifest["historical"][eid] = name

    manifest["ledger"] = {**seed_jsonl_ledger(sandbox),
                          "code_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    atomic_json(sandbox.path("fixture.json"), manifest)
    return manifest
