"""Phase 5.5F: frozen-model inference adapter for the 24 new L1 units.

A predictor for ``Executor`` (same call signature as the synthetic stub). It loads
the real frozen model and its frozen standardizer by verified identity and scores
with the original model code; it never fits, updates or recalibrates anything.

* identity: the detector must be one of the 24 planned ``new_L1_inference`` units,
  equal to its frozen-registry record; the model directory is the registry's, and
  ``ModelStore.load`` re-hashes every file against the registry's manifest hash and
  checks the stored model id;
* classical (Mahalanobis): input columns must be exactly the identity columns,
  ``window_count`` and the frozen 98 features in registry order;
  score = ``model.score(standardizer.transform(X))``;
* dense autoencoder: per-window float32 spectra concatenated mic|acc|gyro with the
  frozen per-modality widths; standardized with the frozen L1 routine
  (``_standardize_in_place``); ``model.score_windows``; segment score = the frozen
  Protocol-B median (``aggregate_window_scores_by_segment``);
* input row/window order is preserved and every call is recorded for coverage.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.experiment.phase4 import ModelStore, aggregate_window_scores_by_segment
from src.experiment.phase5_inputs import IDENTITY_COLUMNS, SCORE_COLUMNS, require

ADAPTER_VERSION = "phase5-frozen-l1-inference-adapter-v1"
MODALITIES = ("mic", "acc", "gyro")
PHASE5_MODEL_ROOT = Path("data/interim/phase5/models")


class FrozenL1Adapter:
    """Real frozen-model scoring for the 24 new L1 units (development/permitted inputs only)."""

    def __init__(self, package, *, model_root=PHASE5_MODEL_ROOT):
        self.package, self.model_root = package, Path(model_root)
        self.planned = frozenset(r["experiment_id"] for r in package.proposed["planned_units"]
                                 if r["operation"] == "new_L1_inference")
        require(len(self.planned) == 24, "the frozen plan must hold exactly 24 new L1 units")
        self._loaded = {}
        self.calls = []

    def _model(self, detector):
        eid = detector.get("experiment_id")
        require(eid in self.planned, "not a planned new L1 inference unit")
        trusted = self.package.registry["detectors"][eid]
        require(detector == trusted, "detector record differs from the frozen registry")
        require(Path(trusted["model_path"]).resolve() == (self.model_root / eid).resolve(),
                "model path differs from the frozen registry")
        if eid not in self._loaded:
            model, standardizer, manifest = ModelStore(self.model_root).load(
                eid, expected_manifest_sha256=trusted["model_manifest"]["model_manifest_sha256"])
            require(manifest["family"] == trusted["family"] and manifest["model_id"] == trusted["model_manifest"]["model_id"]
                    and model.model_id == manifest["model_id"], "frozen model identity mismatch")
            self._loaded[eid] = (model, standardizer)
        return self._loaded[eid]

    def __call__(self, detector, inputs, index=None):
        model, standardizer = self._model(detector)
        family, machine = detector["family"], detector["machine"]
        if family == "mahalanobis_shrinkage":
            columns = tuple(detector["data_provenance"]["feature_columns"])
            require(isinstance(inputs, pd.DataFrame)
                    and tuple(inputs.columns) == IDENTITY_COLUMNS + ("window_count",) + columns,
                    "classical representation differs from the frozen 98-feature layout")
            X = inputs.loc[:, list(columns)].to_numpy(dtype=np.float64)
            require(np.isfinite(X).all(), "non-finite classical feature")
            table = inputs.loc[:, list(IDENTITY_COLUMNS) + ["window_count"]].reset_index(drop=True)
            table["score"] = np.asarray(model.score(standardizer.transform(X)), dtype=np.float64)
            rows = len(X)
        elif family == "dense_autoencoder":
            from src.fusion.l1 import _standardize_in_place
            blocks = detector["data_provenance"]["spectral_blocks"]
            require(isinstance(inputs, dict) and set(inputs) == set(MODALITIES) and index is not None,
                    "spectral inputs and a label-free window index are required")
            for m in MODALITIES:
                width = blocks[m][1] - blocks[m][0]
                require(inputs[m].dtype == np.float32 and inputs[m].shape == (len(index), width),
                        f"{m} spectral representation differs from the frozen block width {width}")
            X = np.concatenate([inputs[m] for m in MODALITIES], axis=1).astype(np.float32, copy=True)
            require(X.shape[1] == blocks["gyro"][1] == model.input_dim, "spectral width differs from the frozen model")
            X = _standardize_in_place(standardizer, X)
            meta = index.loc[:, list(IDENTITY_COLUMNS)].reset_index(drop=True).assign(anomaly_label="withheld")
            table = aggregate_window_scores_by_segment(meta, model.score_windows(X)).drop(columns=["anomaly_label"])
            rows = len(X)
        else:
            raise ValueError(f"family {family!r} is not a new L1 inference family")
        self.calls.append({"experiment_id": detector["experiment_id"], "family": family, "machine": machine,
                           "regime": detector["regime"], "seed": int(detector["seed"]), "input_rows": int(rows),
                           "segments": int(len(table))})
        return table.loc[:, list(SCORE_COLUMNS)]
