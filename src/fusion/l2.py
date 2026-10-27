"""L2 score-level fusion: source-validation calibration and fused q90 thresholds (Phase 5.3).

Implements exactly design v2 §3.3, development side only:

* per modality, robust z = (score - median_v) / IQR_v with median_v and IQR_v from
  that modality's frozen source-validation segment scores (normal only);
* fused score = arithmetic mean of the calibrated scores of the subset, summed in
  the fixed modality order mic, acc, gyro;
* threshold = q90 (``method="higher"``) of the subset's own fused source-validation
  scores, via the frozen ``normal_quantile_threshold``;
* four subsets per unit, 52 units -> 208 calibration records.

Implementation clarification (fixed before any calibration was computed; v2 does
not name it): IQR_v = q75 - q25 with NumPy's default ``linear`` interpolation,
the conventional IQR.  The threshold keeps the frozen ``higher`` method.

A non-finite score or IQR_v == 0 is a technical defect: ``L2CalibrationError``,
never an epsilon or fallback.  No sealed test score is read here.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.evaluation.policy import normal_quantile_threshold
from src.experiment.provenance import canonical_json, canonical_sha256, content_sha256
from src.fusion.assembly import (
    DESIGN_CONFIG,
    DESIGN_CONFIG_SHA256,
    MODALITIES,
    UnitValidationScores,
    assemble_unit_validation_scores,
    load_design_registry,
)


L2_CALIBRATION_VERSION = "phase5-l2-robust-z-mean-v1"
IQR_QUANTILE_METHOD = "linear"
THRESHOLD_QUANTILE = 0.90
THRESHOLD_METHOD = "higher"


class L2CalibrationError(ValueError):
    """Calibration inputs violate the frozen rule; nothing is calibrated."""


def design_subsets(path: str | Path = DESIGN_CONFIG) -> tuple[tuple[str, ...], ...]:
    load_design_registry(path)                       # verifies the design hash first
    subsets = json.loads(Path(path).read_text(encoding="utf-8"))["arms"]["L2_score_level_fusion"]["subsets"]
    canonical = tuple(tuple(m for m in MODALITIES if m in s) for s in subsets)
    if len(canonical) != 4 or len(set(canonical)) != 4 or canonical[0] != MODALITIES:
        raise L2CalibrationError("design subsets are not the four frozen modality subsets")
    return canonical


@dataclass(frozen=True)
class ModalityCalibration:
    median: float
    q25: float
    q75: float
    iqr: float


def calibrate_modality(scores: np.ndarray) -> ModalityCalibration:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise L2CalibrationError("calibration needs a non-empty vector of finite scores")
    q25, q75 = np.quantile(values, [0.25, 0.75], method=IQR_QUANTILE_METHOD)
    iqr = float(q75 - q25)
    if iqr == 0.0:
        raise L2CalibrationError("IQR of the source-validation scores is zero; technical defect, not repaired")
    return ModalityCalibration(float(np.median(values)), float(q25), float(q75), iqr)


def robust_z(scores, calibration: ModalityCalibration) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise L2CalibrationError("non-finite score; technical defect, never imputed")
    return (values - calibration.median) / calibration.iqr


def fuse(calibrated: Mapping[str, np.ndarray], subset: Sequence[str]) -> np.ndarray:
    """Mean of the calibrated scores of ``subset``, accumulated in mic, acc, gyro order."""

    ordered = [m for m in MODALITIES if m in subset]
    if len(ordered) != len(subset) or len(ordered) < 2:
        raise L2CalibrationError(f"invalid fusion subset {subset!r}")
    lengths = {len(calibrated[m]) for m in ordered}
    if len(lengths) != 1:
        raise L2CalibrationError("calibrated modality vectors differ in length")
    total = np.zeros(lengths.pop(), dtype=np.float64)
    for modality in ordered:
        total = total + calibrated[modality]
    return total / len(ordered)


def unit_key(unit: Mapping) -> str:
    return f"{unit['machine']}|{unit['regime']}|{unit['family']}|{unit['seed']}"


def calibration_id(unit: Mapping, subset: Sequence[str]) -> str:
    payload = {"version": L2_CALIBRATION_VERSION, "unit": unit_key(unit), "subset": list(subset),
               "components": {m: unit["phase4_single_modality_ids"][m] for m in subset}}
    return f"p5l2-{content_sha256(payload)[:20]}"


def calibrate_unit(data: UnitValidationScores, subsets) -> tuple[list[dict], pd.DataFrame]:
    unit, frame = data.unit, data.scores
    calibrations = {m: calibrate_modality(frame[m].to_numpy()) for m in MODALITIES}
    calibrated = {m: robust_z(frame[m].to_numpy(), calibrations[m]) for m in MODALITIES}
    records, fused_rows = [], []
    for subset in subsets:
        fused = fuse(calibrated, subset)
        threshold = normal_quantile_threshold(fused, quantile=THRESHOLD_QUANTILE)
        cid = calibration_id(unit, subset)
        records.append({
            "calibration_id": cid, "calibration_version": L2_CALIBRATION_VERSION,
            "unit": unit_key(unit), "machine": unit["machine"], "regime": unit["regime"],
            "family": unit["family"], "seed": int(unit["seed"]), "subset": list(subset),
            "component_experiment_ids": {m: unit["phase4_single_modality_ids"][m] for m in subset},
            "modality_calibration": {m: vars(calibrations[m]) for m in subset},
            "iqr_quantile_method": IQR_QUANTILE_METHOD,
            "threshold": {"quantile": THRESHOLD_QUANTILE, "method": THRESHOLD_METHOD, "value": float(threshold)},
            "validation_segment_count": int(len(frame)),
            "validation_cohort_hash": data.cohort_hash,
            "score_direction": "higher_is_more_anomalous",
            "fused_validation_scores_sha256": content_sha256([float(v) for v in fused]),
            "test_evaluated": False,
        })
        fused_rows.append(pd.DataFrame({"calibration_id": cid, "segment_id": frame["segment_id"].to_numpy(),
                                        "parent_recording": frame["parent_recording"].to_numpy(),
                                        "fused_score": fused}))
    return records, pd.concat(fused_rows, ignore_index=True)


def calibrate_all(pretest_root, design_path: str | Path = DESIGN_CONFIG) -> tuple[list[dict], pd.DataFrame]:
    """All 208 calibration records, in registry order, from the frozen validation scores."""

    subsets = design_subsets(design_path)
    records, frames = [], []
    for unit in load_design_registry(design_path):
        unit_records, fused = calibrate_unit(assemble_unit_validation_scores(unit, pretest_root), subsets)
        records += unit_records
        frames.append(fused)
    ids = [r["calibration_id"] for r in records]
    if len(records) != 208 or len(set(ids)) != 208:
        raise L2CalibrationError(f"expected 208 unique calibration records, built {len(records)}")
    return records, pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------- persistence
RECORDS_NAME = "l2_calibration_records.json"
FUSED_NAME = "l2_fused_validation_scores.csv"
MANIFEST_NAME = "l2_manifest.json"


def write_calibration_artifacts(out_dir, records, fused, *, code_git_sha: str, code_source_hash: str) -> dict:
    out = Path(out_dir)
    targets = [out / RECORDS_NAME, out / FUSED_NAME, out / MANIFEST_NAME]
    if any(p.exists() for p in targets):
        raise FileExistsError(f"L2 calibration artifacts already exist in {out}; a repeat needs a new version")
    out.mkdir(parents=True, exist_ok=True)
    (out / RECORDS_NAME).write_bytes((canonical_json(records) + "\n").encode("utf-8"))
    fused.to_csv(out / FUSED_NAME, index=False, lineterminator="\n", float_format="%.17g")
    manifest = {
        "stage": "multi_sensor_fusion_development", "calibration_version": L2_CALIBRATION_VERSION,
        "design_config_sha256": DESIGN_CONFIG_SHA256, "code_git_sha": code_git_sha,
        "code_source_hash": code_source_hash, "record_count": len(records),
        "files": {RECORDS_NAME: canonical_sha256(out / RECORDS_NAME), FUSED_NAME: canonical_sha256(out / FUSED_NAME)},
        "test_evaluated": False, "protected_inputs_read": False,
    }
    (out / MANIFEST_NAME).write_bytes((canonical_json(manifest) + "\n").encode("utf-8"))
    return manifest


def verify_calibration_artifacts(out_dir, pretest_root, design_path: str | Path = DESIGN_CONFIG) -> dict:
    """Recompute all 208 records from the frozen validation scores and compare with the persisted ones."""

    out = Path(out_dir)
    manifest = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))
    for name, digest in manifest["files"].items():
        if canonical_sha256(out / name) != digest:
            raise L2CalibrationError(f"persisted L2 artifact {name} hash mismatch")
    persisted = json.loads((out / RECORDS_NAME).read_text(encoding="utf-8"))
    records, fused = calibrate_all(pretest_root, design_path)
    if canonical_json(persisted) != canonical_json(records):
        raise L2CalibrationError("persisted L2 calibration records do not reproduce")
    # round_trip parsing: the default fast parser can be one ulp off a %.17g value
    stored = pd.read_csv(out / FUSED_NAME, dtype={"segment_id": str, "parent_recording": str},
                         float_precision="round_trip")
    if not (stored[["calibration_id", "segment_id"]].equals(fused[["calibration_id", "segment_id"]])
            and np.array_equal(stored["fused_score"].to_numpy(), fused["fused_score"].to_numpy())):
        raise L2CalibrationError("persisted fused validation scores do not reproduce")
    return {"records_reproduced": len(records), "fused_rows_reproduced": int(len(fused))}
