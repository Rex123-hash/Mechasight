"""Development-only data assembly for Multi-Sensor Fusion Development (Phase 5.2).

Assembles, from training and source-validation data only, the inputs that the
frozen design (``configs/phase5/fusion_design_v2.json``) needs:

* **L1 classical** — the 98 frozen Phase-3.1 engineered features of all three
  modalities, in the frozen sorted column order the reused Phase-3.2 all-channel
  detectors were fitted on;
* **L1 spectral** — per-window concatenation ``[mic | acc | gyro]`` of the frozen
  Phase-4 log-power spectra, joined on the cache's shared window provenance;
* **L2 inputs** — the frozen per-modality source-validation segment scores of one
  (machine, regime, family, seed) unit, joined on segment identity.

Nothing here fits a model, calibrates or fuses a score, or reads a protected
input or sealed test score: every entry point refuses any role other than
``training`` / ``source_validation`` before a path is built, and the existing
loaders it delegates to keep their own hash, cohort and finiteness checks.
Assembly never fills, drops, reorders or truncates records to make inputs fit;
any mismatch raises ``FusionAssemblyError``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from src.experiment.phase31 import load_feature_artifact, phase31_feature_columns
from src.experiment.phase4 import load_frozen_bundle, modality_feature_columns
from src.experiment.provenance import canonical_sha256, ordered_id_hash
from src.preprocessing.config import load_config
from src.preprocessing.spectral_cache import (
    PROVENANCE_COLUMNS,
    load_spectral_cache,
    load_spectral_cache_metadata,
    spectral_cache_path,
)
from src.preprocessing.spectral_windows import spectral_input_dim


DESIGN_CONFIG = Path("configs/phase5/fusion_design_v2.json")
DESIGN_CONFIG_SHA256 = "2e33299d3c0347ebdc13392d149b248b8ec175f2bea5b2049e5bb2fa996a759a"
DESIGN_STATUS = "PROPOSED — NOT AUTHORIZED FOR EXPERIMENTAL EXECUTION"

MACHINES = ("RoboticArm", "BrushlessMotor")
REGIMES = ("S", "T")
MODALITIES = ("mic", "acc", "gyro")          # spectral concatenation order (design v2)
CLASSICAL_FEATURE_COUNTS = {"mic": 14, "acc": 42, "gyro": 42}
CLASSICAL_FEATURE_TOTAL = 98
SPECTRAL_DIMS = {
    "RoboticArm": {"mic": 800, "acc": 1059, "gyro": 1059},
    "BrushlessMotor": {"mic": 800, "acc": 1068, "gyro": 1068},
}
DEVELOPMENT_ROLES = {"training": "train", "source_validation": "validation"}
# Data pools each development role may contain (frozen split v3): Regime T
# trains on the released target-normal allocation; nothing else ever appears.
ALLOWED_POOLS = {
    ("training", "S"): frozenset({"source_training"}),
    ("training", "T"): frozenset({"source_training", "target_adaptation"}),
    ("source_validation", "S"): frozenset({"source_validation"}),
    ("source_validation", "T"): frozenset({"source_validation"}),
}
PREPROCESSING_CONFIGS = {
    "S": Path("configs/preprocessing/mechasight_few_shot_regime_s.json"),
    "T": Path("configs/preprocessing/mechasight_few_shot_regime_t.json"),
}
SCORE_DIRECTION = "higher_is_more_anomalous"


class FusionAssemblyError(ValueError):
    """Fusion inputs violate the frozen data contract; nothing was assembled."""


def assert_development_role(role: str) -> str:
    """Return the split of a development role; refuse every other role before any I/O."""

    if role not in DEVELOPMENT_ROLES:
        raise FusionAssemblyError(
            f"fusion assembly reads training/source_validation only; refusing role {role!r}"
        )
    return DEVELOPMENT_ROLES[role]


def _check_key(machine: str, regime: str) -> None:
    if machine not in MACHINES or regime not in REGIMES:
        raise FusionAssemblyError(f"unknown machine/regime {machine!r}/{regime!r}")


def _check_development_rows(frame: pd.DataFrame, *, role: str, regime: str, split: str) -> None:
    if set(frame["split"].astype(str)) != {split}:
        raise FusionAssemblyError(f"{role} rows carry splits {sorted(set(frame['split']))}, expected {split!r}")
    pools = set(frame["data_pool"].astype(str))
    if not pools <= ALLOWED_POOLS[(role, regime)]:
        raise FusionAssemblyError(f"{role} rows carry unauthorised data pools {sorted(pools)}")
    if set(frame["anomaly_label"].astype(str)) != {"normal"}:
        raise FusionAssemblyError("development fusion inputs must be normal-only")


# ---------------------------------------------------------------- design registry
def load_design_registry(path: str | Path = DESIGN_CONFIG) -> tuple[dict, ...]:
    """The 52 frozen fusion units of design v2, after verifying the design is unchanged."""

    if canonical_sha256(path) != DESIGN_CONFIG_SHA256:
        raise FusionAssemblyError("fusion design v2 config differs from the reviewed version")
    design = json.loads(Path(path).read_text(encoding="utf-8"))
    if design["status"] != DESIGN_STATUS:
        raise FusionAssemblyError("fusion design status changed")
    units = tuple(design["unit_registry"])
    keys = [(u["machine"], u["regime"], u["family"], u["seed"]) for u in units]
    if len(units) != design["experiment_matrix"]["units"]["count"] or len(set(keys)) != len(keys):
        raise FusionAssemblyError("fusion unit registry is not a set of unique units")
    return units


# ------------------------------------------------------------------ L1 classical
@dataclass(frozen=True)
class ClassicalFusionData:
    X: np.ndarray                       # (segments, 98) float64
    columns: tuple[str, ...]            # frozen Phase-3.1 order (globally sorted)
    modality_of_column: tuple[str, ...]
    segments: pd.DataFrame              # one row per segment, aligned with X
    machine: str
    regime: str
    role: str
    cohort_hash: str
    artifact_sha256: str


def classical_fusion_columns(columns) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate the 98-feature schema and return (columns, modality of each column)."""

    columns = tuple(columns)
    if len(columns) != len(set(columns)):
        raise FusionAssemblyError("duplicated engineered feature columns")
    if columns != tuple(sorted(columns)):
        raise FusionAssemblyError("engineered features are not in the frozen sorted order")
    blocks = {m: modality_feature_columns(columns, m) for m in MODALITIES}
    if {m: len(c) for m, c in blocks.items()} != CLASSICAL_FEATURE_COUNTS:
        raise FusionAssemblyError(f"modality feature counts {[len(c) for c in blocks.values()]} != 14/42/42")
    covered = [c for block in blocks.values() for c in block]
    if len(columns) != CLASSICAL_FEATURE_TOTAL or set(covered) != set(columns) or len(covered) != len(columns):
        raise FusionAssemblyError("engineered features are not exactly the mic/acc/gyro blocks")
    modality = tuple(next(m for m in MODALITIES if c.startswith(f"{m}__")) for c in columns)
    return columns, modality


def assemble_classical_fusion(
    feature_root: str | Path, *, machine: str, regime: str, role: str
) -> ClassicalFusionData:
    split = assert_development_role(role)
    _check_key(machine, regime)
    frame, metadata = load_feature_artifact(feature_root, machine=machine, regime=regime, role=role)
    columns, modality = classical_fusion_columns(phase31_feature_columns(frame))
    for column, expected in (("machine", machine), ("regime", regime)):
        if set(frame[column].astype(str)) != {expected}:
            raise FusionAssemblyError(f"feature artifact {column} differs from {expected!r}")
    _check_development_rows(frame, role=role, regime=regime, split=split)
    segment_ids = tuple(frame["segment_id"].astype(str))
    if len(set(segment_ids)) != len(segment_ids):
        raise FusionAssemblyError("duplicated segment ids in the feature artifact")
    X = frame.loc[:, list(columns)].to_numpy(dtype=np.float64, copy=True)
    if not np.all(np.isfinite(X)):
        raise FusionAssemblyError("non-finite engineered feature")
    segments = frame.loc[:, ["segment_id", "parent_recording", "domain", "data_pool", "split",
                             "window_count"]].reset_index(drop=True)
    return ClassicalFusionData(
        X=X, columns=columns, modality_of_column=modality, segments=segments, machine=machine,
        regime=regime, role=role, cohort_hash=ordered_id_hash(segment_ids),
        artifact_sha256=str(metadata["artifact_sha256"]),
    )


# ------------------------------------------------------------------- L1 spectral
@dataclass(frozen=True)
class SpectralFusionData:
    X: np.ndarray                                  # (windows, 2918 or 2936) float32
    blocks: Mapping[str, tuple[int, int]]          # modality -> [start, stop) columns
    windows: pd.DataFrame                          # shared window provenance, aligned with X
    machine: str
    regime: str
    role: str
    cohort_hash: str                               # ordered hash of the segment ids
    cache_sha256: str


def spectral_fusion_blocks(machine: str, regime: str) -> dict[str, tuple[int, int]]:
    """Column blocks of the concatenation, from the frozen preprocessing config."""

    config = load_config(PREPROCESSING_CONFIGS[regime])
    dims = {m: spectral_input_dim(config, machine, m) for m in MODALITIES}
    if dims != SPECTRAL_DIMS[machine]:
        raise FusionAssemblyError(f"frozen spectral dims {dims} differ from the design {SPECTRAL_DIMS[machine]}")
    blocks, start = {}, 0
    for m in MODALITIES:
        blocks[m] = (start, start + dims[m])
        start += dims[m]
    return blocks


def assemble_spectral_fusion(
    cache_root: str | Path, *, machine: str, regime: str, role: str
) -> SpectralFusionData:
    split = assert_development_role(role)
    _check_key(machine, regime)
    blocks = spectral_fusion_blocks(machine, regime)
    metadata = load_spectral_cache_metadata(cache_root, machine=machine, regime=regime, role=role)
    if (metadata["machine"], metadata["regime"], metadata["role"], metadata["split"]) != (machine, regime, role, split):
        raise FusionAssemblyError("spectral cache metadata identity differs from the request")
    if {m: int(v) for m, v in metadata["input_dims"].items()} != SPECTRAL_DIMS[machine]:
        raise FusionAssemblyError(f"spectral cache dims {metadata['input_dims']} differ from the design")
    path = spectral_cache_path(cache_root, machine=machine, regime=regime, role=role)
    X = None
    windows = None
    for modality in MODALITIES:
        matrix, meta = load_spectral_cache(path, modality=modality, expected_split=split)
        start, stop = blocks[modality]
        if matrix.shape[1] != stop - start:
            raise FusionAssemblyError(f"{modality} width {matrix.shape[1]} != {stop - start}")
        meta = meta.loc[:, list(PROVENANCE_COLUMNS)].reset_index(drop=True)
        if windows is None:
            windows = meta
            X = np.empty((len(meta), blocks[MODALITIES[-1]][1]), dtype=np.float32)
        elif len(meta) != len(windows) or not meta.equals(windows):
            raise FusionAssemblyError(f"{modality} window provenance differs from mic; refusing to concatenate")
        X[:, start:stop] = matrix
        del matrix
    assert windows is not None and X is not None
    if len(windows) != int(metadata["window_count"]) or windows["window_id"].duplicated().any():
        raise FusionAssemblyError("spectral cache window identity is inconsistent")
    _check_development_rows(windows, role=role, regime=regime, split=split)
    _check_window_order(windows)
    segment_ids = tuple(sorted(set(windows["segment_id"])))
    if ordered_id_hash(segment_ids) != metadata["segment_cohort_hash"]:
        raise FusionAssemblyError("spectral cache segment cohort hash mismatch")
    return SpectralFusionData(
        X=X, blocks=blocks, windows=windows, machine=machine, regime=regime, role=role,
        cohort_hash=ordered_id_hash(segment_ids), cache_sha256=str(metadata["cache_sha256"]),
    )


def _check_window_order(windows: pd.DataFrame) -> None:
    """Each segment is one contiguous block, its windows in index order 0..k-1, one parent and domain."""

    segment = windows["segment_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, segment[1:] != segment[:-1]])
    if len(starts) != len(set(segment)):
        raise FusionAssemblyError("a segment's windows are not contiguous")
    grouped = windows.groupby("segment_id", sort=False)
    for segment_id, group in grouped:
        if list(group["window_index"]) != list(range(len(group))):
            raise FusionAssemblyError(f"segment {segment_id} windows are not indexed 0..k-1 in order")
        if group["parent_recording"].nunique() != 1 or group["domain"].nunique() != 1:
            raise FusionAssemblyError(f"segment {segment_id} spans parents or domains")


# ---------------------------------------------------------------------- L2 inputs
@dataclass(frozen=True)
class UnitValidationScores:
    unit: Mapping
    scores: pd.DataFrame        # segment_id, parent_recording, domain, mic, acc, gyro; sorted by segment_id
    experiment_ids: Mapping[str, str]
    cohort_hash: str


def assemble_unit_validation_scores(unit: Mapping, pretest_root: str | Path) -> UnitValidationScores:
    """Join one unit's three frozen source-validation score files on segment identity."""

    frames = {}
    identity = None
    for modality in MODALITIES:
        experiment_id = unit["phase4_single_modality_ids"][modality]
        bundle = load_frozen_bundle(Path(pretest_root) / experiment_id)
        config = bundle.config
        if (bundle.modality, config.machine, config.regime, config.model_family, int(config.model_seed)) != (
            modality, unit["machine"], unit["regime"], unit["family"], int(unit["seed"])
        ):
            raise FusionAssemblyError(f"bundle {experiment_id} does not belong to this unit/modality")
        if bundle.threshold.score_direction != SCORE_DIRECTION:
            raise FusionAssemblyError(f"bundle {experiment_id} score direction is not {SCORE_DIRECTION}")
        frame = pd.read_csv(bundle.bundle_path / "validation_scores.csv",
                            dtype={"segment_id": str, "parent_recording": str, "domain": str},
                            float_precision="round_trip")      # the exact persisted floats
        if set(frame["cohort_role"]) != {"source_validation"} or set(frame["anomaly_label"]) != {"normal"}:
            raise FusionAssemblyError(f"bundle {experiment_id} validation scores are not normal source validation")
        if frame["segment_id"].duplicated().any() or not np.all(np.isfinite(frame["score"].to_numpy(float))):
            raise FusionAssemblyError(f"bundle {experiment_id} validation scores are not unique and finite")
        provenance = json.loads((bundle.bundle_path / "provenance.json").read_text(encoding="utf-8"))
        if ordered_id_hash(tuple(frame["segment_id"])) != provenance["validation_cohort_hash"]:
            raise FusionAssemblyError(f"bundle {experiment_id} validation cohort hash mismatch")
        keyed = frame.loc[:, ["segment_id", "parent_recording", "domain"]].sort_values("segment_id").reset_index(drop=True)
        if identity is None:
            identity = keyed
        elif not keyed.equals(identity):
            raise FusionAssemblyError(f"{modality} validation segments differ from mic; refusing to join")
        frames[modality] = frame.set_index("segment_id")["score"]
    assert identity is not None
    scores = identity.copy()
    for modality in MODALITIES:
        scores[modality] = frames[modality].reindex(scores["segment_id"]).to_numpy(dtype=np.float64)
    return UnitValidationScores(
        unit=unit, scores=scores, experiment_ids=dict(unit["phase4_single_modality_ids"]),
        cohort_hash=ordered_id_hash(tuple(scores["segment_id"])),
    )
