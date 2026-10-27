"""HDF5 persistence for frozen per-window log-power-spectrum inputs.

One cache holds one (machine, regime, split) cohort.  Each modality is a
fixed-width float32 matrix in window order, with the per-window provenance
columns stored alongside so a consumer can never lose the mapping from a row
to its segment, parent recording, domain and label.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np
import pandas as pd

from src.preprocessing.protected_access import (
    PROTECTED_SPECTRAL_CACHE_NAME,
    is_protected_role,
    is_protected_split,
    require_protected_read,
)
from src.preprocessing.spectral_windows import MODALITIES, SPECTRAL_REPRESENTATION_VERSION


SPECTRAL_CACHE_FORMAT = "hdf5_spectral_fixed_v1"
PRETEST_CACHE_ROLES = {"training": "train", "source_validation": "validation"}
PROTECTED_CACHE_ROLE = {"common_evaluation": "test"}
PROVENANCE_COLUMNS = (
    "window_id", "window_index", "segment_id", "parent_recording", "domain",
    "data_pool", "split", "anomaly_label",
)


def assert_pretest_cache_role(role: str) -> str:
    if role not in PRETEST_CACHE_ROLES:
        raise ValueError(
            f"pre-test spectral caches accept training/source_validation only; refusing {role!r}"
        )
    return PRETEST_CACHE_ROLES[role]


def spectral_cache_path(root: str | Path, *, machine: str, regime: str, role: str) -> Path:
    return Path(root) / machine / regime / f"{role}.spectral.h5"


class SpectralCacheWriter:
    def __init__(
        self,
        path: str | Path,
        *,
        machine: str,
        regime: str,
        split: str,
        preprocessing_config_id: str,
        split_version: str,
        input_dims: Mapping[str, int],
    ):
        if set(input_dims) != set(MODALITIES):
            raise ValueError("input dimensions are required for mic, acc and gyro")
        self.path = Path(path)
        self.attrs = {
            "machine": machine,
            "regime": regime,
            "split": split,
            "preprocessing_config_id": preprocessing_config_id,
            "split_version": split_version,
            "cache_format": SPECTRAL_CACHE_FORMAT,
            "spectral_representation_version": SPECTRAL_REPRESENTATION_VERSION,
            "input_dims": {key: int(input_dims[key]) for key in MODALITIES},
        }
        self._handle: h5py.File | None = None
        self._seen: set[str] = set()
        self._rows: list[dict] = []
        self._pending: dict[str, list[np.ndarray]] = {modality: [] for modality in MODALITIES}
        self.flush_rows = 2048

    def __enter__(self) -> "SpectralCacheWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(f"refusing to overwrite spectral cache: {self.path}")
        self._handle = h5py.File(self.path, "w")
        for key, value in self.attrs.items():
            self._handle.attrs[key] = json.dumps(value) if isinstance(value, dict) else value
        for modality in MODALITIES:
            width = self.attrs["input_dims"][modality]
            self._handle.create_dataset(
                modality, shape=(0, width), maxshape=(None, width),
                chunks=(min(4096, 1024), width), dtype="f4",
            )
        return self

    def append(self, record: Mapping, vectors: Mapping[str, np.ndarray]) -> None:
        assert self._handle is not None
        window_id = str(record["window_id"])
        if window_id in self._seen:
            raise ValueError(f"duplicate window id in spectral cache: {window_id}")
        for modality in MODALITIES:
            vector = np.asarray(vectors[modality], dtype=np.float32).reshape(-1)
            width = self.attrs["input_dims"][modality]
            if vector.shape != (width,):
                raise ValueError(
                    f"{modality} vector dimension {vector.shape} differs from frozen ({width},)"
                )
            if not np.all(np.isfinite(vector)):
                raise ValueError(f"{modality} spectral vector must be finite")
            self._pending[modality].append(vector)
        self._seen.add(window_id)
        self._rows.append({column: record[column] for column in PROVENANCE_COLUMNS})
        if len(self._pending["mic"]) >= self.flush_rows:
            self._flush()

    def _flush(self) -> None:
        """Append buffered rows in one resize per modality (row order preserved)."""

        assert self._handle is not None
        for modality in MODALITIES:
            pending = self._pending[modality]
            if not pending:
                continue
            block = np.vstack(pending)
            dataset = self._handle[modality]
            start = dataset.shape[0]
            dataset.resize(start + len(block), axis=0)
            dataset[start:start + len(block)] = block
            self._pending[modality] = []

    def __exit__(self, exc_type, exc, tb) -> None:
        assert self._handle is not None
        try:
            if exc_type is None:
                self._flush()
                frame = pd.DataFrame(self._rows, columns=list(PROVENANCE_COLUMNS))
                if frame.empty:
                    raise ValueError("cannot finalize an empty spectral cache")
                self._handle.create_dataset(
                    "provenance_json",
                    data=frame.to_json(orient="records"),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )
                self._handle.attrs["window_count"] = int(len(frame))
        finally:
            self._handle.close()
            self._handle = None
        if exc_type is not None and self.path.exists():
            self.path.unlink()


def load_spectral_cache(
    path: str | Path,
    *,
    modality: str,
    expected_split: str | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    if modality not in MODALITIES:
        raise ValueError(f"unknown modality {modality!r}")
    # Refuse by name and requested split before the file is opened, then by the
    # stored split before any matrix is read, so a renamed copy is refused too.
    if Path(path).name == PROTECTED_SPECTRAL_CACHE_NAME or is_protected_split(expected_split):
        require_protected_read(f"spectral cache {Path(path).name}")
    with h5py.File(Path(path), "r") as handle:
        if is_protected_split(handle.attrs.get("split")):
            require_protected_read(f"spectral cache {Path(path).name} (stored split=test)")
        attrs = {key: handle.attrs[key] for key in handle.attrs}
        attrs["input_dims"] = json.loads(attrs["input_dims"])
        attrs = {key: (value.item() if hasattr(value, "item") else value) for key, value in attrs.items()}
        if attrs.get("cache_format") != SPECTRAL_CACHE_FORMAT:
            raise ValueError("unsupported spectral cache format")
        if expected_split is not None and attrs.get("split") != expected_split:
            raise ValueError(
                f"spectral cache split {attrs.get('split')!r} differs from expected {expected_split!r}"
            )
        X = np.asarray(handle[modality][()], dtype=np.float32)
        raw = handle["provenance_json"][()]
        raw = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        frame = pd.DataFrame(json.loads(raw))
    frame = frame.loc[:, list(PROVENANCE_COLUMNS)]
    for column in PROVENANCE_COLUMNS:
        if column != "window_index":
            frame[column] = frame[column].astype(str)
    frame["window_index"] = frame["window_index"].astype(int)
    if len(frame) != len(X) or len(frame) != int(attrs["window_count"]):
        raise ValueError("spectral cache row count mismatch")
    if frame["window_id"].duplicated().any():
        raise ValueError("spectral cache contains duplicate window ids")
    if X.shape[1] != attrs["input_dims"][modality] or not np.all(np.isfinite(X)):
        raise ValueError("spectral cache matrix is inconsistent")
    frame.attrs.update(attrs)
    return X, frame


# ------------------------------------------------------------------ builders

def _build_spectral_cache(
    *,
    data_root: str | Path,
    split_manifest_path: str | Path,
    preprocessing_config_path: str | Path,
    cache_root: str | Path,
    machine: str,
    role: str,
    split: str,
    max_segments: int | None,
) -> dict:
    """Stream the frozen Protocol-B pipeline into one spectral cache."""

    from src.experiment.provenance import file_sha256, ordered_id_hash
    from src.preprocessing.config import load_config
    from src.preprocessing.pipeline import (
        iter_processed_windows,
        load_raw_catalog,
        merge_protocol_b_assignments,
    )
    from src.preprocessing.spectral_windows import (
        log_power_spectrum_input,
        spectral_input_dim,
    )

    config = load_config(preprocessing_config_path)
    path = spectral_cache_path(cache_root, machine=machine, regime=config.regime, role=role)
    metadata_path = path.with_suffix(".metadata.json")
    if path.exists() or metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite spectral cache: {path}")
    assignments = pd.read_csv(split_manifest_path)
    catalog = merge_protocol_b_assignments(
        load_raw_catalog(data_root), assignments, config.regime,
        split_version=config.split_version,
    )
    machine_catalog = catalog[catalog["machine"].eq(machine)].copy()
    input_dims = {
        modality: spectral_input_dim(config, machine, modality) for modality in MODALITIES
    }
    segments: list[str] = []
    with SpectralCacheWriter(
        path, machine=machine, regime=config.regime, split=split,
        preprocessing_config_id=config.config_id, split_version=config.split_version,
        input_dims=input_dims,
    ) as writer:
        for record, signals, _features in iter_processed_windows(
            machine_catalog, config, split=split, max_segments=max_segments,
        ):
            if str(record["split"]) != split:
                raise ValueError("preprocessing stream contains an unauthorized split")
            vectors = {
                modality: log_power_spectrum_input(signals[modality], config, machine)
                for modality in MODALITIES
            }
            writer.append(record, vectors)
            if not segments or segments[-1] != str(record["segment_id"]):
                segments.append(str(record["segment_id"]))
    if len(segments) != len(set(segments)):
        raise ValueError("segment windows were not contiguous in the preprocessing stream")
    _, frame = load_spectral_cache(path, modality="mic", expected_split=split)
    metadata = {
        "cache_format": SPECTRAL_CACHE_FORMAT,
        "spectral_representation_version": SPECTRAL_REPRESENTATION_VERSION,
        "machine": machine,
        "regime": config.regime,
        "role": role,
        "split": split,
        "preprocessing_config_id": config.config_id,
        "preprocessing_config_hash": file_sha256(preprocessing_config_path),
        "split_version": config.split_version,
        "split_manifest_hash": file_sha256(split_manifest_path),
        "input_dims": input_dims,
        "segment_count": int(frame["segment_id"].nunique()),
        "window_count": int(len(frame)),
        "segment_cohort_hash": ordered_id_hash(tuple(sorted(set(frame["segment_id"])))),
        "cache_path": str(path),
        "cache_sha256": file_sha256(path),
        "max_segments": max_segments,
        "test_evaluated": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def build_pretest_spectral_cache(
    *,
    data_root: str | Path,
    split_manifest_path: str | Path,
    preprocessing_config_path: str | Path,
    cache_root: str | Path,
    machine: str,
    role: str,
    max_segments: int | None = None,
) -> dict:
    """Training / source-validation caches only; every other role is refused."""

    split = assert_pretest_cache_role(role)
    return _build_spectral_cache(
        data_root=data_root, split_manifest_path=split_manifest_path,
        preprocessing_config_path=preprocessing_config_path, cache_root=cache_root,
        machine=machine, role=role, split=split, max_segments=max_segments,
    )


def build_protected_spectral_cache(
    *,
    data_root: str | Path,
    split_manifest_path: str | Path,
    preprocessing_config_path: str | Path,
    cache_root: str | Path,
    machine: str,
) -> dict:
    """Common-evaluation cache; callable only from the protected score-all stage."""

    require_protected_read("build common-evaluation spectral cache")
    return _build_spectral_cache(
        data_root=data_root, split_manifest_path=split_manifest_path,
        preprocessing_config_path=preprocessing_config_path, cache_root=cache_root,
        machine=machine, role="common_evaluation",
        split=PROTECTED_CACHE_ROLE["common_evaluation"], max_segments=None,
    )


def load_spectral_cache_metadata(cache_root: str | Path, *, machine: str, regime: str, role: str) -> dict:
    from src.experiment.provenance import file_sha256

    if is_protected_role(role):
        require_protected_read(f"spectral cache metadata for role {role}")
    path = spectral_cache_path(cache_root, machine=machine, regime=regime, role=role)
    metadata = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    if file_sha256(path) != metadata["cache_sha256"]:
        raise ValueError(f"spectral cache hash mismatch: {path}")
    return metadata
