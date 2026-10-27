"""Streaming ragged HDF5 cache with configuration-bound identity."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from src.preprocessing.config import PreprocessingConfig
from src.preprocessing.protected_access import is_protected_split, require_protected_read
from src.preprocessing.signals import SensorSignal


def cache_path(
    root: str | Path,
    config: PreprocessingConfig,
    *,
    machine: str,
    split: str,
) -> Path:
    safe = f"{machine}__{config.protocol}__regime-{config.regime}__{split}__{config.config_id}.h5"
    return Path(root) / config.protocol / f"regime_{config.regime}" / safe


def validate_cache_identity(path: str | Path, config: PreprocessingConfig) -> None:
    with h5py.File(path, "r") as handle:
        observed = str(handle.attrs.get("preprocessing_config_id", ""))
        if observed != config.config_id:
            raise ValueError(
                f"incompatible cache: expected {config.config_id}, found {observed or '<missing>'}"
            )


class RaggedHDF5Writer:
    """Append modality-specific variable-length windows without padding them.

    Each modality owns a contiguous value/timestamp store and an offset table.
    A consumer can memory-map/chunk one modality without loading the corpus or
    forcing other modalities to the same sample length.
    """

    def __init__(
        self,
        path: str | Path,
        config: PreprocessingConfig,
        *,
        machine: str,
        split: str,
    ) -> None:
        self.path = Path(path)
        self.config = config
        self.machine = machine
        self.split = split
        self._handle: h5py.File | None = None
        self._groups: dict[str, h5py.Group] = {}
        self.window_count = 0

    def __enter__(self) -> "RaggedHDF5Writer":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents a partial or old cache from being
        # silently overwritten and mistaken for the current configuration.
        self._handle = h5py.File(self.path, "x")
        self._handle.attrs.update({
            "preprocessing_config_id": self.config.config_id,
            "preprocessing_config_json": json.dumps(self.config.canonical_dict(), sort_keys=True),
            "protocol": self.config.protocol,
            "regime": self.config.regime,
            "machine": self.machine,
            "split": self.split,
            "format": self.config.cache_format,
        })
        self._handle.create_dataset(
            "window_id", shape=(0,), maxshape=(None,), chunks=True,
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        return self

    def _group(self, signal: SensorSignal) -> h5py.Group:
        if self._handle is None:
            raise RuntimeError("cache writer is not open")
        if signal.sensor in self._groups:
            group = self._groups[signal.sensor]
            if group["values"].shape[1] != signal.values.shape[1]:
                raise ValueError("channel count changed inside a modality cache")
            return group
        group = self._handle.create_group(signal.sensor)
        channels = signal.values.shape[1]
        group.attrs["channels_json"] = json.dumps(signal.channels)
        group.create_dataset(
            "values", shape=(0, channels), maxshape=(None, channels),
            chunks=(min(8192, max(1, len(signal.values))), channels), dtype="f4",
        )
        group.create_dataset(
            "timestamps", shape=(0,), maxshape=(None,), chunks=True, dtype="f8",
        )
        group.create_dataset(
            "offsets", shape=(0, 2), maxshape=(None, 2), chunks=True, dtype="i8",
        )
        group.create_dataset(
            "lengths", shape=(0,), maxshape=(None,), chunks=True, dtype="i8",
        )
        self._groups[signal.sensor] = group
        return group

    def append(self, window_id: str, signals: dict[str, SensorSignal]) -> None:
        if self._handle is None:
            raise RuntimeError("cache writer is not open")
        if set(signals) != {"mic", "acc", "gyro"}:
            raise ValueError("every cached window must retain all modalities")
        ids = self._handle["window_id"]
        ids.resize((len(ids) + 1,))
        ids[-1] = window_id
        for signal in signals.values():
            group = self._group(signal)
            values = group["values"]
            timestamps = group["timestamps"]
            offsets = group["offsets"]
            lengths = group["lengths"]
            start = len(values)
            stop = start + len(signal.values)
            values.resize((stop, values.shape[1]))
            timestamps.resize((stop,))
            values[start:stop] = signal.values.astype(np.float32, copy=False)
            timestamps[start:stop] = signal.timestamps
            offsets.resize((len(offsets) + 1, 2))
            lengths.resize((len(lengths) + 1,))
            offsets[-1] = (start, stop)
            lengths[-1] = len(signal.values)
        self.window_count += 1

    def __exit__(self, *_exc) -> None:
        if self._handle is not None:
            self._handle.attrs["window_count"] = self.window_count
            self._handle.flush()
            self._handle.close()


def read_window(path: str | Path, sensor: str, index: int) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        if is_protected_split(handle.attrs.get("split")):
            require_protected_read(f"ragged cache window {Path(path).name}")
        group = handle[sensor]
        start, stop = map(int, group["offsets"][index])
        return group["timestamps"][start:stop], group["values"][start:stop]
