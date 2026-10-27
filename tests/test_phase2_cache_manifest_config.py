from __future__ import annotations

from dataclasses import replace
import json

import h5py
import numpy as np
import pandas as pd
import pytest

from src.preprocessing.cache import RaggedHDF5Writer, read_window, validate_cache_identity
from src.preprocessing.config import load_config
from src.preprocessing.manifest import REQUIRED_COLUMNS, validate_manifest
from src.preprocessing.signals import SensorSignal


def _signals(length: int = 4):
    output = {}
    for sensor, channels in (("mic", 1), ("acc", 3), ("gyro", 3)):
        t = np.arange(length, dtype=float) * 0.01
        output[sensor] = SensorSignal(
            sensor, t, np.ones((length, channels)), tuple(f"c{i}" for i in range(channels)),
            length, length,
        )
    return output


def test_config_identity_changes_for_every_cache_relevant_choice():
    base = load_config("configs/preprocessing/mechasight_regime_t.json")
    variants = [
        replace(base, regime="S"),
        replace(base, window_length_s=0.2),
        replace(base, stride_s=0.05),
        replace(base, gyro_warmup_s=0.04),
        replace(base, normalization="train_zscore"),
        replace(base, features=("time",)),
        replace(base, drop_initial_acc_samples={"RoboticArm": 0, "BrushlessMotor": 0}),
    ]
    assert all(variant.config_id != base.config_id for variant in variants)


def test_ragged_cache_round_trip_and_identity(tmp_path):
    config = load_config("configs/preprocessing/mechasight_regime_t.json")
    path = tmp_path / "cache.h5"
    with RaggedHDF5Writer(path, config, machine="RoboticArm", split="train") as writer:
        writer.append("w0", _signals(4))
        writer.append("w1", _signals(6))
    validate_cache_identity(path, config)
    timestamps, values = read_window(path, "acc", 1)
    assert len(timestamps) == 6
    assert values.shape == (6, 3)
    with h5py.File(path, "r") as handle:
        assert int(handle.attrs["window_count"]) == 2


def test_cache_identity_rejects_incompatible_config(tmp_path):
    config = load_config("configs/preprocessing/mechasight_regime_t.json")
    path = tmp_path / "cache.h5"
    with RaggedHDF5Writer(path, config, machine="RoboticArm", split="train"):
        pass
    with pytest.raises(ValueError, match="incompatible cache"):
        validate_cache_identity(path, replace(config, stride_s=0.05))


def test_manifest_traceability_schema():
    record = {column: "value" for column in REQUIRED_COLUMNS}
    record.update({
        "window_id": "unique",
        "window_start_s": 1.0,
        "window_end_s": 1.1,
        "sensor_modalities": json.dumps(["mic", "acc", "gyro"]),
        "raw_file_references": json.dumps({"mic": "raw.parquet"}),
        "sampling_rate_metadata": json.dumps({"mic": {"rate": 16000}}),
    })
    frame = pd.DataFrame([record])
    validate_manifest(frame)


def test_manifest_rejects_duplicate_window_ids():
    record = {column: "value" for column in REQUIRED_COLUMNS}
    record.update({
        "window_id": "duplicate", "window_start_s": 1.0, "window_end_s": 1.1,
        "sensor_modalities": '["mic"]', "raw_file_references": '{"mic":"x"}',
        "sampling_rate_metadata": '{"mic":{"rate":1}}',
    })
    with pytest.raises(ValueError, match="globally unique"):
        validate_manifest(pd.DataFrame([record, record]))
