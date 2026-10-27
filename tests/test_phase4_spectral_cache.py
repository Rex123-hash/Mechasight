"""Single-Modality Benchmarking: spectral window cache persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.spectral_cache import (
    SPECTRAL_CACHE_FORMAT,
    SpectralCacheWriter,
    assert_pretest_cache_role,
    load_spectral_cache,
    spectral_cache_path,
)


def _record(index: int, segment: str, label: str = "normal") -> dict:
    return {
        "window_id": f"{segment}__w{index}",
        "window_index": index,
        "segment_id": segment,
        "parent_recording": segment[:8],
        "domain": "source",
        "data_pool": "source_training",
        "split": "train",
        "anomaly_label": label,
        "machine": "RoboticArm",
        "regime": "S",
        "protocol": "mechasight_experimental_protocol",
        "preprocessing_config_id": "cfg",
        "split_version": "v3",
    }


def _write(tmp_path, dims=(4, 6, 6)):
    path = tmp_path / "cache.h5"
    rng = np.random.default_rng(0)
    rows = []
    with SpectralCacheWriter(
        path, machine="RoboticArm", regime="S", split="train",
        preprocessing_config_id="cfg", split_version="v3",
        input_dims={"mic": dims[0], "acc": dims[1], "gyro": dims[2]},
    ) as writer:
        for segment in ("20240101_a", "20240101_b"):
            for index in range(3):
                vectors = {
                    "mic": rng.normal(size=dims[0]),
                    "acc": rng.normal(size=dims[1]),
                    "gyro": rng.normal(size=dims[2]),
                }
                writer.append(_record(index, segment), vectors)
                rows.append((segment, index, vectors))
    return path, rows


def test_cache_round_trip_preserves_vectors_and_window_order(tmp_path):
    path, rows = _write(tmp_path)
    X, meta = load_spectral_cache(path, modality="acc")
    assert X.shape == (6, 6)
    assert X.dtype == np.float32
    assert list(meta.columns) >= ["window_id", "segment_id", "parent_recording", "domain"]
    for position, (segment, index, vectors) in enumerate(rows):
        assert meta.loc[position, "segment_id"] == segment
        assert meta.loc[position, "window_index"] == index
        np.testing.assert_allclose(X[position], vectors["acc"], rtol=1e-6)


def test_cache_records_identity_attributes(tmp_path):
    path, _ = _write(tmp_path)
    _, meta = load_spectral_cache(path, modality="mic")
    assert meta.attrs["machine"] == "RoboticArm"
    assert meta.attrs["regime"] == "S"
    assert meta.attrs["split"] == "train"
    assert meta.attrs["cache_format"] == SPECTRAL_CACHE_FORMAT
    assert meta.attrs["input_dims"] == {"mic": 4, "acc": 6, "gyro": 6}


def test_writer_refuses_wrong_dimension(tmp_path):
    path = tmp_path / "bad.h5"
    with SpectralCacheWriter(
        path, machine="RoboticArm", regime="S", split="train",
        preprocessing_config_id="cfg", split_version="v3",
        input_dims={"mic": 4, "acc": 6, "gyro": 6},
    ) as writer:
        writer.append(_record(0, "seg"), {
            "mic": np.zeros(4), "acc": np.zeros(6), "gyro": np.zeros(6),
        })
        with pytest.raises(ValueError, match="dimension"):
            writer.append(_record(1, "seg"), {
                "mic": np.zeros(5), "acc": np.zeros(6), "gyro": np.zeros(6),
            })


def test_writer_refuses_duplicate_window_id(tmp_path):
    path = tmp_path / "dup.h5"
    vectors = {"mic": np.zeros(4), "acc": np.zeros(6), "gyro": np.zeros(6)}
    with SpectralCacheWriter(
        path, machine="RoboticArm", regime="S", split="train",
        preprocessing_config_id="cfg", split_version="v3",
        input_dims={"mic": 4, "acc": 6, "gyro": 6},
    ) as writer:
        writer.append(_record(0, "seg"), vectors)
        with pytest.raises(ValueError, match="duplicate"):
            writer.append(_record(0, "seg"), vectors)


def test_loader_refuses_split_mismatch(tmp_path):
    path, _ = _write(tmp_path)
    with pytest.raises(ValueError, match="split"):
        load_spectral_cache(path, modality="mic", expected_split="validation")


def test_pretest_cache_role_guard_refuses_test():
    assert assert_pretest_cache_role("training") == "train"
    assert assert_pretest_cache_role("source_validation") == "validation"
    with pytest.raises(ValueError, match="pre-test"):
        assert_pretest_cache_role("common_evaluation")


def test_cache_path_layout(tmp_path):
    path = spectral_cache_path(tmp_path, machine="BrushlessMotor", regime="T", role="training")
    assert path == tmp_path / "BrushlessMotor" / "T" / "training.spectral.h5"


RAW_ROOT = Path("data/raw/extracted")
SPLIT_MANIFEST = Path("reports/phase2_2/protocol_b_few_shot_adaptation_v3.csv")
REGIME_S = Path("configs/preprocessing/mechasight_few_shot_regime_s.json")


@pytest.mark.skipif(not RAW_ROOT.exists(), reason="IMAD-DS raw data not present")
def test_builder_materializes_training_windows_with_frozen_dimensions(tmp_path):
    from src.preprocessing.spectral_cache import build_pretest_spectral_cache

    metadata = build_pretest_spectral_cache(
        data_root=RAW_ROOT, split_manifest_path=SPLIT_MANIFEST,
        preprocessing_config_path=REGIME_S, cache_root=tmp_path,
        machine="RoboticArm", role="training", max_segments=2,
    )
    assert metadata["role"] == "training" and metadata["split"] == "train"
    assert metadata["input_dims"] == {"mic": 800, "acc": 1059, "gyro": 1059}
    assert metadata["segment_count"] == 2 and metadata["window_count"] == 148
    assert metadata["test_evaluated"] is False
    X, meta = load_spectral_cache(Path(metadata["cache_path"]), modality="gyro", expected_split="train")
    assert X.shape == (148, 1059)
    assert set(meta["anomaly_label"]) == {"normal"}
    assert set(meta["data_pool"]) == {"source_training"}
    assert meta["segment_id"].nunique() == 2


def test_builder_refuses_protected_role(tmp_path):
    from src.preprocessing.spectral_cache import build_pretest_spectral_cache

    with pytest.raises(ValueError, match="pre-test"):
        build_pretest_spectral_cache(
            data_root=RAW_ROOT, split_manifest_path=SPLIT_MANIFEST,
            preprocessing_config_path=REGIME_S, cache_root=tmp_path,
            machine="RoboticArm", role="common_evaluation",
        )
