"""Multi-Sensor Fusion entry hygiene: protected common-evaluation data cannot be read in development.

Single-Modality Benchmarking materialised the common-evaluation spectral caches
and common-test feature artifacts on disk, so "the file does not exist" no
longer protects them.  These tests show every repository reader refuses
test-role contents outside an authorized scope - before the file is opened
where the role is known from the request, and before any dataset is read where
only the stored split reveals it - that refusals mutate nothing, that
training/source-validation reads still work, and that the authorized protected
path is preserved.  The real protected caches are touched only to prove the
refusal; their contents are never opened.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import h5py
import numpy as np
import pandas as pd
import pytest

from src.experiment import phase4 as p4
from src.experiment.phase32 import load_common_test_artifact, prepare_common_test_artifact
from src.preprocessing import spectral_cache as sc
from src.preprocessing.cache import RaggedHDF5Writer, read_window
from src.preprocessing.config import load_config
from src.preprocessing.pipeline import iter_processed_windows
from src.preprocessing.protected_access import (
    PROTECTED_SPECTRAL_CACHE_NAME,
    ProtectedReadRefused,
    active_protected_read,
    authorized_protected_read,
)
from src.preprocessing.signals import SensorSignal
from src.preprocessing.spectral_cache import (
    SpectralCacheWriter,
    build_protected_spectral_cache,
    load_spectral_cache,
    load_spectral_cache_metadata,
    spectral_cache_path,
)

REAL_SPECTRAL_ROOT = Path("data/interim/phase4/spectral")
REAL_COMMON_TEST_ROOT = Path("data/interim/phase32_common_test")
REGIME_T = "configs/preprocessing/mechasight_few_shot_regime_t.json"
DIMS = {"mic": 3, "acc": 4, "gyro": 4}
POOL = {"train": "source_training", "validation": "source_validation", "test": "common_evaluation"}
REAL_H5_FILE = h5py.File


def _spectral(path: Path, split: str, windows: int = 3) -> Path:
    """A tiny synthetic spectral cache with the frozen layout."""

    with SpectralCacheWriter(path, machine="RoboticArm", regime="T", split=split,
                             preprocessing_config_id="cfg", split_version="v3",
                             input_dims=DIMS) as writer:
        for index in range(windows):
            writer.append(
                {"window_id": f"w{index}", "window_index": index, "segment_id": f"s{index // 2}",
                 "parent_recording": "p0", "domain": "source", "data_pool": POOL[split],
                 "split": split, "anomaly_label": "normal"},
                {m: np.full(d, index, dtype=np.float32) for m, d in DIMS.items()},
            )
    return path


def _metadata(path: Path) -> None:
    from src.experiment.provenance import file_sha256

    path.with_suffix(".metadata.json").write_text(
        json.dumps({"cache_sha256": file_sha256(path)}), encoding="utf-8")


def _forbid_h5_open(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("protected file was opened")
    monkeypatch.setattr(sc.h5py, "File", refuse)


class _RecordingFile:
    """h5py.File stand-in that records which datasets a reader touches."""

    touched: list[str] = []

    def __init__(self, *args, **kwargs):
        self._file = REAL_H5_FILE(*args, **kwargs)
        self.attrs = self._file.attrs

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._file.close()

    def __getitem__(self, key):
        _RecordingFile.touched.append(key)
        return self._file[key]


# ----------------------------------------- A/B. development reads are refused
def test_development_read_of_an_existing_protected_spectral_cache_is_refused(tmp_path, monkeypatch):
    path = _spectral(tmp_path / PROTECTED_SPECTRAL_CACHE_NAME, "test")
    assert path.exists() and active_protected_read() is None
    _forbid_h5_open(monkeypatch)                 # refusal must precede opening the file
    for modality in DIMS:
        with pytest.raises(ProtectedReadRefused, match="outside an authorized"):
            load_spectral_cache(path, modality=modality)


def test_requesting_the_test_split_is_refused_whatever_the_file_is_called(tmp_path, monkeypatch):
    path = _spectral(tmp_path / "anything.spectral.h5", "test")
    _forbid_h5_open(monkeypatch)
    with pytest.raises(ProtectedReadRefused):
        load_spectral_cache(path, modality="acc", expected_split="test")


def test_a_renamed_protected_cache_is_refused_before_any_dataset_is_read(tmp_path, monkeypatch):
    path = _spectral(tmp_path / "training.spectral.h5", "test")   # misleading name, test contents
    _RecordingFile.touched = []
    monkeypatch.setattr(sc.h5py, "File", _RecordingFile)
    with pytest.raises(ProtectedReadRefused, match="stored split=test"):
        load_spectral_cache(path, modality="mic", expected_split=None)
    assert _RecordingFile.touched == []


def test_protected_metadata_and_label_free_loaders_refuse_on_existing_artifacts(tmp_path, monkeypatch):
    cache = spectral_cache_path(tmp_path, machine="RoboticArm", regime="T", role="common_evaluation")
    cache.parent.mkdir(parents=True)
    _metadata(_spectral(cache, "test"))
    feature_dir = tmp_path / "features" / "RoboticArm" / "T"
    feature_dir.mkdir(parents=True)
    (feature_dir / "common_test.parquet").write_bytes(b"not read")
    (feature_dir / "common_test.metadata.json").write_text("{}", encoding="utf-8")
    _forbid_h5_open(monkeypatch)
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("protected parquet was read")))

    with pytest.raises(ProtectedReadRefused):
        load_spectral_cache_metadata(tmp_path, machine="RoboticArm", regime="T", role="common_evaluation")
    with pytest.raises(ProtectedReadRefused):
        p4.load_label_free_test_windows(tmp_path, machine="RoboticArm", regime="T", contract=None)
    with pytest.raises(ProtectedReadRefused):
        p4.load_label_free_test_features(tmp_path / "features", machine="RoboticArm", regime="T", contract=None)
    with pytest.raises(ProtectedReadRefused):
        load_common_test_artifact(tmp_path / "features", machine="RoboticArm", regime="T", contract=None)


def test_protected_builders_refuse_before_touching_any_input(tmp_path):
    missing = tmp_path / "does-not-exist"
    with pytest.raises(ProtectedReadRefused):
        build_protected_spectral_cache(data_root=missing, split_manifest_path=missing,
                                       preprocessing_config_path=missing, cache_root=tmp_path,
                                       machine="RoboticArm")
    with pytest.raises(ProtectedReadRefused):
        prepare_common_test_artifact(data_root=missing, split_manifest_path=missing,
                                     preprocessing_config_path=missing, artifact_root=tmp_path,
                                     machine="RoboticArm", test_evaluation_code_sha="x")
    assert list(tmp_path.iterdir()) == []


def test_raw_test_windows_are_refused_at_call_time_not_at_first_iteration():
    config = load_config(REGIME_T)
    empty = pd.DataFrame(columns=["split", "machine", "regime", "protocol",
                                  "parent_recording", "segment_id"])
    with pytest.raises(ProtectedReadRefused, match="raw test windows"):
        iter_processed_windows(empty, config, split="test")     # no next(): refusal is eager
    assert list(iter_processed_windows(empty, config, split="train")) == []
    assert list(iter_processed_windows(empty, config, split="validation")) == []


def test_ragged_cache_test_windows_are_refused_and_train_windows_read(tmp_path):
    config = load_config("configs/preprocessing/mechasight_regime_t.json")

    def signals():
        return {sensor: SensorSignal(sensor, np.arange(4) * 0.01, np.ones((4, channels)),
                                     tuple(f"c{i}" for i in range(channels)), 4, 4)
                for sensor, channels in (("mic", 1), ("acc", 3), ("gyro", 3))}

    for split in ("train", "test"):
        with RaggedHDF5Writer(tmp_path / f"{split}.h5", config, machine="RoboticArm", split=split) as writer:
            writer.append("w0", signals())
    _timestamps, values = read_window(tmp_path / "train.h5", "acc", 0)
    assert values.shape == (4, 3)
    with pytest.raises(ProtectedReadRefused, match="ragged cache"):
        read_window(tmp_path / "test.h5", "acc", 0)


# ------------------------- B/C. the real materialised caches: refused, unchanged
@pytest.mark.skipif(not any(REAL_SPECTRAL_ROOT.rglob(PROTECTED_SPECTRAL_CACHE_NAME)),
                    reason="no materialised protected spectral cache on this machine")
def test_real_materialised_protected_inputs_are_refused_without_being_opened(monkeypatch, protected_state):
    caches = sorted(REAL_SPECTRAL_ROOT.rglob(PROTECTED_SPECTRAL_CACHE_NAME))
    assert len(caches) == 4 and all(path.is_file() for path in caches)
    _forbid_h5_open(monkeypatch)
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("protected parquet was read")))
    for cache in caches:
        machine, regime = cache.parent.parent.name, cache.parent.name
        for modality in DIMS:
            with pytest.raises(ProtectedReadRefused):
                load_spectral_cache(cache, modality=modality)
        with pytest.raises(ProtectedReadRefused):
            load_spectral_cache_metadata(REAL_SPECTRAL_ROOT, machine=machine, regime=regime,
                                         role="common_evaluation")
        with pytest.raises(ProtectedReadRefused):
            p4.load_label_free_test_windows(REAL_SPECTRAL_ROOT, machine=machine, regime=regime, contract=None)
        with pytest.raises(ProtectedReadRefused):
            p4.load_label_free_test_features(REAL_COMMON_TEST_ROOT, machine=machine, regime=regime,
                                             contract=None)
        with pytest.raises(ProtectedReadRefused):
            load_common_test_artifact(REAL_COMMON_TEST_ROOT, machine=machine, regime=regime, contract=None)
    protected_state()


# --------------------------------------------- D. development reads still work
def test_training_and_source_validation_caches_read_normally_outside_any_scope(tmp_path):
    assert active_protected_read() is None
    for role, split in (("training", "train"), ("source_validation", "validation")):
        path = spectral_cache_path(tmp_path, machine="RoboticArm", regime="T", role=role)
        path.parent.mkdir(parents=True, exist_ok=True)
        _metadata(_spectral(path, split))
        X, meta = load_spectral_cache(path, modality="acc", expected_split=split)
        assert X.shape == (3, DIMS["acc"]) and set(meta["split"]) == {split}
        assert load_spectral_cache_metadata(tmp_path, machine="RoboticArm", regime="T", role=role)


# ------------------------------------- authorized protected path is preserved
def test_an_authorized_scope_admits_the_protected_read_and_then_closes(tmp_path):
    path = _spectral(tmp_path / PROTECTED_SPECTRAL_CACHE_NAME, "test")
    with authorized_protected_read(authority="p5-test-batch-fixture", purpose="unit test") as scope:
        assert active_protected_read() == scope
        X, meta = load_spectral_cache(path, modality="gyro", expected_split="test")
        assert X.shape == (3, DIMS["gyro"]) and set(meta["data_pool"]) == {"common_evaluation"}
    assert active_protected_read() is None
    with pytest.raises(ProtectedReadRefused):
        load_spectral_cache(path, modality="gyro")


def test_a_scope_closes_even_when_the_protected_stage_raises(tmp_path):
    with pytest.raises(RuntimeError, match="stage failed"):
        with authorized_protected_read(authority="p5-test-batch-fixture", purpose="unit test"):
            raise RuntimeError("stage failed")
    assert active_protected_read() is None


@pytest.mark.parametrize("authority, purpose", [("", "x"), ("  ", "x"), ("batch", ""), ("batch", " ")])
def test_a_scope_needs_a_named_authority_and_purpose(authority, purpose):
    with pytest.raises(ValueError):
        with authorized_protected_read(authority=authority, purpose=purpose):
            pass


def test_scopes_do_not_nest():
    with authorized_protected_read(authority="a", purpose="outer"):
        with pytest.raises(RuntimeError, match="do not nest"):
            with authorized_protected_read(authority="b", purpose="inner"):
                pass
    assert active_protected_read() is None


# --------------------------------------------------- repository tripwires
SOURCE_ROOTS = (Path("src"), Path("scripts"))
DIRECT_READ = re.compile(r"h5py\.File\(|read_parquet\(|read_table\(|ParquetFile\(|read_hdf\(")
# Reviewed direct HDF5/parquet readers at the Multi-Sensor Fusion entry.  Each
# reads raw recordings, non-test caches, or test data inside its own protected
# stage.  New code must use the guarded loaders; a new direct reader needs a
# reviewed entry here.
DIRECT_READERS = {
    "scripts/audio_noise_probe.py": 1,
    "scripts/authorize_protocol_a_test.py": 1,
    "scripts/investigate_artifacts.py": 1,
    "scripts/run_phase4_protected.py": 1,
    "scripts/run_protocol_a_reproduction.py": 5,
    "src/data/audit.py": 1,
    "src/experiment/focal_toggle_execution.py": 4,
    "src/experiment/phase31.py": 1,
    "src/experiment/phase32.py": 1,
    "src/experiment/phase4.py": 1,
    # Phase 5.5B: projected parquet + numeric-only HDF5; dedicated temp paths,
    # durable fixture capabilities; no production protected scope enabled.
    "src/experiment/phase5_inputs.py": 3,   # 5.5F: + development-schema numeric HDF5 window reader
    "src/experiment/phase5_fixture.py": 1,  # writes synthetic HDF5 fixtures only
    # Phase 5.5F: raw sensor TIMESTAMP column only (development cohorts; protected
    # construction disabled), and the development fixture builder: projected
    # source-validation feature columns, numeric source-validation cache rows, and
    # writing the sandbox HDF5 copy.
    "src/experiment/phase5_window_index.py": 1,
    "src/experiment/phase5_dev_fixture.py": 3,
    # 5.6B: registry-bound development/synthetic projections and numeric HDF5;
    # protected interface always refuses, before any content hash or read.
    "src/experiment/phase5_input_manifest.py": 3,
    "src/experiment/store.py": 1,          # Phase 5.2: guarded ResultStore.read_scores
    "src/fusion/l3.py": 1,                 # Phase 5.4: raw development segments (cohort guard)
    "src/fusion/l3_falsification.py": 1,   # Phase 5.4: clean source of a corrupted development copy
    "src/preprocessing/cache.py": 3,
    "src/preprocessing/signals.py": 1,
    "src/preprocessing/spectral_cache.py": 2,
    "src/reproduction/protocol_a.py": 1,
}
# Every place that opens an authorized protected-read scope.  Granting a scope
# is a protected-evaluation decision, never a development convenience.
AUTHORIZED_SCOPES = {
    "scripts/run_phase32_first_test.py": 1,
    "scripts/run_phase4_protected.py": 2,
}


def _count(pattern: re.Pattern, exclude: str | None = None) -> dict[str, int]:
    counts = {}
    for root in SOURCE_ROOTS:
        for path in sorted(root.rglob("*.py")):
            name = path.as_posix()
            if name == exclude:
                continue
            found = len(pattern.findall(path.read_text(encoding="utf-8")))
            if found:
                counts[name] = found
    return counts


def test_no_new_direct_hdf5_or_parquet_reader_bypasses_the_guarded_loaders():
    assert _count(DIRECT_READ) == DIRECT_READERS


def test_only_reviewed_protected_stages_open_an_authorized_read_scope():
    scopes = _count(re.compile(r"authorized_protected_read\("),
                    exclude="src/preprocessing/protected_access.py")
    assert scopes == AUTHORIZED_SCOPES


# --------------------------------------------- the snapshot itself is sensitive
def test_protected_state_snapshot_detects_edits_additions_and_removals(tmp_path, monkeypatch):
    import conftest as harness

    ledger = tmp_path / "ledger.jsonl"
    ledger.write_bytes(b"{}\n")
    results, pretest, protected, spectral, features = (tmp_path / n for n in
                                                       ("r", "pt", "pr", "sp", "ft"))
    for root in (results, pretest, protected, spectral, features):
        root.mkdir()
    (protected / "score.csv").write_text("a\n", encoding="utf-8")
    (spectral / "c.h5").write_bytes(b"x")
    for name, value in (("REAL_LEDGER", ledger), ("REAL_RESULTS_ROOT", results),
                        ("REAL_PHASE4_PRETEST_ROOT", pretest), ("REAL_PHASE4_PROTECTED_ROOT", protected),
                        ("REAL_PHASE4_SPECTRAL_ROOT", spectral),
                        ("REAL_COMMON_TEST_FEATURE_ROOT", features)):
        monkeypatch.setattr(harness, name, value)

    base = harness.snapshot_protected_state()
    assert harness.snapshot_protected_state() == base
    for mutate in (lambda: ledger.write_bytes(b"{}\n{}\n"),
                   lambda: (protected / "score.csv").write_text("b\n", encoding="utf-8"),
                   lambda: (protected / "new.csv").write_text("", encoding="utf-8"),
                   lambda: (spectral / "c.h5").write_bytes(b"xy"),
                   lambda: (spectral / "c.h5").unlink()):
        mutate()
        assert harness.snapshot_protected_state() != base
        base = harness.snapshot_protected_state()


# --------------------------------------- Phase 5.5C: executor read capabilities
def test_only_the_executor_constructs_read_capabilities_and_projected_readers():
    """Access objects and ProjectedInputs are built only inside phase5_execution.py.

    A caller-built Access is refused at runtime (it is not in the issuing
    executor's capability table); this tripwire keeps new code from even trying,
    and from building a ProjectedInputs with its own verifier.
    """
    access = _count(re.compile(r"\bAccess\("), exclude="src/experiment/phase5_inputs.py")
    readers = _count(re.compile(r"\bProjectedInputs\("), exclude="src/experiment/phase5_inputs.py")
    issued = _count(re.compile(r"_capabilities\b"))
    assert access == {"src/experiment/phase5_execution.py": 1}
    assert readers == {"src/experiment/phase5_execution.py": 1}
    assert set(issued) == {"src/experiment/phase5_execution.py"}
    from dataclasses import fields
    from src.experiment.phase5_inputs import Access
    assert [f.name for f in fields(Access)] == ["executor_identity", "sandbox_identity", "batch_id", "phase", "token"]
