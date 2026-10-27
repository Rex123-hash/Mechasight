"""Multi-Sensor Fusion data foundation (Phase 5.2): development-only assembly contract.

Real-data tests use training/source-validation artifacts only (validation roles
for the spectral caches, to keep memory small; the training caches are verified
by the same code and recorded in docs/phase5_2_data_foundation.md).  Every
refusal path is exercised on synthetic fixtures.  No protected input, sealed
score or test label is read.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from src.experiment.provenance import file_sha256, ordered_id_hash
from src.fusion import assembly as fa
from src.fusion.assembly import FusionAssemblyError
from src.preprocessing.spectral_cache import (
    PROVENANCE_COLUMNS,
    SpectralCacheWriter,
    spectral_cache_path,
)

FEATURE_ROOT = Path("data/interim/phase31_primary")
SPECTRAL_ROOT = Path("data/interim/phase4/spectral")
PRETEST_ROOT = Path("reports/phase4_pretest/experiments")
HAVE_FEATURES = (FEATURE_ROOT / "RoboticArm" / "S" / "training.parquet").exists()
HAVE_SPECTRA = (SPECTRAL_ROOT / "RoboticArm" / "S" / "source_validation.spectral.h5").exists()
CELLS = [(m, r) for m in fa.MACHINES for r in fa.REGIMES]


def _phase4_provenance(machine: str, regime: str, family: str, modality: str = "acc") -> dict:
    unit = next(u for u in fa.load_design_registry()
                if (u["machine"], u["regime"], u["family"]) == (machine, regime, family))
    eid = unit["phase4_single_modality_ids"][modality]
    return json.loads((PRETEST_ROOT / eid / "provenance.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ registry
def test_design_registry_is_the_reviewed_v2_matrix():
    units = fa.load_design_registry()
    assert len(units) == 52
    assert sum(u["l1_source"] == "reuse_phase3_2_sealed" for u in units) == 28
    assert {u["family"] for u in units if u["l1_source"] == "new_fit_phase5"} == {
        "mahalanobis_shrinkage", "dense_autoencoder"}
    assert len({i for u in units for i in u["phase4_single_modality_ids"].values()}) == 156


def test_a_changed_design_config_is_refused(tmp_path):
    altered = tmp_path / "fusion_design_v2.json"
    altered.write_text(Path(fa.DESIGN_CONFIG).read_text(encoding="utf-8").replace('"seed": 4', '"seed": 5', 1),
                       encoding="utf-8")
    with pytest.raises(FusionAssemblyError, match="differs from the reviewed version"):
        fa.load_design_registry(altered)


def test_reused_l1_units_resolve_to_matching_historical_bundles_by_config_only():
    for unit in fa.load_design_registry():
        eid = unit["l1_phase3_2_id"]
        if eid is None:
            continue
        for root in ("reports/phase3_1/validation_only", "reports/phase3_2/experiments"):
            config = json.loads((Path(root) / eid / "config.json").read_text(encoding="utf-8"))
            assert (config["machine"], config["regime"], config["model_family"], int(config["model_seed"])) == (
                unit["machine"], unit["regime"], unit["family"], unit["seed"])


# ---------------------------------------------------------------- role refusal
@pytest.mark.parametrize("role", ["common_evaluation", "test", "reserved_adaptation", "blocked_sibling", ""])
def test_every_non_development_role_is_refused_before_any_io(role, monkeypatch):
    def no_io(*_a, **_k):
        raise AssertionError("I/O attempted for a refused role")
    for name in ("load_feature_artifact", "load_spectral_cache", "load_spectral_cache_metadata"):
        monkeypatch.setattr(fa, name, no_io)
    for assemble in (fa.assemble_classical_fusion, fa.assemble_spectral_fusion):
        with pytest.raises(FusionAssemblyError, match="training/source_validation only"):
            assemble(Path("x"), machine="RoboticArm", regime="S", role=role)


# ------------------------------------------------------------- L1 classical
def test_classical_column_schema_validation():
    good = tuple(sorted([f"acc__f{i:02d}" for i in range(42)] + [f"gyro__f{i:02d}" for i in range(42)]
                        + [f"mic__f{i:02d}" for i in range(14)]))
    columns, modality = fa.classical_fusion_columns(good)
    assert len(columns) == 98 and modality.count("mic") == 14 and modality.count("acc") == 42
    with pytest.raises(FusionAssemblyError, match="duplicated"):
        fa.classical_fusion_columns(good[:-1] + (good[0],))
    with pytest.raises(FusionAssemblyError, match="sorted order"):
        fa.classical_fusion_columns(tuple(reversed(good)))
    with pytest.raises(FusionAssemblyError, match="14/42/42"):
        fa.classical_fusion_columns(good[1:])
    with pytest.raises(FusionAssemblyError):
        fa.classical_fusion_columns(tuple(sorted(good + ("temp__x",))))


@pytest.mark.skipif(not HAVE_FEATURES, reason="development feature artifacts not present")
@pytest.mark.parametrize("machine, regime", CELLS)
@pytest.mark.parametrize("role", ["training", "source_validation"])
def test_real_classical_assembly_matches_the_frozen_cohorts(machine, regime, role):
    data = fa.assemble_classical_fusion(FEATURE_ROOT, machine=machine, regime=regime, role=role)
    assert data.X.shape[1] == 98 and data.X.dtype == np.float64 and np.isfinite(data.X).all()
    assert data.columns == tuple(sorted(data.columns)) and len(set(data.columns)) == 98
    assert [data.modality_of_column.count(m) for m in fa.MODALITIES] == [14, 42, 42]
    assert data.segments["segment_id"].is_unique and len(data.segments) == len(data.X)
    assert set(data.segments["data_pool"]) <= fa.ALLOWED_POOLS[(role, regime)]
    if regime == "T" and role == "training":
        assert (data.segments["data_pool"] == "target_adaptation").sum() == {"RoboticArm": 27, "BrushlessMotor": 18}[machine]
    key = "training_cohort_hash" if role == "training" else "validation_cohort_hash"
    assert data.cohort_hash == _phase4_provenance(machine, regime, "one_class_svm")[key]


@pytest.mark.skipif(not HAVE_FEATURES, reason="development feature artifacts not present")
def test_classical_assembly_is_deterministic():
    a = fa.assemble_classical_fusion(FEATURE_ROOT, machine="BrushlessMotor", regime="T", role="training")
    b = fa.assemble_classical_fusion(FEATURE_ROOT, machine="BrushlessMotor", regime="T", role="training")
    assert np.array_equal(a.X, b.X) and a.columns == b.columns and a.segments.equals(b.segments)


def _synthetic_feature_root(tmp_path: Path, *, mutate=None, role: str = "source_validation") -> Path:
    columns = sorted([f"acc__f{i:02d}" for i in range(42)] + [f"gyro__f{i:02d}" for i in range(42)]
                     + [f"mic__f{i:02d}" for i in range(14)])
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(4, 98)), columns=columns)
    frame.insert(0, "segment_id", [f"seg{i}" for i in range(4)])
    for column, value in (("machine", "RoboticArm"), ("regime", "S"), ("split", "validation"),
                          ("data_pool", "source_validation"), ("domain", "source"), ("parent_recording", "p0"),
                          ("anomaly_label", "normal"), ("window_count", 74)):
        frame[column] = value
    if mutate:
        frame = mutate(frame)
    base = tmp_path / "RoboticArm" / "S"
    base.mkdir(parents=True)
    frame.to_parquet(base / f"{role}.parquet", index=False)
    (base / f"{role}.metadata.json").write_text(json.dumps({
        "role": role, "test_evaluated": False, "artifact_sha256": file_sha256(base / f"{role}.parquet"),
        "segment_count": len(frame), "segment_cohort_hash": ordered_id_hash(tuple(frame["segment_id"])),
    }), encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("mutate, message", [
    (lambda f: f.assign(data_pool="common_evaluation"), "unauthorised data pools"),
    (lambda f: f.assign(split="train"), "expected 'validation'"),
    (lambda f: f.assign(anomaly_label="anomaly"), "normal-only"),
    (lambda f: f.assign(machine="BrushlessMotor"), "machine differs"),
    (lambda f: f.drop(columns=["gyro__f00"]), "14/42/42"),
])
def test_classical_assembly_refuses_contract_violations(tmp_path, mutate, message):
    root = _synthetic_feature_root(tmp_path, mutate=mutate)
    with pytest.raises(FusionAssemblyError, match=message):
        fa.assemble_classical_fusion(root, machine="RoboticArm", regime="S", role="source_validation")


def test_classical_assembly_accepts_a_valid_synthetic_artifact(tmp_path):
    data = fa.assemble_classical_fusion(_synthetic_feature_root(tmp_path), machine="RoboticArm",
                                        regime="S", role="source_validation")
    assert data.X.shape == (4, 98) and list(data.segments["segment_id"]) == [f"seg{i}" for i in range(4)]


# -------------------------------------------------------------- L1 spectral
def _synthetic_cache(root: Path, *, records=None, split: str = "validation", role: str = "source_validation",
                     corrupt=None) -> Path:
    dims = fa.SPECTRAL_DIMS["RoboticArm"]
    path = spectral_cache_path(root, machine="RoboticArm", regime="S", role=role)
    if records is None:
        records = [{"window_id": f"s{s}:{w}", "window_index": w, "segment_id": f"s{s}",
                    "parent_recording": "p0", "domain": "source", "data_pool": "source_validation",
                    "split": split, "anomaly_label": "normal"} for s in range(2) for w in range(3)]
    with SpectralCacheWriter(path, machine="RoboticArm", regime="S", split=split,
                             preprocessing_config_id="cfg", split_version="v3", input_dims=dims) as writer:
        for i, record in enumerate(records):
            writer.append(record, {m: np.full(d, float(i), dtype=np.float32) for m, d in dims.items()})
    if corrupt:
        with h5py.File(path, "r+") as handle:
            corrupt(handle)
    segments = sorted({r["segment_id"] for r in records})
    path.with_suffix(".metadata.json").write_text(json.dumps({
        "machine": "RoboticArm", "regime": "S", "role": role, "split": split, "input_dims": dims,
        "window_count": len(records), "segment_cohort_hash": ordered_id_hash(tuple(segments)),
        "cache_sha256": file_sha256(path),
    }), encoding="utf-8")
    return root


def test_spectral_assembly_concatenates_aligned_rows_in_mic_acc_gyro_order(tmp_path):
    data = fa.assemble_spectral_fusion(_synthetic_cache(tmp_path), machine="RoboticArm", regime="S",
                                       role="source_validation")
    assert data.X.shape == (6, 2918) and data.X.dtype == np.float32
    assert dict(data.blocks) == {"mic": (0, 800), "acc": (800, 1859), "gyro": (1859, 2918)}
    for row in range(6):                       # every block of row i came from writer row i
        assert np.all(data.X[row] == float(row))
    assert list(data.windows.columns) == list(PROVENANCE_COLUMNS)


@pytest.mark.parametrize("modality, perturb, message", [
    ("acc", lambda X, m: (X, m.assign(parent_recording=["p0", "p0", "other", "p0", "p0", "p0"])), "acc window provenance differs"),
    ("acc", lambda X, m: (X, m.assign(window_id=list(reversed(m["window_id"])))), "acc window provenance differs"),
    ("gyro", lambda X, m: (X[:-1], m.iloc[:-1]), "gyro window provenance differs"),
])
def test_spectral_assembly_refuses_provenance_or_row_count_mismatch(tmp_path, monkeypatch, modality, perturb, message):
    root = _synthetic_cache(tmp_path)
    real = fa.load_spectral_cache

    def perturbed(path, *, modality=None, expected_split=None, _target=modality):
        X, meta = real(path, modality=modality, expected_split=expected_split)
        return perturb(X, meta) if modality == _target else (X, meta)

    monkeypatch.setattr(fa, "load_spectral_cache", perturbed)
    with pytest.raises(FusionAssemblyError, match=message):
        fa.assemble_spectral_fusion(root, machine="RoboticArm", regime="S", role="source_validation")


def test_spectral_assembly_refuses_out_of_order_or_interleaved_windows(tmp_path):
    records = [{"window_id": f"s{s}:{w}", "window_index": w, "segment_id": f"s{s}", "parent_recording": "p0",
                "domain": "source", "data_pool": "source_validation", "split": "validation",
                "anomaly_label": "normal"} for w in range(2) for s in range(2)]        # interleaved
    with pytest.raises(FusionAssemblyError, match="not contiguous"):
        fa.assemble_spectral_fusion(_synthetic_cache(tmp_path / "a", records=records), machine="RoboticArm",
                                    regime="S", role="source_validation")
    records = [{**r, "window_index": 1 - r["window_index"]} for r in records if r["segment_id"] == "s0"]
    with pytest.raises(FusionAssemblyError, match="indexed 0..k-1"):
        fa.assemble_spectral_fusion(_synthetic_cache(tmp_path / "b", records=records), machine="RoboticArm",
                                    regime="S", role="source_validation")


def test_spectral_assembly_refuses_a_non_finite_or_missing_modality(tmp_path):
    def poison(handle):
        handle["acc"][1, 5] = np.nan
    with pytest.raises(ValueError):
        fa.assemble_spectral_fusion(_synthetic_cache(tmp_path / "nan", corrupt=poison), machine="RoboticArm",
                                    regime="S", role="source_validation")

    def drop_gyro(handle):
        del handle["gyro"]
    with pytest.raises(KeyError):           # a missing modality is an error, never silently filled
        fa.assemble_spectral_fusion(_synthetic_cache(tmp_path / "missing", corrupt=drop_gyro),
                                    machine="RoboticArm", regime="S", role="source_validation")


def test_spectral_assembly_refuses_a_foreign_pool(tmp_path):
    records = [{"window_id": "s0:0", "window_index": 0, "segment_id": "s0", "parent_recording": "p0",
                "domain": "target", "data_pool": "blocked_sibling", "split": "validation", "anomaly_label": "normal"}]
    with pytest.raises(FusionAssemblyError, match="unauthorised data pools"):
        fa.assemble_spectral_fusion(_synthetic_cache(tmp_path, records=records), machine="RoboticArm",
                                    regime="S", role="source_validation")


@pytest.mark.skipif(not (HAVE_SPECTRA and HAVE_FEATURES), reason="development caches not present")
@pytest.mark.parametrize("machine, regime", CELLS)
def test_real_validation_spectra_align_with_features_and_phase4_cohorts(machine, regime):
    data = fa.assemble_spectral_fusion(SPECTRAL_ROOT, machine=machine, regime=regime, role="source_validation")
    assert data.X.shape[1] == {"RoboticArm": 2918, "BrushlessMotor": 2936}[machine]
    assert data.windows["window_id"].is_unique and set(data.windows["split"]) == {"validation"}
    classical = fa.assemble_classical_fusion(FEATURE_ROOT, machine=machine, regime=regime, role="source_validation")
    assert set(data.windows["segment_id"]) == set(classical.segments["segment_id"])
    assert data.cohort_hash == _phase4_provenance(machine, regime, "dense_autoencoder")["validation_cohort_hash"]


@pytest.mark.skipif(not HAVE_SPECTRA, reason="development caches not present")
def test_spectral_assembly_is_deterministic():
    a = fa.assemble_spectral_fusion(SPECTRAL_ROOT, machine="BrushlessMotor", regime="S", role="source_validation")
    b = fa.assemble_spectral_fusion(SPECTRAL_ROOT, machine="BrushlessMotor", regime="S", role="source_validation")
    assert np.array_equal(a.X, b.X) and a.windows.equals(b.windows) and a.cache_sha256 == b.cache_sha256


# ------------------------------------------------------------ L2 score inputs
def test_all_52_units_join_three_identical_validation_cohorts():
    for unit in fa.load_design_registry():
        data = fa.assemble_unit_validation_scores(unit, PRETEST_ROOT)
        frame = data.scores
        assert list(frame.columns) == ["segment_id", "parent_recording", "domain", "mic", "acc", "gyro"]
        assert list(frame["segment_id"]) == sorted(frame["segment_id"]) and frame["segment_id"].is_unique
        assert len(frame) == {"RoboticArm": 186, "BrushlessMotor": 126}[unit["machine"]]
        assert np.isfinite(frame[["mic", "acc", "gyro"]].to_numpy()).all()
        assert set(frame["domain"]) == {"source"}


def test_unit_assembly_is_invariant_to_modality_enumeration_order():
    unit = fa.load_design_registry()[7]
    shuffled = {**unit, "phase4_single_modality_ids": dict(reversed(list(unit["phase4_single_modality_ids"].items())))}
    a = fa.assemble_unit_validation_scores(unit, PRETEST_ROOT).scores
    b = fa.assemble_unit_validation_scores(shuffled, PRETEST_ROOT).scores
    assert a.equals(b)


def test_unit_assembly_refuses_a_foreign_bundle_or_mismatched_segments(monkeypatch):
    units = fa.load_design_registry()
    unit, other = units[0], next(u for u in units if u["seed"] != units[0]["seed"] or u["family"] != units[0]["family"])
    foreign = {**unit, "phase4_single_modality_ids": {**unit["phase4_single_modality_ids"],
                                                      "acc": other["phase4_single_modality_ids"]["acc"]}}
    with pytest.raises(FusionAssemblyError, match="does not belong"):
        fa.assemble_unit_validation_scores(foreign, PRETEST_ROOT)

    real = pd.read_csv

    def drop_one_acc_row(path, *args, **kwargs):
        frame = real(path, *args, **kwargs)
        return frame.iloc[1:] if Path(path).parent.name == unit["phase4_single_modality_ids"]["acc"] else frame

    monkeypatch.setattr(fa.pd, "read_csv", drop_one_acc_row)
    with pytest.raises(FusionAssemblyError, match="cohort hash mismatch|segments differ"):
        fa.assemble_unit_validation_scores(unit, PRETEST_ROOT)
