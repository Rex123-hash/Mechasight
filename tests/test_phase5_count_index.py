"""5.6B: synthetic attacks, label-free ordinals and permitted development reproduction."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from src.experiment import phase5_input_manifest as mi
from src.experiment.phase5_count_index import (
    INDEX_COLUMNS, CountIndex, _expand_count_rows, build_count_derived_index, build_protected_count_index,
    inference_index, validate_count_index,
)
from src.experiment.phase5_execution import FrozenPackage
from src.experiment.phase5_inputs import Access, Refused, Sandbox
from src.experiment.provenance import canonical_json, content_sha256


@pytest.fixture(scope="module")
def package():
    return FrozenPackage.load()


@pytest.fixture
def fixture(package, tmp_path):
    box = Sandbox(tmp_path / "SYNTHETIC-count-inputs")
    manifest = mi.create_synthetic_count_fixture(box, package)
    return box, manifest


def forged(manifest, transform):
    doc = manifest.snapshot()
    transform(doc)
    raw = canonical_json(doc)
    return replace(manifest, document=raw, identity=sha256(raw.encode()).hexdigest())


@pytest.fixture
def no_artifact_io(monkeypatch):
    counts = {"parquet_reads": 0, "hdf5_opens": 0, "artifact_hashes": 0}
    def parquet(*args, **kwargs):
        counts["parquet_reads"] += 1
        raise AssertionError("unauthorized Parquet read")
    def hdf5(*args, **kwargs):
        counts["hdf5_opens"] += 1
        raise AssertionError("unauthorized HDF5 open")
    original = mi.file_sha256
    def hashed(path):
        if Path(path).name != mi.FIXTURE_NAME:
            counts["artifact_hashes"] += 1
            raise AssertionError("unauthorized artifact hash")
        return original(path)
    monkeypatch.setattr(mi.pd, "read_parquet", parquet)
    monkeypatch.setattr(mi.h5py, "File", hdf5)
    monkeypatch.setattr(mi, "file_sha256", hashed)
    yield counts
    assert counts == {"parquet_reads": 0, "hdf5_opens": 0, "artifact_hashes": 0}


def test_trusted_synthetic_manifest_and_exact_ordinal_index(fixture):
    _, manifest = fixture
    doc = mi.validate_manifest(manifest)
    assert doc["schema"] == mi.SCHEMA and doc["scope"] == mi.SYNTHETIC
    assert len(doc["layouts"]["classical"]) == 102 and doc["registry_sha256"]
    index = build_count_derived_index(manifest)
    assert tuple(index.rows.columns) == INDEX_COLUMNS
    assert index.rows.window_index.tolist() == [0, 1, 0, 1, 2, 0]
    assert index.rows.row.tolist() == list(range(6))
    assert index.rows.segment_row_offset.tolist() == [0, 0, 2, 2, 2, 5]
    assert index.rows.window_count.tolist() == [2, 2, 3, 3, 3, 1]
    assert not any("time" in c or "sample" in c for c in index.rows.columns)
    assert inference_index(index, manifest).window_index.tolist() == [0, 1, 0, 1, 2, 0]


@pytest.mark.parametrize("mutation", ["path", "cohort", "segment", "parent", "domain", "registry", "layout", "hash", "missing_identity", "unit"])
@pytest.mark.parametrize("entry", ["builder", "features", "spectral", "lower_path"])
def test_forged_manifest_rejected_before_artifact_io(fixture, no_artifact_io, mutation, entry):
    _, manifest = fixture
    def change(doc):
        if mutation == "path": doc["artifacts"]["features"]["path"] = "../SYNTHETIC-protected-like.parquet"
        elif mutation == "cohort": doc["cohort_identity"] = "0" * 64
        elif mutation == "segment": doc["segments"][0]["segment_id"] = "SYNTHETIC-forged-protected"
        elif mutation == "parent": doc["segments"][0]["parent_recording"] = "SYNTHETIC-wrong-parent"
        elif mutation == "domain": doc["segments"][0]["domain"] = "target"
        elif mutation == "registry": doc["registry_sha256"] = "0" * 64
        elif mutation == "layout": doc["layouts"]["spectral"]["widths"]["mic"] += 1
        elif mutation == "hash": doc["artifacts"]["features"]["sha256"] = "0" * 64
        elif mutation == "missing_identity": del doc["artifacts"]["features"]["sha256"]
    if mutation != "unit": manifest = forged(manifest, change)
    kw = {"unit": "SYNTHETIC-unregistered-scientific-unit"} if mutation == "unit" else {}
    with pytest.raises(Refused):
        if entry == "builder": build_count_derived_index(manifest, **kw)
        elif entry == "features": mi.read_features(manifest, **kw)
        elif entry == "spectral": mi.read_spectral(manifest, **kw)
        else: mi._checked_path(manifest, "features", **kw)


@pytest.mark.parametrize("entry", ["features", "spectral", "lower_path"])
def test_substituted_external_path_rejected_before_hash(fixture, no_artifact_io, tmp_path, entry):
    _, manifest = fixture
    path = tmp_path / "SYNTHETIC-PROTECTED-LIKE.parquet"
    path.write_bytes(b"SYNTHETIC protected-like sentinel; MUST NOT BE OPENED")
    with pytest.raises(Refused, match="substitution"):
        if entry == "features": mi.read_features(manifest, path=path)
        elif entry == "spectral": mi.read_spectral(manifest, path=path)
        else: mi._checked_path(manifest, "features", path=path)


def test_manifest_hash_and_frozen_registry_tamper_refuse(fixture, no_artifact_io, monkeypatch):
    _, manifest = fixture
    with pytest.raises(Refused, match="manifest hash"):
        build_count_derived_index(replace(manifest, identity="0" * 64))
    real = mi.canonical_bytes
    monkeypatch.setattr(mi, "canonical_bytes", lambda p: real(p) + b" " if Path(p) == mi.ROOT / mi.REGISTRY else real(p))
    with pytest.raises(Refused, match="frozen input reference changed"):
        build_count_derived_index(manifest)


def test_caller_package_registry_tampering_cannot_register_inputs(package, no_artifact_io):
    bad = deepcopy(package)
    bad.registry["status"] = "SYNTHETIC-tampered"
    with pytest.raises(Refused):
        mi.development_manifest(bad, machine="RoboticArm", regime="S")


@pytest.mark.parametrize("case", ["no_authorization", "no_batch", "no_exposure", "wrong_executor", "wrong_batch",
                                  "unapproved_cohort", "feature", "spectral", "fake_development", "raw", "labels", "provenance"])
def test_future_protected_interfaces_always_refuse_without_reading(package, no_artifact_io, case):
    manifest = mi.protected_manifest_template(package, machine="RoboticArm", regime="S")
    assert all(v["path"] is None and v["sha256"] is None for v in manifest.snapshot()["artifacts"].values())
    request = {"authorized": True, "case": case}
    if case in {"wrong_executor", "wrong_batch", "no_exposure"}:
        request["capability"] = Access("SYNTHETIC-wrong-executor", "SYNTHETIC-sandbox", "SYNTHETIC-wrong-batch", "score", "SYNTHETIC-fake-token")
    for call in (build_protected_count_index, mi.read_protected_inputs):
        with pytest.raises(Refused, match="disabled"):
            call(manifest, batch_id="SYNTHETIC-no-durable-exposure", **request)
    with pytest.raises(Refused, match="disabled"):
        build_count_derived_index(manifest)


def test_forged_development_classification_does_not_activate_protected_template(package, no_artifact_io):
    manifest = mi.protected_manifest_template(package, machine="RoboticArm", regime="S")
    fake = forged(manifest, lambda d: d.update(scope=mi.DEVELOPMENT, activation="DEVELOPMENT_ONLY"))
    with pytest.raises(Refused, match="trusted input reference"):
        build_count_derived_index(fake)
    with pytest.raises(TypeError):
        build_count_derived_index(manifest, authorized=True)


@pytest.mark.parametrize("value", [-1, 0, None, float("nan"), .5, 2., True, "2", 2**65])
def test_invalid_window_counts_fail_closed(fixture, value):
    _, manifest = fixture
    features = mi.read_features(manifest)
    layout, _ = mi.read_spectral(manifest)
    features["window_count"] = pd.Series([value, 3, 1], dtype=object)
    with pytest.raises(Refused, match="window_count|overflows"):
        _expand_count_rows(manifest, features, layout)


@pytest.mark.parametrize("mutation", ["duplicate", "unexpected", "missing", "order", "parent", "domain", "missing_count", "label_column"])
def test_feature_identity_order_and_projection_validation(fixture, mutation):
    _, manifest = fixture
    features = mi.read_features(manifest)
    layout, _ = mi.read_spectral(manifest)
    if mutation == "duplicate": features.loc[1, "segment_id"] = features.loc[0, "segment_id"]
    elif mutation == "unexpected": features.loc[0, "segment_id"] = "SYNTHETIC-other"
    elif mutation == "missing": features = features.iloc[:-1]
    elif mutation == "order": features = features.iloc[::-1]
    elif mutation == "parent": features.loc[0, "parent_recording"] = "SYNTHETIC-other"
    elif mutation == "domain": features.loc[0, "domain"] = "target"
    elif mutation == "missing_count": features = features.drop(columns="window_count")
    else: features["anomaly_label"] = "SYNTHETIC-poison"
    with pytest.raises(Refused): _expand_count_rows(manifest, features, layout)


@pytest.mark.parametrize("mutation", ["missing_modality", "rows", "width", "dtype", "declared_total"])
def test_inconsistent_feature_spectral_totals_refuse(fixture, mutation):
    _, manifest = fixture
    features = mi.read_features(manifest)
    layout, _ = mi.read_spectral(manifest)
    if mutation == "missing_modality": del layout["modalities"]["gyro"]
    elif mutation == "rows": layout["modalities"]["acc"]["rows"] += 1
    elif mutation == "width": layout["modalities"]["mic"]["width"] += 1
    elif mutation == "dtype": layout["modalities"]["gyro"]["dtype"] = "float64"
    else: layout["declared_window_count"] += 1
    with pytest.raises(Refused): _expand_count_rows(manifest, features, layout)


@pytest.mark.parametrize("column", ["row", "window_index", "segment_row_offset", "window_count", "cohort_identity", "source_artifact_identity"])
def test_rehashed_index_tampering_fails_against_bound_artifacts(fixture, column):
    _, manifest = fixture
    original = build_count_derived_index(manifest)
    rows = original.rows.copy()
    rows.loc[0, column] = "SYNTHETIC-forged" if column.endswith("identity") else 99
    descriptor = deepcopy(original.descriptor)
    descriptor["content_sha256"] = content_sha256(rows.to_dict("list"))
    with pytest.raises(Refused, match="hash/offset/order/content"):
        validate_count_index(CountIndex(rows, descriptor), manifest)


def test_poisoned_labels_and_provenance_never_requested(fixture, monkeypatch):
    _, manifest = fixture
    real_parquet, real_getitem = pd.read_parquet, h5py.Group.__getitem__
    requested, datasets = [], []
    def projected(path, *args, **kw):
        columns = kw.get("columns")
        assert columns and not any("label" in c or "anomaly" in c for c in columns)
        requested.append(tuple(columns))
        return real_parquet(path, *args, **kw)
    def numeric(self, key):
        datasets.append(key)
        assert key != "provenance_json"
        return real_getitem(self, key)
    monkeypatch.setattr(pd, "read_parquet", projected)
    monkeypatch.setattr(h5py.Group, "__getitem__", numeric)
    result = build_count_derived_index(manifest)
    mi.read_features(manifest, classical=True)
    mi.read_spectral(manifest, values=True)
    # h5py's attribute machinery may access the root group; it is not a dataset.
    assert len(result.rows) == 6 and requested and set(datasets) - {"/"} == set(mi.MODALITIES)


def test_modified_approved_synthetic_artifact_stops_before_deserialization(fixture, monkeypatch):
    box, manifest = fixture
    box.path(mi.FIXTURE_PATHS["features"]).write_bytes(b"SYNTHETIC-corrupted-artifact")
    monkeypatch.setattr(mi.pd, "read_parquet", lambda *a, **k: pytest.fail("deserialized corrupt artifact"))
    with pytest.raises(Refused, match="content identity"):
        build_count_derived_index(manifest)


@pytest.mark.parametrize("mutation", ["missing_modality", "dtype", "width", "rows", "declared_count", "fractional_count", "preprocessing"])
def test_malformed_synthetic_numeric_cache_is_rejected(fixture, mutation):
    box, _ = fixture
    path = box.path(mi.FIXTURE_PATHS["spectral"])
    with h5py.File(path, "r+") as h:
        if mutation == "missing_modality": del h["gyro"]
        elif mutation in {"dtype", "width", "rows"}:
            shape = h["mic"].shape
            del h["mic"]
            h.create_dataset("mic", data=np.zeros((shape[0] + (mutation == "rows"), shape[1] + (mutation == "width")),
                                                  dtype=np.float64 if mutation == "dtype" else np.float32))
        elif mutation == "declared_count": h.attrs["window_count"] = 7
        elif mutation == "fractional_count": h.attrs["window_count"] = 6.5
        else: h.attrs["preprocessing_config_id"] = "SYNTHETIC-forged"
    # Rebinding is allowed only for a SYNTHETIC fixture at fixed temporary paths.
    doc = json.loads(box.path(mi.FIXTURE_NAME).read_text(encoding="utf-8"))
    doc["artifacts"]["spectral"]["sha256"] = mi.file_sha256(path)
    box.path(mi.FIXTURE_NAME).write_text(canonical_json(doc), encoding="utf-8")
    manifest = mi.synthetic_manifest(box)
    with pytest.raises(Refused): build_count_derived_index(manifest)


def test_alias_at_registered_artifact_location_refuses_before_hash(fixture, no_artifact_io, monkeypatch):
    box, manifest = fixture
    target = box.path(mi.FIXTURE_PATHS["spectral"])
    real = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda p: p == target or real(p))
    with pytest.raises(Refused, match="alias/link"):
        build_count_derived_index(manifest)


def test_frozen_split_is_projected_without_label_columns(package, monkeypatch):
    real = mi.pd.read_csv
    projections = []
    def projected(*args, **kw):
        assert tuple(kw.get("usecols", [])) == mi.SPLIT_COLUMNS
        assert not any("label" in c or "anomaly" in c for c in kw["usecols"])
        projections.append(tuple(kw["usecols"]))
        return real(*args, **kw)
    monkeypatch.setattr(mi.pd, "read_csv", projected)
    mi.development_manifest(package, machine="RoboticArm", regime="S")
    mi.protected_manifest_template(package, machine="BrushlessMotor", regime="T")
    assert len(projections) == 2


def test_direct_draft_helper_cannot_read_or_activate_protected_inputs(package, no_artifact_io):
    manifest = mi.protected_manifest_template(package, machine="BrushlessMotor", regime="T")
    with pytest.raises(Refused, match="disabled"):
        _expand_count_rows(manifest, pd.DataFrame(), {})
    with pytest.raises(Refused): mi._checked_path(manifest, "features")


@pytest.mark.parametrize("path", ["reports/SYNTHETIC-protected-like.csv", "../outside.json", "C:/SYNTHETIC/external.json"])
def test_direct_metadata_helpers_cannot_become_arbitrary_readers(path, monkeypatch):
    monkeypatch.setattr(mi, "canonical_bytes", lambda *a: pytest.fail("unregistered file opened"))
    monkeypatch.setattr(mi.subprocess, "run", lambda *a, **k: pytest.fail("unregistered Git object opened"))
    for call in (mi._frozen_bytes, mi._git_bytes):
        with pytest.raises(Refused, match="unregistered frozen metadata"):
            call(path)


def test_all_development_cells_and_24_frozen_models_reproduce(package):
    from scripts.verify_phase5_count_index import development_evidence
    evidence = development_evidence(package)
    assert evidence["windows"] == 40380 and len(evidence["cells"]) == 4
    assert len(evidence["real_frozen_models"]) == 24 and evidence["stub_model_calls"] == 0
    assert evidence["max_abs_score_difference"] == 0. and not evidence["physical_timestamps_reconstructed"]
