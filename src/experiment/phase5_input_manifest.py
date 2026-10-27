"""5.6B input references: permitted development, explicit synthetic, disabled protected.

No protected reader or authorization issuer exists here. A manifest is a reference,
not permission. Every I/O entry reconstructs it from independent trusted metadata.
"""
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
import pandas as pd

from src.experiment.phase5_inputs import IDENTITY_COLUMNS, Refused, Sandbox, require
from src.experiment.provenance import canonical_bytes, canonical_json, content_sha256, file_sha256, ordered_id_hash
from src.preprocessing.config import config_from_dict

ROOT = Path(__file__).resolve().parents[2]
SCIENCE_COMMIT = "d7ea8ac3022acd755a847b77118375fb719ef6e3"
REGISTRY = "reports/phase5_freeze_candidate/scientific_registry.json"
PROPOSED = "reports/phase5_freeze_candidate/proposed_evaluation_manifest.json"
INVENTORY = "reports/phase5_freeze_candidate/integrity_inventory.json"
SPLIT = "reports/phase2_2/protocol_b_few_shot_adaptation_v3.csv"
SCHEMA = "phase5-trusted-input-manifest-v1"
DEVELOPMENT, SYNTHETIC, PROTECTED = "DEVELOPMENT_SOURCE_VALIDATION", "SYNTHETIC_ONLY", "PROTECTED_DISABLED"
MODALITIES = ("mic", "acc", "gyro")
COUNT_COLUMNS = IDENTITY_COLUMNS + ("window_count",)
SPLIT_COLUMNS = ("machine", "regime", "split", "data_pool", "segment_id", "parent_recording", "file_domain")
FIXTURE_NAME = "count_fixture.json"
FIXTURE_PATHS = {"features": "SYNTHETIC/count_features.parquet", "spectral": "SYNTHETIC/count_numeric.h5"}
FROZEN_METADATA_PATHS = frozenset({REGISTRY, PROPOSED, INVENTORY, SPLIT,
    "configs/preprocessing/mechasight_few_shot_regime_s.json", "configs/preprocessing/mechasight_few_shot_regime_t.json"})


def _no_alias(path):
    require(not any(p.is_symlink() or p.is_junction() for p in (path, *path.parents)), "input alias/link refused")
    require(path.resolve(strict=True) == path.absolute(), "input path is not canonical")


@lru_cache(maxsize=None)
def _git_bytes(name):
    # Only internal fixed metadata names reach this helper. Immutable bytes cached;
    # parsed dictionaries are fresh so mutation cannot poison future validation.
    require(name in FROZEN_METADATA_PATHS, "unregistered frozen metadata path")
    return subprocess.run(["git", "show", f"{SCIENCE_COMMIT}:{name}"], cwd=ROOT,
                          capture_output=True, check=True).stdout.replace(b"\r\n", b"\n")


def _frozen_bytes(name):
    require(name in FROZEN_METADATA_PATHS, "unregistered frozen metadata path")
    try:
        path = ROOT / name
        _no_alias(path)
        raw = _git_bytes(name)
        require(canonical_bytes(path) == raw, f"frozen input reference changed: {name}")
        return raw
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Refused("frozen input reference unavailable") from exc


def _registry():
    registry, proposed = json.loads(_frozen_bytes(REGISTRY)), json.loads(_frozen_bytes(PROPOSED))
    require(content_sha256(registry) == proposed["identity"]["registry_sha256"], "scientific registry identity mismatch")
    return registry, proposed


@dataclass(frozen=True)
class InputManifest:
    document: str
    identity: str
    sandbox_root: str | None = None
    fixture_identity: str | None = None

    def snapshot(self):
        """Untrusted copy for inspection/serialization; never authority by itself."""
        return json.loads(self.document)


def _seal(document, *, sandbox_root=None, fixture_identity=None):
    raw = canonical_json(document)
    return InputManifest(raw, sha256(raw.encode("utf-8")).hexdigest(), sandbox_root, fixture_identity)


def _reference_document(machine, regime, scope):
    registry, proposed = _registry()
    units = [u for u in registry["units"] if (u["machine"], u["regime"]) == (machine, regime)]
    require(units and regime in {"S", "T"}, "unregistered machine/regime")
    planned = {r["experiment_id"] for r in proposed["planned_units"] if r["operation"] == "new_L1_inference"}
    detectors = [d for eid, d in registry["detectors"].items()
                 if eid in planned and (d["machine"], d["regime"]) == (machine, regime)]
    maha = next(d for d in detectors if d["family"] == "mahalanobis_shrinkage")
    ae = next(d for d in detectors if d["family"] == "dense_autoencoder")
    features = maha["data_provenance"]["feature_columns"]
    require(len(features) == 98, "frozen classical representation is not 98 features")
    config_path = f"configs/preprocessing/mechasight_few_shot_regime_{regime.lower()}.json"
    config_bytes = _frozen_bytes(config_path)
    config = config_from_dict(json.loads(config_bytes))
    split_bytes = _frozen_bytes(SPLIT)
    split = pd.read_csv(BytesIO(split_bytes), usecols=list(SPLIT_COLUMNS))
    role = "test" if scope == PROTECTED else "validation"
    rows = split[split.machine.eq(machine) & split.regime.eq(regime) & split.split.eq(role)]
    rows = rows.sort_values(["machine", "parent_recording", "segment_id"])
    segments = rows.rename(columns={"file_domain": "domain"}).loc[:, list(IDENTITY_COLUMNS)].to_dict("records")
    cohort = ordered_id_hash([r["segment_id"] for r in segments])
    expected = (proposed["identity"]["cohort_hashes"][machine] if scope == PROTECTED else
                maha["data_provenance"]["source_validation_cohort_hash"])
    require(cohort == expected and len(segments) == len({r["segment_id"] for r in segments}), "frozen cohort mismatch")
    widths = {m: ae["data_provenance"]["spectral_blocks"][m][1] - ae["data_provenance"]["spectral_blocks"][m][0]
              for m in MODALITIES}
    document = {"schema": SCHEMA, "scope": scope, "activation": "DISABLED" if scope == PROTECTED else "DEVELOPMENT_ONLY",
                "scientific_reference_commit": SCIENCE_COMMIT, "registry_sha256": content_sha256(registry),
                "design_sha256": registry["design_sha256"], "machine": machine, "regime": regime,
                "cohort_role": "common_evaluation" if scope == PROTECTED else "source_validation",
                "cohort_identity": cohort, "segments": segments, "ordering": ["machine", "parent_recording", "segment_id", "window_index"],
                "scientific_units": sorted(u["unit"] for u in units), "inference_experiment_ids": sorted(d["experiment_id"] for d in detectors),
                "split_manifest_sha256": sha256(split_bytes).hexdigest(),
                "preprocessing": {"path": config_path, "sha256": sha256(config_bytes).hexdigest(), "config_id": config.config_id},
                "layouts": {"counts": list(COUNT_COLUMNS), "classical": list(COUNT_COLUMNS) + features,
                            "spectral": {"representation": "phase4-log-power-spectrum-v1", "modalities": list(MODALITIES),
                                         "dtype": "float32", "widths": widths}},
                "artifacts": {}, "unresolved": []}
    if scope == PROTECTED:
        # No protected metadata, paths, content hashes, stats or counts are probed.
        document["artifacts"] = {r: {"path": None, "sha256": None, "stat": None} for r in FIXTURE_PATHS}
        document["unresolved"] = ["protected artifact bindings require separately reviewed metadata references and activation",
                                  "window counts and dataset shapes require future authorized post-exposure reads",
                                  "H1-H7 remain human decisions; 232 looks and cap 660 are not authorized"]
        return document
    inventory = json.loads(_frozen_bytes(INVENTORY))["all_ignored_data_and_reports_stat_only"]
    paths = {"features": f"data/interim/phase31_primary/{machine}/{regime}/source_validation.parquet",
             "spectral": f"data/interim/phase4/spectral/{machine}/{regime}/source_validation.spectral.h5"}
    hashes = {"features": maha["data_provenance"]["source_validation_feature_artifact_hash"],
              "spectral": ae["data_provenance"]["source_validation_spectral_cache_hash"]}
    document["artifacts"] = {r: {"path": paths[r], "sha256": hashes[r], "stat": inventory[paths[r]]} for r in paths}
    return document


def development_manifest(package, *, machine, regime):
    package.verify_memory()
    require(content_sha256(package.registry) == content_sha256(_registry()[0]), "caller registry differs from frozen reference")
    return _seal(_reference_document(machine, regime, DEVELOPMENT))


def protected_manifest_template(package, *, machine, regime):
    """Metadata-only disabled template. Unknown artifact fields stay unresolved."""
    package.verify_memory()
    require(content_sha256(package.registry) == content_sha256(_registry()[0]), "caller registry differs from frozen reference")
    return _seal(_reference_document(machine, regime, PROTECTED))


def _synthetic_document(box):
    path = box.path(FIXTURE_NAME)
    _no_alias(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    require(document.get("schema") == SCHEMA and document.get("scope") == SYNTHETIC,
            "synthetic fixture scope/schema required")
    expected = _reference_document(document["machine"], document["regime"], DEVELOPMENT)
    expected.update(scope=SYNTHETIC, activation="SYNTHETIC_TEMP_ONLY", cohort_role="SYNTHETIC",
                    segments=[{"segment_id": f"SYNTHETIC-segment-{i}", "parent_recording": f"SYNTHETIC-parent-{i // 2}",
                               "domain": "source"} for i in range(3)])
    expected["cohort_identity"] = ordered_id_hash([r["segment_id"] for r in expected["segments"]])
    expected["artifacts"] = document["artifacts"]
    require(document == expected, "synthetic fixture has unregistered manifest fields/identities")
    require(set(document["artifacts"]) == set(FIXTURE_PATHS), "synthetic artifacts missing")
    for role, record in document["artifacts"].items():
        require(set(record) == {"path", "sha256", "stat"} and record["path"] == FIXTURE_PATHS[role], "unregistered synthetic input path")
        require(isinstance(record["sha256"], str) and len(record["sha256"]) == 64
                and all(c in "0123456789abcdef" for c in record["sha256"]), "artifact identity missing/malformed")
        require(record["stat"] is None, "synthetic fixture stat field must be null")
        _no_alias(box.path(record["path"]))
    return document


def synthetic_manifest(box):
    require(type(box) is Sandbox, "dedicated synthetic sandbox required")
    document = _synthetic_document(box)
    return _seal(document, sandbox_root=str(box.root), fixture_identity=file_sha256(box.path(FIXTURE_NAME)))


def validate_manifest(manifest, *, unit=None, for_read=False):
    require(type(manifest) is InputManifest, "trusted input-manifest reference required")
    require(type(manifest.document) is str and type(manifest.identity) is str, "input-manifest fields malformed")
    require(sha256(manifest.document.encode("utf-8")).hexdigest() == manifest.identity, "manifest hash mismatch")
    try:
        document = json.loads(manifest.document)
        scope = document["scope"]
        require(scope in {DEVELOPMENT, SYNTHETIC, PROTECTED}, "unknown input scope")
        if for_read:
            require(scope != PROTECTED, "protected input access is disabled; human authorization and durable exposure required")
        if scope == SYNTHETIC:
            require(manifest.sandbox_root and manifest.fixture_identity, "synthetic fixture reference missing")
            box = Sandbox(manifest.sandbox_root)
            expected = _synthetic_document(box)
            require(file_sha256(box.path(FIXTURE_NAME)) == manifest.fixture_identity, "synthetic reference changed")
        else:
            require(manifest.sandbox_root is None and manifest.fixture_identity is None, "foreign input-manifest reference")
            expected = _reference_document(document["machine"], document["regime"], scope)
        require(document == expected, "manifest differs from independently trusted input reference")
        require(unit is None or unit in document["scientific_units"], "unregistered scientific unit for this cell")
        return document
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise Refused("input manifest or reference is malformed/unavailable") from exc


def _checked_path(manifest, role, *, path=None, unit=None):
    doc = validate_manifest(manifest, unit=unit, for_read=True)
    require(role in FIXTURE_PATHS, "raw, labels and provenance inputs are forbidden")
    base = Path(manifest.sandbox_root) if doc["scope"] == SYNTHETIC else ROOT
    # Bind all paths before hashing either input; no caller can supply a new path.
    paths = {r: base / a["path"] for r, a in doc["artifacts"].items()}
    if path is not None:
        requested = Path(path)
        require(".." not in requested.parts and requested.absolute() == paths[role], "input path substitution refused")
    for r, actual in paths.items():
        require(actual.is_relative_to(base) and ".." not in actual.parts, "input path escapes trusted root")
        _no_alias(actual)
        expected = doc["artifacts"][r]["stat"]
        if expected is not None:
            stat = actual.stat()
            require({"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns} == expected, "input stat identity changed")
    actual = paths[role]
    require(file_sha256(actual) == doc["artifacts"][role]["sha256"], "approved artifact content identity mismatch")
    return actual, doc


def read_features(manifest, *, classical=False, path=None, unit=None):
    actual, doc = _checked_path(manifest, "features", path=path, unit=unit)
    columns = doc["layouts"]["classical" if classical else "counts"]
    frame = pd.read_parquet(actual, columns=columns)
    require(list(frame.columns) == columns, "feature projection differs from approved layout")
    return frame


def read_spectral(manifest, *, values=False, path=None, unit=None):
    actual, doc = _checked_path(manifest, "spectral", path=path, unit=unit)
    layout, arrays = {}, {}
    with h5py.File(actual, "r") as handle:
        for key, expected in (("machine", doc["machine"]), ("regime", doc["regime"]),
                              ("preprocessing_config_id", doc["preprocessing"]["config_id"]),
                              ("spectral_representation_version", doc["layouts"]["spectral"]["representation"])):
            require(handle.attrs.get(key) == expected, f"spectral {key} identity mismatch")
        declared = handle.attrs.get("window_count")
        require(declared is None or (isinstance(declared, (int, np.integer)) and not isinstance(declared, (bool, np.bool_))
                                     and declared > 0), "spectral window_count must be a positive integer")
        for m in MODALITIES:
            require(m in handle, "missing modality data")
            dataset = handle[m]
            require(dataset.ndim == 2 and dataset.dtype == np.float32
                    and dataset.shape[1] == doc["layouts"]["spectral"]["widths"][m], "spectral representation layout mismatch")
            layout[m] = {"rows": int(dataset.shape[0]), "width": int(dataset.shape[1]), "dtype": "float32"}
            if values:
                arrays[m] = dataset[()]
                require(np.isfinite(arrays[m]).all(), "nonfinite spectral value")
    return {"modalities": layout, "declared_window_count": None if declared is None else int(declared)}, arrays


def create_synthetic_count_fixture(box, package, *, machine="RoboticArm", regime="S"):
    """Only fake identities and fixed temp paths; no protected registration option."""
    require(type(box) is Sandbox, "dedicated synthetic sandbox required")
    document = development_manifest(package, machine=machine, regime=regime).snapshot()
    document.update(scope=SYNTHETIC, activation="SYNTHETIC_TEMP_ONLY", cohort_role="SYNTHETIC",
                    segments=[{"segment_id": f"SYNTHETIC-segment-{i}", "parent_recording": f"SYNTHETIC-parent-{i // 2}",
                               "domain": "source"} for i in range(3)])
    document["cohort_identity"] = ordered_id_hash([r["segment_id"] for r in document["segments"]])
    frame = pd.DataFrame([{**r, "window_count": k,
                           **{c: 0. for c in document["layouts"]["classical"][4:]},
                           "anomaly_label": "SYNTHETIC-poison-label"}
                          for r, k in zip(document["segments"], [2, 3, 1])])
    paths = {r: box.path(p) for r, p in FIXTURE_PATHS.items()}
    paths["features"].parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(paths["features"], index=False)
    with h5py.File(paths["spectral"], "w") as handle:
        handle.attrs.update(machine=machine, regime=regime, window_count=6,
                            preprocessing_config_id=document["preprocessing"]["config_id"],
                            spectral_representation_version=document["layouts"]["spectral"]["representation"])
        for m in MODALITIES:
            handle.create_dataset(m, data=np.zeros((6, document["layouts"]["spectral"]["widths"][m]), dtype=np.float32))
        handle.create_dataset("provenance_json", data=b'SYNTHETIC poison anomaly_label: never deserialize')
    document["artifacts"] = {r: {"path": FIXTURE_PATHS[r], "sha256": file_sha256(p), "stat": None} for r, p in paths.items()}
    box.path(FIXTURE_NAME).write_text(canonical_json(document), encoding="utf-8")
    return synthetic_manifest(box)


def read_protected_inputs(*args, capability=None, batch_id=None, **request):
    raise Refused("protected inputs disabled in Phase 5.6B: future reviewed executor, human authorization, batch and durable exposure required")
