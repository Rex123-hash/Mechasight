"""Phase-5 projected input interface, executable only in an explicit temp sandbox.

Parquet projection never requests outcome columns. Spectral loading requests
numeric sensor datasets only: the historical monolithic provenance JSON must
not be loaded. A separately hash-bound, label-free window index is mandatory.
No production adapter or protected-read scope is enabled by this module.

Read capabilities (Phase 5.5C/5.5D): an ``Access`` is only a reference to a
capability issued by the executor that owns the ``ProjectedInputs``. It carries
no verifier of its own, and the reader accepts no verifier either: every read is
checked by its bound ``Executor`` (issued capability -> durable exposure state),
so a caller-constructed ``Access`` or a separately constructed reader is refused.
This closes the application-level bypass; it is not a cryptographic guarantee
against code running in the same Python process.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
import uuid

import h5py
import numpy as np
import pandas as pd

from src.experiment.provenance import canonical_json, file_sha256

SYNTHETIC = "SYNTHETIC_ONLY"
IDENTITY_COLUMNS = ("segment_id", "parent_recording", "domain")
SCORE_COLUMNS = (*IDENTITY_COLUMNS, "score", "window_count")
FEATURE_COLUMNS = (*IDENTITY_COLUMNS, "x0", "x1")
INDEX_COLUMNS = (*IDENTITY_COLUMNS, "window_index")
LABEL_COLUMNS = (*IDENTITY_COLUMNS, "fake_label")
# Input schemas (Phase 5.5F). The projected columns of each are fixed in code and the
# frozen registry, never taken from a fixture: SYNTHETIC uses the 5.5B fake inputs;
# DEV uses permitted development source-validation inputs in the frozen L1 layouts
# (98 classical features; float32 log-power spectra + label-free window index).
SYNTHETIC_SCHEMA = "phase5-fixture-synthetic"
DEV_SCHEMA = "phase5-dev-frozen-v1"
# Phase 5.6C: protected-style adapter. MIRROR runs the protected read path on the
# permitted development cohorts: artifacts are resolved only from the trusted 5.6B
# input manifest (never from the fixture), read only after durable exposure, and
# windows come from the count-derived index. PROTECTED is the same path on the real
# common-evaluation cohorts; it is recognised so it can be refused, never executed.
MIRROR_SCHEMA = "phase5-protected-style-dev-mirror-v1"
PROTECTED_SCHEMA = "phase5-protected-common-evaluation-v1"
INPUT_SCHEMAS = frozenset({SYNTHETIC_SCHEMA, DEV_SCHEMA, MIRROR_SCHEMA})


class Refused(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise Refused(message)


def atomic_json(path, payload, *, replace=False, fault=None):
    """Flush data before atomic publication. Interrupted partials remain evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(replace or not path.exists(), f"refusing duplicate file: {path.name}")
    partial = path.with_name(path.name + ".partial")
    raw = (canonical_json(payload) + "\n").encode("utf-8")
    with partial.open("xb") as out:
        middle = max(1, len(raw) // 2)
        out.write(raw[:middle])
        out.flush()
        if fault:
            fault("partial_write", path.name)
        out.write(raw[middle:])
        out.flush()
        os.fsync(out.fileno())
    # Exclusive attempt locking serializes state replacements. Result names
    # are also checked before publication; no ordinary retry overwrites them.
    require(replace or not path.exists(), "duplicate publication")
    os.replace(partial, path)
    if os.name != "nt":
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class Sandbox:
    def __init__(self, root):
        self.root = Path(root).resolve()
        temporary = Path(tempfile.gettempdir()).resolve()
        require(self.root != temporary and self.root.is_relative_to(temporary), "fixture root must be a dedicated temp directory")
        require(not self.root.is_relative_to(Path.cwd().resolve()), "repository paths cannot be fixture roots")
        self.root.mkdir(parents=True, exist_ok=True)
        self.identity = uuid.uuid4().hex
        self.events = []

    def path(self, relative):
        relative = Path(relative)
        require(not relative.is_absolute() and ".." not in relative.parts, "absolute/escaping fixture path refused")
        result = (self.root / relative).resolve()
        require(result.is_relative_to(self.root), "fixture symlink/junction escapes sandbox")
        return result

    def json(self, relative):
        return json.loads(self.path(relative).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Access:
    """Reference to an executor-issued read capability; never authority by itself."""

    executor_identity: str
    sandbox_identity: str
    batch_id: str
    phase: str
    token: str


class ProjectedInputs:
    """Projected reader bound to exactly one issuing ``Executor`` (Phase 5.5D).

    It takes no verifier. Every read resolves the policy from its owner, which must
    be an exact ``Executor`` instance whose ``inputs`` is this reader over the same
    sandbox, and calls ``Executor._verify_capability`` through the class (not the
    instance attribute). A reader built by any other code, or handed a callable,
    an unrelated executor or a subclass, is refused before any file is touched.
    """

    def __init__(self, sandbox, manifest, owner):
        from src.experiment.phase5_execution import Executor
        require(type(owner) is Executor, "ProjectedInputs must be bound to its issuing Executor")
        self.sandbox, self.manifest = sandbox, manifest
        self.__owner = owner

    def __authorize(self, access, phase):
        from src.experiment.phase5_execution import Executor
        owner = self.__owner
        require(type(owner) is Executor and owner.inputs is self and owner.sandbox is self.sandbox
                and owner.fixture is self.manifest, "reader is not bound to its issuing executor")
        Executor._verify_capability(owner, access, phase)

    def _file(self, access, name, phase="score"):
        self.__authorize(access, phase)                # policy first: nothing is opened or hashed before it
        path = self.sandbox.path(name)
        require(name in self.manifest["files"], "undeclared fixture input")
        require(file_sha256(path) == self.manifest["files"][name], "fixture input hash mismatch after exposure")
        self.sandbox.events.append((phase, name))
        return path

    def _project(self, access, name, columns, phase="score"):
        path = self._file(access, name, phase)
        # Column projection occurs in the parquet reader, before a Python frame
        # exists. A drop-after-load implementation is expressly disallowed.
        frame = pd.read_parquet(path, columns=list(columns))
        require(tuple(frame.columns) == columns, "projected schema mismatch")
        return frame

    def features(self, access, cell):
        from src.experiment.phase5_execution import Executor
        columns = Executor._feature_projection(self.__owner)
        if self.manifest.get("schema") in {MIRROR_SCHEMA, PROTECTED_SCHEMA}:
            from src.experiment.phase5_input_manifest import read_features
            manifest = self._trusted(access, cell, "features")
            frame = read_features(manifest, classical=True)
            require(tuple(frame.columns) == columns, "trusted classical layout differs from the frozen projection")
            return frame
        return self._project(access, self.manifest["cells"][cell]["features"], columns)

    def _trusted(self, access, cell, role):
        """Policy, then the independently rebuilt trusted manifest; the fixture supplies no path."""
        self.__authorize(access, "score")             # durable exposure before any manifest, stat or byte
        require(self.manifest.get("schema") == MIRROR_SCHEMA,
                "protected common-evaluation reads are disabled: human authorization H1-H7, review and durable exposure required")
        from src.experiment.phase5_input_manifest import development_manifest
        machine, regime = cell.split("|")
        manifest = development_manifest(self.__owner.package, machine=machine, regime=regime)
        require(manifest.identity == self.manifest["cells"][cell]["input_manifest_identity"],
                "mirror cell is not bound to the trusted input manifest")
        segments = sorted(r["segment_id"] for r in manifest.snapshot()["segments"])
        require(segments == sorted(r["segment_id"] for r in self.manifest["cohorts"][machine]),
                "trusted cohort differs from the executor cohort")
        self.sandbox.events.append(("score", f"trusted:{cell}:{role}"))
        return manifest

    def windows(self, access, cell):
        entry = self.manifest["cells"][cell]
        if self.manifest.get("schema") in {MIRROR_SCHEMA, PROTECTED_SCHEMA}:
            return self._mirror_windows(access, cell)
        if self.manifest.get("schema", SYNTHETIC_SCHEMA) == DEV_SCHEMA:
            return self._dev_windows(access, cell, entry)
        index = self._project(access, entry["index"], INDEX_COLUMNS)
        path = self._file(access, entry["spectral"])
        with h5py.File(path, "r") as handle:
            arrays = {m: np.asarray(handle[m][()], dtype=np.float64) for m in ("mic", "acc", "gyro")}
        require(all(len(a) == len(index) and np.isfinite(a).all() for a in arrays.values()), "spectral/index mismatch")
        require(not index.duplicated(["segment_id", "window_index"]).any(), "duplicate window identity")
        for _, group in index.groupby("segment_id", sort=False):
            require(group["window_index"].tolist() == list(range(len(group))), "window order mismatch")
        return arrays, index

    def _dev_windows(self, access, cell, entry):
        """Label-free index (validated, hash-bound) + numeric float32 spectra, exactly as cached."""
        from src.experiment.phase5_window_index import INDEX_COLUMNS as LABEL_FREE_INDEX, validate_index
        machine, regime = cell.split("|")
        index = self._project(access, entry["index"], LABEL_FREE_INDEX)
        validate_index(index, entry["index_manifest"], expected_segments=self.manifest["cohorts"][machine],
                       machine=machine, regime=regime)
        path = self._file(access, entry["spectral"])
        with h5py.File(path, "r") as handle:
            arrays = {m: np.asarray(handle[m][()]) for m in ("mic", "acc", "gyro")}
        dims = entry["index_manifest"]["representation"]["input_dims"]
        require(all(a.dtype == np.float32 and a.ndim == 2 and len(a) == len(index) and a.shape[1] == dims[m]
                    and np.isfinite(a).all() for m, a in arrays.items()), "spectral/index representation mismatch")
        return arrays, index.loc[:, list(IDENTITY_COLUMNS) + ["window_index"]].reset_index(drop=True)

    def _mirror_windows(self, access, cell):
        """Count-derived label-free index + numeric spectra, both bound to the trusted manifest."""
        from src.experiment.phase5_count_index import build_count_derived_index, inference_index
        from src.experiment.phase5_input_manifest import read_spectral
        manifest = self._trusted(access, cell, "spectral")
        bundle = build_count_derived_index(manifest)
        layout, arrays = read_spectral(manifest, values=True)      # modality datasets only; no provenance_json
        index = inference_index(bundle, manifest)
        require(all(layout["modalities"][m]["rows"] == len(index) == len(arrays[m]) for m in ("mic", "acc", "gyro")),
                "spectral/count-index row mismatch")
        self.sandbox.events.append(("score", f"count-index:{cell}:{bundle.descriptor['content_sha256']}"))
        return arrays, index.reset_index(drop=True)

    def historical(self, access, eid):
        """A historical score table, classified by its trusted frozen-registry identity.

        Phase-4 single-modality tables (L0) may be read under a scoring capability;
        the 28 reused historical L1 comparators are analysis-only. The class comes
        from the owner's registry-derived policy, never from the caller, and is
        enforced before the file path is resolved.
        """
        from src.experiment.phase5_execution import Executor
        phase, name = Executor._historical_artifact(self.__owner, eid)
        return self._project(access, name, SCORE_COLUMNS, phase)

    def labels(self, access, machine):
        return self._project(access, self.manifest["labels"][machine], LABEL_COLUMNS, "analysis")
