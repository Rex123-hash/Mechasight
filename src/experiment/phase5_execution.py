"""Phase 5.5B/5.5C: hash-anchored orchestration; real protected execution DISABLED.

Only explicit synthetic permits over dedicated temporary sandboxes can execute.
There is no production permit constructor, real ledger writer or scope opener.

Phase 5.5C hardening: read capabilities are issued and verified only by the
owning executor (never by a caller-supplied callback); interrupted control-file,
recovery and analysis writes resolve only through explicit audited actions or a
human incident; ``verify_executable_candidate`` binds the executor sources and
synthetic evidence to an externally supplied, independently reviewed commit.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import uuid

import numpy as np
import pandas as pd
import psutil

from src.experiment.fusion_freeze import validate_registry
from src.experiment.phase5_inputs import (
    Access, IDENTITY_COLUMNS, INPUT_SCHEMAS, ProjectedInputs, Refused, SCORE_COLUMNS, SYNTHETIC,
    SYNTHETIC_SCHEMA, Sandbox, atomic_json, require,
)
from src.experiment.provenance import canonical_bytes, canonical_json, canonical_sha256, content_sha256, file_sha256

TRUSTED_COMMIT = "d7ea8ac3022acd755a847b77118375fb719ef6e3"
TRUSTED_MANIFEST = "7f3ae36e42b1ffee66e83915fa7c5138817b842f4cc7be3ecee911c0094c4bab"
PACKAGE = Path("reports/phase5_freeze_candidate")


@lru_cache(maxsize=None)
def committed(path):
    # Immutable Git objects only. Current model/input files are NEVER cached.
    return subprocess.run(["git", "show", f"{TRUSTED_COMMIT}:{Path(path).as_posix()}"],
                          capture_output=True, check=True).stdout


@dataclass(frozen=True)
class FrozenPackage:
    registry: dict
    contract: dict
    proposed: dict
    identity: str
    root: Path

    @classmethod
    def load(cls, root=PACKAGE):
        """Trust old committed identities, never a newly self-hashed manifest."""
        root = Path(root)
        trusted = committed(PACKAGE / "candidate_manifest.json")
        from hashlib import sha256
        require(sha256(trusted).hexdigest() == TRUSTED_MANIFEST, "trusted Git anchor mismatch")
        require(canonical_bytes(root / "candidate_manifest.json") == trusted, "candidate manifest differs from trusted commit")
        manifest = json.loads(trusted)
        documents = {}
        for name, digest in manifest["files"].items():
            require(Path(name).name == name, "unexpected package path")
            require(canonical_sha256(root / name) == digest, "frozen package hash mismatch")
            documents[name] = json.loads((root / name).read_text(encoding="utf-8"))
        # Historical test/doc identities are verified at their original commit.
        # Additive 5.5B tripwire/test changes are bound in the NEW candidate.
        for name, digest in manifest["builder_sources"].items():
            require(sha256(committed(name)).hexdigest() == digest, "historical support-source identity mismatch")
            if name.startswith(("src/", "scripts/")):
                require(canonical_sha256(name) == digest, "freeze builder changed")
        inventory = documents["integrity_inventory.json"]
        for name, digest in inventory["development_and_model_canonical_sha256"].items():
            require(name.startswith(("reports/phase5_development/", "data/interim/phase5/models/",
                                    "data/interim/phase4/models/", "configs/", "src/", "docs/")), "unsafe frozen artifact path")
            require(".." not in Path(name).parts, "unsafe frozen artifact traversal")
            require(canonical_sha256(name) == digest, "frozen artifact/model hash mismatch")
        from hashlib import sha1
        for entry in inventory["tracked_git_inventory"]:
            name = entry["path"]
            if name.startswith(("src/", "scripts/", "configs/")):
                raw = canonical_bytes(name)
                oid = sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
                require(oid == entry["git_blob"], "frozen scientific source/config identity mismatch")
        registry = documents["scientific_registry.json"]
        design = json.loads(Path("configs/phase5/fusion_design_v2.json").read_text(encoding="utf-8"))
        require(canonical_sha256("configs/phase5/fusion_design_v2.json") == registry["design_sha256"], "design identity mismatch")
        validate_registry(registry, design)
        for eid, detector in registry["detectors"].items():
            require(eid == detector["experiment_id"], "model key/identity mismatch")
            require(canonical_sha256(Path(detector["bundle_path"]) / "bundle_manifest.json") == detector["bundle_manifest_sha256"], "bundle identity mismatch")
            if detector["model_path"]:
                model = detector["model_manifest"]
                require(model["experiment_id"] == eid, "model reference mismatch")
                require(canonical_sha256(Path(detector["model_path"]) / "model_manifest.json") == model["model_manifest_sha256"], "model manifest changed")
                expected = {"model.pth" if detector["family"] == "dense_autoencoder" else "model.joblib", "standardizer.joblib"}
                require(set(model["files"]) == expected, "missing model/standardizer")
        proposed = documents["proposed_evaluation_manifest.json"]
        require(proposed["identity"]["registry_sha256"] == content_sha256(registry), "registry identity mismatch")
        return cls(registry, documents["scientific_contract.json"], proposed, TRUSTED_MANIFEST, root)

    def verify_memory(self):
        # Frozen dataclass does not freeze nested dictionaries. Rebind before
        # EVERY attempt so mutation after initial loading cannot grant access.
        expected = json.loads(committed(PACKAGE / "scientific_registry.json"))
        require(self.registry == expected, "in-memory registry differs from trusted identity")
        require(self.proposed == json.loads(committed(PACKAGE / "proposed_evaluation_manifest.json")), "in-memory evaluation plan changed")
        require(self.contract == json.loads(committed(PACKAGE / "scientific_contract.json")), "in-memory statistical contract changed")
        require(self.identity == TRUSTED_MANIFEST, "untrusted package object")
        # Recheck on-disk artifacts, not just the dictionaries loaded earlier.
        # Models may have been removed or replaced between load and execution.
        self.load(self.root)


@dataclass(frozen=True)
class FixturePermit:
    sandbox_identity: str
    package_identity: str
    fixture_hash: str
    action: str
    batch_id: str
    code_identity: str
    recovery_from: str | None = None


FIXTURE_ACTIONS = {"score", "analysis", "recovery", "incident"}
ADMISSION = "recovery.admission"
# Stable coordinator: created on first use and never moved, replaced or deleted (Phase 5.5E).
COORDINATOR = "admission.coordinator"


def _lock_handle(handle):
    """Non-blocking exclusive OS lock on byte 0 of an open handle; OSError if held elsewhere."""
    handle.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_handle(handle):
    handle.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
RECOVERABLE = {"FAILED_TECHNICAL", "EXPOSED", "PRE_EXPOSURE", "REFUSED_PRE_EXPOSURE", "RECOVERING"}


def authorize_fixture(sandbox, package, action, *, recovery_from=None):
    """Explicit fixture authorization only; cannot authorize any real file path.

    Every permit gets a unique batch identity, so a second recovery from the same
    failed batch is a distinct, separately logged authorization (Phase 5.5C).
    """
    require(isinstance(sandbox, Sandbox), "a temp sandbox is required")
    require(action in FIXTURE_ACTIONS, "unknown fixture action")
    package.verify_memory()
    fixture = sandbox.json("fixture.json")
    require(fixture["scope"] == SYNTHETIC, "not a synthetic fixture")
    digest = file_sha256(sandbox.path("fixture.json"))
    batch = "SYNTHETIC-" + content_sha256({"fixture": digest, "package": package.identity, "action": action,
                                         "recovery": recovery_from, "permit": uuid.uuid4().hex})[:24]
    return FixturePermit(sandbox.identity, package.identity, digest, action, batch, source_identity(), recovery_from)


def historical_policy(registry):
    """Read phase of every historical score table, from frozen registry membership only.

    Phase-4 single-modality detectors (each unit's L0) feed L2 derivation and may be
    read while scoring; the reused historical L1 comparators are analysis-only.
    """
    score = {eid for u in registry["units"] for eid in u["L0"].values()}
    analysis = {u["L1"]["experiment_id"] for u in registry["units"] if u["L1"]["source"] != "new_fit_phase5"}
    require(score and analysis and not score & analysis, "historical comparator classes are ambiguous")
    return {**{e: "score" for e in score}, **{e: "analysis" for e in analysis}}


def execute_real_protected(*args, **kwargs):
    raise Refused("Real protected execution is disabled in Phase 5.5B; future human authorization and independent review required")


def synthetic_predictor(detector, inputs, index=None):
    """Test double for the frozen L1 interface; never loads/fits a real model."""
    if isinstance(inputs, dict):
        x = np.concatenate([inputs[m] for m in ("mic", "acc", "gyro")], axis=1)
        frame = index.loc[:, list(IDENTITY_COLUMNS)].copy()
        frame["score"] = np.mean(x * x, axis=1) + detector["seed"] * .01
        grouped = frame.groupby("segment_id", sort=True)
        rows = []
        for sid, part in grouped:
            require(part["parent_recording"].nunique() == part["domain"].nunique() == 1, "window provenance mismatch")
            rows.append({"segment_id": sid, "parent_recording": part["parent_recording"].iloc[0],
                         "domain": part["domain"].iloc[0], "score": float(part["score"].median()), "window_count": len(part)})
        return pd.DataFrame(rows, columns=SCORE_COLUMNS)
    result = inputs.loc[:, list(IDENTITY_COLUMNS)].copy()
    result["score"] = np.mean(inputs[["x0", "x1"]].to_numpy() ** 2, axis=1)
    result["window_count"] = 2
    return result.loc[:, list(SCORE_COLUMNS)]


class Executor:
    def __init__(self, sandbox, package, *, fault=None, predictor=synthetic_predictor):
        require(isinstance(sandbox, Sandbox), "fixture sandbox required")
        self.sandbox, self.package, self.fault, self.predictor = sandbox, package, fault, predictor
        self.fixture = sandbox.json("fixture.json")
        require(self.fixture["scope"] == SYNTHETIC, "real fixture refused")
        self._identity = uuid.uuid4().hex
        self._capabilities = {}
        self._historical_policy = historical_policy(package.registry)
        self.inputs = ProjectedInputs(sandbox, self.fixture, self)
        self.units = {u["unit"]: u for u in package.registry["units"]}
        self.operations = {r["experiment_id"]: r for r in package.proposed["planned_units"]}
        require(len(self.operations) == 232, "exact 232-unit plan required")
        require(self.fixture.get("schema", SYNTHETIC_SCHEMA) in INPUT_SCHEMAS, "unknown fixture input schema")
        from src.experiment.phase5_ledger import ledger_backend
        self.ledger = ledger_backend(self)

    def _inference_kind(self):
        """Classify the callable itself, never a caller's description of it."""
        from src.experiment.phase5_inference import FrozenL1Adapter
        schema = self.fixture.get("schema", SYNTHETIC_SCHEMA)
        if self.predictor is synthetic_predictor:
            require(schema == SYNTHETIC_SCHEMA, "stub predictor requires the synthetic input schema")
            return "synthetic_L1_stub"
        require(type(self.predictor) is FrozenL1Adapter and self.predictor.package is self.package,
                "only the actual frozen L1 adapter or explicit synthetic stub may score")
        require(schema != SYNTHETIC_SCHEMA, "real-model adapter requires the development input schema")
        return "frozen_L1_inference"

    def _computation(self, eid):
        """Expected computation/lineage, also checked on every completed output."""
        from src.experiment.phase5_inference import ADAPTER_VERSION
        op = self.operations[eid]
        kind = self._inference_kind()
        if op["operation"] == "new_L1_inference":
            d = self.package.registry["detectors"][eid]
            result = {"kind": kind, "implementation": (ADAPTER_VERSION if kind == "frozen_L1_inference"
                                                       else "SYNTHETIC_STUB_V1"),
                      "experiment_id": eid, "family": d["family"], "seed": d["seed"]}
            if kind == "frozen_L1_inference":
                result.update(model=d["model_manifest"], fit_code=d["code_git_sha"],
                              adapter_source_sha256=canonical_sha256("src/experiment/phase5_inference.py"))
            else:
                result["source_sha256"] = canonical_sha256("src/experiment/phase5_execution.py")
            return result
        calibration = self.package.registry["calibrations"][eid]
        return {"kind": "derived_L2_scoring", "implementation": "phase5-l2-frozen-robust-z-mean-v1",
                "calibration_id": eid, "calibration_sha256": content_sha256(calibration),
                "subset": calibration["subset"], "source_sha256": canonical_sha256("src/experiment/phase5_execution.py"),
                "component_origin": ("PERMITTED_PHASE4_SOURCE_VALIDATION" if kind == "frozen_L1_inference"
                                     else "SYNTHETIC_FIXTURE"),
                "components": {m: {"experiment_id": component,
                    "fixture_table_sha256": self.fixture["files"][self.fixture["historical"][component]],
                    "frozen_bundle_manifest_sha256": self.package.registry["detectors"][component]["bundle_manifest_sha256"]}
                    for m, component in calibration["component_experiment_ids"].items()}}

    def _feature_projection(self):
        """Trusted classical projection: synthetic columns, or the frozen 98 L1 features from the registry."""
        if self.fixture.get("schema", SYNTHETIC_SCHEMA) == SYNTHETIC_SCHEMA:
            from src.experiment.phase5_inputs import FEATURE_COLUMNS
            return FEATURE_COLUMNS
        lists = {tuple(d["data_provenance"]["feature_columns"]) for d in self.package.registry["detectors"].values()
                 if d["family"] == "mahalanobis_shrinkage" and d["experiment_id"] in self.operations}
        require(len(lists) == 1 and len(next(iter(lists))) == 98, "frozen L1 feature list is not unique")
        return IDENTITY_COLUMNS + ("window_count",) + next(iter(lists))

    def _look_identity(self, eid):
        """Machine, regime, model/calibration identity and cohort hash of one planned look."""
        operation = self.operations[eid]
        unit = self.units[operation["unit"]]
        if operation["operation"] == "new_L1_inference":
            model = self.package.registry["detectors"][eid]["model_manifest"]["model_id"]
        else:
            model = "p5l2:" + self.package.registry["calibrations"][eid]["calibration_version"] + ":" + eid
        from src.experiment.provenance import ordered_id_hash
        cohort = ordered_id_hash(tuple(sorted(r["segment_id"] for r in self.fixture["cohorts"][unit["machine"]])))
        return {"machine": unit["machine"], "regime": unit["regime"], "model_config_identity": model,
                "cohort_hash": cohort, "operation": operation["operation"]}

    def _reservation_hash(self, batch_id):
        return file_sha256(self.sandbox.path(f"reservations/{batch_id}.json"))

    def _hook(self, stage, unit=""):
        if self.fault:
            self.fault(stage, unit)

    def _permit(self, permit, action):
        require(isinstance(permit, FixturePermit), "explicit synthetic authorization required")
        require(permit.sandbox_identity == self.sandbox.identity and permit.action == action, "wrong authorization/sandbox")
        require(permit.code_identity == source_identity(), "executor source changed after fixture authorization")
        self.package.verify_memory()
        require(historical_policy(self.package.registry) == self._historical_policy,
                "historical comparator classification differs from the trusted registry")
        require(permit.package_identity == self.package.identity and
                permit.fixture_hash == file_sha256(self.sandbox.path("fixture.json")), "authorization identity mismatch")

    def _write(self, name, value, *, replace=False, table=False):
        # Result tables keep the historical "partial_write" hook; control files
        # (state, ledger, reservation, exposure, recovery, claim) use "control_partial_write".
        fault = None
        if self.fault:
            fault = self.fault if table else (lambda stage, unit: self.fault("control_" + stage, unit))
        atomic_json(self.sandbox.path(name), value, replace=replace, fault=fault)

    def _event(self, state, **detail):
        path = self.sandbox.path("attempts.jsonl")
        record = {"scope": SYNTHETIC, "state": state, "utc": datetime.now(timezone.utc).isoformat(), **detail}
        with path.open("ab") as out:
            out.write((canonical_json(record) + "\n").encode())
            out.flush()
            os.fsync(out.fileno())

    def _lock_record(self, state="ACTIVE"):
        return {"scope": SYNTHETIC, "pid": os.getpid(), "process_created": psutil.Process().create_time(),
                "executor": self._identity, "state": state}

    def _exclusive_create(self, name, record, refusal):
        """Create ``name`` only if absent (O_EXCL; atomic on NTFS); never replaces."""
        try:
            with self.sandbox.path(name).open("x", encoding="utf-8") as out:
                out.write(canonical_json(record) + "\n")
                out.flush()
                os.fsync(out.fileno())
        except FileExistsError as exc:
            raise Refused(refusal) from exc

    @contextmanager
    def _lock(self):
        """Exclusive attempt lock; retained (and marked) after any failure."""
        self._exclusive_create("attempt.lock", self._lock_record(),
                               "concurrent, duplicate, or interrupted attempt; explicit recovery required")
        with self._hold_lock():
            yield

    @contextmanager
    def _hold_lock(self):
        path = self.sandbox.path("attempt.lock")
        try:
            yield
        except BaseException:
            # Retain the lock; record that this owner's handler ran. If even that
            # write fails, the unmarked lock still refuses until owner death is shown.
            try:
                atomic_json(path, self._lock_record("FAILED_HANDLER_RAN"), replace=True)
            except BaseException:  # noqa: BLE001 - never mask the original failure
                pass
            raise
        else:
            path.unlink()

    @contextmanager
    def _coordinated(self):
        """Stable OS file lock serialising every admission transition (Phase 5.5E).

        Admission acquisition, the recovery owner-check-and-takeover, stale-admission
        validation and quarantine, and admission release all run while this lock is
        held. The coordinator file is created once and never moved or deleted, so its
        identity cannot suffer pathname reuse. The lock is an OS byte-range lock
        (``msvcrt.locking`` on Windows, ``flock`` elsewhere) held on an open handle:
        the OS drops it when the holder exits or crashes, so it never goes stale. It
        is non-blocking: a contender is refused at once and nothing sleeps or polls.
        """
        handle = open(self.sandbox.path(COORDINATOR), "a+b")
        try:
            try:
                _lock_handle(handle)
            except OSError as exc:
                raise Refused("admission coordinator is held by another process; refused, retry later") from exc
            try:
                yield
            finally:
                _unlock_handle(handle)
        finally:
            handle.close()

    def _create_admission(self, permit):
        """Create this recovery's admission (caller holds the coordinator); returns its unique id."""
        admission_id = uuid.uuid4().hex
        self._exclusive_create(ADMISSION, {**self._lock_record(), "batch_id": permit.batch_id,
                                           "admission_id": admission_id},
                               "a recovery admission is present (a recovery is in progress or was interrupted): "
                               "diagnose(); if its owner is dead, resolve_interrupted_writes (incident permit)")
        return admission_id

    def _release_admission(self, admission_id):
        """Remove the admission only if it is still exactly ours (caller holds the coordinator)."""
        path = self.sandbox.path(ADMISSION)
        try:
            current = self.sandbox.json(ADMISSION)
        except (OSError, ValueError) as exc:
            raise Refused("our recovery admission is missing or unreadable; nothing was removed") from exc
        require(current.get("admission_id") == admission_id,
                "the admission at the shared path is not ours; it is left untouched")
        path.unlink()

    @staticmethod
    def _owner_dead(owner):
        try:
            return psutil.Process(owner["pid"]).create_time() != owner["process_created"]
        except psutil.NoSuchProcess:
            return True
        except psutil.AccessDenied:
            return False

    def _owner_released(self):
        """True only if the lock owner recorded its own failure or is provably dead."""
        owner = self.sandbox.json("attempt.lock")
        return owner.get("state") == "FAILED_HANDLER_RAN" or self._owner_dead(owner)

    def _historical_artifact(self, eid):
        """(read phase, fixture file) of a historical table, from its trusted registry identity only."""
        require(isinstance(eid, str) and eid in self._historical_policy, "not a frozen historical comparator identity")
        name = f"historical/{eid}.parquet"
        require(self.fixture["historical"].get(eid) == name, "historical artifact path does not match its trusted identity")
        return self._historical_policy[eid], name

    def _interrupted_writes(self):
        """Unpublished partial files and published-but-unrecorded result tables."""
        partials = sorted(p.relative_to(self.sandbox.root).as_posix() for p in self.sandbox.root.rglob("*.partial")
                          if p.relative_to(self.sandbox.root).parts[0] != "incidents")
        recorded = set()
        if self.sandbox.path("state.json").is_file():
            recorded = {t["file"] for t in self.sandbox.json("state.json")["tables"].values()}
        tables = self.sandbox.path("tables")
        orphans = sorted(f"tables/{p.name}" for p in tables.glob("*.json")) if tables.is_dir() else []
        return partials, [o for o in orphans if o not in recorded]

    def _require_no_interrupted_writes(self):
        partials, orphans = self._interrupted_writes()
        require(not partials and not orphans,
                "interrupted write (partial/orphan) present: resolve_interrupted_writes under an incident permit, "
                "or human incident resolution, is required before any further action")

    # ------------------------------------------------------------- capabilities
    def _issue(self, batch_id, phase):
        token = secrets.token_hex(32)
        access = Access(self._identity, self.sandbox.identity, batch_id, phase, token)
        self._capabilities[token] = access
        return access

    def _revoke(self):
        self._capabilities.clear()

    def _verify_capability(self, access, phase):
        """The only verifier ProjectedInputs consults: issued here, live, matching, durably exposed."""
        require(isinstance(access, Access), "read capability required")
        require(self._capabilities.get(access.token) == access, "unissued, altered, revoked or stale read capability")
        require(access.executor_identity == self._identity and access.sandbox_identity == self.sandbox.identity,
                "read capability belongs to another executor or sandbox")
        require(phase in {"score", "analysis"}, "unknown read phase")
        require(access.phase == phase or (access.phase == "analysis" and phase == "score"),
                "label access requires analysis capability")
        self._check_access(access.batch_id, access.phase)

    def _check_access(self, batch_id, phase):
        try:
            reservation = self.sandbox.json(f"reservations/{batch_id}.json")
            exposure = self.sandbox.json(f"exposures/{batch_id}.json")
            reservation_hash = file_sha256(self.sandbox.path(f"reservations/{batch_id}.json"))
        except (OSError, ValueError) as exc:
            raise Refused("authorization/exposure records missing or unreadable; no read permitted") from exc
        ledger = self.ledger.verify()            # JSONL: the unmodified project history verifier
        require(exposure["authorization_sha256"] == reservation_hash, "authorization not durable/intact")
        ids = {r["experiment_id"] for r in ledger if r.get("batch_id") == batch_id}
        require(ids == set(reservation["attempt_ids"].values()) and exposure["ledger_end"] <= len(ledger), "exposure not durably recorded")
        require(exposure["ledger_payload_sha256"] == self.ledger.prefix_digest(ledger, exposure["ledger_end"]),
                "exposed ledger prefix changed")
        require(len({r["experiment_id"] for r in ledger}) == len(ledger), "duplicate ledger identities")
        records = self._audit_records()
        require(any(r["state"] == "EXPOSURE_RECORDED" and r.get("batch_id") == batch_id
                    for r in records), "exposure audit event missing")
        if phase == "analysis":
            state = self.sandbox.json("state.json")
            require(state["status"] == "SCORED" and state["batch_id"] == batch_id
                    and set(state["tables"]) == set(self.operations), "analysis requires the completed scoring gate")
            require(self.sandbox.path("analysis.claim.json").is_file(), "analysis gate not claimed")
            require(any(r["state"] == "ANALYSIS_CLAIMED" for r in records), "analysis claim not durably logged")
            require(getattr(self, "_analysis_gate_hash", None) == file_sha256(self.sandbox.path("state.json")), "completion gate state changed during analysis")

    def _audit_records(self):
        try:
            records = [json.loads(line) for line in self.sandbox.path("attempts.jsonl").read_text(encoding="utf-8").splitlines()]
        except (OSError, ValueError) as exc:
            raise Refused("partial/corrupt attempt log requires incident resolution") from exc
        require(all(r.get("scope") == SYNTHETIC for r in records), "foreign attempt log")
        return records

    def _access(self, batch_id, phase):
        self._check_access(batch_id, phase)       # durable state first; nothing is issued otherwise
        return self._issue(batch_id, phase)

    def _reserve(self, permit, pending, attempt_ids):
        ledger = self.ledger.verify()
        require(not {r["experiment_id"] for r in ledger} & set(attempt_ids.values()), "duplicate evaluation attempt identity")
        if permit.action == "score":
            require(len(ledger) == self.ledger.base_rows, "the ledger must start at its closed history")
        # Phase 5.6C correction: the look budget is checked before the reservation,
        # batch authorization, ledger append, exposure record or any input read.
        self.ledger.precheck(ledger, attempt_ids)
        self._hook("before_reservation")
        reservation = {"scope": SYNTHETIC, "batch_id": permit.batch_id, "package": self.package.identity,
                       "fixture": permit.fixture_hash, "attempt_ids": attempt_ids, "recovery_from": permit.recovery_from,
                       "code": source_identity(), "ledger_start": len(ledger), "ledger_end": len(ledger) + len(pending)}
        self._write(f"reservations/{permit.batch_id}.json", reservation)
        self._hook("after_authorization")
        ledger = self.ledger.append(permit, pending, attempt_ids, len(ledger))
        self._hook("after_ledger")
        self._write(f"exposures/{permit.batch_id}.json", {
            "scope": SYNTHETIC, "authorization_sha256": file_sha256(self.sandbox.path(f"reservations/{permit.batch_id}.json")),
            "ledger_end": len(ledger), "ledger_payload_sha256": self.ledger.prefix_digest(ledger, len(ledger))})
        self._event("EXPOSURE_RECORDED", batch_id=permit.batch_id, units=len(pending))
        self._hook("after_reservation")
        return self._access(permit.batch_id, "score")

    def _aligned(self, frame, machine):
        require(len(frame.columns) == len(SCORE_COLUMNS) and set(frame.columns) == set(SCORE_COLUMNS), "unexpected score columns / labels / abstention")
        frame = frame.loc[:, list(SCORE_COLUMNS)].sort_values("segment_id").reset_index(drop=True)
        expected = pd.DataFrame(self.fixture["cohorts"][machine]).loc[:, list(IDENTITY_COLUMNS)].sort_values("segment_id").reset_index(drop=True)
        require(frame.loc[:, list(IDENTITY_COLUMNS)].equals(expected), "score cohort/parent/domain alignment mismatch")
        require(not frame["segment_id"].duplicated().any() and np.isfinite(frame["score"].to_numpy(dtype=float)).all(), "missing/nonfinite/duplicate score")
        require((frame["window_count"] > 0).all(), "invalid window count")
        return frame

    def _score(self, access, eid):
        operation = self.operations[eid]
        unit = self.units[operation["unit"]]
        cell = unit["machine"] + "|" + unit["regime"]
        computation = self._computation(eid)
        if operation["operation"] == "new_L1_inference":
            detector = self.package.registry["detectors"][eid]
            # Invoke the supported implementation through its class; an instance
            # attribute/string cannot relabel a custom callback as frozen inference.
            from src.experiment.phase5_inference import FrozenL1Adapter
            predict = (lambda *args: FrozenL1Adapter.__call__(self.predictor, *args)) if computation["kind"] == "frozen_L1_inference" else synthetic_predictor
            if unit["family"] == "dense_autoencoder":
                arrays, index = self.inputs.windows(access, cell)
                result = predict(detector, arrays, index)
            else:
                result = predict(detector, self.inputs.features(access, cell))
            return self._aligned(result, unit["machine"]), computation
        calibration = self.package.registry["calibrations"][eid]
        total, reference = None, None
        for mod in ("mic", "acc", "gyro"):
            if mod not in calibration["subset"]:
                continue
            component = self._aligned(self.inputs.historical(access, unit["L0"][mod]), unit["machine"])
            if reference is None:
                reference = component
                total = np.zeros(len(reference))
            c = calibration["modality_calibration"][mod]
            require(c["iqr"] > 0, "zero IQR")
            total += (component["score"].to_numpy() - c["median"]) / c["iqr"]
        result = reference.copy()
        result["score"] = total / len(calibration["subset"])
        return self._aligned(result, unit["machine"]), computation

    def score_all(self, permit=None):
        self._permit(permit, "score")
        require(not self.sandbox.path("state.json").exists(), "batch already attempted; explicit recovery required")
        with self._lock():
            state = {"scope": SYNTHETIC, "batch_id": permit.batch_id, "status": "PRE_EXPOSURE", "tables": {}, "failed": []}
            self._write("state.json", state)
            return self._execute(permit, state, sorted(self.operations), {eid: eid for eid in self.operations})

    def _provenance(self, eid):
        operation = self.operations[eid]
        if operation["operation"] == "new_L1_inference":
            detector = self.package.registry["detectors"][eid]
            return {"package": self.package.identity, "model": detector["model_manifest"],
                    "contracts": detector["cohort_contracts"], "threshold": detector["threshold"],
                    "fit_code": detector["code_git_sha"], "split_hash": detector["split_manifest_hash"]}
        calibration = self.package.registry["calibrations"][eid]
        return {"package": self.package.identity, "calibration_id": eid,
                "calibration_sha256": content_sha256(calibration), "rule": calibration["calibration_version"],
                "threshold": calibration["threshold"], "components": {
                    m: {"experiment_id": component, "sha256": self.fixture["files"][self.fixture["historical"][component]]}
                    for m, component in calibration["component_experiment_ids"].items()}}

    def _execute(self, permit, state, pending, attempt_ids):
        try:
            access = self._reserve(permit, pending, attempt_ids)
            state["status"] = "EXPOSED"
            self._write("state.json", state, replace=True)
            for eid in pending:
                self._hook("before_read", eid)
                frame, computation = self._score(access, eid)
                self._hook("after_read", eid)
                payload = {"scope": SYNTHETIC, "original_experiment_id": eid, "attempt_id": attempt_ids[eid],
                           "unit": self.operations[eid]["unit"], "operation": self.operations[eid]["operation"],
                           "rows": frame.to_dict(orient="records"), "predictor": computation["implementation"],
                           "computation": computation,
                           "provenance": self._provenance(eid)}
                name = f"tables/{eid}.json"
                self._write(name, payload, table=True)
                self._hook("after_publication", eid)
                state["tables"][eid] = {"file": name, "sha256": file_sha256(self.sandbox.path(name)), "attempt_id": attempt_ids[eid]}
                self._write("state.json", state, replace=True)
                self._event("PERSISTED", batch_id=permit.batch_id, unit=eid, attempt_id=attempt_ids[eid])
            state["status"] = "SCORED"
            state["failed"] = []
            self._write("state.json", state, replace=True)
            self.verify_complete()
            self._event("COMPLETION_GATE_GREEN", batch_id=permit.batch_id, units=232)
            return state
        except BaseException as exc:
            try:
                exposed = any(r.get("batch_id") == permit.batch_id for r in self.ledger.rows())
            except Refused:
                exposed = True        # an unreadable/partial ledger is treated as exposed (conservative)
            state["status"] = "FAILED_TECHNICAL" if exposed else "REFUSED_PRE_EXPOSURE"
            state["failed"] = sorted(set(self.operations) - set(state["tables"]))
            state["failure"] = type(exc).__name__ + ": " + str(exc)
            try:
                self._write("state.json", state, replace=True)
                self._event(state["status"], batch_id=permit.batch_id, failed=state["failed"], error=state["failure"])
            except BaseException:  # noqa: BLE001 - e.g. a leftover .partial; the original failure must surface
                pass
            raise
        finally:
            self._revoke()

    def verify_complete(self):
        self._require_no_interrupted_writes()
        require(self.sandbox.path("state.json").is_file(), "232-unit completion gate refused: no scored batch")
        state = self.sandbox.json("state.json")
        require(state["status"] in {"SCORED", "ANALYSED"} and set(state["tables"]) == set(self.operations), "232-unit completion gate refused")
        actual = {p.name for p in self.sandbox.path("tables").iterdir()}
        require(actual == {eid + ".json" for eid in self.operations}, "partial/extra/missing score output")
        ledger = self.ledger.verify()
        attempts = {r["experiment_id"]: r for r in ledger}
        require(len(attempts) == len(ledger), "duplicate ledger identity")
        checked_batches = set()
        events = self._audit_records()
        for eid, entry in state["tables"].items():
            path = self.sandbox.path(entry["file"])
            require(entry["file"] == f"tables/{eid}.json" and file_sha256(path) == entry["sha256"], "score output hash/path mismatch")
            payload = self.sandbox.json(entry["file"])
            require(payload["scope"] == SYNTHETIC and payload["original_experiment_id"] == eid
                    and payload["unit"] == self.operations[eid]["unit"] and payload["attempt_id"] == entry["attempt_id"], "persisted identity mismatch")
            require(payload["provenance"] == self._provenance(eid), "persisted scientific provenance changed")
            computation = self._computation(eid)
            require(payload.get("computation") == computation and payload.get("predictor") == computation["implementation"],
                    "persisted computation/predictor provenance mismatch")
            require(entry["attempt_id"] in attempts and attempts[entry["attempt_id"]]["original_experiment_id"] == eid, "unlogged result")
            require(sum(r["state"] == "PERSISTED" and r.get("attempt_id") == entry["attempt_id"] for r in events) == 1,
                    "missing/duplicate persisted audit event requires incident resolution")
            batch = attempts[entry["attempt_id"]]["batch_id"]
            if batch not in checked_batches:
                self._check_access(batch, "score")
                checked_batches.add(batch)
            self._aligned(pd.DataFrame(payload["rows"]), self.units[self.operations[eid]["unit"]]["machine"])
        if state["status"] == "ANALYSED":
            require(file_sha256(self.sandbox.path("analysis.json")) == state["analysis_sha256"], "analysis integrity mismatch")
        return state

    def recover(self, permit=None):
        self._permit(permit, "recovery")
        with self._coordinated():
            # Every check and the takeover run while the coordinator is held, so no
            # other recovery or cleaner can act between the owner check and the takeover.
            self._require_no_interrupted_writes()
            admission_id = self._create_admission(permit)
            taken = False
            try:
                pending, attempt_ids, state = self._admitted_takeover(permit)
                taken = True
                self._release_admission(admission_id)
            except BaseException:
                if taken:
                    try:
                        atomic_json(self.sandbox.path("attempt.lock"), self._lock_record("FAILED_HANDLER_RAN"), replace=True)
                    except BaseException:  # noqa: BLE001 - never mask the original failure
                        pass
                try:
                    self._release_admission(admission_id)   # removes it only if it is still ours
                except Refused:
                    pass
                raise
        with self._hold_lock():
            self._hook("after_lock_takeover")
            if not pending:
                # Crash after the last logged publication: finalize verified
                # outputs without inference, input reads or extra ledger rows.
                state["status"], state["failed"] = "SCORED", []
                self._write("state.json", state, replace=True)
                self.verify_complete()
                self._event("RECOVERY_FINALIZED_WITHOUT_READS", batch_id=state["batch_id"])
                return state
            state["recovered_from"] = state["batch_id"]
            state["batch_id"], state["status"] = permit.batch_id, "RECOVERING"
            self._write("state.json", state, replace=True)
            return self._execute(permit, state, pending, attempt_ids)

    def _admitted_takeover(self, permit):
        """Owner check and audited takeover; caller holds the coordinator and its own admission."""
        self._hook("recovery_admitted")
        state = self.sandbox.json("state.json")
        require(state["status"] in RECOVERABLE and permit.recovery_from == state["batch_id"], "wrong recovery authority")
        require(self.sandbox.path("attempt.lock").exists(), "missing interrupted-attempt lock")
        # A hard process exit cannot run our exception handler. Never steal a
        # live process's lock; PID creation time prevents PID-reuse errors.
        require(self._owner_released(), "cannot recover a live/concurrent attempt")
        for entry in state["tables"].values():
            require(file_sha256(self.sandbox.path(entry["file"])) == entry["sha256"], "retained output changed")
        pending = sorted(set(self.operations) - set(state["tables"]))
        if state["status"] in {"FAILED_TECHNICAL", "REFUSED_PRE_EXPOSURE"}:
            require(pending == state["failed"], "recovery pending set mismatch")
        ledger = self.ledger.verify()
        used = {r["experiment_id"] for r in ledger}
        attempt_ids = {}
        for eid in pending:
            if eid not in used:
                # No exposure occurred for this identity before the interruption.
                attempt_ids[eid] = eid
                continue
            k = 1
            while f"{eid}-r{k}" in used:
                k += 1
            attempt_ids[eid] = f"{eid}-r{k}"
        # Audited takeover: archive the stale owner, log the authorization, then
        # establish ownership while admission is still held (the lock never disappears).
        self._write(f"recoveries/{permit.batch_id}.old-lock.json", self.sandbox.json("attempt.lock"))
        self._write(f"recoveries/{permit.batch_id}.json", {"scope": SYNTHETIC, "from": state["batch_id"],
                    "pending": pending, "mapping": attempt_ids, "retained": sorted(state["tables"])})
        self._event("RECOVERY_AUTHORIZED", batch_id=permit.batch_id, from_batch=state["batch_id"],
                    pending=len(pending), new_identities=sum(attempt_ids[e] != e for e in pending))
        atomic_json(self.sandbox.path("attempt.lock"), self._lock_record(), replace=True)
        return pending, attempt_ids, state

    def resolve_interrupted_writes(self, permit=None):
        """Audited quarantine of unpublished partial files and unrecorded result tables.

        Nothing is deleted or rewritten: every file is moved under
        ``incidents/<batch>/`` with its hash recorded first. A quarantined result
        table stays pending, and because its look is already in the ledger the
        next recovery gives it a new ``-r<k>`` identity. Interrupted analysis
        files are never resolved here: they need a human incident decision.
        """
        self._permit(permit, "incident")
        # The whole resolution runs under the admission coordinator: no recovery can
        # create, take over or release an admission between validation and quarantine.
        with self._coordinated():
            return self._resolve_coordinated(permit)

    def _resolve_coordinated(self, permit):
        lock = self.sandbox.path("attempt.lock")
        if lock.exists():
            require(self._owner_released(), "cannot resolve while an attempt may be live")
        admission = self.sandbox.path(ADMISSION)
        validated = None
        if admission.exists():
            stale = self.sandbox.json(ADMISSION)
            require(self._owner_dead(stale), "a live process holds recovery admission; nothing is resolved")
            # Pin the exact admission instance that was judged dead.
            validated = (stale.get("admission_id"), file_sha256(admission))
        partials, orphans = self._interrupted_writes()
        items = [(name, "partial") for name in partials] + [(name, "orphan_table") for name in orphans]
        if validated is not None:
            # A recovery that died inside admission: the owner check/takeover may not
            # have happened. Quarantining the admission only re-opens recovery, which
            # re-runs every check under a fresh admission.
            items.append((ADMISSION, "stale_recovery_admission"))
        # An attempt that died before its first state publication left only a lock:
        # with no state, no reservation and no exposure rows, that lock is quarantined too.
        reservations = self.sandbox.path("reservations")
        never_reserved = (not self.sandbox.path("state.json").exists()
                          and not (reservations.is_dir() and any(reservations.iterdir()))
                          and len(self.ledger.rows()) == self.ledger.base_rows)
        if lock.exists() and never_reserved:
            items.append(("attempt.lock", "stale_lock_before_any_state"))
        require(items, "no interrupted write to resolve")
        require(not any(Path(name).name.startswith("analysis") for name, _ in items),
                "interrupted analysis output: human incident resolution required; no automatic action")
        require(not self.sandbox.path("analysis.claim.json").exists(),
                "analysis already claimed: human incident resolution required")
        record = {"scope": SYNTHETIC, "batch_id": permit.batch_id, "items": [
            {"path": name, "kind": kind, "sha256": file_sha256(self.sandbox.path(name)),
             "published_target_sha256": (file_sha256(self.sandbox.path(name[:-len(".partial")]))
                                         if kind == "partial" and self.sandbox.path(name[:-len(".partial")]).is_file()
                                         else None)} for name, kind in items]}
        self._write(f"incidents/{permit.batch_id}/incident.json", record)
        self._hook("incident_recorded")
        for name, _ in items:
            source = self.sandbox.path(name)
            destination = self.sandbox.path(f"incidents/{permit.batch_id}/{name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if name == ADMISSION:
                # Re-validate the exact instance immediately before moving it ...
                current = self.sandbox.json(ADMISSION)
                require((current.get("admission_id"), file_sha256(source)) == validated and self._owner_dead(current),
                        "admission changed after validation; nothing moved")
            os.replace(source, destination)
            if name == ADMISSION and file_sha256(destination) != validated[1]:
                # ... and never keep a quarantined file that is not the validated one.
                os.replace(destination, source)
                raise Refused("moved admission differs from the validated instance; restored, nothing quarantined")
        self._event("INTERRUPTED_WRITES_QUARANTINED", batch_id=permit.batch_id, items=[n for n, _ in items])
        return record

    def diagnose(self):
        """Read-only classification of the sandbox state and the only permitted next action."""
        try:
            records = self._audit_records() if self.sandbox.path("attempts.jsonl").exists() else []
        except Refused:
            return {"classification": "CORRUPT_AUDIT_LOG", "next": "human incident resolution"}
        partials, orphans = self._interrupted_writes()
        state = self.sandbox.json("state.json") if self.sandbox.path("state.json").is_file() else None
        claimed = (self.sandbox.path("analysis.claim.json").exists()
                   or any(r["state"] == "ANALYSIS_CLAIMED" for r in records))
        completed = any(r["state"] == "ANALYSIS_COMPLETED" for r in records)
        base = {"partials": partials, "orphans": orphans, "status": state and state["status"],
                "lock_present": self.sandbox.path("attempt.lock").exists(), "analysis_claimed": claimed}
        if claimed and not completed:
            return {**base, "classification": "INTERRUPTED_ANALYSIS",
                    "next": "human incident resolution; labels may have been read, so no automatic re-analysis"}
        if self.sandbox.path(ADMISSION).exists():
            return {**base, "classification": "INTERRUPTED_RECOVERY_ADMISSION",
                    "next": "if its owner is dead: resolve_interrupted_writes (incident permit), then recover; "
                            "otherwise wait for the live recovery"}
        if partials or orphans:
            return {**base, "classification": "INTERRUPTED_WRITE", "next": "resolve_interrupted_writes (incident permit), then recover"}
        if state is None:
            if base["lock_present"]:
                return {**base, "classification": "STALE_LOCK_BEFORE_STATE",
                        "next": "resolve_interrupted_writes (incident permit), then score_all"}
            return {**base, "classification": "NOT_STARTED", "next": "score_all"}
        if state["status"] in RECOVERABLE:
            return {**base, "classification": "RECOVERABLE", "next": "recover (recovery permit)"}
        if state["status"] == "SCORED":
            return {**base, "classification": "SCORED", "next": "analyse_once (analysis permit)"}
        return {**base, "classification": state["status"], "next": "none"}

    def analyse_once(self, permit=None):
        self._permit(permit, "analysis")
        self.verify_complete()
        require(not self.sandbox.path("analysis.claim.json").exists(), "analyse-once already claimed; no implicit retry")
        require(not any(r["state"] == "ANALYSIS_CLAIMED" for r in self._audit_records()),
                "analysis previously claimed per the attempt log; human incident resolution required")
        state = self.sandbox.json("state.json")
        require(state["status"] == "SCORED", "analysis requires a completed, not yet analysed batch")
        with self._lock():
            self._write("analysis.claim.json", {"scope": SYNTHETIC, "batch_id": permit.batch_id,
                                               "package": self.package.identity, "fixture": permit.fixture_hash})
            # Durable, append-only record of the claim BEFORE any label access.
            self._event("ANALYSIS_CLAIMED", batch_id=state["batch_id"],
                        claim_sha256=file_sha256(self.sandbox.path("analysis.claim.json")))
            self._hook("after_analysis_claim")
            self._analysis_gate_hash = file_sha256(self.sandbox.path("state.json"))
            # Analysis after a recovery uses its durable exposure batch, not a
            # synthetic permit's originally proposed batch identifier.
            access = self._access(state["batch_id"], "analysis")
            try:
                from src.experiment.phase5_analysis import analyse_synthetic
                report = analyse_synthetic(self, access)
                self._hook("after_analysis_compute")
                self._write("analysis.json", report)
            finally:
                self._revoke()
            state["status"] = "ANALYSED"
            state["analysis_sha256"] = file_sha256(self.sandbox.path("analysis.json"))
            self._write("state.json", state, replace=True)
            self._event("ANALYSIS_COMPLETED", batch_id=state["batch_id"])
            return report


def source_identity():
    return content_sha256({p: canonical_sha256(p) for p in EXECUTOR_SOURCES})


# ------------------------------------------------------------------ executable-candidate identity (N4)
CANDIDATE_ROOT = Path("reports/phase5_executable_freeze_candidate")
CANDIDATE_STATUS = "CANDIDATE — NOT AUTHORIZED FOR REAL PROTECTED EVALUATION."
EXECUTOR_SOURCES = ("src/experiment/phase5_execution.py", "src/experiment/phase5_inputs.py",
                    "src/experiment/phase5_analysis.py", "src/experiment/phase5_fixture.py",
                    # Phase 5.5F integration adapters (part of the executor identity).
                    "src/experiment/phase5_window_index.py", "src/experiment/phase5_inference.py",
                    "src/experiment/phase5_ledger.py", "src/experiment/phase5_dev_fixture.py")
CANDIDATE_SOURCES = EXECUTOR_SOURCES + (
    "scripts/run_phase5_fixture_executor.py", "tests/test_phase5_executor.py",
    "tests/test_phase5_synthetic_statistics.py", "tests/_recovery_race_worker.py", "tests/test_phase5_freeze_candidate.py",
    "tests/test_phase5_protected_read_guard.py", "docs/phase5_5b_synthetic_executor.md",
    "docs/phase5_5c_executor_hardening.md", "docs/phase5_5d_security_corrections.md",
    "docs/phase5_5e_admission_race_fix.md",
    "tests/test_phase5_integration.py", "docs/phase5_5f_integration.md",
    "tests/test_phase5_integration_corrections.py", "docs/phase5_5g_integration_corrections.md",
)
CANDIDATE_FILES = ("synthetic_evidence.json", "integration_evidence.json", "focused_tests.xml",
                   "regression_tests.xml", "development_sources.json")
CANDIDATE_KEYS = {
    "status", "stage", "review_status", "real_execution_enabled", "authorization_issued", "independently_approved",
    "trusted_package_commit", "trusted_package_manifest_sha256", "original_scientific_registry_sha256",
    "executor_source_identity", "sources", "files", "generated_from_head", "tracked_tree_clean_at_generation",
    "proposed_real_looks", "real_ledger_rows", "restrictions", "interfaces", "coverage",
}


def verify_executable_candidate(reviewed_commit, *, committed_reader=None, root=CANDIDATE_ROOT):
    """Bind the working executor and its synthetic evidence to an externally supplied reviewed commit.

    ``reviewed_commit`` must come from the independent-review / human-authorization
    record, never from this code. The candidate manifest is read from that commit's
    Git objects; the working manifest, every candidate source and evidence file, the
    executor identity and the package trust anchors must all equal it. Editing code
    or metadata therefore cannot approve itself: a new hash only passes once a
    reviewer names the commit that contains it.
    """
    from hashlib import sha256
    require(isinstance(reviewed_commit, str) and re.fullmatch(r"[0-9a-f]{40}", reviewed_commit or ""),
            "a full reviewed commit SHA is required")
    read = committed_reader or (lambda path: subprocess.run(
        ["git", "show", f"{reviewed_commit}:{Path(path).as_posix()}"], capture_output=True, check=True).stdout)
    root = Path(root)
    try:
        trusted = read(CANDIDATE_ROOT / "manifest.json")
    except (subprocess.CalledProcessError, OSError, KeyError) as exc:
        raise Refused("candidate manifest absent from the reviewed commit") from exc
    require(canonical_bytes(root / "manifest.json") == trusted.replace(b"\r\n", b"\n"),
            "working candidate manifest differs from the reviewed commit")
    manifest = json.loads(trusted)
    require(set(manifest) == CANDIDATE_KEYS, "unexpected candidate manifest structure")
    require(manifest["status"] == CANDIDATE_STATUS and manifest["real_execution_enabled"] is False
            and manifest["authorization_issued"] is False, "candidate status/authorization fields changed")
    require(manifest["trusted_package_commit"] == TRUSTED_COMMIT
            and manifest["trusted_package_manifest_sha256"] == TRUSTED_MANIFEST, "trust anchors differ from the reviewed candidate")
    require(tuple(sorted(manifest["sources"])) == tuple(sorted(CANDIDATE_SOURCES))
            and tuple(sorted(manifest["files"])) == tuple(sorted(CANDIDATE_FILES)), "candidate inventory changed")
    for path, digest in manifest["sources"].items():
        require(sha256(read(path).replace(b"\r\n", b"\n")).hexdigest() == digest, "reviewed-commit source differs from its manifest")
        require(canonical_sha256(path) == digest, f"candidate source modified after review: {path}")
    for name, digest in manifest["files"].items():
        require(sha256(read(CANDIDATE_ROOT / name).replace(b"\r\n", b"\n")).hexdigest() == digest,
                "reviewed-commit evidence differs from its manifest")
        require(canonical_sha256(root / name) == digest, f"candidate evidence modified after review: {name}")
    expected_identity = content_sha256({p: manifest["sources"][p] for p in EXECUTOR_SOURCES})
    require(manifest["executor_source_identity"] == expected_identity == source_identity(), "executor identity mismatch")
    evidence = json.loads(canonical_bytes(root / "synthetic_evidence.json"))
    require(evidence["executor_source_identity"] == expected_identity, "synthetic evidence was produced by other code")
    require(FrozenPackage.load().identity == manifest["trusted_package_manifest_sha256"], "frozen package identity mismatch")
    return manifest
