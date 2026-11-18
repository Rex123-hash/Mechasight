"""Phase 5.6C-r2: protected-activation candidate (disabled), tested without protected data.

Covers the artifact-binding interface, the human batch-authorization record, token
separation, the preflight order and the disabled real path. Uses development mirrors,
synthetic records and temporary sandboxes only; the real ledger is only read.
"""
from __future__ import annotations

import copy
import json
import secrets
import shutil
import tempfile
from pathlib import Path

import pytest

from src.experiment import phase5_input_manifest as mi
from src.experiment import phase5_protected as pp
from src.experiment.phase5_execution import Executor, FrozenPackage, authorize_fixture
from src.experiment.phase5_inference import FrozenL1Adapter
from src.experiment.phase5_inputs import Refused, Sandbox, atomic_json
from src.experiment.phase5_mirror_fixture import create_mirror_fixture
from src.experiment.provenance import canonical_json, file_sha256

REAL_LEDGER = Path("reports/experiment_results/test_look_ledger.jsonl")


@pytest.fixture(scope="module")
def frozen():
    return FrozenPackage.load()


@pytest.fixture(scope="module")
def records(frozen):
    bindings = pp.rehearsal_bindings(frozen)
    digest = pp.validate_bindings(bindings, frozen)
    scoring, analysis = secrets.token_hex(32), secrets.token_hex(32)
    return bindings, digest, pp.rehearsal_authorization(frozen, digest, scoring, analysis), scoring, analysis


def _count_reads(monkeypatch):
    reads = []
    real_parquet, real_h5 = mi.pd.read_parquet, mi.h5py.File
    monkeypatch.setattr(mi.pd, "read_parquet", lambda *a, **k: reads.append("parquet") or real_parquet(*a, **k))
    monkeypatch.setattr(mi.h5py, "File", lambda *a, **k: reads.append("hdf5") or real_h5(*a, **k))
    return reads


# ------------------------------------------------------------------ artifact bindings
def test_protected_proposal_comes_from_frozen_inventory_and_is_never_executable(frozen, monkeypatch):
    stats = []
    monkeypatch.setattr(Path, "stat", lambda self, *a, **k: stats.append(str(self)) or pytest.fail(f"stat of {self}"))
    proposal = pp.protected_bindings_proposal(frozen)
    monkeypatch.undo()
    assert stats == [] and proposal["status"] == pp.PROPOSED and proposal["scope"] == pp.PROTECTED
    plan = frozen.proposed["identity"]["cohort_hashes"]
    for cell, entry in proposal["cells"].items():
        assert entry["cohort_identity"] == plan[entry["machine"]]
        assert all(a["sha256"] is None for a in entry["artifacts"].values())
    with pytest.raises(Refused, match="protected execution is disabled"):
        pp.validate_bindings(proposal, frozen)


@pytest.mark.parametrize("tamper", ["extra_field", "path", "cohort", "registry", "missing_cell", "stat"])
def test_rehearsal_bindings_must_equal_the_trusted_development_manifests(frozen, records, tamper):
    bindings = copy.deepcopy(records[0])
    cell = bindings["cells"]["RoboticArm|S"]
    if tamper == "extra_field":
        bindings["approved_by_self"] = True
    elif tamper == "path":
        cell["artifacts"]["features"]["path"] = "data/interim/phase32_common_test/RoboticArm/S/common_test.parquet"
    elif tamper == "cohort":
        cell["cohort_identity"] = "0" * 64
    elif tamper == "registry":
        bindings["registry_sha256"] = "0" * 64
    elif tamper == "missing_cell":
        del bindings["cells"]["BrushlessMotor|T"]
    else:
        cell["artifacts"]["spectral"]["stat"]["bytes"] += 1
    with pytest.raises(Refused):
        pp.validate_bindings(bindings, frozen)


# ------------------------------------------------------------------ authorization and tokens
def test_authorization_template_is_unsigned_and_rejected(frozen):
    proposal = pp.protected_bindings_proposal(frozen)
    from src.experiment.provenance import content_sha256
    template = pp.authorization_template(frozen, content_sha256(proposal), "a" * 40)
    assert all(d["decision"] == "UNRESOLVED" for d in template["human_decisions"].values())
    assert template["scoring_token_sha256"] is None and template["analysis_token_sha256"] is None
    with pytest.raises(Refused):
        pp.validate_authorization(template, frozen, bindings_sha256=content_sha256(proposal), scope=pp.PROTECTED)


@pytest.mark.parametrize("edit", ["counts", "cap", "decision", "reviewer", "same_tokens", "bindings", "plan", "extra"])
def test_altered_authorization_record_is_refused(frozen, records, edit):
    _, digest, record, _, _ = records
    record = copy.deepcopy(record)
    if edit == "counts":
        record["ledger_end_count"] = 429
    elif edit == "cap":
        record["max_total_rows"] = 892
    elif edit == "decision":
        record["human_decisions"]["H5"]["decision"] = "UNRESOLVED"
    elif edit == "reviewer":
        record["independent_reviewer"]["verdict"] = "YELLOW"
    elif edit == "same_tokens":
        record["analysis_token_sha256"] = record["scoring_token_sha256"]
    elif edit == "bindings":
        record["bindings_sha256"] = "0" * 64
    elif edit == "plan":
        record["planned_ids_sha256"] = "0" * 64
    else:
        record["self_issued"] = True
    with pytest.raises(Refused):
        pp.validate_authorization(record, frozen, bindings_sha256=digest, scope=pp.REHEARSAL)


def test_tokens_are_phase_separated(records):
    _, _, record, scoring, analysis = records
    pp.check_token(record, scoring, "score")
    pp.check_token(record, analysis, "analysis")
    with pytest.raises(Refused, match="cannot authorize score"):
        pp.check_token(record, analysis, "score")
    with pytest.raises(Refused, match="cannot authorize analysis"):
        pp.check_token(record, scoring, "analysis")
    with pytest.raises(Refused, match="does not match"):
        pp.check_token(record, secrets.token_hex(32), "score")


def test_rehearsal_authorization_can_never_authorize_protected_scope(frozen, records):
    _, digest, record, _, _ = records
    with pytest.raises(Refused):
        pp.validate_authorization(record, frozen, bindings_sha256=digest, scope=pp.PROTECTED)


# ------------------------------------------------------------------ disabled real path
def test_protected_score_and_analysis_refuse_before_any_read_or_ledger_change(frozen, records, monkeypatch):
    bindings, _, record, scoring, analysis = records
    before = file_sha256(REAL_LEDGER)
    reads = _count_reads(monkeypatch)
    assert pp.protected_execution_enabled() is False
    with pytest.raises(Refused, match="protected execution is disabled"):
        pp.score_all(frozen, scope=pp.PROTECTED, bindings=bindings, authorization=record, token=scoring)
    with pytest.raises(Refused, match="protected execution is disabled"):
        pp.preflight(frozen, scope=pp.PROTECTED, bindings=bindings, authorization=record, token=analysis, phase="analysis")
    assert reads == [] and file_sha256(REAL_LEDGER) == before == pp.REAL_LEDGER_SHA256


def test_preflight_checks_real_ledger_identity(frozen, records, monkeypatch):
    bindings, _, record, scoring, _ = records
    pp.preflight(frozen, scope=pp.REHEARSAL, bindings=bindings, authorization=record, token=scoring, phase="score")
    monkeypatch.setattr(pp, "REAL_LEDGER_SHA256", "0" * 64)
    with pytest.raises(Refused, match="closed 196-row identity"):
        pp.preflight(frozen, scope=pp.REHEARSAL, bindings=bindings, authorization=record, token=scoring, phase="score")


# ------------------------------------------------------------------ binding enforced at the executor read path
@pytest.fixture(scope="module")
def bound_template(frozen, records):
    root = Path(tempfile.mkdtemp(prefix="mechasight-SYNTHETIC-5-6c-r2-template-"))
    box = Sandbox(root)
    fixture = create_mirror_fixture(box, frozen)
    box.path("bindings.json").write_text(canonical_json(records[0]), encoding="utf-8")
    digest = file_sha256(box.path("bindings.json"))
    fixture["files"]["bindings.json"] = digest
    fixture["bindings"] = {"file": "bindings.json", "scope": pp.REHEARSAL, "sha256": digest}
    atomic_json(box.path("fixture.json"), fixture, replace=True)
    yield root
    shutil.rmtree(root, ignore_errors=True)


@pytest.mark.parametrize("tamper", ["record_changed", "stat_mismatch", "protected_scope"])
def test_binding_mismatch_stops_after_exposure_before_artifact_reads(frozen, bound_template, tmp_path, monkeypatch, tamper):
    root = tmp_path / "SYNTHETIC-bound"
    shutil.copytree(bound_template, root)
    box = Sandbox(root)
    if tamper == "record_changed":
        document = json.loads(box.path("bindings.json").read_text(encoding="utf-8"))
        document["content_hash_policy"] = "edited after authorization"
        box.path("bindings.json").write_text(canonical_json(document), encoding="utf-8")
    elif tamper == "stat_mismatch":
        document = json.loads(box.path("bindings.json").read_text(encoding="utf-8"))
        for cell in document["cells"].values():
            cell["artifacts"]["features"]["stat"]["mtime_ns"] += 1
        box.path("bindings.json").write_text(canonical_json(document), encoding="utf-8")
        fixture = box.json("fixture.json")
        fixture["files"]["bindings.json"] = fixture["bindings"]["sha256"] = file_sha256(box.path("bindings.json"))
        atomic_json(box.path("fixture.json"), fixture, replace=True)
    else:
        fixture = box.json("fixture.json")
        fixture["bindings"]["scope"] = pp.PROTECTED
        atomic_json(box.path("fixture.json"), fixture, replace=True)
    executor = Executor(box, frozen, predictor=FrozenL1Adapter(frozen))
    reads = _count_reads(monkeypatch)
    with pytest.raises(Refused, match="binding|rehearsal bindings"):
        executor.score_all(authorize_fixture(box, frozen, "score"))
    assert reads == []                                                   # no bound artifact was opened
    assert box.json("state.json")["status"] == "FAILED_TECHNICAL"        # exposure recorded, failure attributable
