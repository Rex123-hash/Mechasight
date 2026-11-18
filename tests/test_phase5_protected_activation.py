"""The activated protected chain end to end, with REAL activation checks (docs/phase5_protected_architecture.md).

No preflight, activation, tree or source check is mocked. Authority comes from a temporary
Git *records repository* (``pp.GIT_ROOT``):

1. a "reviewed" commit holding byte copies of the running protected executable and the closed
   196-row ledger (tracked, as in the project repository);
2. one later commit adding the three synthetic human records (bindings, batch authorization,
   ACTIVATION.json) bound to that commit;
3. for the recovery test, a further commit adding recovery records.

``verified_activation`` therefore runs exactly as in production: committed and unmodified records,
a single introducing commit, a non-circular reviewed ancestor, no source change since review and a
running source identical to the reviewed Git objects.

Only DATA locations are redirected:
- protected artifacts -> permitted development source-validation artifacts;
- sealed Phase-4 / Phase-3.2 stores -> temporary copies built from development scores;
- the real ledger -> the records repository's sealed copy (committed after every operation, as the
  protocol requires);
- batch authorizations and results -> temporary directories.

Labels are relabeled to synthetic parent-homogeneous values after the real projection read,
because development validation has no faults. Every credential, token and record here is
SYNTHETIC, lives in temporary directories and can never be accepted as real protected
authority: the real records repository (the project) has no activation records. Nothing here
is a protected result.
"""
from __future__ import annotations

from datetime import date
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import time

import pandas as pd
import pytest

from src.experiment import phase5_execution as ex_module
from src.experiment import phase5_input_manifest as mi
from src.experiment import phase5_protected as pp
from src.experiment.ledger_history import verify_ledger_history
from src.experiment.phase5_execution import Executor, FixturePermit, FrozenPackage, authorize_fixture
from src.experiment.phase5_inference import FrozenL1Adapter
from src.experiment.phase5_inputs import Refused, SCORE_COLUMNS, Sandbox
from src.experiment import phase5_ledger as ledger_module
from src.experiment.phase5_ledger import ledger_sealed
from src.experiment.provenance import canonical_bytes, canonical_sha256, content_sha256, file_sha256, ordered_id_hash

REAL_LEDGER = Path("reports/experiment_results/test_look_ledger.jsonl")
SYNTHETIC_BY = "SYNTHETIC test fixture (not a human decision)"


@pytest.fixture(autouse=True)
def _ledger_primitives_never_touch_the_real_ledger(monkeypatch):
    """Harness guard: the frozen conftest guards TestLookLedger; this guards the replace and seal primitives."""
    from src.experiment import phase5_ledger as _ledger_module
    real_ledger = Path("reports/experiment_results/test_look_ledger.jsonl").resolve()
    real_replace, real_seal = _ledger_module._replace_sealed, _ledger_module.seal_ledger

    def guarded_replace(source, target):
        assert Path(target).resolve() != real_ledger, "TEST HARNESS GUARD: refusing to replace the real ledger"
        return real_replace(source, target)

    def guarded_seal(path):
        assert Path(path).resolve() != real_ledger, "TEST HARNESS GUARD: refusing to seal the real ledger"
        return real_seal(path)
    monkeypatch.setattr(_ledger_module, "_replace_sealed", guarded_replace)
    monkeypatch.setattr(_ledger_module, "seal_ledger", guarded_seal)


# ------------------------------------------------------------------ module data (development only)
@pytest.fixture(scope="module")
def frozen():
    return FrozenPackage.load()


@pytest.fixture(scope="module")
def env(frozen, tmp_path_factory):
    """Development manifests, synthetic approved bindings and sealed-store copies (built once)."""
    dev = {cell: mi.development_manifest(frozen, machine=cell.split("|")[0], regime=cell.split("|")[1]).snapshot()
           for cell in pp.CELLS}
    root = tmp_path_factory.mktemp("SYNTHETIC-sealed")
    p4, p32 = root / "phase4_staging", root / "phase3_2"
    (p4 / "scores").mkdir(parents=True)
    tables, detectors = {}, frozen.registry["detectors"]
    for unit in frozen.registry["units"]:
        for eid in unit["L0"].values():
            frame = pd.read_csv(Path(detectors[eid]["bundle_path"]) / "validation_scores.csv", usecols=list(SCORE_COLUMNS),
                                dtype={"segment_id": str, "parent_recording": str}, float_precision="round_trip")
            frame.loc[:, list(SCORE_COLUMNS)].to_csv(p4 / "scores" / f"{eid}.csv", index=False, lineterminator="\n")
            tables[eid] = {"file": f"{eid}.csv", "sha256": canonical_sha256(p4 / "scores" / f"{eid}.csv"),
                           "cohort_hash": ordered_id_hash(tuple(frame["segment_id"]))}
        if unit["L1"]["source"] != "new_fit_phase5":
            eid = unit["L1"]["experiment_id"]
            frame = pd.read_csv(Path(detectors[eid]["bundle_path"]) / "validation_scores.csv",
                                dtype={"segment_id": str, "parent_recording": str}, float_precision="round_trip")
            bundle = p32 / eid
            bundle.mkdir(parents=True, exist_ok=True)
            frame.loc[:, ["segment_id", "parent_recording", "domain", "anomaly_label", "score"]].to_parquet(
                bundle / "scores.parquet", index=False)
            (bundle / "bundle_manifest.json").write_text(json.dumps(
                {"experiment_id": eid, "files": {"scores.parquet": file_sha256(bundle / "scores.parquet")}}), encoding="utf-8")
            (bundle / "provenance.json").write_text(json.dumps(
                {"test_cohort_hash": ordered_id_hash(tuple(frame["segment_id"]))}), encoding="utf-8")
    (p4 / "score_manifest.json").write_text(json.dumps({"tables": tables}), encoding="utf-8")
    cells = {}
    for cell in pp.CELLS:
        machine, regime = cell.split("|")
        artifacts = {}
        for role in pp.ROLES:
            path = dev[cell]["artifacts"][role]["path"]
            stat = (pp.ROOT / path).stat()
            artifacts[role] = {"path": path, "stat": {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}, "sha256": None}
        cells[cell] = {"machine": machine, "regime": regime, "cohort_identity": dev[cell]["cohort_identity"],
                       "artifacts": artifacts}
    plan = pp._plan(frozen)
    bindings = {"schema": pp.BINDING_SCHEMA, "scope": pp.PROTECTED, "status": pp.APPROVED, "trusted_commit": pp.TRUSTED_COMMIT,
                "registry_sha256": plan["registry_sha256"], "plan_identity_sha256": plan["identity_sha256"], "cells": cells,
                "content_hash_policy": "H3 (SYNTHETIC test bindings over development artifacts)",
                "approval": {"by": SYNTHETIC_BY, "on": date.today().isoformat()}}
    return {"dev": dev, "p4": p4, "p32": p32, "bindings": bindings}


# ------------------------------------------------------------------ synthetic records repository
def git(root, *args):
    return subprocess.run(["git", "-C", str(root), "-c", "user.name=SYNTHETIC test", "-c", "user.email=synthetic@example.invalid",
                           "-c", "core.autocrlf=false", *args], capture_output=True, text=True, check=True).stdout.strip()


def _write_json(root, relative, document):
    path = Path(root) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


LEDGER_IN_REPO = Path("reports/experiment_results/test_look_ledger.jsonl")


def _copy_executable(root, mutate=None):
    for name in pp.PROTECTED_EXECUTABLE:
        target = Path(root) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = canonical_bytes(pp.ROOT / name)
        target.write_bytes(mutate(name, raw) if mutate else raw)
    ledger = Path(root) / LEDGER_IN_REPO
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_bytes(REAL_LEDGER.read_bytes())
    (Path(root) / ".gitignore").write_text("*.jsonl.lock\n*.jsonl.pending-*\n", encoding="utf-8", newline="\n")


def commit_all(root, message):
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", message)
    return git(root, "rev-parse", "HEAD")


def make_records(frozen, env, reviewed):
    """SYNTHETIC bindings, batch authorization and activation bound to ``reviewed``; and the tokens."""
    bindings = json.loads(json.dumps(env["bindings"]))
    scoring, analysis = secrets.token_hex(32), secrets.token_hex(32)
    record = pp.authorization_template(frozen, content_sha256(bindings), reviewed)
    today = date.today().isoformat()
    record.update(human_decisions={h: {"decision": "APPROVED", "by": SYNTHETIC_BY, "on": today} for h in pp.HUMAN_DECISIONS},
                  independent_reviewer={"name": "SYNTHETIC test reviewer", "reviewed_commit": reviewed, "verdict": "GREEN"},
                  scoring_token_sha256=pp.token_sha256(scoring), analysis_token_sha256=pp.token_sha256(analysis),
                  issued_by=SYNTHETIC_BY, issued_on=today)
    activation = {"schema": pp.ACTIVATION_SCHEMA, "enabled": True, "activated_by": SYNTHETIC_BY, "activated_on": today,
                  "reviewed_commit": reviewed, "authorization_sha256": content_sha256(record),
                  "bindings_sha256": content_sha256(bindings)}
    return {pp.APPROVED_BINDINGS_PATH: bindings, pp.AUTHORIZATION_PATH: record, pp.ACTIVATION_PATH: activation}, \
        {"score": scoring, "analysis": analysis}


def build_repo(root, frozen, env, *, edit=None, commit_records=True):
    """reviewed executable commit -> one records commit. ``edit(records)`` may alter records before writing."""
    root.mkdir(parents=True)
    git(root, "init", "-q")
    _copy_executable(root)
    reviewed = commit_all(root, "SYNTHETIC reviewed executable")
    records, tokens = make_records(frozen, env, reviewed)
    if edit:
        edit(records)
    for path, document in records.items():
        if document is not None:
            _write_json(root, path, document)
    if commit_records:
        commit_all(root, "SYNTHETIC human records")
    return {"root": root, "reviewed": reviewed, "records": records, "tokens": tokens}


def redirect(monkeypatch, tmp, frozen, env, repo):
    """Point the authority at the records repository and the data locations at development copies."""
    monkeypatch.setattr(pp, "GIT_ROOT", repo["root"])
    ledger = Path(repo["root"]) / LEDGER_IN_REPO
    if not ledger.exists():
        shutil.copyfile(REAL_LEDGER, ledger)
    if not ledger_sealed(ledger):
        ledger_module.seal_ledger(ledger)           # sealing the real ledger is a human activation step
    monkeypatch.setattr(pp, "REAL_LEDGER", ledger)
    monkeypatch.setattr(pp, "BATCH_AUTH_ROOT", tmp / "batches")
    monkeypatch.setattr(pp, "RESULTS_ROOT", tmp / "results")
    dev = env["dev"]
    monkeypatch.setattr(pp, "_protected_paths", lambda m, r: {k: dev[f"{m}|{r}"]["artifacts"][k]["path"] for k in pp.ROLES})
    real_plan = pp._plan
    monkeypatch.setattr(pp, "_plan", lambda package=None: {**real_plan(package), "cohorts": {
        m: dev[f"{m}|S"]["cohort_identity"] for m in ("RoboticArm", "BrushlessMotor")}})
    real_reference = mi._reference_document

    def reference(machine, regime, scope):
        if scope != mi.PROTECTED:
            return real_reference(machine, regime, scope)
        doc = real_reference(machine, regime, mi.DEVELOPMENT)
        doc.update(scope=mi.PROTECTED, activation="DISABLED", cohort_role="common_evaluation", artifacts={},
                   unresolved=["test redirection: development cohort stands in for the protected cohort"])
        return doc
    monkeypatch.setattr(mi, "_reference_document", reference)
    monkeypatch.setattr(pp, "PHASE4_STAGING", env["p4"])
    monkeypatch.setattr(pp, "PHASE32_STORE", env["p32"])
    real_labels = mi.read_protected_labels

    def labels(manifest, *, capability=None):
        frame = real_labels(manifest, capability=capability)          # the real, capability-checked projection read
        parents = sorted(frame["parent_recording"].unique())
        anomalous = set(parents[len(parents) // 2:])
        return frame.assign(anomaly_label=["loosescrewsA" if p in anomalous else "normal" for p in frame["parent_recording"]])
    monkeypatch.setattr(mi, "read_protected_labels", labels)
    return ledger


@pytest.fixture
def active(frozen, env, tmp_path, monkeypatch):
    repo = build_repo(tmp_path / "records", frozen, env)
    ledger = redirect(monkeypatch, tmp_path, frozen, env, repo)
    return {**repo, "ledger": ledger, "tmp": tmp_path}


def staging(tmp, name="staging"):
    return str(tmp.parent / f"{tmp.name}-{name}")


# ------------------------------------------------------------------ the principal end-to-end test
def test_activated_chain_reproduces_all_24_l1_and_208_l2_with_real_activation_checks(frozen, env, active, monkeypatch,
                                                                                     record_property):
    real_before = file_sha256(REAL_LEDGER)
    chain = pp.verified_activation()                                   # the real verifier, against the records repo
    assert chain["reviewed_commit"] == active["reviewed"] and chain["records_commit"] != active["reviewed"]
    tokens = active["tokens"]
    with pytest.raises(Refused, match="cannot authorize score"):
        pp.score_all(frozen, scope=pp.PROTECTED, token=tokens["analysis"])
    with pytest.raises(Refused, match="does not match"):
        pp.score_all(frozen, scope=pp.PROTECTED, token=secrets.token_hex(32))
    with pytest.raises(Refused, match="supplied authorization differs"):
        pp.score_all(frozen, scope=pp.PROTECTED, authorization={**chain["authorization"], "issued_by": "edited"},
                     token=tokens["score"])

    # Every protected read must happen after the 232 looks are durable and the exposure is recorded.
    ledger, reads, label_reads, bypass = active["ledger"], [], [], {}
    from src.experiment import phase4 as phase4_module

    def observe(kind):
        rows = len(verify_ledger_history(ledger, authorization_root=pp.BATCH_AUTH_ROOT).rows)
        exposures = list(Path(staging(active["tmp"])).joinpath("exposures").glob("*.json"))
        reads.append((kind, rows, len(exposures)))
    real_parquet, real_h5, real_staged = mi.pd.read_parquet, mi.h5py.File, phase4_module.ScoreStaging.read_scores
    monkeypatch.setattr(mi.pd, "read_parquet", lambda *a, **k: observe("parquet") or real_parquet(*a, **k))
    monkeypatch.setattr(mi.h5py, "File", lambda *a, **k: observe("hdf5") or real_h5(*a, **k))
    monkeypatch.setattr(phase4_module.ScoreStaging, "read_scores", lambda self, eid: observe("sealed") or real_staged(self, eid))
    real_labels = mi.read_protected_labels
    monkeypatch.setattr(mi, "read_protected_labels", lambda m, **k: label_reads.append(
        Path(staging(active["tmp"])).joinpath("analysis.claim.json").is_file()) or real_labels(m, **k))
    # Direct-helper bypass attempts DURING scoring, with the live scoring capability: labels stay unreadable.
    issued = []
    real_issue = Executor._issue
    monkeypatch.setattr(Executor, "_issue", lambda self, b, p: issued.append(real_issue(self, b, p)) or issued[-1])
    real_hook = Executor._hook

    def hook(self, stage, unit=""):
        if stage == "before_read" and not bypass:
            access = issued[-1]
            for name, call in {"labels_via_frame": lambda: pp.protected_label_frame(self, access, "RoboticArm"),
                               "labels_via_reader": lambda: real_labels(
                                   mi.protected_manifest(frozen, machine="RoboticArm", regime="S"), capability=(self, access)),
                               "analysis_comparator": lambda: pp.read_sealed_comparator(
                                   self, access, next(iter(self._historical_policy.keys() - {
                                       e for u in frozen.registry["units"] for e in u["L0"].values()})), "analysis",
                                   expected_sha256="0" * 64)}.items():
                try:
                    call()
                    bypass[name] = "ACCEPTED"
                except Refused as exc:
                    bypass[name] = str(exc)
        return real_hook(self, stage, unit)
    monkeypatch.setattr(Executor, "_hook", hook)

    box, executor, state, checks = pp.score_all(frozen, scope=pp.PROTECTED, token=tokens["score"],
                                                workdir=staging(active["tmp"]))
    assert state["status"] == "SCORED" and len(state["tables"]) == 232
    assert reads and all(rows == 428 and exposures == 1 for _, rows, exposures in reads)       # exposure before every read
    assert label_reads == [] and not any("labels" in n for _, n in box.events)                   # no label access while scoring
    assert set(bypass) == {"labels_via_frame", "labels_via_reader", "analysis_comparator"}
    assert all(v != "ACCEPTED" and "never readable while scoring" in v for v in bypass.values()), bypass

    # Ledger accounting: 196 -> 428, the planned 232 identities, each with its frozen look identity.
    history = verify_ledger_history(ledger, authorization_root=pp.BATCH_AUTH_ROOT)
    rows = history.rows
    assert len(rows) == 428 and all(f"scope={pp.PROTECTED}" in r["reason"] for r in rows[196:])
    assert sorted(r["experiment_id"] for r in rows[196:]) == sorted(executor.operations)
    for r in rows[196:]:
        identity = executor._look_identity(r["experiment_id"])
        assert (r["machine"], r["regime"], r["model_config_identity"], r["test_cohort_hash"], r["git_sha"]) == (
            identity["machine"], identity["regime"], identity["model_config_identity"], identity["cohort_hash"],
            active["reviewed"])
    batch_record = json.loads((pp.BATCH_AUTH_ROOT / chain["authorization"]["batch_id"] / "batch_authorization.json")
                              .read_text(encoding="utf-8"))
    assert (batch_record["ledger_start_count"], batch_record["ledger_end_count"], batch_record["code_git_sha"]) == (
        196, 428, active["reviewed"])

    # All 24 frozen L1 models: 3,744 scores, frozen thresholds and model identities, against the frozen references.
    l1 = [e for e, op in executor.operations.items() if op["operation"] == "new_L1_inference"]
    l2 = [e for e, op in executor.operations.items() if op["operation"] == "derived_L2_scoring"]
    assert len(l1) == 24 and len(l2) == 208
    l1_rows, l1_diff = 0, 0.0
    for eid in l1:
        payload = box.json(state["tables"][eid]["file"])
        detector = frozen.registry["detectors"][eid]
        assert payload["computation"]["kind"] == "frozen_L1_inference" and payload["computation"]["model"] == detector["model_manifest"]
        assert payload["provenance"]["threshold"] == detector["threshold"] and payload["provenance"]["model"] == detector["model_manifest"]
        got = pd.DataFrame(payload["rows"]).set_index("segment_id")
        ref = pd.read_csv(Path(detector["bundle_path"]) / "validation_scores.csv", dtype={"segment_id": str},
                          float_precision="round_trip").set_index("segment_id")
        assert sorted(got.index) == sorted(ref.index)
        l1_rows += len(got)
        l1_diff = max(l1_diff, float((got["score"] - ref.loc[got.index, "score"]).abs().max()))
    assert l1_rows == 3744 and l1_diff <= 1e-9
    # All 208 L2 units: 32,448 rows, exact calibration identities and thresholds, against the persisted fused scores.
    fused = pd.read_csv("reports/phase5_development/l2/l2_fused_validation_scores.csv", dtype={"segment_id": str},
                        float_precision="round_trip").set_index(["calibration_id", "segment_id"])["fused_score"]
    l2_rows, l2_diff = 0, 0.0
    for eid in l2:
        payload = box.json(state["tables"][eid]["file"])
        calibration = frozen.registry["calibrations"][eid]
        assert payload["computation"]["calibration_sha256"] == content_sha256(calibration)
        assert payload["computation"]["component_origin"] == "SEALED_PHASE4_COMMON_EVALUATION"
        assert payload["provenance"]["threshold"] == calibration["threshold"]
        assert payload["provenance"]["calibration_sha256"] == content_sha256(calibration)
        l2_rows += len(payload["rows"])
        l2_diff = max(l2_diff, max(abs(r["score"] - fused[(eid, r["segment_id"])]) for r in payload["rows"]))
    assert l2_rows == 32448 and l2_diff <= 1e-12
    for key, value in {"l1_models": len(l1), "l1_scores": l1_rows, "l1_max_abs_diff": l1_diff, "l2_units": len(l2),
                       "l2_rows": l2_rows, "l2_max_abs_diff": l2_diff, "ledger_rows_after_score": len(rows),
                       "protected_reads_after_exposure": len(reads), "reviewed_commit_is_records_repo": True}.items():
        record_property(key, value)
    # H3: post-exposure hashes kept by this executor only, for the 8 bound development artifacts.
    h3 = json.loads(box.path("h3_observed_hashes.json").read_text(encoding="utf-8"))
    assert set(h3) == {a["path"] for c in env["bindings"]["cells"].values() for a in c["artifacts"].values()}
    assert all(file_sha256(pp.ROOT / p) == d for p, d in h3.items())
    assert sum(n.startswith("trusted:") for _, n in box.events) == 24

    # After the operation its authority is gone: a direct reader call with the (stale) capability is refused.
    with pytest.raises(Refused, match="no protected authority"):
        mi.read_features(mi.protected_manifest(frozen, machine="RoboticArm", regime="S"), capability=(executor, issued[-1]))
    with pytest.raises(Refused, match="executor capability"):
        mi.read_features(mi.protected_manifest(frozen, machine="RoboticArm", regime="S"))
    with pytest.raises(Refused, match="no protected authority"):
        executor.ledger.append(type("P", (), {"batch_id": "SYNTHETIC-direct"})(), l1[:1], {l1[0]: l1[0] + "-r1"}, 428)
    assert len(verify_ledger_history(ledger, authorization_root=pp.BATCH_AUTH_ROOT).rows) == 428

    # Protocol: the next operation starts only from a committed ledger.
    with pytest.raises(Refused, match="not committed"):
        pp.analyse_once(frozen, box, executor, scope=pp.PROTECTED, token=tokens["analysis"])
    commit_all(active["root"], "SYNTHETIC: commit the ledger after score-all (protocol)")
    # Analysis: the scoring token cannot analyse; the analysis token runs analyse-once exactly once.
    with pytest.raises(Refused, match="cannot authorize analysis"):
        pp.analyse_once(frozen, box, executor, scope=pp.PROTECTED, token=tokens["score"])
    report = pp.analyse_once(frozen, box, executor, scope=pp.PROTECTED, token=tokens["analysis"])
    assert report["scope"] == pp.PROTECTED and report["not_real_performance"] is False
    assert report["authorization_sha256"] == chain["authorization_sha256"] == checks["authorization_sha256"]
    assert report["primary_contrasts"] == 140 and report["primary_intervals"] == 420 and report["replicates"] == 10_000
    assert label_reads and all(label_reads)                       # every label read came after the durable claim
    events = [json.loads(line)["state"] for line in box.path("attempts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events.index("COMPLETION_GATE_GREEN") < events.index("ANALYSIS_CLAIMED") < events.index("ANALYSIS_COMPLETED")
    with pytest.raises(Refused, match="already claimed"):
        pp.analyse_once(frozen, box, executor, scope=pp.PROTECTED, token=tokens["analysis"])
    assert len(verify_ledger_history(ledger, authorization_root=pp.BATCH_AUTH_ROOT).rows) == 428   # analysis adds no look
    archive = pp.RESULTS_ROOT / chain["authorization"]["batch_id"]
    assert (archive / "analysis.json").is_file() and (archive / "state.json").is_file()
    import importlib.util
    spec = importlib.util.spec_from_file_location("integrate_results", "paper/tools/integrate_results.py")
    integrator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(integrator)
    monkeypatch.setattr(integrator, "TAB", active["tmp"] / "tables")
    (active["tmp"] / "tables").mkdir()
    integrator.render(str(box.path("analysis.json")))
    assert (active["tmp"] / "tables" / "tab_primary_results.tex").is_file()
    assert file_sha256(REAL_LEDGER) == real_before == pp.REAL_LEDGER_SHA256       # the real ledger is untouched


# ------------------------------------------------------------------ crash and recovery: 428 -> 660, never 892
def _recovery_record(chain, batch_suffix, recovers, token):
    return {"schema": pp.RECOVERY_SCHEMA, "batch_id": f"{chain['authorization']['batch_id']}-recovery-{batch_suffix}",
            "recovers": recovers, "authorization_sha256": chain["authorization_sha256"], "approved_by": SYNTHETIC_BY,
            "approved_on": date.today().isoformat(), "token_sha256": pp.token_sha256(token)}


def test_crash_and_recovery_reach_660_and_the_892_attempt_fails_before_mutation(frozen, env, active, monkeypatch,
                                                                                record_property):
    chain = pp.verified_activation()
    batch = chain["authorization"]["batch_id"]
    r1, r2 = secrets.token_hex(32), secrets.token_hex(32)
    names = {k: (pp.RECOVERY_ROOT / f"recovery-{k}.json").as_posix() for k in (1, 2)}
    _write_json(active["root"], names[1], _recovery_record(chain, 1, batch, r1))
    _write_json(active["root"], names[2], _recovery_record(chain, 2, f"{batch}-recovery-1", r2))
    with pytest.raises(Refused, match="uncommitted"):
        pp.committed_recovery_record(names[1])
    commit_all(active["root"], "SYNTHETIC recovery records")
    crash = {"on": True}
    real_hook = Executor._hook

    def fault(self, stage, unit=""):
        if crash["on"] and stage == "before_read":
            raise OSError("SYNTHETIC crash after exposure, before any read")
        return real_hook(self, stage, unit)
    monkeypatch.setattr(Executor, "_hook", fault)
    work = staging(active["tmp"])
    with pytest.raises(OSError):
        pp.score_all(frozen, scope=pp.PROTECTED, token=active["tokens"]["score"], workdir=work)
    box = Sandbox(work)
    rows = lambda: len(verify_ledger_history(active["ledger"], authorization_root=pp.BATCH_AUTH_ROOT).rows)  # noqa: E731
    assert rows() == 428 and box.json("state.json")["status"] == "FAILED_TECHNICAL"
    commit_all(active["root"], "SYNTHETIC: commit the ledger after the interrupted score-all (protocol)")
    with pytest.raises(Refused, match="recovery token does not match"):
        pp.recover(frozen, box, recovery_record=names[1], token=r2)
    with pytest.raises(Refused, match="recoveries"):
        pp.recover(frozen, box, recovery_record="reports/phase5_protected/elsewhere.json", token=r1)
    with pytest.raises(OSError):                                       # recovery 1 exposes 232 -r1 looks, then crashes
        pp.recover(frozen, box, recovery_record=names[1], token=r1)
    assert rows() == 660 and box.json("state.json")["batch_id"] == f"{batch}-recovery-1"
    commit_all(active["root"], "SYNTHETIC: commit the ledger after the interrupted recovery (protocol)")
    crash["on"] = False
    ledger_before = file_sha256(active["ledger"])
    with pytest.raises(Refused, match="look budget exceeded"):          # 660 + 232 would be 892
        pp.recover(frozen, box, recovery_record=names[2], token=r2)
    assert rows() == 660 and file_sha256(active["ledger"]) == ledger_before
    assert not (pp.BATCH_AUTH_ROOT / f"{batch}-recovery-2").exists()    # no partial authorization
    assert not box.path(f"exposures/{batch}-recovery-2.json").exists()
    ids = [r["experiment_id"] for r in verify_ledger_history(active["ledger"], authorization_root=pp.BATCH_AUTH_ROOT).rows]
    assert len(ids) == len(set(ids)) == 660 and sum(i.endswith("-r1") for i in ids) == 232
    record_property("ledger_rows_sequence", "196 -> 428 (crash) -> 660 (recovery 1 crash) -> 892 refused, stays 660")


# ------------------------------------------------------------------ B1: activation authority adversarial tests
def _refused(match):
    with pytest.raises(Refused, match=match):
        pp.verified_activation()


def test_real_repository_has_no_activation_and_refuses_before_any_protected_stat(frozen, monkeypatch):
    assert pp.GIT_ROOT == pp.ROOT and not (pp.ROOT / pp.ACTIVATION_PATH).exists()
    assert pp.protected_execution_enabled() is False
    stats = []
    real_stat = Path.stat
    monkeypatch.setattr(Path, "stat", lambda self, *a, **k: stats.append(str(self)) or real_stat(self, *a, **k))
    for call in (pp.verified_activation, pp.load_protected_records,
                 lambda: mi.protected_manifest(frozen, machine="RoboticArm", regime="S"),
                 lambda: pp.validate_bindings(pp.protected_bindings_proposal(frozen), frozen),
                 lambda: pp.score_all(frozen, scope=pp.PROTECTED, token=secrets.token_hex(32))):
        with pytest.raises(Refused, match="protected execution is disabled"):
            call()
    assert not any("phase32_common_test" in s or "common_evaluation.spectral" in s for s in stats)
    assert not Path(str(REAL_LEDGER) + ".lock").exists()                 # refused before the single-process lock
    assert not hasattr(pp, "PROTECTED_EXECUTION_ENABLED") and not hasattr(pp, "_APPROVED")
    assert not hasattr(mi, "OBSERVED_PROTECTED_HASHES") and not hasattr(pp, "authorize_protected")
    assert not hasattr(pp, "issue_grant") and not hasattr(pp, "redeem_grant") and not hasattr(pp, "ProtectedGrant")


@pytest.mark.parametrize("case", ["missing_activation", "missing_authorization", "missing_bindings"])
def test_missing_records_fail_closed(frozen, env, tmp_path, monkeypatch, case):
    drop = {"missing_activation": pp.ACTIVATION_PATH, "missing_authorization": pp.AUTHORIZATION_PATH,
            "missing_bindings": pp.APPROVED_BINDINGS_PATH}[case]
    repo = build_repo(tmp_path / "records", frozen, env, edit=lambda r: r.__setitem__(drop, None))
    redirect(monkeypatch, tmp_path, frozen, env, repo)
    _refused("protected execution is disabled")


@pytest.mark.parametrize("edit, match", [
    (lambda r: r[pp.ACTIVATION_PATH].update(enabled=False), "not enabled"),
    (lambda r: r[pp.ACTIVATION_PATH].update(extra="x"), "missing/unexpected"),
    (lambda r: r[pp.ACTIVATION_PATH].pop("authorization_sha256"), "missing/unexpected"),
    (lambda r: r[pp.ACTIVATION_PATH].update(schema="phase5-protected-activation-v1"), "unknown schema"),
    (lambda r: r[pp.ACTIVATION_PATH].update(reviewed_commit="abc123"), "full 40-hex"),
    (lambda r: r[pp.ACTIVATION_PATH].update(authorization_sha256="0" * 64), "bound to another authorization"),
    (lambda r: r[pp.ACTIVATION_PATH].update(bindings_sha256="0" * 64), "bound to another authorization record or other bindings"),
    (lambda r: r[pp.AUTHORIZATION_PATH]["human_decisions"]["H4"].update(decision="REJECTED"), "bound to another authorization"),
])
def test_malformed_or_inconsistent_records_fail_closed(frozen, env, tmp_path, monkeypatch, edit, match):
    repo = build_repo(tmp_path / "records", frozen, env, edit=edit)
    redirect(monkeypatch, tmp_path, frozen, env, repo)
    _refused(match)


def test_invalid_json_record_fails_closed(frozen, env, tmp_path, monkeypatch):
    repo = build_repo(tmp_path / "records", frozen, env, commit_records=False)
    (repo["root"] / pp.ACTIVATION_PATH).write_text("{not json", encoding="utf-8")
    commit_all(repo["root"], "SYNTHETIC records with a broken activation")
    redirect(monkeypatch, tmp_path, frozen, env, repo)
    _refused("not valid JSON")


def test_uncommitted_records_fail_closed(frozen, env, tmp_path, monkeypatch):
    repo = build_repo(tmp_path / "records", frozen, env, commit_records=False)
    redirect(monkeypatch, tmp_path, frozen, env, repo)
    _refused("uncommitted")


def test_records_altered_in_the_working_tree_fail_closed(active):
    path = active["root"] / pp.ACTIVATION_PATH
    path.write_text(path.read_text(encoding="utf-8").replace('"enabled": true', '"enabled": true '), encoding="utf-8")
    _refused("uncommitted or modified")


def test_a_record_modified_in_a_later_commit_fails_closed(active):
    document = active["records"][pp.ACTIVATION_PATH]
    _write_json(active["root"], pp.ACTIVATION_PATH, {**document, "activated_on": "1999-01-01"})
    commit_all(active["root"], "SYNTHETIC later edit of the activation record")
    _refused("modified after it was committed")


def test_records_split_across_commits_fail_closed(frozen, env, tmp_path, monkeypatch):
    repo = build_repo(tmp_path / "records", frozen, env, commit_records=False)
    git(repo["root"], "add", pp.APPROVED_BINDINGS_PATH.as_posix())
    git(repo["root"], "commit", "-q", "-m", "SYNTHETIC bindings first")
    commit_all(repo["root"], "SYNTHETIC the rest later")
    redirect(monkeypatch, tmp_path, frozen, env, repo)
    _refused("introduced by one commit")


def test_activation_cannot_enable_a_previously_reviewed_executable(frozen, env, tmp_path, monkeypatch):
    """An earlier reviewed commit (different code) named by otherwise valid, committed records is refused."""
    root = tmp_path / "records"
    root.mkdir()
    git(root, "init", "-q")
    _copy_executable(root, mutate=lambda name, raw: raw + b"\n# SYNTHETIC earlier reviewed version\n"
                     if name == "src/experiment/phase5_inputs.py" else raw)
    earlier = commit_all(root, "SYNTHETIC earlier reviewed executable")
    _copy_executable(root)
    commit_all(root, "SYNTHETIC current executable (not reviewed)")
    records, tokens = make_records(frozen, env, earlier)
    for path, document in records.items():
        _write_json(root, path, document)
    commit_all(root, "SYNTHETIC records naming the earlier review")
    redirect(monkeypatch, tmp_path, frozen, env, {"root": root})
    _refused("source or configuration changed since the reviewed commit|differs from the reviewed commit")


def test_running_source_must_equal_the_reviewed_git_objects(frozen, env, tmp_path, monkeypatch):
    root = tmp_path / "records"
    root.mkdir()
    git(root, "init", "-q")
    _copy_executable(root, mutate=lambda name, raw: raw.replace(b"OPERATION_LOCK_WAIT_SECONDS = 5.0", b"OPERATION_LOCK_WAIT_SECONDS = 6.0")
                     if name == "src/experiment/phase5_protected.py" else raw)
    reviewed = commit_all(root, "SYNTHETIC reviewed executable that is not the running one")
    records, _ = make_records(frozen, env, reviewed)
    for path, document in records.items():
        _write_json(root, path, document)
    commit_all(root, "SYNTHETIC records")
    redirect(monkeypatch, tmp_path, frozen, env, {"root": root})
    _refused("running src/experiment/phase5_protected.py differs from the reviewed commit")


def test_self_referential_binding_is_refused(frozen, env, tmp_path, monkeypatch):
    """Records inside the very commit they name (reviewed == records commit) are refused."""
    root = tmp_path / "records"
    root.mkdir()
    git(root, "init", "-q")
    _copy_executable(root)
    base = commit_all(root, "SYNTHETIC base")
    records, _ = make_records(frozen, env, base)
    for path, document in records.items():
        _write_json(root, path, document)
    with_records = commit_all(root, "SYNTHETIC executable plus records")
    records, _ = make_records(frozen, env, with_records)
    for path, document in records.items():
        _write_json(root, path, document)
    git(root, "add", "-A")
    git(root, "commit", "-q", "--amend", "-m", "SYNTHETIC records naming their own commit")
    redirect(monkeypatch, tmp_path, frozen, env, {"root": root})
    _refused("records must be committed after|self-referential|does not exist")


def test_source_change_after_review_and_dirty_or_untracked_code_are_refused(active):
    target = active["root"] / "src/experiment/phase5_inputs.py"
    original = target.read_bytes()
    target.write_bytes(original + b"\n# SYNTHETIC uncommitted edit\n")
    _refused("tracked file src/experiment/phase5_inputs.py differs from HEAD")
    target.write_bytes(original)
    (active["root"] / "src/experiment/sitecustomize.py").write_text("# SYNTHETIC\n", encoding="utf-8")
    _refused("untracked file src/experiment/sitecustomize.py")
    (active["root"] / "src/experiment/sitecustomize.py").unlink()
    pp.verified_activation()                                           # restored: verifies again
    target.write_bytes(original + b"\n# SYNTHETIC committed edit after review\n")
    commit_all(active["root"], "SYNTHETIC source change after review")
    _refused("source or configuration changed since the reviewed commit")


def test_activation_naming_a_commit_outside_the_history_is_refused(frozen, env, tmp_path, monkeypatch):
    repo = build_repo(tmp_path / "records", frozen, env, edit=lambda r: _rebind(r, "1" * 40))
    redirect(monkeypatch, tmp_path, frozen, env, repo)
    _refused("does not exist")


def _rebind(records, commit):
    records[pp.AUTHORIZATION_PATH].update(reviewed_commit=commit)
    records[pp.AUTHORIZATION_PATH]["independent_reviewer"]["reviewed_commit"] = commit
    records[pp.ACTIVATION_PATH].update(reviewed_commit=commit, authorization_sha256=content_sha256(records[pp.AUTHORIZATION_PATH]))


def test_activation_and_authorization_naming_different_commits_is_refused(frozen, env, tmp_path, monkeypatch):
    def edit(records):
        records[pp.ACTIVATION_PATH].update(reviewed_commit="2" * 40)
    repo = build_repo(tmp_path / "records", frozen, env, edit=edit)
    redirect(monkeypatch, tmp_path, frozen, env, repo)
    _refused("name different commits")


def test_activation_is_revalidated_on_every_operation(frozen, active):
    pp.verified_activation()
    (active["root"] / pp.AUTHORIZATION_PATH).unlink()                 # removed after an earlier success
    _refused("protected execution is disabled")
    git(active["root"], "checkout", "--", pp.AUTHORIZATION_PATH.as_posix())
    pp.verified_activation()


# ------------------------------------------------------------------ operation authority: requests, not grants
@pytest.fixture
def protected_box(frozen, active):
    work = staging(active["tmp"], "grant-box")
    box = Sandbox(work)
    pp.create_protected_fixture(box, frozen)
    return box


def _in_protected_process(call):
    """Run an executor operation the way the drivers do: inside the single protected process."""
    with pp.single_protected_operation():
        return call()


def _ledger_untouched(active, before):
    assert file_sha256(active["ledger"]) == before and ledger_sealed(active["ledger"])
    assert len(verify_ledger_history(active["ledger"], authorization_root=pp.BATCH_AUTH_ROOT).rows) == 196
    assert not pp.BATCH_AUTH_ROOT.exists() or not list(pp.BATCH_AUTH_ROOT.rglob("batch_authorization.json"))


def test_the_executor_accepts_only_a_protected_request_and_derives_authority_itself(frozen, active, protected_box):
    executor = Executor(protected_box, frozen)
    tokens, before = active["tokens"], file_sha256(active["ledger"])
    batch = active["records"][pp.AUTHORIZATION_PATH]["batch_id"]
    with pytest.raises(Refused, match="not a synthetic fixture"):
        authorize_fixture(protected_box, frozen, "score")
    forged_permit = FixturePermit(protected_box.identity, frozen.identity, file_sha256(protected_box.path("fixture.json")),
                                  "score", batch, ex_module.source_identity())
    forged_authority = pp.ProtectedAuthority(*(["SYNTHETIC-forged"] * 16))
    for call in (lambda: executor.score_all(forged_permit), lambda: executor.score_all(None),
                 lambda: executor.score_all(forged_authority), lambda: executor.analyse_once(forged_permit),
                 lambda: executor.recover(forged_permit)):
        with pytest.raises(Refused, match="requires a ProtectedRequest"):
            _in_protected_process(call)
    with pytest.raises(Refused, match="single protected process"):    # a valid request outside the driver's process lock
        executor.score_all(pp.ProtectedRequest("score", tokens["score"]))
    for request, match in ((pp.ProtectedRequest("analysis", tokens["analysis"]), "cannot start score"),
                           (pp.ProtectedRequest("score", tokens["analysis"]), "cannot authorize score"),
                           (pp.ProtectedRequest("score", secrets.token_hex(32)), "does not match")):
        with pytest.raises(Refused, match=match):
            _in_protected_process(lambda: executor.score_all(request))
    with pytest.raises(Refused, match="cannot start incident"):
        _in_protected_process(lambda: executor.resolve_interrupted_writes(pp.ProtectedRequest("incident", tokens["score"])))
    assert tokens["score"] not in repr(pp.ProtectedRequest("score", tokens["score"]))      # a token is never printed
    assert executor._authority is None and not protected_box.path("state.json").exists()
    _ledger_untouched(active, before)


def test_authority_exists_only_during_its_operation(frozen, active, protected_box, monkeypatch):
    executor = Executor(protected_box, frozen)
    before = file_sha256(active["ledger"])
    _crash_before_reservation(monkeypatch)
    with pytest.raises(OSError, match="pre-exposure"):
        _in_protected_process(lambda: executor.score_all(pp.ProtectedRequest("score", active["tokens"]["score"])))
    assert executor._authority is None and executor._capabilities == {}
    _ledger_untouched(active, before)


def _crash_before_reservation(monkeypatch):
    real_hook = Executor._hook

    def fault(self, stage, unit=""):
        if stage == "before_reservation":
            raise OSError("SYNTHETIC pre-exposure crash")
        return real_hook(self, stage, unit)
    monkeypatch.setattr(Executor, "_hook", fault)


@pytest.mark.parametrize("change, match", [
    ("activation_edited_uncommitted", "uncommitted or modified"),
    ("authorization_edited_in_a_later_commit", "modified after it was committed"),
    ("executor_source_changed", "executor source changed during the protected operation"),
])
def test_authority_is_rederived_at_the_ledger_append(frozen, active, protected_box, monkeypatch, change, match):
    """Valid at the start of the operation, changed before the append: the append's own check refuses."""
    executor = Executor(protected_box, frozen)
    before = file_sha256(active["ledger"])
    real_hook, seen = Executor._hook, []

    def hook(self, stage, unit=""):
        if stage == "before_reservation" and not seen:
            seen.append(stage)
            if change == "activation_edited_uncommitted":
                path = active["root"] / pp.ACTIVATION_PATH
                path.write_text(path.read_text(encoding="utf-8").replace('"enabled": true', '"enabled": true '),
                                encoding="utf-8")
            elif change == "authorization_edited_in_a_later_commit":
                _write_json(active["root"], pp.AUTHORIZATION_PATH,
                            {**active["records"][pp.AUTHORIZATION_PATH], "issued_on": "1999-01-01"})
                commit_all(active["root"], "SYNTHETIC later edit of the batch authorization")
            else:
                monkeypatch.setattr(ex_module, "source_identity", lambda: "0" * 64)
        return real_hook(self, stage, unit)
    monkeypatch.setattr(Executor, "_hook", hook)
    with pytest.raises(Refused, match=match):
        _in_protected_process(lambda: executor.score_all(pp.ProtectedRequest("score", active["tokens"]["score"])))
    assert seen == ["before_reservation"] and executor._authority is None
    _ledger_untouched(active, before)


def test_the_executor_refuses_a_fixture_bound_to_another_authorization(frozen, active, protected_box):
    from src.experiment.phase5_inputs import atomic_json
    fixture = protected_box.json("fixture.json")
    atomic_json(protected_box.path("fixture.json"), {**fixture, "authorization_sha256": "0" * 64}, replace=True)
    with pytest.raises(Refused, match="another authorization record"):
        Executor(protected_box, frozen)
    atomic_json(protected_box.path("fixture.json"), fixture, replace=True)
    executor = Executor(protected_box, frozen)
    atomic_json(protected_box.path("fixture.json"), {**fixture, "recipe": "edited"}, replace=True)
    with pytest.raises(Refused, match="bound to another authorization or reviewed commit"):
        _in_protected_process(lambda: executor.score_all(pp.ProtectedRequest("score", active["tokens"]["score"])))


# ------------------------------------------------------------------ one protected process, one ledger writer
def test_a_second_protected_operation_is_refused_while_one_holds_the_process_lock(frozen, active, monkeypatch, tmp_path):
    """Another OS process holds the ledger lock (as a running protected operation does)."""
    ready = tmp_path / "SYNTHETIC-holder-ready"
    holder = subprocess.Popen([sys.executable, "-c", (
        "import sys, time, pathlib; sys.path.insert(0, sys.argv[1]);"
        "from src.experiment.phase5_ledger import ledger_lock;"
        "lock = ledger_lock(sys.argv[2]); lock.__enter__(); pathlib.Path(sys.argv[3]).write_text('1'); time.sleep(120)"),
        str(pp.ROOT), str(active["ledger"]), str(ready)])
    try:
        deadline = time.monotonic() + 60
        while not ready.exists():
            assert holder.poll() is None and time.monotonic() < deadline, "lock holder did not start"
            time.sleep(0.05)
        monkeypatch.setattr(pp, "OPERATION_LOCK_WAIT_SECONDS", 0.5)
        before = file_sha256(active["ledger"])
        work = staging(active["tmp"], "second")
        for call in (lambda: pp.score_all(frozen, scope=pp.PROTECTED, token=active["tokens"]["score"], workdir=work),
                     lambda: pp.analyse_once(frozen, None, None, scope=pp.PROTECTED, token=active["tokens"]["analysis"]),
                     lambda: pp.recover(frozen, None, recovery_record="reports/phase5_protected/recoveries/x.json",
                                        token=secrets.token_hex(32))):
            with pytest.raises(Refused, match="another protected operation is running"):
                call()
        assert not Path(work).exists()
        _ledger_untouched(active, before)
    finally:
        holder.kill()
        holder.wait(30)


def test_the_protected_ledger_has_no_writer_outside_a_running_operation(frozen, active, protected_box, tmp_path):
    """Direct unauthorized calls of the ledger writer, with every other condition valid."""
    from src.experiment.phase5_fixture import create_fixture
    from src.experiment.phase5_inputs import atomic_json
    assert not hasattr(ledger_module, "sealed_append") and not hasattr(ledger_module, "settle_interrupted_append")
    executor = Executor(protected_box, frozen)
    ledger, before = executor.ledger, file_sha256(active["ledger"])
    assert type(ledger) is ledger_module.ProtectedJsonlLedger
    ids = {eid: eid for eid in sorted(executor.operations)}
    fake = type("P", (), {"batch_id": active["records"][pp.AUTHORIZATION_PATH]["batch_id"]})()
    with pytest.raises(Refused, match="no protected authority"):
        ledger.append(fake, sorted(ids), ids, 196)                          # the public writer, outside an operation
    with pp.single_protected_operation():
        with pytest.raises(Refused, match="no protected authority"):
            ledger.append(fake, sorted(ids), ids, 196)                      # inside the process lock, still no authority
        with pytest.raises(Refused, match="no protected authority"):
            ledger._commit(fake, sorted(ids), ids, 196)                     # the private writer runs the same checks
    with pytest.raises(Refused, match="only under the ledger lock"):
        ledger._commit(fake, sorted(ids), ids, 196)
    # A synthetic backend aimed at the protected ledger, also through a hard link, is refused.
    box = Sandbox(staging(active["tmp"], "synthetic-alias"))
    create_fixture(box, frozen)
    box.path("ledger").mkdir(parents=True, exist_ok=True)
    os.link(active["ledger"], box.path("ledger/alias.jsonl"))
    fixture = box.json("fixture.json")
    fixture["ledger"] = {"kind": "jsonl", "path": "ledger/alias.jsonl", "authorization_root": "ledger/authorizations",
                         "base_rows": 196, "code_git_sha": "0" * 40}
    atomic_json(box.path("fixture.json"), fixture, replace=True)
    with pytest.raises(Refused, match="never a synthetic executor backend"):
        Executor(box, frozen)
    _ledger_untouched(active, before)


# ------------------------------------------------------------------ detection: what the seal does not prevent
@pytest.mark.parametrize("tamper, match", [
    ("replace_with_the_closed_196_rows", "does not extend the committed ledger"),
    ("edit_an_appended_row", "does not extend the committed ledger"),
    ("edit_a_historical_row", "closed 196-row history|not byte-identical"),
    ("append_a_foreign_row", "unauthorized history|not a PROTECTED"),
    ("rewrite_the_batch_digest", "differ from the ledger bytes its authorization recorded"),
])
def test_same_user_replacement_is_not_prevented_but_is_detected_before_the_next_operation(frozen, active, interrupted,
                                                                                         tamper, match):
    """Any same-user process can replace or rewrite the sealed ledger (the seal is no OS boundary against that).
    The audit detects it, and the next protected operation refuses before any mutation."""
    it, path = interrupted, active["ledger"]
    lines = path.read_bytes().splitlines(keepends=True)
    if tamper == "rewrite_the_batch_digest":
        record_path = pp.BATCH_AUTH_ROOT / it["batch"] / "batch_authorization.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record_path.write_text(json.dumps({**record, "ledger_sha256_after": "0" * 64}), encoding="utf-8")
    else:
        if tamper == "replace_with_the_closed_196_rows":
            data = b"".join(lines[:196])
        elif tamper == "edit_an_appended_row":
            data = b"".join(lines[:300] + [lines[300].replace(b'"machine":"', b'"machine":"X')] + lines[301:])
        elif tamper == "edit_a_historical_row":
            data = b"".join(lines[:10] + [lines[10].replace(b'"reason":"', b'"reason":"edited ')] + lines[11:])
        else:
            data = b"".join(lines) + lines[-1].replace(b'"experiment_id":"', b'"experiment_id":"FOREIGN-')
        replacement = path.with_name("SYNTHETIC-replacement.jsonl")
        replacement.write_bytes(data)
        ledger_module.seal_ledger(replacement)
        ledger_module._replace_sealed(replacement, path)        # a same-user namespace replacement: NOT prevented
        assert path.read_bytes() == data and ledger_sealed(path)
    tampered = file_sha256(path)
    with pytest.raises(Refused, match=match):
        pp.audit_real_ledger(pp.verified_activation(), frozen)
    with pytest.raises(Refused, match=match):
        pp.recover(frozen, it["box"], recovery_record=it["name"], token=it["token"])
    assert file_sha256(path) == tampered                               # refused before any mutation
    assert not (pp.BATCH_AUTH_ROOT / f"{it['batch']}-recovery-1" / "batch_authorization.json").exists()


def test_the_audit_passes_on_an_intact_ledger_and_the_script_audits_the_real_one(frozen, active, interrupted):
    report = pp.audit_real_ledger(pp.verified_activation(), frozen)
    assert report["rows"] == report["committed_rows"] == 428 and report["sealed"] and report["detection_only"]
    assert [(b["start"], b["end"]) for b in report["batches"]] == [(196, 428)]
    real = subprocess.run([sys.executable, "scripts/verify_phase5_ledger.py"], capture_output=True, text=True)
    assert real.returncode == 0, real.stderr
    out = json.loads(real.stdout)
    assert out["rows"] == 196 and out["committed_rows"] == 196 and out["sealed"] is False and out["batches"] == []
    assert out["ledger_sha256"] == pp.REAL_LEDGER_SHA256


def test_cli_takes_no_caller_supplied_reviewed_commit():
    source = Path("scripts/run_phase5_protected.py").read_text(encoding="utf-8")
    assert "--reviewed-commit" not in source and "reviewed_commit=" not in source
    import inspect
    for fn in (pp.score_all, pp.analyse_once, pp.recover, pp.preflight):
        assert "reviewed_commit" not in inspect.signature(fn).parameters


# ------------------------------------------------------------------ recovery authorization
def _ledger_rows(active):
    return len(verify_ledger_history(active["ledger"], authorization_root=pp.BATCH_AUTH_ROOT).rows)


@pytest.fixture
def interrupted(frozen, active, monkeypatch):
    """A protected batch that crashed after its durable 232-look exposure (ledger 428), and one committed
    SYNTHETIC recovery record for it. Everything after this is the window between execution and recovery."""
    chain = pp.verified_activation()
    batch = chain["authorization"]["batch_id"]
    token = secrets.token_hex(32)
    name = (pp.RECOVERY_ROOT / "recovery-1.json").as_posix()
    record = _recovery_record(chain, 1, batch, token)
    _write_json(active["root"], name, record)
    commit_all(active["root"], "SYNTHETIC recovery record")
    real_hook = Executor._hook

    def fault(self, stage, unit=""):
        if stage == "before_read":
            raise OSError("SYNTHETIC crash after exposure, before any read")
        return real_hook(self, stage, unit)
    monkeypatch.setattr(Executor, "_hook", fault)
    work = staging(active["tmp"], "interrupted")
    with pytest.raises(OSError):
        pp.score_all(frozen, scope=pp.PROTECTED, token=active["tokens"]["score"], workdir=work)
    monkeypatch.setattr(Executor, "_hook", real_hook)
    box = Sandbox(work)
    assert _ledger_rows(active) == 428 and box.json("state.json")["status"] == "FAILED_TECHNICAL"
    commit_all(active["root"], "SYNTHETIC: commit the ledger after the interrupted score-all (protocol)")
    return {"box": box, "batch": batch, "name": name, "record": record, "token": token, "chain": chain,
            "ledger_sha": file_sha256(active["ledger"]), "real_hook": real_hook}


def _assert_no_recovery_mutation(active, it):
    assert file_sha256(active["ledger"]) == it["ledger_sha"] and _ledger_rows(active) == 428
    assert ledger_sealed(active["ledger"])
    assert not (pp.BATCH_AUTH_ROOT / f"{it['batch']}-recovery-1" / "batch_authorization.json").exists()
    assert not it["box"].path(f"exposures/{it['batch']}-recovery-1.json").exists()


def _change(active, it, change):
    """Changes made between the original execution and the recovery (all SYNTHETIC, in the records repository)."""
    root, name = active["root"], it["name"]
    if change == "recovery_edited_uncommitted":
        _write_json(root, name, {**it["record"], "token_sha256": pp.token_sha256(secrets.token_hex(32))})
    elif change == "recovery_edited_in_a_later_commit":
        _write_json(root, name, {**it["record"], "approved_on": "1999-01-01"})
        commit_all(root, "SYNTHETIC later edit of the recovery record")
    elif change == "recovery_revoked":
        git(root, "rm", "-q", name)
        commit_all(root, "SYNTHETIC revocation of the recovery record")
    elif change == "authorization_edited_uncommitted":
        _write_json(root, pp.AUTHORIZATION_PATH, {**active["records"][pp.AUTHORIZATION_PATH], "issued_on": "1999-01-01"})
    elif change == "authorization_edited_in_a_later_commit":
        _write_json(root, pp.AUTHORIZATION_PATH, {**active["records"][pp.AUTHORIZATION_PATH], "issued_on": "1999-01-01"})
        commit_all(root, "SYNTHETIC later edit of the batch authorization")
    elif change == "activation_revoked":
        git(root, "rm", "-q", pp.ACTIVATION_PATH.as_posix())
        commit_all(root, "SYNTHETIC revocation of the activation")
    elif change == "source_changed_since_review":
        target = root / "src/experiment/phase5_execution.py"
        target.write_bytes(target.read_bytes() + b"\n# SYNTHETIC post-review change\n")
        commit_all(root, "SYNTHETIC source change after review")
    elif change == "recovery_for_another_batch":
        it["name"] = (pp.RECOVERY_ROOT / "recovery-other.json").as_posix()
        _write_json(root, it["name"], {**it["record"], "recovers": "SYNTHETIC-another-batch"})
        commit_all(root, "SYNTHETIC recovery record for another batch")
    elif change == "recovery_naming_another_authorization":
        it["name"] = (pp.RECOVERY_ROOT / "recovery-other.json").as_posix()
        _write_json(root, it["name"], {**it["record"], "authorization_sha256": "0" * 64})
        commit_all(root, "SYNTHETIC recovery record bound to another authorization")
    elif change == "ledger_unsealed":
        os.chmod(active["ledger"], 0o644)
    else:
        raise AssertionError(change)


CHANGES = [
    ("recovery_edited_uncommitted", "uncommitted or modified"),
    ("recovery_edited_in_a_later_commit", "modified after it was committed"),
    ("recovery_revoked", "missing or revoked"),
    ("authorization_edited_uncommitted", "uncommitted or modified"),
    ("authorization_edited_in_a_later_commit", "modified after it was committed"),
    ("activation_revoked", "protected execution is disabled"),
    ("source_changed_since_review", "source or configuration changed since the reviewed commit"),
    ("recovery_for_another_batch", "not bound to this batch"),
    ("recovery_naming_another_authorization", "not bound to this batch"),
    ("ledger_unsealed", "not sealed read-only"),
]


@pytest.mark.parametrize("change, match", CHANGES)
def test_recovery_revalidates_authority_changed_between_execution_and_recovery(frozen, active, interrupted, change, match):
    it = interrupted
    _change(active, it, change)
    with pytest.raises(Refused, match=match):
        pp.recover(frozen, it["box"], recovery_record=it["name"], token=it["token"])
    if change == "ledger_unsealed":
        ledger_module.seal_ledger(active["ledger"])
    _assert_no_recovery_mutation(active, it)


def test_recovery_refuses_a_wrong_token_and_a_record_outside_the_recovery_root(frozen, active, interrupted):
    it = interrupted
    for record, token, match in ((it["name"], secrets.token_hex(32), "recovery token does not match"),
                                 (it["name"], active["tokens"]["score"], "recovery token does not match"),
                                 ("reports/phase5_protected/elsewhere.json", it["token"], "recoveries"),
                                 ((pp.RECOVERY_ROOT / "never-committed.json").as_posix(), it["token"], "missing or revoked")):
        with pytest.raises(Refused, match=match):
            pp.recover(frozen, it["box"], recovery_record=record, token=token)
    _assert_no_recovery_mutation(active, it)


def test_a_recovery_request_outside_the_driver_or_with_another_phase_is_refused(frozen, active, interrupted):
    it = interrupted
    executor = Executor(it["box"], frozen)
    with pytest.raises(Refused, match="single protected process"):
        executor.recover(pp.ProtectedRequest("recovery", it["token"], it["name"]))
    with pytest.raises(Refused, match="closed 196-row identity"):            # after exposure, no new score operation
        _in_protected_process(lambda: Executor(it["box"], frozen).score_all(pp.ProtectedRequest("score", active["tokens"]["score"])))
    with pytest.raises(Refused, match="cannot start recovery"):
        _in_protected_process(lambda: executor.recover(pp.ProtectedRequest("score", it["token"], it["name"])))
    assert executor._authority is None
    _assert_no_recovery_mutation(active, it)


@pytest.mark.parametrize("change, match", [("recovery_edited_uncommitted", "uncommitted or modified"),
                                           ("recovery_revoked", "missing or revoked"),
                                           ("authorization_edited_in_a_later_commit", "modified after it was committed")])
def test_authority_revoked_mid_recovery_is_refused_before_the_ledger_append(frozen, active, interrupted, monkeypatch,
                                                                            change, match):
    """Valid at the start; changed after the takeover; the next boundary (the locked append) refuses."""
    it = interrupted
    seen = []

    def revoke(self, stage, unit=""):
        if stage == "after_lock_takeover" and not seen:
            seen.append(stage)
            _change(active, it, change)
        return it["real_hook"](self, stage, unit)
    monkeypatch.setattr(Executor, "_hook", revoke)
    with pytest.raises(Refused, match=match):
        pp.recover(frozen, it["box"], recovery_record=it["name"], token=it["token"])
    assert seen == ["after_lock_takeover"]
    _assert_no_recovery_mutation(active, it)


def test_protected_scope_refuses_an_unsealed_real_ledger_before_any_mutation(frozen, active):
    assert ledger_sealed(active["ledger"])
    os.chmod(active["ledger"], 0o644)
    before = file_sha256(active["ledger"])
    with pytest.raises(Refused, match="not sealed read-only"):
        pp.score_all(frozen, scope=pp.PROTECTED, token=active["tokens"]["score"], workdir=staging(active["tmp"], "unsealed"))
    assert file_sha256(active["ledger"]) == before and _ledger_rows(active) == 196
    assert not pp.BATCH_AUTH_ROOT.exists() or not any(pp.BATCH_AUTH_ROOT.iterdir())
    ledger_module.seal_ledger(active["ledger"])
