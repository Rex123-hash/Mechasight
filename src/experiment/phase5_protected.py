"""Protected execution: disabled unless a human activates it. See docs/phase5_protected_architecture.md.

The design is deliberately small:

* **Authority** comes only from committed Git records. ``verified_activation()`` re-derives it every
  time and fails closed. The three human records must be committed together, after the reviewed
  commit and never edited; no code may change since review; the running protected executable must be
  byte-identical to the reviewed commit.
* **One protected process.** The drivers (``score_all``, ``analyse_once``, ``recover``) hold the
  ledger lock, a cross-process OS lock, for the whole operation. A second protected operation, in any
  process, is refused before it does anything.
* **No grants.** A caller hands the executor a ``ProtectedRequest`` (phase, token and, for a recovery,
  the committed recovery record), which carries no authority. The executor derives a
  ``ProtectedAuthority`` itself with the full preflight (``authorize``), holds it for that one
  operation only and re-derives it at every protected boundary (``current_authority``): each read and
  the ledger append. Nothing reusable is ever handed out.
* **One ledger writer.** ``phase5_ledger.JsonlLedger.append``, which begins with the authority, budget
  and history checks.
* **Detection.** ``phase5_ledger_audit`` independently re-checks the ledger bytes against the reviewed
  history constant, the per-batch digests and the Git-committed ledger. Every protected preflight
  fails closed on a mismatch.

This module issues no authorization, activation or token.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
from contextlib import ExitStack, contextmanager
import re
import subprocess
import uuid

from src.experiment.phase5_inputs import Refused, Sandbox, require
from src.experiment.provenance import canonical_bytes, canonical_json, canonical_sha256, content_sha256, file_sha256

ROOT = Path(__file__).resolve().parents[2]
# The Git repository holding the committed human records and the reviewed commit. It is the project
# repository; tests point it at a temporary records repository (synthetic authority, never real).
GIT_ROOT = ROOT

BINDING_SCHEMA = "phase5-protected-artifact-bindings-v1"
AUTHORIZATION_SCHEMA = "phase5-protected-batch-authorization-v1"
ACTIVATION_SCHEMA = "phase5-protected-activation-v2"
REHEARSAL, PROTECTED = "REHEARSAL_DEVELOPMENT_MIRROR", "PROTECTED_COMMON_EVALUATION"
PROPOSED, APPROVED = "PROPOSED_UNAPPROVED", "APPROVED"
CELLS = ("RoboticArm|S", "RoboticArm|T", "BrushlessMotor|S", "BrushlessMotor|T")
ROLES = ("features", "spectral")
HUMAN_DECISIONS = ("H1", "H2", "H3", "H4", "H5", "H6", "H7")
TRUSTED_COMMIT = "d7ea8ac3022acd755a847b77118375fb719ef6e3"
REAL_LEDGER = Path("reports/experiment_results/test_look_ledger.jsonl")
REAL_LEDGER_SHA256 = "f75584611c18ff387bee8c06f1fcc999c3046def876ba13dea1c744fce64de5f"
PROTECTED_ROOT = Path("reports/phase5_protected")
ACTIVATION_PATH = PROTECTED_ROOT / "ACTIVATION.json"
APPROVED_BINDINGS_PATH = PROTECTED_ROOT / "artifact_bindings.json"
AUTHORIZATION_PATH = PROTECTED_ROOT / "batch_authorization_record.json"
RECORD_PATHS = (ACTIVATION_PATH, AUTHORIZATION_PATH, APPROVED_BINDINGS_PATH)
RECOVERY_ROOT = PROTECTED_ROOT / "recoveries"
BATCH_AUTH_ROOT = PROTECTED_ROOT / "batches"
RESULTS_ROOT = PROTECTED_ROOT / "results"
PHASE4_STAGING = Path("reports/phase4_protected/staging")
PHASE32_STORE = Path("reports/phase3_2/experiments")
# Paths that may change between the reviewed commit and execution: records and documents, never code.
POST_REVIEW_PATHS = ("reports/phase5_protected/", "docs/", "paper/", "defence/")
CODE_PATHS = ("src/", "scripts/", "configs/", "tests/")
# The running protected executable; each file must equal its Git object at the reviewed commit.
PROTECTED_EXECUTABLE = (
    "src/experiment/phase5_execution.py", "src/experiment/phase5_inputs.py", "src/experiment/phase5_analysis.py",
    "src/experiment/phase5_fixture.py", "src/experiment/phase5_window_index.py", "src/experiment/phase5_inference.py",
    "src/experiment/phase5_ledger.py", "src/experiment/phase5_dev_fixture.py", "src/experiment/phase5_protected.py",
    "src/experiment/phase5_input_manifest.py", "src/experiment/phase5_count_index.py",
    "src/experiment/phase5_mirror_fixture.py", "src/experiment/ledger.py", "src/experiment/ledger_history.py",
    "src/experiment/provenance.py", "src/experiment/fusion_freeze.py", "src/experiment/phase4.py",
    "src/experiment/store.py", "src/preprocessing/protected_access.py", "scripts/run_phase5_protected.py",
    "src/experiment/phase5_ledger_audit.py",
)
ACTIONS = ("score", "analysis", "recovery")
OPERATION_LOCK_WAIT_SECONDS = 5.0
RECOVERY_SCHEMA = "phase5-protected-recovery-authorization-v1"
RECOVERY_KEYS = {"schema", "batch_id", "recovers", "authorization_sha256", "approved_by", "approved_on", "token_sha256"}
ACTIVATION_KEYS = {"schema", "enabled", "activated_by", "activated_on", "reviewed_commit",
                   "authorization_sha256", "bindings_sha256"}
BINDING_KEYS = {"schema", "scope", "status", "trusted_commit", "registry_sha256", "plan_identity_sha256",
                "cells", "content_hash_policy", "approval"}
CELL_KEYS = {"machine", "regime", "cohort_identity", "artifacts"}
ARTIFACT_KEYS = {"path", "stat", "sha256"}
AUTH_KEYS = {"schema", "scope", "batch_id", "reviewed_commit", "bindings_sha256", "registry_sha256",
             "plan_identity_sha256", "planned_ids_sha256", "ledger_start_count", "ledger_end_count",
             "experiment_count", "max_total_rows", "human_decisions", "independent_reviewer",
             "scoring_token_sha256", "analysis_token_sha256", "issued_by", "issued_on"}
DISABLED = ("protected execution is disabled: it needs the committed human records (ACTIVATION.json, the approved "
            "artifact bindings and the batch authorization), added together after the independently reviewed commit")
_SHA = re.compile(r"[0-9a-f]{40}")


def _protected_paths(machine, regime):
    """The registered common-evaluation artifacts (the same files Phase 3.2/4 used)."""
    return {"features": f"data/interim/phase32_common_test/{machine}/{regime}/common_test.parquet",
            "spectral": f"data/interim/phase4/spectral/{machine}/{regime}/common_evaluation.spectral.h5"}


def _plan(package=None):
    """The frozen plan, from the package or (no package) from the frozen Git metadata itself."""
    if package is None:
        from src.experiment.phase5_input_manifest import _registry
        registry, proposed = _registry()
    else:
        registry, proposed = package.registry, package.proposed
    ids = sorted(r["experiment_id"] for r in proposed["planned_units"])
    require(len(ids) == 232 == len(set(ids)), "frozen plan does not hold 232 unique identities")
    return {"batch_id": proposed["proposed_batch_id"], "ids": ids, "ids_sha256": content_sha256(ids),
            "identity_sha256": content_sha256(proposed["identity"]), "cohorts": proposed["identity"]["cohort_hashes"],
            "registry_sha256": content_sha256(registry)}


# ---------------------------------------------------------------- committed-record authority (B1)
def _git(*args, check=True):
    result = subprocess.run(["git", "-C", str(GIT_ROOT), "-c", "core.quotepath=off", *args], capture_output=True)
    if check and result.returncode != 0:
        raise Refused(f"activation refused: git {args[0]} failed ({result.stderr.decode(errors='replace').strip()[:200]})")
    return result


@lru_cache(maxsize=4096)
def _git_object(root, commit, path):
    """Bytes of ``commit:path`` (LF form). Cached only by full commit SHA: Git objects are immutable."""
    result = subprocess.run(["git", "-C", root, "show", f"{commit}:{path}"], capture_output=True)
    return None if result.returncode != 0 else result.stdout.replace(b"\r\n", b"\n")


@lru_cache(maxsize=1024)
def _history_facts(root, head, path):
    """Commits of ``path`` reachable from ``head`` (immutable for a fixed head SHA)."""
    result = subprocess.run(["git", "-C", root, "log", "--format=%H", head, "--", path], capture_output=True, text=True)
    return tuple(result.stdout.split()) if result.returncode == 0 else None


@lru_cache(maxsize=1024)
def _is_ancestor(root, ancestor, descendant):
    return subprocess.run(["git", "-C", root, "merge-base", "--is-ancestor", ancestor, descendant],
                          capture_output=True).returncode == 0


@lru_cache(maxsize=1024)
def _changed_since(root, reviewed, head):
    result = subprocess.run(["git", "-C", root, "-c", "core.quotepath=off", "diff", "--name-only", "--no-renames",
                             reviewed, head], capture_output=True, text=True)
    return tuple(result.stdout.split("\n")) if result.returncode == 0 else None


def _relative(path):
    """A POSIX path relative to the records repository, or None for a location outside it."""
    path = Path(path)
    if path.is_absolute():
        try:
            return path.resolve().relative_to(Path(GIT_ROOT).resolve()).as_posix()
        except ValueError:
            return None
    return path.as_posix()


def _allowed_dirty():
    """Execution evidence that may differ from HEAD: the real ledger, its lock, batch records, results."""
    exact, prefixes = set(), []
    for path in (REAL_LEDGER, Path(str(REAL_LEDGER) + ".lock")):
        if _relative(path):
            exact.add(_relative(path))
    for path in (BATCH_AUTH_ROOT, RESULTS_ROOT):
        if _relative(path):
            prefixes.append(_relative(path).rstrip("/") + "/")
    return exact, tuple(prefixes)


def _tree_state():
    """(HEAD SHA, porcelain entries). Refused unless the tracked tree is clean outside execution evidence."""
    head = _git("rev-parse", "--verify", "HEAD").stdout.decode().strip()
    require(_SHA.fullmatch(head), "activation refused: the records repository has no commit")
    raw = _git("status", "--porcelain=v1", "-z", "--untracked-files=all", "--no-renames").stdout.decode("utf-8", "replace")
    exact, prefixes = _allowed_dirty()
    records = {p.as_posix() for p in RECORD_PATHS}
    for entry in filter(None, raw.split("\0")):
        code, name = entry[:2], entry[3:]
        if name in exact or name.startswith(prefixes):
            continue
        require(name not in records and not name.startswith(RECOVERY_ROOT.as_posix() + "/"),
                f"activation refused: human record {name} is uncommitted or modified ({code.strip()})")
        require(code == "??", f"activation refused: tracked file {name} differs from HEAD; the tree must be clean")
        require(not name.startswith(CODE_PATHS + (PROTECTED_ROOT.as_posix() + "/",)),
                f"activation refused: untracked file {name} in a code or record path")
    return head


def _committed_record(path, head):
    """A tracked record, byte-identical to HEAD, introduced by one commit and never modified since."""
    name = Path(path).as_posix()
    local = Path(GIT_ROOT) / name
    require(local.is_file(), DISABLED)
    committed = _git_object(str(GIT_ROOT), head, name)
    require(committed is not None, f"activation refused: {name} exists but is not committed")
    require(canonical_bytes(local) == committed, f"activation refused: {name} differs from its committed version")
    history = _history_facts(str(GIT_ROOT), head, name)
    require(history is not None and len(history) == 1,
            f"activation refused: {name} was modified after it was committed; a new record needs a new review")
    try:
        document = json.loads(committed)
    except ValueError as exc:
        raise Refused(f"activation refused: {name} is not valid JSON") from exc
    require(isinstance(document, dict), f"activation refused: {name} is malformed")
    return document, history[0]


def _validate_activation(record):
    require(set(record) == ACTIVATION_KEYS and record["schema"] == ACTIVATION_SCHEMA,
            "activation refused: ACTIVATION.json has missing/unexpected fields or an unknown schema")
    require(record["enabled"] is True and isinstance(record["activated_by"], str) and record["activated_by"].strip()
            and isinstance(record["activated_on"], str) and record["activated_on"].strip(),
            "activation refused: the activation record is not enabled by a named person with a date")
    for key in ("authorization_sha256", "bindings_sha256"):
        require(isinstance(record[key], str) and re.fullmatch(r"[0-9a-f]{64}", record[key]),
                f"activation refused: {key} malformed")
    require(isinstance(record["reviewed_commit"], str) and _SHA.fullmatch(record["reviewed_commit"]),
            "activation refused: the reviewed commit must be a full 40-hex SHA")


def protected_source_identity(reviewed_commit=None):
    """Identity of the running protected executable; with ``reviewed_commit``, checked against its Git objects."""
    hashes = {}
    for name in PROTECTED_EXECUTABLE:
        hashes[name] = canonical_sha256(ROOT / name)
        if reviewed_commit is not None:
            reviewed = _git_object(str(GIT_ROOT), reviewed_commit, name)
            require(reviewed is not None and sha256(reviewed).hexdigest() == hashes[name],
                    f"activation refused: running {name} differs from the reviewed commit")
    return content_sha256(hashes)


def verified_activation():
    """Re-derive the complete protected authority from committed records. Fails closed; never cached.

    Reads only the records repository (Git metadata and the three human records) and the running
    source files. No protected artifact is statted, hashed or opened.
    """
    for path in RECORD_PATHS:
        require((Path(GIT_ROOT) / path).is_file(), DISABLED)
    head = _tree_state()
    activation, c1 = _committed_record(ACTIVATION_PATH, head)
    authorization, c2 = _committed_record(AUTHORIZATION_PATH, head)
    bindings, c3 = _committed_record(APPROVED_BINDINGS_PATH, head)
    require(c1 == c2 == c3, "activation refused: the three human records must be introduced by one commit")
    _validate_activation(activation)
    reviewed = activation["reviewed_commit"]
    require(activation["authorization_sha256"] == content_sha256(authorization)
            and activation["bindings_sha256"] == content_sha256(bindings),
            "activation refused: ACTIVATION.json is bound to another authorization record or other bindings")
    _validate_bindings(bindings, None)
    require(bindings["scope"] == PROTECTED and bindings["status"] == APPROVED, "activation refused: bindings not approved")
    validate_authorization(authorization, None, bindings_sha256=activation["bindings_sha256"], scope=PROTECTED)
    require(authorization["reviewed_commit"] == reviewed == authorization["independent_reviewer"]["reviewed_commit"],
            "activation refused: the activation record, authorization and review name different commits")
    require(_git("cat-file", "-t", reviewed, check=False).stdout.decode().strip() == "commit",
            "activation refused: the reviewed commit does not exist in the records repository")
    require(reviewed != head and reviewed != c1 and _is_ancestor(str(GIT_ROOT), reviewed, c1)
            and _is_ancestor(str(GIT_ROOT), c1, head),
            "activation refused: the records must be committed after (and on top of) the reviewed commit")
    require(all(_git_object(str(GIT_ROOT), reviewed, p.as_posix()) is None for p in RECORD_PATHS),
            "activation refused: the reviewed commit already contains activation records (self-referential binding)")
    changed = _changed_since(str(GIT_ROOT), reviewed, head)
    ledger = _relative(REAL_LEDGER)
    require(changed is not None and all(not n or n.startswith(POST_REVIEW_PATHS) or n == ledger for n in changed),
            "activation refused: source or configuration changed since the reviewed commit")
    identity = protected_source_identity(reviewed)
    return {"head": head, "reviewed_commit": reviewed, "records_commit": c1,
            "activation": activation, "activation_sha256": content_sha256(activation),
            "authorization": authorization, "authorization_sha256": content_sha256(authorization),
            "bindings": bindings, "bindings_sha256": content_sha256(bindings), "protected_source_identity": identity}


def protected_execution_enabled():
    """Informational only (never authority): whether a verified activation exists right now."""
    try:
        verified_activation()
        return True
    except Refused:
        return False


def load_protected_records():
    """The committed approved bindings and authorization record, after full verification."""
    chain = verified_activation()
    return chain["bindings"], chain["authorization"]


def committed_recovery_record(path):
    """H4: a recovery authorization committed under reports/phase5_protected/recoveries/ after activation.

    Revocation is a later commit that deletes or edits the record: a deleted record is missing, an
    edited one has a history of more than one commit, and both are refused here on every call.
    """
    chain = verified_activation()
    name = _relative(path)
    require(name is not None and name.startswith(RECOVERY_ROOT.as_posix() + "/") and name.endswith(".json")
            and ".." not in Path(name).parts, "recovery records live under reports/phase5_protected/recoveries/")
    require((Path(GIT_ROOT) / name).is_file(), f"recovery refused: {name} is missing or revoked")
    record, commit = _committed_record(name, chain["head"])
    require(commit != chain["records_commit"] and _is_ancestor(str(GIT_ROOT), chain["records_commit"], commit),
            "recovery record must be committed after the activation records")
    return record, chain


# ---------------------------------------------------------------- artifact bindings
def protected_bindings_proposal(package):
    """Metadata-only proposal from the frozen d7ea8ac inventory. Opens and stats no protected file."""
    inventory = json.loads(subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{TRUSTED_COMMIT}:reports/phase5_freeze_candidate/integrity_inventory.json"],
        capture_output=True, check=True).stdout)["all_ignored_data_and_reports_stat_only"]
    plan = _plan(package)
    cells = {}
    for cell in CELLS:
        machine, regime = cell.split("|")
        paths = _protected_paths(machine, regime)
        require(all(p in inventory for p in paths.values()), f"{cell}: protected artifact absent from the frozen inventory")
        cells[cell] = {"machine": machine, "regime": regime, "cohort_identity": plan["cohorts"][machine],
                       "artifacts": {r: {"path": p, "stat": {k: inventory[p][k] for k in ("bytes", "mtime_ns")},
                                         "sha256": None} for r, p in paths.items()}}
    return {"schema": BINDING_SCHEMA, "scope": PROTECTED, "status": PROPOSED, "trusted_commit": TRUSTED_COMMIT,
            "registry_sha256": plan["registry_sha256"], "plan_identity_sha256": plan["identity_sha256"], "cells": cells,
            "content_hash_policy": ("H3: content hashes are computed only after the batch exposure is recorded, by "
                                    "byte hashing without deserialization; pre-exposure integrity is the stat identity"),
            "approval": None}


def rehearsal_bindings(package):
    """Bindings of the development mirror: the same structure, bound to permitted development artifacts."""
    from src.experiment.phase5_input_manifest import development_manifest
    plan = _plan(package)
    cells = {}
    for cell in CELLS:
        machine, regime = cell.split("|")
        doc = development_manifest(package, machine=machine, regime=regime).snapshot()
        cells[cell] = {"machine": machine, "regime": regime, "cohort_identity": doc["cohort_identity"],
                       "artifacts": {r: dict(doc["artifacts"][r]) for r in ROLES}}
    return {"schema": BINDING_SCHEMA, "scope": REHEARSAL, "status": APPROVED, "trusted_commit": TRUSTED_COMMIT,
            "registry_sha256": plan["registry_sha256"], "plan_identity_sha256": plan["identity_sha256"], "cells": cells,
            "content_hash_policy": "development artifacts: content hashes from the frozen registry",
            "approval": {"by": "REHEARSAL (no human approval; development data only)", "on": None}}


def _validate_bindings(document, package):
    """Structure, frozen identities and scope rules (no activation gate; no path is touched)."""
    plan = _plan(package)
    require(isinstance(document, dict) and set(document) == BINDING_KEYS, "binding record has missing/unexpected fields")
    require(document["schema"] == BINDING_SCHEMA and document["trusted_commit"] == TRUSTED_COMMIT
            and document["registry_sha256"] == plan["registry_sha256"]
            and document["plan_identity_sha256"] == plan["identity_sha256"], "binding record is not bound to the frozen package")
    require(document["scope"] in {REHEARSAL, PROTECTED}, "unknown binding scope")
    require(isinstance(document["cells"], dict) and set(document["cells"]) == set(CELLS),
            "binding record must cover exactly the four cells")
    if document["scope"] == PROTECTED:
        require(document["status"] == APPROVED and isinstance(document["approval"], dict)
                and document["approval"].get("by"), "protected bindings are not approved")
    for cell, entry in document["cells"].items():
        machine, regime = cell.split("|")
        require(isinstance(entry, dict) and set(entry) == CELL_KEYS and (entry["machine"], entry["regime"]) == (machine, regime),
                f"{cell}: binding cell malformed")
        require(isinstance(entry["artifacts"], dict) and set(entry["artifacts"]) == set(ROLES),
                f"{cell}: binding artifacts must be features and spectral")
        for role, artifact in entry["artifacts"].items():
            require(isinstance(artifact, dict) and set(artifact) == ARTIFACT_KEYS and isinstance(artifact["stat"], dict)
                    and set(artifact["stat"]) == {"bytes", "mtime_ns"}, f"{cell}/{role}: artifact binding malformed")
            if document["scope"] == PROTECTED:
                require(entry["cohort_identity"] == plan["cohorts"][machine], f"{cell}: protected cohort identity mismatch")
                require(artifact["path"] == _protected_paths(machine, regime)[role], f"{cell}/{role}: unregistered protected path")
    return content_sha256(document)


def validate_bindings(document, package):
    """Public validation. Protected bindings are only valid as the verified, committed approved record."""
    if isinstance(document, dict) and document.get("scope") == PROTECTED:
        chain = verified_activation()                 # refused before any path below is statted, hashed or opened
        require(content_sha256(document) == chain["bindings_sha256"],
                "protected bindings differ from the committed, approved bindings record")
    elif isinstance(document, dict) and document.get("scope") == REHEARSAL:
        require(document == rehearsal_bindings(package), "rehearsal bindings differ from the trusted development manifests")
    return _validate_bindings(document, package)


def check_cell_binding(box, fixture, cell, manifest):
    """Executor-side: the bound record, its hash and the read path must agree (called after exposure)."""
    entry = fixture["bindings"]
    require(isinstance(entry, dict) and set(entry) == {"file", "scope", "sha256"} and entry["scope"] in {REHEARSAL, PROTECTED},
            "binding scope not executable")
    if fixture["scope"] == PROTECTED:
        require(entry["scope"] == PROTECTED, "binding scope differs from the batch scope")
    else:
        require(entry["scope"] == REHEARSAL, "only rehearsal bindings are executable outside an authorized protected batch")
    path = box.path(entry["file"])
    require(file_sha256(path) == entry["sha256"] == fixture["files"].get(entry["file"]), "binding record changed after authorization")
    document = json.loads(path.read_text(encoding="utf-8"))
    if entry["scope"] == PROTECTED:
        require(content_sha256(document) == verified_activation()["bindings_sha256"],
                "binding record differs from the committed, approved bindings")
    bound = document["cells"][cell]["artifacts"]
    doc = manifest.snapshot()
    for role in ROLES:
        require({k: bound[role][k] for k in ("path", "sha256")} == {k: doc["artifacts"][role][k] for k in ("path", "sha256")},
                f"{cell}/{role}: read path or content identity differs from the bound artifact")
        stat = (ROOT / doc["artifacts"][role]["path"]).stat()
        require({"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns} == bound[role]["stat"],
                f"{cell}/{role}: artifact stat identity differs from the binding")


# ---------------------------------------------------------------- human batch authorization
def token_sha256(token):
    require(isinstance(token, str) and len(token) >= 32, "token missing or too short")
    return sha256(token.encode("utf-8")).hexdigest()


def authorization_template(package, bindings_sha256, reviewed_commit):
    """An UNSIGNED template for the human authorizer. Every decision is left UNRESOLVED."""
    plan = _plan(package)
    return {"schema": AUTHORIZATION_SCHEMA, "scope": PROTECTED, "batch_id": plan["batch_id"],
            "reviewed_commit": reviewed_commit, "bindings_sha256": bindings_sha256,
            "registry_sha256": plan["registry_sha256"], "plan_identity_sha256": plan["identity_sha256"],
            "planned_ids_sha256": plan["ids_sha256"], "ledger_start_count": 196, "ledger_end_count": 428,
            "experiment_count": 232, "max_total_rows": 660,
            "human_decisions": {h: {"decision": "UNRESOLVED", "by": None, "on": None} for h in HUMAN_DECISIONS},
            "independent_reviewer": {"name": None, "reviewed_commit": None, "verdict": None},
            "scoring_token_sha256": None, "analysis_token_sha256": None, "issued_by": None, "issued_on": None}


def validate_authorization(record, package, *, bindings_sha256, scope):
    plan = _plan(package)
    require(isinstance(record, dict) and set(record) == AUTH_KEYS, "authorization record has missing/unexpected fields")
    require(record["schema"] == AUTHORIZATION_SCHEMA and record["scope"] == scope, "authorization schema/scope mismatch")
    require(isinstance(record["batch_id"], str) and (record["batch_id"] == plan["batch_id"] if scope == PROTECTED
                                                     else record["batch_id"].startswith("REHEARSAL-")),
            "authorization batch identity mismatch")
    require((record["ledger_start_count"], record["ledger_end_count"], record["experiment_count"], record["max_total_rows"])
            == (196, 428, 232, 660), "authorization counts differ from the frozen plan (196 -> 428, 232 looks, cap 660)")
    require(record["registry_sha256"] == plan["registry_sha256"] and record["plan_identity_sha256"] == plan["identity_sha256"]
            and record["planned_ids_sha256"] == plan["ids_sha256"], "authorization is not bound to the frozen plan")
    require(record["bindings_sha256"] == bindings_sha256, "authorization is bound to other artifact bindings")
    decisions = record["human_decisions"]
    require(isinstance(decisions, dict) and set(decisions) == set(HUMAN_DECISIONS) and all(
        isinstance(d, dict) and d.get("decision") == "APPROVED" and d.get("by") and d.get("on") for d in decisions.values()),
        "H1-H7 must each be APPROVED by a named person with a date")
    reviewer = record["independent_reviewer"]
    require(isinstance(reviewer, dict) and reviewer.get("name") and reviewer.get("verdict") == "GREEN"
            and reviewer.get("reviewed_commit") == record["reviewed_commit"], "independent review of the reviewed commit missing")
    require(record["issued_by"] and record["issued_on"], "authorization issuer missing")
    for key in ("scoring_token_sha256", "analysis_token_sha256"):
        require(isinstance(record[key], str) and len(record[key]) == 64, f"{key} missing")
    require(record["scoring_token_sha256"] != record["analysis_token_sha256"], "scoring and analysis tokens must differ")
    return content_sha256(record)


def check_token(record, token, phase):
    key = {"score": "scoring_token_sha256", "analysis": "analysis_token_sha256"}[phase]
    other = "analysis_token_sha256" if phase == "score" else "scoring_token_sha256"
    digest = token_sha256(token)
    require(digest != record[other], f"the {'analysis' if phase == 'score' else 'scoring'} token cannot authorize {phase}")
    require(digest == record[key], f"{phase} token does not match the authorization record")
    return digest


def validate_recovery(record, authorization, token=None, *, digest=None):
    """H4: every recovery batch needs its own human authorization and token (or, on redemption, its digest)."""
    require(isinstance(record, dict) and set(record) == RECOVERY_KEYS and record["schema"] == RECOVERY_SCHEMA,
            "recovery authorization malformed")
    batch = authorization["batch_id"]
    require(record["authorization_sha256"] == content_sha256(authorization)
            and isinstance(record["batch_id"], str) and record["batch_id"].startswith(batch + "-recovery-")
            and isinstance(record["recovers"], str)
            and (record["recovers"] == batch or record["recovers"].startswith(batch + "-recovery-"))
            and record["recovers"] != record["batch_id"]
            and record["approved_by"] and record["approved_on"], "recovery authorization is not bound to this batch")
    require((token_sha256(token) if digest is None else digest) == record["token_sha256"]
            and record["token_sha256"] not in {authorization["scoring_token_sha256"], authorization["analysis_token_sha256"]},
            "recovery token does not match its own record")
    return record


# ---------------------------------------------------------------- preflight
def preflight(package, *, scope, bindings=None, authorization=None, token, phase, recovery_record=None):
    """Every pre-exposure check, in order. Opens no protected path.

    Protected scope takes its records only from ``verified_activation()``; records passed by the
    caller must equal the committed ones. A recovery takes the path of its committed record.
    """
    from src.experiment.ledger_history import ledger_lines, verify_ledger_history
    from src.experiment.phase5_ledger import look_budget
    require(phase in ACTIONS, "unknown phase")
    chain, recovery = None, None
    if scope == PROTECTED:
        chain = verified_activation()                 # refused here if anything is missing or altered
        require(bindings is None or content_sha256(bindings) == chain["bindings_sha256"],
                "supplied bindings differ from the committed, approved bindings")
        require(authorization is None or content_sha256(authorization) == chain["authorization_sha256"],
                "supplied authorization differs from the committed authorization record")
        bindings, authorization = chain["bindings"], chain["authorization"]
        if phase == "recovery":
            recovery, chain = committed_recovery_record(recovery_record)
            recovery_record = _relative(recovery_record)
    else:
        require(scope == REHEARSAL and phase != "recovery", "unknown scope / rehearsal has no recovery path")
    package.verify_memory()
    budget = look_budget(package)
    audit = None
    if scope == PROTECTED:
        from src.experiment.phase5_ledger import ledger_sealed
        require(ledger_sealed(REAL_LEDGER), "the real ledger is not sealed read-only: sealing it is a required human "
                                            "activation step (scripts/seal_phase5_ledger.py)")
        audit = audit_real_ledger(chain, package)      # detection: refused on any integrity mismatch
        require(audit["committed_rows"] == audit["rows"],
                f"the ledger has {audit['rows'] - audit['committed_rows']} rows that are not committed: commit the ledger and "
                "its batch records after every protected operation (also a crashed one) before starting the next")
        rows = audit["rows"]
    else:
        rows = len(verify_ledger_history(REAL_LEDGER, authorization_root=BATCH_AUTH_ROOT).rows)
    if phase == "score":
        require(file_sha256(REAL_LEDGER) == REAL_LEDGER_SHA256 and rows == budget["baseline_rows"],
                "real ledger is not at its closed 196-row identity")
    else:
        # After the batch exposure: the closed 196-row history must be byte-identical and the cap respected.
        prefix = sha256(b"".join(ledger_lines(REAL_LEDGER)[:budget["baseline_rows"]])).hexdigest()
        require(prefix == REAL_LEDGER_SHA256 and budget["baseline_rows"] <= rows <= budget["max_total_rows"],
                "real ledger history changed or exceeds the look cap")
    bindings_sha256 = _validate_bindings(bindings, package) if scope == PROTECTED else validate_bindings(bindings, package)
    authorization_sha256 = validate_authorization(authorization, package, bindings_sha256=bindings_sha256, scope=scope)
    if phase == "recovery":
        validate_recovery(recovery, authorization, token)
        digest = recovery["token_sha256"]
    else:
        digest = check_token(authorization, token, phase)
    return {"bindings_sha256": bindings_sha256, "authorization_sha256": authorization_sha256, "budget": budget,
            "real_ledger_rows": rows, "ledger_audit": audit, "chain": chain, "recovery": recovery,
            "recovery_path": recovery_record if phase == "recovery" else None, "token_sha256": digest}


def audit_real_ledger(chain, package):
    """The independent ledger audit with the Git anchor of the records repository (detection only)."""
    from src.experiment.phase5_ledger import look_budget
    from src.experiment.phase5_ledger_audit import audit_ledger
    budget = look_budget(package)
    name = _relative(REAL_LEDGER)
    committed = _git_object(str(GIT_ROOT), chain["head"], name) if name else None
    return audit_ledger(REAL_LEDGER, authorization_root=BATCH_AUTH_ROOT, baseline_rows=budget["baseline_rows"],
                        baseline_sha256=REAL_LEDGER_SHA256, max_total_rows=budget["max_total_rows"],
                        scope_marker=PROTECTED, committed_bytes=committed, require_committed_anchor=True)


# ---------------------------------------------------------------- operation authority (no grants)
@dataclass(frozen=True)
class ProtectedRequest:
    """What a caller hands a protected executor operation. It carries NO authority.

    The executor derives the authority from it itself (``authorize``), at the start of the operation.
    """

    action: str
    token: str = field(repr=False)
    recovery_record: str | None = None


@dataclass(frozen=True)
class ProtectedAuthority:
    """The authority of ONE running protected operation.

    It is built only by ``authorize`` inside the executor's operation, held as ``executor._authority``,
    re-derived at every boundary and cleared when the operation ends. It is never returned to a caller.
    The fields mirror a FixturePermit, plus the verified chain it was derived under.
    """

    sandbox_identity: str
    package_identity: str
    fixture_hash: str
    action: str
    batch_id: str
    code_identity: str
    recovery_from: str | None
    executor_identity: str
    token_sha256: str
    authorization_sha256: str
    activation_sha256: str
    bindings_sha256: str
    reviewed_commit: str
    protected_source_identity: str
    recovery_sha256: str | None
    recovery_path: str | None


def _require_single_operation():
    from src.experiment.phase5_ledger import ledger_lock_held
    require(ledger_lock_held(REAL_LEDGER), "protected operations run only inside the single protected process: "
                                           "the driver must hold the ledger lock for the whole operation")


def authorize(executor, request, action):
    """Executor-side, at the start of one protected operation: the full preflight, then the operation's authority."""
    from src.experiment.phase5_execution import Executor, source_identity
    require(type(executor) is Executor and executor.scope == PROTECTED, "protected authority is derived by a protected executor only")
    require(type(request) is ProtectedRequest,
            "protected scope requires a ProtectedRequest (phase and token); synthetic permits and other objects are refused")
    require(action in ACTIONS and request.action == action, f"a {request.action} request cannot start {action}")
    _require_single_operation()
    checks = preflight(executor.package, scope=PROTECTED, token=request.token, phase=action,
                       recovery_record=request.recovery_record)
    chain, box = checks["chain"], executor.sandbox
    fixture = box.json("fixture.json")
    require(fixture == executor.fixture and fixture["scope"] == PROTECTED
            and fixture["authorization_sha256"] == chain["authorization_sha256"]
            and fixture["ledger"]["code_git_sha"] == chain["reviewed_commit"],
            "the executor's batch fixture is bound to another authorization or reviewed commit")
    recovery = checks["recovery"]
    batch = {"score": chain["authorization"]["batch_id"],
             "analysis": f"{chain['authorization']['batch_id']}-analysis-{uuid.uuid4().hex[:8]}",
             "recovery": recovery["batch_id"] if recovery else None}[action]
    return ProtectedAuthority(box.identity, executor.package.identity, file_sha256(box.path("fixture.json")), action, batch,
                              source_identity(), recovery["recovers"] if recovery else None, executor._identity,
                              checks["token_sha256"], chain["authorization_sha256"], chain["activation_sha256"],
                              chain["bindings_sha256"], chain["reviewed_commit"], chain["protected_source_identity"],
                              content_sha256(recovery) if recovery else None, checks["recovery_path"])


def _chain_matches(authority, chain):
    return ((chain["authorization_sha256"], chain["activation_sha256"], chain["bindings_sha256"], chain["reviewed_commit"],
             chain["protected_source_identity"])
            == (authority.authorization_sha256, authority.activation_sha256, authority.bindings_sha256,
                authority.reviewed_commit, authority.protected_source_identity))


def _verify_recovery_authority(authority, chain):
    """Re-read the committed recovery record now; it must still be the one the operation started under.

    Refuses a revoked (deleted), edited, uncommitted or replaced record, a record for another batch or
    another interrupted batch, and a token that is not that record's token.
    """
    require(authority.action == "recovery" and authority.recovery_path and authority.recovery_sha256
            and authority.recovery_from, "recovery authority without its recovery record")
    record, recovery_chain = committed_recovery_record(authority.recovery_path)
    require(_chain_matches(authority, recovery_chain), "the activation chain changed during the recovery")
    require(content_sha256(record) == authority.recovery_sha256, "the recovery authorization record changed during the recovery")
    validate_recovery(record, chain["authorization"], digest=authority.token_sha256)
    require(record["batch_id"] == authority.batch_id and record["recovers"] == authority.recovery_from,
            "recovery batch identity mismatch")
    return record


def current_authority(executor, *, actions):
    """The running operation's authority, re-derived now against the committed records. Every protected boundary."""
    from src.experiment.phase5_execution import Executor, source_identity
    require(type(executor) is Executor and executor.scope == PROTECTED, "protected authority requires a protected executor")
    authority = executor._authority
    require(type(authority) is ProtectedAuthority and authority.action in actions,
            "no protected authority: protected boundaries are reachable only inside a running protected operation")
    _require_single_operation()
    require(authority.code_identity == source_identity(), "executor source changed during the protected operation")
    chain = verified_activation()
    require(_chain_matches(authority, chain), "the activation chain changed during the protected operation")
    record = chain["authorization"]
    if authority.action == "recovery":
        _verify_recovery_authority(authority, chain)
    else:
        key = {"score": "scoring_token_sha256", "analysis": "analysis_token_sha256"}[authority.action]
        require(authority.token_sha256 == record[key], "the operation's token does not match the phase token")
        require(authority.batch_id == record["batch_id"] if authority.action == "score"
                else authority.batch_id.startswith(record["batch_id"] + "-analysis-"), "batch identity mismatch")
    return authority, chain


def verify_reader_capability(capability, phase):
    """Every protected I/O boundary: the running operation's authority, a fresh chain and durable exposure.

    ``capability`` is ``(executor, access)``. Labels (``phase == "analysis"``) need an analysis operation
    and an analysis capability, which the executor issues only after the completion gate and the durable claim.
    """
    from src.experiment.phase5_execution import Executor
    from src.experiment.phase5_inputs import Access
    require(isinstance(capability, tuple) and len(capability) == 2, "protected reads require an executor capability")
    executor, access = capability
    require(type(access) is Access, "protected reads require an executor-issued read capability")
    require(phase in {"score", "analysis"}, "unknown read phase")
    authority, chain = current_authority(executor, actions=ACTIONS)
    if phase == "analysis":
        require(authority.action == "analysis" and access.phase == "analysis",
                "labels and analysis-only inputs are never readable while scoring: an analysis operation and capability are required")
    elif authority.action == "analysis":
        require(access.phase == "analysis", "an analysis batch reads only under its analysis capability")
    else:
        require(access.phase == "score" and access.batch_id == authority.batch_id,
                "the scoring capability is bound to its own batch")
    Executor._verify_capability(executor, access, phase)       # issued, live, durably exposed (and gated for analysis)
    return chain


def require_write_authority(executor, permit):
    """The real-ledger append (under the ledger lock) needs the running score/recovery authority of this batch."""
    authority, _ = current_authority(executor, actions=("score", "recovery"))
    require(authority is permit and authority.batch_id == permit.batch_id, "ledger append for another operation or batch")
    return authority


# ---------------------------------------------------------------- sealed stores and labels (after exposure)
def _phase4_manifest():
    return json.loads((PHASE4_STAGING / "score_manifest.json").read_text(encoding="utf-8"))


def _phase32_bundle_hash(eid):
    return json.loads((PHASE32_STORE / eid / "bundle_manifest.json").read_text(encoding="utf-8"))["files"]["scores.parquet"]


def read_sealed_comparator(executor, access, eid, phase, *, expected_sha256):
    """Sealed historical tables through their sanctioned, scope-guarded readers (called after exposure).

    L0 (Phase 4, label-free, scoring): ScoreStaging.read_scores, hash-checked against the Phase-4 score
    manifest. Reused L1 (Phase 3.2, analysis-only): ResultStore.read_scores, projected to score columns;
    labels in that table are dropped and never returned.
    """
    from src.experiment.phase5_inputs import IDENTITY_COLUMNS, SCORE_COLUMNS
    verify_reader_capability((executor, access), phase)
    if phase == "score":
        from src.experiment.phase4 import ScoreStaging
        entry = _phase4_manifest()["tables"][eid]
        require(canonical_sha256(PHASE4_STAGING / "scores" / entry["file"]) == entry["sha256"] == expected_sha256,
                f"sealed Phase-4 table hash mismatch for {eid}")
        return ScoreStaging(PHASE4_STAGING).read_scores(eid).loc[:, list(SCORE_COLUMNS)]
    require(phase == "analysis", "unknown comparator phase")
    from src.experiment.store import ResultStore
    require(_phase32_bundle_hash(eid) == expected_sha256, f"sealed Phase-3.2 bundle hash mismatch for {eid}")
    frame = ResultStore(PHASE32_STORE).read_scores(eid)
    if "window_count" not in frame:
        from src.experiment.phase5_input_manifest import protected_manifest, read_features
        unit = next(u for u in executor.package.registry["units"] if u["L1"]["experiment_id"] == eid)
        counts = read_features(protected_manifest(executor.package, machine=unit["machine"], regime=unit["regime"]),
                               capability=(executor, access))
        frame = frame.assign(window_count=frame["segment_id"].astype(str).map(counts.set_index("segment_id")["window_count"]))
    frame = frame.assign(**{c: frame[c].astype(str) for c in IDENTITY_COLUMNS})
    return frame.loc[:, list(SCORE_COLUMNS)]


def protected_label_frame(executor, access, machine):
    """H5: labels by column projection of the bound common-test artifact, mapped to the analysis vocabulary."""
    from src.experiment.phase5_fixture import ANOMALY, NORMAL
    from src.experiment.phase5_input_manifest import protected_manifest, read_protected_labels
    verify_reader_capability((executor, access), "analysis")
    frame = read_protected_labels(protected_manifest(executor.package, machine=machine, regime="S"),
                                  capability=(executor, access))
    labels = frame["anomaly_label"].astype(str)
    require(not labels.eq("").any(), "protected labels missing")
    out = frame.loc[:, ["segment_id", "parent_recording", "domain"]].astype(str)
    out["fake_label"] = [NORMAL if v == "normal" else ANOMALY for v in labels]   # column name kept by the frozen analysis
    return out


def create_protected_fixture(box, package):
    """A3: the protected batch's executor fixture, from the verified records. Metadata only."""
    from src.experiment.phase5_input_manifest import protected_manifest
    from src.experiment.phase5_inputs import IDENTITY_COLUMNS, PROTECTED_SCHEMA, atomic_json
    from src.experiment.phase5_ledger import look_budget
    chain = verified_activation()
    fixture = {"scope": PROTECTED, "schema": PROTECTED_SCHEMA, "recipe": "phase5-protected-common-evaluation-v1",
               "data_origin": "PROTECTED common-evaluation cohort; approved artifact bindings; authorized batch",
               "files": {}, "cells": {}, "cohorts": {}, "historical": {}, "labels": {}, "package_identity": package.identity,
               "real_model_inference": True, "authorization_sha256": chain["authorization_sha256"]}
    for machine in ("RoboticArm", "BrushlessMotor"):
        cohorts = []
        for regime in ("S", "T"):
            manifest = protected_manifest(package, machine=machine, regime=regime)
            fixture["cells"][f"{machine}|{regime}"] = {"input_manifest_identity": manifest.identity}
            cohorts.append(sorted(tuple(r[c] for c in IDENTITY_COLUMNS) for r in manifest.snapshot()["segments"]))
        require(cohorts[0] == cohorts[1], f"{machine}: S/T protected cohorts differ")
        fixture["cohorts"][machine] = [dict(zip(IDENTITY_COLUMNS, r)) for r in cohorts[0]]
        fixture["labels"][machine] = "H5:column-projection"
    phase4 = _phase4_manifest()["tables"]
    for unit in package.registry["units"]:
        reused = unit["L1"]["source"] != "new_fit_phase5"
        for eid in list(unit["L0"].values()) + ([unit["L1"]["experiment_id"]] if reused else []):
            name = f"historical/{eid}.parquet"            # logical name; bytes stay in their sealed stores
            fixture["historical"][eid] = name
            fixture["files"][name] = phase4[eid]["sha256"] if eid in phase4 else _phase32_bundle_hash(eid)
    box.path("bindings.json").write_text(canonical_json(chain["bindings"]), encoding="utf-8")
    digest = file_sha256(box.path("bindings.json"))
    fixture["files"]["bindings.json"] = digest
    fixture["bindings"] = {"file": "bindings.json", "scope": PROTECTED, "sha256": digest}
    fixture["ledger"] = {"kind": "protected_jsonl", "base_rows": look_budget(package)["baseline_rows"],
                         "authorization_root": str(BATCH_AUTH_ROOT), "code_git_sha": chain["reviewed_commit"]}
    atomic_json(box.path("fixture.json"), fixture)
    return fixture


def _archive(box, batch_id, names):
    """Copy the staging evidence into the results directory (no protected input bytes)."""
    import shutil
    target = RESULTS_ROOT / batch_id
    target.mkdir(parents=True, exist_ok=True)
    for name in names:
        source = box.path(name)
        if source.is_dir():
            shutil.copytree(source, target / name, dirs_exist_ok=True)
        elif source.exists():
            (target / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target / name)
    return target


# ---------------------------------------------------------------- drivers
@contextmanager
def single_protected_operation():
    """The one protected process: hold the ledger lock (a cross-process OS lock) for the whole operation."""
    from src.experiment.phase5_ledger import ledger_lock
    stack = ExitStack()
    try:
        stack.enter_context(ledger_lock(REAL_LEDGER, timeout=OPERATION_LOCK_WAIT_SECONDS))
    except Refused as exc:
        raise Refused("another protected operation is running (it holds the ledger lock); refused before any mutation") from exc
    with stack:
        yield


def _write_audit(batch_id, chain, package, label):
    from src.experiment.phase5_inputs import atomic_json
    target = RESULTS_ROOT / batch_id
    target.mkdir(parents=True, exist_ok=True)
    atomic_json(target / f"ledger_audit.{label}.json", audit_real_ledger(chain, package), replace=True)


def score_all(package, *, scope, bindings=None, authorization=None, token, workdir=None):
    """Score-all. PROTECTED needs the verified activation; REHEARSAL runs the identical chain on the mirror."""
    from src.experiment.phase5_execution import Executor, authorize_fixture
    from src.experiment.phase5_inference import FrozenL1Adapter
    from src.experiment.phase5_inputs import atomic_json
    from src.experiment.phase5_mirror_fixture import create_mirror_fixture
    if scope == PROTECTED:
        import tempfile
        from src.preprocessing.protected_access import authorized_protected_read
        verified_activation()                       # disabled: refused before any lock, file or protected stat
        with single_protected_operation():
            checks = preflight(package, scope=scope, bindings=bindings, authorization=authorization, token=token, phase="score")
            batch = checks["chain"]["authorization"]["batch_id"]
            box = Sandbox(workdir or tempfile.mkdtemp(prefix="mechasight-PROTECTED-staging-"))
            create_protected_fixture(box, package)
            executor = Executor(box, package, predictor=FrozenL1Adapter(package))
            try:
                with authorized_protected_read(authority=batch, purpose="phase5.7 score-all"):
                    state = executor.score_all(ProtectedRequest("score", token))
            finally:
                atomic_json(box.path("h3_observed_hashes.json"), dict(sorted(executor._protected_hashes.items())), replace=True)
            _archive(box, batch, ["fixture.json", "bindings.json", "state.json", "attempts.jsonl",
                                  "reservations", "exposures", "tables", "h3_observed_hashes.json"])
            _write_audit(batch, verified_activation(), package, "after-score")
        return box, executor, state, checks
    checks = preflight(package, scope=scope, bindings=bindings, authorization=authorization, token=token, phase="score")
    box = Sandbox(workdir)
    fixture = create_mirror_fixture(box, package)
    box.path("bindings.json").write_text(canonical_json(bindings), encoding="utf-8")
    digest = file_sha256(box.path("bindings.json"))
    fixture["files"]["bindings.json"] = digest
    fixture["bindings"] = {"file": "bindings.json", "scope": REHEARSAL, "sha256": digest}
    box.path("authorization.json").write_text(canonical_json(authorization), encoding="utf-8")
    fixture["files"]["authorization.json"] = file_sha256(box.path("authorization.json"))
    atomic_json(box.path("fixture.json"), fixture, replace=True)
    executor = Executor(box, package, predictor=FrozenL1Adapter(package))
    state = executor.score_all(authorize_fixture(box, package, "score"))
    return box, executor, state, checks


def analyse_once(package, box, executor, *, scope, bindings=None, authorization=None, token):
    """Analyse-once under the separate analysis token; labels only after the completion gate."""
    if scope == PROTECTED:
        from src.experiment.phase5_execution import Executor
        from src.experiment.phase5_inference import FrozenL1Adapter
        from src.preprocessing.protected_access import authorized_protected_read
        verified_activation()
        with single_protected_operation():
            checks = preflight(package, scope=scope, bindings=bindings, authorization=authorization, token=token,
                               phase="analysis")
            batch = checks["chain"]["authorization"]["batch_id"]
            if executor is None:
                executor = Executor(box, package, predictor=FrozenL1Adapter(package))
            require(executor.sandbox is box, "analysis must run on the scoring batch's staging sandbox")
            with authorized_protected_read(authority=batch, purpose="phase5.8 analyse-once"):
                report = executor.analyse_once(ProtectedRequest("analysis", token))
            _archive(box, batch, ["analysis.claim.json", "analysis.json", "state.json", "attempts.jsonl"])
            _write_audit(batch, verified_activation(), package, "at-analysis")
        return report
    preflight(package, scope=scope, bindings=bindings, authorization=authorization, token=token, phase="analysis")
    from src.experiment.phase5_execution import authorize_fixture
    require(executor.sandbox is box and file_sha256(box.path("authorization.json")) == executor.fixture["files"]["authorization.json"],
            "analysis must use the scoring batch's authorization record")
    return executor.analyse_once(authorize_fixture(box, package, "analysis"))


def recover(package, box, *, recovery_record, token):
    """H4: resume an interrupted protected batch under its own committed recovery record (new -r<k> looks)."""
    from src.experiment.phase5_execution import Executor
    from src.experiment.phase5_inference import FrozenL1Adapter
    from src.preprocessing.protected_access import authorized_protected_read
    verified_activation()
    with single_protected_operation():
        checks = preflight(package, scope=PROTECTED, token=token, phase="recovery", recovery_record=recovery_record)
        batch = checks["chain"]["authorization"]["batch_id"]
        executor = Executor(box, package, predictor=FrozenL1Adapter(package))
        with authorized_protected_read(authority=checks["recovery"]["batch_id"], purpose="phase5.7 recovery"):
            state = executor.recover(ProtectedRequest("recovery", token, recovery_record))
        _archive(box, batch, ["state.json", "attempts.jsonl", "reservations", "exposures", "recoveries", "tables"])
        _write_audit(batch, verified_activation(), package, f"after-{checks['recovery']['batch_id']}")
    return executor, state


def rehearsal_authorization(package, bindings_sha256, scoring_token, analysis_token):
    """A REHEARSAL-scope authorization with synthetic tokens (never valid for protected scope)."""
    record = authorization_template(package, bindings_sha256, "REHEARSAL")
    record.update(scope=REHEARSAL, batch_id="REHEARSAL-" + bindings_sha256[:16],
                  human_decisions={h: {"decision": "APPROVED", "by": "REHEARSAL (synthetic)", "on": date.today().isoformat()}
                                   for h in HUMAN_DECISIONS},
                  independent_reviewer={"name": "REHEARSAL (synthetic)", "reviewed_commit": "REHEARSAL", "verdict": "GREEN"},
                  scoring_token_sha256=token_sha256(scoring_token), analysis_token_sha256=token_sha256(analysis_token),
                  issued_by="REHEARSAL (synthetic)", issued_on=date.today().isoformat())
    return record
