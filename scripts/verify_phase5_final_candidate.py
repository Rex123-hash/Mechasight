"""Phase 5.6C final executable freeze candidate: write or verify; never protected execution.

``--write-candidate`` binds sources, the frozen scientific package, the evaluation plan
and analysis contract, the development-mirror evidence and actual JUnit records into
``reports/phase5_final_candidate/``. ``--verify-candidate SHA`` checks those bytes
against an externally supplied commit. Passing that check is a self-consistency
check, not an independent review or a human authorization.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment.phase5_execution import (
    CANDIDATE_SOURCES, EXECUTOR_SOURCES, PACKAGE, TRUSTED_COMMIT, TRUSTED_MANIFEST, FrozenPackage, source_identity,
)
from src.experiment.phase5_inputs import require
from src.experiment.provenance import canonical_bytes, canonical_sha256, content_sha256, file_sha256

STAGE = "5.6C-r1 (look-cap ordering correction)"
START = "6411322a72145c5fe63280f758379523be9cb3ec"
STATUS = "FINAL EXECUTABLE FREEZE CANDIDATE — UNREVIEWED; NOT AUTHORIZED FOR REAL PROTECTED EVALUATION."
OUTPUT = Path("reports/phase5_final_candidate")
REAL_LEDGER = Path("reports/experiment_results/test_look_ledger.jsonl")
REAL_LEDGER_SHA256 = "f75584611c18ff387bee8c06f1fcc999c3046def876ba13dea1c744fce64de5f"
NEW_SOURCES = (
    # 5.6B trusted manifests and count-derived index
    "src/experiment/phase5_input_manifest.py", "src/experiment/phase5_count_index.py",
    "scripts/verify_phase5_count_index.py", "tests/test_phase5_count_index.py", "docs/phase5_6b_count_derived_index.md",
    "docs/phase5_6_protected_evaluation_readiness.md",
    # 5.6C protected-style adapter, mirror, batch interface, final candidate
    "src/experiment/phase5_mirror_fixture.py", "scripts/run_phase5_mirror.py", "tests/test_phase5_mirror_adapter.py",
    "scripts/verify_phase5_final_candidate.py", "docs/phase5_6c_final_candidate.md",
)
SOURCES = tuple(sorted(set(CANDIDATE_SOURCES + NEW_SOURCES)))
EVIDENCE = {"mirror_evidence.json": Path("reports/phase5_6c_mirror/evidence.json"),
            "count_index_evidence.json": Path("reports/phase5_6b_candidate/evidence.json")}
FILES = ("mirror_evidence.json", "count_index_evidence.json", "focused_tests.xml", "regression_tests.xml",
         "authorization_checklist.json", "cap_precheck_evidence.json", "integrity_report.json")
PROTECTED_STAT_ROOTS = ("data/interim/phase4/spectral/", "data/interim/phase32_common_test/",
                        "reports/phase4_protected/", "reports/phase3_4_results/")
REGRESSION_SCOPE = {
    "run": ["tests/test_phase5_executor.py", "tests/test_phase5_integration_corrections.py",
            "tests/test_phase5_integration.py (all except the 4 test_B_ frozen-adapter tests)"],
    "not_rerun_after_correction": {
        "tests/test_phase5_integration.py test_B_* (4)": "frozen-adapter score path; unchanged by the ledger correction",
        "tests/test_phase5_synthetic_statistics.py": "statistics unchanged",
        "tests/test_phase5_freeze_candidate.py": "scientific freeze metadata unchanged",
        "development-mirror model run": "score path unchanged; retained evidence from commit 49069ee (24 L1 + 208 L2, max diff 0.0)"},
    "never_run": "Phase-3.4 protected-score replay and conftest content snapshot (they read protected score bytes)"}


def cap_precheck_evidence():
    """Re-run the 660 -> 892 counterexample on a temporary synthetic JSONL ledger (stub predictor)."""
    from src.experiment.phase5_execution import Executor, authorize_fixture
    from src.experiment.phase5_fixture import create_fixture
    from src.experiment.phase5_inputs import Refused, Sandbox, atomic_json
    from src.experiment.phase5_ledger import look_budget, seed_jsonl_ledger
    package = FrozenPackage.load()
    box = Sandbox(tempfile.mkdtemp(prefix="mechasight-SYNTHETIC-5-6c-cap-"))
    create_fixture(box, package)
    fixture = box.json("fixture.json")
    fixture["ledger"] = {**seed_jsonl_ledger(box), "code_git_sha": "0" * 40}
    atomic_json(box.path("fixture.json"), fixture, replace=True)
    executor = Executor(box, package)

    def crash(stage, unit):
        if stage == "after_reservation":
            raise OSError("SYNTHETIC crash after exposure")

    executor.fault = crash
    for action in ("score", "recovery"):
        try:
            if action == "score":
                executor.score_all(authorize_fixture(box, package, "score"))
            else:
                executor.recover(authorize_fixture(box, package, "recovery", recovery_from=box.json("state.json")["batch_id"]))
        except OSError:
            pass
    executor.fault = None
    ledger = box.path("ledger/test_look_ledger.jsonl")
    before = {"rows": len(ledger.read_bytes().splitlines()), "sha256": file_sha256(ledger),
              "authorizations": len(list(box.path("ledger/authorizations").iterdir())),
              "reservations": len(list(box.path("reservations").iterdir())),
              "exposures": len(list(box.path("exposures").iterdir())), "reads": len(box.events)}
    permit = authorize_fixture(box, package, "recovery", recovery_from=box.json("state.json")["batch_id"])
    try:
        executor.recover(permit)
        refusal = None
    except Refused as exc:
        refusal = str(exc)
    after = {"rows": len(ledger.read_bytes().splitlines()), "sha256": file_sha256(ledger),
             "authorizations": len(list(box.path("ledger/authorizations").iterdir())),
             "reservations": len(list(box.path("reservations").iterdir())),
             "exposures": len(list(box.path("exposures").iterdir())), "reads": len(box.events)}
    require(refusal and "look budget exceeded before any mutation" in refusal and before == after
            and before["rows"] == 660, "cap counterexample not refused before mutation")
    return {"scope": "SYNTHETIC temporary JSONL ledger; stub predictor; no real data", "budget": look_budget(package),
            "cap_semantics": "maximum TOTAL ledger length 660 (= 428 + 232, proposal H4, not authorized) = 464 looks above the 196 baseline",
            "counterexample": "ledger at 660 rows, recovery batch of 232 pending looks",
            "previous_behaviour": "appended to 892 rows before validate_batches refused reads",
            "refusal": refusal, "before": before, "after": after,
            "rejected_batch_authorization_published": box.path(f"ledger/authorizations/{permit.batch_id}").exists(),
            "rejected_batch_state": box.json("state.json")["status"]}


def integrity_report(package):
    """Stat/metadata and git only; no protected bytes are read."""
    inventory = json.loads(subprocess.run(["git", "show", f"{TRUSTED_COMMIT}:reports/phase5_freeze_candidate/integrity_inventory.json"],
                                          capture_output=True, check=True).stdout)["all_ignored_data_and_reports_stat_only"]
    checked, changed = 0, []
    for name, stat in inventory.items():
        if name.startswith(PROTECTED_STAT_ROOTS) and isinstance(stat, dict):
            checked += 1
            actual = Path(name).stat() if Path(name).exists() else None
            if actual is None or {"bytes": actual.st_size, "mtime_ns": actual.st_mtime_ns} != {k: stat[k] for k in ("bytes", "mtime_ns")}:
                changed.append(name)
    new = [p.as_posix() for root in PROTECTED_STAT_ROOTS[:2] for p in Path(root).rglob("*")
           if p.is_file() and p.as_posix() not in inventory]
    tracked = subprocess.run(["git", "status", "--porcelain", "--", "reports/phase3_4_results", "reports/phase4_protected",
                              "reports/experiment_results", "reports/phase5_freeze_candidate", "data"],
                             capture_output=True, text=True, check=True).stdout.splitlines()
    return {"real_ledger": {"rows": len(REAL_LEDGER.read_bytes().splitlines()), "sha256": file_sha256(REAL_LEDGER)},
            "frozen_package_identity": package.identity, "frozen_package_verified": True,
            "protected_stat_snapshot": {"entries": checked, "changed_or_missing": changed, "new_files": new,
                                        "method": "size+mtime vs d7ea8ac inventory; no protected bytes read"},
            "tracked_frozen_and_protected_git_changes": tracked,
            "protected_inference_or_outcome_analysis": "none",
            "first_defence_directory": "not read or modified by this tooling"}
SCIENTIFIC_FILES = ("reports/phase5_freeze_candidate/scientific_registry.json",
                    "reports/phase5_freeze_candidate/proposed_evaluation_manifest.json",
                    "reports/phase5_freeze_candidate/scientific_contract.json",
                    "reports/phase5_freeze_candidate/candidate_manifest.json",
                    "configs/phase5/fusion_design_v2.json",
                    "reports/phase2_2/protocol_b_few_shot_adaptation_v3.csv")
HUMAN_DECISIONS = {
    "H1": "approve the count-derived label-free index method for the protected cohorts",
    "H2": "confirm index construction is covered by the 232-look scoring batch, after its append and exposure record",
    "H3": "approve post-exposure byte hashing of protected caches/artifacts against declared Phase-4 hashes",
    "H4": "retry policy: a new authorization per recovery batch; proposed hard cap 660 appended looks",
    "H5": "analysis label source: column projection of common-test feature artifacts under a separate analysis token",
    "H6": "name the independent reviewer and the reviewed implementation SHA the authorization binds",
    "H7": "acknowledge the reversed Phase-4 materialisation order (exposure first, then reads)",
}
STILL_DISABLED = ("execute_real_protected", "read_protected_inputs", "build_protected_count_index",
                  "build_protected_window_index", "append_protected_batch", "PROTECTED input schema")


def _junit(path):
    require(path.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve()), "test report must be temporary")
    tree = ET.fromstring(path.read_bytes())
    counts = {k: sum(int(s.attrib.get(k, 0)) for s in tree.iter("testsuite")) for k in ("tests", "failures", "errors", "skipped")}
    require(counts["tests"] and not counts["failures"] and not counts["errors"], "test evidence has failures/errors")
    for case in tree.iter("testcase"):                      # machine-local timings/paths are not evidence
        case.attrib.pop("time", None)
    for suite in tree.iter("testsuite"):
        for key in ("time", "timestamp", "hostname"):
            suite.attrib.pop(key, None)
    raw = ET.tostring(tree, encoding="utf-8", xml_declaration=False).replace(b"\r", b"").replace(b"\n", b" ")
    return raw, counts


def scientific_bindings(package):
    detectors = package.registry["detectors"]
    return {
        "trusted_package_commit": TRUSTED_COMMIT, "trusted_package_manifest_sha256": TRUSTED_MANIFEST,
        "scientific_package_identity": package.identity,
        "registry_content_sha256": content_sha256(package.registry),
        "design_v2_sha256": package.registry["design_sha256"],
        "evaluation_plan_identity": package.proposed["identity"],
        "planned_units": len(package.proposed["planned_units"]),
        "scientific_files_canonical_sha256": {p: canonical_sha256(p) for p in SCIENTIFIC_FILES},
        "models_and_standardizers": {e: {"model_manifest_sha256": d["model_manifest"]["model_manifest_sha256"],
                                         "files": d["model_manifest"]["files"]}
                                     for e, d in sorted(detectors.items()) if d.get("model_path")},
        "bundle_manifests_sha256": content_sha256({e: d["bundle_manifest_sha256"] for e, d in detectors.items()}),
        "thresholds_sha256": content_sha256({e: d["threshold"] for e, d in detectors.items()}),
        "calibrations_sha256": content_sha256(package.registry["calibrations"]),
        "cohort_contracts_sha256": content_sha256({e: d.get("cohort_contracts") for e, d in detectors.items()}),
        "protected_cohort_hashes": package.proposed["identity"]["cohort_hashes"],
    }


def authorization_checklist():
    return {"status": "OPEN — nothing below is approved, decided or issued by this candidate",
            "human_decisions": {k: {"decision": v, "state": "UNRESOLVED"} for k, v in HUMAN_DECISIONS.items()},
            "required_before_any_protected_read": [
                "independent review of this candidate at a named commit (not this self-check)",
                "H1-H7 recorded by a human with the reviewer and reviewed SHA",
                "protected artifact bindings resolved from approved metadata references (currently UNRESOLVED)",
                "a committed batch authorization record for 232 looks, ledger 196 -> 428",
                "a human-issued scoring token; a separate human-issued analysis token",
                "activation of the disabled entry points in a reviewed change"],
            "still_disabled": list(STILL_DISABLED),
            "issued_tokens": [], "authorized_looks": 0}


def write_candidate(focused, regression):
    package = FrozenPackage.load()
    require(file_sha256(REAL_LEDGER) == REAL_LEDGER_SHA256 and len(REAL_LEDGER.read_bytes().splitlines()) == 196,
            "real ledger changed")
    mirror = json.loads(EVIDENCE["mirror_evidence.json"].read_text(encoding="utf-8"))
    require(mirror["units"] == 232 and mirror["ledger"]["rows_before_after"] == [196, 428]
            and mirror["real_model_results"]["l1_max_abs_diff_vs_frozen_bundle_validation_scores"] == 0.0
            and mirror["real_model_results"]["l2_max_abs_diff_vs_persisted_l2_fused_validation"] == 0.0,
            "mirror evidence does not meet acceptance")
    tests, test_counts = _junit(focused)
    regressions, regression_counts = _junit(regression)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, source in EVIDENCE.items():
        (OUTPUT / name).write_bytes(canonical_bytes(source))
    (OUTPUT / "focused_tests.xml").write_bytes(tests)
    (OUTPUT / "regression_tests.xml").write_bytes(regressions)
    for name, payload in (("cap_precheck_evidence.json", cap_precheck_evidence()), ("integrity_report.json", integrity_report(package))):
        (OUTPUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    (OUTPUT / "authorization_checklist.json").write_text(json.dumps(authorization_checklist(), indent=2, sort_keys=True) + "\n",
                                                         encoding="utf-8", newline="\n")
    manifest = {"status": STATUS, "stage": STAGE, "starting_checkpoint": START,
                "real_execution_enabled": False, "independently_approved": False, "authorization_issued": False,
                "executor_source_identity": source_identity(), "executor_sources": list(EXECUTOR_SOURCES),
                "sources": {p: canonical_sha256(p) for p in SOURCES},
                "files": {p: canonical_sha256(OUTPUT / p) for p in FILES},
                "scientific": scientific_bindings(package),
                "tests": {"focused": test_counts, "regression": regression_counts, "regression_scope": REGRESSION_SCOPE},
                "real_ledger": {"rows": 196, "sha256": REAL_LEDGER_SHA256},
                "generated_from_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "supersedes": {"phase5_executable_freeze_candidate (5.5)": "historical; executor sources changed in 5.6C",
                               "phase5_6b_candidate": "historical component candidate; executor sources changed in 5.6C"},
                "note": ("Self-generated candidate. Verifying it against a commit proves byte consistency only; "
                         "it is not an independent review, approval or authorization.")}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"executor_source_identity": manifest["executor_source_identity"], "sources": len(SOURCES),
                      "focused": test_counts, "regression": regression_counts}))


def verify_candidate(commit):
    require(isinstance(commit, str) and re.fullmatch("[0-9a-f]{40}", commit), "full external commit SHA required")

    def git(path):
        return subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, check=True).stdout.replace(b"\r\n", b"\n")

    trusted = git(OUTPUT.as_posix() + "/manifest.json")
    require(canonical_bytes(OUTPUT / "manifest.json") == trusted, "working candidate differs from the named commit")
    manifest = json.loads(trusted)
    require(manifest["status"] == STATUS and manifest["stage"] == STAGE and manifest["starting_checkpoint"] == START
            and manifest["real_execution_enabled"] is False and manifest["independently_approved"] is False
            and manifest["authorization_issued"] is False, "candidate status/authority mismatch")
    require(set(manifest["sources"]) == set(SOURCES) and set(manifest["files"]) == set(FILES), "candidate inventory mismatch")
    for path, digest in manifest["sources"].items():
        require(sha256(git(path)).hexdigest() == digest == canonical_sha256(path), f"candidate source mismatch: {path}")
    for name, digest in manifest["files"].items():
        require(sha256(git(f"{OUTPUT.as_posix()}/{name}")).hexdigest() == digest == canonical_sha256(OUTPUT / name),
                f"candidate evidence mismatch: {name}")
    require(manifest["executor_source_identity"] == source_identity(), "executor identity mismatch")
    package = FrozenPackage.load(PACKAGE)
    require(manifest["scientific"] == scientific_bindings(package), "scientific package bindings changed")
    require(file_sha256(REAL_LEDGER) == REAL_LEDGER_SHA256, "real ledger changed")
    checklist = json.loads((OUTPUT / "authorization_checklist.json").read_text(encoding="utf-8"))
    require(checklist["issued_tokens"] == [] and checklist["authorized_looks"] == 0
            and all(v["state"] == "UNRESOLVED" for v in checklist["human_decisions"].values()), "checklist claims authority")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write-candidate", action="store_true")
    mode.add_argument("--verify-candidate")
    parser.add_argument("--focused-report", type=Path)
    parser.add_argument("--regression-report", type=Path)
    args = parser.parse_args()
    if args.verify_candidate:
        m = verify_candidate(args.verify_candidate)
        print(json.dumps({"verified_commit": args.verify_candidate, "status": m["status"],
                          "executor_source_identity": m["executor_source_identity"],
                          "self_check_only": True}))
    else:
        require(args.focused_report is not None and args.regression_report is not None, "focused and regression reports required")
        write_candidate(args.focused_report, args.regression_report)
