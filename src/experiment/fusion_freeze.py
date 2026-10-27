"""Phase 5.5 candidate assembly; metadata and development artifacts only.

This module has no evaluator, authorisation writer, scope opener or model loader.
Historical output locations below are used ONLY for their JSON identity manifests.
Protected input caches and raw recordings are inventoried by stat, never opened.
The split-assignment CSV is not parsed: it also contains protected labels.
"""
from __future__ import annotations

from dataclasses import asdict
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess

from src.experiment.phase32 import experiment_config_from_dict
from src.experiment.provenance import canonical_sha256, content_sha256, hash_form, ordered_id_hash

BASE = "e69e85e4616c8e8ded1f0a04bc0f50a0ff16e825"
STATUS = "CANDIDATE — NOT YET AUTHORIZED FOR PROTECTED EVALUATION."
DESIGN = Path("configs/phase5/fusion_design_v2.json")
DESIGN_HASH = "2e33299d3c0347ebdc13392d149b248b8ec175f2bea5b2049e5bb2fa996a759a"
OUTPUT = Path("reports/phase5_freeze_candidate")
DEV = Path("reports/phase5_development")
PRETEST = Path("reports/phase4_pretest/experiments")
HISTORICAL_L1 = Path("reports/phase3_2/experiments")
SCORE_MANIFEST = Path("reports/phase4_protected/staging/score_manifest.json")
MODALITIES = ("mic", "acc", "gyro")
SUBSETS = (MODALITIES, ("acc", "gyro"), ("mic", "gyro"), ("mic", "acc"))
COHORT_FIELDS = ("training_cohort_hash", "training_contract_hash", "validation_cohort_hash", "validation_contract_hash")
BUNDLE_FILES = {"config.json", "provenance.json", "threshold.json", "validation_scores.csv", "validation_summary.json"}
GAPS = [
    "Independent read-only pre-test audit and explicit human authorization have not occurred.",
    "The V2 section-13 protected executor and its fixture tests are absent. No new scope is enabled. "
    "A separately reviewed implementation and updated freeze are required before Phase 5.6; this metadata candidate is not execution-ready.",
    "Ignored protected caches/raw inputs have stat identities only; content integrity must be verified inside a future authorized scope AFTER exposure is recorded.",
    "Full split-based unit_binding was not rerun: it parses protected labels. Its existing successful report is pinned; "
    "fresh metadata binding uses the frozen Phase-4 pre-test cohort contracts instead.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def unit_key(unit):
    return "|".join(str(unit[k]) for k in ("machine", "regime", "family", "seed"))


def verify_files(directory, files, allowed):
    """Only explicit development-bundle/model filenames can be content-hashed."""
    require(set(files) <= allowed, f"unexpected file in development manifest: {directory}")
    for name, digest in files.items():
        require(hash_form(directory / name, digest) != "NEITHER", f"artifact hash mismatch: {directory / name}")


def bundle(directory, eid, *, model_root=None):
    manifest = read_json(directory / "bundle_manifest.json")
    require(manifest["experiment_id"] == eid and set(manifest["files"]) == BUNDLE_FILES, "bundle identity/file set")
    verify_files(directory, manifest["files"], BUNDLE_FILES)
    config, provenance, threshold = [read_json(directory / n) for n in ("config.json", "provenance.json", "threshold.json")]
    require(experiment_config_from_dict(config).experiment_id == eid == provenance["experiment_id"], "experiment id does not recompute")
    require(not provenance["test_evaluated"], "expected validation-only bundle")
    require(threshold["quantile"] == .9 and threshold["quantile_method"] == "higher", "q90 threshold policy changed")
    require(threshold["source_role"] == "source_validation" and threshold["source_experiment_id"] == eid, "threshold source")
    require(ordered_id_hash(threshold["source_segment_ids"]) == provenance["validation_cohort_hash"] == threshold["source_cohort_hash"], "threshold cohort")
    model = provenance.get("model_manifest")
    if model_root is not None:
        require(model is not None, "missing persisted model")
        directory_model = model_root / eid
        require(canonical_sha256(directory_model / "model_manifest.json") == model["model_manifest_sha256"], "model manifest hash")
        stored = read_json(directory_model / "model_manifest.json")
        require(stored == {k: v for k, v in model.items() if k != "model_manifest_sha256"}, "model manifest identity")
        require(len(stored["files"]) == 2, "model/standardizer file set")
        verify_files(directory_model, stored["files"], {"model.pth", "model.joblib", "standardizer.joblib"})
    compact_threshold = {k: v for k, v in threshold.items() if k != "source_segment_ids"}
    compact_threshold["source_segment_count"] = len(threshold["source_segment_ids"])
    record = {
        "experiment_id": eid, "bundle_path": directory.as_posix(),
        "bundle_manifest_sha256": canonical_sha256(directory / "bundle_manifest.json"),
        "machine": config["machine"], "regime": config["regime"], "family": config["model_family"],
        "seed": config["model_seed"], "modality": provenance.get("modality", "all_channel"),
        "code_git_sha": config["code_git_sha"], "preprocessing_config_hash": config["preprocessing_config_hash"],
        "split_manifest_hash": config["split_manifest_hash"],
        "cohort_contracts": {k: provenance[k] for k in COHORT_FIELDS},
        "threshold": compact_threshold, "model_manifest": model,
        "model_path": (model_root / eid).as_posix() if model_root else None,
        "stored_hyperparameters": config["model_hyperparameters"],
        "recorded_environment": config["dependency_environment"],
        "data_provenance": provenance["score_provenance"].get("data_provenance", {}),
    }
    return record, config, provenance


def metadata_binding(record, config, provenance, unit, reference_contracts, recorded):
    """Fresh label/recipe/contract binding, without opening split labels or model objects.

    Complements (does not impersonate a rerun of) src.fusion.identity.unit_binding.
    The full verifier's successful, immutable completion record is also required.
    """
    from src.fusion import l1
    from src.fusion.assembly import SPECTRAL_DIMS, PREPROCESSING_CONFIGS
    from src.models.dense_autoencoder import DENSE_AE_SPEC, AUTOENCODER_IMPLEMENTATION_VERSION
    from src.models.mahalanobis import MAHALANOBIS_SPEC, MAHALANOBIS_IMPLEMENTATION_VERSION
    from src.preprocessing.config import load_config

    expected = {k: unit[k] for k in ("machine", "regime", "family", "seed")}
    score = provenance["score_provenance"]
    dim = sum(SPECTRAL_DIMS[unit["machine"]].values()) if unit["family"] == "dense_autoencoder" else None
    hyper = l1._hyperparameters(unit["family"], input_dim=dim, code_source_hash=config["model_hyperparameters"]["phase5_source_hash"])
    preprocessing = load_config(PREPROCESSING_CONFIGS[unit["regime"]])
    if dim is not None:
        identity = {"version": AUTOENCODER_IMPLEMENTATION_VERSION, "family": unit["family"],
                    "spec": DENSE_AE_SPEC.to_dict(), "input_dim": dim, "seed": unit["seed"]}
        representation, version = l1.L1_SPECTRAL_REPRESENTATION, l1.L1_SPECTRAL_FEATURE_VERSION
        normalization = "train_only_per_bin_zscore_of_log_power_spectrum"
    else:
        identity = {"version": MAHALANOBIS_IMPLEMENTATION_VERSION, "family": unit["family"],
                    "hyperparameters": dict(MAHALANOBIS_SPEC.hyperparameters), "seed": "deterministic"}
        representation, version = l1.L1_CLASSICAL_REPRESENTATION, l1.L1_CLASSICAL_FEATURE_VERSION
        normalization = "train_only_feature_zscore_after_segment_median"
    labels = {k: record[k] for k in expected}
    checks = {
        "labels": labels == expected == provenance["fusion_unit"] == {
            "machine": score["machine"], "regime": score["regime"], "family": score["model_family"], "seed": score["model_seed"]}
            == {"machine": config["machine"], "regime": config["regime"], "family": config["model_family"], "seed": config["model_seed"]},
        "experiment_ids": all(v == record["experiment_id"] for v in (
            config["experiment_id"], provenance["experiment_id"], score["experiment_id"], record["model_manifest"]["experiment_id"])),
        "recipe": config["model_hyperparameters"] == hyper,
        "representation": (config["feature_representation"], config["feature_version"], config["normalization_strategy"], provenance["modality"])
                          == (representation, version, normalization, l1.L1_MODALITY),
        "preprocessing": all(config[k] == getattr(preprocessing, k) for k in ("window_length_s", "stride_s", "split_version"))
                         and config["preprocessing_config_id"] == preprocessing.config_id,
        "cohort_contracts": record["cohort_contracts"] == reference_contracts,
        "model_identity": record["model_manifest"]["model_id"] == score["model_id"] == "p4-model-" + content_sha256(identity)[:20]
                          and record["model_manifest"]["family"] == unit["family"],
        "recorded_unit_verifier": recorded["binding"]["binding_valid"] and all(recorded["binding"]["checks"].values())
                                  and recorded["binding"]["experiment_id"] == record["experiment_id"]
                                  and recorded["binding"]["unit"] == unit_key(unit),
        "recorded_byte_identity": recorded["identity"]["identity_valid"] and recorded["identity"]["experiment_id"] == record["experiment_id"]
                                 and recorded["identity"]["code_git_sha"] == config["code_git_sha"]
                                 and recorded["identity"]["phase5_source_hash"] == hyper["phase5_source_hash"]
                                 and all(recorded["identity"][k] == config[k] for k in ("split_manifest_hash", "preprocessing_config_hash")),
    }
    require(all(checks.values()), f"L1 metadata binding failed: {unit_key(unit)}: {checks}")
    return checks


def validate_registry(registry, design):
    """Pure structural checks also used by adversarial fixture tests."""
    units, detectors, calibrations = registry["units"], registry["detectors"], registry["calibrations"]
    require(len(units) == 52 and {u["unit"] for u in units} == {unit_key(u) for u in design["unit_registry"]}, "scientific unit set")
    require(len(detectors) == 208 and len(calibrations) == 208, "detector/calibration count")
    new, old, singles = [], [], []
    for u in units:
        expected = next(r for r in design["unit_registry"] if unit_key(r) == u["unit"])
        labels = {k: expected[k] for k in ("machine", "regime", "family", "seed")}
        require({k: u[k] for k in labels} == labels and u["L1"]["source"] == expected["l1_source"], "unit labels/source")
        require(u["L0"] == expected["phase4_single_modality_ids"], "L0 identity")
        singles.extend(u["L0"].values())
        for eid in [*u["L0"].values(), u["L1"]["experiment_id"]]:
            require(eid in detectors and {k: detectors[eid][k] for k in labels} == labels, "detector belongs to different scientific unit")
            threshold = detectors[eid]["threshold"]
            require(math.isfinite(threshold["value"]) and threshold["quantile"] == .9
                    and threshold["quantile_method"] == "higher", "detector threshold")
        if expected["l1_source"] == "new_fit_phase5":
            new.append(u["L1"]["experiment_id"])
        else:
            require(u["L1"]["experiment_id"] == expected["l1_phase3_2_id"], "reused L1 identity")
            old.append(u["L1"]["experiment_id"])
        require(set(u["L2"]) == {"+".join(s) for s in SUBSETS}, "four L2 subsets required")
        for subset in SUBSETS:
            cid = u["L2"]["+".join(subset)]
            require(cid in calibrations, "missing calibration")
            r = calibrations[cid]
            require(r["calibration_id"] == cid and r["unit"] == u["unit"] and tuple(r["subset"]) == subset, "calibration unit/subset")
            require({k: r[k] for k in labels} == labels, "calibration labels")
            require(r["component_experiment_ids"] == {m: u["L0"][m] for m in subset}, "calibration components")
            require(r["threshold"]["quantile"] == .9 and r["threshold"]["method"] == "higher" and r["iqr_quantile_method"] == "linear", "calibration policy")
            require(not r["test_evaluated"] and math.isfinite(r["threshold"]["value"])
                    and set(r["modality_calibration"]) == set(subset)
                    and all(c["iqr"] > 0 and all(math.isfinite(v) for v in c.values()) for c in r["modality_calibration"].values()), "invalid calibration")
            require(all(r["validation_cohort_hash"] == detectors[eid]["cohort_contracts"]["validation_cohort_hash"] for eid in r["component_experiment_ids"].values()), "calibration cohort")
        require(u["L3"] == fallback_map(u), "L3 fallback/abstention mapping")
    require(len(set(singles)) == 156 and len(set(new)) == 24 and len(set(old)) == 28, "unique arm identities")
    require(set(singles) | set(new) | set(old) == set(detectors), "unreferenced detector")
    require({cid for u in units for cid in u["L2"].values()} == set(calibrations), "unreferenced calibration")


def fallback_map(unit):
    return {
        "all_valid": unit["L2"]["mic+acc+gyro"],
        "one_value_invalid": {m: unit["L2"]["+".join(s for s in MODALITIES if s != m)] for m in MODALITIES},
        "two_value_invalid_survivor": dict(unit["L0"]),
        "all_value_invalid": "ABSTAIN_SENSOR_INVALID", "any_alignment_or_absent_failure": "ABSTAIN_ALIGNMENT_FAILURE",
        "single_threshold_rule": "original L0 raw score and its raw q90; never robust-z vs raw threshold",
        "integrity_flag": "always emitted independently of anomaly decision",
    }


def build_registry(design):
    completion = read_json(DEV / "l1/l1_completion_and_reproduction.json")
    require(all(completion[k] for k in ("all_identities_valid", "all_reproduced", "all_units_bound", "all_standardizers_equal_to_phase4")), "prior L1 completion failed")
    new = {}
    for directory in sorted((DEV / "l1/experiments").iterdir()):
        if directory.is_dir():
            record, config, provenance = bundle(directory, directory.name, model_root=Path("data/interim/phase5/models"))
            key = unit_key(record)
            require(key not in new, "duplicate L1 fit for unit")
            new[key] = (record, config, provenance)
    require(len(new) == 24 and len(completion["units"]) == 24, "L1 fit count")
    verified = {r["experiment_id"]: r for r in completion["units"]}
    require(len(verified) == 24, "duplicate completion identity")
    records = read_json(DEV / "l2/l2_calibration_records.json")
    calibrations = {r["calibration_id"]: r for r in records}
    require(len(calibrations) == len(records) == 208, "duplicate/missing calibration")
    units, detectors = [], {}
    phase4_manifest = read_json(SCORE_MANIFEST)
    for expected in design["unit_registry"]:
        key = unit_key(expected)
        l0 = expected["phase4_single_modality_ids"]
        for modality, eid in l0.items():
            record, _, provenance = bundle(PRETEST / eid, eid, model_root=Path("data/interim/phase4/models"))
            require(record["modality"] == modality, "single modality mismatch")
            require(eid in phase4_manifest["tables"], "missing historical score metadata")
            record["historical_score_identity"] = phase4_manifest["tables"][eid]
            detectors[eid] = record
        contracts = detectors[l0["mic"]]["cohort_contracts"]
        require(all(detectors[eid]["cohort_contracts"] == contracts for eid in l0.values()), "modality cohort mismatch")
        if expected["l1_source"] == "new_fit_phase5":
            record, config, provenance = new[key]
            record["metadata_binding_checks"] = metadata_binding(record, config, provenance, expected, contracts, verified[record["experiment_id"]])
        else:
            eid = expected["l1_phase3_2_id"]
            record, _, _ = bundle(Path("reports/phase3_1/validation_only") / eid, eid)
            require(record["cohort_contracts"] == contracts, "reused L1 cohorts")
            historic_prov = read_json(HISTORICAL_L1 / eid / "provenance.json")
            require(all(historic_prov[k] == contracts[k] for k in ("training_cohort_hash", "validation_cohort_hash")), "historical L1 cohorts")
            require(historic_prov["threshold_provenance"] == read_json(Path(record["bundle_path"]) / "threshold.json"), "historical L1 threshold")
            record["historical_bundle_manifest"] = (HISTORICAL_L1 / eid / "bundle_manifest.json").as_posix()
            record["historical_bundle_manifest_sha256"] = canonical_sha256(record["historical_bundle_manifest"])
            record["model_storage_note"] = "No serialized historical model; reuse sealed scores, never refit."
        detectors[record["experiment_id"]] = record
        matching = [r for r in records if r["unit"] == key]
        u = {"unit": key, **{k: expected[k] for k in ("machine", "regime", "family", "seed")},
             "L0": l0, "L1": {"source": expected["l1_source"], "experiment_id": record["experiment_id"]},
             "L2": {"+".join(r["subset"]): r["calibration_id"] for r in matching}}
        u["L3"] = fallback_map(u)
        units.append(u)
    registry = {"status": STATUS, "base_commit": BASE, "design_sha256": DESIGN_HASH,
                "units": units, "detectors": detectors, "calibrations": calibrations}
    validate_registry(registry, design)
    return registry


def evaluation_manifest(registry, design, ledger_rows):
    validate_registry(registry, design)
    planned = []
    for u in registry["units"]:
        if u["L1"]["source"] == "new_fit_phase5":
            planned.append({"experiment_id": u["L1"]["experiment_id"], "unit": u["unit"], "operation": "new_L1_inference"})
        for subset, cid in u["L2"].items():
            planned.append({"experiment_id": cid, "unit": u["unit"], "subset": subset, "operation": "derived_L2_scoring"})
    planned.sort(key=lambda row: row["experiment_id"])
    ids = [p["experiment_id"] for p in planned]
    require(len(ids) == len(set(ids)) == design["experiment_matrix"]["new_protected_looks"] == 232, "planned look count/identity")
    require(len(ledger_rows) == 196 and not set(ids) & {r["experiment_id"] for r in ledger_rows}, "ledger collision or changed start")
    identity = {"base_commit": BASE, "registry_sha256": content_sha256(registry), "planned_ids": sorted(ids),
                "cohort_hashes": {m: c["test_cohort_hash"] for m, c in design["inherits_unchanged"]["common_test_cohort"].items()}}
    return {"status": STATUS, "authorization_granted": False, "unit_state": "PLANNED",
            "proposed_batch_id": "p5-candidate-batch-" + content_sha256(identity)[:24],
            "identity": identity, "scoring_head": None, "scoring_head_requirement": "bind separately at authorization; approved ids/cohorts must equal this proposal",
            "planned_units": planned, "accounting": design["experiment_matrix"],
            "required_order_and_recovery": design["protected_execution"], "unresolved_authorization_gates": GAPS}


def inventory():
    """Git-object inventory, not extraction of historical scientific payloads."""
    tracked = []
    output = git("ls-tree", "-r", "-l", BASE)
    for line in output.splitlines():
        meta, path = line.split("\t", 1)
        mode, kind, oid, size = meta.split()
        tracked.append({"path": path, "git_blob": oid, "bytes": int(size) if size != "-" else None})
    stats = {}
    for root in (Path("data/raw"), Path("data/interim/phase4/spectral"), Path("data/interim/phase32_common_test"),
                 Path("data/interim/protocol_a"), Path("data/interim/phase3_4")):
        stats[root.as_posix()] = {p.as_posix(): {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                                  for p in sorted(root.rglob("*")) if p.is_file()}
    # Include every ignored scientific-data file, including legacy caches whose path
    # does not match a recent stage name. Stat identities do not certify contents.
    ignored = git("ls-files", "--others", "--ignored", "--exclude-standard", "--", "data", "reports")
    extra = {}
    for name in ignored.splitlines():
        p = Path(name)
        if p.is_file():
            st = p.stat()
            extra[name] = {"bytes": st.st_size, "mtime_ns": st.st_mtime_ns}
    return {"base_commit": BASE, "tracked_git_inventory": tracked, "stat_only_inputs": stats,
            "all_ignored_data_and_reports_stat_only": extra,
            "limits": "Git blob IDs inventory committed history without opening payloads. Stats are not cryptographic content verification. "
                      "Previously recorded table hashes are metadata evidence; protected contents are not re-read by this candidate builder."}


def prepare():
    from src.experiment.ledger_history import verify_ledger_history
    from src.fusion.l2 import verify_calibration_artifacts
    from src.fusion.identity import source_hash_at_commit
    from scripts.run_phase5_development import SOURCE_IDENTITY_PATHS
    from scripts.run_phase5_l3 import SOURCE_IDENTITY_PATHS as L3_SOURCE_PATHS

    require(git("branch", "--show-current") == "phase5-multisensor-fusion", "unexpected branch")
    require(git("merge-base", "HEAD", BASE) == BASE, "base commit not ancestor")
    require(canonical_sha256(DESIGN) == DESIGN_HASH, "frozen design changed")
    # New candidate files and the metadata-reference tripwire are the only tracked
    # changes permitted. No historical outputs are diffed into the console.
    allowed = {"src/experiment/fusion_freeze.py", "scripts/prepare_phase5_freeze_candidate.py",
               "tests/test_phase5_freeze_candidate.py", "tests/test_phase5_sealed_score_guard.py",
               "docs/phase5_5_pretest_freeze_candidate.md"}
    changed = git("diff", "--name-only", BASE, "--").splitlines()
    require(all(p in allowed or p.startswith(OUTPUT.as_posix() + "/") for p in changed), "historical/frozen tracked file changed")
    history = verify_ledger_history()
    require(len(history.rows) == 196, "ledger changed")
    before = canonical_sha256(Path("reports/experiment_results/test_look_ledger.jsonl"))
    design = read_json(DESIGN)
    registry = build_registry(design)
    l2_check = verify_calibration_artifacts(DEV / "l2", PRETEST)
    source_hashes = {}
    for record in registry["detectors"].values():
        if "metadata_binding_checks" in record:
            commit = record["code_git_sha"]
            if commit not in source_hashes:
                source_hashes[commit] = source_hash_at_commit(commit, SOURCE_IDENTITY_PATHS)
            require(source_hashes[commit] == record["stored_hyperparameters"]["phase5_source_hash"], "fitting source hash mismatch")
    l3 = read_json(DEV / "l3/falsification/falsification_manifest.json")
    require(content_sha256({str(p): canonical_sha256(p) for p in L3_SOURCE_PATHS}) == l3["identity"]["l3_source_hash"], "L3 source changed")
    verify_files(DEV / "l3/falsification", l3["files"], {
        "detector_scores.csv", "runs.csv", "ungated_dead_input_scores.csv", "unit_decisions.csv"})
    require(canonical_sha256(DEV / "l3/clean_scan/clean_scan_summary.json") == l3["clean_scan_summary_sha256"], "clean scan summary hash")
    require(read_json(DEV / "l3/falsification/assertions.json")["all_pass_criteria_passed"], "recorded L3 assertions fail")
    frozen_files = {}
    for root in (DEV, Path("data/interim/phase5/models"), Path("data/interim/phase4/models")):
        frozen_files.update({p.as_posix(): canonical_sha256(p) for p in sorted(root.rglob("*")) if p.is_file()})
    for p in (DESIGN, Path("docs/phase5_fusion_design_v2.md"), Path("src/fusion/identity.py"), Path("src/preprocessing/protected_access.py"),
              Path("docs/phase5_entry_hygiene.md"), Path("docs/phase5_4_incident_protected_raw_read.md"),
              Path("docs/phase5_4_sensor_integrity_fusion.md"), Path("docs/phase5_3_line_ending_provenance.md")):
        frozen_files[p.as_posix()] = canonical_sha256(p)
    require(before == canonical_sha256(Path("reports/experiment_results/test_look_ledger.jsonl")), "ledger mutated during preparation")
    environment = {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "torch", "h5py", "pytest")}
    contract_keys = ("research_question", "hypotheses", "estimand", "machines", "regimes", "modalities", "families", "arms", "integrity_gate",
                     "inherits_unchanged", "outcomes", "interpretation", "contrasts", "statistics", "claim_boundary", "unresolved_until_measured", "protected_execution")
    documents = {
        "scientific_registry.json": registry,
        "proposed_evaluation_manifest.json": evaluation_manifest(registry, design, history.rows),
        "scientific_contract.json": {"status": STATUS, "design_sha256": DESIGN_HASH, **{k: design[k] for k in contract_keys}},
        "integrity_inventory.json": {**inventory(), "development_and_model_canonical_sha256": frozen_files,
                                     "ledger_sha256": before, "closed_segments": [asdict(s) for s in history.segments]},
        "validation_report.json": {"status": STATUS, "candidate_metadata_checks": "PASS", "execution_ready": False,
                                   "l1_metadata_bindings": 24, "l0_model_manifests_verified": 156, "reused_l1_metadata": 28,
                                   "l2_development_reproduction": l2_check, "l3_assertions": "existing E0-E6 PASS; not rerun",
                                   "ledger_rows_before_after": [196, 196], "protected_contents_read": False,
                                   "full_unit_verifier_rerun": False, "unresolved_authorization_gates": GAPS,
                                   "candidate_environment": {"python": platform.python_version(), **environment},
                                   "l3_official_identity": l3["identity"]},
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, document in documents.items():
        (OUTPUT / name).write_text(json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    manifest = {"status": STATUS, "base_commit": BASE, "execution_ready": False,
                "files": {name: canonical_sha256(OUTPUT / name) for name in documents},
                "builder_sources": {p: canonical_sha256(p) for p in (
                    "src/experiment/fusion_freeze.py", "scripts/prepare_phase5_freeze_candidate.py",
                    "tests/test_phase5_freeze_candidate.py", "tests/test_phase5_sealed_score_guard.py",
                    "tests/test_phase5_protected_read_guard.py", "docs/phase5_5_pretest_freeze_candidate.md")}}
    (OUTPUT / "candidate_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return {"status": STATUS, "units": 52, "new_L1": 24, "L2": 208, "proposed_looks": 232, "ledger": 196, "execution_ready": False}
