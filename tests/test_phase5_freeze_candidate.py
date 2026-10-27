"""Candidate validation, adversarial metadata mutations and a read-boundary check.

No historical replay and no fixture that snapshots protected payload bytes.
"""
from copy import deepcopy
import builtins
from pathlib import Path

import pytest

from src.experiment import fusion_freeze as freeze
from src.experiment.provenance import canonical_sha256, content_sha256


@pytest.fixture
def registry():
    return freeze.read_json(freeze.OUTPUT / "scientific_registry.json")


@pytest.fixture
def design():
    return freeze.read_json(freeze.DESIGN)


def test_manifest_hashes_and_immutable_scientific_contract(design):
    manifest = freeze.read_json(freeze.OUTPUT / "candidate_manifest.json")
    assert manifest["status"] == freeze.STATUS and manifest["execution_ready"] is False
    for name, digest in manifest["files"].items():
        assert canonical_sha256(freeze.OUTPUT / name) == digest
    for name, digest in manifest["builder_sources"].items():
        # The old candidate remains immutable. 5.5B adds reviewed test/tripwire
        # entries, whose current identities live in the new executable freeze.
        # Historical support identities are checked against the original commit.
        from hashlib import sha256
        from src.experiment.phase5_execution import committed
        assert sha256(committed(name)).hexdigest() == digest
        if name.startswith(("src/", "scripts/")):
            assert canonical_sha256(name) == digest
    contract = freeze.read_json(freeze.OUTPUT / "scientific_contract.json")
    for name in set(contract) - {"status", "design_sha256"}:
        assert contract[name] == design[name]


def test_actual_ids_fallbacks_and_proposed_accounting(registry, design):
    freeze.validate_registry(registry, design)
    ledger = [{"experiment_id": f"old-{i}"} for i in range(196)]
    proposed = freeze.evaluation_manifest(registry, design, ledger)
    committed = freeze.read_json(freeze.OUTPUT / "proposed_evaluation_manifest.json")
    assert proposed == committed
    assert proposed["authorization_granted"] is False and proposed["scoring_head"] is None
    assert sum(r["operation"] == "new_L1_inference" for r in proposed["planned_units"]) == 24
    assert sum(r["operation"] == "derived_L2_scoring" for r in proposed["planned_units"]) == 208
    assert proposed["accounting"]["ledger"] == {"expected_start": 196, "rows_appended": 232, "expected_end": 428}
    actual = {r["experiment_id"] for r in freeze.read_json(freeze.DEV / "l1/l1_completion_and_reproduction.json")["units"]}
    assert actual == {u["L1"]["experiment_id"] for u in registry["units"] if u["L1"]["source"] == "new_fit_phase5"}


@pytest.mark.parametrize("mutation", [
    "missing_unit", "duplicate_unit", "wrong_seed", "wrong_family", "wrong_component", "wrong_subset",
    "wrong_calibration_unit", "missing_calibration", "wrong_threshold_policy", "zero_iqr", "wrong_cohort",
    "wrong_pair_fallback", "wrong_single_fallback", "missing_abstention", "missing_detector",
    "wrong_unit_labels", "wrong_source", "nonfinite_threshold",
])
def test_corrupted_metadata_refuses(registry, design, mutation):
    data = deepcopy(registry)
    u = data["units"][0]
    cid = u["L2"]["mic+acc+gyro"]
    calibration = data["calibrations"][cid]
    detector = data["detectors"][u["L0"]["mic"]]
    if mutation == "missing_unit":
        data["units"].pop()
    elif mutation == "duplicate_unit":
        data["units"][-1] = deepcopy(u)
    elif mutation == "wrong_seed":
        detector["seed"] += 1
    elif mutation == "wrong_family":
        detector["family"] = "unregistered"
    elif mutation == "wrong_component":
        calibration["component_experiment_ids"]["mic"] = u["L0"]["acc"]
    elif mutation == "wrong_subset":
        calibration["subset"] = ["mic", "acc"]
    elif mutation == "wrong_calibration_unit":
        calibration["unit"] = "foreign-unit"
    elif mutation == "missing_calibration":
        del data["calibrations"][cid]
    elif mutation == "wrong_threshold_policy":
        calibration["threshold"]["method"] = "linear"
    elif mutation == "zero_iqr":
        calibration["modality_calibration"]["mic"]["iqr"] = 0
    elif mutation == "wrong_cohort":
        calibration["validation_cohort_hash"] = "foreign-cohort"
    elif mutation == "wrong_pair_fallback":
        u["L3"]["one_value_invalid"]["mic"] = cid
    elif mutation == "wrong_single_fallback":
        u["L3"]["two_value_invalid_survivor"]["mic"] = u["L0"]["acc"]
    elif mutation == "missing_abstention":
        del u["L3"]["all_value_invalid"]
    elif mutation == "missing_detector":
        del data["detectors"][u["L0"]["mic"]]
    elif mutation == "wrong_unit_labels":
        u["regime"] = "other"
    elif mutation == "wrong_source":
        u["L1"]["source"] = "new_fit_phase5"
    elif mutation == "nonfinite_threshold":
        detector["threshold"]["value"] = float("nan")
    with pytest.raises(ValueError):
        freeze.validate_registry(data, design)


@pytest.mark.parametrize("case", ["collision", "wrong_start"])
def test_exposure_plan_refuses_ledger_collision_or_changed_start(registry, design, case):
    ledger = [{"experiment_id": f"old-{i}"} for i in range(196)]
    if case == "collision":
        ledger[0]["experiment_id"] = next(iter(registry["calibrations"]))
    else:
        ledger.pop()
    with pytest.raises(ValueError, match="ledger collision"):
        freeze.evaluation_manifest(registry, design, ledger)


def test_manifest_path_cannot_escape_to_another_file(tmp_path):
    with pytest.raises(ValueError, match="unexpected file"):
        freeze.verify_files(tmp_path, {"../unexpected.csv": "hash"}, freeze.BUNDLE_FILES)


@pytest.mark.parametrize("field", ["seed", "recipe", "cohort", "model", "experiment"])
def test_new_l1_binding_refuses_relabelled_or_changed_fit_metadata(registry, design, field):
    unit = next(u for u in design["unit_registry"] if u["l1_source"] == "new_fit_phase5")
    u = next(u for u in registry["units"] if u["unit"] == freeze.unit_key(unit))
    record = deepcopy(registry["detectors"][u["L1"]["experiment_id"]])
    config = freeze.read_json(Path(record["bundle_path"]) / "config.json")
    provenance = freeze.read_json(Path(record["bundle_path"]) / "provenance.json")
    contracts = registry["detectors"][u["L0"]["mic"]]["cohort_contracts"]
    report = freeze.read_json(freeze.DEV / "l1/l1_completion_and_reproduction.json")
    recorded = next(r for r in report["units"] if r["experiment_id"] == record["experiment_id"])
    if field == "seed":
        config["model_seed"] += 1
    elif field == "recipe":
        config["model_hyperparameters"]["unexpected_parameter"] = 1
    elif field == "cohort":
        record["cohort_contracts"]["training_contract_hash"] = "foreign"
    elif field == "model":
        record["model_manifest"]["model_id"] = "foreign"
    elif field == "experiment":
        provenance["score_provenance"]["experiment_id"] = "foreign"
    with pytest.raises(ValueError, match="metadata binding failed"):
        freeze.metadata_binding(record, config, provenance, unit, contracts, recorded)


def test_registry_rebuild_without_protected_contents_or_full_split_verifier(monkeypatch, registry, design):
    """Refuse before open; actual metadata rebuild must still succeed."""
    from src.fusion import identity
    original_path_open, original_open = Path.open, builtins.open

    def check(path):
        name = Path(path).as_posix().replace("\\", "/")
        if ("data/raw/" in name or "common_evaluation" in name or "common_test" in name
                or "protocol_b_few_shot_adaptation_v3.csv" in name):
            raise AssertionError(f"protected input/label read attempted: {name}")
        if "phase4_protected" in name:
            assert name.endswith("staging/score_manifest.json"), name
        if "phase3_2/experiments/" in name:
            assert Path(name).name in {"provenance.json", "bundle_manifest.json"}, name
        assert not name.endswith((".parquet", ".h5")), name

    def guarded_path_open(self, *args, **kwargs):
        check(self)
        return original_path_open(self, *args, **kwargs)

    def guarded_open(file, *args, **kwargs):
        if not isinstance(file, int):
            check(file)
        return original_open(file, *args, **kwargs)

    def refuse_full_verifier(*args, **kwargs):
        raise AssertionError("full verifier would parse protected split labels")

    monkeypatch.setattr(Path, "open", guarded_path_open)
    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(identity, "unit_binding", refuse_full_verifier)
    monkeypatch.setattr(identity, "bundle_identity", refuse_full_verifier)
    assert content_sha256(freeze.build_registry(design)) == content_sha256(registry)
