"""Phase 5.3 per-bundle identity verification: truthful mixed CRLF/LF input-hash provenance."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from src.fusion import identity

C679 = "c6798967a34d0cf430782496b1a6139c66de1116"
PREPROC_S = Path("configs/preprocessing/mechasight_few_shot_regime_s.json")
SPLIT = Path("reports/phase2_2/protocol_b_few_shot_adaptation_v3.csv")


def test_raw_hash_form_classifies_lf_crlf_and_rejects_anything_else():
    committed = b"a\nbc\n"
    assert identity.raw_hash_form(sha256(b"a\nbc\n").hexdigest(), committed) == "LF"
    assert identity.raw_hash_form(sha256(b"a\r\nbc\r\n").hexdigest(), committed) == "CRLF"
    assert identity.raw_hash_form(sha256(b"a\nbX\n").hexdigest(), committed) == "NEITHER"
    assert identity.raw_hash_form("0" * 64, committed) == "NEITHER"


@pytest.mark.parametrize("path, crlf, lf", [
    (PREPROC_S, "c892619a1208007a642a9b94e2c0cf6b1abb891608b9d2ad59e2f6da2ab491b9",
     "bb934f9d4d970e3c283fdc038202fd5e86e2fe251d9cf81873babcf467572406"),
    (SPLIT, "68e5406d63e96cf5ee418ae58808f5c3e7790889ba1b8497420642d1f0d8409e",
     "dedddce2d24abf24d9d498a247fcc7517d1002a143849baf664511db4decc47e"),
])
def test_both_recorded_serialisations_of_the_same_committed_input_are_recognised(path, crlf, lf):
    committed = identity.committed_bytes(C679, path)
    assert identity.raw_hash_form(crlf, committed) == "CRLF"      # Phase 4 and the main checkout
    assert identity.raw_hash_form(lf, committed) == "LF"          # a fresh checkout honouring eol=lf


def test_source_hash_is_computed_from_the_bundles_own_commit():
    import sys
    sys.path.insert(0, "scripts")
    import run_phase5_development as driver
    assert identity.source_hash_at_commit(C679, driver.SOURCE_IDENTITY_PATHS).startswith("f1c01c9671b4")


def _committed_bundles():
    from src.fusion.l1 import load_l1_bundle
    import subprocess
    tracked = subprocess.run(["git", "ls-files", "reports/phase5_development/l1/experiments"],
                             capture_output=True, text=True, check=True).stdout.split()
    return [load_l1_bundle(d) for d in sorted({Path(t).parent for t in tracked})]


def test_every_committed_bundle_has_a_valid_truthful_identity():
    import sys
    sys.path.insert(0, "scripts")
    import run_phase5_development as driver
    bundles = _committed_bundles()
    assert len(bundles) >= 14
    for bundle in bundles:
        result = identity.bundle_identity(bundle, source_paths=driver.SOURCE_IDENTITY_PATHS,
                                          preprocessing_configs=driver.PREPROCESSING_CONFIGS,
                                          split_manifest=driver.SPLIT_MANIFEST)
        assert result["identity_valid"] and result["experiment_id_recomputes"], bundle["experiment_id"]
        assert result["preprocessing_config_hash_form"] in ("CRLF", "LF")
        assert result["preprocessing_config_hash_form"] == result["split_manifest_hash_form"]


def test_a_falsified_raw_hash_or_id_is_rejected():
    import copy
    import sys
    sys.path.insert(0, "scripts")
    import run_phase5_development as driver
    bundle = _committed_bundles()[0]
    kwargs = dict(source_paths=driver.SOURCE_IDENTITY_PATHS, preprocessing_configs=driver.PREPROCESSING_CONFIGS,
                  split_manifest=driver.SPLIT_MANIFEST)
    forged = copy.deepcopy(bundle)
    forged["config"]["split_manifest_hash"] = "f" * 64
    result = identity.bundle_identity(forged, **kwargs)
    assert result["split_manifest_hash_form"] == "NEITHER" and not result["identity_valid"]
    renamed = copy.deepcopy(bundle)
    renamed["experiment_id"] = "p3-" + "0" * 24
    assert not identity.bundle_identity(renamed, **kwargs)["identity_valid"]


# ---------------------------------------------- scientific-unit binding (independent audit finding A)
import copy as _copy

from src.experiment.phase32 import experiment_config_from_dict as _config_from_dict


def _registry_unit(machine, regime, family, seed):
    from src.fusion.assembly import load_design_registry
    return next(u for u in load_design_registry()
                if (u["machine"], u["regime"], u["family"], int(u["seed"])) == (machine, regime, family, int(seed)))


def _label(bundle):
    u = bundle["provenance"]["fusion_unit"]
    return u["machine"], u["regime"], u["family"], int(u["seed"])


def _relabel(bundle, *, machine=None, regime=None, family=None, seed=None, recompute_ids=True):
    """A forged bundle whose every unit label is changed and whose own id fields stay self-consistent."""

    forged = _copy.deepcopy(bundle)
    config, prov = forged["config"], forged["provenance"]
    score = prov["score_provenance"]
    for key, value in (("machine", machine), ("regime", regime), ("family", family), ("seed", seed)):
        if value is None:
            continue
        prov["fusion_unit"][key] = value
        config[{"family": "model_family", "seed": "model_seed"}.get(key, key)] = value
        score[{"family": "model_family", "seed": "model_seed"}.get(key, key)] = value
    if recompute_ids:
        new_id = _config_from_dict({**config, "experiment_id": None}).experiment_id
        for holder in (forged, config, prov, score, prov["model_manifest"]):
            holder["experiment_id"] = new_id
    return forged


def _bound(bundle, **kwargs):
    return identity.unit_binding(bundle, _registry_unit(*_label(bundle)), **kwargs)


@pytest.fixture(scope="module")
def bundles():
    return {_label(b): b for b in _committed_bundles()}


def test_E_every_committed_bundle_is_bound_to_its_registered_unit(bundles):
    from src.fusion.l1 import l1_units
    from src.fusion.assembly import load_design_registry
    assert len(bundles) == 24
    for unit in l1_units(load_design_registry()):
        key = (unit["machine"], unit["regime"], unit["family"], int(unit["seed"]))
        result = identity.unit_binding(bundles[key], unit, model_root="data/interim/phase5/models")
        assert result["binding_valid"], (key, result["checks"])


def test_A_swapped_seed_labels_between_two_valid_bundles_are_rejected(bundles):
    a = bundles[("BrushlessMotor", "S", "dense_autoencoder", 1)]
    b = bundles[("BrushlessMotor", "S", "dense_autoencoder", 2)]
    # the audit counterexample: only the provenance unit labels are swapped
    for bundle, seed in ((a, 2), (b, 1)):
        swapped = _copy.deepcopy(bundle)
        swapped["provenance"]["fusion_unit"]["seed"] = seed
        result = _bound(swapped)
        assert not result["binding_valid"] and not result["checks"]["unit_labels_match_registry"]
    # a self-consistent forgery: every label changed and every id field recomputed
    forged = _relabel(a, seed=2)
    result = _bound(forged)
    assert not result["binding_valid"] and not result["checks"]["model_identity_matches_unit"]


@pytest.mark.parametrize("source, change", [
    (("RoboticArm", "S", "mahalanobis_shrinkage", 0), {"machine": "BrushlessMotor"}),
    (("BrushlessMotor", "T", "mahalanobis_shrinkage", 0), {"regime": "S"}),
    (("RoboticArm", "T", "dense_autoencoder", 3), {"machine": "BrushlessMotor"}),
    (("RoboticArm", "S", "dense_autoencoder", 0), {"regime": "T"}),
])
def test_B_swapped_machine_or_regime_labels_are_rejected_even_when_self_consistent(bundles, source, change):
    forged = _relabel(bundles[source], **change)
    result = _bound(forged)
    assert result["checks"]["experiment_id_consistent"]          # the forgery is internally consistent
    assert not result["binding_valid"]
    assert not result["checks"]["cohort_contracts_match_unit"]


def test_C_a_modified_model_family_label_is_rejected(bundles):
    source = bundles[("BrushlessMotor", "S", "mahalanobis_shrinkage", 0)]
    forged = _relabel(source, family="dense_autoencoder")
    result = _bound(forged)
    assert not result["binding_valid"]
    assert not result["checks"]["recipe_matches_frozen"] and not result["checks"]["model_identity_matches_unit"]


@pytest.mark.parametrize("key, value", [("learning_rate", 0.01), ("hidden", [512, 64]), ("patience", 20),
                                        ("input_dim", 2918), ("bottleneck", 32)])
def test_D_parameters_differing_from_the_frozen_recipe_are_rejected(bundles, key, value):
    forged = _copy.deepcopy(bundles[("BrushlessMotor", "T", "dense_autoencoder", 4)])
    forged["config"]["model_hyperparameters"][key] = value
    new_id = _config_from_dict({**forged["config"], "experiment_id": None}).experiment_id
    for holder in (forged, forged["config"], forged["provenance"], forged["provenance"]["score_provenance"],
                   forged["provenance"]["model_manifest"]):
        holder["experiment_id"] = new_id
    result = _bound(forged)
    assert not result["binding_valid"] and key in result["mismatched_recipe_keys"]


def test_D_a_representation_field_differing_from_the_frozen_l1_recipe_is_rejected(bundles):
    forged = _copy.deepcopy(bundles[("RoboticArm", "S", "mahalanobis_shrinkage", 0)])
    forged["config"]["normalization_strategy"] = "none"
    result = _bound(forged)
    assert not result["binding_valid"] and result["mismatched_representation_fields"] == ["normalization_strategy"]


def test_F_both_truthful_line_ending_forms_stay_accepted(bundles):
    import sys
    sys.path.insert(0, "scripts")
    import run_phase5_development as driver
    forms = {}
    for bundle in bundles.values():
        ident = identity.bundle_identity(bundle, source_paths=driver.SOURCE_IDENTITY_PATHS,
                                         preprocessing_configs=driver.PREPROCESSING_CONFIGS,
                                         split_manifest=driver.SPLIT_MANIFEST)
        assert ident["identity_valid"] and _bound(bundle)["binding_valid"]
        forms[ident["preprocessing_config_hash_form"]] = forms.get(ident["preprocessing_config_hash_form"], 0) + 1
    assert forms == {"CRLF": 14, "LF": 10}
