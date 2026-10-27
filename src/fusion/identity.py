"""Per-bundle identity verification for Phase-5 L1 development fits (verifier only).

Each L1 bundle carries its own truthful provenance: the code commit it ran from,
the Phase-5 source hash, and raw SHA-256 hashes of the input files it read.
The same committed file can be serialised with CRLF (an older checkout) or LF
(a fresh checkout honouring ``eol=lf``), so the same scientific input can
legitimately carry two raw hashes.  This module checks, per bundle, without
assuming every bundle shares one serialisation:

* the experiment id recomputes from the bundle's stored config;
* the recorded Phase-5 source hash equals the source hash of the committed
  files at the bundle's own code commit;
* each recorded raw input hash is the SHA-256 of the committed content at that
  commit in LF form or in CRLF form (anything else is ``NEITHER`` -> failure).

Everything is read from git objects at the bundle's commit, never from the
working tree, so the check is independent of how this checkout is serialised.
It is not part of the fitting code and does not change any experiment identity.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import subprocess
from typing import Iterable, Mapping

from src.experiment.phase32 import experiment_config_from_dict
from src.experiment.provenance import content_sha256


def committed_bytes(commit: str, path: str | Path) -> bytes:
    """Exact committed content of ``path`` at ``commit`` (the git blob, LF form)."""

    return subprocess.run(["git", "show", f"{commit}:{Path(path).as_posix()}"],
                          capture_output=True, check=True).stdout


def raw_hash_form(recorded: str, committed: bytes) -> str:
    """LF or CRLF if ``recorded`` hashes that serialisation of the committed content, else NEITHER."""

    lf = committed.replace(b"\r\n", b"\n")
    if recorded == sha256(lf).hexdigest():
        return "LF"
    if recorded == sha256(lf.replace(b"\n", b"\r\n")).hexdigest():
        return "CRLF"
    return "NEITHER"


def source_hash_at_commit(commit: str, paths: Iterable[Path]) -> str:
    """The Phase-5 source hash as it would be computed from a checkout of ``commit``."""

    return content_sha256({str(p): sha256(committed_bytes(commit, p).replace(b"\r\n", b"\n")).hexdigest()
                           for p in paths})


def bundle_identity(bundle: Mapping, *, source_paths: Iterable[Path], preprocessing_configs: Mapping[str, Path],
                    split_manifest: Path) -> dict:
    config = bundle["config"]
    commit = config["code_git_sha"]
    recomputed = experiment_config_from_dict(config).experiment_id
    preprocessing_form = raw_hash_form(config["preprocessing_config_hash"],
                                       committed_bytes(commit, preprocessing_configs[config["regime"]]))
    split_form = raw_hash_form(config["split_manifest_hash"], committed_bytes(commit, split_manifest))
    recorded_source = config["model_hyperparameters"]["phase5_source_hash"]
    source_ok = recorded_source == source_hash_at_commit(commit, source_paths)
    return {
        "experiment_id": bundle["experiment_id"],
        "code_git_sha": commit,
        "experiment_id_recomputes": recomputed == bundle["experiment_id"] == config["experiment_id"],
        "phase5_source_hash": recorded_source,
        "source_hash_matches_commit": source_ok,
        "preprocessing_config_hash": config["preprocessing_config_hash"],
        "preprocessing_config_hash_form": preprocessing_form,
        "split_manifest_hash": config["split_manifest_hash"],
        "split_manifest_hash_form": split_form,
        "identity_valid": (recomputed == bundle["experiment_id"] and source_ok
                           and "NEITHER" not in (preprocessing_form, split_form)),
    }


# ------------------------------------------------------------------ scientific-unit binding
# The frozen L1 representation of each family (src/fusion/l1.py run_l1_fit); the smallest
# mapping that lets the verifier compare a stored config with the permitted recipe.
L1_REPRESENTATION = {
    "mahalanobis_shrinkage": ("feature_fusion_classical", "train_only_feature_zscore_after_segment_median"),
    "dense_autoencoder": ("feature_fusion_spectral", "train_only_per_bin_zscore_of_log_power_spectrum"),
}


def _expected_input_dim(machine: str) -> int:
    from src.fusion.assembly import SPECTRAL_DIMS

    return int(sum(SPECTRAL_DIMS[machine].values()))


def _expected_model_id(family: str, seed: int, machine: str) -> str | None:
    """The model identity the frozen recipe gives this (family, seed, machine); None if not permitted."""

    try:
        if family == "dense_autoencoder":
            from src.models.dense_autoencoder import DenseAutoencoder

            return DenseAutoencoder(input_dim=_expected_input_dim(machine), seed=seed).model_id
        if family == "mahalanobis_shrinkage":
            from src.models.mahalanobis import ShrinkageMahalanobis

            return ShrinkageMahalanobis(seed=seed).model_id
    except ValueError:
        return None
    return None


def _frozen_contract_hashes(machine: str, regime: str, split_manifest: Path) -> tuple[str, str]:
    """(training, source-validation) frozen cohort-contract hashes of one machine and regime."""

    import pandas as pd

    from src.experiment.data import SourceValidationCohortContract, TrainingCohortContract
    from src.fusion.assembly import PREPROCESSING_CONFIGS
    from src.preprocessing.config import load_config

    config = load_config(PREPROCESSING_CONFIGS[regime])
    assignments = pd.read_csv(split_manifest)
    return (TrainingCohortContract.from_frozen_assignments(assignments, config, machine=machine).frozen_manifest_hash,
            SourceValidationCohortContract.from_frozen_assignments(assignments, config, machine=machine)
            .frozen_manifest_hash)


def unit_binding(bundle: Mapping, unit: Mapping, *, model_root: str | Path | None = None,
                 split_manifest: str | Path = Path("reports/phase2_2/protocol_b_few_shot_adaptation_v3.csv")) -> dict:
    """Bind one L1 bundle to the registered scientific unit it is counted for.

    ``bundle_identity`` shows a bundle is internally self-consistent; this shows it is the
    *right* bundle: every unit label (provenance, stored config, score provenance, model
    manifest) equals the registry unit, the stored hyperparameters equal the frozen recipe
    for that family and machine, the representation fields equal the frozen L1 ones, and
    the model identity equals the one the recipe gives that family/seed/input size.
    A bundle whose own hash fields agree but whose labels or recipe were altered fails.
    """

    from src.fusion import l1
    from src.preprocessing.config import load_config
    from src.fusion.assembly import PREPROCESSING_CONFIGS

    expected = {"machine": unit["machine"], "regime": unit["regime"], "family": unit["family"],
                "seed": int(unit["seed"])}
    config = bundle["config"]
    provenance = bundle["provenance"]
    score = provenance["score_provenance"]
    manifest = provenance["model_manifest"]
    family, machine, regime, seed = expected["family"], expected["machine"], expected["regime"], expected["seed"]

    labels = {
        "provenance_fusion_unit": {**provenance["fusion_unit"], "seed": int(provenance["fusion_unit"]["seed"])},
        "config": {"machine": config["machine"], "regime": config["regime"], "family": config["model_family"],
                   "seed": int(config["model_seed"])},
        "score_provenance": {"machine": score["machine"], "regime": score["regime"], "family": score["model_family"],
                             "seed": int(score["model_seed"])},
    }
    ids = {bundle["experiment_id"], config["experiment_id"], provenance["experiment_id"], score["experiment_id"],
           manifest["experiment_id"]}
    try:
        recomputed = experiment_config_from_dict(config).experiment_id
    except Exception:  # noqa: BLE001 - an unparseable config simply fails the binding
        recomputed = None

    stored_hyper = config["model_hyperparameters"]
    recipe = None
    if family in l1.L1_FAMILIES:
        recipe = l1._hyperparameters(
            family, input_dim=_expected_input_dim(machine) if family == "dense_autoencoder" else None,
            code_source_hash=stored_hyper.get("phase5_source_hash", ""))
    kind, normalization = L1_REPRESENTATION.get(family, (None, None))
    frozen_preprocessing = load_config(PREPROCESSING_CONFIGS[regime]) if regime in PREPROCESSING_CONFIGS else None
    representation = {
        "feature_representation": (config["feature_representation"],
                                   l1.L1_SPECTRAL_REPRESENTATION if kind == "feature_fusion_spectral"
                                   else l1.L1_CLASSICAL_REPRESENTATION),
        "feature_version": (config["feature_version"],
                            l1.L1_SPECTRAL_FEATURE_VERSION if kind == "feature_fusion_spectral"
                            else l1.L1_CLASSICAL_FEATURE_VERSION),
        "normalization_strategy": (config["normalization_strategy"], normalization),
        "preprocessing_config_id": (config["preprocessing_config_id"],
                                    frozen_preprocessing.config_id if frozen_preprocessing else None),
        "window_length_s": (config["window_length_s"],
                            frozen_preprocessing.window_length_s if frozen_preprocessing else None),
        "stride_s": (config["stride_s"], frozen_preprocessing.stride_s if frozen_preprocessing else None),
        "split_version": (config["split_version"], frozen_preprocessing.split_version if frozen_preprocessing else None),
        "modality": (provenance.get("modality"), l1.L1_MODALITY),
    }
    expected_model = _expected_model_id(family, seed, machine)
    # Machine and regime fix the frozen training/validation cohorts, which a relabelled bundle
    # cannot share (the deterministic Mahalanobis recipe and model id are machine-independent).
    try:
        train_hash, validation_hash = _frozen_contract_hashes(machine, regime, Path(split_manifest))
    except (KeyError, ValueError):
        train_hash = validation_hash = None
    checks = {
        "unit_registered_as_new_l1_fit": unit.get("l1_source") == "new_fit_phase5" and family in l1.L1_FAMILIES,
        "unit_labels_match_registry": all(v == expected for v in labels.values()),
        "experiment_id_consistent": len(ids) == 1 and recomputed in ids,
        "recipe_matches_frozen": recipe is not None and stored_hyper == recipe,
        "representation_matches_frozen": all(a == b for a, b in representation.values()),
        "cohort_contracts_match_unit": (train_hash is not None
                                        and provenance.get("training_contract_hash") == train_hash
                                        and provenance.get("validation_contract_hash") == validation_hash),
        "model_identity_matches_unit": (expected_model is not None and manifest["family"] == family
                                        and manifest["model_id"] == expected_model == score["model_id"]),
    }
    if model_root is not None:
        from src.experiment.phase4 import ModelStore

        try:
            stored = ModelStore(model_root).verify(bundle["experiment_id"],
                                                   expected_manifest_sha256=manifest["model_manifest_sha256"])
            checks["model_files_match_manifest"] = (stored["model_id"] == manifest["model_id"]
                                                    and stored["family"] == family)
        except (OSError, ValueError, KeyError):
            checks["model_files_match_manifest"] = False
    return {"unit": f"{machine}|{regime}|{family}|{seed}", "experiment_id": bundle["experiment_id"],
            "checks": checks, "labels": labels,
            "mismatched_representation_fields": sorted(k for k, (a, b) in representation.items() if a != b),
            "mismatched_recipe_keys": sorted(k for k in set(stored_hyper) | set(recipe or {})
                                             if (recipe or {}).get(k) != stored_hyper.get(k)),
            "binding_valid": all(checks.values())}
