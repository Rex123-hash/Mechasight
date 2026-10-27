"""L1 feature-level fusion: the 24 new development fits of design v2 (Phase 5.3).

Each fit is the frozen Phase-4 single-modality recipe with the modality input
replaced by all three modalities:

* ``mahalanobis_shrinkage`` on the 98 Phase-3.1 engineered features in their
  frozen sorted order (the order the 28 reused Phase-3.2 all-channel detectors
  were fitted on), train-only feature z-score;
* ``dense_autoencoder`` on the per-window concatenation [mic | acc | gyro] of the
  frozen log-power spectra (2918 / 2936 inputs), train-only per-bin z-score,
  unchanged architecture, optimiser, loss, schedule, early stopping and
  best-validation checkpoint.

Training reads only the frozen training cohort; source validation is used only
for the frozen early-stopping/checkpoint rule and the q90 threshold.  The
Phase-4 cohort contracts, standardiser, model classes, evaluation gate,
threshold rule, model store and result store are reused unchanged; nothing in
``src/experiment/phase4.py`` is modified.

The autoencoder standardiser is fitted per modality block exactly as Phase 4
fitted it (same rows, same float64 accumulation), so it equals the three
Phase-4 single-modality standardisers concatenated; ``verify_standardizer_
equivalence`` checks this against the stored Phase-4 artifacts.  The fused
training matrix is standardised in place to avoid a second full-size copy.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.experiment.config import ExperimentConfig
from src.experiment.constants import SCORE_DIRECTION
from src.experiment.data import (
    SourceValidationCohortContract,
    TrainingCohortContract,
    fit_training_standardizer,
    load_estimator_training_data,
    load_source_validation_data,
)
from src.experiment.gate import EvaluationGate
from src.experiment.phase31 import DATASET_RELEASE_IDENTITY, load_feature_artifact, phase31_feature_columns
from src.experiment.phase4 import (
    ModelStore,
    PHASE4_EVALUATION_STAGE,
    _build_model,
    _model_hyperparameters,
    aggregate_window_scores_by_segment,
    dependency_environment,
    load_frozen_bundle,
    modality_feature_columns,
    rewrite_bundle_manifest,
    validated_training_window_rows,
    validated_validation_window_rows,
)
from src.experiment.pretest import PretestResultStore
from src.experiment.provenance import file_sha256, initial_state_hash, ordered_id_hash
from src.experiment.scores import ScoreBatch, ScoreProvenance, select_protocol_b_threshold
from src.fusion.assembly import (
    DESIGN_CONFIG_SHA256,
    MODALITIES,
    PREPROCESSING_CONFIGS,
    FusionAssemblyError,
    assemble_classical_fusion,
    assemble_spectral_fusion,
    spectral_fusion_blocks,
)
from src.preprocessing.config import load_config
from src.preprocessing.normalization import EPSILON, FittedChannelStandardizer
from src.preprocessing.spectral_cache import load_spectral_cache, spectral_cache_path


STAGE_NAME = "multi_sensor_fusion_development"
PHASE5_SCHEMA_VERSION = "phase5-fusion-development-v1"
L1_MODALITY = "fusion[mic+acc+gyro]"
L1_FAMILIES = ("mahalanobis_shrinkage", "dense_autoencoder")
L1_CLASSICAL_FEATURE_VERSION = "phase3.1-segment-median-engineered-v1"
L1_CLASSICAL_REPRESENTATION = "median_by_segment(existing_100ms_nonoverlap_time_and_frequency_features)"
L1_SPECTRAL_FEATURE_VERSION = "phase5-l1-log-power-spectrum-concatenation-v1"
L1_SPECTRAL_REPRESENTATION = "feature_fusion[mic|acc|gyro]:log_power_spectrum_window(phase4-log-power-spectrum-v1)"
STANDARDIZE_CHUNK = 8192


class L1FitError(RuntimeError):
    """An L1 fit violated the frozen recipe; no bundle was written."""


def l1_units(registry) -> tuple[Mapping, ...]:
    units = tuple(u for u in registry if u["l1_source"] == "new_fit_phase5")
    if len(units) != 24 or {u["family"] for u in units} != set(L1_FAMILIES):
        raise L1FitError("design registry does not hold exactly the 24 planned L1 fits")
    return units


def _hyperparameters(family: str, *, input_dim: int | None, code_source_hash: str) -> dict[str, Any]:
    payload = _model_hyperparameters(family, L1_MODALITY, input_dim=input_dim, code_source_hash="")
    payload.pop("phase4_source_hash")
    payload.update({
        "stage": STAGE_NAME,
        "stage_schema_version": PHASE5_SCHEMA_VERSION,
        "phase5_source_hash": code_source_hash,
        "fusion_level": "L1_feature_level",
        "fusion_modalities": list(MODALITIES),
        "design_config_sha256": DESIGN_CONFIG_SHA256,
    })
    return payload


# ----------------------------------------------------------- dense autoencoder
def _fit_block_standardizers(
    cache_root, *, machine: str, regime: str, contract, config
) -> tuple[FittedChannelStandardizer, pd.DataFrame, np.ndarray, tuple[str, ...]]:
    """Fit one standardiser per modality exactly as Phase 4 did, then concatenate."""

    path = spectral_cache_path(cache_root, machine=machine, regime=regime, role="training")
    means, stds = [], []
    reference_meta = rows = segment_ids = None
    count = None
    for modality in MODALITIES:
        X, meta = load_spectral_cache(path, modality=modality, expected_split="train")
        modality_rows = validated_training_window_rows(meta, contract, config)
        if reference_meta is None:
            reference_meta, rows = meta, modality_rows
            segment_ids = tuple(sorted(set(meta.iloc[rows]["segment_id"])))
        elif not np.array_equal(modality_rows, rows) or not meta.equals(reference_meta):
            raise L1FitError(f"{modality} training rows differ from mic")
        selected = X if np.array_equal(rows, np.arange(len(X))) else X[rows]
        block = FittedChannelStandardizer.fit([selected.astype(np.float64)], split="train", segment_ids=segment_ids)
        means.append(block.mean)
        stds.append(block.std)
        count = block.n_samples if count is None else count
        if block.n_samples != count:
            raise L1FitError("modality standardisers saw different sample counts")
        del X, selected
    fused = FittedChannelStandardizer(
        mean=np.concatenate(means), std=np.concatenate(stds), n_samples=int(count),
        fit_split="train", fit_segment_ids=segment_ids, epsilon=EPSILON,
    )
    return fused, reference_meta, rows, segment_ids


def _standardize_in_place(standardizer: FittedChannelStandardizer, X: np.ndarray) -> np.ndarray:
    """Same arithmetic as phase4.standardize_float32, written back into ``X``."""

    for start in range(0, len(X), STANDARDIZE_CHUNK):
        X[start:start + STANDARDIZE_CHUNK] = standardizer.transform(X[start:start + STANDARDIZE_CHUNK]).astype(np.float32)
    return X


def _spectral_inputs(cache_root, *, machine, regime, role, rows, reference_meta=None):
    data = assemble_spectral_fusion(cache_root, machine=machine, regime=regime, role=role)
    if reference_meta is not None:
        expected = reference_meta.loc[:, list(data.windows.columns)].reset_index(drop=True)
        if not data.windows.equals(expected):
            raise L1FitError(f"{role} fused window provenance differs from the contract-validated rows")
    X = data.X if np.array_equal(rows, np.arange(len(data.X))) else data.X[rows]
    meta = data.windows.iloc[rows].reset_index(drop=True)
    return X, meta, data


def fit_dense_autoencoder_l1(*, unit, cache_root, config, train_contract, validation_contract):
    machine, regime, seed = unit["machine"], unit["regime"], int(unit["seed"])
    blocks = spectral_fusion_blocks(machine, regime)
    input_dim = blocks[MODALITIES[-1]][1]
    standardizer, train_reference, train_rows, training_segment_ids = _fit_block_standardizers(
        cache_root, machine=machine, regime=regime, contract=train_contract, config=config)
    X_train, train_meta, train_data = _spectral_inputs(cache_root, machine=machine, regime=regime,
                                                       role="training", rows=train_rows,
                                                       reference_meta=train_reference)
    training_cache_sha256 = train_data.cache_sha256
    del train_data
    _standardize_in_place(standardizer, X_train)
    val_path = spectral_cache_path(cache_root, machine=machine, regime=regime, role="source_validation")
    _, val_reference = load_spectral_cache(val_path, modality="mic", expected_split="validation")
    validation_rows = validated_validation_window_rows(val_reference, validation_contract, config)
    X_validation, validation_meta, validation_data = _spectral_inputs(
        cache_root, machine=machine, regime=regime, role="source_validation",
        rows=validation_rows, reference_meta=val_reference)
    X_validation = _standardize_in_place(standardizer, np.array(X_validation, dtype=np.float32, copy=True))
    model = _build_model("dense_autoencoder", seed, input_dim=input_dim)
    initial_hash = initial_state_hash(model.network)
    model.fit(X_train, X_validation)
    window_scores = model.score_windows(X_validation)
    table = aggregate_window_scores_by_segment(validation_meta, window_scores)
    provenance = {
        "training_cohort_hash": ordered_id_hash(training_segment_ids),
        "source_validation_cohort_hash": ordered_id_hash(tuple(sorted(set(validation_meta["segment_id"])))),
        "training_spectral_cache_hash": training_cache_sha256,
        "source_validation_spectral_cache_hash": validation_data.cache_sha256,
        "spectral_blocks": {m: list(v) for m, v in blocks.items()},
        "initial_state_sha256": initial_hash,
    }
    counts = {"training_windows": int(len(X_train)), "validation_windows": int(len(X_validation))}
    return model, standardizer, table, input_dim, provenance, counts, dict(model.history)


# --------------------------------------------------------------- Mahalanobis
def fit_mahalanobis_l1(*, unit, feature_root, train_contract, validation_contract):
    machine, regime, seed = unit["machine"], unit["regime"], int(unit["seed"])
    train_frame, train_meta_json = load_feature_artifact(feature_root, machine=machine, regime=regime, role="training")
    validation_frame, validation_meta_json = load_feature_artifact(
        feature_root, machine=machine, regime=regime, role="source_validation")
    features = phase31_feature_columns(train_frame)
    if phase31_feature_columns(validation_frame) != features:
        raise L1FitError("training and source-validation feature schemas differ")
    assembled = assemble_classical_fusion(feature_root, machine=machine, regime=regime, role="training")
    if assembled.columns != features or len(features) != 98:
        raise L1FitError("L1 classical features differ from the frozen 98-column assembly")
    training = load_estimator_training_data(train_frame, train_contract, feature_columns=features)
    standardizer = fit_training_standardizer(train_frame, train_contract, feature_columns=features).standardizer
    validation = load_source_validation_data(validation_frame, validation_contract, feature_columns=features)
    model = _build_model("mahalanobis_shrinkage", seed).fit(standardizer.transform(training.X))
    scores = model.score(standardizer.transform(validation.X))
    table = pd.DataFrame({
        "segment_id": validation.segment_ids,
        "parent_recording": validation.parent_recordings,
        "domain": ("source",) * len(validation.segment_ids),
        "anomaly_label": validation.anomaly_labels,
        "score": scores,
        "window_count": validation_frame.set_index("segment_id").loc[list(validation.segment_ids), "window_count"].to_numpy(),
    }).sort_values("segment_id").reset_index(drop=True)
    provenance = {
        "training_cohort_hash": ordered_id_hash(training.segment_ids),
        "source_validation_cohort_hash": ordered_id_hash(validation.segment_ids),
        "training_feature_artifact_hash": train_meta_json["artifact_sha256"],
        "source_validation_feature_artifact_hash": validation_meta_json["artifact_sha256"],
        "feature_columns": list(features),
    }
    counts = {"training_windows": int(train_frame["window_count"].sum()),
              "validation_windows": int(validation_frame["window_count"].sum())}
    return model, standardizer, table, None, provenance, counts, {}


# ------------------------------------------------------------------- runner
def run_l1_fit(
    *, unit: Mapping, feature_root, spectral_cache_root, model_root, result_root,
    split_manifest_path, code_git_sha: str, code_source_hash: str,
    phase4_pretest_root=None, phase4_model_root=None,
) -> dict[str, Any]:
    """Fit one planned L1 unit; persist the model and an immutable validation-only bundle."""

    family, machine, regime, seed = unit["family"], unit["machine"], unit["regime"], int(unit["seed"])
    if unit.get("l1_source") != "new_fit_phase5" or family not in L1_FAMILIES:
        raise L1FitError(f"unit {machine}/{regime}/{family}/{seed} is not a planned new L1 fit")
    config_path = PREPROCESSING_CONFIGS[regime]
    config = load_config(config_path)
    assignments = pd.read_csv(split_manifest_path)
    train_contract = TrainingCohortContract.from_frozen_assignments(assignments, config, machine=machine)
    validation_contract = SourceValidationCohortContract.from_frozen_assignments(assignments, config, machine=machine)
    started = perf_counter()
    if family == "dense_autoencoder":
        model, standardizer, table, input_dim, data_provenance, counts, history = fit_dense_autoencoder_l1(
            unit=unit, cache_root=spectral_cache_root, config=config,
            train_contract=train_contract, validation_contract=validation_contract)
        feature_count, feature_version, representation = input_dim, L1_SPECTRAL_FEATURE_VERSION, L1_SPECTRAL_REPRESENTATION
        normalization = "train_only_per_bin_zscore_of_log_power_spectrum"
    else:
        model, standardizer, table, input_dim, data_provenance, counts, history = fit_mahalanobis_l1(
            unit=unit, feature_root=feature_root, train_contract=train_contract,
            validation_contract=validation_contract)
        feature_count, feature_version, representation = 98, L1_CLASSICAL_FEATURE_VERSION, L1_CLASSICAL_REPRESENTATION
        normalization = "train_only_feature_zscore_after_segment_median"
    data_provenance["split_manifest_hash"] = file_sha256(split_manifest_path)
    # The fused standardiser must equal the unit's stored Phase-4 standardisers;
    # checked before anything is written, so a mismatch leaves no artifact.
    if phase4_pretest_root is not None:
        data_provenance["standardizer_equivalence"] = verify_standardizer_equivalence(
            unit, standardizer, phase4_pretest_root=phase4_pretest_root, phase4_model_root=phase4_model_root,
            feature_columns=data_provenance.get("feature_columns"))

    experiment = ExperimentConfig(
        timestamp_utc=datetime.now(timezone.utc).isoformat(), machine=machine, protocol=config.protocol,
        regime=regime, model_family=family,
        model_hyperparameters=_hyperparameters(family, input_dim=input_dim, code_source_hash=code_source_hash),
        model_seed=seed, preprocessing_config_id=config.config_id,
        preprocessing_config_hash=file_sha256(config_path), split_version=config.split_version,
        split_manifest_hash=file_sha256(split_manifest_path), dataset_release_identity=DATASET_RELEASE_IDENTITY,
        feature_representation=representation, feature_version=feature_version,
        window_length_s=config.window_length_s, stride_s=config.stride_s,
        normalization_strategy=normalization, code_git_sha=code_git_sha,
        dependency_environment=dependency_environment(),
    )
    score_provenance = ScoreProvenance(
        experiment_id=experiment.experiment_id, machine=machine, protocol=config.protocol, regime=regime,
        model_id=model.model_id, model_family=family, model_seed=seed,
        preprocessing_config_id=config.config_id, preprocessing_config_hash=file_sha256(config_path),
        split_version=config.split_version, score_direction=SCORE_DIRECTION, data_provenance=data_provenance,
        config_provenance={"code_git_sha": code_git_sha, "phase5_source_hash": code_source_hash,
                           "stage_schema_version": PHASE5_SCHEMA_VERSION, "modality": L1_MODALITY},
    )
    batch = ScoreBatch(
        provenance=score_provenance, scores=table["score"].to_numpy(),
        segment_ids=tuple(table["segment_id"].astype(str)),
        parent_recording_ids=tuple(table["parent_recording"].astype(str)),
        cohort_roles=("source_validation",) * len(table), domains=tuple(table["domain"].astype(str)),
        anomaly_labels=tuple(table["anomaly_label"].astype(str)),
    )
    EvaluationGate(experiment, score_provenance).evaluate_validation(batch)
    threshold = select_protocol_b_threshold(batch, expected_provenance=score_provenance)
    model_manifest = ModelStore(model_root).write(
        experiment_id=experiment.experiment_id, family=family, model_id=model.model_id,
        model=model, standardizer=standardizer)
    scores = batch.scores
    summary = {
        "evaluation_stage": PHASE4_EVALUATION_STAGE, "stage": STAGE_NAME, "test_evaluated": False,
        "descriptive_only_no_model_ranking": True, "modality": L1_MODALITY, "fusion_level": "L1_feature_level",
        "training_segment_count": len(train_contract.authorized_segment_ids),
        "training_window_count": counts["training_windows"],
        "source_validation_segment_count": len(validation_contract.segment_ids),
        "source_validation_window_count": counts["validation_windows"],
        "feature_count": int(feature_count), "score_count": int(len(scores)),
        "finite_score_count": int(np.isfinite(scores).sum()),
        "score_min": float(np.min(scores)), "score_median": float(np.median(scores)),
        "score_max": float(np.max(scores)), "score_q90": float(threshold.value),
        "fit_seconds": float(history.get("fit_seconds") or getattr(model, "fit_seconds_", 0.0) or 0.0),
        "total_seconds": float(perf_counter() - started), "model_saved": True,
        "model_manifest_sha256": model_manifest["model_manifest_sha256"], "training_history": history,
    }
    bundle_path = PretestResultStore(result_root).write_bundle(
        config=experiment,
        provenance={
            "evaluation_stage": PHASE4_EVALUATION_STAGE, "stage": STAGE_NAME,
            "stage_schema_version": PHASE5_SCHEMA_VERSION, "test_evaluated": False,
            "score_direction": SCORE_DIRECTION, "modality": L1_MODALITY, "fusion_level": "L1_feature_level",
            "design_config_sha256": DESIGN_CONFIG_SHA256,
            "fusion_unit": {k: unit[k] for k in ("machine", "regime", "family", "seed")},
            "training_cohort_hash": data_provenance["training_cohort_hash"],
            "validation_cohort_hash": data_provenance["source_validation_cohort_hash"],
            "training_contract_hash": train_contract.frozen_manifest_hash,
            "validation_contract_hash": validation_contract.frozen_manifest_hash,
            "score_provenance": asdict(score_provenance), "model_manifest": model_manifest,
            "completion_status": "COMPLETE",
        },
        summary=summary, threshold=threshold,
        validation_scores=table.assign(cohort_role="source_validation")[
            ["segment_id", "parent_recording", "cohort_role", "domain", "anomaly_label", "score", "window_count"]],
    )
    rewrite_bundle_manifest(bundle_path)
    return {
        "experiment_id": experiment.experiment_id, "model_id": model.model_id, "machine": machine,
        "regime": regime, "family": family, "seed": seed, "feature_count": int(feature_count),
        "threshold_q90": float(threshold.value), "epochs_run": history.get("epochs_run"),
        "best_epoch": history.get("best_epoch"), "best_validation_window_mse": history.get("best_validation_window_mse"),
        "stopped_early": history.get("stopped_early"), "fit_seconds": summary["fit_seconds"],
        "total_seconds": summary["total_seconds"], "bundle": str(bundle_path).replace("\\", "/"),
        "test_evaluated": False,
    }


# ------------------------------------------------------------ verification
def verify_standardizer_equivalence(unit: Mapping, standardizer: FittedChannelStandardizer, *,
                                    phase4_pretest_root, phase4_model_root, feature_columns=None) -> dict:
    """The fused standardiser must equal the unit's three stored Phase-4 standardisers."""

    store = ModelStore(phase4_model_root)
    offset = 0
    for modality in MODALITIES:
        eid = unit["phase4_single_modality_ids"][modality]
        bundle = load_frozen_bundle(Path(phase4_pretest_root) / eid)
        _model, phase4_std, _manifest = store.load(eid, expected_manifest_sha256=bundle.model_manifest_sha256)
        if feature_columns is None:
            width = phase4_std.mean.size
            index = np.arange(offset, offset + width)
            offset += width
        else:
            names = modality_feature_columns(feature_columns, modality)
            index = np.array([list(feature_columns).index(n) for n in names])
        if not (np.array_equal(standardizer.mean[index], phase4_std.mean)
                and np.array_equal(standardizer.std[index], phase4_std.std)
                and standardizer.n_samples == phase4_std.n_samples
                and standardizer.fit_segment_ids == phase4_std.fit_segment_ids):
            raise L1FitError(f"fused standardiser differs from the Phase-4 {modality} standardiser {eid}")
    return {"equal_to_phase4_standardizers": True,
            "phase4_ids": dict(unit["phase4_single_modality_ids"])}


# ------------------------------------------------------------ reproduction
def load_l1_bundle(path) -> dict:
    """Read one Phase-5 L1 bundle after verifying every file against its manifest."""

    import json
    from src.experiment.provenance import canonical_sha256

    bundle = Path(path)
    manifest = json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["files"].items():
        if canonical_sha256(bundle / name) != digest:
            raise L1FitError(f"L1 bundle file {name} hash mismatch in {bundle.name}")
    provenance = json.loads((bundle / "provenance.json").read_text(encoding="utf-8"))
    if (provenance.get("stage"), provenance.get("test_evaluated"), provenance.get("completion_status")) != (
            STAGE_NAME, False, "COMPLETE"):
        raise L1FitError(f"{bundle.name} is not a complete Phase-5 development bundle")
    return {
        "experiment_id": manifest["experiment_id"], "path": bundle,
        "config": json.loads((bundle / "config.json").read_text(encoding="utf-8")),
        "provenance": provenance,
        "threshold": json.loads((bundle / "threshold.json").read_text(encoding="utf-8")),
        "validation_scores": pd.read_csv(bundle / "validation_scores.csv", dtype={"segment_id": str, "parent_recording": str},
                                         float_precision="round_trip"),
    }


def reproduce_l1_bundle(path, *, model_root, feature_root, spectral_cache_root, split_manifest_path) -> dict:
    """Re-score source validation from the stored model; compare with the bundle's recorded scores."""

    from src.evaluation.policy import normal_quantile_threshold

    bundle = load_l1_bundle(path)
    unit = bundle["provenance"]["fusion_unit"]
    family, machine, regime = unit["family"], unit["machine"], unit["regime"]
    model, standardizer, _ = ModelStore(model_root).load(
        bundle["experiment_id"], expected_manifest_sha256=bundle["provenance"]["model_manifest"]["model_manifest_sha256"])
    config = load_config(PREPROCESSING_CONFIGS[regime])
    contract = SourceValidationCohortContract.from_frozen_assignments(
        pd.read_csv(split_manifest_path), config, machine=machine)
    if family == "dense_autoencoder":
        val_path = spectral_cache_path(spectral_cache_root, machine=machine, regime=regime, role="source_validation")
        _, reference = load_spectral_cache(val_path, modality="mic", expected_split="validation")
        rows = validated_validation_window_rows(reference, contract, config)
        X, meta, _ = _spectral_inputs(spectral_cache_root, machine=machine, regime=regime,
                                      role="source_validation", rows=rows, reference_meta=reference)
        X = _standardize_in_place(standardizer, np.array(X, dtype=np.float32, copy=True))
        table = aggregate_window_scores_by_segment(meta, model.score_windows(X))
    else:
        frame, _ = load_feature_artifact(feature_root, machine=machine, regime=regime, role="source_validation")
        features = phase31_feature_columns(frame)
        validation = load_source_validation_data(frame, contract, feature_columns=features)
        table = pd.DataFrame({"segment_id": validation.segment_ids,
                              "score": model.score(standardizer.transform(validation.X))}
                             ).sort_values("segment_id").reset_index(drop=True)
    recorded = bundle["validation_scores"].sort_values("segment_id").reset_index(drop=True)
    if not table["segment_id"].astype(str).reset_index(drop=True).equals(recorded["segment_id"]):
        raise L1FitError(f"{bundle['experiment_id']} reproduced segments differ")
    diff = float(np.max(np.abs(table["score"].to_numpy(dtype=np.float64) - recorded["score"].to_numpy(dtype=np.float64))))
    threshold = normal_quantile_threshold(table["score"].to_numpy(dtype=np.float64), quantile=0.9)
    return {"experiment_id": bundle["experiment_id"], "max_abs_score_difference": diff,
            "threshold_recorded": float(bundle["threshold"]["value"]), "threshold_reproduced": float(threshold),
            "reproduced": diff <= 1e-9 and abs(threshold - float(bundle["threshold"]["value"])) <= 1e-9}
