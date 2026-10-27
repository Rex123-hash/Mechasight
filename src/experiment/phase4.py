"""Single-Modality Model Benchmarking (legacy roadmap name: Phase 4).

Pre-test side: fit every frozen (machine, regime, modality, family, seed)
configuration on the authorized normal training cohort, score source
validation, freeze its q90 threshold and persist an immutable bundle plus a
hash-manifested model artifact.

Protected side: one atomically authorized batch of exactly 156 looks; the
scoring loop reads no anomaly label and computes no metric.  Metrics exist
only after ``phase4_analysis.analyse_once`` verifies every score table.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version as package_version
import json
from pathlib import Path
import platform
import time
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping

import joblib
import numpy as np
import pandas as pd

from src.experiment.config import ExperimentConfig
from src.experiment.constants import SCORE_DIRECTION
from src.experiment.data import (
    FORBIDDEN_MODEL_FEATURES,
    SourceValidationCohortContract,
    TrainingCohortContract,
    fit_training_standardizer,
    load_estimator_training_data,
    load_source_validation_data,
)
from src.experiment.gate import EvaluationGate
from src.experiment.ledger import TestLookLedger, TestLookRecord
from src.experiment.phase31 import (
    DATASET_RELEASE_IDENTITY,
    load_feature_artifact,
    phase31_feature_columns,
)
from src.experiment.phase32 import (
    CommonTestCohortContract,
    common_test_artifact_paths,
    experiment_config_from_dict,
)
from src.experiment.pretest import PretestResultStore
from src.experiment.provenance import (
    canonical_bytes,
    canonical_json,
    canonical_sha256,
    content_sha256,
    file_sha256,
    ordered_id_hash,
)
from src.experiment.scores import (
    ScoreBatch,
    ScoreProvenance,
    ThresholdSelection,
    select_protocol_b_threshold,
)
from src.models.classical import ClassicalAnomalyBaseline, primary_spec
from src.models.dense_autoencoder import (
    AUTOENCODER_IMPLEMENTATION_VERSION,
    DENSE_AE_SPEC,
    DenseAutoencoder,
)
from src.models.mahalanobis import (
    MAHALANOBIS_IMPLEMENTATION_VERSION,
    MAHALANOBIS_SPEC,
    ShrinkageMahalanobis,
)
from src.preprocessing.config import MECHASIGHT_PROTOCOL, PreprocessingConfig, load_config
from src.preprocessing.normalization import EPSILON, FittedChannelStandardizer
from src.preprocessing.protected_access import require_protected_read, require_sealed_score_read
from src.preprocessing.spectral_cache import (
    load_spectral_cache,
    load_spectral_cache_metadata,
    spectral_cache_path,
)
from src.preprocessing.spectral_windows import (
    SPECTRAL_REPRESENTATION_VERSION,
    spectral_input_dim,
)


STAGE_NAME = "single_modality_model_benchmarking"
PHASE4_SCHEMA_VERSION = "phase4-single-modality-benchmark-v1"
PHASE4_EVALUATION_STAGE = "validation_only_pretest"
REPORTING_POLICY_ID = "p4-reporting-single-modality-v1"
TEST_REASON = "single-modality model benchmarking: one pre-registered 156-look batch"

MACHINES = ("RoboticArm", "BrushlessMotor")
REGIMES = ("S", "T")
MODALITIES = ("mic", "acc", "gyro")
FAMILIES = (
    "statistical_reference",
    "isolation_forest",
    "one_class_svm",
    "mahalanobis_shrinkage",
    "dense_autoencoder",
)
CLASSICAL_FAMILIES = frozenset({"statistical_reference", "isolation_forest", "one_class_svm"})
STOCHASTIC_FAMILIES = frozenset({"isolation_forest", "dense_autoencoder"})
MODEL_SEEDS = (0, 1, 2, 3, 4)
DETERMINISTIC_SEED = 0
EXPECTED_EXPERIMENT_COUNT = 156
LEDGER_EXPECTED_START = 40
EXPECTED_TEST_SEGMENTS = {"RoboticArm": 466, "BrushlessMotor": 312}

CLASSICAL_FEATURE_VERSION = "phase4-single-modality-segment-median-engineered-v1"
AUTOENCODER_FEATURE_VERSION = "phase4-single-modality-log-power-spectrum-v1"
REQUESTED_METRICS = ("auroc", "average_precision", "standardized_pauroc_fpr_0_10", "threshold_metrics")


def expected_seeds(family: str) -> tuple[int, ...]:
    if family not in FAMILIES:
        raise ValueError(f"model family is not part of the frozen benchmark: {family!r}")
    return MODEL_SEEDS if family in STOCHASTIC_FAMILIES else (DETERMINISTIC_SEED,)


@dataclass(frozen=True, order=True)
class PlannedExperiment:
    machine: str
    regime: str
    modality: str
    family: str
    seed: int


def experiment_plan() -> tuple[PlannedExperiment, ...]:
    plan = tuple(
        PlannedExperiment(machine, regime, modality, family, seed)
        for machine in MACHINES
        for regime in REGIMES
        for modality in MODALITIES
        for family in FAMILIES
        for seed in expected_seeds(family)
    )
    if len(plan) != EXPECTED_EXPERIMENT_COUNT:  # pragma: no cover - frozen arithmetic
        raise AssertionError("frozen experiment count drifted")
    return plan


def modality_feature_columns(columns: Iterable[str], modality: str) -> tuple[str, ...]:
    if modality not in MODALITIES:
        raise ValueError(f"unknown modality {modality!r}")
    prefix = f"{modality}__"
    selected = tuple(sorted(str(column) for column in columns if str(column).startswith(prefix)))
    if not selected:
        raise ValueError(f"no engineered features found for modality {modality!r}")
    return selected


def aggregate_window_scores_by_segment(meta: pd.DataFrame, window_scores: np.ndarray) -> pd.DataFrame:
    """Frozen Protocol-B rule: segment score = median of its window scores."""

    scores = np.asarray(window_scores, dtype=np.float64).reshape(-1)
    if len(scores) != len(meta) or not np.all(np.isfinite(scores)):
        raise ValueError("one finite window score is required per window row")
    frame = meta.loc[:, ["segment_id", "parent_recording", "domain", "anomaly_label"]].copy()
    frame["score"] = scores
    for column in ("parent_recording", "domain", "anomaly_label"):
        if (frame.groupby("segment_id")[column].nunique() > 1).any():
            raise ValueError(f"window provenance column {column!r} varies within a segment")
    grouped = frame.groupby("segment_id", sort=True)
    table = grouped.agg(
        parent_recording=("parent_recording", "first"),
        domain=("domain", "first"),
        anomaly_label=("anomaly_label", "first"),
        score=("score", "median"),
        window_count=("score", "size"),
    ).reset_index()
    table["window_count"] = table["window_count"].astype(int)
    return table


def dependency_environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": package_version("numpy"),
        "pandas": package_version("pandas"),
        "scikit-learn": package_version("scikit-learn"),
        "scipy": package_version("scipy"),
        "joblib": package_version("joblib"),
        "torch": package_version("torch"),
        "h5py": package_version("h5py"),
    }


def standardize_float32(standardizer: FittedChannelStandardizer, X: np.ndarray, chunk: int = 8192) -> np.ndarray:
    output = np.empty(X.shape, dtype=np.float32)
    for start in range(0, len(X), chunk):
        output[start:start + chunk] = standardizer.transform(X[start:start + chunk]).astype(np.float32)
    return output


def write_bytes_durably(path: Path, data: bytes, *, attempts: int = 6) -> None:
    """``Path.write_bytes`` with a short retry: Windows file scanners briefly lock new files."""

    for attempt in range(attempts):
        try:
            Path(path).write_bytes(data)
            return
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.25 * (attempt + 1))


def normalize_text_artifacts_to_lf(directory: Path, *, names: Iterable[str]) -> None:
    """Rewrite text artifacts with LF so working-tree and git-blob hashes agree (A6)."""

    for name in names:
        path = directory / name
        if path.exists():
            write_bytes_durably(path, canonical_bytes(path))


BUNDLE_TEXT_FILES = (
    "config.json", "provenance.json", "validation_summary.json", "threshold.json",
    "validation_scores.csv",
)


def rewrite_bundle_manifest(bundle: Path) -> None:
    """Normalise a pre-test bundle to LF and re-hash its manifest canonically."""

    normalize_text_artifacts_to_lf(bundle, names=BUNDLE_TEXT_FILES)
    manifest_path = bundle / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = {name: canonical_sha256(bundle / name) for name in manifest["files"]}
    manifest["hash_form"] = "canonical_sha256 (LF-normalised text)"
    write_bytes_durably(manifest_path, (canonical_json(manifest) + "\n").encode("utf-8"))


# ------------------------------------------------------------------ model store

class ModelStore:
    """Hash-manifested persistence of fitted models outside the reports tree."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def directory(self, experiment_id: str) -> Path:
        return self.root / experiment_id

    def write(
        self,
        *,
        experiment_id: str,
        family: str,
        model_id: str,
        model: Any,
        standardizer: FittedChannelStandardizer,
    ) -> dict[str, Any]:
        directory = self.directory(experiment_id)
        if directory.exists():
            raise FileExistsError(f"model artifact already exists: {directory}")
        directory.mkdir(parents=True)
        model_path = directory / ("model.pth" if family == "dense_autoencoder" else "model.joblib")
        model.save(model_path)
        standardizer_path = directory / "standardizer.joblib"
        joblib.dump(standardizer, standardizer_path)
        files = {path.name: canonical_sha256(path) for path in (model_path, standardizer_path)}
        manifest = {
            "schema_version": PHASE4_SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "family": family,
            "model_id": model_id,
            "files": files,
        }
        manifest_path = directory / "model_manifest.json"
        write_bytes_durably(manifest_path, (canonical_json(manifest) + "\n").encode("utf-8"))
        manifest["model_manifest_sha256"] = canonical_sha256(manifest_path)
        return manifest

    def verify(self, experiment_id: str, *, expected_manifest_sha256: str) -> dict[str, Any]:
        directory = self.directory(experiment_id)
        manifest_path = directory / "model_manifest.json"
        if canonical_sha256(manifest_path) != expected_manifest_sha256:
            raise ValueError(f"model manifest hash mismatch for {experiment_id}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("experiment_id") != experiment_id:
            raise ValueError("model manifest belongs to another experiment")
        for name, digest in manifest["files"].items():
            if canonical_sha256(directory / name) != digest:
                raise ValueError(f"model artifact {name} hash mismatch for {experiment_id}")
        return manifest

    def load(self, experiment_id: str, *, expected_manifest_sha256: str) -> tuple[Any, FittedChannelStandardizer, dict]:
        manifest = self.verify(experiment_id, expected_manifest_sha256=expected_manifest_sha256)
        directory = self.directory(experiment_id)
        family = manifest["family"]
        if family == "dense_autoencoder":
            model = DenseAutoencoder.load(directory / "model.pth")
        elif family == "mahalanobis_shrinkage":
            model = ShrinkageMahalanobis.load(directory / "model.joblib")
        elif family in CLASSICAL_FAMILIES:
            model = ClassicalAnomalyBaseline.load(directory / "model.joblib")
        else:
            raise ValueError(f"unknown stored family {family!r}")
        if model.model_id != manifest["model_id"]:
            raise ValueError("stored model identity differs from its manifest")
        standardizer = joblib.load(directory / "standardizer.joblib")
        return model, standardizer, manifest


# ------------------------------------------------------------------ window cohorts

def _window_frame(meta: pd.DataFrame, config: PreprocessingConfig, machine: str) -> pd.DataFrame:
    frame = meta.copy()
    frame["machine"] = machine
    frame["protocol"] = MECHASIGHT_PROTOCOL
    frame["regime"] = config.regime
    frame["split_version"] = config.split_version
    frame["preprocessing_version"] = config.version
    frame["preprocessing_config_id"] = config.config_id
    frame["spectral_row"] = np.arange(len(frame), dtype=np.float64)
    return frame


def validated_training_window_rows(
    meta: pd.DataFrame, contract: TrainingCohortContract, config: PreprocessingConfig
) -> np.ndarray:
    """Run the frozen training contract on window provenance; return row order."""

    data = load_estimator_training_data(
        _window_frame(meta, config, contract.machine), contract, feature_columns=("spectral_row",)
    )
    return data.X[:, 0].astype(int)


def validated_validation_window_rows(
    meta: pd.DataFrame, contract: SourceValidationCohortContract, config: PreprocessingConfig
) -> np.ndarray:
    data = load_source_validation_data(
        _window_frame(meta, config, contract.machine), contract, feature_columns=("spectral_row",)
    )
    return data.X[:, 0].astype(int)


# ------------------------------------------------------------------ pre-test runner

def _build_model(family: str, seed: int, *, input_dim: int | None = None):
    if family in CLASSICAL_FAMILIES:
        return ClassicalAnomalyBaseline(primary_spec(family), seed=seed)
    if family == "mahalanobis_shrinkage":
        return ShrinkageMahalanobis(seed=seed)
    if family == "dense_autoencoder":
        assert input_dim is not None
        return DenseAutoencoder(input_dim=input_dim, seed=seed)
    raise ValueError(f"unknown family {family!r}")


def _model_hyperparameters(family: str, modality: str, *, input_dim: int | None, code_source_hash: str) -> dict[str, Any]:
    if family in CLASSICAL_FAMILIES:
        spec = primary_spec(family)
        payload = {
            **dict(spec.hyperparameters),
            "implementation": spec.implementation,
            "implementation_version": "phase3.1-classical-baselines-v1",
            "model_spec_id": spec.identity,
            "stochastic": spec.stochastic,
            "score_transform": spec.score_transform,
            "training_feature_aggregation": "segment_median",
            "feature_standardization": "training_cohort_zscore",
        }
    elif family == "mahalanobis_shrinkage":
        payload = {
            **dict(MAHALANOBIS_SPEC.hyperparameters),
            "implementation": MAHALANOBIS_SPEC.implementation,
            "implementation_version": MAHALANOBIS_IMPLEMENTATION_VERSION,
            "model_spec_id": MAHALANOBIS_SPEC.identity,
            "stochastic": False,
            "score_transform": MAHALANOBIS_SPEC.score_transform,
            "training_feature_aggregation": "segment_median",
            "feature_standardization": "training_cohort_zscore",
        }
    else:
        payload = {
            **DENSE_AE_SPEC.to_dict(),
            "implementation": "src.models.dense_autoencoder.DenseAutoencoder",
            "implementation_version": AUTOENCODER_IMPLEMENTATION_VERSION,
            "stochastic": True,
            "input_dim": int(input_dim or 0),
            "window_score": "mean squared reconstruction error of the standardized log-power window",
            "segment_score": "median of window scores",
            "input_standardization": "training_cohort_per_bin_zscore",
            "early_stopping_cohort": "source_validation_window_mse",
        }
    payload.update({
        "modality": modality,
        "stage": STAGE_NAME,
        "stage_schema_version": PHASE4_SCHEMA_VERSION,
        "standardization_epsilon": EPSILON,
        "phase4_source_hash": code_source_hash,
    })
    return payload


def run_one_pretest_experiment(
    *,
    machine: str,
    regime: str,
    modality: str,
    family: str,
    seed: int,
    feature_root: str | Path,
    spectral_cache_root: str | Path,
    model_root: str | Path,
    result_root: str | Path,
    split_manifest_path: str | Path,
    preprocessing_config_path: str | Path,
    code_git_sha: str,
    code_source_hash: str,
) -> dict[str, Any]:
    """Fit one frozen configuration; source-validation scores and q90 only."""

    if seed not in expected_seeds(family):
        raise ValueError(f"{family} does not run seed {seed}")
    if modality not in MODALITIES:
        raise ValueError(f"unknown modality {modality!r}")
    config = load_config(preprocessing_config_path)
    if config.regime != regime:
        raise ValueError("preprocessing config regime differs from the requested regime")
    assignments = pd.read_csv(split_manifest_path)
    train_contract = TrainingCohortContract.from_frozen_assignments(assignments, config, machine=machine)
    validation_contract = SourceValidationCohortContract.from_frozen_assignments(
        assignments, config, machine=machine
    )
    data_provenance: dict[str, str] = {
        "split_manifest_hash": file_sha256(split_manifest_path),
    }
    started = perf_counter()

    if family == "dense_autoencoder":
        input_dim = spectral_input_dim(config, machine, modality)
        train_meta_json = load_spectral_cache_metadata(
            spectral_cache_root, machine=machine, regime=regime, role="training"
        )
        validation_meta_json = load_spectral_cache_metadata(
            spectral_cache_root, machine=machine, regime=regime, role="source_validation"
        )
        X_train_raw, train_meta = load_spectral_cache(
            spectral_cache_path(spectral_cache_root, machine=machine, regime=regime, role="training"),
            modality=modality, expected_split="train",
        )
        X_validation_raw, validation_meta = load_spectral_cache(
            spectral_cache_path(spectral_cache_root, machine=machine, regime=regime, role="source_validation"),
            modality=modality, expected_split="validation",
        )
        train_rows = validated_training_window_rows(train_meta, train_contract, config)
        validation_rows = validated_validation_window_rows(validation_meta, validation_contract, config)
        X_train_raw = X_train_raw[train_rows]
        X_validation_raw = X_validation_raw[validation_rows]
        train_meta = train_meta.iloc[train_rows].reset_index(drop=True)
        validation_meta = validation_meta.iloc[validation_rows].reset_index(drop=True)
        training_segment_ids = tuple(sorted(set(train_meta["segment_id"])))
        standardizer = FittedChannelStandardizer.fit(
            [X_train_raw.astype(np.float64)], split="train", segment_ids=training_segment_ids
        )
        X_train = standardize_float32(standardizer, X_train_raw)
        X_validation = standardize_float32(standardizer, X_validation_raw)
        model = _build_model(family, seed, input_dim=input_dim).fit(X_train, X_validation)
        score_started = perf_counter()
        window_scores = model.score_windows(X_validation)
        table = aggregate_window_scores_by_segment(validation_meta, window_scores)
        score_seconds = perf_counter() - score_started
        feature_count = input_dim
        feature_representation = (
            f"single_modality[{modality}]:log_power_spectrum_window({SPECTRAL_REPRESENTATION_VERSION})"
        )
        feature_version = AUTOENCODER_FEATURE_VERSION
        normalization_strategy = "train_only_per_bin_zscore_of_log_power_spectrum"
        training_windows = int(len(X_train))
        validation_windows = int(len(X_validation))
        data_provenance.update({
            "training_cohort_hash": ordered_id_hash(training_segment_ids),
            "source_validation_cohort_hash": ordered_id_hash(tuple(sorted(set(validation_meta["segment_id"])))),
            "training_spectral_cache_hash": train_meta_json["cache_sha256"],
            "source_validation_spectral_cache_hash": validation_meta_json["cache_sha256"],
        })
        training_history = dict(model.history)
    else:
        train_frame, train_meta_json = load_feature_artifact(
            feature_root, machine=machine, regime=regime, role="training"
        )
        validation_frame, validation_meta_json = load_feature_artifact(
            feature_root, machine=machine, regime=regime, role="source_validation"
        )
        all_features = phase31_feature_columns(train_frame)
        if phase31_feature_columns(validation_frame) != all_features:
            raise ValueError("training and source-validation feature schemas differ")
        features = modality_feature_columns(all_features, modality)
        training = load_estimator_training_data(train_frame, train_contract, feature_columns=features)
        fitted = fit_training_standardizer(train_frame, train_contract, feature_columns=features)
        standardizer = fitted.standardizer
        validation = load_source_validation_data(
            validation_frame, validation_contract, feature_columns=features
        )
        X_train = standardizer.transform(training.X)
        X_validation = standardizer.transform(validation.X)
        model = _build_model(family, seed).fit(X_train)
        score_started = perf_counter()
        validation_scores = model.score(X_validation)
        score_seconds = perf_counter() - score_started
        table = pd.DataFrame({
            "segment_id": validation.segment_ids,
            "parent_recording": validation.parent_recordings,
            "domain": ("source",) * len(validation.segment_ids),
            "anomaly_label": validation.anomaly_labels,
            "score": validation_scores,
            "window_count": validation_frame.set_index("segment_id").loc[list(validation.segment_ids), "window_count"].to_numpy(),
        }).sort_values("segment_id").reset_index(drop=True)
        feature_count = len(features)
        input_dim = None
        feature_representation = (
            f"single_modality[{modality}]:median_by_segment(existing_100ms_nonoverlap_time_and_frequency_features)"
        )
        feature_version = CLASSICAL_FEATURE_VERSION
        normalization_strategy = "train_only_feature_zscore_after_segment_median"
        training_windows = int(train_frame["window_count"].sum())
        validation_windows = int(validation_frame["window_count"].sum())
        data_provenance.update({
            "training_cohort_hash": ordered_id_hash(training.segment_ids),
            "source_validation_cohort_hash": ordered_id_hash(validation.segment_ids),
            "training_feature_artifact_hash": train_meta_json["artifact_sha256"],
            "source_validation_feature_artifact_hash": validation_meta_json["artifact_sha256"],
        })
        training_history = {}

    experiment = ExperimentConfig(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        machine=machine,
        protocol=config.protocol,
        regime=regime,
        model_family=family,
        model_hyperparameters=_model_hyperparameters(
            family, modality, input_dim=input_dim, code_source_hash=code_source_hash
        ),
        model_seed=seed,
        preprocessing_config_id=config.config_id,
        preprocessing_config_hash=file_sha256(preprocessing_config_path),
        split_version=config.split_version,
        split_manifest_hash=file_sha256(split_manifest_path),
        dataset_release_identity=DATASET_RELEASE_IDENTITY,
        feature_representation=feature_representation,
        feature_version=feature_version,
        window_length_s=config.window_length_s,
        stride_s=config.stride_s,
        normalization_strategy=normalization_strategy,
        code_git_sha=code_git_sha,
        dependency_environment=dependency_environment(),
    )
    score_provenance = ScoreProvenance(
        experiment_id=experiment.experiment_id,
        machine=machine,
        protocol=config.protocol,
        regime=regime,
        model_id=model.model_id,
        model_family=family,
        model_seed=seed,
        preprocessing_config_id=config.config_id,
        preprocessing_config_hash=file_sha256(preprocessing_config_path),
        split_version=config.split_version,
        score_direction=SCORE_DIRECTION,
        data_provenance=data_provenance,
        config_provenance={
            "code_git_sha": code_git_sha,
            "phase4_source_hash": code_source_hash,
            "stage_schema_version": PHASE4_SCHEMA_VERSION,
            "modality": modality,
        },
    )
    batch = ScoreBatch(
        provenance=score_provenance,
        scores=table["score"].to_numpy(),
        segment_ids=tuple(table["segment_id"].astype(str)),
        parent_recording_ids=tuple(table["parent_recording"].astype(str)),
        cohort_roles=("source_validation",) * len(table),
        domains=tuple(table["domain"].astype(str)),
        anomaly_labels=tuple(table["anomaly_label"].astype(str)),
    )
    EvaluationGate(experiment, score_provenance).evaluate_validation(batch)
    threshold = select_protocol_b_threshold(batch, expected_provenance=score_provenance)

    model_manifest = ModelStore(model_root).write(
        experiment_id=experiment.experiment_id,
        family=family,
        model_id=model.model_id,
        model=model,
        standardizer=standardizer,
    )
    scores = batch.scores
    summary = {
        "evaluation_stage": PHASE4_EVALUATION_STAGE,
        "stage": STAGE_NAME,
        "test_evaluated": False,
        "descriptive_only_no_model_ranking": True,
        "modality": modality,
        "training_segment_count": len(train_contract.authorized_segment_ids),
        "training_window_count": training_windows,
        "source_validation_segment_count": len(validation_contract.segment_ids),
        "source_validation_window_count": validation_windows,
        "feature_count": int(feature_count),
        "score_count": int(len(scores)),
        "finite_score_count": int(np.isfinite(scores).sum()),
        "score_min": float(np.min(scores)),
        "score_q05": float(np.quantile(scores, 0.05, method="higher")),
        "score_q25": float(np.quantile(scores, 0.25, method="higher")),
        "score_median": float(np.median(scores)),
        "score_q75": float(np.quantile(scores, 0.75, method="higher")),
        "score_max": float(np.max(scores)),
        "score_q90": float(threshold.value),
        "score_q95": float(np.quantile(scores, 0.95, method="higher")),
        "fit_seconds": float(getattr(model, "fit_seconds_", None) or training_history.get("fit_seconds", 0.0)),
        "validation_scoring_seconds": float(score_seconds),
        "total_seconds": float(perf_counter() - started),
        "model_saved": True,
        "model_manifest_sha256": model_manifest["model_manifest_sha256"],
        "training_history": training_history,
    }
    table_out = table.assign(cohort_role="source_validation")[
        ["segment_id", "parent_recording", "cohort_role", "domain", "anomaly_label", "score", "window_count"]
    ]
    bundle_path = PretestResultStore(result_root).write_bundle(
        config=experiment,
        provenance={
            "evaluation_stage": PHASE4_EVALUATION_STAGE,
            "stage": STAGE_NAME,
            "stage_schema_version": PHASE4_SCHEMA_VERSION,
            "test_evaluated": False,
            "score_direction": SCORE_DIRECTION,
            "modality": modality,
            "training_cohort_hash": data_provenance["training_cohort_hash"],
            "validation_cohort_hash": data_provenance["source_validation_cohort_hash"],
            "training_contract_hash": train_contract.frozen_manifest_hash,
            "validation_contract_hash": validation_contract.frozen_manifest_hash,
            "score_provenance": asdict(score_provenance),
            "model_manifest": model_manifest,
        },
        summary=summary,
        threshold=threshold,
        validation_scores=table_out,
    )
    rewrite_bundle_manifest(bundle_path)
    bundle = bundle_path
    return {
        "experiment_id": experiment.experiment_id,
        "model_id": model.model_id,
        "machine": machine,
        "regime": regime,
        "modality": modality,
        "family": family,
        "seed": seed,
        "stochastic": family in STOCHASTIC_FAMILIES,
        "feature_count": int(feature_count),
        "threshold_q90": float(threshold.value),
        "fit_seconds": summary["fit_seconds"],
        "total_seconds": summary["total_seconds"],
        "epochs_run": training_history.get("epochs_run"),
        "best_epoch": training_history.get("best_epoch"),
        "best_validation_window_mse": training_history.get("best_validation_window_mse"),
        "stopped_early": training_history.get("stopped_early"),
        "bundle": str(bundle),
        "test_evaluated": False,
    }


# ------------------------------------------------------------------ frozen bundles

@dataclass(frozen=True)
class FrozenBundle:
    experiment_id: str
    config: ExperimentConfig
    modality: str
    threshold: ThresholdSelection
    score_provenance: ScoreProvenance
    model_manifest_sha256: str
    bundle_path: Path


def load_frozen_bundle(path: str | Path) -> FrozenBundle:
    bundle = Path(path)
    manifest = json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["files"].items():
        if canonical_sha256(bundle / name) != digest:
            raise ValueError(f"bundle file {name} hash mismatch in {bundle.name}")
    config = experiment_config_from_dict(json.loads((bundle / "config.json").read_text(encoding="utf-8")))
    if config.experiment_id != manifest["experiment_id"] or bundle.name != config.experiment_id:
        raise ValueError("bundle identity mismatch")
    provenance = json.loads((bundle / "provenance.json").read_text(encoding="utf-8"))
    if provenance.get("test_evaluated") is not False or provenance.get("stage") != STAGE_NAME:
        raise ValueError("bundle is not a Phase-4 pre-test bundle")
    threshold_payload = json.loads((bundle / "threshold.json").read_text(encoding="utf-8"))
    threshold_payload["source_segment_ids"] = tuple(threshold_payload["source_segment_ids"])
    threshold = ThresholdSelection(**threshold_payload)
    if threshold.source_experiment_id != config.experiment_id:
        raise ValueError("bundle threshold belongs to another experiment")
    provenance_payload = dict(provenance["score_provenance"])
    score_provenance = ScoreProvenance(**provenance_payload)
    return FrozenBundle(
        experiment_id=config.experiment_id,
        config=config,
        modality=provenance["modality"],
        threshold=threshold,
        score_provenance=score_provenance,
        model_manifest_sha256=provenance["model_manifest"]["model_manifest_sha256"],
        bundle_path=bundle,
    )


def load_frozen_bundles(result_root: str | Path) -> dict[str, FrozenBundle]:
    root = Path(result_root)
    bundles = {}
    for directory in sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")):
        bundle = load_frozen_bundle(directory)
        bundles[bundle.experiment_id] = bundle
    return bundles


def assert_complete_plan(bundles: Mapping[str, FrozenBundle]) -> None:
    observed = {
        PlannedExperiment(
            b.config.machine, b.config.regime, b.modality, b.config.model_family, b.config.model_seed
        )
        for b in bundles.values()
    }
    expected = set(experiment_plan())
    if observed != expected or len(bundles) != EXPECTED_EXPERIMENT_COUNT:
        missing = sorted(expected - observed)[:5]
        extra = sorted(observed - expected)[:5]
        raise ValueError(
            f"frozen bundles do not cover the 156-experiment plan exactly "
            f"(missing {missing}, unexpected {extra})"
        )


# ------------------------------------------------------------------ protected batch

@dataclass(frozen=True)
class PlannedBatch:
    batch_id: str
    experiment_ids: tuple[str, ...]
    test_cohort_hashes_by_machine: Mapping[str, str]
    code_git_sha: str
    reporting_policy_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "experiment_count": len(self.experiment_ids),
            "experiment_ids": list(self.experiment_ids),
            "test_cohort_hashes_by_machine": dict(self.test_cohort_hashes_by_machine),
            "code_git_sha": self.code_git_sha,
            "reporting_policy_id": self.reporting_policy_id,
            "ledger_expected_start": LEDGER_EXPECTED_START,
        }


def plan_protected_batch(
    bundles: Mapping[str, FrozenBundle],
    *,
    test_cohort_hashes_by_machine: Mapping[str, str],
    code_git_sha: str,
) -> PlannedBatch:
    assert_complete_plan(bundles)
    if set(test_cohort_hashes_by_machine) != set(MACHINES):
        raise ValueError("a frozen test cohort hash is required per machine")
    ids = tuple(sorted(bundles))
    batch_id = "p4-test-batch-" + content_sha256({
        "experiment_ids": list(ids),
        "cohort_hashes": {
            experiment_id: test_cohort_hashes_by_machine[bundles[experiment_id].config.machine]
            for experiment_id in ids
        },
        "policy": REPORTING_POLICY_ID,
        "code": code_git_sha,
    })[:24]
    return PlannedBatch(
        batch_id=batch_id,
        experiment_ids=ids,
        test_cohort_hashes_by_machine=dict(test_cohort_hashes_by_machine),
        code_git_sha=code_git_sha,
        reporting_policy_id=REPORTING_POLICY_ID,
    )


@dataclass(frozen=True)
class BatchAuthorization:
    batch_id: str
    experiment_ids: frozenset[str]
    experiment_test_cohort_hashes: Mapping[str, str]
    ledger_start_count: int
    ledger_end_count: int


def authorize_protected_batch(
    *,
    ledger: TestLookLedger,
    planned: PlannedBatch,
    bundles: Mapping[str, FrozenBundle],
    expected_ledger_start: int = LEDGER_EXPECTED_START,
) -> BatchAuthorization:
    """Append all 156 looks atomically before any scorer is invoked."""

    if len(planned.experiment_ids) != EXPECTED_EXPERIMENT_COUNT:
        raise ValueError("protected batch must contain exactly 156 experiments")
    if set(planned.experiment_ids) != set(bundles):
        raise ValueError("planned batch and frozen bundles differ")
    if len(ledger.read()) != expected_ledger_start:
        raise RuntimeError("unexpected test-look ledger starting count")
    timestamp = datetime.now(timezone.utc).isoformat()
    cohort_hashes = {
        experiment_id: planned.test_cohort_hashes_by_machine[bundles[experiment_id].config.machine]
        for experiment_id in planned.experiment_ids
    }
    reason = (
        f"{TEST_REASON}; batch={planned.batch_id}; policy={planned.reporting_policy_id}; "
        "score_all_then_analyse_once; no metric during scoring"
    )
    records = tuple(
        TestLookRecord(
            timestamp_utc=timestamp,
            experiment_id=experiment_id,
            git_sha=planned.code_git_sha,
            machine=bundles[experiment_id].config.machine,
            protocol=bundles[experiment_id].config.protocol,
            regime=bundles[experiment_id].config.regime,
            model_config_identity=bundles[experiment_id].score_provenance.model_id,
            test_cohort_hash=cohort_hashes[experiment_id],
            reason=reason,
            metrics_requested=REQUESTED_METRICS,
        )
        for experiment_id in planned.experiment_ids
    )
    ledger.append_many_atomic(records, expected_existing_count=expected_ledger_start)
    return BatchAuthorization(
        batch_id=planned.batch_id,
        experiment_ids=frozenset(planned.experiment_ids),
        experiment_test_cohort_hashes=cohort_hashes,
        ledger_start_count=expected_ledger_start,
        ledger_end_count=expected_ledger_start + len(records),
    )


# ------------------------------------------------------------------ label-free test inputs

@dataclass(frozen=True)
class LabelFreeTestFeatures:
    """Segment feature matrix for scoring; anomaly labels are deliberately absent."""

    frame: pd.DataFrame
    feature_names: tuple[str, ...]
    cohort_hash: str
    machine: str
    regime: str


LABEL_COLUMNS = ("anomaly_label", "is_anomaly", "label")


def load_label_free_test_features(
    root: str | Path, *, machine: str, regime: str, contract: CommonTestCohortContract
) -> LabelFreeTestFeatures:
    require_protected_read("label-free common-test features")
    artifact_path, metadata_path = common_test_artifact_paths(root, machine=machine, regime=regime)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if file_sha256(artifact_path) != metadata.get("artifact_sha256"):
        raise ValueError("common-test feature artifact hash mismatch")
    frame = pd.read_parquet(artifact_path)
    features = phase31_feature_columns(frame)
    keep = ["segment_id", "parent_recording", "domain", "window_count", *features]
    frame = frame.loc[:, keep].sort_values("segment_id").reset_index(drop=True)
    segment_ids = tuple(frame["segment_id"].astype(str))
    if frozenset(segment_ids) != contract.segment_ids or len(segment_ids) != len(set(segment_ids)):
        raise ValueError("common-test segment identities do not match the frozen cohort")
    cohort_hash = ordered_id_hash(segment_ids)
    if cohort_hash != metadata.get("cohort_hash"):
        raise ValueError("common-test artifact cohort hash mismatch")
    if set(frame.columns) & set(LABEL_COLUMNS):
        raise AssertionError("label column survived label-free loading")
    return LabelFreeTestFeatures(frame, features, cohort_hash, machine, regime)


@dataclass(frozen=True)
class LabelFreeTestWindows:
    X: dict[str, np.ndarray]
    meta: pd.DataFrame
    cohort_hash: str
    machine: str
    regime: str


def load_label_free_test_windows(
    cache_root: str | Path, *, machine: str, regime: str, contract: CommonTestCohortContract
) -> LabelFreeTestWindows:
    require_protected_read("label-free common-test spectral windows")
    metadata = load_spectral_cache_metadata(cache_root, machine=machine, regime=regime, role="common_evaluation")
    path = spectral_cache_path(cache_root, machine=machine, regime=regime, role="common_evaluation")
    matrices = {}
    meta = None
    for modality in MODALITIES:
        X, frame = load_spectral_cache(path, modality=modality, expected_split="test")
        matrices[modality] = X
        meta = frame if meta is None else meta
    assert meta is not None
    meta = meta.drop(columns=[column for column in LABEL_COLUMNS if column in meta.columns])
    segment_ids = tuple(sorted(set(meta["segment_id"])))
    if frozenset(segment_ids) != contract.segment_ids:
        raise ValueError("common-test window cohort does not match the frozen cohort")
    if set(meta["data_pool"]) != {"common_evaluation"} or set(meta["split"]) != {"test"}:
        raise ValueError("common-test window cache carries the wrong role")
    cohort_hash = ordered_id_hash(segment_ids)
    if cohort_hash != metadata["segment_cohort_hash"]:
        raise ValueError("common-test window cache cohort hash mismatch")
    return LabelFreeTestWindows(matrices, meta, cohort_hash, machine, regime)


# ------------------------------------------------------------------ score all

SCORE_COLUMNS = ("segment_id", "parent_recording", "domain", "score", "window_count")


def score_one_experiment(
    bundle: FrozenBundle,
    *,
    model_store: ModelStore,
    test_features: LabelFreeTestFeatures | None,
    test_windows: LabelFreeTestWindows | None,
    authorization: BatchAuthorization,
) -> pd.DataFrame:
    if bundle.experiment_id not in authorization.experiment_ids:
        raise ValueError("experiment lacks durable batch authorization")
    if authorization.ledger_end_count - authorization.ledger_start_count != EXPECTED_EXPERIMENT_COUNT:
        raise ValueError("batch authorization receipt is incomplete")
    model, standardizer, _manifest = model_store.load(
        bundle.experiment_id, expected_manifest_sha256=bundle.model_manifest_sha256
    )
    family = bundle.config.model_family
    if family == "dense_autoencoder":
        if test_windows is None:
            raise ValueError("autoencoder scoring requires the common-test spectral cache")
        if test_windows.cohort_hash != authorization.experiment_test_cohort_hashes[bundle.experiment_id]:
            raise ValueError("authorized and supplied test cohort hashes differ")
        X = standardize_float32(standardizer, test_windows.X[bundle.modality])
        window_scores = model.score_windows(X)
        meta = test_windows.meta.assign(anomaly_label="withheld")
        table = aggregate_window_scores_by_segment(meta, window_scores).drop(columns=["anomaly_label"])
    else:
        if test_features is None:
            raise ValueError("classical scoring requires the common-test feature artifact")
        if test_features.cohort_hash != authorization.experiment_test_cohort_hashes[bundle.experiment_id]:
            raise ValueError("authorized and supplied test cohort hashes differ")
        features = modality_feature_columns(test_features.feature_names, bundle.modality)
        X = standardizer.transform(test_features.frame.loc[:, list(features)].to_numpy(dtype=np.float64))
        scores = model.score(X)
        table = pd.DataFrame({
            "segment_id": test_features.frame["segment_id"].astype(str),
            "parent_recording": test_features.frame["parent_recording"].astype(str),
            "domain": test_features.frame["domain"].astype(str),
            "score": scores,
            "window_count": test_features.frame["window_count"].astype(int),
        })
    table = table.loc[:, list(SCORE_COLUMNS)].sort_values("segment_id").reset_index(drop=True)
    if not np.all(np.isfinite(table["score"])) or len(table) != EXPECTED_TEST_SEGMENTS[bundle.config.machine]:
        raise ValueError("scorer returned an invalid score table")
    if ordered_id_hash(tuple(table["segment_id"])) != authorization.experiment_test_cohort_hashes[bundle.experiment_id]:
        raise ValueError("score table cohort differs from the authorized test cohort")
    return table


class ScoreStaging:
    """Sealed staging area: score tables and their hashes, nothing derived."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.scores = self.root / "scores"
        self.manifest_path = self.root / "score_manifest.json"
        self.state_path = self.root / "state.json"

    def open(self, authorization: BatchAuthorization) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            raise FileExistsError("score staging already opened; a repeat needs a new versioned identity")
        self.scores.mkdir(exist_ok=True)
        self._write_manifest({
            "schema_version": PHASE4_SCHEMA_VERSION,
            "batch_id": authorization.batch_id,
            "expected_count": EXPECTED_EXPERIMENT_COUNT,
            "ledger_start_count": authorization.ledger_start_count,
            "ledger_end_count": authorization.ledger_end_count,
            "tables": {},
            "metrics_computed": False,
        })

    def _write_manifest(self, payload: dict) -> None:
        temporary = self.manifest_path.with_suffix(".tmp")
        write_bytes_durably(temporary, (canonical_json(payload) + "\n").encode("utf-8"))
        temporary.replace(self.manifest_path)

    def read_manifest(self) -> dict:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def write_scores(self, experiment_id: str, table: pd.DataFrame, *, cohort_hash: str) -> str:
        if tuple(table.columns) != SCORE_COLUMNS:
            raise ValueError("score table must carry exactly the label-free score columns")
        path = self.scores / f"{experiment_id}.csv"
        if path.exists():
            raise FileExistsError(f"score table already staged for {experiment_id}")
        table.to_csv(path, index=False, lineterminator="\n")
        digest = canonical_sha256(path)
        manifest = self.read_manifest()
        if experiment_id in manifest["tables"]:
            raise ValueError("experiment already recorded in the score manifest")
        manifest["tables"][experiment_id] = {"file": path.name, "sha256": digest, "cohort_hash": cohort_hash}
        self._write_manifest(manifest)
        return digest

    def verify_complete(self, expected_ids: Iterable[str]) -> dict:
        manifest = self.read_manifest()
        expected = set(expected_ids)
        if set(manifest["tables"]) != expected or len(manifest["tables"]) != EXPECTED_EXPERIMENT_COUNT:
            raise ValueError(
                f"score staging is incomplete: {len(manifest['tables'])} of {EXPECTED_EXPERIMENT_COUNT} tables"
            )
        for experiment_id, entry in manifest["tables"].items():
            path = self.scores / entry["file"]
            if canonical_sha256(path) != entry["sha256"]:
                raise ValueError(f"staged score table hash mismatch for {experiment_id}")
        return manifest

    def read_scores(self, experiment_id: str) -> pd.DataFrame:
        require_sealed_score_read(f"{experiment_id}.csv", reader="ScoreStaging.read_scores")
        frame = pd.read_csv(self.scores / f"{experiment_id}.csv", dtype={"segment_id": str, "parent_recording": str, "domain": str})
        if tuple(frame.columns) != SCORE_COLUMNS:
            raise ValueError("staged score table schema drifted")
        return frame


def score_all(
    *,
    bundles: Mapping[str, FrozenBundle],
    authorization: BatchAuthorization,
    model_store: ModelStore,
    test_features: Mapping[tuple[str, str], LabelFreeTestFeatures],
    test_windows: Mapping[tuple[str, str], LabelFreeTestWindows],
    staging: ScoreStaging,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Score every authorized bundle; write tables only.  No labels, no metrics."""

    if set(bundles) != authorization.experiment_ids:
        raise ValueError("bundles and authorization differ")
    manifest = staging.read_manifest()
    if manifest["batch_id"] != authorization.batch_id:
        raise ValueError("staging belongs to another batch")
    for experiment_id in sorted(bundles):
        bundle = bundles[experiment_id]
        key = (bundle.config.machine, bundle.config.regime)
        table = score_one_experiment(
            bundle,
            model_store=model_store,
            test_features=test_features.get(key),
            test_windows=test_windows.get(key),
            authorization=authorization,
        )
        digest = staging.write_scores(
            experiment_id, table,
            cohort_hash=authorization.experiment_test_cohort_hashes[experiment_id],
        )
        if progress is not None:
            progress({"experiment_id": experiment_id, "staged_sha256": digest})
    return staging.verify_complete(bundles)


# ------------------------------------------------------------------ reproduction check

def verify_bundle_reproduces_validation(
    bundle: FrozenBundle,
    *,
    model_store: ModelStore,
    feature_root: str | Path,
    spectral_cache_root: str | Path,
    split_manifest_path: str | Path,
    preprocessing_config_path: str | Path,
    atol: float = 1e-9,
) -> dict[str, Any]:
    """Reload the frozen artifact and re-score source validation; must match exactly."""

    config = load_config(preprocessing_config_path)
    machine, regime = bundle.config.machine, bundle.config.regime
    if config.regime != regime:
        raise ValueError("preprocessing config regime differs from the bundle regime")
    assignments = pd.read_csv(split_manifest_path)
    validation_contract = SourceValidationCohortContract.from_frozen_assignments(
        assignments, config, machine=machine
    )
    model, standardizer, _manifest = model_store.load(
        bundle.experiment_id, expected_manifest_sha256=bundle.model_manifest_sha256
    )
    recorded = pd.read_csv(
        bundle.bundle_path / "validation_scores.csv", dtype={"segment_id": str}
    ).sort_values("segment_id").reset_index(drop=True)
    if bundle.config.model_family == "dense_autoencoder":
        X_raw, meta = load_spectral_cache(
            spectral_cache_path(spectral_cache_root, machine=machine, regime=regime, role="source_validation"),
            modality=bundle.modality, expected_split="validation",
        )
        rows = validated_validation_window_rows(meta, validation_contract, config)
        X = standardize_float32(standardizer, X_raw[rows])
        table = aggregate_window_scores_by_segment(meta.iloc[rows].reset_index(drop=True), model.score_windows(X))
    else:
        frame, _meta = load_feature_artifact(feature_root, machine=machine, regime=regime, role="source_validation")
        features = modality_feature_columns(phase31_feature_columns(frame), bundle.modality)
        validation = load_source_validation_data(frame, validation_contract, feature_columns=features)
        table = pd.DataFrame({
            "segment_id": validation.segment_ids,
            "score": model.score(standardizer.transform(validation.X)),
        })
    table = table.sort_values("segment_id").reset_index(drop=True)
    if list(table["segment_id"]) != list(recorded["segment_id"]):
        raise ValueError(f"{bundle.experiment_id}: validation cohort differs on reload")
    difference = float(np.max(np.abs(table["score"].to_numpy() - recorded["score"].to_numpy())))
    if difference > atol:
        raise ValueError(
            f"{bundle.experiment_id}: reloaded model does not reproduce its frozen validation scores "
            f"(max abs difference {difference:.3e})"
        )
    threshold = float(np.quantile(table["score"].to_numpy(), 0.90, method="higher"))
    if abs(threshold - bundle.threshold.value) > atol:
        raise ValueError(f"{bundle.experiment_id}: reloaded q90 differs from the frozen threshold")
    return {
        "experiment_id": bundle.experiment_id,
        "family": bundle.config.model_family,
        "modality": bundle.modality,
        "max_abs_score_difference": difference,
        "threshold_reproduced": True,
    }
