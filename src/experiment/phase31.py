"""Phase-3.1 pre-test feature preparation and baseline execution.

This module has no common-test execution path.  Its public cohort guard accepts
only the complete frozen training cohort and the source-normal validation
cohort, and it rejects every other role before a preprocessing iterator or
model scorer can be invoked.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version as package_version
import json
from pathlib import Path
import platform
from time import perf_counter
from typing import Iterable, Iterator, Mapping

import numpy as np
import pandas as pd

from src.experiment.config import ExperimentConfig
from src.experiment.constants import (
    PHASE31_EVALUATION_STAGE,
    PHASE31_PREPROCESSING_POLICY_VERSION,
    SCORE_DIRECTION,
)
from src.experiment.data import (
    SourceValidationCohortContract,
    TrainingCohortContract,
    fit_training_standardizer,
    load_estimator_training_data,
    load_source_validation_data,
)
from src.experiment.gate import EvaluationGate
from src.experiment.pretest import PretestResultStore
from src.experiment.provenance import content_sha256, file_sha256, ordered_id_hash
from src.experiment.scores import (
    ScoreBatch,
    ScoreProvenance,
    select_protocol_b_threshold,
)
from src.models.classical import ClassicalAnomalyBaseline, primary_spec
from src.preprocessing.config import PreprocessingConfig, load_config
from src.preprocessing.normalization import EPSILON
from src.preprocessing.pipeline import (
    iter_processed_windows,
    load_raw_catalog,
    merge_protocol_b_assignments,
)


ACCEPTED_PHASE31_HEAD = "3bf046dd8edb29d99f1d246b33f88674a5b53a2d"
DATASET_RELEASE_IDENTITY = "IMAD-DS:zenodo-12665499:verified"
FEATURE_VERSION = "phase3.1-segment-median-engineered-v1"
PRIMARY_FEATURE_REPRESENTATION = (
    "median_by_segment(existing_100ms_nonoverlap_time_and_frequency_features)"
)
ALLOWED_PHASE31_ROLES = frozenset({"training", "source_validation"})
ROLE_TO_SPLIT = {"training": "train", "source_validation": "validation"}
DETERMINISTIC_SEED = 0
ISOLATION_FOREST_SEEDS = (0, 1, 2, 3, 4)


def assert_phase31_role(role: str) -> str:
    """Fail before data materialisation or scoring for every unauthorized role."""

    if role not in ALLOWED_PHASE31_ROLES:
        raise ValueError(
            "Phase-3.1 authorizes only training and source_validation; "
            f"refusing role {role!r} before materialisation/scoring"
        )
    return ROLE_TO_SPLIT[role]


def phase31_feature_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    """Return the explicit, stable engineered-feature schema."""

    columns = tuple(sorted(str(column) for column in frame.columns if "__" in str(column)))
    if not columns:
        raise ValueError("Phase-3.1 artifact contains no engineered feature columns")
    values = frame.loc[:, columns].to_numpy(dtype=np.float64, copy=False)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("Phase-3.1 engineered features must all be finite")
    return columns


def aggregate_windows_by_segment(
    windows: Iterable[tuple[Mapping, Mapping, Mapping[str, float]]],
    *,
    role: str,
) -> pd.DataFrame:
    """Median-aggregate one contiguous preprocessing stream per segment."""

    expected_split = assert_phase31_role(role)
    rows: list[dict] = []
    current_id: str | None = None
    current_record: Mapping | None = None
    current_features: list[Mapping[str, float]] = []

    def flush() -> None:
        nonlocal current_id, current_record, current_features
        if current_record is None:
            return
        names = tuple(sorted(current_features[0]))
        if any(tuple(sorted(item)) != names for item in current_features):
            raise ValueError("window feature schemas differ within a segment")
        matrix = np.asarray(
            [[float(item[name]) for name in names] for item in current_features],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"non-finite window feature in segment {current_id}")
        trace_keys = (
            "machine", "protocol", "regime", "split", "data_pool", "domain",
            "parent_recording", "segment_id", "anomaly_label",
            "preprocessing_config_id", "preprocessing_version", "split_version",
        )
        row = {key: current_record[key] for key in trace_keys}
        row["window_count"] = int(len(current_features))
        row.update({name: float(value) for name, value in zip(names, np.median(matrix, axis=0))})
        rows.append(row)

    for record, _signals, features in windows:
        if str(record.get("split")) != expected_split:
            raise ValueError("preprocessing stream contains an unauthorized split")
        segment_id = str(record.get("segment_id", ""))
        if not segment_id:
            raise ValueError("preprocessing stream lacks segment identity")
        if current_id is not None and segment_id != current_id:
            flush()
            current_features = []
        current_id = segment_id
        current_record = record
        current_features.append(features)
    flush()
    if not rows:
        raise ValueError("cannot build an empty Phase-3.1 feature artifact")
    frame = pd.DataFrame(rows).sort_values("segment_id").reset_index(drop=True)
    if frame["segment_id"].duplicated().any():
        raise ValueError("segment aggregation produced duplicate identities")
    phase31_feature_columns(frame)
    return frame


def feature_artifact_paths(
    root: str | Path, *, machine: str, regime: str, role: str
) -> tuple[Path, Path]:
    assert_phase31_role(role)
    base = Path(root) / machine / regime
    return base / f"{role}.parquet", base / f"{role}.metadata.json"


def prepare_feature_artifact(
    *,
    data_root: str | Path,
    split_manifest_path: str | Path,
    preprocessing_config_path: str | Path,
    artifact_root: str | Path,
    machine: str,
    role: str,
) -> dict:
    """Build one immutable training/validation segment-feature artifact."""

    split = assert_phase31_role(role)
    artifact_path, metadata_path = feature_artifact_paths(
        artifact_root, machine=machine, regime=load_config(preprocessing_config_path).regime,
        role=role,
    )
    conflicts = [path for path in (artifact_path, metadata_path) if path.exists()]
    if conflicts:
        raise FileExistsError("refusing to overwrite Phase-3.1 artifact: " + ", ".join(map(str, conflicts)))
    config = load_config(preprocessing_config_path)
    assignments = pd.read_csv(split_manifest_path)
    catalog = merge_protocol_b_assignments(
        load_raw_catalog(data_root), assignments, config.regime,
        split_version=config.split_version,
    )
    machine_catalog = catalog[catalog["machine"].eq(machine)].copy()
    windows = iter_processed_windows(machine_catalog, config, split=split)
    frame = aggregate_windows_by_segment(windows, role=role)

    if role == "training":
        contract = TrainingCohortContract.from_frozen_assignments(
            assignments, config, machine=machine
        )
        load_estimator_training_data(
            frame, contract, feature_columns=phase31_feature_columns(frame)
        )
    else:
        contract = SourceValidationCohortContract.from_frozen_assignments(
            assignments, config, machine=machine
        )
        load_source_validation_data(
            frame, contract, feature_columns=phase31_feature_columns(frame)
        )

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(artifact_path, index=False)
    metadata = {
        "policy_version": PHASE31_PREPROCESSING_POLICY_VERSION,
        "feature_version": FEATURE_VERSION,
        "role": role,
        "split": split,
        "machine": machine,
        "regime": config.regime,
        "preprocessing_config_id": config.config_id,
        "preprocessing_config_hash": file_sha256(preprocessing_config_path),
        "split_version": config.split_version,
        "split_manifest_hash": file_sha256(split_manifest_path),
        "segment_count": int(len(frame)),
        "window_count": int(frame["window_count"].sum()),
        "feature_count": len(phase31_feature_columns(frame)),
        "segment_cohort_hash": ordered_id_hash(tuple(frame["segment_id"].astype(str))),
        "artifact_sha256": file_sha256(artifact_path),
        "test_evaluated": False,
        "evaluation_stage": PHASE31_EVALUATION_STAGE,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def load_feature_artifact(
    root: str | Path, *, machine: str, regime: str, role: str
) -> tuple[pd.DataFrame, dict]:
    assert_phase31_role(role)
    artifact_path, metadata_path = feature_artifact_paths(
        root, machine=machine, regime=regime, role=role
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("role") != role or metadata.get("test_evaluated") is not False:
        raise ValueError("Phase-3.1 feature-artifact metadata is incompatible")
    if file_sha256(artifact_path) != metadata.get("artifact_sha256"):
        raise ValueError("Phase-3.1 feature artifact hash mismatch")
    frame = pd.read_parquet(artifact_path)
    if len(frame) != int(metadata["segment_count"]):
        raise ValueError("Phase-3.1 feature artifact row count mismatch")
    if ordered_id_hash(tuple(frame["segment_id"].astype(str))) != metadata["segment_cohort_hash"]:
        raise ValueError("Phase-3.1 feature artifact cohort hash mismatch")
    phase31_feature_columns(frame)
    return frame, metadata


def dependency_environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": package_version("numpy"),
        "pandas": package_version("pandas"),
        "scikit-learn": package_version("scikit-learn"),
        "scipy": package_version("scipy"),
        "joblib": package_version("joblib"),
    }


def phase31_source_hash(paths: Iterable[str | Path]) -> str:
    return content_sha256({str(Path(path)): file_sha256(path) for path in paths})


def seed_plan(family: str) -> tuple[int, ...]:
    return ISOLATION_FOREST_SEEDS if family == "isolation_forest" else (DETERMINISTIC_SEED,)


def run_one_baseline(
    *,
    machine: str,
    regime: str,
    family: str,
    seed: int,
    feature_root: str | Path,
    result_root: str | Path,
    split_manifest_path: str | Path,
    preprocessing_config_path: str | Path,
    code_source_hash: str,
) -> dict:
    """Fit one declared baseline and persist source-validation results only."""

    assert_phase31_role("training")
    assert_phase31_role("source_validation")
    spec = primary_spec(family)
    if not spec.stochastic and seed != DETERMINISTIC_SEED:
        raise ValueError("deterministic Phase-3.1 models have one canonical seed entry (0)")
    train_frame, train_meta = load_feature_artifact(
        feature_root, machine=machine, regime=regime, role="training"
    )
    validation_frame, validation_meta = load_feature_artifact(
        feature_root, machine=machine, regime=regime, role="source_validation"
    )
    config = load_config(preprocessing_config_path)
    assignments = pd.read_csv(split_manifest_path)
    train_contract = TrainingCohortContract.from_frozen_assignments(
        assignments, config, machine=machine
    )
    validation_contract = SourceValidationCohortContract.from_frozen_assignments(
        assignments, config, machine=machine
    )
    features = phase31_feature_columns(train_frame)
    if phase31_feature_columns(validation_frame) != features:
        raise ValueError("training and source-validation feature schemas differ")
    training = load_estimator_training_data(
        train_frame, train_contract, feature_columns=features
    )
    fitted_scaler = fit_training_standardizer(
        train_frame, train_contract, feature_columns=features
    )
    validation = load_source_validation_data(
        validation_frame, validation_contract, feature_columns=features
    )
    X_train = fitted_scaler.standardizer.transform(training.X)
    X_validation = fitted_scaler.standardizer.transform(validation.X)

    model = ClassicalAnomalyBaseline(spec, seed=seed).fit(X_train)
    score_started = perf_counter()
    validation_scores = model.score(X_validation)
    score_seconds = perf_counter() - score_started
    model_hyperparameters = {
        **dict(spec.hyperparameters),
        "implementation": spec.implementation,
        "implementation_version": "phase3.1-classical-baselines-v1",
        "model_spec_id": spec.identity,
        "stochastic": spec.stochastic,
        "score_transform": spec.score_transform,
        "training_feature_aggregation": "segment_median",
        "feature_standardization": "training_cohort_zscore",
        "standardization_epsilon": EPSILON,
        "phase31_source_hash": code_source_hash,
    }
    experiment = ExperimentConfig(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        machine=machine,
        protocol=config.protocol,
        regime=regime,
        model_family=family,
        model_hyperparameters=model_hyperparameters,
        model_seed=seed,
        preprocessing_config_id=config.config_id,
        preprocessing_config_hash=file_sha256(preprocessing_config_path),
        split_version=config.split_version,
        split_manifest_hash=file_sha256(split_manifest_path),
        dataset_release_identity=DATASET_RELEASE_IDENTITY,
        feature_representation=PRIMARY_FEATURE_REPRESENTATION,
        feature_version=FEATURE_VERSION,
        window_length_s=config.window_length_s,
        stride_s=config.stride_s,
        normalization_strategy="train_only_feature_zscore_after_segment_median",
        code_git_sha=ACCEPTED_PHASE31_HEAD,
        dependency_environment=dependency_environment(),
    )
    training_hash = ordered_id_hash(training.segment_ids)
    validation_hash = ordered_id_hash(validation.segment_ids)
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
        data_provenance={
            "split_manifest_hash": file_sha256(split_manifest_path),
            "training_cohort_hash": training_hash,
            "source_validation_cohort_hash": validation_hash,
            "training_feature_artifact_hash": train_meta["artifact_sha256"],
            "source_validation_feature_artifact_hash": validation_meta["artifact_sha256"],
        },
        config_provenance={
            "accepted_head": ACCEPTED_PHASE31_HEAD,
            "phase31_source_hash": code_source_hash,
            "preprocessing_policy_version": PHASE31_PREPROCESSING_POLICY_VERSION,
        },
    )
    batch = ScoreBatch(
        provenance=score_provenance,
        scores=validation_scores,
        segment_ids=validation.segment_ids,
        parent_recording_ids=validation.parent_recordings,
        cohort_roles=("source_validation",) * len(validation.segment_ids),
        domains=("source",) * len(validation.segment_ids),
        anomaly_labels=validation.anomaly_labels,
    )
    EvaluationGate(experiment, score_provenance).evaluate_validation(batch)
    threshold = select_protocol_b_threshold(batch, expected_provenance=score_provenance)
    table = pd.DataFrame({
        "segment_id": validation.segment_ids,
        "parent_recording": validation.parent_recordings,
        "cohort_role": batch.cohort_roles,
        "domain": batch.domains,
        "anomaly_label": validation.anomaly_labels,
        "score": validation_scores,
    }).sort_values("segment_id").reset_index(drop=True)
    summary = {
        "evaluation_stage": PHASE31_EVALUATION_STAGE,
        "test_evaluated": False,
        "descriptive_only_no_model_ranking": True,
        "training_segment_count": len(training.segment_ids),
        "training_window_count": int(train_frame["window_count"].sum()),
        "source_validation_segment_count": len(validation.segment_ids),
        "source_validation_window_count": int(validation_frame["window_count"].sum()),
        "feature_count": len(features),
        "score_count": int(len(validation_scores)),
        "finite_score_count": int(np.isfinite(validation_scores).sum()),
        "non_finite_score_count": int((~np.isfinite(validation_scores)).sum()),
        "score_min": float(np.min(validation_scores)),
        "score_q05": float(np.quantile(validation_scores, 0.05, method="higher")),
        "score_q25": float(np.quantile(validation_scores, 0.25, method="higher")),
        "score_median": float(np.median(validation_scores)),
        "score_q75": float(np.quantile(validation_scores, 0.75, method="higher")),
        "score_max": float(np.max(validation_scores)),
        "score_q90": float(threshold.value),
        "score_q95": float(np.quantile(validation_scores, 0.95, method="higher")),
        "fit_seconds": float(model.fit_seconds_ or 0.0),
        "validation_scoring_seconds": float(score_seconds),
        "model_saved": False,
        "model_serialized_size_bytes": "not_applicable_model_not_saved",
    }
    bundle = PretestResultStore(result_root).write_bundle(
        config=experiment,
        provenance={
            "evaluation_stage": PHASE31_EVALUATION_STAGE,
            "test_evaluated": False,
            "score_direction": SCORE_DIRECTION,
            "training_cohort_hash": training_hash,
            "validation_cohort_hash": validation_hash,
            "training_contract_hash": train_contract.frozen_manifest_hash,
            "validation_contract_hash": validation_contract.frozen_manifest_hash,
            "score_provenance": asdict(score_provenance),
        },
        summary=summary,
        threshold=threshold,
        validation_scores=table,
    )
    return {
        "experiment_id": experiment.experiment_id,
        "model_id": model.model_id,
        "machine": machine,
        "regime": regime,
        "family": family,
        "seed": seed,
        "stochastic": spec.stochastic,
        "threshold_q90": threshold.value,
        "training_segments": len(training.segment_ids),
        "validation_segments": len(validation.segment_ids),
        "feature_count": len(features),
        "fit_seconds": summary["fit_seconds"],
        "validation_scoring_seconds": score_seconds,
        "bundle": str(bundle),
        "test_evaluated": False,
        "evaluation_stage": PHASE31_EVALUATION_STAGE,
    }
