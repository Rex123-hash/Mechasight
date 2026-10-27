"""Locked Phase-3.2 common-test evaluation primitives.

The module is intentionally configuration-driven.  It can audit and
reproduce the 28 immutable Phase-3.1 experiments before a test look, then
requires a durable batch-ledger receipt before any model scorer is invoked.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from src.evaluation.reporting import expected_seeds
from src.experiment.config import BootstrapConfig, ExperimentConfig
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
from src.experiment.metrics import (
    RANKING_METRICS,
    protocol_b_ranking_metrics,
    protocol_b_threshold_metrics,
    ranking_metric_value,
)
from src.experiment.phase31 import (
    ACCEPTED_PHASE31_HEAD,
    phase31_feature_columns,
    load_feature_artifact,
)
from src.experiment.provenance import content_sha256, file_sha256, ordered_id_hash
from src.experiment.scores import (
    ScoreBatch,
    ScoreProvenance,
    ThresholdSelection,
    select_protocol_b_threshold,
)
from src.models.classical import ClassicalAnomalyBaseline, primary_spec
from src.preprocessing.config import MECHASIGHT_PROTOCOL, PreprocessingConfig, load_config
from src.preprocessing.pipeline import (
    iter_processed_windows,
    load_raw_catalog,
    merge_protocol_b_assignments,
)
from src.preprocessing.protected_access import require_protected_read
from src.preprocessing.splits import (
    BLOCKED_SIBLING_POOL,
    BLOCKED_SIBLING_SPLIT,
    PROTOCOL_B_V3_PREPROCESSING_VERSION,
    PROTOCOL_B_V3_SPLIT_VERSION,
    assert_few_shot_adaptation_cohorts,
)


PHASE32_SCHEMA_VERSION = "phase3.2-first-locked-test-v1"
PHASE31_SOURCE_SNAPSHOT = "93c33580054e9e45f228e385ba726bb43ad0dfb3531f83bb9f3c9c5fb003e229"
REPORTING_POLICY_ID = "p311-reporting-5e76ab7dabd5689c7b0ced68"
EXPECTED_TEST_SEGMENTS = {"RoboticArm": 466, "BrushlessMotor": 312}
REQUESTED_METRICS = (*RANKING_METRICS, "threshold_metrics")
TEST_REASON = "first pre-registered Protocol-B baseline evaluation"


@dataclass(frozen=True)
class CommonTestCohortContract:
    machine: str
    regime: str
    protocol: str
    split_version: str
    preprocessing_version: str
    preprocessing_config_id: str
    segment_ids: frozenset[str]
    frozen_manifest_hash: str

    @classmethod
    def from_frozen_assignments(
        cls,
        assignments: pd.DataFrame,
        config: PreprocessingConfig,
        *,
        machine: str,
    ) -> "CommonTestCohortContract":
        if config.protocol != MECHASIGHT_PROTOCOL:
            raise ValueError("Phase-3.2 test evaluation requires Protocol B")
        if config.split_version != PROTOCOL_B_V3_SPLIT_VERSION:
            raise ValueError("Phase-3.2 refuses a stale split version")
        if config.version != PROTOCOL_B_V3_PREPROCESSING_VERSION:
            raise ValueError("Phase-3.2 refuses a stale preprocessing version")
        assert_few_shot_adaptation_cohorts(assignments)
        part = assignments[
            assignments["machine"].eq(machine)
            & assignments["regime"].eq(config.regime)
            & assignments["split"].eq("test")
            & assignments["data_pool"].eq("common_evaluation")
        ].copy()
        expected = EXPECTED_TEST_SEGMENTS[machine]
        if len(part) != expected or part["segment_id"].nunique() != expected:
            raise AssertionError(f"{machine} common test must contain {expected} segments")
        for column, value in (
            ("protocol", config.protocol),
            ("split_version", config.split_version),
            ("preprocessing_version", config.version),
            ("split", "test"),
            ("data_pool", "common_evaluation"),
        ):
            if set(part[column].astype(str)) != {value}:
                raise ValueError(f"common-test {column} mismatch")
        if BLOCKED_SIBLING_SPLIT in set(part["split"].astype(str)) or (
            BLOCKED_SIBLING_POOL in set(part["data_pool"].astype(str))
        ):
            raise ValueError("blocked siblings are forbidden in common-test evaluation")
        identity_rows = part[
            ["machine", "regime", "segment_id", "parent_recording", "split", "data_pool"]
        ].sort_values("segment_id").to_dict("records")
        return cls(
            machine=machine,
            regime=config.regime,
            protocol=config.protocol,
            split_version=config.split_version,
            preprocessing_version=config.version,
            preprocessing_config_id=config.config_id,
            segment_ids=frozenset(part["segment_id"].astype(str)),
            frozen_manifest_hash=content_sha256(identity_rows),
        )


@dataclass(frozen=True)
class EstimatorTestData:
    X: np.ndarray
    feature_names: tuple[str, ...]
    segment_ids: tuple[str, ...]
    parent_recordings: tuple[str, ...]
    domains: tuple[str, ...]
    anomaly_labels: tuple[str, ...]
    operating_conditions: tuple[str, ...]
    background_conditions: tuple[str, ...]
    cohort_hash: str
    machine: str
    regime: str


def load_common_test_data(
    frame: pd.DataFrame,
    contract: CommonTestCohortContract,
    *,
    feature_columns: Iterable[str],
) -> EstimatorTestData:
    features = tuple(feature_columns)
    if not features or set(features) & FORBIDDEN_MODEL_FEATURES:
        raise ValueError("test estimator features must be explicit and label/provenance-free")
    required = {
        "machine", "protocol", "regime", "split", "data_pool", "domain",
        "segment_id", "parent_recording", "anomaly_label", "operating_condition",
        "background_condition", "preprocessing_config_id", "preprocessing_version",
        "split_version", *features,
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"common-test frame is missing fields: {sorted(missing)}")
    expected_scalars = {
        "machine": contract.machine,
        "protocol": contract.protocol,
        "regime": contract.regime,
        "split": "test",
        "data_pool": "common_evaluation",
        "preprocessing_config_id": contract.preprocessing_config_id,
        "preprocessing_version": contract.preprocessing_version,
        "split_version": contract.split_version,
    }
    for column, expected in expected_scalars.items():
        if set(frame[column].astype(str)) != {expected}:
            raise ValueError(f"wrong common-test {column}")
    segment_ids = tuple(frame["segment_id"].astype(str))
    if len(segment_ids) != len(set(segment_ids)) or frozenset(segment_ids) != contract.segment_ids:
        raise ValueError("common-test segment identities do not match the frozen cohort")
    if frame["parent_recording"].astype(str).eq("").any():
        raise ValueError("common-test parent identities are required")
    if frame["anomaly_label"].astype(str).eq("").any():
        raise ValueError("common-test labels are required for authorized final evaluation")
    values = frame.loc[:, features].to_numpy(dtype=np.float64, copy=True)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("common-test features must be finite")
    return EstimatorTestData(
        X=values,
        feature_names=features,
        segment_ids=segment_ids,
        parent_recordings=tuple(frame["parent_recording"].astype(str)),
        domains=tuple(frame["domain"].astype(str)),
        anomaly_labels=tuple(frame["anomaly_label"].astype(str)),
        operating_conditions=tuple(frame["operating_condition"].astype(str)),
        background_conditions=tuple(frame["background_condition"].astype(str)),
        cohort_hash=ordered_id_hash(segment_ids),
        machine=contract.machine,
        regime=contract.regime,
    )


def _aggregate_common_test_windows(windows: Iterable[tuple[Mapping, Mapping, Mapping]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    current_id: str | None = None
    current_record: Mapping | None = None
    current_features: list[Mapping[str, float]] = []

    def flush() -> None:
        nonlocal current_record, current_features
        if current_record is None:
            return
        names = tuple(sorted(current_features[0]))
        if any(tuple(sorted(item)) != names for item in current_features):
            raise ValueError("common-test window feature schemas differ")
        matrix = np.asarray(
            [[float(item[name]) for name in names] for item in current_features], dtype=np.float64
        )
        if not np.all(np.isfinite(matrix)):
            raise ValueError("common-test window features contain non-finite values")
        trace = {
            key: current_record[key]
            for key in (
                "machine", "protocol", "regime", "split", "data_pool", "domain",
                "parent_recording", "segment_id", "anomaly_label", "operating_condition",
                "background_condition", "preprocessing_config_id", "preprocessing_version",
                "split_version",
            )
        }
        trace["window_count"] = len(current_features)
        trace.update({name: float(value) for name, value in zip(names, np.median(matrix, axis=0))})
        rows.append(trace)

    for record, _signals, features in windows:
        if record.get("split") != "test" or record.get("data_pool") != "common_evaluation":
            raise ValueError("Phase-3.2 materializer accepts common_evaluation test rows only")
        segment_id = str(record.get("segment_id", ""))
        if not segment_id:
            raise ValueError("common-test segment identity is missing")
        if current_id is not None and segment_id != current_id:
            flush()
            current_features = []
        current_id = segment_id
        current_record = record
        current_features.append(features)
    flush()
    if not rows:
        raise ValueError("common-test artifact cannot be empty")
    return pd.DataFrame(rows).sort_values("segment_id").reset_index(drop=True)


def common_test_artifact_paths(
    root: str | Path, *, machine: str, regime: str
) -> tuple[Path, Path]:
    base = Path(root) / machine / regime
    return base / "common_test.parquet", base / "common_test.metadata.json"


def prepare_common_test_artifact(
    *,
    data_root: str | Path,
    split_manifest_path: str | Path,
    preprocessing_config_path: str | Path,
    artifact_root: str | Path,
    machine: str,
    test_evaluation_code_sha: str,
) -> dict[str, Any]:
    require_protected_read("build common-test feature artifact")
    config = load_config(preprocessing_config_path)
    artifact_path, metadata_path = common_test_artifact_paths(
        artifact_root, machine=machine, regime=config.regime
    )
    if artifact_path.exists() or metadata_path.exists():
        raise FileExistsError("refusing to overwrite common-test feature artifact")
    assignments = pd.read_csv(split_manifest_path)
    contract = CommonTestCohortContract.from_frozen_assignments(
        assignments, config, machine=machine
    )
    catalog = merge_protocol_b_assignments(
        load_raw_catalog(data_root), assignments, config.regime,
        split_version=config.split_version,
    )
    machine_catalog = catalog[catalog["machine"].eq(machine)].copy()
    frame = _aggregate_common_test_windows(
        iter_processed_windows(machine_catalog, config, split="test")
    )
    features = phase31_feature_columns(frame)
    loaded = load_common_test_data(frame, contract, feature_columns=features)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(artifact_path, index=False)
    metadata = {
        "schema_version": PHASE32_SCHEMA_VERSION,
        "test_evaluation_code_sha": test_evaluation_code_sha,
        "machine": machine,
        "regime": config.regime,
        "role": "common_evaluation",
        "segment_count": len(frame),
        "window_count": int(frame["window_count"].sum()),
        "feature_count": len(features),
        "cohort_hash": loaded.cohort_hash,
        "frozen_manifest_contract_hash": contract.frozen_manifest_hash,
        "preprocessing_config_id": config.config_id,
        "preprocessing_config_hash": file_sha256(preprocessing_config_path),
        "split_manifest_hash": file_sha256(split_manifest_path),
    }
    metadata["artifact_sha256"] = file_sha256(artifact_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def load_common_test_artifact(
    root: str | Path,
    *,
    machine: str,
    regime: str,
    contract: CommonTestCohortContract,
) -> tuple[pd.DataFrame, EstimatorTestData, dict[str, Any]]:
    require_protected_read("common-test feature artifact")
    artifact_path, metadata_path = common_test_artifact_paths(root, machine=machine, regime=regime)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if file_sha256(artifact_path) != metadata.get("artifact_sha256"):
        raise ValueError("common-test feature artifact hash mismatch")
    frame = pd.read_parquet(artifact_path)
    features = phase31_feature_columns(frame)
    loaded = load_common_test_data(frame, contract, feature_columns=features)
    if loaded.cohort_hash != metadata.get("cohort_hash") or len(frame) != metadata.get("segment_count"):
        raise ValueError("common-test artifact metadata mismatch")
    return frame, loaded, metadata


def experiment_config_from_dict(raw: Mapping[str, Any]) -> ExperimentConfig:
    names = {field.name for field in fields(ExperimentConfig)}
    values = {name: raw[name] for name in names}
    values["bootstrap"] = BootstrapConfig(**dict(values["bootstrap"]))
    return ExperimentConfig(**values)


@dataclass
class ReproducedExperiment:
    config: ExperimentConfig
    model: ClassicalAnomalyBaseline
    standardizer: Any
    threshold: ThresholdSelection
    feature_names: tuple[str, ...]
    model_id: str
    training_cohort_hash: str
    validation_cohort_hash: str
    normalizer_identity: str
    validation_exact_match: bool


def _load_verified_phase31_bundle(bundle: Path) -> tuple[dict, dict, dict, pd.DataFrame]:
    manifest = json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if file_sha256(bundle / name) != expected:
            raise ValueError(f"frozen Phase-3.1 bundle hash mismatch: {bundle.name}/{name}")
    raw_config = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
    provenance = json.loads((bundle / "provenance.json").read_text(encoding="utf-8"))
    threshold = json.loads((bundle / "threshold.json").read_text(encoding="utf-8"))
    scores = pd.read_csv(
        bundle / "validation_scores.csv", float_precision="round_trip"
    ).sort_values("segment_id").reset_index(drop=True)
    return raw_config, provenance, threshold, scores


def reproduce_frozen_experiment(
    *,
    bundle: str | Path,
    feature_root: str | Path,
    split_manifest_path: str | Path,
    preprocessing_config_path: str | Path,
) -> ReproducedExperiment:
    bundle = Path(bundle)
    raw_config, provenance, raw_threshold, frozen_scores = _load_verified_phase31_bundle(bundle)
    experiment = experiment_config_from_dict(raw_config)
    if experiment.experiment_id != bundle.name or raw_config.get("experiment_id") != bundle.name:
        raise ValueError("frozen experiment identity mismatch")
    if experiment.code_git_sha != ACCEPTED_PHASE31_HEAD:
        raise ValueError("frozen experiment accepted-base SHA mismatch")
    if experiment.model_hyperparameters.get("phase31_source_hash") != PHASE31_SOURCE_SNAPSHOT:
        raise ValueError("frozen experiment model-source snapshot mismatch")
    spec = primary_spec(experiment.model_family)
    for key, expected in spec.hyperparameters.items():
        if experiment.model_hyperparameters.get(key) != expected:
            raise ValueError(f"frozen model hyperparameter mismatch: {key}")
    if experiment.model_seed not in expected_seeds(experiment.model_family):
        raise ValueError("frozen experiment seed is not authorized")

    preprocessing = load_config(preprocessing_config_path)
    if preprocessing.config_id != experiment.preprocessing_config_id:
        raise ValueError("frozen preprocessing identity mismatch")
    if file_sha256(preprocessing_config_path) != experiment.preprocessing_config_hash:
        raise ValueError("frozen preprocessing hash mismatch")
    assignments = pd.read_csv(split_manifest_path)
    training_frame, _training_meta = load_feature_artifact(
        feature_root, machine=experiment.machine, regime=experiment.regime, role="training"
    )
    validation_frame, _validation_meta = load_feature_artifact(
        feature_root, machine=experiment.machine, regime=experiment.regime,
        role="source_validation",
    )
    feature_names = phase31_feature_columns(training_frame)
    if phase31_feature_columns(validation_frame) != feature_names:
        raise ValueError("frozen training/validation feature schema mismatch")
    training_contract = TrainingCohortContract.from_frozen_assignments(
        assignments, preprocessing, machine=experiment.machine
    )
    validation_contract = SourceValidationCohortContract.from_frozen_assignments(
        assignments, preprocessing, machine=experiment.machine
    )
    training = load_estimator_training_data(
        training_frame, training_contract, feature_columns=feature_names
    )
    validation = load_source_validation_data(
        validation_frame, validation_contract, feature_columns=feature_names
    )
    fitted = fit_training_standardizer(
        training_frame, training_contract, feature_columns=feature_names
    )
    normalizer_identity = content_sha256({
        "feature_names": feature_names,
        "mean": fitted.standardizer.mean.tolist(),
        "std": fitted.standardizer.std.tolist(),
        "epsilon": fitted.standardizer.epsilon,
        "training_cohort_hash": ordered_id_hash(training.segment_ids),
    })
    model = ClassicalAnomalyBaseline(spec, seed=experiment.model_seed).fit(
        fitted.standardizer.transform(training.X)
    )
    if model.model_id != provenance["score_provenance"]["model_id"]:
        raise ValueError("reproduced model identity mismatch")
    reproduced_scores = model.score(fitted.standardizer.transform(validation.X))
    frozen_ids = tuple(frozen_scores["segment_id"].astype(str))
    if validation.segment_ids != frozen_ids:
        raise ValueError("reproduced validation segment identities differ")
    if not np.array_equal(reproduced_scores, frozen_scores["score"].to_numpy(dtype=float)):
        raise ValueError("reproduced validation scores are not exactly equal")
    score_provenance = ScoreProvenance(**provenance["score_provenance"])
    batch = ScoreBatch(
        provenance=score_provenance,
        scores=reproduced_scores,
        segment_ids=validation.segment_ids,
        parent_recording_ids=validation.parent_recordings,
        cohort_roles=("source_validation",) * len(validation.segment_ids),
        domains=("source",) * len(validation.segment_ids),
        anomaly_labels=validation.anomaly_labels,
    )
    EvaluationGate(experiment, score_provenance).evaluate_validation(batch)
    threshold = select_protocol_b_threshold(batch, expected_provenance=score_provenance)
    frozen_threshold = ThresholdSelection(
        value=float(raw_threshold["value"]),
        policy_version=raw_threshold["policy_version"],
        quantile=float(raw_threshold["quantile"]),
        quantile_method=raw_threshold["quantile_method"],
        score_direction=raw_threshold["score_direction"],
        source_experiment_id=raw_threshold["source_experiment_id"],
        source_cohort_hash=raw_threshold["source_cohort_hash"],
        source_segment_ids=tuple(raw_threshold["source_segment_ids"]),
        source_role=raw_threshold["source_role"],
    )
    if threshold != frozen_threshold:
        raise ValueError("reproduced q90 threshold is not exactly equal to frozen threshold")
    training_hash = ordered_id_hash(training.segment_ids)
    validation_hash = ordered_id_hash(validation.segment_ids)
    if training_hash != provenance["training_cohort_hash"]:
        raise ValueError("reproduced training cohort hash mismatch")
    if validation_hash != provenance["validation_cohort_hash"]:
        raise ValueError("reproduced validation cohort hash mismatch")
    return ReproducedExperiment(
        config=experiment,
        model=model,
        standardizer=fitted.standardizer,
        threshold=frozen_threshold,
        feature_names=feature_names,
        model_id=model.model_id,
        training_cohort_hash=training_hash,
        validation_cohort_hash=validation_hash,
        normalizer_identity=normalizer_identity,
        validation_exact_match=True,
    )


@dataclass(frozen=True)
class TestBatchAuthorization:
    __test__ = False

    batch_id: str
    test_evaluation_code_sha: str
    reporting_policy_id: str
    experiment_ids: frozenset[str]
    experiment_test_cohort_hashes: Mapping[str, str]
    ledger_start_count: int
    ledger_end_count: int


def authorize_first_test_batch(
    *,
    ledger: TestLookLedger,
    reproduced: Mapping[str, ReproducedExperiment],
    expected_experiment_ids: frozenset[str],
    test_cohort_hashes_by_machine: Mapping[str, str],
    test_evaluation_code_sha: str,
    reporting_policy_id: str = REPORTING_POLICY_ID,
    expected_ledger_start: int = 0,
) -> TestBatchAuthorization:
    if frozenset(reproduced) != expected_experiment_ids or len(reproduced) != 28:
        raise ValueError("first-test authorization requires exactly the 28 frozen experiment IDs")
    if reporting_policy_id != REPORTING_POLICY_ID:
        raise ValueError("first-test reporting-policy identity mismatch")
    if len(ledger.read()) != expected_ledger_start:
        raise RuntimeError("unexpected test-look ledger starting count")
    exp_hashes = {
        experiment_id: test_cohort_hashes_by_machine[item.config.machine]
        for experiment_id, item in reproduced.items()
    }
    batch_id = f"p32-test-batch-{content_sha256({
        'experiment_ids': sorted(reproduced),
        'cohort_hashes': exp_hashes,
        'policy': reporting_policy_id,
        'code': test_evaluation_code_sha,
    })[:24]}"
    timestamp = datetime.now(timezone.utc).isoformat()
    reason = (
        f"{TEST_REASON}; batch={batch_id}; policy={reporting_policy_id}; "
        "slices=complete_per_machine,source_domain,target_domain,healthy_target"
    )
    records = tuple(
        TestLookRecord(
            timestamp_utc=timestamp,
            experiment_id=experiment_id,
            git_sha=test_evaluation_code_sha,
            machine=item.config.machine,
            protocol=item.config.protocol,
            regime=item.config.regime,
            model_config_identity=item.model_id,
            test_cohort_hash=exp_hashes[experiment_id],
            reason=reason,
            metrics_requested=REQUESTED_METRICS,
        )
        for experiment_id, item in sorted(reproduced.items())
    )
    ledger.append_many_atomic(records, expected_existing_count=expected_ledger_start)
    return TestBatchAuthorization(
        batch_id=batch_id,
        test_evaluation_code_sha=test_evaluation_code_sha,
        reporting_policy_id=reporting_policy_id,
        experiment_ids=frozenset(reproduced),
        experiment_test_cohort_hashes=exp_hashes,
        ledger_start_count=expected_ledger_start,
        ledger_end_count=expected_ledger_start + len(records),
    )


def invoke_authorized_scorer(
    *,
    experiment_id: str,
    test_cohort_hash: str,
    authorization: TestBatchAuthorization,
    scorer: Callable[[np.ndarray], np.ndarray],
    standardized_test_X: np.ndarray,
) -> np.ndarray:
    if experiment_id not in authorization.experiment_ids:
        raise ValueError("experiment lacks durable first-test authorization")
    if authorization.experiment_test_cohort_hashes.get(experiment_id) != test_cohort_hash:
        raise ValueError("authorized and supplied test cohort hashes differ")
    if authorization.ledger_end_count - authorization.ledger_start_count != 28:
        raise ValueError("first-test authorization receipt is incomplete")
    scores = np.asarray(scorer(standardized_test_X), dtype=np.float64).reshape(-1)
    if len(scores) != len(standardized_test_X) or not np.all(np.isfinite(scores)):
        raise ValueError("authorized scorer returned invalid scores")
    return scores


def binary_labels(labels: Iterable[str]) -> np.ndarray:
    return np.asarray([str(label).lower() != "normal" for label in labels], dtype=np.int8)


def ranking_metrics_or_undefined(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    if len(np.unique(np.asarray(labels))) != 2:
        return {
            "status": "UNDEFINED — ONE CLASS PRESENT",
            **{metric: None for metric in RANKING_METRICS},
        }
    return {"status": "defined", **protocol_b_ranking_metrics(labels, scores)}


@dataclass
class EvaluatedExperiment:
    reproduced: ReproducedExperiment
    test_data: EstimatorTestData
    scores: np.ndarray
    score_table: pd.DataFrame
    complete_metrics: dict[str, Any]
    domain_metrics: dict[str, dict[str, Any]]
    healthy_target: dict[str, Any]


def evaluate_authorized_experiment(
    reproduced: ReproducedExperiment,
    test_data: EstimatorTestData,
    authorization: TestBatchAuthorization,
) -> EvaluatedExperiment:
    experiment = reproduced.config
    if (test_data.machine, test_data.regime) != (experiment.machine, experiment.regime):
        raise ValueError("test data machine/regime mismatch")
    if test_data.feature_names != reproduced.feature_names:
        raise ValueError("test feature schema differs from frozen training schema")
    if reproduced.threshold.source_experiment_id != experiment.experiment_id:
        raise ValueError("frozen threshold belongs to another experiment")
    X = reproduced.standardizer.transform(test_data.X)
    scores = invoke_authorized_scorer(
        experiment_id=experiment.experiment_id,
        test_cohort_hash=test_data.cohort_hash,
        authorization=authorization,
        scorer=reproduced.model.score,
        standardized_test_X=X,
    )
    labels = binary_labels(test_data.anomaly_labels)
    ranking = protocol_b_ranking_metrics(labels, scores)
    threshold = protocol_b_threshold_metrics(
        labels, scores, threshold=reproduced.threshold.value
    ).to_dict()
    complete = {**ranking, **threshold}
    domains = np.asarray(test_data.domains)
    domain_metrics = {}
    for domain in ("source", "target"):
        mask = domains == domain
        domain_metrics[domain] = {
            "sample_count": int(mask.sum()),
            **ranking_metrics_or_undefined(labels[mask], scores[mask]),
        }
    healthy_target_mask = (domains == "target") & (labels == 0)
    healthy_scores = scores[healthy_target_mask]
    fp = int(np.sum(healthy_scores > reproduced.threshold.value))
    healthy_n = int(healthy_target_mask.sum())
    healthy_target = {
        "healthy_target_sample_count": healthy_n,
        "false_positive": fp,
        "false_positive_rate": fp / healthy_n if healthy_n else None,
    }
    table = pd.DataFrame({
        "segment_id": test_data.segment_ids,
        "parent_recording": test_data.parent_recordings,
        "machine": experiment.machine,
        "regime": experiment.regime,
        "model_family": experiment.model_family,
        "model_seed": experiment.model_seed,
        "domain": test_data.domains,
        "anomaly_label": test_data.anomaly_labels,
        "operating_condition": test_data.operating_conditions,
        "background_condition": test_data.background_conditions,
        "score": scores,
    })
    return EvaluatedExperiment(
        reproduced=reproduced,
        test_data=test_data,
        scores=scores,
        score_table=table,
        complete_metrics=complete,
        domain_metrics=domain_metrics,
        healthy_target=healthy_target,
    )


def score_frame(result: EvaluatedExperiment) -> pd.DataFrame:
    return pd.DataFrame({
        "segment_id": result.test_data.segment_ids,
        "parent_recording": result.test_data.parent_recordings,
        "machine": result.reproduced.config.machine,
        "domain": result.test_data.domains,
        "is_anomaly": binary_labels(result.test_data.anomaly_labels),
        "score": result.scores,
    }).sort_values("segment_id").reset_index(drop=True)


def validate_family_score_frames(
    family: str, frames_by_seed: Mapping[int, pd.DataFrame]
) -> pd.DataFrame:
    if set(frames_by_seed) != set(expected_seeds(family)):
        raise ValueError("family score frames do not contain the frozen seed set")
    base: pd.DataFrame | None = None
    identity_columns = ["segment_id", "parent_recording", "machine", "domain", "is_anomaly"]
    for seed in expected_seeds(family):
        frame = frames_by_seed[seed].sort_values("segment_id").reset_index(drop=True)
        if base is None:
            base = frame
        elif not frame[identity_columns].equals(base[identity_columns]):
            raise ValueError("family seed score frames do not share identical evaluation units")
    assert base is not None
    return base


@dataclass(frozen=True)
class CommonBootstrapDraws:
    sample_weights: np.ndarray
    identity: str
    replicates: int
    seed: int


def make_common_bootstrap_draws(
    frame: pd.DataFrame,
    *,
    replicates: int = 10_000,
    seed: int = 1995,
) -> CommonBootstrapDraws:
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    machines = set(frame["machine"].astype(str))
    if len(machines) != 1:
        raise ValueError("bootstrap requires one machine cohort")
    if set(frame["is_anomaly"].astype(int)) != {0, 1}:
        raise ValueError("class-stratified bootstrap requires both classes")
    homogeneous = frame.groupby("parent_recording")["is_anomaly"].nunique()
    if (homogeneous > 1).any():
        raise ValueError("bootstrap parent recording spans classes")
    machine = next(iter(machines))
    if machine == "BrushlessMotor" and not frame.groupby("parent_recording").size().eq(1).all():
        raise ValueError("BrushlessMotor bootstrap requires one segment per recording")
    rng = np.random.default_rng(seed)
    parent_values = frame["parent_recording"].astype(str).to_numpy()
    weights = np.zeros((replicates, len(frame)), dtype=np.uint16)
    parents = frame[["parent_recording", "is_anomaly"]].drop_duplicates()
    for _label, class_parents in parents.groupby("is_anomaly", sort=True):
        ids = class_parents["parent_recording"].astype(str).to_numpy()
        selected = rng.integers(0, len(ids), size=(replicates, len(ids)))
        for parent_index, parent_id in enumerate(ids):
            multiplicity = np.sum(selected == parent_index, axis=1, dtype=np.uint16)
            weights[:, parent_values == parent_id] = multiplicity[:, None]
    identity = "p32-draws-" + sha256(
        b"|".join((
            str(seed).encode(),
            ordered_id_hash(tuple(frame["segment_id"].astype(str))).encode(),
            weights.tobytes(),
        ))
    ).hexdigest()[:24]
    return CommonBootstrapDraws(weights, identity, replicates, seed)


@dataclass(frozen=True)
class BootstrapDistribution:
    metric: str
    point_estimate: float
    values: np.ndarray
    replicates_requested: int
    valid_replicates: int
    invalid_replicates: int
    ci_lower: float | None
    ci_upper: float | None
    draw_identity: str


def _score_groups(scores: np.ndarray, *, descending: bool) -> list[np.ndarray]:
    order = np.argsort(scores, kind="mergesort")
    if descending:
        order = order[::-1]
    ordered = scores[order]
    boundaries = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1], True])
    return [order[boundaries[index]:boundaries[index + 1]] for index in range(len(boundaries) - 1)]


def weighted_bootstrap_metric(
    labels: np.ndarray,
    scores: np.ndarray,
    sample_weights: np.ndarray,
    *,
    metric: str,
) -> np.ndarray:
    """Vectorized weighted equivalent of duplicate-row bootstrap metrics."""

    y = np.asarray(labels, dtype=np.int8).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    weights = np.asarray(sample_weights, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[1] != len(y) or len(y) != len(s):
        raise ValueError("bootstrap weight/label/score dimensions differ")
    if not np.all(np.isfinite(s)) or not set(np.unique(y)) <= {0, 1}:
        raise ValueError("bootstrap metric inputs are invalid")
    total_positive = weights[:, y == 1].sum(axis=1)
    total_negative = weights[:, y == 0].sum(axis=1)
    valid = (total_positive > 0) & (total_negative > 0)
    output = np.full(len(weights), np.nan, dtype=np.float64)

    if metric == "auroc":
        numerator = np.zeros(len(weights), dtype=np.float64)
        cumulative_negative = np.zeros(len(weights), dtype=np.float64)
        for group in _score_groups(s, descending=False):
            group_weights = weights[:, group]
            group_labels = y[group]
            positive = group_weights[:, group_labels == 1].sum(axis=1)
            negative = group_weights[:, group_labels == 0].sum(axis=1)
            numerator += positive * (cumulative_negative + 0.5 * negative)
            cumulative_negative += negative
        output[valid] = numerator[valid] / (
            total_positive[valid] * total_negative[valid]
        )
        return output

    groups = _score_groups(s, descending=True)
    if metric == "average_precision":
        cumulative_positive = np.zeros(len(weights), dtype=np.float64)
        cumulative_total = np.zeros(len(weights), dtype=np.float64)
        area = np.zeros(len(weights), dtype=np.float64)
        for group in groups:
            group_weights = weights[:, group]
            group_labels = y[group]
            positive = group_weights[:, group_labels == 1].sum(axis=1)
            total = group_weights.sum(axis=1)
            cumulative_positive += positive
            cumulative_total += total
            precision = np.divide(
                cumulative_positive, cumulative_total,
                out=np.zeros_like(cumulative_positive), where=cumulative_total > 0,
            )
            area += np.divide(
                positive, total_positive,
                out=np.zeros_like(positive), where=total_positive > 0,
            ) * precision
        output[valid] = area[valid]
        return output

    if metric == "standardized_pauroc_fpr_0_10":
        max_fpr = 0.10
        cumulative_positive = np.zeros(len(weights), dtype=np.float64)
        cumulative_negative = np.zeros(len(weights), dtype=np.float64)
        previous_tpr = np.zeros(len(weights), dtype=np.float64)
        previous_fpr = np.zeros(len(weights), dtype=np.float64)
        partial_area = np.zeros(len(weights), dtype=np.float64)
        active = valid.copy()
        for group in groups:
            group_weights = weights[:, group]
            group_labels = y[group]
            cumulative_positive += group_weights[:, group_labels == 1].sum(axis=1)
            cumulative_negative += group_weights[:, group_labels == 0].sum(axis=1)
            tpr = np.divide(
                cumulative_positive, total_positive,
                out=np.zeros_like(cumulative_positive), where=total_positive > 0,
            )
            fpr = np.divide(
                cumulative_negative, total_negative,
                out=np.zeros_like(cumulative_negative), where=total_negative > 0,
            )
            full = active & (fpr <= max_fpr)
            partial_area[full] += 0.5 * (tpr[full] + previous_tpr[full]) * (
                fpr[full] - previous_fpr[full]
            )
            crossing = active & (previous_fpr < max_fpr) & (fpr > max_fpr)
            fraction = np.divide(
                max_fpr - previous_fpr,
                fpr - previous_fpr,
                out=np.zeros_like(fpr), where=fpr > previous_fpr,
            )
            interpolated_tpr = previous_tpr + fraction * (tpr - previous_tpr)
            partial_area[crossing] += 0.5 * (
                interpolated_tpr[crossing] + previous_tpr[crossing]
            ) * (max_fpr - previous_fpr[crossing])
            active &= fpr <= max_fpr
            previous_tpr = tpr
            previous_fpr = fpr
        minimum_area = 0.5 * max_fpr**2
        maximum_area = max_fpr
        output[valid] = 0.5 * (
            1.0 + (partial_area[valid] - minimum_area) / (maximum_area - minimum_area)
        )
        return output
    raise ValueError(f"unknown ranking metric {metric!r}")


def family_bootstrap_distribution(
    family: str,
    frames_by_seed: Mapping[int, pd.DataFrame],
    *,
    metric: str,
    draws: CommonBootstrapDraws,
) -> BootstrapDistribution:
    base = validate_family_score_frames(family, frames_by_seed)
    ordered_frames = {
        seed: frames_by_seed[seed].sort_values("segment_id").reset_index(drop=True)
        for seed in expected_seeds(family)
    }
    point = float(np.mean([
        ranking_metric_value(metric, base["is_anomaly"], ordered_frames[seed]["score"])
        for seed in expected_seeds(family)
    ]))
    seed_values = np.vstack([
        weighted_bootstrap_metric(
            base["is_anomaly"].to_numpy(),
            ordered_frames[seed]["score"].to_numpy(),
            draws.sample_weights,
            metric=metric,
        )
        for seed in expected_seeds(family)
    ])
    values = np.mean(seed_values, axis=0)
    valid = values[np.isfinite(values)]
    lower, upper = (None, None)
    if len(valid):
        lower, upper = map(float, np.quantile(valid, [0.025, 0.975]))
    return BootstrapDistribution(
        metric=metric,
        point_estimate=point,
        values=values,
        replicates_requested=draws.replicates,
        valid_replicates=len(valid),
        invalid_replicates=int(draws.replicates - len(valid)),
        ci_lower=lower,
        ci_upper=upper,
        draw_identity=draws.identity,
    )


def paired_distribution_difference(
    first: BootstrapDistribution,
    second: BootstrapDistribution,
    *,
    orientation: str,
) -> dict[str, Any]:
    if (
        first.metric != second.metric
        or len(first.values) != len(second.values)
        or first.draw_identity != second.draw_identity
    ):
        raise ValueError("paired bootstrap distributions are incompatible")
    differences = first.values - second.values
    valid = differences[np.isfinite(differences)]
    lower, upper = (None, None)
    if len(valid):
        lower, upper = map(float, np.quantile(valid, [0.025, 0.975]))
    return {
        "metric": first.metric,
        "orientation": orientation,
        "point_delta": first.point_estimate - second.point_estimate,
        "replicates_requested": len(differences),
        "valid_replicates": len(valid),
        "invalid_replicates": int(len(differences) - len(valid)),
        "ci_lower": lower,
        "ci_upper": upper,
    }
