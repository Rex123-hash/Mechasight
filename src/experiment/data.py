"""Validated, label-free model-facing training data contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.experiment.constants import (
    BRUSHLESS_MOTOR_ADAPTATION_SEGMENTS,
    BRUSHLESS_MOTOR_VALIDATION_SEGMENTS,
    ROBOTIC_ARM_ADAPTATION_SEGMENTS,
    ROBOTIC_ARM_VALIDATION_PARENTS,
    ROBOTIC_ARM_VALIDATION_SEGMENTS,
)
from src.experiment.provenance import content_sha256
from src.preprocessing.config import MECHASIGHT_PROTOCOL, PreprocessingConfig
from src.preprocessing.normalization import FittedChannelStandardizer
from src.preprocessing.splits import (
    BLOCKED_SIBLING_POOL,
    BLOCKED_SIBLING_SPLIT,
    PROTOCOL_B_V3_PREPROCESSING_VERSION,
    PROTOCOL_B_V3_SPLIT_VERSION,
    assert_few_shot_adaptation_cohorts,
)


FORBIDDEN_MODEL_FEATURES = frozenset({
    "anomaly_label", "is_anomaly", "label", "target", "y",
    "machine", "protocol", "regime", "split", "split_version",
    "preprocessing_version", "preprocessing_config_id", "data_pool",
    "segment_id", "parent_recording", "window_id", "domain", "file_domain",
    "operating_condition", "background_condition", "domain_shift_op", "domain_shift_env",
})


@dataclass(frozen=True)
class TrainingCohortContract:
    """Authorized segment identities for one complete Protocol-B fit cohort."""

    machine: str
    protocol: str
    regime: str
    split_version: str
    preprocessing_version: str
    preprocessing_config_id: str
    source_segment_ids: frozenset[str]
    adaptation_segment_ids: frozenset[str]
    frozen_manifest_hash: str

    @property
    def authorized_segment_ids(self) -> frozenset[str]:
        return self.source_segment_ids | self.adaptation_segment_ids

    @property
    def authorized_roles(self) -> frozenset[str]:
        roles = {"source_training"}
        if self.regime == "T":
            roles.add("target_adaptation")
        return frozenset(roles)

    @classmethod
    def from_frozen_assignments(
        cls,
        assignments: pd.DataFrame,
        config: PreprocessingConfig,
        *,
        machine: str,
    ) -> "TrainingCohortContract":
        """Bind a preprocessing config to the accepted Phase-2.2 segment freeze."""

        if config.protocol != MECHASIGHT_PROTOCOL:
            raise ValueError("Phase-3 Protocol-B training requires the MechaSight protocol")
        if config.split_version != PROTOCOL_B_V3_SPLIT_VERSION:
            raise ValueError("Phase-3 training refuses a stale split version")
        if config.version != PROTOCOL_B_V3_PREPROCESSING_VERSION:
            raise ValueError("Phase-3 training refuses a stale preprocessing version")
        assert_few_shot_adaptation_cohorts(assignments)
        part = assignments[
            assignments["machine"].eq(machine) & assignments["regime"].eq(config.regime)
        ].copy()
        if part.empty:
            raise ValueError("frozen assignments do not contain the requested machine/regime")
        for column, expected in (
            ("protocol", config.protocol),
            ("split_version", config.split_version),
            ("preprocessing_version", config.version),
        ):
            observed = set(part[column].astype(str))
            if observed != {expected}:
                raise ValueError(f"frozen assignment {column} mismatch: {sorted(observed)}")
        source = frozenset(
            part.loc[
                part["data_pool"].eq("source_training") & part["split"].eq("train"),
                "segment_id",
            ].astype(str)
        )
        adaptation = frozenset(
            part.loc[
                part["data_pool"].eq("target_adaptation") & part["split"].eq("train"),
                "segment_id",
            ].astype(str)
        )
        expected_adaptation = {
            "RoboticArm": ROBOTIC_ARM_ADAPTATION_SEGMENTS,
            "BrushlessMotor": BRUSHLESS_MOTOR_ADAPTATION_SEGMENTS,
        }[machine] if config.regime == "T" else 0
        if len(adaptation) != expected_adaptation:
            raise AssertionError(
                f"{machine} regime {config.regime} adaptation count is {len(adaptation)}, "
                f"expected {expected_adaptation}"
            )
        identity_rows = part[
            ["machine", "regime", "segment_id", "parent_recording", "split", "data_pool"]
        ].sort_values(["machine", "regime", "segment_id"]).to_dict("records")
        return cls(
            machine=machine,
            protocol=config.protocol,
            regime=config.regime,
            split_version=config.split_version,
            preprocessing_version=config.version,
            preprocessing_config_id=config.config_id,
            source_segment_ids=source,
            adaptation_segment_ids=adaptation,
            frozen_manifest_hash=content_sha256(identity_rows),
        )


@dataclass(frozen=True)
class EstimatorTrainingData:
    """Estimator input matrix plus non-feature reproducibility identities.

    There is deliberately no anomaly-label or target field. Callers pass only
    ``X`` to an estimator. The remaining fields are audit identities.
    """

    X: np.ndarray
    feature_names: tuple[str, ...]
    sample_ids: tuple[str, ...]
    segment_ids: tuple[str, ...]
    parent_recordings: tuple[str, ...]
    data_pools: tuple[str, ...]
    machine: str
    protocol: str
    regime: str
    preprocessing_config_id: str
    split_version: str
    frozen_manifest_hash: str


def _normal_mask(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower().eq("normal")


def load_estimator_training_data(
    frame: pd.DataFrame,
    contract: TrainingCohortContract,
    *,
    feature_columns: Iterable[str],
    require_complete_cohort: bool = True,
) -> EstimatorTrainingData:
    """Validate provenance and return a label-free estimator matrix."""

    features = tuple(feature_columns)
    if not features:
        raise ValueError("at least one explicit feature column is required")
    forbidden = set(features) & FORBIDDEN_MODEL_FEATURES
    if forbidden:
        raise ValueError(f"provenance/label fields cannot be estimator features: {sorted(forbidden)}")
    domain_column = "domain" if "domain" in frame.columns else "file_domain"
    required = {
        "machine", "protocol", "regime", "split_version", "preprocessing_version",
        "preprocessing_config_id",
        "split", "data_pool", "segment_id", "parent_recording", "anomaly_label",
        domain_column, *features,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"training frame is missing fields: {sorted(missing)}")
    if frame.empty:
        raise ValueError("training cohort cannot be empty")
    expected_scalars = {
        "machine": contract.machine,
        "protocol": contract.protocol,
        "regime": contract.regime,
        "split_version": contract.split_version,
        "preprocessing_version": contract.preprocessing_version,
        "preprocessing_config_id": contract.preprocessing_config_id,
    }
    for column, expected in expected_scalars.items():
        observed = set(frame[column].astype(str))
        if observed != {expected}:
            raise ValueError(f"wrong {column}: expected {expected!r}, found {sorted(observed)}")
    if not set(frame["split"].astype(str)) <= {"train"}:
        raise ValueError("model-fitting data may contain train rows only")
    if BLOCKED_SIBLING_SPLIT in set(frame["split"].astype(str)) or (
        BLOCKED_SIBLING_POOL in set(frame["data_pool"].astype(str))
    ):
        raise ValueError("blocked siblings are forbidden in model-fitting data")
    roles = set(frame["data_pool"].astype(str))
    if not roles <= contract.authorized_roles:
        raise ValueError(f"unauthorized training role: {sorted(roles - contract.authorized_roles)}")
    if not _normal_mask(frame["anomaly_label"]).all():
        raise ValueError("model-fitting data must be normal-only")
    source = frame["data_pool"].eq("source_training")
    adaptation = frame["data_pool"].eq("target_adaptation")
    if not frame.loc[source, domain_column].astype(str).eq("source").all():
        raise ValueError("source_training rows must be source-domain")
    if adaptation.any() and not frame.loc[adaptation, domain_column].astype(str).eq("target").all():
        raise ValueError("target_adaptation rows must be target-domain")
    segment_ids = tuple(frame["segment_id"].astype(str))
    observed_segments = frozenset(segment_ids)
    unauthorized = observed_segments - contract.authorized_segment_ids
    if unauthorized:
        raise ValueError(f"training rows contain unauthorized segment ids: {sorted(unauthorized)[:5]}")
    if require_complete_cohort and observed_segments != contract.authorized_segment_ids:
        missing_ids = contract.authorized_segment_ids - observed_segments
        raise ValueError(f"training data do not cover the complete frozen cohort: {len(missing_ids)} missing")
    if "window_id" in frame:
        sample_ids = tuple(frame["window_id"].astype(str))
    else:
        sample_ids = segment_ids
    if any(not item for item in sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("model-fitting sample identities must be present and unique")
    values = frame.loc[:, features].to_numpy(dtype=np.float64, copy=True)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("estimator features must be a finite two-dimensional matrix")
    return EstimatorTrainingData(
        X=values,
        feature_names=features,
        sample_ids=sample_ids,
        segment_ids=segment_ids,
        parent_recordings=tuple(frame["parent_recording"].astype(str)),
        data_pools=tuple(frame["data_pool"].astype(str)),
        machine=contract.machine,
        protocol=contract.protocol,
        regime=contract.regime,
        preprocessing_config_id=contract.preprocessing_config_id,
        split_version=contract.split_version,
        frozen_manifest_hash=contract.frozen_manifest_hash,
    )


@dataclass(frozen=True)
class ValidatedTrainingStandardizer:
    standardizer: FittedChannelStandardizer
    contract: TrainingCohortContract
    feature_names: tuple[str, ...]


def fit_training_standardizer(
    frame: pd.DataFrame,
    contract: TrainingCohortContract,
    *,
    feature_columns: Iterable[str],
    require_complete_cohort: bool = True,
) -> ValidatedTrainingStandardizer:
    """Public normalizer fit path; cohort validation always precedes fitting."""

    training = load_estimator_training_data(
        frame,
        contract,
        feature_columns=feature_columns,
        require_complete_cohort=require_complete_cohort,
    )
    fitted = FittedChannelStandardizer.fit(
        [training.X], split="train", segment_ids=training.segment_ids
    )
    return ValidatedTrainingStandardizer(fitted, contract, training.feature_names)


@dataclass(frozen=True)
class SourceValidationCohortContract:
    """Authorized source-normal validation identities for Phase-3.1 scoring."""

    machine: str
    protocol: str
    regime: str
    split_version: str
    preprocessing_version: str
    preprocessing_config_id: str
    segment_ids: frozenset[str]
    parent_recording_ids: frozenset[str]
    frozen_manifest_hash: str

    @classmethod
    def from_frozen_assignments(
        cls,
        assignments: pd.DataFrame,
        config: PreprocessingConfig,
        *,
        machine: str,
    ) -> "SourceValidationCohortContract":
        if config.protocol != MECHASIGHT_PROTOCOL:
            raise ValueError("Phase-3.1 validation requires Protocol B")
        if config.split_version != PROTOCOL_B_V3_SPLIT_VERSION:
            raise ValueError("Phase-3.1 validation refuses a stale split version")
        if config.version != PROTOCOL_B_V3_PREPROCESSING_VERSION:
            raise ValueError("Phase-3.1 validation refuses a stale preprocessing version")
        assert_few_shot_adaptation_cohorts(assignments)
        part = assignments[
            assignments["machine"].eq(machine)
            & assignments["regime"].eq(config.regime)
            & assignments["split"].eq("validation")
            & assignments["data_pool"].eq("source_validation")
        ].copy()
        if part.empty:
            raise ValueError("frozen assignments contain no source validation cohort")
        expected_segments = {
            "RoboticArm": ROBOTIC_ARM_VALIDATION_SEGMENTS,
            "BrushlessMotor": BRUSHLESS_MOTOR_VALIDATION_SEGMENTS,
        }[machine]
        if len(part) != expected_segments or part["segment_id"].nunique() != expected_segments:
            raise AssertionError(
                f"{machine} source validation must contain {expected_segments} segments"
            )
        parent_ids = frozenset(part["parent_recording"].astype(str))
        if machine == "RoboticArm" and len(parent_ids) != ROBOTIC_ARM_VALIDATION_PARENTS:
            raise AssertionError("RoboticArm source validation must contain exactly six parents")
        identity_rows = part[
            ["machine", "regime", "segment_id", "parent_recording", "split", "data_pool"]
        ].sort_values(["machine", "regime", "segment_id"]).to_dict("records")
        return cls(
            machine=machine,
            protocol=config.protocol,
            regime=config.regime,
            split_version=config.split_version,
            preprocessing_version=config.version,
            preprocessing_config_id=config.config_id,
            segment_ids=frozenset(part["segment_id"].astype(str)),
            parent_recording_ids=parent_ids,
            frozen_manifest_hash=content_sha256(identity_rows),
        )


@dataclass(frozen=True)
class EstimatorValidationData:
    X: np.ndarray
    feature_names: tuple[str, ...]
    sample_ids: tuple[str, ...]
    segment_ids: tuple[str, ...]
    parent_recordings: tuple[str, ...]
    anomaly_labels: tuple[str, ...]
    machine: str
    protocol: str
    regime: str
    preprocessing_config_id: str
    split_version: str
    frozen_manifest_hash: str


def load_source_validation_data(
    frame: pd.DataFrame,
    contract: SourceValidationCohortContract,
    *,
    feature_columns: Iterable[str],
    require_complete_cohort: bool = True,
) -> EstimatorValidationData:
    """Return only the authorized source-normal validation matrix."""

    features = tuple(feature_columns)
    if not features:
        raise ValueError("at least one explicit feature column is required")
    forbidden = set(features) & FORBIDDEN_MODEL_FEATURES
    if forbidden:
        raise ValueError(f"provenance/label fields cannot be estimator features: {sorted(forbidden)}")
    required = {
        "machine", "protocol", "regime", "split_version", "preprocessing_version",
        "preprocessing_config_id", "split", "data_pool", "segment_id",
        "parent_recording", "anomaly_label", "domain", *features,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"validation frame is missing fields: {sorted(missing)}")
    if frame.empty:
        raise ValueError("source validation cohort cannot be empty")
    expected_scalars = {
        "machine": contract.machine,
        "protocol": contract.protocol,
        "regime": contract.regime,
        "split_version": contract.split_version,
        "preprocessing_version": contract.preprocessing_version,
        "preprocessing_config_id": contract.preprocessing_config_id,
    }
    for column, expected in expected_scalars.items():
        observed = set(frame[column].astype(str))
        if observed != {expected}:
            raise ValueError(f"wrong {column}: expected {expected!r}, found {sorted(observed)}")
    if set(frame["split"].astype(str)) != {"validation"}:
        raise ValueError("Phase-3.1 scoring accepts validation rows only")
    if set(frame["data_pool"].astype(str)) != {"source_validation"}:
        raise ValueError("Phase-3.1 scoring accepts source_validation role only")
    if set(frame["domain"].astype(str)) != {"source"}:
        raise ValueError("Phase-3.1 validation must be source-domain only")
    if not _normal_mask(frame["anomaly_label"]).all():
        raise ValueError("Phase-3.1 validation must be normal-only")
    segment_ids = tuple(frame["segment_id"].astype(str))
    observed_segments = frozenset(segment_ids)
    unauthorized = observed_segments - contract.segment_ids
    if unauthorized:
        raise ValueError(f"validation rows contain unauthorized segment ids: {sorted(unauthorized)[:5]}")
    if require_complete_cohort and observed_segments != contract.segment_ids:
        missing_ids = contract.segment_ids - observed_segments
        raise ValueError(f"validation data do not cover the complete frozen cohort: {len(missing_ids)} missing")
    sample_ids = tuple(
        frame["window_id"].astype(str) if "window_id" in frame else frame["segment_id"].astype(str)
    )
    if any(not item for item in sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("validation sample identities must be present and unique")
    values = frame.loc[:, features].to_numpy(dtype=np.float64, copy=True)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("validation features must be a finite two-dimensional matrix")
    return EstimatorValidationData(
        X=values,
        feature_names=features,
        sample_ids=sample_ids,
        segment_ids=segment_ids,
        parent_recordings=tuple(frame["parent_recording"].astype(str)),
        anomaly_labels=tuple(frame["anomaly_label"].astype(str)),
        machine=contract.machine,
        protocol=contract.protocol,
        regime=contract.regime,
        preprocessing_config_id=contract.preprocessing_config_id,
        split_version=contract.split_version,
        frozen_manifest_hash=contract.frozen_manifest_hash,
    )
