"""Strict Phase-3 experiment configuration and deterministic identity."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Mapping

from src.experiment.constants import (
    ADAPTATION_POLICY_VERSION,
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_INFERENCE_SCOPE,
    BOOTSTRAP_POLICY_VERSION,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    BRUSHLESS_MOTOR_ACQUISITION_SESSION_CONFOUND_ID,
    BRUSHLESS_MOTOR_LIMITATION,
    EXPERIMENT_SCHEMA_VERSION,
    METRIC_POLICY_VERSION,
    MODEL_SEEDS,
    PRIMARY_BOOTSTRAP_STRATA,
    PRIMARY_REPORTING_SLICES,
    ROBOTIC_ARM_ADAPTATION_LIMITATION,
    ROBOTIC_ARM_ADAPTATION_PARENT_IDS,
    ROBOTIC_ARM_ADAPTATION_PARENT_LIMITATION_ID,
    ROBOTIC_ARM_ADAPTATION_PARENTS,
    ROBOTIC_ARM_ADAPTATION_SEGMENTS,
    ROBOTIC_ARM_ADAPTATION_CLOSURE_SEGMENTS,
    ROBOTIC_ARM_BLOCKED_SIBLING_SEGMENTS,
    ROBOTIC_ARM_VALIDATION_PARENT_IDS,
    ROBOTIC_ARM_VALIDATION_PARENT_LIMITATION_ID,
    ROBOTIC_ARM_VALIDATION_LIMITATION,
    ROBOTIC_ARM_VALIDATION_PARENTS,
    ROBOTIC_ARM_VALIDATION_SEGMENTS,
    SCORE_DIRECTION,
    THRESHOLD_POLICY_VERSION,
)
from src.experiment.provenance import content_sha256
from src.preprocessing.config import VALID_PROTOCOLS, VALID_REGIMES


@dataclass(frozen=True)
class BootstrapConfig:
    replicates: int = BOOTSTRAP_REPLICATES
    seed: int = BOOTSTRAP_SEED
    confidence: float = BOOTSTRAP_CONFIDENCE
    ci_method: str = "percentile"
    paired: bool = True
    policy_version: str = BOOTSTRAP_POLICY_VERSION
    method: str = "class_stratified_cluster_bootstrap"
    strata: tuple[str, ...] = PRIMARY_BOOTSTRAP_STRATA
    reporting_slices: tuple[str, ...] = PRIMARY_REPORTING_SLICES
    inference_scope: str = BOOTSTRAP_INFERENCE_SCOPE

    def __post_init__(self) -> None:
        if self.replicates <= 0 or not 0 < self.confidence < 1:
            raise ValueError("invalid bootstrap configuration")
        if self.seed != BOOTSTRAP_SEED:
            raise ValueError("Phase-3 bootstrap seed is frozen to 1995")
        if self.ci_method != "percentile" or not self.paired:
            raise ValueError("Phase-3 freezes paired percentile bootstrap inference")
        frozen = (
            self.policy_version == BOOTSTRAP_POLICY_VERSION
            and self.method == "class_stratified_cluster_bootstrap"
            and tuple(self.strata) == PRIMARY_BOOTSTRAP_STRATA
            and tuple(self.reporting_slices) == PRIMARY_REPORTING_SLICES
            and self.inference_scope == BOOTSTRAP_INFERENCE_SCOPE
        )
        if not frozen:
            raise ValueError("Phase-3.0.1 primary bootstrap design is frozen")


@dataclass(frozen=True)
class ExperimentConfig:
    timestamp_utc: str
    machine: str
    protocol: str
    regime: str
    model_family: str
    model_hyperparameters: Mapping[str, Any]
    model_seed: int
    preprocessing_config_id: str
    preprocessing_config_hash: str
    split_version: str
    split_manifest_hash: str
    dataset_release_identity: str
    feature_representation: str
    feature_version: str
    window_length_s: float
    stride_s: float
    normalization_strategy: str
    code_git_sha: str
    dependency_environment: Mapping[str, str]
    adaptation_policy_version: str = ADAPTATION_POLICY_VERSION
    anomaly_score_direction: str = SCORE_DIRECTION
    segment_aggregation_rule: str = "median"
    threshold_policy_version: str = THRESHOLD_POLICY_VERSION
    metric_policy_version: str = METRIC_POLICY_VERSION
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    schema_version: str = EXPERIMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EXPERIMENT_SCHEMA_VERSION:
            raise ValueError("unsupported experiment schema version")
        try:
            datetime.fromisoformat(self.timestamp_utc.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("timestamp_utc must be ISO-8601") from error
        if self.protocol not in VALID_PROTOCOLS or self.regime not in VALID_REGIMES:
            raise ValueError("experiment protocol/regime is invalid")
        if self.model_seed not in MODEL_SEEDS:
            raise ValueError(f"model seed must be one of {list(MODEL_SEEDS)}")
        if self.anomaly_score_direction != SCORE_DIRECTION:
            raise ValueError("Phase-3 scores must use higher-is-more-anomalous direction")
        if self.segment_aggregation_rule != "median":
            raise ValueError("Phase-3 segment aggregation is frozen to median")
        if self.window_length_s <= 0 or self.stride_s <= 0:
            raise ValueError("window length and stride must be positive")
        required_text = (
            self.machine, self.model_family, self.preprocessing_config_id,
            self.preprocessing_config_hash, self.split_version, self.split_manifest_hash,
            self.dataset_release_identity, self.feature_representation, self.feature_version,
            self.normalization_strategy, self.code_git_sha,
        )
        if any(not str(value).strip() for value in required_text):
            raise ValueError("experiment identity fields must be non-empty")
        if not self.dependency_environment:
            raise ValueError("dependency/environment provenance is required")

    def scientific_identity_payload(self) -> dict[str, Any]:
        """Fields that determine the scientific computation, excluding run time."""

        data = asdict(self)
        data.pop("timestamp_utc")
        # Environment is persisted but code/dependency patch-location metadata does
        # not rename an otherwise identical scientific configuration.
        data.pop("dependency_environment")
        return data

    @property
    def experiment_id(self) -> str:
        return f"p3-{content_sha256(self.scientific_identity_payload())[:24]}"

    @property
    def scientific_limitations(self) -> tuple[str, ...]:
        limitations = []
        if self.machine == "BrushlessMotor":
            limitations.append(BRUSHLESS_MOTOR_LIMITATION)
        if self.machine == "RoboticArm":
            limitations.append(ROBOTIC_ARM_VALIDATION_LIMITATION)
            limitations.append(ROBOTIC_ARM_ADAPTATION_LIMITATION)
        return tuple(limitations)

    @property
    def scientific_limitation_ids(self) -> tuple[str, ...]:
        if self.machine == "BrushlessMotor":
            return (BRUSHLESS_MOTOR_ACQUISITION_SESSION_CONFOUND_ID,)
        if self.machine == "RoboticArm":
            return (
                ROBOTIC_ARM_VALIDATION_PARENT_LIMITATION_ID,
                ROBOTIC_ARM_ADAPTATION_PARENT_LIMITATION_ID,
            )
        return ()

    @property
    def cohort_provenance(self) -> dict[str, Any]:
        if self.machine != "RoboticArm":
            return {}
        return {
            "source_validation": {
                "segment_count": ROBOTIC_ARM_VALIDATION_SEGMENTS,
                "parent_count": ROBOTIC_ARM_VALIDATION_PARENTS,
                "parent_ids": list(ROBOTIC_ARM_VALIDATION_PARENT_IDS),
            },
            "target_adaptation": {
                "released_segment_count": ROBOTIC_ARM_ADAPTATION_SEGMENTS,
                "parent_count": ROBOTIC_ARM_ADAPTATION_PARENTS,
                "parent_ids": list(ROBOTIC_ARM_ADAPTATION_PARENT_IDS),
                "blocked_sibling_segment_count": ROBOTIC_ARM_BLOCKED_SIBLING_SEGMENTS,
                "parent_closure_segment_count": ROBOTIC_ARM_ADAPTATION_CLOSURE_SEGMENTS,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["experiment_id"] = self.experiment_id
        data["scientific_limitations"] = list(self.scientific_limitations)
        data["scientific_limitation_ids"] = list(self.scientific_limitation_ids)
        data["cohort_provenance"] = self.cohort_provenance
        return data
