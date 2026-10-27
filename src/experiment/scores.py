"""Structured anomaly-score provenance and safe threshold selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from src.evaluation.policy import normal_quantile_threshold
from src.experiment.constants import (
    SCORE_DIRECTION,
    THRESHOLD_POLICY_VERSION,
)
from src.experiment.provenance import content_sha256, ordered_id_hash


@dataclass(frozen=True)
class ScoreProvenance:
    experiment_id: str
    machine: str
    protocol: str
    regime: str
    model_id: str
    model_family: str
    model_seed: int
    preprocessing_config_id: str
    preprocessing_config_hash: str
    split_version: str
    score_direction: str
    data_provenance: Mapping[str, str]
    config_provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        required_text = (
            self.experiment_id, self.machine, self.protocol, self.regime,
            self.model_id, self.model_family, self.preprocessing_config_id,
            self.preprocessing_config_hash, self.split_version,
        )
        if any(not str(value).strip() for value in required_text):
            raise ValueError("score provenance fields must be non-empty")
        if self.regime not in {"S", "T"}:
            raise ValueError("score provenance regime must be S or T")
        if self.score_direction != SCORE_DIRECTION:
            raise ValueError(f"score direction must be {SCORE_DIRECTION!r}")
        if not self.data_provenance or not self.config_provenance:
            raise ValueError("score data/config provenance must be present")

    @property
    def identity(self) -> tuple:
        return (
            self.experiment_id, self.machine, self.protocol, self.regime,
            self.model_id, self.model_family, self.model_seed,
            self.preprocessing_config_id, self.preprocessing_config_hash,
            self.split_version, self.score_direction,
            content_sha256(dict(self.data_provenance)),
            content_sha256(dict(self.config_provenance)),
        )


@dataclass(frozen=True)
class ScoreBatch:
    provenance: ScoreProvenance
    scores: np.ndarray
    segment_ids: tuple[str, ...]
    parent_recording_ids: tuple[str, ...]
    cohort_roles: tuple[str, ...]
    domains: tuple[str, ...]
    anomaly_labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        scores = np.asarray(self.scores, dtype=np.float64).reshape(-1)
        object.__setattr__(self, "scores", scores)
        lengths = {
            len(scores), len(self.segment_ids), len(self.parent_recording_ids),
            len(self.cohort_roles), len(self.domains),
        }
        if self.anomaly_labels is not None:
            lengths.add(len(self.anomaly_labels))
        if lengths != {len(scores)} or len(scores) == 0:
            raise ValueError("score values and per-segment provenance must have one non-empty length")
        if not np.all(np.isfinite(scores)):
            raise ValueError("anomaly scores must be finite")
        if any(not str(value) for value in self.segment_ids):
            raise ValueError("segment identities must be present")
        if len(set(self.segment_ids)) != len(self.segment_ids):
            raise ValueError("score batches require unique segment identities")
        if any(not str(value) for value in self.parent_recording_ids):
            raise ValueError("parent-recording identities must be present")

    @property
    def cohort_hash(self) -> str:
        return ordered_id_hash(self.segment_ids)

    @property
    def is_anomaly(self) -> np.ndarray:
        if self.anomaly_labels is None:
            raise ValueError("labelled evaluation scores are required")
        return np.asarray(
            [str(label).lower() != "normal" for label in self.anomaly_labels], dtype=bool
        )


def assert_matching_score_provenance(
    observed: ScoreProvenance, expected: ScoreProvenance
) -> None:
    if observed.identity != expected.identity:
        raise ValueError("score batch has stale or wrong experiment provenance")


@dataclass(frozen=True)
class ThresholdSelection:
    value: float
    policy_version: str
    quantile: float
    quantile_method: str
    score_direction: str
    source_experiment_id: str
    source_cohort_hash: str
    source_segment_ids: tuple[str, ...]
    source_role: str


def select_protocol_b_threshold(
    batch: ScoreBatch,
    *,
    expected_provenance: ScoreProvenance,
) -> ThresholdSelection:
    """Select q90 only from the frozen source-normal validation cohort."""

    assert_matching_score_provenance(batch.provenance, expected_provenance)
    if batch.provenance.score_direction != SCORE_DIRECTION:
        raise ValueError("threshold selection requires higher-is-more-anomalous scores")
    expected_cohort_hash = batch.provenance.data_provenance.get(
        "source_validation_cohort_hash"
    )
    if not expected_cohort_hash or batch.cohort_hash != expected_cohort_hash:
        raise ValueError("threshold segment ids do not match the frozen source-validation cohort")
    roles = set(batch.cohort_roles)
    if roles != {"source_validation"}:
        raise ValueError("threshold source must be a pure source_validation cohort")
    if set(batch.domains) != {"source"}:
        raise ValueError("threshold source must be source-domain only")
    if batch.anomaly_labels is None:
        raise ValueError("threshold provenance requires explicit normal labels")
    if any(str(label).lower() != "normal" for label in batch.anomaly_labels):
        raise ValueError("threshold source must contain normal scores only")
    threshold = normal_quantile_threshold(batch.scores, quantile=0.90)
    return ThresholdSelection(
        value=threshold,
        policy_version=THRESHOLD_POLICY_VERSION,
        quantile=0.90,
        quantile_method="higher",
        score_direction=SCORE_DIRECTION,
        source_experiment_id=batch.provenance.experiment_id,
        source_cohort_hash=batch.cohort_hash,
        source_segment_ids=batch.segment_ids,
        source_role="source_validation",
    )
