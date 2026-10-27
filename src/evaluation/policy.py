"""Frozen Phase-2.1 comparison, threshold, and aggregation invariants."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.preprocessing.config import PreprocessingConfig


@dataclass(frozen=True)
class PairedExperimentSpec:
    regime: str
    training_pool_id: str
    adaptation_access: bool
    test_segment_ids: frozenset[str]
    test_parent_recordings: frozenset[str]
    anomaly_test_segment_ids: frozenset[str]
    preprocessing_signature: str
    window_configuration: str
    feature_configuration: str
    model_configuration: str
    score_aggregation: str
    metric_implementation: str


def assert_paired_experiment_specs(
    regime_s: PairedExperimentSpec, regime_t: PairedExperimentSpec
) -> None:
    """The only permitted difference is target-adaptation access/training pool."""

    if (regime_s.regime, regime_t.regime) != ("S", "T"):
        raise AssertionError("paired specs must be ordered Regime S then Regime T")
    invariant_fields = (
        "test_segment_ids", "test_parent_recordings", "anomaly_test_segment_ids",
        "preprocessing_signature", "window_configuration", "feature_configuration",
        "model_configuration", "score_aggregation", "metric_implementation",
    )
    for field in invariant_fields:
        if getattr(regime_s, field) != getattr(regime_t, field):
            raise AssertionError(f"paired S/T invariant differs: {field}")
    if regime_s.adaptation_access or not regime_t.adaptation_access:
        raise AssertionError("S must reserve and T must access target adaptation data")
    if regime_s.training_pool_id == regime_t.training_pool_id:
        raise AssertionError("S/T training pool identities must record adaptation difference")


def preprocessing_pair_signature(config: PreprocessingConfig) -> str:
    """Canonical comparison signature excluding only the regime label."""

    data = config.canonical_dict()
    data.pop("regime")
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def assert_preprocessing_pair(
    regime_s: PreprocessingConfig, regime_t: PreprocessingConfig
) -> None:
    if (regime_s.regime, regime_t.regime) != ("S", "T"):
        raise AssertionError("expected Regime S and Regime T preprocessing configs")
    if preprocessing_pair_signature(regime_s) != preprocessing_pair_signature(regime_t):
        raise AssertionError("S/T preprocessing differs beyond the regime field")


def normal_quantile_threshold(
    normal_segment_scores: np.ndarray, *, quantile: float = 0.90
) -> float:
    """Unsupervised threshold from permitted normal calibration scores only."""

    scores = np.asarray(normal_segment_scores, dtype=np.float64).ravel()
    if scores.size == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("finite normal calibration scores are required")
    if not 0 < quantile < 1:
        raise ValueError("quantile must be strictly between zero and one")
    return float(np.quantile(scores, quantile, method="higher"))


def aggregate_segment_scores(
    window_ids: list[str], window_scores: np.ndarray, *, method: str = "median"
) -> dict[str, Any]:
    """Aggregate linked windows without discarding their provenance."""

    scores = np.asarray(window_scores, dtype=np.float64).ravel()
    if len(window_ids) != scores.size or scores.size == 0:
        raise ValueError("one non-empty score is required per contributing window")
    if not np.all(np.isfinite(scores)):
        raise ValueError("window scores must be finite")
    if method != "median":
        raise ValueError("Phase-2.1 freezes Protocol-B primary aggregation to median")
    return {
        "segment_score": float(np.median(scores)),
        "aggregation": method,
        "contributing_window_ids": tuple(window_ids),
        "n_windows": int(scores.size),
    }


def load_evaluation_policy(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def draw_parent_cluster_sample(
    frame: pd.DataFrame,
    *,
    seed: int,
    parent_column: str = "parent_recording",
    strata: tuple[str, ...] = ("domain", "is_anomaly", "operating_condition"),
) -> pd.DataFrame:
    """Draw one stratified parent bootstrap and retain every child segment.

    Duplicate parent draws receive distinct ``bootstrap_cluster_instance``
    values.  This is a sampling primitive only; Phase 2.1 computes no model
    statistic.
    """

    required = {parent_column, *strata}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"cluster bootstrap fields missing: {sorted(missing)}")
    homogeneity = frame.groupby(parent_column)[list(strata)].nunique(dropna=False)
    if (homogeneity > 1).any().any():
        raise ValueError("a parent recording spans bootstrap strata")
    parents = frame[[parent_column, *strata]].drop_duplicates()
    rng = np.random.default_rng(seed)
    outputs = []
    instance = 0
    group_key: str | list[str] = list(strata) if len(strata) > 1 else strata[0]
    for _, cell in parents.groupby(group_key, sort=True, dropna=False):
        parent_ids = cell[parent_column].astype(str).to_numpy()
        for selected in rng.choice(parent_ids, size=len(parent_ids), replace=True):
            child = frame[frame[parent_column].astype(str).eq(str(selected))].copy()
            child["bootstrap_cluster_instance"] = instance
            child["bootstrap_parent_recording"] = str(selected)
            outputs.append(child)
            instance += 1
    return pd.concat(outputs, ignore_index=True) if outputs else frame.iloc[0:0].copy()
