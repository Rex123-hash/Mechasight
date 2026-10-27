"""Primary class-stratified cluster bootstrap for Phase 3 inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

import numpy as np
import pandas as pd

from src.experiment.constants import (
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_INFERENCE_SCOPE,
    BOOTSTRAP_POLICY_VERSION,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    PRIMARY_BOOTSTRAP_STRATA,
    PRIMARY_REPORTING_SLICES,
)
from src.experiment.metrics import ranking_metric_value


@dataclass(frozen=True)
class BootstrapResult:
    metric: str
    point_estimate: float | None
    confidence_level: float
    ci_lower: float | None
    ci_upper: float | None
    number_requested: int
    number_valid: int
    number_invalid: int
    reporting_slice: tuple[tuple[str, str], ...]
    resampling_unit: str
    design: str = BOOTSTRAP_POLICY_VERSION
    inference_scope: str = BOOTSTRAP_INFERENCE_SCOPE
    strata: tuple[str, ...] = PRIMARY_BOOTSTRAP_STRATA
    ci_method: str = "percentile"


def _validate_primary_frame(
    frame: pd.DataFrame,
    *,
    parent_column: str,
    label_column: str,
) -> tuple[tuple[tuple[str, str], ...], str]:
    required = {parent_column, label_column, *PRIMARY_REPORTING_SLICES}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"bootstrap frame missing fields: {sorted(missing)}")
    if frame.empty:
        raise ValueError("bootstrap frame cannot be empty")

    reporting_slice: list[tuple[str, str]] = []
    for column in PRIMARY_REPORTING_SLICES:
        values = frame[column].drop_duplicates()
        if len(values) != 1:
            raise ValueError(
                f"primary bootstrap requires evaluation to be sliced by {column} before resampling"
            )
        reporting_slice.append((column, str(values.iloc[0])))

    homogeneous = frame.groupby(parent_column)[label_column].nunique(dropna=False)
    if (homogeneous > 1).any():
        raise ValueError("a parent recording spans anomaly classes")

    machine = dict(reporting_slice)["machine"]
    if machine == "RoboticArm":
        resampling_unit = "physical_parent_recording"
    elif machine == "BrushlessMotor":
        cluster_sizes = frame.groupby(parent_column).size()
        if not cluster_sizes.eq(1).all():
            raise ValueError(
                "BrushlessMotor primary bootstrap requires the audited one-segment-per-recording structure"
            )
        resampling_unit = "released_recording_segment_one_to_one"
    else:
        raise ValueError(f"unsupported machine for Phase-3 primary bootstrap: {machine!r}")
    return tuple(reporting_slice), resampling_unit


def draw_class_stratified_cluster_ids(
    frame: pd.DataFrame,
    *,
    rng: np.random.Generator,
    parent_column: str = "parent_recording",
    label_column: str = "is_anomaly",
) -> dict[Hashable, tuple[str, ...]]:
    """Sample parent/recording clusters within class, preserving class cluster counts."""

    parents = frame[[parent_column, label_column]].drop_duplicates()
    draws: dict[Hashable, tuple[str, ...]] = {}
    for label, class_parents in parents.groupby(label_column, sort=True, dropna=False):
        ids = class_parents[parent_column].astype(str).to_numpy()
        draws[label] = tuple(str(value) for value in rng.choice(ids, size=len(ids), replace=True))
    return draws


def draw_stratified_cluster_indices(
    frame: pd.DataFrame,
    *,
    rng: np.random.Generator,
    parent_column: str = "parent_recording",
    label_column: str = "is_anomaly",
) -> np.ndarray:
    """Draw class-stratified clusters and expand every draw to all child segments."""

    draws = draw_class_stratified_cluster_ids(
        frame, rng=rng, parent_column=parent_column, label_column=label_column
    )
    parent_values = frame[parent_column].astype(str)
    indices: list[object] = []
    for selected_ids in draws.values():
        for selected in selected_ids:
            indices.extend(frame.index[parent_values.eq(selected)].tolist())
    return np.asarray(indices)


def _interval(values: list[float], confidence: float) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    alpha = 1.0 - confidence
    lower, upper = np.quantile(np.asarray(values), [alpha / 2, 1 - alpha / 2])
    return float(lower), float(upper)


def bootstrap_metric(
    frame: pd.DataFrame,
    *,
    metric: str,
    score_column: str = "score",
    label_column: str = "is_anomaly",
    parent_column: str = "parent_recording",
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = BOOTSTRAP_CONFIDENCE,
) -> BootstrapResult:
    reporting_slice, resampling_unit = _validate_primary_frame(
        frame, parent_column=parent_column, label_column=label_column
    )
    if score_column not in frame:
        raise ValueError(f"bootstrap score column {score_column!r} is missing")
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    try:
        point = ranking_metric_value(metric, frame[label_column], frame[score_column])
    except ValueError:
        point = None
    rng = np.random.default_rng(seed)
    values: list[float] = []
    invalid = 0
    for _ in range(replicates):
        sample = frame.loc[draw_stratified_cluster_indices(
            frame, rng=rng, parent_column=parent_column, label_column=label_column
        )]
        try:
            values.append(ranking_metric_value(metric, sample[label_column], sample[score_column]))
        except ValueError:
            invalid += 1
    lower, upper = _interval(values, confidence)
    return BootstrapResult(
        metric=metric,
        point_estimate=point,
        confidence_level=confidence,
        ci_lower=lower,
        ci_upper=upper,
        number_requested=replicates,
        number_valid=len(values),
        number_invalid=invalid,
        reporting_slice=reporting_slice,
        resampling_unit=resampling_unit,
    )


def bootstrap_paired_difference(
    frame: pd.DataFrame,
    *,
    metric: str,
    score_a_column: str = "score_a",
    score_b_column: str = "score_b",
    label_column: str = "is_anomaly",
    parent_column: str = "parent_recording",
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = BOOTSTRAP_CONFIDENCE,
) -> BootstrapResult:
    reporting_slice, resampling_unit = _validate_primary_frame(
        frame, parent_column=parent_column, label_column=label_column
    )
    for column in (score_a_column, score_b_column):
        if column not in frame:
            raise ValueError(f"paired score column {column!r} is missing")
    try:
        point = ranking_metric_value(metric, frame[label_column], frame[score_a_column]) - (
            ranking_metric_value(metric, frame[label_column], frame[score_b_column])
        )
    except ValueError:
        point = None
    rng = np.random.default_rng(seed)
    values: list[float] = []
    invalid = 0
    for _ in range(replicates):
        # One draw is expanded once; both models receive identical rows and multiplicities.
        sample = frame.loc[draw_stratified_cluster_indices(
            frame, rng=rng, parent_column=parent_column, label_column=label_column
        )]
        try:
            a = ranking_metric_value(metric, sample[label_column], sample[score_a_column])
            b = ranking_metric_value(metric, sample[label_column], sample[score_b_column])
            values.append(a - b)
        except ValueError:
            invalid += 1
    lower, upper = _interval(values, confidence)
    return BootstrapResult(
        metric=f"delta_{metric}",
        point_estimate=point,
        confidence_level=confidence,
        ci_lower=lower,
        ci_upper=upper,
        number_requested=replicates,
        number_valid=len(values),
        number_invalid=invalid,
        reporting_slice=reporting_slice,
        resampling_unit=resampling_unit,
    )
