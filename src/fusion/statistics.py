"""Phase-5 statistical procedures (design v2 §7-8), implemented and tested on synthetic data only.

* Threshold metrics at a frozen threshold - healthy-target FPR and recall - with
  the shared class-stratified parent-cluster draws (``make_common_bootstrap_draws``)
  and the frozen R8 seed averaging; returned as the same ``BootstrapDistribution``
  the Phase-3.2/Phase-4 ranking bootstrap returns, so paired contrasts reuse
  ``paired_distribution_difference`` unchanged.
* Ranking metrics reuse ``phase4_analysis.family_bootstrap`` unchanged.
* The v2 interpretation rules: per-metric category, operating axis, joint category.

Nothing in this module reads data; callers supply frames.  In Phase 5.3 it is
exercised only on synthetic fixtures - no protected metric is computed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from src.experiment.phase32 import BootstrapDistribution, CommonBootstrapDraws, paired_distribution_difference


THRESHOLD_METRICS = ("healthy_target_fpr", "recall")
FAVOURABLE = {"auroc": "higher", "average_precision": "higher", "standardized_pauroc_fpr_0_10": "higher",
              "recall": "higher", "healthy_target_fpr": "lower"}
MARGINS = {"auroc": 0.02, "average_precision": 0.02, "healthy_target_fpr": 0.10, "recall": 0.05}
MIN_VALID_FRACTION = 0.9            # 9000 of the frozen 10,000 replicates


def _denominator(frame: pd.DataFrame, metric: str) -> np.ndarray:
    """Evaluation units a threshold metric is defined over (healthy target normals, or anomalies)."""

    anomaly = frame["is_anomaly"].to_numpy().astype(bool)
    if metric == "healthy_target_fpr":
        return (~anomaly) & frame["domain"].eq("target").to_numpy()
    if metric == "recall":
        return anomaly
    raise ValueError(f"unknown threshold metric {metric!r}")


def threshold_metric_point(frame: pd.DataFrame, threshold: float, metric: str) -> float:
    denominator = _denominator(frame, metric)
    if not denominator.any():
        return float("nan")
    alarm = frame["score"].to_numpy(dtype=np.float64) > threshold
    return float(alarm[denominator].mean())


def threshold_metric_bootstrap(
    frames_by_seed: Mapping[int, pd.DataFrame],
    thresholds_by_seed: Mapping[int, float],
    *,
    seeds: tuple[int, ...],
    metric: str,
    draws: CommonBootstrapDraws,
) -> BootstrapDistribution:
    """Seed-averaged threshold metric per replicate (R8), alarm = score > frozen threshold."""

    if set(frames_by_seed) != set(seeds) or set(thresholds_by_seed) != set(seeds):
        raise ValueError("frames and thresholds must cover exactly the frozen seed set")
    ordered = {s: frames_by_seed[s].sort_values("segment_id").reset_index(drop=True) for s in seeds}
    base = ordered[seeds[0]]
    for s in seeds[1:]:
        if not (ordered[s]["segment_id"].equals(base["segment_id"])
                and ordered[s]["is_anomaly"].equals(base["is_anomaly"])
                and ordered[s]["domain"].equals(base["domain"])):
            raise ValueError("seed frames do not share identical evaluation units")
    weights = draws.sample_weights
    if weights.shape[1] != len(base):
        raise ValueError("bootstrap draws do not match the evaluation cohort")
    denominator = _denominator(base, metric)
    denominator_weight = weights[:, denominator].sum(axis=1, dtype=np.float64)
    per_seed, points = [], []
    for s in seeds:
        alarm = ordered[s]["score"].to_numpy(dtype=np.float64) > float(thresholds_by_seed[s])
        numerator = weights[:, denominator & alarm].sum(axis=1, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            per_seed.append(np.where(denominator_weight > 0, numerator / denominator_weight, np.nan))
        points.append(threshold_metric_point(ordered[s], float(thresholds_by_seed[s]), metric))
    values = np.mean(np.vstack(per_seed), axis=0)       # NaN propagates: invalid stays invalid
    valid = values[np.isfinite(values)]
    lower, upper = (None, None)
    if len(valid):
        lower, upper = map(float, np.quantile(valid, [0.025, 0.975]))
    return BootstrapDistribution(
        metric=metric, point_estimate=float(np.mean(points)), values=values,
        replicates_requested=draws.replicates, valid_replicates=int(len(valid)),
        invalid_replicates=int(draws.replicates - len(valid)), ci_lower=lower, ci_upper=upper,
        draw_identity=draws.identity,
    )


def paired_metric_difference(first: BootstrapDistribution, second: BootstrapDistribution) -> dict:
    """first - second on shared draws; the frozen Phase-3.2/Phase-4 paired difference."""

    return paired_distribution_difference(first, second, orientation=f"{first.metric}:first_minus_second")


# ---------------------------------------------------------- interpretation
@dataclass(frozen=True)
class MetricCategory:
    category: str            # EQUIVALENT | IMPROVEMENT | DEGRADATION | INDETERMINATE
    tag: str | None          # DETECTABLE_BUT_NEGLIGIBLE | MEANINGFUL | BELOW_MARGIN | None


def classify_metric(contrast: Mapping, *, metric: str, replicates_requested: int | None = None) -> MetricCategory:
    """v2 per-metric category; EQUIVALENT is tested first and needs the whole CI inside the margin."""

    margin, favourable = MARGINS[metric], FAVOURABLE[metric]
    lower, upper, point = contrast["ci_lower"], contrast["ci_upper"], contrast["point_delta"]
    requested = replicates_requested or contrast["replicates_requested"]
    if lower is None or upper is None or contrast["valid_replicates"] < MIN_VALID_FRACTION * requested:
        return MetricCategory("INDETERMINATE", None)
    if -margin < lower and upper < margin:
        excludes_zero = lower > 0 or upper < 0
        return MetricCategory("EQUIVALENT", "DETECTABLE_BUT_NEGLIGIBLE" if excludes_zero else None)
    sign = 1.0 if favourable == "higher" else -1.0
    if lower > 0 or upper < 0:
        favourable_side = (sign * lower > 0) if lower > 0 else (sign * upper > 0)
        size = "MEANINGFUL" if abs(point) >= margin else "BELOW_MARGIN"
        return MetricCategory("IMPROVEMENT" if favourable_side else "DEGRADATION", size)
    return MetricCategory("INDETERMINATE", None)


def operating_axis(hfpr: MetricCategory, recall: MetricCategory) -> str:
    cats = {hfpr.category, recall.category}
    if "IMPROVEMENT" in cats and "DEGRADATION" in cats:
        return "O_tradeoff"
    if "INDETERMINATE" in cats:
        return "O?"
    if "IMPROVEMENT" in cats:
        return "O+"
    if "DEGRADATION" in cats:
        return "O-"
    return "O0"


def joint_category(ranking: MetricCategory, operating: str) -> str:
    r = ranking.category
    if r == "INDETERMINATE" or operating == "O?":
        return "INDETERMINATE"
    if r == "IMPROVEMENT":
        return {"O+": "BOTH_IMPROVED", "O0": "RANKING_GAIN_OPERATION_PRESERVED"}.get(operating, "TRADE_OFF")
    if r == "EQUIVALENT":
        return {"O+": "OPERATING_GAIN_RANKING_PRESERVED", "O0": "EQUIVALENT", "O-": "UNDERPERFORMANCE",
                "O_tradeoff": "TRADE_OFF"}[operating]
    return "TRADE_OFF" if operating == "O+" else "UNDERPERFORMANCE"
