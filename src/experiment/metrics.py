"""Protocol-B metric definitions backed by scikit-learn 1.9.0."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.experiment.constants import METRIC_POLICY_VERSION


RANKING_METRICS = (
    "auroc",
    "average_precision",
    "standardized_pauroc_fpr_0_10",
)


def _validated_binary(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.int8).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y.size == 0 or y.size != s.size:
        raise ValueError("one non-empty score is required per binary label")
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError("labels must be binary 0/1")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite")
    return y, s


def protocol_b_ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | str]:
    y, s = _validated_binary(labels, scores)
    if len(np.unique(y)) != 2:
        raise ValueError("ranking metrics require both normal and anomaly classes")
    return {
        "metric_policy_version": METRIC_POLICY_VERSION,
        "auroc": float(roc_auc_score(y, s)),
        "average_precision": float(average_precision_score(y, s)),
        # sklearn's max_fpr API returns McClish-standardized partial AUC,
        # not the raw geometric area under the truncated ROC curve.
        "standardized_pauroc_fpr_0_10": float(roc_auc_score(y, s, max_fpr=0.10)),
    }


def ranking_metric_value(name: str, labels: np.ndarray, scores: np.ndarray) -> float:
    if name not in RANKING_METRICS:
        raise ValueError(f"unknown ranking metric {name!r}")
    return float(protocol_b_ranking_metrics(labels, scores)[name])


@dataclass(frozen=True)
class ThresholdMetrics:
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int
    false_positive_rate: float | None
    recall: float | None
    precision: float | None
    f1: float | None
    decision_rule: str = "anomaly if score > threshold"

    def to_dict(self) -> dict:
        return asdict(self)


def protocol_b_threshold_metrics(
    labels: np.ndarray, scores: np.ndarray, *, threshold: float
) -> ThresholdMetrics:
    y, s = _validated_binary(labels, scores)
    predicted = s > float(threshold)
    normal = y == 0
    anomaly = y == 1
    tn = int(np.sum(normal & ~predicted))
    fp = int(np.sum(normal & predicted))
    fn = int(np.sum(anomaly & ~predicted))
    tp = int(np.sum(anomaly & predicted))
    fpr = fp / (fp + tn) if fp + tn else None
    recall = tp / (tp + fn) if tp + fn else None
    precision = tp / (tp + fp) if tp + fp else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    return ThresholdMetrics(tn, fp, fn, tp, fpr, recall, precision, f1)
