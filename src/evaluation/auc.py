"""Metric compatibility helpers; no model evaluation is run in Phase 2."""

from __future__ import annotations

import numpy as np


def official_pairwise_auc(normal_scores: np.ndarray, anomaly_scores: np.ndarray) -> float:
    """Pinned upstream AUC: mean(normal < anomaly), with exact ties worth zero."""

    normal = np.asarray(normal_scores, dtype=np.float64).ravel()
    anomaly = np.asarray(anomaly_scores, dtype=np.float64).ravel()
    if normal.size == 0 or anomaly.size == 0:
        raise ValueError("both normal and anomaly scores are required")
    return float(np.mean(normal[:, None] < anomaly[None, :]))
