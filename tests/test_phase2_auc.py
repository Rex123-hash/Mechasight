import numpy as np

from src.evaluation.auc import official_pairwise_auc


def test_official_auc_gives_ties_zero_credit():
    assert official_pairwise_auc(np.array([1.0]), np.array([1.0])) == 0.0


def test_official_auc_is_pairwise_strict_less_than_fraction():
    normal = np.array([0.0, 2.0])
    anomaly = np.array([1.0, 3.0])
    assert official_pairwise_auc(normal, anomaly) == 0.75
