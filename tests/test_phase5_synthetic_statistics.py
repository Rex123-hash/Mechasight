"""Phase 5.5C: non-degenerate SYNTHETIC check of the frozen bootstrap statistics.

The 5.5B fixture separated its fake classes perfectly, so every AUROC contrast was
trivially equivalent with zero-width intervals. Here class scores overlap, parents
hold several segments, two seeds are averaged, and replicate values are recomputed
independently (row expansion by bootstrap multiplicity + scikit-learn). No MechaSight
data is read; nothing here is a performance claim.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.experiment.phase32 import make_common_bootstrap_draws, paired_distribution_difference
from src.experiment.phase4_analysis import family_bootstrap
from src.fusion.statistics import threshold_metric_bootstrap

REPLICATES = 2_000


def _cohort():
    rng = np.random.default_rng(20261025)
    rows = []
    for parent in range(16):                       # 8 normal + 8 fault parents, 3 segments each
        anomaly = int(parent >= 8)
        for k in range(3):
            rows.append({"segment_id": f"SYN-{parent:02}-{k}", "parent_recording": f"SYN-P{parent:02}",
                         "machine": "RoboticArm", "is_anomaly": anomaly,
                         "domain": "target" if parent % 4 in (0, 1) else "source"})
    frame = pd.DataFrame(rows).sort_values("segment_id").reset_index(drop=True)
    base = rng.normal(0.0, 1.0, len(frame)) + 0.9 * frame["is_anomaly"].to_numpy()   # overlapping classes
    seeds = {s: frame.assign(score=base + rng.normal(0.0, 0.3, len(frame))) for s in (0, 1)}
    other = {s: frame.assign(score=base + 0.4 * frame["is_anomaly"].to_numpy() + rng.normal(0.0, 0.3, len(frame)))
             for s in (0, 1)}
    return frame, seeds, other


@pytest.fixture(scope="module")
def setup():
    frame, seeds, other = _cohort()
    draws = make_common_bootstrap_draws(frame, replicates=REPLICATES, seed=1995)
    return frame, seeds, other, draws


def _expanded_auroc(labels, scores, weights):
    keep = np.repeat(np.arange(len(labels)), weights.astype(int))
    if len(set(labels[keep])) < 2:
        return np.nan
    return roc_auc_score(labels[keep], scores[keep])


def test_draws_are_parent_aware_and_class_stratified(setup):
    frame, _, _, draws = setup
    weights = draws.sample_weights
    for _, group in frame.groupby("parent_recording"):
        columns = weights[:, group.index.to_numpy()]
        assert (columns == columns[:, :1]).all()                    # one multiplicity per parent
    for anomaly in (0, 1):
        members = frame.index[frame["is_anomaly"].eq(anomaly)].to_numpy()
        assert (weights[:, members].sum(axis=1) == len(members)).all()   # class size preserved


def test_seed_averaged_auroc_replicates_match_an_independent_computation(setup):
    frame, seeds, _, draws = setup
    result = family_bootstrap(seeds, seeds=(0, 1), metric="auroc", draws=draws)
    labels = frame["is_anomaly"].to_numpy()
    for r in range(40):
        expected = np.mean([_expanded_auroc(labels, seeds[s]["score"].to_numpy(), draws.sample_weights[r])
                            for s in (0, 1)])
        assert result.values[r] == pytest.approx(expected, abs=1e-12)
    assert result.point_estimate == pytest.approx(np.mean([roc_auc_score(labels, seeds[s]["score"]) for s in (0, 1)]))
    assert 0.5 < result.point_estimate < 1.0 and result.ci_upper - result.ci_lower > 0.05


def test_paired_contrasts_have_real_uncertainty_and_self_contrast_is_zero(setup):
    _, seeds, other, draws = setup
    a = family_bootstrap(seeds, seeds=(0, 1), metric="auroc", draws=draws)
    b = family_bootstrap(other, seeds=(0, 1), metric="auroc", draws=draws)
    difference = paired_distribution_difference(b, a, orientation="first_minus_second")
    assert difference["ci_upper"] - difference["ci_lower"] > 0.01
    assert difference["point_delta"] > 0
    same = paired_distribution_difference(a, a, orientation="first_minus_second")
    assert same["ci_lower"] == same["ci_upper"] == 0.0


def test_healthy_target_fpr_replicates_match_a_manual_weighted_count(setup):
    frame, seeds, _, draws = setup
    thresholds = {0: 0.3, 1: 0.35}
    result = threshold_metric_bootstrap(seeds, thresholds, seeds=(0, 1), metric="healthy_target_fpr", draws=draws)
    healthy = (frame["is_anomaly"].eq(0) & frame["domain"].eq("target")).to_numpy()
    for r in range(40):
        w = draws.sample_weights[r].astype(float)
        if w[healthy].sum() == 0:
            assert np.isnan(result.values[r])
            continue
        per_seed = [(w * (healthy & (seeds[s]["score"].to_numpy() > thresholds[s]))).sum() / w[healthy].sum()
                    for s in (0, 1)]
        assert result.values[r] == pytest.approx(np.mean(per_seed), abs=1e-12)
    assert 0 < result.valid_replicates <= REPLICATES and result.ci_upper > result.ci_lower


def test_s4_domain_slice_by_masked_shared_weights_equals_the_slice_of_each_joint_resample(setup):
    """S4 clarification: a slice interval is the slice metric of each full-cohort joint resample."""
    frame, seeds, _, draws = setup
    mask = frame["domain"].eq("target").to_numpy()
    sliced = {s: f.loc[mask].reset_index(drop=True) for s, f in seeds.items()}
    shared = replace(draws, sample_weights=draws.sample_weights[:, mask], identity=draws.identity + ":target")
    result = family_bootstrap(sliced, seeds=(0, 1), metric="auroc", draws=shared)
    labels = frame["is_anomaly"].to_numpy()
    for r in range(40):
        per_seed = [_expanded_auroc(labels[mask], seeds[s]["score"].to_numpy()[mask], draws.sample_weights[r][mask])
                    for s in (0, 1)]
        expected = np.mean(per_seed)
        if np.isnan(expected):
            assert np.isnan(result.values[r])
        else:
            assert result.values[r] == pytest.approx(expected, abs=1e-12)
