"""Phase 5.3 statistical procedures on synthetic cohorts only.

These verify the implementation of design v2 §7-8.  They are not evidence about
MechaSight performance, and no protected metric is computed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.experiment.phase32 import make_common_bootstrap_draws
from src.experiment.phase4_analysis import family_bootstrap
from src.fusion import statistics as st
from src.fusion.statistics import MetricCategory


def _cohort(seed: int = 0, *, target_parents: int = 6, cluster_sizes=(1, 2, 5)) -> pd.DataFrame:
    """RoboticArm-like: parents with unequal cluster sizes, source/target normals, anomalies."""

    rng = np.random.default_rng(seed)
    rows = []
    for p in range(40):
        domain = "target" if p < target_parents else "source"
        for s in range(cluster_sizes[p % len(cluster_sizes)]):
            rows.append({"segment_id": f"n{p}_{s}", "parent_recording": f"np{p}", "is_anomaly": 0, "domain": domain})
    for p in range(30):
        for s in range(cluster_sizes[p % len(cluster_sizes)]):
            rows.append({"segment_id": f"a{p}_{s}", "parent_recording": f"ap{p}", "is_anomaly": 1,
                         "domain": "target" if p % 3 == 0 else "source"})
    frame = pd.DataFrame(rows)
    frame["machine"] = "RoboticArm"
    frame["score"] = rng.normal(size=len(frame)) + 1.5 * frame["is_anomaly"]
    return frame


def _draws(frame, replicates=2000, seed=1995):
    return make_common_bootstrap_draws(frame.sort_values("segment_id").reset_index(drop=True),
                                       replicates=replicates, seed=seed)


def _seeds(frame, n=3, shift=0.0):
    rng = np.random.default_rng(42)
    return {s: frame.assign(score=frame["score"] + shift + 0.1 * rng.normal(size=len(frame))) for s in range(n)}


# -------------------------------------------------------- threshold metrics
def test_point_estimates_are_the_plain_threshold_rates():
    frame = _cohort()
    healthy = frame[(frame.is_anomaly == 0) & (frame.domain == "target")]
    assert st.threshold_metric_point(frame, 0.5, "healthy_target_fpr") == float((healthy.score > 0.5).mean())
    assert st.threshold_metric_point(frame, 0.5, "recall") == float((frame[frame.is_anomaly == 1].score > 0.5).mean())
    with pytest.raises(ValueError):
        st.threshold_metric_point(frame, 0.5, "auroc")


def test_identity_weights_reproduce_the_point_estimate_exactly():
    frame = _cohort().sort_values("segment_id").reset_index(drop=True)
    draws = _draws(frame, replicates=4)
    draws = type(draws)(np.ones_like(draws.sample_weights), draws.identity, 4, draws.seed)
    dist = st.threshold_metric_bootstrap({0: frame}, {0: 0.5}, seeds=(0,), metric="recall", draws=draws)
    assert np.all(dist.values == dist.point_estimate)


def test_seed_averaging_follows_the_frozen_r8_rule():
    frame = _cohort()
    draws = _draws(frame, replicates=500)
    seeds = _seeds(frame)
    thresholds = {s: 0.4 + 0.05 * s for s in seeds}
    joint = st.threshold_metric_bootstrap(seeds, thresholds, seeds=(0, 1, 2), metric="healthy_target_fpr", draws=draws)
    singles = [st.threshold_metric_bootstrap({s: seeds[s]}, {s: thresholds[s]}, seeds=(s,),
                                             metric="healthy_target_fpr", draws=draws).values for s in seeds]
    assert np.allclose(joint.values, np.mean(singles, axis=0), equal_nan=True)
    assert joint.point_estimate == pytest.approx(np.mean([st.threshold_metric_point(seeds[s], thresholds[s],
                                                                                    "healthy_target_fpr") for s in seeds]))


def test_bootstrap_is_deterministic_and_paired_on_shared_draws():
    frame = _cohort()
    a = st.threshold_metric_bootstrap(_seeds(frame), {s: 0.5 for s in range(3)}, seeds=(0, 1, 2),
                                      metric="recall", draws=_draws(frame))
    b = st.threshold_metric_bootstrap(_seeds(frame), {s: 0.5 for s in range(3)}, seeds=(0, 1, 2),
                                      metric="recall", draws=_draws(frame))
    assert np.array_equal(a.values, b.values, equal_nan=True) and a.draw_identity == b.draw_identity
    same = st.paired_metric_difference(a, b)
    assert same["point_delta"] == 0.0 and (same["ci_lower"], same["ci_upper"]) == (0.0, 0.0)
    other = st.threshold_metric_bootstrap(_seeds(frame), {s: 0.5 for s in range(3)}, seeds=(0, 1, 2),
                                          metric="recall", draws=_draws(frame, seed=7))
    with pytest.raises(ValueError, match="incompatible"):
        st.paired_metric_difference(a, other)


def test_a_paired_shift_is_detected_with_the_right_sign():
    frame = _cohort()
    draws = _draws(frame)
    base = st.threshold_metric_bootstrap(_seeds(frame), {s: 0.5 for s in range(3)}, seeds=(0, 1, 2),
                                         metric="recall", draws=draws)
    higher = st.threshold_metric_bootstrap(_seeds(frame, shift=1.0), {s: 0.5 for s in range(3)}, seeds=(0, 1, 2),
                                           metric="recall", draws=draws)
    diff = st.paired_metric_difference(higher, base)
    assert diff["point_delta"] > 0 and diff["ci_lower"] > 0


def test_replicates_without_any_healthy_target_unit_are_invalid_and_counted():
    frame = _cohort(target_parents=1, cluster_sizes=(1,))       # one healthy-target parent among 40 normals
    dist = st.threshold_metric_bootstrap({0: frame}, {0: 0.5}, seeds=(0,), metric="healthy_target_fpr",
                                         draws=_draws(frame, replicates=1000))
    assert 0 < dist.invalid_replicates < 1000
    assert dist.valid_replicates + dist.invalid_replicates == 1000
    assert np.isnan(dist.values).sum() == dist.invalid_replicates


def test_frames_must_share_identical_units_and_cover_the_seed_set():
    frame = _cohort()
    draws = _draws(frame, replicates=10)
    with pytest.raises(ValueError, match="identical evaluation units"):
        st.threshold_metric_bootstrap({0: frame, 1: frame.iloc[1:]}, {0: 0.5, 1: 0.5}, seeds=(0, 1),
                                      metric="recall", draws=draws)
    with pytest.raises(ValueError, match="frozen seed set"):
        st.threshold_metric_bootstrap({0: frame}, {0: 0.5}, seeds=(0, 1), metric="recall", draws=draws)


def test_ranking_metrics_reuse_the_frozen_family_bootstrap_on_the_same_draws():
    frame = _cohort()
    draws = _draws(frame, replicates=200)
    dist = family_bootstrap(_seeds(frame), seeds=(0, 1, 2), metric="auroc", draws=draws)
    assert dist.draw_identity == draws.identity and 0.5 < dist.point_estimate <= 1.0


# ------------------------------------------------------- interpretation rules
def _c(lower, upper, point, valid=10000, requested=10000):
    return {"ci_lower": lower, "ci_upper": upper, "point_delta": point,
            "valid_replicates": valid, "replicates_requested": requested}


@pytest.mark.parametrize("contrast, metric, expected", [
    (_c(-0.01, 0.015, 0.002), "auroc", MetricCategory("EQUIVALENT", None)),
    (_c(0.002, 0.015, 0.008), "auroc", MetricCategory("EQUIVALENT", "DETECTABLE_BUT_NEGLIGIBLE")),
    (_c(0.01, 0.08, 0.04), "auroc", MetricCategory("IMPROVEMENT", "MEANINGFUL")),
    (_c(0.003, 0.03, 0.012), "auroc", MetricCategory("IMPROVEMENT", "BELOW_MARGIN")),
    (_c(-0.09, -0.03, -0.06), "auroc", MetricCategory("DEGRADATION", "MEANINGFUL")),
    (_c(-0.30, 0.30, 0.001), "auroc", MetricCategory("INDETERMINATE", None)),       # small point, wide CI
    (_c(-0.01, 0.01, 0.0, valid=8999), "auroc", MetricCategory("INDETERMINATE", None)),
    (_c(None, None, 0.0, valid=0), "auroc", MetricCategory("INDETERMINATE", None)),
    (_c(-0.25, -0.12, -0.18), "healthy_target_fpr", MetricCategory("IMPROVEMENT", "MEANINGFUL")),
    (_c(0.12, 0.25, 0.18), "healthy_target_fpr", MetricCategory("DEGRADATION", "MEANINGFUL")),
    (_c(-0.05, 0.08, 0.01), "healthy_target_fpr", MetricCategory("EQUIVALENT", None)),
    (_c(-0.20, 0.20, 0.0), "healthy_target_fpr", MetricCategory("INDETERMINATE", None)),
    (_c(-0.12, -0.06, -0.09), "recall", MetricCategory("DEGRADATION", "MEANINGFUL")),
])
def test_metric_categories_follow_design_v2(contrast, metric, expected):
    assert st.classify_metric(contrast, metric=metric) == expected


@pytest.mark.parametrize("h, r, axis", [
    ("IMPROVEMENT", "EQUIVALENT", "O+"), ("IMPROVEMENT", "IMPROVEMENT", "O+"),
    ("DEGRADATION", "EQUIVALENT", "O-"), ("IMPROVEMENT", "DEGRADATION", "O_tradeoff"),
    ("EQUIVALENT", "EQUIVALENT", "O0"), ("IMPROVEMENT", "INDETERMINATE", "O?"),
    ("INDETERMINATE", "EQUIVALENT", "O?"),
])
def test_operating_axis(h, r, axis):
    assert st.operating_axis(MetricCategory(h, None), MetricCategory(r, None)) == axis


@pytest.mark.parametrize("ranking, operating, joint", [
    ("IMPROVEMENT", "O+", "BOTH_IMPROVED"), ("IMPROVEMENT", "O0", "RANKING_GAIN_OPERATION_PRESERVED"),
    ("EQUIVALENT", "O+", "OPERATING_GAIN_RANKING_PRESERVED"), ("IMPROVEMENT", "O-", "TRADE_OFF"),
    ("IMPROVEMENT", "O_tradeoff", "TRADE_OFF"), ("DEGRADATION", "O+", "TRADE_OFF"),
    ("EQUIVALENT", "O_tradeoff", "TRADE_OFF"), ("EQUIVALENT", "O0", "EQUIVALENT"),
    ("DEGRADATION", "O-", "UNDERPERFORMANCE"), ("DEGRADATION", "O0", "UNDERPERFORMANCE"),
    ("DEGRADATION", "O_tradeoff", "UNDERPERFORMANCE"), ("EQUIVALENT", "O-", "UNDERPERFORMANCE"),
    ("INDETERMINATE", "O+", "INDETERMINATE"), ("IMPROVEMENT", "O?", "INDETERMINATE"),
])
def test_joint_categories_follow_design_v2(ranking, operating, joint):
    assert st.joint_category(MetricCategory(ranking, None), operating) == joint
