"""Single-Modality Benchmarking: analyse-once metric layer on synthetic scores."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.experiment.phase4_analysis import (
    CONTRAST_METRICS,
    MODALITY_PAIRS,
    dominance_table,
    experiment_metrics,
    family_bootstrap,
    join_labels,
    paired_contrast,
    seed_summary,
)
from src.experiment.phase32 import make_common_bootstrap_draws


def _labels(n_source_normal=6, n_source_anomaly=4, n_target_normal=3, n_target_anomaly=3):
    rows = []
    index = 0
    for domain, label, count in (
        ("source", "normal", n_source_normal), ("source", "anomaly", n_source_anomaly),
        ("target", "normal", n_target_normal), ("target", "anomaly", n_target_anomaly),
    ):
        for _ in range(count):
            rows.append({
                "segment_id": f"seg{index:03d}", "parent_recording": f"p{index:03d}",
                "domain": domain, "anomaly_label": label,
                "operating_condition": "op", "background_condition": "bg",
            })
            index += 1
    return pd.DataFrame(rows)


def _scores(labels: pd.DataFrame, seed: int, strength: float = 3.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.normal(size=len(labels))
    base = base + strength * labels["anomaly_label"].eq("anomaly").to_numpy()
    return pd.DataFrame({
        "segment_id": labels["segment_id"], "parent_recording": labels["parent_recording"],
        "domain": labels["domain"], "score": base, "window_count": 5,
    })


def test_join_labels_attaches_binary_labels_and_refuses_missing_segments():
    labels = _labels()
    scores = _scores(labels, 0)
    joined = join_labels(scores, labels)
    assert list(joined["is_anomaly"]) == labels["anomaly_label"].eq("anomaly").astype(int).tolist()
    with pytest.raises(ValueError, match="segment"):
        join_labels(scores.iloc[:-1], labels)
    wrong_domain = scores.copy()
    wrong_domain.loc[0, "domain"] = "target"
    with pytest.raises(ValueError, match="domain"):
        join_labels(wrong_domain, labels)


def test_experiment_metrics_on_perfect_separation():
    labels = _labels()
    scores = _scores(labels, 0, strength=100.0)
    joined = join_labels(scores, labels)
    result = experiment_metrics(joined, threshold=50.0)
    assert result["complete"]["auroc"] == 1.0
    assert result["complete"]["average_precision"] == 1.0
    assert result["source"]["auroc"] == 1.0 and result["target"]["auroc"] == 1.0
    assert result["complete"]["threshold"]["false_positive"] == 0
    assert result["complete"]["threshold"]["true_positive"] == 7
    assert result["complete"]["threshold"]["phase321_count_identity_f1"] == 1.0
    assert result["healthy_target"]["false_positive_rate"] == 0.0
    assert result["healthy_target"]["healthy_target_sample_count"] == 3
    assert result["target"]["prevalence"] == pytest.approx(0.5)
    assert result["complete"]["ap_minus_prevalence"] == pytest.approx(1 - 7 / 16)


def test_experiment_metrics_no_alarm_convention():
    labels = _labels()
    scores = _scores(labels, 0, strength=1.0)
    joined = join_labels(scores, labels)
    result = experiment_metrics(joined, threshold=1e9)
    threshold = result["complete"]["threshold"]
    assert threshold["phase321_count_identity_f1"] == 0.0
    assert threshold["precision"] == "undefined - no positive predictions"
    assert threshold["produced_no_positive_predictions"] is True


def test_family_bootstrap_seed_averages_and_paired_contrast_sign():
    labels = _labels(20, 20, 10, 10)
    labels["machine"] = "BrushlessMotor"
    strong = {seed: join_labels(_scores(labels, seed, 4.0), labels) for seed in (0, 1, 2)}
    weak = {seed: join_labels(_scores(labels, seed, 0.2), labels) for seed in (0, 1, 2)}
    draws = make_common_bootstrap_draws(strong[0].assign(machine="BrushlessMotor"), replicates=300, seed=1995)
    a = family_bootstrap(strong, seeds=(0, 1, 2), metric="auroc", draws=draws)
    b = family_bootstrap(weak, seeds=(0, 1, 2), metric="auroc", draws=draws)
    assert a.point_estimate > b.point_estimate
    assert a.valid_replicates == 300
    contrast = paired_contrast(a, b, orientation="strong_minus_weak")
    assert contrast["ci_lower"] > 0 and contrast["point_delta"] > 0
    summary = seed_summary({seed: 0.9 + 0.01 * seed for seed in (0, 1, 2)})
    assert summary["mean"] == pytest.approx(0.91) and summary["sd"] > 0 and summary["seed_count"] == 3


def test_dominance_rule_requires_every_family_to_agree():
    rows = []
    for family in ("a", "b", "c"):
        rows.append({
            "machine": "RoboticArm", "regime": "S", "family": family, "pair": "acc_minus_mic",
            "metric": "auroc", "point_delta": 0.1, "ci_lower": 0.02, "ci_upper": 0.2,
        })
    rows.append({
        "machine": "RoboticArm", "regime": "S", "family": "a", "pair": "gyro_minus_mic",
        "metric": "auroc", "point_delta": 0.1, "ci_lower": 0.02, "ci_upper": 0.2,
    })
    rows.append({
        "machine": "RoboticArm", "regime": "S", "family": "b", "pair": "gyro_minus_mic",
        "metric": "auroc", "point_delta": 0.01, "ci_lower": -0.05, "ci_upper": 0.1,
    })
    rows.append({
        "machine": "RoboticArm", "regime": "S", "family": "c", "pair": "gyro_minus_mic",
        "metric": "auroc", "point_delta": 0.1, "ci_lower": 0.02, "ci_upper": 0.2,
    })
    table = dominance_table(pd.DataFrame(rows), families=("a", "b", "c"))
    by_pair = table.set_index("pair")
    assert by_pair.loc["acc_minus_mic", "verdict"] == "first_dominates"
    assert by_pair.loc["gyro_minus_mic", "verdict"] == "no_consistent_dominance"
    assert by_pair.loc["gyro_minus_mic", "families_excluding_zero"] == 2
    assert MODALITY_PAIRS == (("acc", "mic"), ("gyro", "mic"), ("acc", "gyro"))
    assert CONTRAST_METRICS == ("auroc", "average_precision", "standardized_pauroc_fpr_0_10")
