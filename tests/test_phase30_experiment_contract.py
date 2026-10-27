"""Phase 3.0 experiment identity, persistence, ledger, and evaluation gate."""

from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from src.experiment.config import ExperimentConfig
from src.experiment.gate import EvaluationGate
from src.experiment.ledger import TestLookLedger, TestLookRecord
from src.experiment.provenance import ordered_id_hash
from src.experiment.scores import ScoreBatch, ScoreProvenance, select_protocol_b_threshold
from src.experiment.store import ResultStore


def _config(**changes) -> ExperimentConfig:
    values = dict(
        timestamp_utc="2026-08-24T00:00:00+00:00",
        machine="RoboticArm",
        protocol="mechasight_experimental_protocol",
        regime="S",
        model_family="statistical_reference",
        model_hyperparameters={"distance": "mahalanobis"},
        model_seed=0,
        preprocessing_config_id="phase2.2-v3-test",
        preprocessing_config_hash="config-hash",
        split_version="protocol_b_few_shot_adaptation_v3",
        split_manifest_hash="manifest-hash",
        dataset_release_identity="IMAD-DS:zenodo-12665499:verified",
        feature_representation="engineered_features",
        feature_version="phase2-feature-v1",
        window_length_s=0.1,
        stride_s=0.1,
        normalization_strategy="none",
        code_git_sha="abc123",
        dependency_environment={"python": "3.14.4", "scikit-learn": "1.9.0"},
    )
    values.update(changes)
    return ExperimentConfig(**values)


def _provenance(config: ExperimentConfig) -> ScoreProvenance:
    return ScoreProvenance(
        experiment_id=config.experiment_id,
        machine=config.machine,
        protocol=config.protocol,
        regime=config.regime,
        model_id="statistical-reference-v1",
        model_family=config.model_family,
        model_seed=config.model_seed,
        preprocessing_config_id=config.preprocessing_config_id,
        preprocessing_config_hash=config.preprocessing_config_hash,
        split_version=config.split_version,
        score_direction=config.anomaly_score_direction,
        data_provenance={
            "split_manifest_hash": config.split_manifest_hash,
            "source_validation_cohort_hash": ordered_id_hash(
                tuple(f"s{i}" for i in range(10))
            ),
            "common_evaluation_cohort_hash": ordered_id_hash(
                tuple(f"s{i}" for i in range(4))
            ),
        },
        config_provenance={"git_sha": config.code_git_sha},
    )


def _scores(config: ExperimentConfig, role: str, labels: tuple[str, ...]) -> ScoreBatch:
    n = len(labels)
    return ScoreBatch(
        provenance=_provenance(config),
        scores=np.linspace(0.1, 0.9, n),
        segment_ids=tuple(f"s{i}" for i in range(n)),
        parent_recording_ids=tuple(f"p{i}" for i in range(n)),
        cohort_roles=(role,) * n,
        domains=("source",) * n,
        anomaly_labels=labels,
    )


def test_experiment_id_is_deterministic_and_ignores_timestamp():
    first = _config()
    second = _config(timestamp_utc="2026-08-25T00:00:00+00:00")
    assert first.experiment_id == second.experiment_id


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_seed", 1),
        ("window_length_s", 0.2),
        ("stride_s", 0.05),
        ("normalization_strategy", "train_zscore"),
        ("split_manifest_hash", "changed"),
        ("model_hyperparameters", {"distance": "euclidean"}),
    ],
)
def test_scientifically_relevant_change_changes_experiment_id(field, value):
    base = _config()
    assert replace(base, **{field: value}).experiment_id != base.experiment_id


def test_model_seed_policy_is_exactly_zero_through_four():
    for seed in range(5):
        assert _config(model_seed=seed).model_seed == seed
    with pytest.raises(ValueError, match="model seed"):
        _config(model_seed=5)


def test_machine_limitations_are_embedded_in_config_metadata():
    assert "six independent" in _config().scientific_limitations[0]
    assert "27 target-normal segments from 18" in _config().scientific_limitations[1]
    bm = _config(machine="BrushlessMotor")
    assert "acquisition-time/class confound" in bm.scientific_limitations[0]


def test_result_store_writes_complete_bundle_and_refuses_overwrite(tmp_path):
    config = _config()
    store = ResultStore(tmp_path / "results")
    scores = pd.DataFrame({
        "segment_id": ["s1", "s2"],
        "parent_recording": ["p1", "p2"],
        "score": [0.1, 0.9],
    })
    cohort_hash = ordered_id_hash(("s1", "s2"))
    path = store.write_bundle(
        config=config,
        provenance={
            "test_cohort_hash": cohort_hash,
            "score_direction": "higher_is_more_anomalous",
            "threshold_provenance": {"source": "source_validation"},
        },
        metrics={"auroc": 1.0},
        scores=scores,
    )
    assert {item.name for item in path.iterdir()} == {
        "config.json", "provenance.json", "metrics.json", "scores.parquet",
        "bundle_manifest.json",
    }
    persisted = json.loads((path / "provenance.json").read_text(encoding="utf-8"))
    assert persisted["scientific_limitation_ids"] == list(config.scientific_limitation_ids)
    assert persisted["cohort_provenance"]["source_validation"]["parent_count"] == 6
    assert persisted["cohort_provenance"]["target_adaptation"]["parent_count"] == 18
    with pytest.raises(FileExistsError):
        store.write_bundle(
            config=config,
            provenance={
                "test_cohort_hash": cohort_hash,
                "score_direction": "higher_is_more_anomalous",
                "threshold_provenance": {"source": "source_validation"},
            },
            metrics={"auroc": 0.0},
            scores=scores,
        )


def _record(index: int) -> TestLookRecord:
    return TestLookRecord(
        timestamp_utc=f"2026-08-24T00:00:0{index}+00:00",
        experiment_id=f"p3-{index}",
        git_sha="abc",
        machine="RoboticArm",
        protocol="mechasight_experimental_protocol",
        regime="S",
        model_config_identity="model-v1",
        test_cohort_hash="test-hash",
        reason="synthetic unit test",
        metrics_requested=("auroc",),
    )


def test_test_look_ledger_is_append_only(tmp_path):
    ledger = TestLookLedger(tmp_path / "test_looks.jsonl")
    ledger.append(_record(1))
    first_bytes = ledger.path.read_bytes()
    ledger.append(_record(2))
    assert ledger.path.read_bytes().startswith(first_bytes)
    assert [item["experiment_id"] for item in ledger.read()] == ["p3-1", "p3-2"]


def test_validation_gate_does_not_create_test_look(tmp_path):
    config = _config()
    provenance = _provenance(config)
    gate = EvaluationGate(config, provenance)
    batch = _scores(config, "source_validation", ("normal",) * 10)
    threshold = select_protocol_b_threshold(batch, expected_provenance=provenance)
    ledger = TestLookLedger(tmp_path / "test_looks.jsonl")
    result = gate.evaluate_validation(batch, threshold=threshold)
    assert result["evaluation_scope"] == "validation"
    assert ledger.read() == []


def test_test_gate_logs_before_returning_synthetic_metrics(tmp_path):
    config = _config()
    provenance = _provenance(config)
    gate = EvaluationGate(config, provenance)
    validation = _scores(config, "source_validation", ("normal",) * 10)
    threshold = select_protocol_b_threshold(validation, expected_provenance=provenance)
    test = _scores(config, "common_evaluation", ("normal", "normal", "fault", "fault"))
    ledger = TestLookLedger(tmp_path / "test_looks.jsonl")
    result = gate.evaluate_test(
        test, threshold=threshold, ledger=ledger, reason="synthetic gate test"
    )
    assert result["evaluation_scope"] == "test"
    assert len(ledger.read()) == 1
    assert ledger.read()[0]["test_cohort_hash"] == test.cohort_hash


def test_test_gate_rejects_validation_batch_as_test(tmp_path):
    config = _config()
    provenance = _provenance(config)
    gate = EvaluationGate(config, provenance)
    validation = _scores(config, "source_validation", ("normal",) * 10)
    threshold = select_protocol_b_threshold(validation, expected_provenance=provenance)
    with pytest.raises(ValueError, match="common_evaluation"):
        gate.evaluate_test(
            validation,
            threshold=threshold,
            ledger=TestLookLedger(tmp_path / "ledger.jsonl"),
            reason="must fail",
        )
