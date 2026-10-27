"""Phase 3.0 model-facing data and normalization safety contracts."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from src.experiment.data import (
    TrainingCohortContract,
    fit_training_standardizer,
    load_estimator_training_data,
)
from src.preprocessing.config import load_config


def _contract(regime: str = "T") -> TrainingCohortContract:
    return TrainingCohortContract(
        machine="RoboticArm",
        protocol="mechasight_experimental_protocol",
        regime=regime,
        split_version="protocol_b_few_shot_adaptation_v3",
        preprocessing_version="phase2.2-v3",
        preprocessing_config_id="phase2.2-v3-test",
        source_segment_ids=frozenset({"s1", "s2"}),
        adaptation_segment_ids=frozenset({"a1"}) if regime == "T" else frozenset(),
        frozen_manifest_hash="frozen-hash",
    )


def _frame(regime: str = "T") -> pd.DataFrame:
    ids = ["s1", "s2", "a1"] if regime == "T" else ["s1", "s2"]
    roles = ["source_training", "source_training", "target_adaptation"][:len(ids)]
    domains = ["source", "source", "target"][:len(ids)]
    return pd.DataFrame({
        "machine": "RoboticArm",
        "protocol": "mechasight_experimental_protocol",
        "regime": regime,
        "split_version": "protocol_b_few_shot_adaptation_v3",
        "preprocessing_version": "phase2.2-v3",
        "preprocessing_config_id": "phase2.2-v3-test",
        "split": "train",
        "data_pool": roles,
        "segment_id": ids,
        "parent_recording": [f"p-{value}" for value in ids],
        "window_id": [f"w-{value}" for value in ids],
        "anomaly_label": "normal",
        "domain": domains,
        "f1": [1.0, 2.0, 3.0][:len(ids)],
        "f2": [3.0, 4.0, 5.0][:len(ids)],
    })


def test_estimator_interface_is_label_free_and_regime_t_includes_released_adaptation():
    data = load_estimator_training_data(
        _frame(), _contract(), feature_columns=("f1", "f2")
    )
    assert data.X.shape == (3, 2)
    assert set(data.data_pools) == {"source_training", "target_adaptation"}
    assert not hasattr(data, "anomaly_label")
    assert not hasattr(data, "y")


@pytest.mark.parametrize("forbidden", ["anomaly_label", "domain", "data_pool", "segment_id"])
def test_estimator_rejects_label_and_provenance_as_features(forbidden):
    with pytest.raises(ValueError, match="cannot be estimator features"):
        load_estimator_training_data(
            _frame(), _contract(), feature_columns=(forbidden,)
        )


def test_regime_s_rejects_target_adaptation():
    with pytest.raises(ValueError, match="wrong regime"):
        load_estimator_training_data(
            _frame("T"), _contract("S"), feature_columns=("f1",)
        )


def test_training_loader_rejects_test_row():
    bad = _frame()
    bad.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="train rows only"):
        load_estimator_training_data(bad, _contract(), feature_columns=("f1",))


def test_training_loader_rejects_anomalous_row():
    bad = _frame()
    bad.loc[0, "anomaly_label"] = "fault"
    with pytest.raises(ValueError, match="normal-only"):
        load_estimator_training_data(bad, _contract(), feature_columns=("f1",))


def test_training_loader_rejects_blocked_sibling():
    bad = _frame()
    bad.loc[0, ["split", "data_pool", "segment_id"]] = [
        "blocked_sibling", "adaptation_sibling_blocked", "blocked-1"
    ]
    with pytest.raises(ValueError, match="train rows only|blocked siblings"):
        load_estimator_training_data(bad, _contract(), feature_columns=("f1",))


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("split_version", "protocol_b_common_test_v2", "wrong split_version"),
        ("preprocessing_version", "phase2.1-v2", "wrong preprocessing_version"),
        ("regime", "S", "wrong regime"),
        ("protocol", "official_baseline_reproduction", "wrong protocol"),
        ("preprocessing_config_id", "stale-config", "wrong preprocessing_config_id"),
        ("machine", "BrushlessMotor", "wrong machine"),
    ],
)
def test_training_loader_rejects_wrong_provenance(column, value, message):
    bad = _frame()
    bad[column] = value
    with pytest.raises(ValueError, match=message):
        load_estimator_training_data(bad, _contract(), feature_columns=("f1",))


def test_training_loader_rejects_duplicate_sample_identity():
    bad = _frame()
    bad.loc[1, "window_id"] = bad.loc[0, "window_id"]
    with pytest.raises(ValueError, match="unique"):
        load_estimator_training_data(bad, _contract(), feature_columns=("f1",))


def test_training_loader_requires_complete_frozen_cohort():
    with pytest.raises(ValueError, match="complete frozen cohort"):
        load_estimator_training_data(
            _frame().iloc[:-1], _contract(), feature_columns=("f1",)
        )


@pytest.mark.parametrize(
    ("machine", "regime", "source_count", "adaptation_count", "total"),
    [
        ("RoboticArm", "S", 1528, 0, 1528),
        ("RoboticArm", "T", 1528, 27, 1555),
        ("BrushlessMotor", "S", 1137, 0, 1137),
        ("BrushlessMotor", "T", 1137, 18, 1155),
    ],
)
def test_real_phase22_manifest_binds_exact_regime_fit_cohorts(
    machine, regime, source_count, adaptation_count, total
):
    assignments = pd.read_csv("reports/phase2_2/protocol_b_few_shot_adaptation_v3.csv")
    config = load_config(
        f"configs/preprocessing/mechasight_few_shot_regime_{regime.lower()}.json"
    )
    contract = TrainingCohortContract.from_frozen_assignments(
        assignments, config, machine=machine
    )
    assert len(contract.source_segment_ids) == source_count
    assert len(contract.adaptation_segment_ids) == adaptation_count
    assert len(contract.authorized_segment_ids) == total


def test_safe_normalizer_fits_only_after_complete_training_validation():
    fitted = fit_training_standardizer(
        _frame(), _contract(), feature_columns=("f1", "f2")
    )
    assert fitted.standardizer.fit_split == "train"
    assert fitted.standardizer.fit_segment_ids == ("s1", "s2", "a1")
    assert fitted.standardizer.n_samples == 3


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("split", "validation"),
        ("split", "test"),
        ("anomaly_label", "fault"),
        ("data_pool", "adaptation_sibling_blocked"),
        ("protocol", "official_baseline_reproduction"),
        ("regime", "S"),
        ("split_version", "protocol_b_common_test_v2"),
        ("preprocessing_version", "phase2.1-v2"),
        ("preprocessing_config_id", "old-config"),
    ],
)
def test_safe_normalizer_rejects_unauthorized_manifest_membership(column, value):
    bad = _frame()
    bad.loc[0, column] = value
    with pytest.raises(ValueError):
        fit_training_standardizer(bad, _contract(), feature_columns=("f1", "f2"))


def test_contract_identity_change_is_detected():
    wrong = replace(_contract(), preprocessing_config_id="different")
    with pytest.raises(ValueError, match="preprocessing_config_id"):
        load_estimator_training_data(_frame(), wrong, feature_columns=("f1",))
