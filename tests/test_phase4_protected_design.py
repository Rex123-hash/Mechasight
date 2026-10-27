"""Single-Modality Benchmarking: score-all / analyse-once protected design.

Everything here runs on synthetic bundles, a tmp ledger and tmp staging.  The
suite-wide guard in conftest refuses the real ledger and result roots.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import pytest

from src.experiment.config import ExperimentConfig
from src.experiment.constants import SCORE_DIRECTION
from src.experiment.ledger import TestLookLedger
from src.experiment.phase4 import (
    EXPECTED_EXPERIMENT_COUNT,
    LEDGER_EXPECTED_START,
    SCORE_COLUMNS,
    BatchAuthorization,
    FrozenBundle,
    ScoreStaging,
    assert_complete_plan,
    authorize_protected_batch,
    experiment_plan,
    plan_protected_batch,
    score_one_experiment,
)
from src.experiment.scores import ScoreProvenance, ThresholdSelection
from src.preprocessing.protected_access import ProtectedReadRefused, authorized_protected_read


COHORTS = {"RoboticArm": "a" * 64, "BrushlessMotor": "b" * 64}


def _bundle(item, index: int) -> FrozenBundle:
    config = ExperimentConfig(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        machine=item.machine,
        protocol="mechasight_experimental_protocol",
        regime=item.regime,
        model_family=item.family,
        model_hyperparameters={"modality": item.modality, "index": index},
        model_seed=item.seed,
        preprocessing_config_id="cfg",
        preprocessing_config_hash="c" * 64,
        split_version="protocol_b_few_shot_adaptation_v3",
        split_manifest_hash="d" * 64,
        dataset_release_identity="IMAD-DS:test",
        feature_representation=f"single_modality[{item.modality}]",
        feature_version="v",
        window_length_s=0.1,
        stride_s=0.1,
        normalization_strategy="z",
        code_git_sha="e" * 40,
        dependency_environment={"python": "3"},
    )
    provenance = ScoreProvenance(
        experiment_id=config.experiment_id,
        machine=item.machine,
        protocol=config.protocol,
        regime=item.regime,
        model_id=f"model-{index}",
        model_family=item.family,
        model_seed=item.seed,
        preprocessing_config_id="cfg",
        preprocessing_config_hash="c" * 64,
        split_version=config.split_version,
        score_direction=SCORE_DIRECTION,
        data_provenance={"training_cohort_hash": "t" * 64, "source_validation_cohort_hash": "v" * 64},
        config_provenance={"modality": item.modality},
    )
    threshold = ThresholdSelection(
        value=0.5, policy_version="p", quantile=0.9, quantile_method="higher",
        score_direction=SCORE_DIRECTION, source_experiment_id=config.experiment_id,
        source_cohort_hash="v" * 64, source_segment_ids=("s1",), source_role="source_validation",
    )
    return FrozenBundle(
        experiment_id=config.experiment_id, config=config, modality=item.modality,
        threshold=threshold, score_provenance=provenance,
        model_manifest_sha256="f" * 64, bundle_path=None,
    )


@pytest.fixture(scope="module")
def bundles():
    return {b.experiment_id: b for b in (_bundle(item, i) for i, item in enumerate(experiment_plan()))}


def test_synthetic_plan_covers_156_unique_identities(bundles):
    assert len(bundles) == EXPECTED_EXPERIMENT_COUNT
    assert_complete_plan(bundles)


def test_incomplete_plan_is_refused(bundles):
    partial = dict(list(bundles.items())[:-1])
    with pytest.raises(ValueError, match="156-experiment plan"):
        assert_complete_plan(partial)


def test_batch_identity_is_deterministic_and_content_bound(bundles):
    first = plan_protected_batch(bundles, test_cohort_hashes_by_machine=COHORTS, code_git_sha="e" * 40)
    second = plan_protected_batch(bundles, test_cohort_hashes_by_machine=COHORTS, code_git_sha="e" * 40)
    assert first.batch_id == second.batch_id and first.batch_id.startswith("p4-test-batch-")
    other = plan_protected_batch(bundles, test_cohort_hashes_by_machine=COHORTS, code_git_sha="0" * 40)
    assert other.batch_id != first.batch_id
    assert first.to_dict()["ledger_expected_start"] == LEDGER_EXPECTED_START == 40
    assert first.to_dict()["experiment_count"] == 156


def test_authorization_appends_156_records_atomically_to_a_fresh_ledger(tmp_path, bundles):
    ledger = TestLookLedger(tmp_path / "ledger.jsonl")
    for _ in range(LEDGER_EXPECTED_START):
        pass
    planned = plan_protected_batch(bundles, test_cohort_hashes_by_machine=COHORTS, code_git_sha="e" * 40)
    with pytest.raises(RuntimeError, match="starting count"):
        authorize_protected_batch(ledger=ledger, planned=planned, bundles=bundles)
    authorization = authorize_protected_batch(
        ledger=ledger, planned=planned, bundles=bundles, expected_ledger_start=0
    )
    records = ledger.read()
    assert len(records) == 156
    assert authorization.ledger_end_count - authorization.ledger_start_count == 156
    assert {r["experiment_id"] for r in records} == set(bundles)
    assert all("no metric during scoring" in r["reason"] for r in records)
    assert all(r["test_cohort_hash"] == COHORTS[r["machine"]] for r in records)


def test_staging_refuses_labels_duplicates_and_incomplete_analysis(tmp_path, bundles):
    authorization = BatchAuthorization(
        batch_id="p4-test-batch-x", experiment_ids=frozenset(bundles),
        experiment_test_cohort_hashes={k: COHORTS[b.config.machine] for k, b in bundles.items()},
        ledger_start_count=40, ledger_end_count=196,
    )
    staging = ScoreStaging(tmp_path / "staging")
    staging.open(authorization)
    with pytest.raises(FileExistsError):
        staging.open(authorization)
    experiment_id = next(iter(bundles))
    good = pd.DataFrame({
        "segment_id": ["s1", "s2"], "parent_recording": ["p", "p"], "domain": ["source", "target"],
        "score": [0.1, 0.9], "window_count": [3, 3],
    })
    with pytest.raises(ValueError, match="label-free"):
        staging.write_scores(experiment_id, good.assign(anomaly_label="normal"), cohort_hash="x")
    staging.write_scores(experiment_id, good, cohort_hash="x")
    with pytest.raises(FileExistsError):
        staging.write_scores(experiment_id, good, cohort_hash="x")
    with pytest.raises(ValueError, match="incomplete: 1 of 156"):
        staging.verify_complete(bundles)
    manifest = staging.read_manifest()
    assert manifest["metrics_computed"] is False
    # Staged scores are sealed protected outputs: readable only inside an authorized scope.
    with pytest.raises(ProtectedReadRefused):
        staging.read_scores(experiment_id)
    with authorized_protected_read(authority="p4-test-batch-x", purpose="unit test fixture"):
        assert set(staging.read_scores(experiment_id).columns) == set(SCORE_COLUMNS)
    assert "anomaly_label" not in json.dumps(manifest)


def test_scoring_refuses_unauthorized_experiment(tmp_path, bundles):
    experiment_id, bundle = next(iter(bundles.items()))
    authorization = BatchAuthorization(
        batch_id="p4-test-batch-x", experiment_ids=frozenset(set(bundles) - {experiment_id}),
        experiment_test_cohort_hashes={}, ledger_start_count=40, ledger_end_count=196,
    )
    with pytest.raises(ValueError, match="authorization"):
        score_one_experiment(
            bundle, model_store=None, test_features=None, test_windows=None,
            authorization=authorization,
        )


def test_scoring_refuses_incomplete_receipt(tmp_path, bundles):
    experiment_id, bundle = next(iter(bundles.items()))
    authorization = BatchAuthorization(
        batch_id="p4-test-batch-x", experiment_ids=frozenset(bundles),
        experiment_test_cohort_hashes={}, ledger_start_count=40, ledger_end_count=100,
    )
    with pytest.raises(ValueError, match="incomplete"):
        score_one_experiment(
            bundle, model_store=None, test_features=None, test_windows=None,
            authorization=authorization,
        )


def _synthetic_labels(machine: str, n: int) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for index in range(n):
        anomaly = index % 2 == 1
        # RoboticArm parents hold several class-homogeneous segments; BM is 1:1.
        parent = f"{machine}-p{index // 4}-{int(anomaly)}" if machine == "RoboticArm" else f"{machine}-p{index}"
        rows.append({
            "segment_id": f"{machine}-s{index:03d}", "parent_recording": parent,
            "domain": "source" if index % 3 else "target",
            "anomaly_label": "anomaly" if anomaly else "normal",
            "operating_condition": "op", "background_condition": "bg",
        })
    return pd.DataFrame(rows)


def test_analyse_once_runs_over_a_complete_synthetic_batch_and_is_final(tmp_path, bundles, monkeypatch):
    from src.experiment import phase4 as p4
    from src.experiment.phase4_analysis import analyse_once

    counts = {"RoboticArm": 24, "BrushlessMotor": 20}
    monkeypatch.setattr(p4, "EXPECTED_TEST_SEGMENTS", counts)
    labels = {machine: _synthetic_labels(machine, n) for machine, n in counts.items()}
    cohorts = {m: p4.ordered_id_hash(tuple(f["segment_id"])) for m, f in labels.items()}
    authorization = BatchAuthorization(
        batch_id="p4-test-batch-synthetic", experiment_ids=frozenset(bundles),
        experiment_test_cohort_hashes={k: cohorts[b.config.machine] for k, b in bundles.items()},
        ledger_start_count=40, ledger_end_count=196,
    )
    staging = ScoreStaging(tmp_path / "staging")
    staging.open(authorization)
    rng = np.random.default_rng(0)
    for experiment_id, bundle in bundles.items():
        frame = labels[bundle.config.machine]
        strength = {"mic": 0.5, "acc": 3.0, "gyro": 2.0}[bundle.modality]
        scores = rng.normal(size=len(frame)) + strength * frame["anomaly_label"].eq("anomaly").to_numpy()
        table = pd.DataFrame({
            "segment_id": frame["segment_id"], "parent_recording": frame["parent_recording"],
            "domain": frame["domain"], "score": scores, "window_count": 3,
        }).loc[:, list(SCORE_COLUMNS)]
        staging.write_scores(experiment_id, table, cohort_hash=cohorts[bundle.config.machine])
    with authorized_protected_read(authority="p4-test-batch-fixture", purpose="unit test fixture"):
        result = analyse_once(
            staging=staging, bundles=bundles, labels_by_machine=labels, out_dir=tmp_path / "analysis",
            families=p4.FAMILIES, seeds_for=p4.expected_seeds, modalities=p4.MODALITIES, replicates=40,
        )
    assert result["experiment_count"] == 156
    family = pd.read_csv(tmp_path / "analysis" / "family_bootstrap_metrics.csv")
    assert len(family) == 2 * 2 * 3 * 5
    contrasts = pd.read_csv(tmp_path / "analysis" / "modality_paired_contrasts.csv")
    assert len(contrasts) == 2 * 2 * 5 * 3 * 3
    dominance = pd.read_csv(tmp_path / "analysis" / "modality_dominance.csv")
    assert set(dominance["verdict"]) <= {"first_dominates", "second_dominates", "no_consistent_dominance"}
    regime = pd.read_csv(tmp_path / "analysis" / "regime_paired_contrasts.csv")
    assert len(regime) == 2 * 3 * 5 * 3
    individual = pd.read_csv(tmp_path / "analysis" / "individual_experiment_metrics.csv")
    assert len(individual) == 156 and individual["auroc"].between(0, 1).all()
    assert staging.read_manifest()["metrics_computed"] is True
    with pytest.raises(RuntimeError, match="final"):
        analyse_once(
            staging=staging, bundles=bundles, labels_by_machine=labels, out_dir=tmp_path / "again",
            families=p4.FAMILIES, seeds_for=p4.expected_seeds, modalities=p4.MODALITIES, replicates=40,
        )
