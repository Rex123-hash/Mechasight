"""Phase 5.3 L1 feature-level fusion: frozen-recipe inheritance and development-only fitting.

The 20 dense-AE fits take minutes each, so the suite checks their recipe,
standardiser semantics, initialisation and checkpoint rule on synthetic data
and one real BrushlessMotor standardiser equivalence; the full fits are
verified by ``scripts/run_phase5_development.py l1-verify``.  The fast
Mahalanobis fit runs end to end into a temporary directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.experiment.phase4 import _model_hyperparameters, standardize_float32
from src.experiment.provenance import initial_state_hash
from src.fusion import l1
from src.fusion.assembly import load_design_registry
from src.models.dense_autoencoder import DENSE_AE_SPEC, DenseAutoencoder, DenseAutoencoderSpec
from src.preprocessing.normalization import FittedChannelStandardizer

PRETEST = Path("reports/phase4_pretest/experiments")
PHASE4_MODELS = Path("data/interim/phase4/models")
HAVE_DATA = Path("data/interim/phase31_primary/BrushlessMotor/S/training.parquet").exists() and PHASE4_MODELS.is_dir()
SPLIT = Path("reports/phase2_2/protocol_b_few_shot_adaptation_v3.csv")


def test_registry_holds_exactly_the_24_planned_new_fits():
    units = l1.l1_units(load_design_registry())
    assert len(units) == 24
    assert sum(u["family"] == "mahalanobis_shrinkage" for u in units) == 4
    assert sum(u["family"] == "dense_autoencoder" for u in units) == 20
    assert {(u["machine"], u["regime"]) for u in units} == {(m, r) for m in ("RoboticArm", "BrushlessMotor") for r in "ST"}


@pytest.mark.parametrize("family, input_dim", [("dense_autoencoder", 2918), ("mahalanobis_shrinkage", None)])
def test_l1_hyperparameters_are_the_frozen_phase4_recipe(family, input_dim):
    fused = l1._hyperparameters(family, input_dim=input_dim, code_source_hash="h")
    frozen = _model_hyperparameters(family, "acc", input_dim=input_dim, code_source_hash="x")
    changed = {"modality", "stage", "stage_schema_version", "phase4_source_hash", "phase5_source_hash",
               "fusion_level", "fusion_modalities", "design_config_sha256"}
    assert {k: v for k, v in fused.items() if k not in changed} == {k: v for k, v in frozen.items() if k not in changed}
    assert fused["modality"] == "fusion[mic+acc+gyro]" and fused["fusion_modalities"] == ["mic", "acc", "gyro"]


def test_block_standardiser_equals_a_whole_matrix_fit():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 30)) * rng.uniform(0.1, 9, size=30) + rng.normal(size=30)
    whole = FittedChannelStandardizer.fit([X], split="train", segment_ids=("a",))
    blocks = [FittedChannelStandardizer.fit([X[:, s]], split="train", segment_ids=("a",))
              for s in (slice(0, 10), slice(10, 21), slice(21, 30))]
    assert np.array_equal(whole.mean, np.concatenate([b.mean for b in blocks]))
    assert np.array_equal(whole.std, np.concatenate([b.std for b in blocks]))


def test_in_place_standardisation_is_bitwise_the_phase4_arithmetic():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(20000, 12)).astype(np.float32)
    std = FittedChannelStandardizer.fit([X.astype(np.float64)], split="train", segment_ids=("a",))
    expected = standardize_float32(std, X)
    result = l1._standardize_in_place(std, X.copy())
    assert result.dtype == np.float32 and np.array_equal(result, expected)


def test_autoencoder_initialisation_is_seed_deterministic_at_fused_width():
    a, b = DenseAutoencoder(input_dim=2936, seed=3), DenseAutoencoder(input_dim=2936, seed=3)
    assert initial_state_hash(a.network) == initial_state_hash(b.network)
    assert initial_state_hash(a.network) != initial_state_hash(DenseAutoencoder(input_dim=2936, seed=4).network)
    assert a.spec == DENSE_AE_SPEC and a.parameter_count > DenseAutoencoder(input_dim=1068, seed=3).parameter_count


def test_frozen_checkpoint_rule_restores_the_best_validation_epoch():
    rng = np.random.default_rng(0)
    spec = DenseAutoencoderSpec(**{**DENSE_AE_SPEC.to_dict(), "max_epochs": 30, "patience": 3})
    model = DenseAutoencoder(input_dim=8, seed=0, spec=spec)
    train = rng.normal(size=(256, 8)).astype(np.float32)
    validation = rng.normal(size=(64, 8)).astype(np.float32)
    model.fit(train, validation)
    losses = model.history["validation_window_mse"]
    assert model.history["best_epoch"] == int(np.argmin(losses)) + 1
    assert model.history["best_validation_window_mse"] == min(losses)
    assert float(np.mean(model.score_windows(validation))) == pytest.approx(min(losses), rel=1e-6)


def test_a_reused_or_unknown_unit_is_refused_before_any_io(monkeypatch):
    monkeypatch.setattr(l1, "load_config", lambda *a, **k: (_ for _ in ()).throw(AssertionError("I/O")))
    reused = next(u for u in load_design_registry() if u["l1_source"] == "reuse_phase3_2_sealed")
    with pytest.raises(l1.L1FitError, match="not a planned new L1 fit"):
        l1.run_l1_fit(unit=reused, feature_root="x", spectral_cache_root="x", model_root="x", result_root="x",
                      split_manifest_path="x", code_git_sha="x", code_source_hash="x")


@pytest.mark.skipif(not HAVE_DATA, reason="development artifacts not present")
def test_real_mahalanobis_l1_fit_is_validation_only_reproducible_and_matches_phase4(tmp_path):
    unit = next(u for u in l1.l1_units(load_design_registry())
                if u["family"] == "mahalanobis_shrinkage" and u["machine"] == "RoboticArm" and u["regime"] == "T")
    kwargs = dict(unit=unit, feature_root="data/interim/phase31_primary", spectral_cache_root="data/interim/phase4/spectral",
                  split_manifest_path=SPLIT, code_git_sha="test-sha", code_source_hash="test-src",
                  phase4_pretest_root=PRETEST, phase4_model_root=PHASE4_MODELS)
    first = l1.run_l1_fit(model_root=tmp_path / "m1", result_root=tmp_path / "r1", **kwargs)
    second = l1.run_l1_fit(model_root=tmp_path / "m2", result_root=tmp_path / "r2", **kwargs)
    assert first["experiment_id"] == second["experiment_id"] and first["feature_count"] == 98
    a = l1.load_l1_bundle(first["bundle"])
    b = l1.load_l1_bundle(second["bundle"])
    assert a["validation_scores"]["score"].equals(b["validation_scores"]["score"])            # deterministic
    prov = a["provenance"]
    assert prov["test_evaluated"] is False and prov["completion_status"] == "COMPLETE"
    assert prov["score_provenance"]["data_provenance"]["standardizer_equivalence"]["equal_to_phase4_standardizers"]
    phase4 = json.loads((PRETEST / unit["phase4_single_modality_ids"]["acc"] / "provenance.json").read_text(encoding="utf-8"))
    assert prov["training_cohort_hash"] == phase4["training_cohort_hash"]
    assert prov["validation_cohort_hash"] == phase4["validation_cohort_hash"]
    scores = a["validation_scores"]["score"].to_numpy()
    assert a["threshold"]["value"] == float(np.quantile(scores, 0.9, method="higher"))
    assert set(a["validation_scores"]["cohort_role"]) == {"source_validation"}
    check = l1.reproduce_l1_bundle(first["bundle"], model_root=tmp_path / "m1",
                                   feature_root="data/interim/phase31_primary",
                                   spectral_cache_root="data/interim/phase4/spectral", split_manifest_path=SPLIT)
    assert check["reproduced"] and check["max_abs_score_difference"] == 0.0
    with pytest.raises(FileExistsError):                                                        # immutable
        l1.run_l1_fit(model_root=tmp_path / "m1", result_root=tmp_path / "r1", **kwargs)


@pytest.mark.skipif(not HAVE_DATA, reason="development artifacts not present")
def test_real_fused_spectral_standardiser_equals_the_phase4_standardisers():
    """BrushlessMotor S: the per-block fit is bit-identical to the stored Phase-4 AE standardisers."""

    from src.experiment.data import TrainingCohortContract
    from src.preprocessing.config import load_config

    unit = next(u for u in l1.l1_units(load_design_registry())
                if u["family"] == "dense_autoencoder" and u["machine"] == "BrushlessMotor" and u["regime"] == "S")
    config = load_config("configs/preprocessing/mechasight_few_shot_regime_s.json")
    contract = TrainingCohortContract.from_frozen_assignments(pd.read_csv(SPLIT), config, machine="BrushlessMotor")
    standardizer, _meta, rows, _ids = l1._fit_block_standardizers(
        "data/interim/phase4/spectral", machine="BrushlessMotor", regime="S", contract=contract, config=config)
    assert standardizer.mean.size == 2936 and len(rows) == standardizer.n_samples
    assert l1.verify_standardizer_equivalence(unit, standardizer, phase4_pretest_root=PRETEST,
                                              phase4_model_root=PHASE4_MODELS)["equal_to_phase4_standardizers"]


def test_a_tampered_l1_bundle_is_refused(tmp_path):
    bundle = tmp_path / "p3-x"
    bundle.mkdir()
    (bundle / "provenance.json").write_text(json.dumps({"stage": l1.STAGE_NAME, "test_evaluated": False,
                                                        "completion_status": "COMPLETE"}), encoding="utf-8")
    from src.experiment.provenance import canonical_sha256
    (bundle / "bundle_manifest.json").write_text(json.dumps({"experiment_id": "p3-x", "files": {
        "provenance.json": canonical_sha256(bundle / "provenance.json")}}), encoding="utf-8")
    (bundle / "provenance.json").write_text(json.dumps({"stage": "other"}), encoding="utf-8")
    with pytest.raises(l1.L1FitError, match="hash mismatch"):
        l1.load_l1_bundle(bundle)
