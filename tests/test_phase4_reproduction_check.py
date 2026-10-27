"""Reload-and-rescore check on real pre-test bundles (validation data only)."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pytest

from src.experiment.phase4 import (
    ModelStore,
    load_frozen_bundles,
    verify_bundle_reproduces_validation,
)
from src.experiment.provenance import canonical_sha256

PRETEST = Path("reports/phase4_pretest/experiments")
MODELS = Path("data/interim/phase4/models")
SPLIT = Path("reports/phase2_2/protocol_b_few_shot_adaptation_v3.csv")
FEATURES = Path("data/interim/phase31_primary")
SPECTRAL = Path("data/interim/phase4/spectral")

pytestmark = pytest.mark.skipif(
    not (PRETEST.exists() and MODELS.exists() and FEATURES.exists()),
    reason="real pre-test bundles not present",
)


def _one_classical_bundle():
    bundles = load_frozen_bundles(PRETEST)
    for bundle in bundles.values():
        if bundle.config.model_family == "mahalanobis_shrinkage":
            return bundle
    pytest.skip("no mahalanobis bundle present")


def _config(bundle) -> Path:
    return Path(f"configs/preprocessing/mechasight_few_shot_regime_{bundle.config.regime.lower()}.json")


def test_real_bundle_reproduces_exactly():
    bundle = _one_classical_bundle()
    result = verify_bundle_reproduces_validation(
        bundle, model_store=ModelStore(MODELS), feature_root=FEATURES,
        spectral_cache_root=SPECTRAL, split_manifest_path=SPLIT, preprocessing_config_path=_config(bundle),
    )
    assert result["max_abs_score_difference"] < 1e-9 and result["threshold_reproduced"]


def test_tampered_model_artifact_is_detected(tmp_path):
    bundle = _one_classical_bundle()
    store_root = tmp_path / "models"
    shutil.copytree(MODELS / bundle.experiment_id, store_root / bundle.experiment_id)
    # Tampering without updating the manifest: hash mismatch.
    standardizer_path = store_root / bundle.experiment_id / "standardizer.joblib"
    standardizer = joblib.load(standardizer_path)
    tampered = type(standardizer)(
        mean=standardizer.mean + 0.5, std=standardizer.std, n_samples=standardizer.n_samples,
        fit_split=standardizer.fit_split, fit_segment_ids=standardizer.fit_segment_ids,
    )
    joblib.dump(tampered, standardizer_path)
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_bundle_reproduces_validation(
            bundle, model_store=ModelStore(store_root), feature_root=FEATURES,
            spectral_cache_root=SPECTRAL, split_manifest_path=SPLIT, preprocessing_config_path=_config(bundle),
        )
    # Tampering *with* a consistent manifest but a stale bundle: score mismatch.
    manifest_path = store_root / bundle.experiment_id / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["standardizer.joblib"] = canonical_sha256(standardizer_path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    from dataclasses import replace
    stale = replace(bundle, model_manifest_sha256=canonical_sha256(manifest_path))
    with pytest.raises(ValueError, match="does not reproduce"):
        verify_bundle_reproduces_validation(
            stale, model_store=ModelStore(store_root), feature_root=FEATURES,
            spectral_cache_root=SPECTRAL, split_manifest_path=SPLIT, preprocessing_config_path=_config(bundle),
        )
