"""Single-Modality Benchmarking: Ledoit-Wolf shrinkage Mahalanobis baseline."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.covariance import LedoitWolf

from src.models.mahalanobis import (
    MAHALANOBIS_IMPLEMENTATION_VERSION,
    MAHALANOBIS_SPEC,
    ShrinkageMahalanobis,
)


def _data(seed=0, n=300, d=6):
    rng = np.random.default_rng(seed)
    basis = rng.normal(size=(d, d))
    return rng.normal(size=(n, d)) @ basis


def test_spec_is_deterministic_squared_distance():
    assert MAHALANOBIS_SPEC.family == "mahalanobis_shrinkage"
    assert MAHALANOBIS_SPEC.stochastic is False
    assert MAHALANOBIS_SPEC.hyperparameters["covariance_estimator"] == "sklearn.covariance.LedoitWolf"
    assert MAHALANOBIS_SPEC.hyperparameters["assume_centered"] is False
    assert "squared" in MAHALANOBIS_SPEC.score_transform
    assert MAHALANOBIS_IMPLEMENTATION_VERSION.startswith("phase4-")


def test_score_equals_ledoit_wolf_squared_mahalanobis():
    X = _data()
    model = ShrinkageMahalanobis().fit(X)
    reference = LedoitWolf().fit(X).mahalanobis(X[:20])
    np.testing.assert_allclose(model.score(X[:20]), reference)
    assert np.all(model.score(X[:20]) >= 0)


def test_fit_is_deterministic_across_calls():
    X = _data()
    a = ShrinkageMahalanobis().fit(X).score(X)
    b = ShrinkageMahalanobis().fit(X).score(X)
    np.testing.assert_array_equal(a, b)


def test_off_distribution_points_score_higher():
    X = _data()
    model = ShrinkageMahalanobis().fit(X)
    inside = model.score(X).mean()
    outside = model.score(X + 10.0).mean()
    assert outside > inside


def test_refuses_schema_mismatch_and_non_finite():
    X = _data()
    model = ShrinkageMahalanobis().fit(X)
    with pytest.raises(ValueError, match="feature-schema"):
        model.score(X[:, :3])
    bad = X.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        ShrinkageMahalanobis().fit(bad)


def test_save_and_load_round_trip(tmp_path):
    X = _data()
    model = ShrinkageMahalanobis().fit(X)
    path = model.save(tmp_path / "m.joblib")
    loaded = ShrinkageMahalanobis.load(path)
    np.testing.assert_array_equal(loaded.score(X[:5]), model.score(X[:5]))
    assert loaded.model_id == model.model_id
    assert loaded.n_features_in_ == X.shape[1]
