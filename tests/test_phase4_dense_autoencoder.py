"""Single-Modality Benchmarking: frozen Protocol-B dense autoencoder."""

from __future__ import annotations

import numpy as np
import pytest

from src.experiment.provenance import file_sha256
from src.models.dense_autoencoder import (
    AUTOENCODER_IMPLEMENTATION_VERSION,
    DENSE_AE_SPEC,
    DenseAutoencoder,
    DenseAutoencoderSpec,
)


def _windows(seed: int, n: int = 512, d: int = 24):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 3))
    mixing = rng.normal(size=(3, d))
    return (latent @ mixing + 0.05 * rng.normal(size=(n, d))).astype(np.float32)


def test_frozen_spec_values():
    assert DENSE_AE_SPEC.hidden == (256, 64)
    assert DENSE_AE_SPEC.bottleneck == 16
    assert DENSE_AE_SPEC.activation == "relu"
    assert DENSE_AE_SPEC.output == "linear"
    assert DENSE_AE_SPEC.loss == "mse"
    assert DENSE_AE_SPEC.optimizer == "adam"
    assert DENSE_AE_SPEC.learning_rate == 1e-3
    assert DENSE_AE_SPEC.weight_decay == 0.0
    assert DENSE_AE_SPEC.batch_size == 256
    assert DENSE_AE_SPEC.max_epochs == 200
    assert DENSE_AE_SPEC.patience == 10
    assert DENSE_AE_SPEC.checkpoint_selection == "best_source_validation_window_mse"
    assert AUTOENCODER_IMPLEMENTATION_VERSION.startswith("phase4-")


def test_parameter_count_matches_frozen_architecture():
    model = DenseAutoencoder(input_dim=800, seed=0)
    expected = 2 * (800 * 256 + 256 * 64 + 64 * 16) + 2 * (256 + 64) + 16 + 800
    assert model.parameter_count == expected


def test_training_is_deterministic_for_a_seed():
    spec = DenseAutoencoderSpec(hidden=(16, 8), bottleneck=4, max_epochs=5, patience=3)
    train, validation = _windows(1), _windows(2, n=128)
    a = DenseAutoencoder(input_dim=24, seed=3, spec=spec).fit(train, validation)
    b = DenseAutoencoder(input_dim=24, seed=3, spec=spec).fit(train, validation)
    np.testing.assert_array_equal(a.score_windows(validation), b.score_windows(validation))
    assert a.history["validation_window_mse"] == b.history["validation_window_mse"]


def test_different_seeds_give_different_models():
    spec = DenseAutoencoderSpec(hidden=(16, 8), bottleneck=4, max_epochs=3, patience=3)
    train, validation = _windows(1), _windows(2, n=128)
    a = DenseAutoencoder(input_dim=24, seed=0, spec=spec).fit(train, validation)
    b = DenseAutoencoder(input_dim=24, seed=1, spec=spec).fit(train, validation)
    assert not np.array_equal(a.score_windows(validation), b.score_windows(validation))


def test_early_stopping_keeps_best_validation_checkpoint():
    spec = DenseAutoencoderSpec(hidden=(16, 8), bottleneck=4, max_epochs=60, patience=2)
    train, validation = _windows(1), _windows(2, n=128)
    model = DenseAutoencoder(input_dim=24, seed=0, spec=spec).fit(train, validation)
    history = model.history
    losses = history["validation_window_mse"]
    assert history["best_epoch"] == int(np.argmin(losses)) + 1
    assert history["epochs_run"] == len(losses)
    assert history["epochs_run"] <= spec.max_epochs
    if history["stopped_early"]:
        assert history["epochs_run"] == history["best_epoch"] + spec.patience
    # Scores come from the best checkpoint, so they reproduce the best epoch's loss.
    assert model.score_windows(validation).mean() == pytest.approx(min(losses), rel=1e-5)


def test_window_score_is_mean_squared_reconstruction_error():
    spec = DenseAutoencoderSpec(hidden=(16, 8), bottleneck=4, max_epochs=2, patience=2)
    train, validation = _windows(1), _windows(2, n=32)
    model = DenseAutoencoder(input_dim=24, seed=0, spec=spec).fit(train, validation)
    reconstructed = model.reconstruct(validation)
    manual = np.mean((reconstructed - validation) ** 2, axis=1)
    np.testing.assert_allclose(model.score_windows(validation), manual, rtol=1e-5)
    assert model.score_windows(validation).shape == (32,)


def test_reduced_input_learns_better_than_untrained():
    spec = DenseAutoencoderSpec(hidden=(16, 8), bottleneck=4, max_epochs=30, patience=30)
    train, validation = _windows(1), _windows(2, n=128)
    model = DenseAutoencoder(input_dim=24, seed=0, spec=spec)
    before = model.score_windows(validation).mean()
    model.fit(train, validation)
    assert model.score_windows(validation).mean() < before


def test_save_load_round_trip_and_hash(tmp_path):
    spec = DenseAutoencoderSpec(hidden=(16, 8), bottleneck=4, max_epochs=2, patience=2)
    train, validation = _windows(1), _windows(2, n=32)
    model = DenseAutoencoder(input_dim=24, seed=4, spec=spec).fit(train, validation)
    path = model.save(tmp_path / "ae.pth")
    loaded = DenseAutoencoder.load(path)
    np.testing.assert_array_equal(loaded.score_windows(validation), model.score_windows(validation))
    assert loaded.seed == 4 and loaded.input_dim == 24 and loaded.spec == spec
    assert loaded.history == model.history
    assert loaded.model_id == model.model_id
    assert len(file_sha256(path)) == 64


def test_refuses_dimension_mismatch_and_non_finite():
    spec = DenseAutoencoderSpec(hidden=(16, 8), bottleneck=4, max_epochs=1, patience=1)
    model = DenseAutoencoder(input_dim=24, seed=0, spec=spec)
    with pytest.raises(ValueError, match="dimension"):
        model.score_windows(np.zeros((3, 10), dtype=np.float32))
    bad = np.zeros((3, 24), dtype=np.float32)
    bad[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        model.fit(bad, bad)
