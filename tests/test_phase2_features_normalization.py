from __future__ import annotations

import numpy as np
import pytest

from src.features.extraction import Band, frequency_features, one_sided_power, time_features
from src.preprocessing.normalization import FittedChannelStandardizer, standardize_window


def test_time_features_known_constant_and_rms():
    values = np.array([[3.0], [3.0], [3.0], [3.0]])
    features = time_features(values)
    assert features["mean"][0] == 3.0
    assert features["rms"][0] == 3.0
    assert features["variance"][0] == 0.0
    assert features["energy"][0] == 36.0
    assert features["crest_factor"][0] == 1.0


def test_sine_frequency_psd_peak_centroid_and_parseval_energy():
    fs = 1000.0
    t = np.arange(1000) / fs
    values = np.sin(2 * np.pi * 50 * t)[:, None]
    features = frequency_features(values, fs, bands=[Band("tone", 45, 55)])
    assert features["dominant_frequency_hz"][0] == 50.0
    assert np.isclose(features["spectral_centroid_hz"][0], 50.0, atol=1e-9)
    assert np.isclose(features["spectral_energy"][0], 0.5, atol=1e-12)
    assert np.isclose(features["band_energy_tone"][0], 0.5, atol=1e-12)


def test_one_sided_power_matches_mean_square_on_non_sine_signal():
    rng = np.random.default_rng(11)
    values = rng.normal(size=(257, 2))
    _, power = one_sided_power(values, 100.0)
    assert np.allclose(power.sum(axis=0), np.mean(values**2, axis=0))


def test_std_window_is_per_channel_and_deterministic():
    values = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    transformed = standardize_window(values)
    assert np.allclose(transformed.mean(axis=0), 0.0, atol=1e-12)
    assert np.all(transformed.std(axis=0) < 1.0)
    assert np.allclose(transformed, standardize_window(values))


def test_train_fitted_scaler_ignores_validation_and_test_populations():
    train = [np.array([[0.0], [2.0]])]
    scaler = FittedChannelStandardizer.fit(train, split="train", segment_ids=["train-a"])
    validation = scaler.transform(np.array([[100.0]]))[0, 0]
    test = scaler.transform(np.array([[-100.0]]))[0, 0]
    assert scaler.mean[0] == 1.0
    assert validation > 90
    assert test < -90
    assert scaler.fit_segment_ids == ("train-a",)


def test_scaler_refuses_non_train_fit():
    with pytest.raises(ValueError, match="train only"):
        FittedChannelStandardizer.fit(
            [np.array([[1.0]])], split="validation", segment_ids=["v"]
        )
