"""Single-Modality Benchmarking: frozen log-power-spectrum window inputs."""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing.config import load_config
from src.preprocessing.signals import SensorSignal
from src.preprocessing.spectral_windows import (
    LOG_POWER_EPSILON,
    SPECTRAL_REPRESENTATION_VERSION,
    fixed_window_samples,
    log_power_spectrum_input,
    spectral_input_dim,
)


REGIME_S = "configs/preprocessing/mechasight_few_shot_regime_s.json"


@pytest.mark.parametrize(
    ("machine", "sensor", "expected"),
    [
        ("RoboticArm", "mic", 1598),
        ("RoboticArm", "acc", 704),
        ("RoboticArm", "gyro", 704),
        ("BrushlessMotor", "mic", 1598),
        ("BrushlessMotor", "acc", 711),
        ("BrushlessMotor", "gyro", 711),
    ],
)
def test_fixed_window_samples_is_floor_rate_times_window_minus_two(machine, sensor, expected):
    config = load_config(REGIME_S)
    assert fixed_window_samples(config, machine, sensor) == expected


@pytest.mark.parametrize(
    ("machine", "modality", "expected"),
    [
        ("RoboticArm", "mic", 800),
        ("RoboticArm", "acc", 1059),
        ("RoboticArm", "gyro", 1059),
        ("BrushlessMotor", "mic", 800),
        ("BrushlessMotor", "acc", 1068),
        ("BrushlessMotor", "gyro", 1068),
    ],
)
def test_spectral_input_dim_matches_frozen_dimensions(machine, modality, expected):
    config = load_config(REGIME_S)
    assert spectral_input_dim(config, machine, modality) == expected


def _signal(sensor: str, rate: float, samples: int, channels: int, seed: int = 0) -> SensorSignal:
    rng = np.random.default_rng(seed)
    timestamps = np.arange(samples) / rate
    values = rng.normal(size=(samples, channels))
    names = tuple(f"{sensor}_{index}" for index in range(channels))
    return SensorSignal(sensor, timestamps, values, names, samples, samples)


def test_log_power_input_is_channel_major_and_finite():
    config = load_config(REGIME_S)
    rate = config.resampling.target_rates_hz["RoboticArm"]["acc"]
    signal = _signal("acc", rate, 707, 3)
    vector = log_power_spectrum_input(signal, config, "RoboticArm")
    assert vector.shape == (1059,)
    assert np.all(np.isfinite(vector))
    # Channel-major layout: bins of channel 0 first, then channel 1, then 2.
    single = log_power_spectrum_input(
        SensorSignal("acc", signal.timestamps, signal.values[:, :1], ("acc_0",), 707, 707),
        config, "RoboticArm",
    )
    assert single.shape == (353,)
    np.testing.assert_allclose(vector[:353], single)


def test_log_power_input_truncates_to_fixed_length_deterministically():
    config = load_config(REGIME_S)
    rate = config.resampling.target_rates_hz["RoboticArm"]["mic"]
    longer = _signal("mic", rate, 1600, 1)
    shorter = SensorSignal(
        "mic", longer.timestamps[:1598], longer.values[:1598], longer.channels, 1598, 1598
    )
    np.testing.assert_allclose(
        log_power_spectrum_input(longer, config, "RoboticArm"),
        log_power_spectrum_input(shorter, config, "RoboticArm"),
    )


def test_log_power_input_refuses_windows_shorter_than_fixed_length():
    config = load_config(REGIME_S)
    rate = config.resampling.target_rates_hz["RoboticArm"]["mic"]
    with pytest.raises(ValueError, match="fixed-length"):
        log_power_spectrum_input(_signal("mic", rate, 1597, 1), config, "RoboticArm")


def test_measured_minimum_windows_are_accepted():
    """Smallest counts observed over all train/validation windows (2026-09-13)."""
    config = load_config(REGIME_S)
    for machine, sensor, samples, channels in (
        ("RoboticArm", "mic", 1598, 1), ("RoboticArm", "acc", 705, 3),
        ("BrushlessMotor", "mic", 1599, 1), ("BrushlessMotor", "gyro", 712, 3),
    ):
        rate = config.resampling.target_rates_hz[machine][sensor]
        vector = log_power_spectrum_input(_signal(sensor, rate, samples, channels), config, machine)
        assert vector.shape == (spectral_input_dim(config, machine, sensor),)


def test_log_power_uses_frozen_epsilon_on_silent_window():
    config = load_config(REGIME_S)
    rate = config.resampling.target_rates_hz["RoboticArm"]["mic"]
    silent = _signal("mic", rate, 1600, 1)
    silent = SensorSignal(
        "mic", silent.timestamps, np.zeros_like(silent.values), silent.channels, 1600, 1600
    )
    vector = log_power_spectrum_input(silent, config, "RoboticArm")
    assert LOG_POWER_EPSILON == 1e-12
    np.testing.assert_allclose(vector, np.log(LOG_POWER_EPSILON))
    assert SPECTRAL_REPRESENTATION_VERSION.startswith("phase4-")
