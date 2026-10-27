"""Frozen log-power-spectrum window inputs for the single-modality dense AE.

One window of one modality becomes one fixed-length vector:

    frozen ``resample_regular`` -> truncate to ``fixed_window_samples``
    -> frozen ``one_sided_power`` -> ``log(power + 1e-12)``
    -> concatenate channels (channel-major) within that modality only.

The fixed length is ``floor(window_length_s * target_rate) - 2``.  Measured
over all 194,165 authorized training/validation windows (2026-09-13, before
any model was fit), the frozen resampler yields counts from two below to one
above that floor (RoboticArm mic 1598..1600, IMU 705..707; BrushlessMotor mic
1599..1600, acc 713..714, gyro 712..714).  ``-1`` was the first proposal and
was falsified by 121 RoboticArm mic windows; ``-2`` covers every observed
window with one sample of headroom.  A shorter window is a technical defect.
"""

from __future__ import annotations

import math

import numpy as np

from src.features.extraction import one_sided_power
from src.preprocessing.config import PreprocessingConfig
from src.preprocessing.signals import SensorSignal, resample_regular


SPECTRAL_REPRESENTATION_VERSION = "phase4-log-power-spectrum-v1"
LOG_POWER_EPSILON = 1e-12
FIXED_LENGTH_MARGIN = 2
MODALITIES = ("mic", "acc", "gyro")
MODALITY_CHANNELS = {"mic": 1, "acc": 3, "gyro": 3}


def fixed_window_samples(config: PreprocessingConfig, machine: str, sensor: str) -> int:
    rate = float(config.resampling.target_rates_hz[machine][sensor])
    return int(math.floor(config.window_length_s * rate + 1e-9)) - FIXED_LENGTH_MARGIN


def spectral_bins(config: PreprocessingConfig, machine: str, sensor: str) -> int:
    return fixed_window_samples(config, machine, sensor) // 2 + 1


def spectral_input_dim(config: PreprocessingConfig, machine: str, modality: str) -> int:
    if modality not in MODALITIES:
        raise ValueError(f"unknown modality {modality!r}")
    return spectral_bins(config, machine, modality) * MODALITY_CHANNELS[modality]


def log_power_spectrum_input(
    signal: SensorSignal, config: PreprocessingConfig, machine: str
) -> np.ndarray:
    """Return the frozen fixed-length log-power vector for one window."""

    rate = float(config.resampling.target_rates_hz[machine][signal.sensor])
    regular = resample_regular(signal, rate, method=config.resampling.method)
    length = fixed_window_samples(config, machine, signal.sensor)
    if len(regular.values) < length:
        raise ValueError(
            f"{signal.sensor} window has {len(regular.values)} resampled samples, "
            f"below the frozen fixed-length {length}"
        )
    _frequency, power = one_sided_power(regular.values[:length], rate)
    vector = np.log(power + LOG_POWER_EPSILON).T.reshape(-1)
    if not np.all(np.isfinite(vector)):
        raise ValueError("log-power spectrum produced non-finite values")
    return vector.astype(np.float64)
