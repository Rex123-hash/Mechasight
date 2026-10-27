"""Small, documented feature set for later classical baselines.

Features are computed per channel.  Time-domain features operate on native
samples and need no resampling.  Frequency-domain features require a regular
grid and an explicit sampling rate; callers must use the configured resampling
path when timestamps are nonuniform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


TIME_FEATURES = (
    "mean", "std", "rms", "variance", "peak_abs", "peak_to_peak",
    "skewness", "kurtosis_excess", "crest_factor", "energy",
)
FREQUENCY_FEATURES = (
    "dominant_frequency_hz", "spectral_energy", "spectral_centroid_hz",
    "spectral_spread_hz",
)


FEATURE_DEFINITIONS = {
    "mean": "Arithmetic mean; same unit as the channel.",
    "std": "Population standard deviation; same unit as the channel.",
    "rms": "sqrt(mean(x^2)); same unit as the channel.",
    "variance": "Population variance; squared channel unit.",
    "peak_abs": "max(abs(x)); same unit as the channel.",
    "peak_to_peak": "max(x)-min(x); same unit as the channel.",
    "skewness": "Third standardized central moment; dimensionless.",
    "kurtosis_excess": "Fourth standardized central moment minus 3; dimensionless.",
    "crest_factor": "peak_abs/RMS; dimensionless.",
    "energy": "sum(x^2); squared channel unit times samples.",
    "dominant_frequency_hz": "Largest non-DC one-sided FFT power bin, in Hz.",
    "spectral_energy": "One-sided Parseval power sum (= mean square), squared channel unit.",
    "spectral_centroid_hz": "Power-weighted mean frequency, in Hz.",
    "spectral_spread_hz": "Power-weighted standard deviation about centroid, in Hz.",
    "band_energy": "One-sided Parseval power within [low, high), squared channel unit.",
}


@dataclass(frozen=True)
class Band:
    name: str
    low_hz: float
    high_hz: float

    def __post_init__(self) -> None:
        if self.low_hz < 0 or self.high_hz <= self.low_hz:
            raise ValueError("frequency band must satisfy 0 <= low < high")


def _as_channels(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or len(array) == 0:
        raise ValueError("features require a non-empty (samples, channels) array")
    if not np.all(np.isfinite(array)):
        raise ValueError("features require finite values")
    return array


def time_features(values: np.ndarray) -> dict[str, np.ndarray]:
    x = _as_channels(values)
    mean = x.mean(axis=0)
    centered = x - mean
    variance = np.mean(np.square(centered), axis=0)
    std = np.sqrt(variance)
    rms = np.sqrt(np.mean(np.square(x), axis=0))
    peak = np.max(np.abs(x), axis=0)
    safe_std = np.where(std > 0, std, np.nan)
    skewness = np.mean(np.power(centered / safe_std, 3), axis=0)
    kurtosis = np.mean(np.power(centered / safe_std, 4), axis=0) - 3.0
    # Constant signals have defined zero higher moments here, avoiding NaNs in
    # feature caches while recording their lack of distributional shape.
    skewness = np.nan_to_num(skewness, nan=0.0)
    kurtosis = np.nan_to_num(kurtosis, nan=0.0)
    crest = np.divide(peak, rms, out=np.zeros_like(peak), where=rms > 0)
    return {
        "mean": mean,
        "std": std,
        "rms": rms,
        "variance": variance,
        "peak_abs": peak,
        "peak_to_peak": np.ptp(x, axis=0),
        "skewness": skewness,
        "kurtosis_excess": kurtosis,
        "crest_factor": crest,
        "energy": np.sum(np.square(x), axis=0),
    }


def one_sided_power(values: np.ndarray, sampling_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Return FFT bins and one-sided Parseval power per channel."""

    x = _as_channels(values)
    if sampling_rate_hz <= 0:
        raise ValueError("sampling rate must be positive")
    n = len(x)
    spectrum = np.fft.rfft(x, axis=0)
    power = np.square(np.abs(spectrum)) / (n * n)
    if n % 2 == 0:
        if len(power) > 2:
            power[1:-1] *= 2.0
    elif len(power) > 1:
        power[1:] *= 2.0
    frequency = np.fft.rfftfreq(n, d=1.0 / sampling_rate_hz)
    return frequency, power


def frequency_features(
    values: np.ndarray,
    sampling_rate_hz: float,
    *,
    bands: Iterable[Band] = (),
) -> dict[str, np.ndarray]:
    frequency, power = one_sided_power(values, sampling_rate_hz)
    total = power.sum(axis=0)
    non_dc = power.copy()
    non_dc[0] = 0.0
    dominant = frequency[np.argmax(non_dc, axis=0)]
    centroid = np.divide(
        (frequency[:, None] * power).sum(axis=0), total,
        out=np.zeros_like(total), where=total > 0,
    )
    spread = np.sqrt(np.divide(
        (np.square(frequency[:, None] - centroid) * power).sum(axis=0), total,
        out=np.zeros_like(total), where=total > 0,
    ))
    result = {
        "dominant_frequency_hz": dominant,
        "spectral_energy": total,
        "spectral_centroid_hz": centroid,
        "spectral_spread_hz": spread,
    }
    for band in bands:
        mask = (frequency >= band.low_hz) & (frequency < band.high_hz)
        result[f"band_energy_{band.name}"] = power[mask].sum(axis=0)
    return result


def flatten_features(
    sensor: str, channels: tuple[str, ...], features: dict[str, np.ndarray]
) -> dict[str, float]:
    output: dict[str, float] = {}
    for feature, values in features.items():
        for channel, value in zip(channels, values, strict=True):
            output[f"{sensor}__{channel}__{feature}"] = float(value)
    return output
