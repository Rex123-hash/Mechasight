"""Timestamp-first loading, alignment, artefact handling, and windowing."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from src.preprocessing.config import OFFICIAL_PROTOCOL, PreprocessingConfig


SENSOR_KEYS = ("mic", "acc", "gyro")


@dataclass(frozen=True)
class SensorSignal:
    sensor: str
    timestamps: np.ndarray
    values: np.ndarray
    channels: tuple[str, ...]
    raw_sample_count: int
    transformed_sample_count: int
    transformation: str = "raw"
    alignment_start_s: float | None = None

    def __post_init__(self) -> None:
        if self.timestamps.ndim != 1:
            raise ValueError("timestamps must be one-dimensional")
        if self.values.ndim != 2:
            raise ValueError("values must have shape (samples, channels)")
        if len(self.timestamps) != len(self.values):
            raise ValueError("timestamp/value sample counts differ")
        if len(self.channels) != self.values.shape[1]:
            raise ValueError("channel names do not match values")
        finite = np.isfinite(self.timestamps)
        if finite.any() and not np.all(finite[: np.count_nonzero(finite)]):
            raise ValueError("non-finite timestamp padding is allowed only at the tail")
        physical = self.timestamps[finite]
        if len(physical) > 1 and not np.all(np.diff(physical) > 0):
            raise ValueError("physical timestamps must be strictly increasing")


@dataclass(frozen=True)
class WindowedSignals:
    window_index: int
    start_s: float
    end_s: float
    signals: dict[str, SensorSignal]


def read_sensor_parquet(path: str | Path, sensor: str) -> SensorSignal:
    frame = pd.read_parquet(path)
    time_candidates = [column for column in frame.columns if str(column).lower() in {"time", "timestamp", "t"}]
    if len(time_candidates) != 1:
        raise ValueError(f"expected one timestamp column in {path}, found {time_candidates}")
    time_column = time_candidates[0]
    channels = tuple(str(column) for column in frame.columns if column != time_column)
    timestamps = pd.to_numeric(frame[time_column], errors="raise").to_numpy(np.float64)
    values = frame[list(channels)].to_numpy(np.float64, copy=True)
    order = np.argsort(timestamps, kind="stable")
    timestamps, values = timestamps[order], values[order]
    if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"non-increasing timestamps in {path}")
    return SensorSignal(sensor, timestamps, values, channels, len(values), len(values))


def apply_startup_transformations(
    signals: dict[str, SensorSignal], machine: str, config: PreprocessingConfig
) -> tuple[dict[str, SensorSignal], dict[str, int]]:
    """Apply explicit machine-aware startup corrections before alignment."""

    output = dict(signals)
    log = {"accelerometer_samples_dropped": 0, "gyro_samples_dropped": 0}

    acc = output.get("acc")
    n_drop = int(config.drop_initial_acc_samples.get(machine, 0))
    if acc is not None and n_drop:
        if len(acc.timestamps) <= n_drop:
            raise ValueError("accelerometer startup rule removes the complete stream")
        output["acc"] = replace(
            acc,
            timestamps=acc.timestamps[n_drop:],
            values=acc.values[n_drop:],
            transformed_sample_count=len(acc.values) - n_drop,
            transformation=f"drop_initial_samples:{n_drop}",
        )
        log["accelerometer_samples_dropped"] = n_drop

    gyro = output.get("gyro")
    if gyro is not None and config.gyro_warmup_s > 0:
        cutoff = float(gyro.timestamps[0] + config.gyro_warmup_s)
        first = int(np.searchsorted(gyro.timestamps, cutoff, side="left"))
        if first >= len(gyro.timestamps):
            raise ValueError("gyro warm-up removes the complete stream")
        output["gyro"] = replace(
            gyro,
            timestamps=gyro.timestamps[first:],
            values=gyro.values[first:],
            transformed_sample_count=len(gyro.values) - first,
            transformation=f"crop_warmup_s:{config.gyro_warmup_s}",
            # The upstream and experimental rule starts coverage at the exact
            # physical cutoff, not at the next discrete gyro sample.
            alignment_start_s=cutoff,
        )
        log["gyro_samples_dropped"] = first
    return output, log


def common_coverage(signals: dict[str, SensorSignal]) -> tuple[float, float]:
    if set(signals) != set(SENSOR_KEYS):
        raise ValueError(f"all modalities are required, found {sorted(signals)}")
    if any(len(signal.timestamps) == 0 for signal in signals.values()):
        raise ValueError("cannot align an empty sensor stream")
    start = max(
        float(signal.alignment_start_s)
        if signal.alignment_start_s is not None
        else float(signal.timestamps[0])
        for signal in signals.values()
    )
    end = min(float(signal.timestamps[-1]) for signal in signals.values())
    if end <= start:
        raise ValueError(f"sensor streams have no common coverage ({start}, {end})")
    return start, end


def window_bounds(
    start_s: float, end_s: float, length_s: float, stride_s: float
) -> list[tuple[float, float]]:
    """Return deterministic half-open physical windows fully inside coverage."""

    if length_s <= 0 or stride_s <= 0:
        raise ValueError("window length and stride must be positive")
    tolerance = 1e-12 * max(1.0, abs(start_s), abs(end_s))
    available = end_s - start_s
    if available + tolerance < length_s:
        return []
    count = int(np.floor((available - length_s + tolerance) / stride_s)) + 1
    return [
        (float(start_s + index * stride_s), float(start_s + index * stride_s + length_s))
        for index in range(count)
    ]


def _slice_signal(
    signal: SensorSignal,
    start_s: float,
    end_s: float,
    *,
    right_inclusive: bool,
) -> SensorSignal:
    left = int(np.searchsorted(signal.timestamps, start_s, side="left"))
    right = int(np.searchsorted(
        signal.timestamps, end_s, side="right" if right_inclusive else "left"
    ))
    return replace(
        signal,
        timestamps=signal.timestamps[left:right].copy(),
        values=signal.values[left:right].copy(),
        transformed_sample_count=right - left,
        transformation=signal.transformation + ("|right_inclusive" if right_inclusive else "|half_open"),
    )


def _official_shape(signal: SensorSignal, config: PreprocessingConfig) -> SensorSignal:
    expected = int(config.official_fixed_rates_hz[signal.sensor] * config.window_length_s)
    values = signal.values[:expected]
    timestamps = signal.timestamps[:expected]
    if len(values) < expected:
        pad = expected - len(values)
        values = np.vstack([values, np.zeros((pad, signal.values.shape[1]), dtype=np.float64)])
        # Padded values have no physical timestamps. NaN makes that explicit.
        timestamps = np.concatenate([timestamps, np.full(pad, np.nan)])
    return replace(
        signal,
        timestamps=timestamps,
        values=values,
        transformed_sample_count=expected,
        transformation=signal.transformation + f"|official_pad_truncate:{expected}",
    )


def iter_aligned_windows(
    signals: dict[str, SensorSignal], config: PreprocessingConfig
) -> Iterator[WindowedSignals]:
    start_s, end_s = common_coverage(signals)
    bounds = window_bounds(start_s, end_s, config.window_length_s, config.stride_s)
    right_inclusive = config.protocol == OFFICIAL_PROTOCOL
    for index, (start, end) in enumerate(bounds):
        sliced = {
            sensor: _slice_signal(signal, start, end, right_inclusive=right_inclusive)
            for sensor, signal in signals.items()
        }
        if config.protocol == OFFICIAL_PROTOCOL:
            sliced = {sensor: _official_shape(signal, config) for sensor, signal in sliced.items()}
        yield WindowedSignals(index, start, end, sliced)


def resample_regular(
    signal: SensorSignal, target_rate_hz: float, *, method: str = "linear"
) -> SensorSignal:
    """Interpolate a stream to a regular grid without extrapolating its edges."""

    if method != "linear":
        raise ValueError("only scientifically documented linear interpolation is supported")
    if target_rate_hz <= 0:
        raise ValueError("target rate must be positive")
    finite = np.isfinite(signal.timestamps)
    timestamps = signal.timestamps[finite]
    values = signal.values[finite]
    if len(timestamps) < 2:
        raise ValueError("at least two physical samples are required for resampling")
    duration = float(timestamps[-1] - timestamps[0])
    count = int(np.floor(duration * target_rate_hz + 1e-12)) + 1
    grid = timestamps[0] + np.arange(count, dtype=np.float64) / target_rate_hz
    grid = grid[grid <= timestamps[-1] + 1e-12]
    transformed = np.column_stack([
        np.interp(grid, timestamps, values[:, channel])
        for channel in range(values.shape[1])
    ])
    return replace(
        signal,
        timestamps=grid,
        values=transformed,
        transformed_sample_count=len(grid),
        transformation=signal.transformation + f"|linear_resample:{target_rate_hz:g}Hz",
    )


def effective_rate_hz(signal: SensorSignal) -> float:
    finite = signal.timestamps[np.isfinite(signal.timestamps)]
    if len(finite) < 2 or finite[-1] <= finite[0]:
        return float("nan")
    return float((len(finite) - 1) / (finite[-1] - finite[0]))
