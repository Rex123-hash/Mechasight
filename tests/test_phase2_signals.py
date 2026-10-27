from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.preprocessing.config import load_config
from src.preprocessing.signals import (
    SensorSignal,
    apply_startup_transformations,
    common_coverage,
    iter_aligned_windows,
    resample_regular,
    window_bounds,
)


def _signal(sensor: str, timestamps: np.ndarray, channels: int = 1) -> SensorSignal:
    values = np.column_stack([timestamps + channel for channel in range(channels)])
    return SensorSignal(
        sensor, timestamps, values, tuple(f"c{c}" for c in range(channels)),
        len(timestamps), len(timestamps),
    )


def test_alignment_uses_physical_intersection_not_index():
    signals = {
        "mic": _signal("mic", np.arange(0.0, 1.0, 0.01)),
        "acc": _signal("acc", np.arange(0.05, 1.05, 0.02)),
        "gyro": _signal("gyro", np.arange(0.02, 0.92, 0.015)),
    }
    start, end = common_coverage(signals)
    assert np.isclose(start, 0.05)
    assert np.isclose(end, signals["gyro"].timestamps[-1])


def test_window_bounds_are_half_open_and_do_not_exceed_coverage():
    bounds = window_bounds(0.05, 0.55, 0.2, 0.1)
    assert bounds == [(0.05, 0.25), (0.15000000000000002, 0.35000000000000003),
                      (0.25, 0.45), (0.35000000000000003, 0.55)]
    assert all(end <= 0.55 for _, end in bounds)


def test_bm_startup_rule_drops_one_acc_sample_only():
    config = load_config("configs/preprocessing/mechasight_regime_t.json")
    t = np.arange(10, dtype=float) * 0.001
    signals = {sensor: _signal(sensor, t, 3 if sensor != "mic" else 1) for sensor in ("mic", "acc", "gyro")}
    output, log = apply_startup_transformations(signals, "BrushlessMotor", replace(config, gyro_warmup_s=0.0))
    assert len(output["acc"].timestamps) == 9
    assert len(output["mic"].timestamps) == 10
    assert log["accelerometer_samples_dropped"] == 1


def test_ra_startup_rule_does_not_change_accelerometer():
    config = load_config("configs/preprocessing/mechasight_regime_t.json")
    t = np.arange(50, dtype=float) * 0.001
    signals = {sensor: _signal(sensor, t, 3 if sensor != "mic" else 1) for sensor in ("mic", "acc", "gyro")}
    output, log = apply_startup_transformations(signals, "RoboticArm", replace(config, gyro_warmup_s=0.0))
    assert len(output["acc"].timestamps) == 50
    assert log["accelerometer_samples_dropped"] == 0


def test_gyro_warmup_crop_preserves_absolute_alignment():
    config = load_config("configs/preprocessing/mechasight_regime_t.json")
    t = 10.0 + np.arange(101, dtype=float) * 0.001
    signals = {sensor: _signal(sensor, t, 3 if sensor != "mic" else 1) for sensor in ("mic", "acc", "gyro")}
    output, log = apply_startup_transformations(signals, "RoboticArm", config)
    assert output["gyro"].timestamps[0] >= 10.035
    assert np.isclose(common_coverage(output)[0], 10.035)
    assert log["gyro_samples_dropped"] == 35


def test_irregular_sampling_linear_resampling_has_regular_grid_and_no_extrapolation():
    t = np.array([0.0, 0.09, 0.21, 0.31, 0.41])
    signal = SensorSignal("acc", t, (2 * t)[:, None], ("x",), 5, 5)
    regular = resample_regular(signal, 10.0)
    assert np.allclose(np.diff(regular.timestamps), 0.1)
    assert regular.timestamps[-1] <= t[-1]
    assert np.allclose(regular.values[:, 0], 2 * regular.timestamps)
    assert regular.raw_sample_count == 5
    assert regular.transformed_sample_count == 5


def test_official_windows_have_fixed_native_lengths():
    config = load_config("configs/preprocessing/official_baseline_reproduction.json")
    signals = {
        "mic": _signal("mic", np.arange(0.0, 0.21, 1 / 16000)),
        "acc": _signal("acc", np.arange(0.0, 0.21, 1 / 7135), 3),
        "gyro": _signal("gyro", np.arange(0.0, 0.21, 1 / 7135), 3),
    }
    config = replace(config, gyro_warmup_s=0.0)
    window = next(iter_aligned_windows(signals, config))
    assert len(window.signals["mic"].values) == 1600
    assert len(window.signals["acc"].values) == 706
    assert len(window.signals["gyro"].values) == 706


def test_official_first_window_starts_at_exact_gyro_cutoff_not_next_sample():
    config = load_config("configs/preprocessing/official_baseline_reproduction.json")
    t = np.arange(0.0, 0.3, 0.0013)
    signals = {sensor: _signal(sensor, t, 3 if sensor != "mic" else 1) for sensor in ("mic", "acc", "gyro")}
    transformed, _ = apply_startup_transformations(signals, "RoboticArm", config)
    first = next(iter_aligned_windows(transformed, config))
    assert np.isclose(first.start_s, 0.035)
