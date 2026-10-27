"""Protocol-A faithful reproduction of the published IMAD-DS official baseline.

Upstream: https://github.com/augustif/IMAD-DS_baseline
Pinned commit: 592bcbbed58893dc26bbab3240c5d7a3105a1a7f

This module reproduces the behaviour of the upstream ``preprocess.ipynb``,
``baseline.ipynb``, ``model.py`` and ``utilities.py`` at that commit.  It is a
historical comparability anchor, **not** a MechaSight method.

FIDELITY RULES
--------------
Upstream behaviour is preserved even where MechaSight knows better.  In
particular this module deliberately keeps:

* the fixed rate constants 16000 / 7063 / 7063 Hz, not the measured
  machine-specific effective rates;
* right-inclusive pandas datetime slicing, so adjacent 100 ms windows can share
  a boundary sample;
* zero-pad / truncate to ``int(fs * 0.1)`` samples;
* no BrushlessMotor first-accelerometer-sample correction;
* the released train/test split, including the RoboticArm parent-recording
  overlap, which Protocol A exists to preserve;
* window-level (not recording-level) train/validation splitting;
* per-window standardisation applied independently to train, validation and
  test, with no train-fitted statistics;
* the upstream "best model" checkpoint behaviour described in
  :class:`AutoencoderFC` -- see ``UPSTREAM_DEFECTS``.

Nothing here may import Protocol-B splits, features, thresholds or models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

UPSTREAM_REPOSITORY = "https://github.com/augustif/IMAD-DS_baseline"
UPSTREAM_COMMIT = "592bcbbed58893dc26bbab3240c5d7a3105a1a7f"
PROTOCOL_A_IMPLEMENTATION_VERSION = "phase3.3-protocol-a-reproduction-v1"

SENSORS: tuple[str, ...] = ("imp23absu_mic", "ism330dhcx_acc", "ism330dhcx_gyro")
LABEL_NAMES: tuple[str, ...] = (
    "segment_id", "split_label", "anomaly_label", "domain_shift_op", "domain_shift_env",
)

WINDOW_SIZE = pd.to_timedelta("100ms")
GYROSCOPE_WARM_UP_TIME = pd.to_timedelta("35ms")

#: Upstream fixed rate constants. NOT the measured effective rates.
OFFICIAL_SENSOR_SPEC: dict[str, dict[str, int]] = {
    "imp23absu_mic": {"fs": 16000, "number_of_channel": 1},
    "ism330dhcx_acc": {"fs": 7063, "number_of_channel": 3},
    "ism330dhcx_gyro": {"fs": 7063, "number_of_channel": 3},
}

#: Upstream training hyperparameters, read from baseline.ipynb at the pinned commit.
OFFICIAL_PARAMS: dict[str, Any] = {
    "layer_dims": [2048, 2048, 2048, 16],
    "lr": 0.0001,
    "criterion": "MSE",
    "batch_size": 1024,
    "num_epochs": 1000,
    "patience": 3,
    "normalisation": "std_window",
    "valid_size": 0.1,
    "seed": 1995,
}

#: Upstream behaviours that are defects but are reproduced deliberately.
UPSTREAM_DEFECTS: tuple[dict[str, str], ...] = (
    {
        "id": "best_model_state_is_a_live_reference",
        "location": "model.py AutoencoderFC.fit",
        "description": (
            "best_model_state = self.state_dict() stores references to the live "
            "parameter tensors rather than a deep copy, so the saved 'best' state "
            "keeps mutating as training continues. The final load_state_dict "
            "therefore restores the LAST epoch's weights, not the best-validation "
            "epoch's weights."
        ),
        "reproduced": "yes - not corrected; correcting it would change results",
    },
    {
        "id": "right_inclusive_window_slicing",
        "location": "preprocess.ipynb sensor_df[start:end]",
        "description": (
            "pandas label-based datetime slicing is right-inclusive, so adjacent "
            "100 ms windows can share a boundary sample."
        ),
        "reproduced": "yes",
    },
    {
        "id": "window_level_train_validation_split",
        "location": "baseline.ipynb train_test_split",
        "description": (
            "the train/validation split is taken at window level, so windows from "
            "one physical recording appear on both sides."
        ),
        "reproduced": "yes",
    },
    {
        "id": "fixed_imu_rate_constant",
        "location": "preprocess.ipynb sensor_dict",
        "description": (
            "7063 Hz is applied to both machines, although the measured effective "
            "IMU rate differs per machine (RA 7063.29, BM 7135.72)."
        ),
        "reproduced": "yes",
    },
    {
        "id": "bare_except_on_segment_read",
        "location": "preprocess.ipynb",
        "description": "unreadable segments are printed and skipped silently.",
        "reproduced": "behaviour preserved, but every skip is recorded rather than only printed",
    },
)


def official_window_length(sensor: str) -> int:
    """int(fs * 0.1) exactly as upstream computes it."""

    spec = OFFICIAL_SENSOR_SPEC[sensor]
    return int(spec["fs"] * WINDOW_SIZE.total_seconds())


# ------------------------------------------------------------------ preprocess
def official_segment_bounds(
    frames: Mapping[str, pd.DataFrame]
) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    """Upstream common-interval rule and window count.

    start = max(gyro_first + 35 ms, max(all sensors' first timestamps))
    end   = min(all sensors' last timestamps)
    n     = (end - start) // 100 ms
    """

    mins = [frames[s].index[0] for s in OFFICIAL_SENSOR_SPEC]
    maxs = [frames[s].index[-1] for s in OFFICIAL_SENSOR_SPEC]
    start = max(frames["ism330dhcx_gyro"].index[0] + GYROSCOPE_WARM_UP_TIME, max(mins))
    end = min(maxs)
    number_of_window = (end - start) // WINDOW_SIZE
    return start, end, int(number_of_window)


def official_window(
    sensor_df: pd.DataFrame, start: pd.Timestamp, sensor: str
) -> np.ndarray:
    """One upstream window: right-inclusive slice, then zero-pad or truncate.

    Returns (channels, window_length) to match the upstream HDF5 layout.
    """

    spec = OFFICIAL_SENSOR_SPEC[sensor]
    length = official_window_length(sensor)
    channels = spec["number_of_channel"]
    end = start + WINDOW_SIZE
    # Right-inclusive on purpose: this is pandas label slicing, as upstream.
    values = sensor_df[start:end].values
    if len(values) < length:
        padding = np.zeros((length - len(values), channels))
        values = np.vstack([values, padding])
    else:
        values = values[:length, :]
    return values.T


def read_official_sensor_frame(path: str) -> pd.DataFrame:
    """Upstream segment read: parquet -> datetime index in seconds -> sorted."""

    frame = pd.read_parquet(path)
    frame["Time"] = pd.to_datetime(frame["Time"], unit="s")
    frame = frame.set_index("Time")
    frame = frame.sort_index()
    return frame


def standardize_window(data: np.ndarray) -> np.ndarray:
    """Upstream per-window, per-channel standardisation on an (N, C, L) array.

    The epsilon sits in the denominator here, which is the code path the
    published results used.  Mutates in place, exactly as upstream does.
    """

    array = np.asarray(data)
    if array.ndim != 3:
        raise ValueError("standardize_window expects an (N, C, L) array")
    n, c, _ = array.shape
    mean_ = array.mean(2).reshape(n, c, 1)
    std_ = array.std(2).reshape(n, c, 1)
    array -= mean_
    array /= std_ + 1e-5
    return array


# ------------------------------------------------------------------- scoring
def mse_per_sample(actual: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    """Upstream utilities.MSE: mean squared error over dim=1, one value per row."""

    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(estimate, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("MSE expects two identically shaped 2-D arrays")
    return np.mean((a - b) ** 2, axis=1)


def sensor_slices(
    window_lengths: Sequence[int], num_channels: Sequence[int]
) -> list[tuple[int, int]]:
    """Concatenation offsets used by upstream sensor_specific_loss."""

    bounds, start = [], 0
    for length, channels in zip(window_lengths, num_channels, strict=True):
        width = int(length) * int(channels)
        bounds.append((start, start + width))
        start += width
    return bounds


def group_by_segment_id_median(
    labels: pd.DataFrame, score_columns: Iterable[str]
) -> pd.DataFrame:
    """Upstream group_by_segment_id with aggregation_type='median'.

    Label columns take the first value in each group; score columns take the
    median of the segment's window scores.
    """

    columns = list(score_columns)
    grouping: dict[str, str] = {
        "split_label": "first",
        "anomaly_label": "first",
        "domain_shift_op": "first",
        "domain_shift_env": "first",
    }
    if "combined_label" in labels.columns:
        grouping = {"combined_label": "first", **grouping}
    for column in columns:
        grouping[column] = "median"
    return labels.groupby("segment_id").agg(grouping).reset_index()


def official_single_auc(frame: pd.DataFrame, anomaly_score_column: str) -> dict[str, float]:
    """Upstream calculate_single_auc: strict pairwise, exact ties score zero.

    Returns the three published slices in upstream order: all, source, target.
    """

    source_mask = frame["split_label"].isin(["Normal_Source_Test", "Anomaly_Source_Test"])
    target_mask = frame["split_label"].isin(["Normal_Target_Test", "Anomaly_Target_Test"])
    normal_mask = frame["anomaly_label"] == "normal"
    anomaly_mask = frame["anomaly_label"] != "normal"

    domains = {
        "all": (frame[normal_mask], frame[anomaly_mask]),
        "source": (frame[source_mask & normal_mask], frame[source_mask & anomaly_mask]),
        "target": (frame[target_mask & normal_mask], frame[target_mask & anomaly_mask]),
    }
    out: dict[str, float] = {}
    for name, (normal, anomaly) in domains.items():
        n = np.asarray(normal[anomaly_score_column], dtype=np.float64)
        a = np.asarray(anomaly[anomaly_score_column], dtype=np.float64)
        if n.size == 0 or a.size == 0:
            raise ValueError(f"AUC slice {name!r} lacks one of the two classes")
        # int(normal < anomaly): exact ties contribute zero, as upstream.
        out[name] = float(np.sum(n[:, None] < a[None, :]) / (n.size * a.size))
    return out


# ------------------------------------------------------------------- torch
@dataclass(frozen=True)
class TrainingOutcome:
    epochs_run: int
    stopped_early: bool
    best_valid_loss: float
    best_epoch: int
    training_losses: tuple[float, ...]
    valid_losses: tuple[float, ...]
    checkpoint_semantics: str = (
        "upstream: state_dict() reference, so the restored weights are the last "
        "epoch's, not the best-validation epoch's"
    )
    extra: dict = field(default_factory=dict)


def build_official_autoencoder(window_lengths, num_channels, params):
    """Upstream AutoencoderFC: Linear -> BatchNorm1d -> ReLU blocks, mirrored.

    torch is imported lazily so that the pure-numpy parts of this module remain
    importable in environments without a deep-learning framework.
    """

    import torch.nn as nn

    input_dim = int(sum(int(w) * int(c) for w, c in zip(window_lengths, num_channels,
                                                        strict=True)))
    dims = list(params["layer_dims"])

    encoder: list = []
    current = input_dim
    for dim in dims[:-1]:
        encoder += [nn.Linear(current, dim), nn.BatchNorm1d(dim), nn.ReLU()]
        current = dim
    encoder.append(nn.Linear(current, dims[-1]))

    decoder: list = []
    current = dims[-1]
    for dim in reversed(dims[:-1]):
        decoder += [nn.Linear(current, dim), nn.BatchNorm1d(dim), nn.ReLU()]
        current = dim
    decoder.append(nn.Linear(current, input_dim))

    class AutoencoderFC(nn.Module):
        """Faithful reproduction of the upstream fully connected autoencoder."""

        def __init__(self) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.encoder = nn.Sequential(*encoder)
            self.decoder = nn.Sequential(*decoder)

        def forward(self, x):
            x = x.view(-1, self.input_dim)
            encoded = self.encoder(x)
            return encoded, self.decoder(encoded)

    return AutoencoderFC()
