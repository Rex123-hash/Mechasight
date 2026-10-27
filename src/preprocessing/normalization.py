"""Leakage-safe deterministic and train-fitted normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


EPSILON = 1e-5


def standardize_window(values: np.ndarray, epsilon: float = EPSILON) -> np.ndarray:
    """Authors' per-window, per-channel std_window operation.

    Input is (samples, channels); population standard deviation (ddof=0) is
    used exactly as in the pinned upstream implementation.
    """

    array = np.asarray(values, dtype=np.float64)
    mean = array.mean(axis=0, keepdims=True)
    std = array.std(axis=0, ddof=0, keepdims=True)
    return (array - mean) / (std + epsilon)


@dataclass(frozen=True)
class FittedChannelStandardizer:
    mean: np.ndarray
    std: np.ndarray
    n_samples: int
    fit_split: str
    fit_segment_ids: tuple[str, ...]
    epsilon: float = EPSILON

    @classmethod
    def fit(
        cls,
        batches: Iterable[np.ndarray],
        *,
        split: str,
        segment_ids: Iterable[str],
        epsilon: float = EPSILON,
    ) -> "FittedChannelStandardizer":
        if split != "train":
            raise ValueError("learned normalization statistics may be fit on train only")
        total = None
        total_square = None
        count = 0
        for batch in batches:
            values = np.asarray(batch, dtype=np.float64)
            if values.ndim != 2:
                raise ValueError("normalizer batches must be (samples, channels)")
            batch_sum = values.sum(axis=0)
            batch_square = np.square(values).sum(axis=0)
            total = batch_sum if total is None else total + batch_sum
            total_square = batch_square if total_square is None else total_square + batch_square
            count += len(values)
        if count == 0 or total is None or total_square is None:
            raise ValueError("cannot fit a scaler on no samples")
        mean = total / count
        variance = np.maximum(total_square / count - np.square(mean), 0.0)
        return cls(
            mean=mean,
            std=np.sqrt(variance),
            n_samples=count,
            fit_split=split,
            fit_segment_ids=tuple(segment_ids),
            epsilon=epsilon,
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != self.mean.size:
            raise ValueError("transform channels do not match fitted statistics")
        return (array - self.mean) / (self.std + self.epsilon)
