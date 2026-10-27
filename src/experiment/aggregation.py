"""Frozen median window-to-segment anomaly-score aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SegmentScore:
    segment_id: str
    parent_recording_id: str
    segment_score: float
    aggregation: str
    contributing_window_ids: tuple[str, ...]
    n_windows: int


def aggregate_segment_windows(
    *,
    segment_ids: list[str] | tuple[str, ...],
    parent_recording_ids: list[str] | tuple[str, ...],
    window_ids: list[str] | tuple[str, ...],
    window_scores: np.ndarray,
) -> SegmentScore:
    scores = np.asarray(window_scores, dtype=np.float64).reshape(-1)
    lengths = {len(segment_ids), len(parent_recording_ids), len(window_ids), len(scores)}
    if lengths != {len(scores)} or len(scores) == 0:
        raise ValueError("one non-empty score and identity is required per window")
    if any(not str(window_id) for window_id in window_ids):
        raise ValueError("window ids cannot be missing")
    if len(set(window_ids)) != len(window_ids):
        raise ValueError("duplicate window ids are forbidden")
    if len(set(segment_ids)) != 1:
        raise ValueError("mixed segment ids cannot be aggregated")
    if len(set(parent_recording_ids)) != 1:
        raise ValueError("mixed parent recordings cannot be aggregated")
    if not np.all(np.isfinite(scores)):
        raise ValueError("window scores must be finite")
    return SegmentScore(
        segment_id=str(segment_ids[0]),
        parent_recording_id=str(parent_recording_ids[0]),
        segment_score=float(np.median(scores)),
        aggregation="median",
        contributing_window_ids=tuple(sorted(map(str, window_ids))),
        n_windows=len(scores),
    )
