"""Timestamp analysis for multi-rate sensor recordings.

Pure functions over timestamp arrays. Nothing here knows about IMAD-DS's folder
layout, so every function is testable on synthetic input.

Phase 1 scope: these functions *measure*. They do not resample, filter, or
correct anything. Any irregularity found is reported, never silently repaired.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

# A dt more than this multiple of the median dt is treated as a gap (a likely
# dropped-sample event) rather than ordinary jitter. Reporting threshold only --
# it has no effect on any data.
GAP_FACTOR = 3.0

# If one float64 ULP at the timestamps' magnitude exceeds this fraction of the
# median sample interval, per-sample dt values are quantisation-limited and the
# median/mean rate estimators become biased. Absolute (unix-epoch) timestamps at
# kHz sampling rates hit this: one ULP at ~1.7e9 s is ~2.4e-7 s, which is 0.38%
# of a 16 kHz sample interval. The span-based estimator is unaffected, because it
# divides by a large elapsed time rather than by a single interval.
PRECISION_WARN_RATIO = 1e-3

# If the per-sample intervals take fewer than this many distinct values, the
# stored timestamps are quantised to a coarse grid rather than being continuous.
# Quantisation biases 1/median(dt) and 1/mean(dt) towards the grid, while the
# span estimator (n-1)/elapsed stays accurate. Detecting it matters: without it,
# a quantisation artefact would be mistaken for the true sampling rate.
QUANTISATION_UNIQUE_DT_WARN = 20

# An interval this far from the median is "irregular" -- beyond what 1 us
# timestamp rounding can explain at these rates, so it indicates a genuine
# second cadence (e.g. FIFO batching) rather than a display artefact.
IRREGULAR_DT_TOLERANCE = 0.05


@dataclass(frozen=True)
class RateStats:
    """Timing statistics for one sensor stream within one segment."""

    n_samples: int
    t_start: float
    t_end: float
    span_s: float
    # Two independent rate estimators. They disagree when the stream has gaps:
    # the median-based one describes the instantaneous rate, the span-based one
    # the average rate including any stalls.
    rate_from_median_dt: float
    rate_from_mean_dt: float
    rate_from_span: float
    dt_mean: float
    dt_median: float
    dt_std: float
    dt_min: float
    dt_max: float
    dt_p01: float
    dt_p99: float
    n_duplicate_timestamps: int
    n_nonmonotonic: int
    is_strictly_increasing: bool
    n_gaps: int
    max_gap_s: float
    total_gap_s: float
    # Numerical trustworthiness of the per-sample estimators above.
    ulp_s: float
    ulp_over_dt: float
    precision_limited: bool
    # Storage quantisation of the timestamp grid (distinct from float64 ULP).
    n_unique_dt: int
    timestamp_resolution_s: float
    quantised: bool
    quantisation_rate_bias_pct: float
    # Sampling regularity: what fraction of intervals sit far enough from the
    # median that timestamp rounding cannot explain them.
    frac_dt_irregular: float
    dt_ratio_p99_p01: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def analyse_timestamps(t: np.ndarray, gap_factor: float = GAP_FACTOR) -> RateStats:
    """Measure timing properties of a 1-D array of timestamps in seconds.

    The array is analysed as given -- it is NOT sorted first, because whether it
    arrives sorted is itself a finding worth reporting.
    """
    t = np.asarray(t, dtype=np.float64).ravel()
    n = int(t.size)

    if n == 0:
        nan = float("nan")
        return RateStats(
            n_samples=0, t_start=nan, t_end=nan, span_s=nan,
            rate_from_median_dt=nan, rate_from_mean_dt=nan, rate_from_span=nan,
            dt_mean=nan, dt_median=nan, dt_std=nan, dt_min=nan, dt_max=nan,
            dt_p01=nan, dt_p99=nan,
            n_duplicate_timestamps=0, n_nonmonotonic=0,
            is_strictly_increasing=False,
            n_gaps=0, max_gap_s=nan, total_gap_s=0.0,
            ulp_s=nan, ulp_over_dt=nan, precision_limited=False,
            n_unique_dt=0, timestamp_resolution_s=nan, quantised=False,
            quantisation_rate_bias_pct=nan, frac_dt_irregular=nan,
            dt_ratio_p99_p01=nan,
        )

    t_start, t_end = float(t[0]), float(t[-1])
    span = t_end - t_start

    if n == 1:
        nan = float("nan")
        return RateStats(
            n_samples=1, t_start=t_start, t_end=t_end, span_s=0.0,
            rate_from_median_dt=nan, rate_from_mean_dt=nan, rate_from_span=nan,
            dt_mean=nan, dt_median=nan, dt_std=nan, dt_min=nan, dt_max=nan,
            dt_p01=nan, dt_p99=nan,
            n_duplicate_timestamps=0, n_nonmonotonic=0,
            is_strictly_increasing=True,
            n_gaps=0, max_gap_s=nan, total_gap_s=0.0,
            ulp_s=float(np.spacing(abs(t_start))) if t_start else 0.0,
            ulp_over_dt=nan, precision_limited=False,
            n_unique_dt=0, timestamp_resolution_s=nan, quantised=False,
            quantisation_rate_bias_pct=nan, frac_dt_irregular=nan,
            dt_ratio_p99_p01=nan,
        )

    # Diff relative to the first sample. This does not recover precision already
    # lost when the values were stored, but it avoids adding more of it here.
    dt = np.diff(t - t[0])
    dt_median = float(np.median(dt))
    dt_mean = float(np.mean(dt))

    n_duplicates = int(np.count_nonzero(dt == 0.0))
    n_nonmonotonic = int(np.count_nonzero(dt < 0.0))

    # Gaps are judged against the median dt, which is robust to the gaps
    # themselves. Only meaningful when the median dt is positive.
    if dt_median > 0:
        gap_mask = dt > gap_factor * dt_median
        n_gaps = int(np.count_nonzero(gap_mask))
        # Excess time beyond one nominal sample interval, i.e. the time actually
        # "missing" rather than the raw gap length.
        total_gap = float(np.sum(dt[gap_mask] - dt_median)) if n_gaps else 0.0
    else:
        n_gaps, total_gap = 0, 0.0

    ulp = float(np.spacing(max(abs(t_start), abs(t_end)))) if (t_start or t_end) else 0.0
    ulp_over_dt = _safe_div(ulp, dt_median) if dt_median > 0 else float("nan")

    # Quantisation: how many distinct dt values exist, and what grid spacing do
    # they imply? Rounded to 1e-12 s so float noise does not inflate the count.
    unique_dt = np.unique(np.round(dt, 12))
    n_unique_dt = int(unique_dt.size)
    if unique_dt.size >= 2:
        steps = np.diff(unique_dt)
        steps = steps[steps > 0]
        resolution = float(np.min(steps)) if steps.size else float("nan")
    else:
        resolution = float("nan")
    quantised = bool(n_unique_dt <= QUANTISATION_UNIQUE_DT_WARN)

    # How far the per-sample estimator is pulled from the span estimator by that
    # quantisation. The span estimator is the reference because it divides by a
    # long elapsed time and is therefore insensitive to the grid.
    if dt_median > 0:
        frac_irregular = float(
            np.count_nonzero(np.abs(dt - dt_median) > IRREGULAR_DT_TOLERANCE * dt_median)
            / dt.size
        )
    else:
        frac_irregular = float("nan")
    dt_p01_value, dt_p99_value = float(np.percentile(dt, 1)), float(np.percentile(dt, 99))

    rate_span = _safe_div(n - 1, span)
    rate_median = _safe_div(1.0, dt_median)
    bias_pct = (
        float(100.0 * (rate_median - rate_span) / rate_span)
        if np.isfinite(rate_span) and rate_span
        else float("nan")
    )

    return RateStats(
        n_samples=n,
        t_start=t_start,
        t_end=t_end,
        span_s=float(span),
        rate_from_median_dt=_safe_div(1.0, dt_median),
        rate_from_mean_dt=_safe_div(1.0, dt_mean),
        rate_from_span=_safe_div(n - 1, span),
        dt_mean=dt_mean,
        dt_median=dt_median,
        dt_std=float(np.std(dt)),
        dt_min=float(np.min(dt)),
        dt_max=float(np.max(dt)),
        dt_p01=float(np.percentile(dt, 1)),
        dt_p99=float(np.percentile(dt, 99)),
        n_duplicate_timestamps=n_duplicates,
        n_nonmonotonic=n_nonmonotonic,
        is_strictly_increasing=bool(np.all(dt > 0)),
        n_gaps=n_gaps,
        max_gap_s=float(np.max(dt)),
        total_gap_s=total_gap,
        ulp_s=ulp,
        ulp_over_dt=ulp_over_dt,
        precision_limited=bool(
            np.isfinite(ulp_over_dt) and ulp_over_dt > PRECISION_WARN_RATIO
        ),
        n_unique_dt=n_unique_dt,
        timestamp_resolution_s=resolution,
        quantised=quantised,
        quantisation_rate_bias_pct=bias_pct,
        frac_dt_irregular=frac_irregular,
        dt_ratio_p99_p01=_safe_div(dt_p99_value, dt_p01_value),
    )


def rate_drift(t: np.ndarray, n_chunks: int = 4) -> dict[str, float]:
    """Split a stream into equal-sample chunks and compare their rates.

    Detects within-recording clock drift, which a single whole-file rate
    estimate would average away.
    """
    t = np.asarray(t, dtype=np.float64).ravel()
    if t.size < n_chunks * 2:
        return {"drift_n_chunks": 0, "drift_rate_min": float("nan"),
                "drift_rate_max": float("nan"), "drift_rate_ratio": float("nan")}

    rates = []
    for chunk in np.array_split(t, n_chunks):
        if chunk.size >= 2:
            span = float(chunk[-1] - chunk[0])
            if span > 0:
                rates.append((chunk.size - 1) / span)

    if not rates:
        return {"drift_n_chunks": 0, "drift_rate_min": float("nan"),
                "drift_rate_max": float("nan"), "drift_rate_ratio": float("nan")}

    lo, hi = float(min(rates)), float(max(rates))
    return {
        "drift_n_chunks": len(rates),
        "drift_rate_min": lo,
        "drift_rate_max": hi,
        "drift_rate_ratio": _safe_div(hi, lo),
    }


@dataclass(frozen=True)
class AlignmentStats:
    """How well several sensor streams overlap in time within one segment."""

    start_spread_s: float          # max(start) - min(start)
    end_spread_s: float            # max(end) - min(end)
    intersection_start: float
    intersection_end: float
    intersection_s: float
    union_s: float
    coverage_ratio: float          # intersection / union; 1.0 == perfect overlap

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyse_alignment(spans: dict[str, tuple[float, float]]) -> AlignmentStats:
    """Compare (start, end) pairs across sensors of one segment.

    `spans` maps sensor name -> (t_start, t_end). A negative `intersection_s`
    means the sensors do not overlap at all, which is reported as-is rather than
    clamped to zero, because the magnitude is diagnostic.
    """
    if not spans:
        nan = float("nan")
        return AlignmentStats(nan, nan, nan, nan, nan, nan, nan)

    starts = [s for s, _ in spans.values()]
    ends = [e for _, e in spans.values()]

    inter_start, inter_end = max(starts), min(ends)
    union_start, union_end = min(starts), max(ends)
    intersection = inter_end - inter_start
    union = union_end - union_start

    return AlignmentStats(
        start_spread_s=float(max(starts) - min(starts)),
        end_spread_s=float(max(ends) - min(ends)),
        intersection_start=float(inter_start),
        intersection_end=float(inter_end),
        intersection_s=float(intersection),
        union_s=float(union),
        coverage_ratio=_safe_div(intersection, union),
    )


def discarded_fraction(
    span: tuple[float, float], keep: tuple[float, float]
) -> float:
    """Fraction of one sensor's recorded duration lost by cropping to `keep`.

    Used to quantify the cost of the reference pipeline's
    max(start)/min(end) intersection plus gyroscope warm-up cut.
    """
    t0, t1 = span
    k0, k1 = keep
    total = t1 - t0
    if total <= 0:
        return float("nan")
    kept = max(0.0, min(t1, k1) - max(t0, k0))
    return float(1.0 - kept / total)
