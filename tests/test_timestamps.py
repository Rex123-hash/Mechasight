"""Tests for src/data/timestamps.py.

These check that the measurement code behaves correctly on synthetic streams
whose properties are known by construction. None of them assert a constant taken
from the IMAD-DS paper -- the point is to catch wrong data handling, not to
re-state published claims.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.timestamps import (
    analyse_alignment,
    analyse_timestamps,
    discarded_fraction,
    rate_drift,
)


def _uniform(rate: float, n: int, t0: float = 0.0) -> np.ndarray:
    return t0 + np.arange(n, dtype=np.float64) / rate


class TestRateEstimation:
    def test_recovers_known_rate_from_uniform_stream(self):
        stats = analyse_timestamps(_uniform(7063.0, 5000))
        assert stats.n_samples == 5000
        for estimate in (
            stats.rate_from_median_dt,
            stats.rate_from_mean_dt,
            stats.rate_from_span,
        ):
            assert estimate == pytest.approx(7063.0, rel=1e-9)

    def test_span_and_median_estimators_diverge_when_a_gap_exists(self):
        """A dropped block lowers the average rate but not the instantaneous one.

        This is the whole reason both estimators are reported: if the dataset has
        dropped samples, quoting only one of them would misdescribe it.
        """
        t = np.concatenate([_uniform(1000.0, 1000), _uniform(1000.0, 1000, t0=2.0)])
        stats = analyse_timestamps(t)

        assert stats.rate_from_median_dt == pytest.approx(1000.0, rel=1e-9)
        assert stats.rate_from_span < 900.0
        assert stats.n_gaps == 1
        # ~1 s of samples are missing (the 1 s hole minus one nominal interval).
        assert stats.total_gap_s == pytest.approx(1.0, abs=1e-3)

    def test_epoch_offset_degrades_per_sample_estimators_and_is_flagged(self):
        """float64 unix-epoch timestamps quantise dt at kHz rates.

        One ULP at ~1.7e9 s is ~2.4e-7 s, i.e. 0.38% of a 16 kHz sample interval,
        so median/mean-dt rate estimates acquire a bias that no amount of
        averaging removes. This is a property of the stored values, not of the
        analyser, and it must be surfaced rather than hidden -- otherwise an
        estimated sampling rate would be quoted with false confidence.
        """
        plain = analyse_timestamps(_uniform(16000.0, 4000))
        shifted = analyse_timestamps(_uniform(16000.0, 4000, t0=1.7e9))

        assert not plain.precision_limited
        assert shifted.precision_limited
        assert shifted.ulp_over_dt > 1e-3

        # The per-sample estimator drifts off the true rate ...
        assert abs(shifted.rate_from_median_dt - 16000.0) > 1.0
        # ... while the span-based estimator stays accurate, because it divides
        # by a long elapsed time instead of a single interval.
        assert shifted.rate_from_span == pytest.approx(16000.0, rel=1e-5)

    def test_quantised_timestamps_bias_per_sample_rate_but_not_span_rate(self):
        """Storing timestamps on a coarse grid snaps dt and biases 1/median(dt).

        Reproduces the mechanism observed in the real release: a true rate whose
        interval is not a whole number of grid steps makes every dt round to a
        neighbouring grid value, so the per-sample estimator reports the grid
        rate rather than the true one. The span estimator is unaffected because
        it divides by a long elapsed time.
        """
        true_rate = 16003.076
        t = np.round(np.arange(83200) / true_rate, 6)  # 1 us grid
        stats = analyse_timestamps(t)

        assert stats.quantised
        assert stats.n_unique_dt <= 3
        assert stats.timestamp_resolution_s == pytest.approx(1e-6, rel=1e-3)
        assert stats.rate_from_span == pytest.approx(true_rate, rel=1e-6)
        # 1/62us = 16129 Hz, well away from the true rate.
        assert stats.rate_from_median_dt > true_rate + 100
        assert stats.quantisation_rate_bias_pct > 0.5

    def test_bimodal_cadence_is_reported_as_irregular(self):
        """A stream with two interleaved intervals is not uniformly sampled.

        Mirrors the IMU behaviour found in the release: most intervals sit at
        one value while a minority sit at a clearly shorter one. Quoting a
        single rate for such a stream hides the irregularity, so it must be
        measured separately.
        """
        rng = np.random.default_rng(2)
        intervals = np.where(rng.random(20000) < 0.03, 122.5e-6, 140.6e-6)
        t = np.concatenate([[0.0], np.cumsum(intervals)])
        stats = analyse_timestamps(t)

        assert stats.frac_dt_irregular == pytest.approx(0.03, abs=0.01)
        assert stats.dt_ratio_p99_p01 > 1.1
        assert stats.n_gaps == 0  # shorter intervals are not gaps

    def test_uniform_stream_has_no_irregularity(self):
        stats = analyse_timestamps(_uniform(7135.0, 20000))
        assert stats.frac_dt_irregular == pytest.approx(0.0, abs=1e-9)
        assert stats.dt_ratio_p99_p01 == pytest.approx(1.0, rel=1e-6)

    def test_unquantised_stream_is_not_flagged(self):
        rng = np.random.default_rng(1)
        t = np.sort(_uniform(7000.0, 20000) + rng.normal(0, 1e-7, 20000))
        stats = analyse_timestamps(t)
        assert not stats.quantised
        assert stats.n_unique_dt > 100

    def test_relative_timestamps_are_not_precision_limited(self):
        stats = analyse_timestamps(_uniform(7063.0, 5000))
        assert not stats.precision_limited
        assert stats.rate_from_median_dt == pytest.approx(7063.0, rel=1e-9)

    def test_jitter_widens_dt_spread_but_keeps_median_rate(self):
        rng = np.random.default_rng(0)
        t = np.sort(_uniform(1000.0, 5000) + rng.normal(0, 1e-5, 5000))
        stats = analyse_timestamps(t)
        assert stats.rate_from_median_dt == pytest.approx(1000.0, rel=0.05)
        assert stats.dt_std > 0


class TestDegenerateInput:
    def test_empty_stream_is_reported_not_crashed(self):
        stats = analyse_timestamps(np.array([]))
        assert stats.n_samples == 0
        assert not stats.is_strictly_increasing
        assert np.isnan(stats.rate_from_median_dt)

    def test_single_sample_stream(self):
        stats = analyse_timestamps(np.array([5.0]))
        assert stats.n_samples == 1
        assert stats.span_s == 0.0
        assert np.isnan(stats.rate_from_span)

    def test_constant_timestamps_do_not_produce_infinite_rate(self):
        stats = analyse_timestamps(np.zeros(100))
        assert stats.n_duplicate_timestamps == 99
        assert not stats.is_strictly_increasing
        assert np.isnan(stats.rate_from_median_dt)


class TestMonotonicity:
    def test_duplicates_counted(self):
        t = np.array([0.0, 1.0, 1.0, 2.0, 3.0, 3.0])
        stats = analyse_timestamps(t)
        assert stats.n_duplicate_timestamps == 2
        assert stats.n_nonmonotonic == 0
        assert not stats.is_strictly_increasing

    def test_backwards_step_counted_separately_from_duplicates(self):
        t = np.array([0.0, 1.0, 0.5, 2.0])
        stats = analyse_timestamps(t)
        assert stats.n_nonmonotonic == 1
        assert stats.n_duplicate_timestamps == 0

    def test_unsorted_input_is_not_silently_sorted(self):
        """Arrival order is itself a finding; the analyser must not hide it."""
        ordered = _uniform(100.0, 50)
        shuffled = ordered[::-1]
        assert analyse_timestamps(shuffled).n_nonmonotonic == 49


class TestDrift:
    def test_constant_rate_gives_ratio_of_one(self):
        drift = rate_drift(_uniform(6667.0, 8000), n_chunks=4)
        assert drift["drift_n_chunks"] == 4
        assert drift["drift_rate_ratio"] == pytest.approx(1.0, rel=1e-6)

    def test_accelerating_clock_is_detected(self):
        slow = _uniform(1000.0, 2000)
        fast = _uniform(2000.0, 2000, t0=slow[-1] + 1e-3)
        drift = rate_drift(np.concatenate([slow, fast]), n_chunks=4)
        assert drift["drift_rate_ratio"] > 1.5

    def test_too_short_stream_reports_no_chunks_rather_than_failing(self):
        assert rate_drift(np.array([0.0, 1.0]), n_chunks=4)["drift_n_chunks"] == 0


class TestAlignment:
    def test_identical_spans_give_full_coverage(self):
        stats = analyse_alignment({"mic": (0.0, 10.0), "acc": (0.0, 10.0)})
        assert stats.intersection_s == pytest.approx(10.0)
        assert stats.coverage_ratio == pytest.approx(1.0)
        assert stats.start_spread_s == 0.0

    def test_staggered_starts_reduce_intersection(self):
        stats = analyse_alignment(
            {"mic": (0.0, 10.0), "acc": (0.5, 10.0), "gyro": (0.2, 9.7)}
        )
        assert stats.start_spread_s == pytest.approx(0.5)
        assert stats.end_spread_s == pytest.approx(0.3)
        assert stats.intersection_s == pytest.approx(9.2)
        assert stats.union_s == pytest.approx(10.0)

    def test_non_overlapping_sensors_give_negative_intersection(self):
        """Reported as negative, not clamped -- the magnitude is diagnostic."""
        stats = analyse_alignment({"mic": (0.0, 1.0), "acc": (5.0, 6.0)})
        assert stats.intersection_s < 0

    def test_empty_input_does_not_crash(self):
        assert np.isnan(analyse_alignment({}).intersection_s)


class TestDiscardedFraction:
    def test_nothing_discarded_when_keep_covers_span(self):
        assert discarded_fraction((0.0, 10.0), (0.0, 10.0)) == pytest.approx(0.0)

    def test_head_crop_fraction(self):
        # Cropping the first 35 ms of a 10 s stream loses 0.35% of it.
        assert discarded_fraction((0.0, 10.0), (0.035, 10.0)) == pytest.approx(0.0035)

    def test_disjoint_keep_discards_everything(self):
        assert discarded_fraction((0.0, 1.0), (5.0, 6.0)) == pytest.approx(1.0)

    def test_zero_length_span_is_nan_not_division_error(self):
        assert np.isnan(discarded_fraction((3.0, 3.0), (0.0, 10.0)))
