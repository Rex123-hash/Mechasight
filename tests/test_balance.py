"""Tests for the condition/label shortcut audit (Q4 / risk R6).

The permutation null is the point of this module: raw mutual information is
biased upward by sample size and category count, so a bare MI value cannot
distinguish a real association from chance. These tests check that the detector
fires on a planted association and stays quiet on an independent one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.balance import (
    analyse_balance,
    condition_class_table,
    mutual_information_with_permutation,
)


def _independent(n: int = 300, n_values: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    condition = pd.Series(rng.integers(0, n_values, n)).map(lambda v: f"c{v}")
    is_anomaly = pd.Series(rng.random(n) < 0.5)
    return condition, is_anomaly


class TestMutualInformation:
    def test_independent_condition_shows_no_excess_over_null(self):
        condition, is_anomaly = _independent()
        result = mutual_information_with_permutation(condition, is_anomaly,
                                                     n_permutations=500)
        assert result["interpretable"]
        # Raw MI is non-zero purely from finite-sample bias; the null absorbs it.
        assert result["mutual_information_bits"] > 0
        assert result["excess_over_null_bits"] < 0.02
        assert result["p_value"] > 0.05

    def test_perfectly_predictive_condition_is_detected(self):
        """A condition value that determines the class must be caught."""
        condition = pd.Series(["a"] * 150 + ["b"] * 150)
        is_anomaly = pd.Series([False] * 150 + [True] * 150)
        result = mutual_information_with_permutation(condition, is_anomaly,
                                                     n_permutations=500)
        # One full bit: knowing the condition removes all uncertainty.
        assert result["mutual_information_bits"] > 0.99
        assert result["excess_over_null_bits"] > 0.9
        assert result["p_value"] < 0.01

    def test_partial_association_is_detected(self):
        rng = np.random.default_rng(3)
        condition = pd.Series(["a"] * 200 + ["b"] * 200)
        probability = np.where(condition == "a", 0.75, 0.25)
        is_anomaly = pd.Series(rng.random(400) < probability)
        result = mutual_information_with_permutation(condition, is_anomaly,
                                                     n_permutations=500)
        assert result["excess_over_null_bits"] > 0.05
        assert result["p_value"] < 0.01

    def test_raw_mi_alone_would_mislead_with_many_categories(self):
        """Why the null matters: many categories inflate MI even under independence.

        With one sample per category, raw MI is maximal, yet nothing is learned.
        A rule based on raw MI alone would call this a shortcut.
        """
        condition = pd.Series([f"c{i}" for i in range(60)])
        rng = np.random.default_rng(5)
        is_anomaly = pd.Series(rng.random(60) < 0.5)
        result = mutual_information_with_permutation(condition, is_anomaly,
                                                     n_permutations=500)
        assert result["mutual_information_bits"] > 0.9      # looks alarming
        assert result["excess_over_null_bits"] < 0.15       # but the null matches it
        assert result["p_value"] > 0.05                     # so: not significant

    def test_permutation_p_value_is_never_zero(self):
        condition = pd.Series(["a"] * 100 + ["b"] * 100)
        is_anomaly = pd.Series([False] * 100 + [True] * 100)
        result = mutual_information_with_permutation(condition, is_anomaly,
                                                     n_permutations=100)
        assert result["p_value"] > 0

    def test_single_valued_condition_is_marked_uninterpretable(self):
        result = mutual_information_with_permutation(
            pd.Series(["a"] * 50), pd.Series([True, False] * 25), n_permutations=50
        )
        assert result["interpretable"] is False
        assert np.isnan(result["mutual_information_bits"])

    def test_single_class_target_is_marked_uninterpretable(self):
        result = mutual_information_with_permutation(
            pd.Series(["a", "b"] * 25), pd.Series([True] * 50), n_permutations=50
        )
        assert result["interpretable"] is False


class TestConditionClassTable:
    def test_flags_class_exclusive_condition_values(self):
        frame = pd.DataFrame({
            "machine": "M",
            "file_domain": "source",
            "cond": ["a"] * 10 + ["b"] * 10,
            "is_anomaly": [False] * 10 + [True] * 10,
        })
        table = condition_class_table(frame, "cond", extra_group=["machine", "file_domain"])
        assert table["class_exclusive"].all()
        assert table["anomaly_fraction"].tolist() == [0.0, 1.0]

    def test_balanced_condition_is_not_flagged(self):
        # Each condition value must contain BOTH classes, so the pairing has to
        # cycle at different periods -- ["a","b"]*20 against [True,False]*20
        # would make "a" always True, i.e. class-exclusive after all.
        frame = pd.DataFrame({
            "machine": "M",
            "file_domain": "source",
            "cond": ["a", "a", "b", "b"] * 10,
            "is_anomaly": [True, False] * 20,
        })
        table = condition_class_table(frame, "cond", extra_group=["machine", "file_domain"])
        assert not table["class_exclusive"].any()
        assert table["anomaly_fraction"].tolist() == [0.5, 0.5]

    def test_counts_sum_to_input_size(self):
        frame = pd.DataFrame({
            "machine": "M", "file_domain": "source",
            "cond": ["a", "b", "c"] * 10,
            "is_anomaly": ([True] * 15 + [False] * 15),
        })
        table = condition_class_table(frame, "cond", extra_group=["machine", "file_domain"])
        assert int(table["total"].sum()) == len(frame)


class TestAnalyseBalance:
    def test_reports_missing_condition_columns_rather_than_failing(self):
        frame = pd.DataFrame({"is_anomaly": [True, False] * 10, "file_domain": "source"})
        result = analyse_balance(frame, n_permutations=50)
        assert set(result["condition_columns_missing"]) == {
            "domain_shift_op", "domain_shift_env"
        }
        assert result["conditions"] == {}

    def test_runs_per_domain_as_well_as_overall(self):
        rng = np.random.default_rng(7)
        n = 200
        frame = pd.DataFrame({
            "domain_shift_op": rng.choice(["s1", "s2", "s3"], n),
            "domain_shift_env": rng.choice(["e1", "e2"], n),
            "file_domain": rng.choice(["source", "target"], n),
            "is_anomaly": rng.random(n) < 0.5,
        })
        result = analyse_balance(frame, n_permutations=200)
        entry = result["conditions"]["domain_shift_op"]
        assert set(entry["by_domain"]) == {"source", "target"}
        assert "joint_condition" in result
