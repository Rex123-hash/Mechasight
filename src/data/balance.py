"""Label/condition balance analysis — the shortcut-risk audit (Q4, risk R6).

Question: does a segment's operating/environmental condition, on its own, predict
whether that segment is anomalous? If it does, any condition-aware model has a
shortcut to the answer that has nothing to do with the machine's health, and the
deferred tree-model recast is unsafe.

Metadata only. No signal content is read here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score

CONDITION_COLUMNS = ["domain_shift_op", "domain_shift_env"]


def _as_labels(series: pd.Series) -> np.ndarray:
    return series.astype("string").fillna("<NA>").to_numpy()


def mutual_information_with_permutation(
    condition: pd.Series,
    is_anomaly: pd.Series,
    n_permutations: int = 2000,
    seed: int = 0,
) -> dict:
    """Mutual information between a condition and the anomaly flag, with a null.

    Raw MI is biased upward on small samples and with many condition values, so
    a bare MI number is not interpretable. The permutation null says how much MI
    this many samples and this many categories would produce by chance alone;
    only an excess over that null is evidence of a real association.
    """
    cond = _as_labels(condition)
    target = is_anomaly.astype(bool).to_numpy()

    n = int(cond.size)
    if n == 0 or np.unique(cond).size < 2 or np.unique(target).size < 2:
        return {
            "n": n,
            "n_condition_values": int(np.unique(cond).size),
            "mutual_information_bits": float("nan"),
            "null_mean_bits": float("nan"),
            "null_p95_bits": float("nan"),
            "excess_over_null_bits": float("nan"),
            "p_value": float("nan"),
            "interpretable": False,
        }

    to_bits = 1.0 / np.log(2.0)
    observed = mutual_info_score(cond, target) * to_bits

    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=np.float64)
    shuffled = target.copy()
    for i in range(n_permutations):
        rng.shuffle(shuffled)
        null[i] = mutual_info_score(cond, shuffled) * to_bits

    # +1 correction: a permutation p-value can never legitimately be zero.
    p_value = float((np.count_nonzero(null >= observed) + 1) / (n_permutations + 1))

    return {
        "n": n,
        "n_condition_values": int(np.unique(cond).size),
        "mutual_information_bits": float(observed),
        "null_mean_bits": float(null.mean()),
        "null_p95_bits": float(np.percentile(null, 95)),
        "excess_over_null_bits": float(observed - null.mean()),
        "p_value": p_value,
        "interpretable": True,
    }


def condition_class_table(
    frame: pd.DataFrame, condition_column: str, extra_group: list[str] | None = None
) -> pd.DataFrame:
    """Counts of normal vs anomalous segments per condition value.

    `n_exclusive_class` flags condition values that occur in only one class --
    the clearest form of shortcut, because such a value alone determines the label.
    """
    group = (extra_group or []) + [condition_column]
    table = (
        frame.assign(_is_anomaly=frame["is_anomaly"].map({True: "anomaly", False: "normal"}))
        .groupby(group + ["_is_anomaly"], dropna=False)
        .size()
        .unstack("_is_anomaly", fill_value=0)
        .reset_index()
    )
    for column in ("normal", "anomaly"):
        if column not in table.columns:
            table[column] = 0
    table["total"] = table["normal"] + table["anomaly"]
    table["anomaly_fraction"] = np.where(
        table["total"] > 0, table["anomaly"] / table["total"], np.nan
    )
    table["class_exclusive"] = (table["normal"] == 0) | (table["anomaly"] == 0)
    table.insert(0, "condition_column", condition_column)
    return table


def analyse_balance(frame: pd.DataFrame, n_permutations: int = 2000) -> dict:
    """Full Q4 analysis for one machine's *test* metadata.

    Restricted to test rows by the caller: training data is normal-only, so a
    condition/label association is only meaningful within the test set.
    """
    results: dict = {"n_test_segments": int(len(frame)), "conditions": {}}

    present = [c for c in CONDITION_COLUMNS if c in frame.columns]
    results["condition_columns_present"] = present
    results["condition_columns_missing"] = [
        c for c in CONDITION_COLUMNS if c not in frame.columns
    ]

    for column in present:
        entry: dict = {
            "values": sorted(frame[column].astype("string").fillna("<NA>").unique().tolist()),
        }
        entry["overall"] = mutual_information_with_permutation(
            frame[column], frame["is_anomaly"], n_permutations=n_permutations
        )
        # Also within each domain: a condition can be uninformative overall yet
        # predictive inside one domain, which pooling would mask.
        by_domain = {}
        if "file_domain" in frame.columns:
            for domain, part in frame.groupby("file_domain", dropna=False):
                by_domain[str(domain)] = mutual_information_with_permutation(
                    part[column], part["is_anomaly"], n_permutations=n_permutations
                )
        entry["by_domain"] = by_domain
        results["conditions"][column] = entry

    # Joint condition (operational + environmental together).
    if len(present) > 1:
        joint = frame[present].astype("string").fillna("<NA>").agg("|".join, axis=1)
        results["joint_condition"] = mutual_information_with_permutation(
            joint, frame["is_anomaly"], n_permutations=n_permutations
        )

    return results
