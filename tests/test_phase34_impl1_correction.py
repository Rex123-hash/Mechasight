"""Phase 3.4 IMPL-1 — fold-specific anomaly reference in the concordance estimator.

Synthetic tests prove the matching rule structurally; read-only checks reproduce the
committed corrected artifacts from the persisted scores. No inference anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.experiment import focal_toggle_execution as fx
from src.experiment.focal_concordance import (ANOMALY_REFERENCE_SIZE, concordance_per_segment,
                                              parent_concordance_delta, parent_concordance_delta_by_fold)
from src.experiment.provenance import canonical_sha256

RESULTS = Path("reports/phase3_4_results")
OLD4, OLD5 = "6441ed66b88074417ee0772e2618af0842050d2533d48029bb07a831ec1384bd", "427b04cb96c97fe1dd218d515c1f0fb3378e65551d95c9a30d858f1347807188"


def _synthetic(folds=(1, 2, 3), parents=12, segs=3, seed=0, shift=0.3):
    """One row per (segment, arm, fold-model). Anomaly reference rows differ per fold model."""
    rng = np.random.default_rng(seed)
    rows = []
    for f in folds:
        for arm in ("PRESENT", "ABSENT"):
            offset = f * 10.0                      # fold models live on wildly different raw scales
            for j in range(ANOMALY_REFERENCE_SIZE):
                rows.append({"segment_id": f"anom{j}", "parent_id": f"ap{j}", "arm": arm, "anomaly_label": "anomaly",
                             "focal_fold": f, "total_loss": offset + rng.normal(3, 1)})
            for p in range(parents):
                for s in range(segs):
                    rows.append({"segment_id": f"f{f}p{p}s{s}", "parent_id": f"f{f}p{p}", "arm": arm,
                                 "anomaly_label": "normal", "focal_fold": f,
                                 "total_loss": offset + rng.normal(3, 1) - (shift if arm == "PRESENT" else 0)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------ structural proofs
def test_each_parent_reference_is_exactly_232_rows_from_its_own_fold_and_arm():
    result, audits = parent_concordance_delta_by_fold(_synthetic(), bootstrap=False)
    assert len(audits) == 6                                     # 3 folds x 2 arms
    for a in audits:
        assert a.n_reference == 232
        assert a.reference_folds == (a.fold,)
        assert a.reference_arms == (a.arm,)
    assert result.n_parents == 36


def test_reference_rows_from_another_fold_can_never_enter_a_parent_concordance():
    """Manual recomputation for one fold with ONLY that fold's own reference equals the estimator."""
    df = _synthetic()
    result, _ = parent_concordance_delta_by_fold(df, bootstrap=False)
    f = 2
    manual = {}
    for arm in ("PRESENT", "ABSENT"):
        sub = df[(df.focal_fold == f) & (df.arm == arm)]
        normal, anomaly = sub[sub.anomaly_label == "normal"], sub[sub.anomaly_label != "normal"]
        assert len(anomaly) == 232
        manual[arm] = pd.Series(concordance_per_segment(normal.total_loss, anomaly.total_loss), index=normal.segment_id.to_numpy())
    delta = (manual["PRESENT"] - manual["ABSENT"])
    parent_of = df.drop_duplicates("segment_id").set_index("segment_id").parent_id
    expected = delta.groupby(parent_of.reindex(delta.index).to_numpy()).mean()
    for p, v in expected.items():
        assert result.parent_deltas[p] == pytest.approx(v, abs=1e-15)


def test_changing_or_removing_another_folds_anomaly_rows_cannot_affect_a_folds_parent_deltas():
    df = _synthetic()
    base, _ = parent_concordance_delta_by_fold(df, bootstrap=False)
    fold2 = base.parent_deltas[[p for p in base.parent_deltas.index if p.startswith("f2")]]
    # (a) perturb fold-3's anomaly reference wildly
    pert = df.copy()
    m = (pert.focal_fold == 3) & (pert.anomaly_label != "normal")
    pert.loc[m, "total_loss"] = -1000.0                     # now BELOW every fold-2 normal score
    r_pert, _ = parent_concordance_delta_by_fold(pert, bootstrap=False)
    assert r_pert.parent_deltas[fold2.index].equals(fold2)
    # (b) remove fold 3 entirely
    r_drop, _ = parent_concordance_delta_by_fold(df[df.focal_fold != 3], bootstrap=False)
    assert r_drop.parent_deltas[fold2.index].equals(fold2)
    assert r_drop.fold_deltas[2] == base.fold_deltas[2]
    # (c) the historical pooled estimator does NOT have this property - the defect IMPL-1
    hist_base = parent_concordance_delta(df, bootstrap=False)
    hist_pert = parent_concordance_delta(pert, bootstrap=False)
    assert not hist_pert.parent_deltas[fold2.index].equals(hist_base.parent_deltas[fold2.index])


def test_wrong_reference_size_or_missing_fold_tag_is_refused():
    df = _synthetic()
    with pytest.raises(ValueError, match="expected exactly 232"):
        parent_concordance_delta_by_fold(df.drop(df[(df.focal_fold == 1) & (df.segment_id == "anom0") & (df.arm == "PRESENT")].index), bootstrap=False)
    bad = df.copy(); bad.loc[bad.index[0], "focal_fold"] = np.nan
    with pytest.raises(ValueError, match="carry the fold"):
        parent_concordance_delta_by_fold(bad, bootstrap=False)
    with pytest.raises(ValueError, match="missing columns"):
        parent_concordance_delta_by_fold(df.drop(columns=["focal_fold"]), bootstrap=False)


def test_corrected_estimator_equals_historical_when_only_one_fold_is_present():
    """With a single fold the pooled reference IS the fold's own reference: both agree exactly."""
    df = _synthetic(folds=(4,))
    a, _ = parent_concordance_delta_by_fold(df, replicates=500, seed=27182)
    b = parent_concordance_delta(df, replicates=500, seed=27182, stratify_by_fold=True)
    assert a.point_estimate == b.point_estimate and (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)
    assert a.parent_deltas.equals(b.parent_deltas)


def test_bootstrap_is_fold_stratified_reproducible_and_conditional():
    df = _synthetic()
    a, _ = parent_concordance_delta_by_fold(df, replicates=1000, seed=27182)
    b, _ = parent_concordance_delta_by_fold(df, replicates=1000, seed=27182)
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high) and a.bootstrap_stratified_by_fold is True
    assert a.interval_is_conditional_on_trained_models is True and a.bootstrap_seed == 27182
    assert a.point_estimate == pytest.approx(a.parent_deltas.mean(), abs=1e-15)   # equal-parent, no segment weighting


# ---------------------------------------------- real artifacts (read-only)
def _corrected(n):
    return json.loads((RESULTS / f"phase3_4_impl1_corrected_{n}fold_analysis.json").read_text(encoding="utf-8"))


def test_48_and_60_parent_computations_reproduce_independently_from_persisted_scores():
    a4, a5 = _corrected(4), _corrected(5)
    r4 = json.loads(json.dumps(fx.analyse_corrected([2, 3, 4, 5])))
    r5 = json.loads(json.dumps(fx.analyse_corrected([1, 2, 3, 4, 5])))
    assert r4 == a4 and r5 == a5
    assert a4["primary"]["n_focal_parents"] == 48 and a5["primary"]["n_focal_parents"] == 60
    for a in (a4, a5):
        for ref in a["reference_audit"]:
            assert ref["n_reference"] == 232 and ref["reference_folds"] == [ref["fold"]] and ref["reference_arms"] == [ref["arm"]]
    # folds 2-5 are identical between the two corrected analyses (fold-local by construction)
    for k in ("2", "3", "4", "5"):
        assert a4["primary"]["fold_level_deltas_descriptive"][k] == a5["primary"]["fold_level_deltas_descriptive"][k]
    assert all(a5["parent_deltas"][k] == v for k, v in a4["parent_deltas"].items())
    for a in (a4, a5):
        assert canonical_sha256(RESULTS / f"phase3_4_impl1_corrected_{a['primary']['n_focal_parents'] // 12}fold_analysis.json") == \
            (RESULTS / f"phase3_4_impl1_corrected_{a['primary']['n_focal_parents'] // 12}fold_analysis.sha256").read_text(encoding="utf-8").strip()


def test_historical_artifacts_are_retained_and_secondary_metrics_were_unaffected():
    assert canonical_sha256(RESULTS / "phase3_4_inc1_salvage_analysis.json") == OLD4
    assert canonical_sha256(RESULTS / "phase3_4_five_fold_recovered_analysis.json") == OLD5
    old4 = json.loads((RESULTS / "phase3_4_inc1_salvage_analysis.json").read_text(encoding="utf-8"))
    old5 = json.loads((RESULTS / "phase3_4_five_fold_recovered_analysis.json").read_text(encoding="utf-8"))
    a4, a5 = _corrected(4), _corrected(5)
    for key in ("S1_segment_weighted_focal_auroc_delta_by_fold",
                "S4_raw_focal_normal_score_shift_by_fold_SCALE_DEPENDENT",
                "S5_anomaly_reference_raw_shift_by_fold_specificity"):
        assert old4["secondary"][key] == a4["secondary"][key]
        assert old5["secondary"][key] == a5["secondary"][key]
    # the primary WAS affected: historical fold-2..5 deltas differ between old 4- and 5-fold
    assert any(old4["salvage"]["fold_level_deltas_descriptive"][k] != old5["primary"]["fold_level_deltas_descriptive"][k]
               for k in ("2", "3", "4", "5"))
