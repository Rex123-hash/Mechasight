"""Phase 3.4 primary estimator: parent-equal-weighted focal concordance delta.

Frozen by ``configs/phase3/roboticarm_focal_toggle_execution_v1.json``
(``p34c-exec-0eecc19f81e200f5c301ca25``). This module performs no model
inference: it consumes already-persisted per-segment scores.

The per-segment quantity is a segment's individual contribution to
normal-versus-anomaly concordance against a fixed anomaly reference scored by
the same arm::

    C_i(a) = (1 / N_anomaly) * sum_j  I( score_i(a) < score_anomaly_j(a) )

Exact ties score zero, matching the upstream AUC convention. Because ``C_i``
depends only on the rank of segment ``i`` within its own arm, it is invariant
under any strictly increasing transform of that arm's scores, which is what
makes it comparable across two independently trained models.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BOOTSTRAP_SEED = 27182
BOOTSTRAP_REPLICATES = 10_000


def concordance_per_segment(normal_scores, anomaly_scores) -> np.ndarray:
    """C_i for each normal segment against the fixed anomaly reference.

    Both arrays must come from the SAME model arm.
    """
    normal = np.asarray(normal_scores, dtype=np.float64)
    anomaly = np.asarray(anomaly_scores, dtype=np.float64)
    if normal.ndim != 1 or anomaly.ndim != 1:
        raise ValueError("expected one-dimensional score arrays")
    if normal.size == 0 or anomaly.size == 0:
        raise ValueError("both classes must be non-empty")
    # int(normal < anomaly): exact ties contribute zero, as upstream does.
    return (normal[:, None] < anomaly[None, :]).sum(axis=1) / anomaly.size


def segment_weighted_auroc(normal_scores, anomaly_scores) -> float:
    """Ordinary strict pairwise AUROC; equals the mean of ``C_i`` exactly."""
    return float(concordance_per_segment(normal_scores, anomaly_scores).mean())


@dataclass(frozen=True)
class ConcordanceDelta:
    """Result of the frozen primary estimator. Carries no model state."""

    point_estimate: float
    n_parents: int
    n_segments: int
    parent_deltas: pd.Series
    segment_deltas: pd.Series
    ci_low: float | None = None
    ci_high: float | None = None
    bootstrap_replicates: int = 0
    bootstrap_seed: int | None = None
    bootstrap_stratified_by_fold: bool = True
    interval_is_conditional_on_trained_models: bool = True
    fold_deltas: pd.Series | None = field(default=None)


def parent_concordance_delta(
    scores: pd.DataFrame,
    *,
    bootstrap: bool = True,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    stratify_by_fold: bool = True,
) -> ConcordanceDelta:
    """Parent-equal-weighted focal concordance delta from persisted scores.

    ``scores`` must hold one row per (segment, arm) for the focal normal
    segments and the fixed anomaly reference, with columns ``segment_id``,
    ``parent_id``, ``arm`` (``PRESENT``/``ABSENT``), ``anomaly_label``,
    ``total_loss`` and, when available, ``focal_fold``.

    The bootstrap resamples PARENTS, never segments, so the 203 segment-level
    deltas are never treated as independent observations. With
    ``stratify_by_fold`` (the frozen default) it resamples 12 parents within
    each fold, holding the fold composition -- and therefore the mixture of
    fitted model pairs -- fixed. The interval is explicitly CONDITIONAL ON THE
    FITTED MODELS: it does not estimate training-seed, model-pair or
    site-population uncertainty. Five folds cannot support a cluster-variance
    estimate, so fold is a stratification level, never a resampling level.
    """
    required = {"segment_id", "parent_id", "arm", "anomaly_label", "total_loss"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"scores is missing columns: {sorted(missing)}")
    if set(scores["arm"]) != {"PRESENT", "ABSENT"}:
        raise ValueError("scores must contain exactly the arms PRESENT and ABSENT")

    per_arm: dict[str, pd.Series] = {}
    for arm, sub in scores.groupby("arm"):
        normal = sub[sub.anomaly_label == "normal"]
        anomaly = sub[sub.anomaly_label != "normal"]
        values = concordance_per_segment(normal.total_loss, anomaly.total_loss)
        per_arm[arm] = pd.Series(values, index=normal.segment_id.to_numpy(), name=arm)

    present, absent = per_arm["PRESENT"], per_arm["ABSENT"]
    if set(present.index) != set(absent.index):
        raise ValueError("the two arms must score exactly the same focal normal segments")
    absent = absent.reindex(present.index)

    segment_deltas = (present - absent).rename("delta_i")
    parent_of = (scores.drop_duplicates("segment_id")
                 .set_index("segment_id")["parent_id"].reindex(segment_deltas.index))
    # MEAN within parent, per the frozen decision: parents hold a median of 3
    # segments and 11 of 60 hold exactly one, where a median would be unstable.
    parent_deltas = segment_deltas.groupby(parent_of.to_numpy()).mean().sort_index()
    parent_deltas.name = "delta_parent"

    fold_deltas = None
    if "focal_fold" in scores.columns:
        fold_of = (scores.drop_duplicates("parent_id")
                   .set_index("parent_id")["focal_fold"].reindex(parent_deltas.index))
        if fold_of.notna().all():
            fold_deltas = parent_deltas.groupby(fold_of.to_numpy()).mean()
            fold_deltas.name = "delta_fold"

    point = float(parent_deltas.mean())
    low = high = None
    if bootstrap:
        values = parent_deltas.to_numpy()
        rng = np.random.default_rng(seed)
        if stratify_by_fold:
            if fold_deltas is None:
                raise ValueError(
                    "fold-stratified bootstrap needs a 'focal_fold' column for every parent")
            fold_labels = fold_of.to_numpy()
            blocks = [np.flatnonzero(fold_labels == f) for f in np.unique(fold_labels)]
            # Resample WITHIN each fold, keeping 12 parents per fold. Each fold's
            # parents share one fitted PRESENT/ABSENT pair, so holding the fold
            # composition fixed keeps the model mixture of the estimand intact;
            # free resampling would let between-model differences masquerade as
            # parent-sampling noise.
            means = np.zeros(replicates)
            for block in blocks:
                draws = rng.integers(0, block.size, size=(replicates, block.size))
                means += values[block][draws].sum(axis=1)
            means /= values.size
        else:
            draws = rng.integers(0, values.size, size=(replicates, values.size))
            means = values[draws].mean(axis=1)
        low, high = (float(x) for x in np.percentile(means, [2.5, 97.5]))

    return ConcordanceDelta(
        point_estimate=point,
        n_parents=int(parent_deltas.size),
        n_segments=int(segment_deltas.size),
        parent_deltas=parent_deltas,
        segment_deltas=segment_deltas,
        ci_low=low, ci_high=high,
        bootstrap_replicates=replicates if bootstrap else 0,
        bootstrap_seed=seed if bootstrap else None,
        bootstrap_stratified_by_fold=bool(stratify_by_fold and bootstrap),
        fold_deltas=fold_deltas,
    )


# ----------------------------------------------------------------------------
# IMPL-1 correction (Phase 3.4, after e19cffd)
#
# ``parent_concordance_delta`` above is retained UNCHANGED as the historical
# estimator: when handed a frame pooled over several folds it ranks each focal
# segment against the anomaly rows of EVERY fold's model of the same arm
# (232 x n_folds rows), whereas the frozen scientific text intends the 232
# anomaly segments scored by the SAME fold's model and SAME arm. The function
# below makes fold identity an explicit part of the reference matching.
# ----------------------------------------------------------------------------
ANOMALY_REFERENCE_SIZE = 232


@dataclass(frozen=True)
class FoldReferenceAudit:
    """Per (fold, arm): how many anomaly-reference rows were used and from where."""

    fold: int
    arm: str
    n_reference: int
    reference_folds: tuple[int, ...]
    reference_arms: tuple[str, ...]
    n_focal_normal: int


def parent_concordance_delta_by_fold(
    scores: pd.DataFrame,
    *,
    bootstrap: bool = True,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    reference_size: int = ANOMALY_REFERENCE_SIZE,
) -> tuple[ConcordanceDelta, list[FoldReferenceAudit]]:
    """Fold-specific-reference parent concordance delta (IMPL-1 corrected).

    ``scores`` holds one row per (segment, arm, fold-model): focal normal
    segments and the anomaly reference, each row carrying the ``focal_fold`` of
    the fitted model that produced its score. For every fold ``f`` and arm ``a``
    the concordance of that fold's focal normal segments is computed against
    EXACTLY the ``reference_size`` anomaly rows scored by fold ``f``'s model of
    arm ``a`` - never against another fold's model. Everything downstream
    (per-segment delta, parent mean, equal-parent mean, fold-stratified parent
    bootstrap with percentile 95% interval) is the frozen definition.
    """
    required = {"segment_id", "parent_id", "arm", "anomaly_label", "total_loss", "focal_fold"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"scores is missing columns: {sorted(missing)}")
    if scores["focal_fold"].isna().any():
        raise ValueError("every row, including anomaly-reference rows, must carry the fold of its model")
    if set(scores["arm"]) != {"PRESENT", "ABSENT"}:
        raise ValueError("scores must contain exactly the arms PRESENT and ABSENT")

    audits: list[FoldReferenceAudit] = []
    per_arm_fold: dict[tuple[str, int], pd.Series] = {}
    for (fold, arm), sub in scores.groupby(["focal_fold", "arm"]):
        fold = int(fold)
        normal = sub[sub.anomaly_label == "normal"]
        anomaly = sub[sub.anomaly_label != "normal"]
        # the reference is exactly the fixed anomaly cohort, scored by THIS fold's model of THIS arm
        if anomaly.segment_id.nunique() != reference_size or len(anomaly) != reference_size:
            raise ValueError(f"fold {fold} arm {arm}: expected exactly {reference_size} anomaly-reference "
                             f"rows from its own model, found {len(anomaly)} rows / "
                             f"{anomaly.segment_id.nunique()} unique")
        if set(anomaly.focal_fold.astype(int)) != {fold} or set(anomaly.arm) != {arm}:
            raise ValueError("reference rows must come from the same fold and arm")
        values = concordance_per_segment(normal.total_loss, anomaly.total_loss)
        per_arm_fold[(arm, fold)] = pd.Series(values, index=normal.segment_id.to_numpy())
        audits.append(FoldReferenceAudit(fold, str(arm), int(len(anomaly)),
                                         tuple(sorted(set(anomaly.focal_fold.astype(int)))),
                                         tuple(sorted(set(anomaly.arm))), int(len(normal))))

    folds = sorted({f for _, f in per_arm_fold})
    present = pd.concat([per_arm_fold[("PRESENT", f)] for f in folds])
    absent = pd.concat([per_arm_fold[("ABSENT", f)] for f in folds])
    if set(present.index) != set(absent.index) or not present.index.is_unique:
        raise ValueError("the two arms must score exactly the same focal normal segments, once each")
    absent = absent.reindex(present.index)

    segment_deltas = (present - absent).rename("delta_i")
    normals = scores[scores.anomaly_label == "normal"].drop_duplicates("segment_id").set_index("segment_id")
    parent_of = normals["parent_id"].reindex(segment_deltas.index)
    parent_deltas = segment_deltas.groupby(parent_of.to_numpy()).mean().sort_index()
    parent_deltas.name = "delta_parent"
    fold_of = normals.drop_duplicates("parent_id").set_index("parent_id")["focal_fold"].reindex(parent_deltas.index).astype(int)
    fold_deltas = parent_deltas.groupby(fold_of.to_numpy()).mean()
    fold_deltas.name = "delta_fold"

    point = float(parent_deltas.mean())
    low = high = None
    if bootstrap:
        values = parent_deltas.to_numpy()
        rng = np.random.default_rng(seed)
        fold_labels = fold_of.to_numpy()
        blocks = [np.flatnonzero(fold_labels == f) for f in np.unique(fold_labels)]
        means = np.zeros(replicates)
        for block in blocks:
            draws = rng.integers(0, block.size, size=(replicates, block.size))
            means += values[block][draws].sum(axis=1)
        means /= values.size
        low, high = (float(x) for x in np.percentile(means, [2.5, 97.5]))

    result = ConcordanceDelta(
        point_estimate=point,
        n_parents=int(parent_deltas.size),
        n_segments=int(segment_deltas.size),
        parent_deltas=parent_deltas,
        segment_deltas=segment_deltas,
        ci_low=low, ci_high=high,
        bootstrap_replicates=replicates if bootstrap else 0,
        bootstrap_seed=seed if bootstrap else None,
        bootstrap_stratified_by_fold=bool(bootstrap),
        fold_deltas=fold_deltas,
    )
    return result, audits
