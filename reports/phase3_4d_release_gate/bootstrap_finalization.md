# Phase 3.4D — Bootstrap Finalisation

**Applies regardless of the design decision.** The primary estimand itself is unchanged.

---

## What changed

| | |
|---|---|
| **Was (3.4C)** | free resampling of the 60 focal parents |
| **Now (frozen)** | **fold-stratified parent bootstrap** |

## The frozen procedure

For each of **10,000** replicates, under seed **27182**:

1. for each of the five folds, sample **12** focal parents **with replacement** from that fold's own
   12 focal parents;
2. retain each selected parent's **complete** focal-normal segment set;
3. compute one parent-level delta per selected parent (MEAN of its segment deltas, unchanged);
4. combine the 5 × 12 = **60** resampled parent deltas and take the parent-equal mean.

Interval: **percentile 95%**.

## Why stratifying is the coherent choice

Each fold's 12 parents share **one fitted PRESENT/ABSENT model pair**. Free resampling could draw 20
parents from one fold and 5 from another, silently changing the mixture of fitted models that the
estimand averages over — so between-model differences would masquerade as parent-sampling noise.

Stratifying holds the fold composition, and therefore the model mixture, **fixed**. That is the
correct bootstrap for an interval we have already declared conditional on those models. It is also
the more honest of the two: the free version's extra width comes from a source it cannot legitimately
estimate with five clusters.

Empirically, on synthetic data with a known effect, the stratified interval is slightly narrower
(width 0.1318 vs 0.1396) with an **identical point estimate**, exactly as expected.

## What this interval is, and is not

> **CONDITIONAL ON THE TEN FITTED MODELS.**

It does **not** estimate:

- training-seed uncertainty,
- model-pair sampling uncertainty,
- site or general-population uncertainty.

The **five fold-level point estimates are reported separately** as a descriptive spread indicating
model-level variability.

**N = 5 folds must never be called an inferential sample.** Fold is a *stratification* level, never a
*resampling* level: five clusters cannot support a cluster-variance estimate, and resampling them
would produce an unstable, misleading interval.

## Implementation

`src/experiment/focal_concordance.py` — `parent_concordance_delta(..., stratify_by_fold=True)`,
the frozen default. The result object carries `bootstrap_stratified_by_fold` and
`interval_is_conditional_on_trained_models` so the qualifier travels with the number rather than
living only in prose.
