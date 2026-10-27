# Sibling-Exposure Sensitivity Study

<!-- Legacy study name: Phase 3.4 -->

## Benchmark Integrity and Evaluation Validation — COMPLETE

<!-- Legacy roadmap name: Phase 3; status: CLOSED -->

The benchmark integrity and evaluation validation work is complete and closed. It includes
classical baseline evaluation, official baseline reproduction and the RoboticArm sibling-exposure
sensitivity study. The next research stage is Single-Modality Model Benchmarking, which has not
started.

Under the frozen RoboticArm sensitivity design, no clear detectable change in focal-parent
normal-vs-anomaly concordance was associated with same-parent sibling exposure. The estimated
effect was small and weakly positive: **delta_parent_concordance = +0.00787, 95% CI
[−0.00061, +0.01779]**. The interval includes zero; the study does not establish a causal leakage
effect. The corrected analysis used the already-frozen raw scores, without retraining or new
protected inference.

**Scientific result going forward:** the **IMPL-1-corrected five-fold analysis**
(`reports/phase3_4_results/phase3_4_impl1_corrected_5fold_analysis.json`,
SHA-256 `ecd431f61128d7ead8c2d7082cc266370c73226422d91554b06f663962af3349`).

> Question (pre-registered, `p34d-exec-e5470b025229f59a06f4c772`): for donor-matchable focal
> RoboticArm parents, does exposure to same-parent training siblings change parent-weighted
> normal-vs-anomaly concordance? Estimator: per-segment concordance `C_i(a)` against the fixed
> 232-segment anomaly reference **scored by the same fold's model of the same arm**,
> `delta_i = C_i(PRESENT) − C_i(ABSENT)`, parent mean, equal-parent mean; fold-stratified parent
> bootstrap (12 parents/fold, 10,000 replicates, seed 27182, percentile 95 %).

---

## 1. Technical history and provenance

The study was developed on `phase3-fold1-technical-recovery` and merged into `master` at
`2da3a58`. The safety tag `phase3-pre-fold1-recovery` remains at `a8b3d6c`.

1. **Technical evaluation interruption (INC-1).** After the ten test looks were authorised (Step D, `576b8a1`), a
   COMMIT-B-era test invoked the real scorer against the real results root; the ledger gate now
   passed, the single-use claim for fold-1 PRESENT was acquired and `SCORE_STARTED` appended
   before the test's inference sentinel raised. Zero protected forwards, zero score rows, no
   `SCORE_COMPLETED`, no artifact. Claim and record preserved byte-for-byte ever since.
2. **Four-fold incident-contingent analysis** (`64694e6` freeze, E1 `ac71988`, E2 `a8b3d6c`).
   Eight retained looks scored once each; salvage estimand over folds 2–5 / 48 parents.
3. **Technical recovery of fold 1** (amendment `a799934`, raw `8282575`, analysis `e19cffd`).
   Versioned amendment frozen before scoring, evaluator proven byte-identical to the GREEN audit,
   both fold-1 checkpoints re-evaluated once each under new recovery identities; INC-1 untouched.
4. **Analysis implementation issue (IMPL-1).** During the five-fold analysis, folds 2–5 descriptive deltas shifted
   between the four- and five-fold analyses although the artifacts were identical.
5. **Why IMPL-1 matters.** The analysis code concatenated every fold's frame — focal normals plus
   that fold's 232 anomaly-reference rows — and passed the pooled frame to
   `parent_concordance_delta`, which ranks each focal segment against **all** anomaly rows of its
   arm in the frame (232 × n_folds rows from different fitted models). `C_i` is a within-model rank
   statistic; mixing anomaly scores from other folds' models compares scores across independently
   trained networks on different raw scales — exactly what the rank definition was chosen to avoid —
   and makes each fold's result depend on which other folds are present.
6. **Original computed values retained historically** (superseded for interpretation, artifacts
   untouched): four-fold pooled Δ = +0.00844, CI [−0.00141, +0.02038] (`6441ed66…`);
   five-fold pooled Δ = +0.00809, CI [+0.00009, +0.01764] (`427b04cb…`).
7. **Corrected four-fold result** (folds 2–5, 48 parents): see §2.
8. **Corrected five-fold result** (folds 1–5, 60 parents): see §2.
9. **The corrected five-fold result is the scientific result used going forward.**

## 2. Results

| Analysis | Reference | Δ_parent_concordance | 95 % CI | Parents +/0/− | Status |
|---|---|---|---|---|---|
| Four-fold (folds 2–5, 48 parents) | pooled (IMPL-1) | +0.00844 | [−0.00141, +0.02038] | 17/6/25 | historical, superseded |
| Five-fold (folds 1–5, 60 parents) | pooled (IMPL-1) | +0.00809 | [+0.00009, +0.01764] | 28/6/26 | historical, superseded |
| **Four-fold corrected** | fold-specific | **+0.00789** | **[−0.00255, +0.02049]** | 18/8/22 | corrected, incident-contingent |
| **Five-fold corrected — SCIENTIFIC RESULT** | fold-specific | **+0.00787** | **[−0.00061, +0.01779]** | 28/10/22 | corrected, original design |

Corrected fold-level deltas (descriptive, one fitted pair each; identical in both corrected
analyses because they are fold-local by construction): fold 1 +0.00780 · fold 2 +0.03073 ·
fold 3 +0.00742 · fold 4 −0.00920 · fold 5 +0.00262. Focal normal segments 203 (158 in folds 2–5).

Per-(fold, arm) reference audit in each corrected artifact: exactly 232 anomaly rows, all from the
same fold and arm. Corrected analyses reproduce exactly on recomputation.

**Secondary quantities.** S1 (segment-weighted focal AUROC Δ by fold: +0.0088 / +0.0331 / +0.0023 /
−0.0011 / +0.0028), S4 (raw focal-normal shift) and S5 (anomaly-reference raw shift) were always
computed inside each fold's own sub-frame — **UNAFFECTED**, values identical to the historical
artifacts. Affected and corrected: primary point estimate, CI, fold-level deltas, parent deltas,
sign counts.

## 3. Interpretation (within the frozen claim boundary)

Across all five pre-registered fold pairs, with each fold's focal parents ranked against its own
model's anomaly reference, same-parent sibling exposure is associated with a small positive shift
in focal-parent normal-vs-anomaly concordance: **+0.008 (under one percentage point of rank
position), 95 % interval [−0.0006, +0.0178], which includes zero.** 28 of 60 parents moved
positive, 10 were unchanged, 22 moved negative. The fold-2 fitted pair (+0.031) contributes most
of the mean; fold 4 is negative (−0.009); the other three are within ±0.008. The corrected result
is materially the same as the historical pooled values; the correction changes the reference
definition, not the conclusion. Read plainly: **under the pre-registered RoboticArm sensitivity
design, we detected no clear change in focal-parent normal-vs-anomaly concordance associated with
same-parent sibling exposure; the estimated effect was small and weakly positive
(Δ = +0.0079, 95 % CI [−0.0006, +0.0178]).** Reported as computed; nothing tuned.

Not claimed: a causal leakage effect; that the published baseline is invalid or inflated by any
amount. Conditional on the ten fitted models (no training-seed, model-pair or site/population
uncertainty). Residual V3 parent gap 0–3 (ABSENT larger); donor material metadata-matched but
signal content differs; training volume ≈ 76–85 % of official; fold 1 completed after a documented
zero-exposure technical recovery decided after the four-fold result was known and accepted
regardless of value.

## 4. Provenance state

Ledger 40 (28 B + 2 A + 10 Phase-3.4), unchanged since Step D. Claims 11: 8 retained completed,
INC-1 `[SCORE_STARTED]` (original namespace, never reused), 2 recovery completed. Raw score
artifacts 10 × 464 rows, unchanged. Protected inference during the correction: 0. Estimator
module: historical `parent_concordance_delta` retained unchanged; corrected
`parent_concordance_delta_by_fold` added with fold identity as an explicit matching key.
Correction manifest: `reports/phase3_4_results/impl1_correction_manifest.json`.
