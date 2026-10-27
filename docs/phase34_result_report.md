# Phase 3.4 — Result Report: Focal-Parent Sibling-Toggle Sensitivity (RoboticArm)

**Analysis label:** INCIDENT-CONTINGENT SECONDARY SENSITIVITY ANALYSIS
**Recovery config:** `p34r-inc1-4f86592159eadbbe276d9c6f` (parent `p34d-exec-e5470b025229f59a06f4c772`)
**Evaluator:** `64694e6` (protected evaluator unchanged from COMMIT B `c10fdfc`, audited GREEN)
**E1 provenance commit:** `ac71988` · **Analysis artifact:** `reports/phase3_4_results/phase3_4_inc1_salvage_analysis.json`
(SHA-256 `6441ed66b88074417ee0772e2618af0842050d2533d48029bb07a831ec1384bd`)

> **Status of the pre-registered analysis.** The original design had 5 paired folds / 60 focal
> parents. INC-1 consumed fold-1 PRESENT before inference (zero protected forwards, zero score
> observed); the frozen single-use policy prohibited retry; fold-1 ABSENT was intentionally left
> unused. **The original five-fold primary analysis is NOT FULLY ESTIMABLE.** The result below is a
> pre-score, incident-contingent, secondary sensitivity analysis over folds 2–5 / 48 parents. It is
> not confirmatory and not a repaired version of the primary analysis.

---

## 1. Question (frozen salvage estimand)

*For donor-matchable focal RoboticArm parents in intact folds 2–5, does exposure to same-parent
training siblings change parent-weighted normal-vs-anomaly concordance?*

## 2. Population and inputs

- Retained folds 2, 3, 4, 5 · 4 complete PRESENT/ABSENT pairs · 8 models scored once each (E1).
- 48 focal parents (12 per fold), 158 focal normal test segments (descriptive), fixed 232-segment
  anomaly reference scored by the same arm.
- Inputs: the eight hash-verified E1 artifacts (combined table SHA-256 `199474ef…58875`). No
  inference, no claim, no exposure write, no ledger change occurred during analysis.

## 3. Estimator and uncertainty (frozen before any score existed)

Per-segment concordance `C_i(a)` against the anomaly reference within arm `a`; `delta_i =
C_i(PRESENT) − C_i(ABSENT)`; parent-level mean over the parent's focal normal segments;
**equal-parent mean over the 48 parents**. Uncertainty: fold-stratified parent bootstrap (12 parents
with replacement within each retained fold → 48 per replicate), 10,000 replicates, seed 27182,
percentile 95 % interval — conditional on the eight fitted models and the four retained folds.
Implementation: unchanged `parent_concordance_delta` (`focal_concordance.py` byte-identical to COMMIT B).

## 4. Result (reported as computed; nothing tuned)

| Quantity | Value |
|---|---|
| **Δ_parent_concordance (point estimate, 48 parents)** | **+0.0084** |
| 95 % bootstrap interval | **[−0.0014, +0.0204]** — crosses zero |
| Parents with Δ > 0 / = 0 / < 0 | 17 / 6 / 25 |
| Fold-level Δ (descriptive, one fitted pair each) | fold 2: +0.0296 · fold 3: +0.0032 · fold 4: −0.0020 · fold 5: +0.0029 |
| Focal normal segments (descriptive) | 158 |

Secondary descriptive quantities (per fold, PRESENT − ABSENT): S1 segment-weighted focal AUROC delta
fold 2 +0.0331, 3 +0.0023, 4 −0.0011, 5 +0.0028; S4 raw focal-normal score shift (scale-dependent)
−0.0038 / +0.0009 / −0.0016 / −0.0001; S5 anomaly-reference raw shift −0.0012 / +0.0006 / −0.0013 /
−0.0011. In-memory recomputation reproduces every value exactly.

## 5. Interpretation (within the frozen claim boundary)

The parent-equal-weighted concordance delta is small and positive (+0.008 on the concordance scale,
i.e. under one percentage point of rank-position against the anomaly reference), and its 95 %
interval includes zero. More parents moved negative than positive (25 vs 17). The mean is pulled by
fold 2, whose fitted pair shows a larger positive delta (+0.030); folds 3–5 are within ±0.003.
Read plainly: **in this four-fold sensitivity analysis we detected no clear change in focal-parent
normal-vs-anomaly concordance associated with exposure to same-parent training siblings; the
estimated effect was small and weakly positive and not distinguishable from zero.** This is a
null-compatible result, not a failure of the experiment. (Historical: superseded by the IMPL-1
corrected analyses, see `docs/phase34_final_report.md`.)

What it does not say: it is not a causal leakage claim; it does not state that the published
baseline is invalid or inflated by any amount; it does not estimate the pre-registered five-fold
quantity; the fold-2 value is one fitted pair and is descriptive only.

## 6. Required limitations

- Original design 5 folds / 60 parents; salvage uses folds 2–5 / 48 parents only, for a non-random
  procedural reason (INC-1), fixed before any score existed.
- INC-1: fold-1 PRESENT consumed before inference; zero protected forwards; zero score observed;
  retry prohibited by the frozen single-use policy; fold-1 ABSENT intentionally unused.
- Original five-fold primary analysis incomplete; this result is incident-contingent and secondary.
- The interval is conditional on eight fitted models and four folds; it carries no training-seed,
  model-pair, site/population, or missing-fold-1 uncertainty. N = 4 folds is not an inferential sample.
- Residual V3 parent-count gap remains 0–3, ABSENT larger.
- Donor material is metadata-matched (segment count, windows, 25 strata) but its signal content differs.
- Training volume remains approximately 76–85 % of official training in both arms.
- No causal leakage claim is made.

## 7. Provenance chain

COMMIT A `107579f` → F1/F2 `dcb57d1` → F3 `e93886c` → COMMIT B `c10fdfc` → Step D `576b8a1` →
INC-1 recovery freeze `64694e6` → evaluator diff audit GREEN → E1 `ac71988` → E2 (this commit).
Ledger: 40 entries (28 B + 2 A + 10 Phase-3.4), append-only, unchanged since Step D. Claims: 9
(8 retained completed + INC-1 incomplete). Score artifacts: 8 × 464 rows.
