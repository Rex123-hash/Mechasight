# Phase 3.4 — Fold-1 Recovery Result Report: Both Analyses

> **SUPERSEDED FOR CURRENT SCIENTIFIC INTERPRETATION**
>
> This report documents the technical recovery and the historical intermediate analysis.
> A subsequent analysis correction introduced fold-specific anomaly references using the
> already-frozen raw scores. The current scientific result is **delta = +0.00787,
> 95% CI [−0.00061, +0.01779]**. See the [Sibling-Exposure Sensitivity Study](phase34_final_report.md).
> The original report below is retained as a historical record.

**Amendment:** `fold1-technical-recovery-v1` (commit `a799934`) · **Raw recovery scores:** `8282575`
**Safety tag:** `phase3-pre-fold1-recovery` = `a8b3d6c` · **Branch:** `phase3-fold1-technical-recovery`
**Five-fold artifact:** `reports/phase3_4_results/phase3_4_five_fold_recovered_analysis.json`
(SHA-256 `427b04cb96c97fe1dd218d515c1f0fb3378e65551d95c9a30d858f1347807188`)

> The four-fold incident-contingent result (A) is preserved exactly as committed at `a8b3d6c`.
> The five-fold result (B) was completed after a documented zero-exposure technical recovery of
> fold-1 PRESENT (INC-1) using the same frozen models, without retraining or tuning. Both are
> reported; neither was selected over the other; the decision to recover was made after A was known
> and is not justified by A's direction.

---

## A. Previous incident-contingent four-fold result — PRESERVED UNCHANGED

| Quantity | Value |
|---|---|
| Δ_parent_concordance (48 parents, folds 2–5) | **+0.00844** |
| 95 % fold-stratified parent bootstrap | **[−0.00141, +0.02038]** — includes zero |
| Parents Δ > 0 / = 0 / < 0 | 17 / 6 / 25 |
| Artifact SHA-256 | `6441ed66…1384bd` (commit `a8b3d6c`) |

## B. Recovered original five-fold result

| Quantity | Value |
|---|---|
| **Δ_parent_concordance (60 parents, folds 1–5)** | **+0.00809** |
| **95 % fold-stratified parent bootstrap (10,000 reps, seed 27182)** | **[+0.00009, +0.01764]** — excludes zero by a hair |
| Parents Δ > 0 / = 0 / < 0 | 28 / 6 / 26 |
| Fold-level Δ (descriptive, one fitted pair each) | fold 1: +0.00744 · 2: +0.02999 · 3: +0.00294 · 4: −0.00249 · 5: +0.00256 |
| Focal normal segments (descriptive) | 203 |
| S1 segment-weighted focal AUROC Δ by fold | +0.0088 / +0.0331 / +0.0023 / −0.0011 / +0.0028 |

Fold 1 (recovered pair): Δ = +0.0074 — inside the range of the other folds. In-memory
recomputation reproduces the artifact exactly.

## C. Four-fold vs five-fold comparison

- Point estimates agree: +0.0084 (A) vs +0.0081 (B); adding fold 1 moved the mean by −0.0004.
- The interval narrowed with 60 parents and its lower bound moved from −0.0014 to +0.0001. A
  lower bound of 0.00009 on a concordance scale is not a meaningful separation from zero; it is
  a boundary case and is reported as such, not as a change in conclusion.
- Sign counts: A 17/6/25 → B 28/6/26 (fold 1 contributed 11 positive, 0 zero, 1 negative parent).
- Fold 2's single fitted pair remains the dominant positive contribution in both analyses.
- **No strong disagreement** between A and B.

## D. Interpretation (within the frozen claim boundary)

Across all five pre-registered fold pairs, exposure to same-parent training siblings is
associated with a small positive shift in focal-parent normal-vs-anomaly concordance
(≈ +0.008, under one percentage point of rank position against the anomaly reference). The
five-fold interval sits essentially at zero at its lower end; the four-fold interval includes
zero. Read together: the effect, if any, is small, weakly positive, and driven mostly by one
fold's fitted pair. This is not a causal leakage claim, does not state that the published
baseline is invalid or inflated by any amount, and is conditional on the ten fitted models
(no training-seed, model-pair or site/population uncertainty). Residual V3 parent gap 0–3
(ABSENT larger); donor material metadata-matched but signal content differs; training volume
≈ 76–85 % of official.

## E. Finding IMPL-1 — disclosed, not corrected

While computing B it became visible that the **frozen estimator implementation pools the
anomaly reference across all folds present in the analysis frame** (1,160 anomaly rows per arm
with five folds; 928 with four), whereas the pre-registered text states `N_anomaly = 232` within
the same arm. Each focal segment is therefore ranked against anomaly scores produced by every
fold's model of its arm, not only its own fold's model. This is why the fold-level descriptive
deltas for folds 2–5 differ slightly between A and B (e.g. fold 2: 0.029649 → 0.029988) even though
the underlying artifacts are byte-identical. The estimator was frozen and audited before any
score existed; both committed results use it consistently. **Nothing was changed after seeing
results.** A per-fold-reference variant would be a new post-hoc analysis and requires
independent authorisation; it is not computed here. Recorded in
`fold1_recovery_result_manifest.json` (`finding_IMPL_1`).

## F. Provenance

INC-1 original claim + `[SCORE_STARTED]` byte-unchanged (tested against `576b8a1`). Recovery
identities `p34r-f1rec-78c8f486fdb7c822af87` (1P, checkpoint `85386f2d…`) and
`p34r-f1rec-e83624b2152b673c30c5` (1A, checkpoint `4b4d848b…`), one evaluation each, claims
bound to amendment commit `a799934`; artifacts `c5d51937…` and `e08e89b4…`; combined
`cdda80f9…`. Ledger 40, unchanged since Step D. Evaluator `_run_protected_inference` and
persistence byte-identical to the GREEN audit at `64694e6`. Phase-4 failure policy:
`docs/phase4_failure_policy_note.md`.
