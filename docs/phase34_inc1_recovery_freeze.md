# Phase 3.4 — INC-1 Recovery Freeze (pre-scoring)

**Stage:** between D and E of the audited sequence A–H. **Nothing scored. No protected inference.**
**Recovery config identity:** `p34r-inc1-4f86592159eadbbe276d9c6f` · **parent config:** `p34d-exec-e5470b025229f59a06f4c772`
**Frozen at HEAD before this commit:** `576b8a1`

> Mandatory statements (they travel with every downstream result):
> the original design had 5 paired folds / 60 focal parents · INC-1 consumed fold-1 PRESENT before
> inference · zero protected forwards occurred in INC-1 · zero score was observed from INC-1 · the
> frozen single-use policy prohibited retry · fold-1 ABSENT was intentionally left unused · the
> original five-fold primary analysis is incomplete · the folds 2–5 analysis is a pre-score
> incident-contingent secondary sensitivity analysis.

---

## 1. Incident record — INC-1

| Field | Value |
|---|---|
| Experiment | `p34d-92fd40fc93c8c8d2e819` (fold 1, PRESENT, seed 1995) |
| Checkpoint | `85386f2df404e5db49923b604e08676b9d6d934b49c5d3c633f785e5c676060e` (unchanged on disk) |
| Authorization (ledger) | `2026-09-11T19:49:04.480058Z`, `git_sha c10fdfc…` |
| Claim (`score_claims/…claim`) | `CLAIMED`, `2026-09-11T19:50:25.721473Z` |
| Attempt history | `[SCORE_STARTED]` at `…25.722314Z`; **no** `SCORE_COMPLETED` |
| Score artifact | none (no `scores/` directory exists) |
| Protected forward passes | **0** |
| Test performance observed | **none** |
| Cause | COMMIT B version of `test_score_refuses_before_any_claim_or_exposure` called `score_experiment` against the real results root after step D; the ledger gate legitimately passed |
| Scorer behaviour | exactly as frozen (F1/F3) |

**Zero-forward evidence.** (i) `_run_protected_inference` — the only function that reads protected
test data — was replaced by a sentinel that raised immediately and read nothing; (ii) the scorer
writes `SCORE_COMPLETED` only after an artifact is persisted and hashed, and the append-only attempt
record ends at `SCORE_STARTED`; (iii) no per-segment score row for any experiment exists anywhere in
the repository; (iv) the claim and attempt files are committed at `576b8a1` exactly as created and
are byte-identical now (tested).

## 2. Frozen decisions

- **Fold-1 PRESENT — NO RETRY.** Permanently `CONSUMED / INCOMPLETE`. The claim is not deleted, not
  replaced, not recovered. "No forward occurred" is not permission to re-score.
- **Fold-1 ABSENT (`p34d-e56529a2f56f98fd4a7a`) — DO NOT SCORE.** Permanently `AUTHORIZED / UNUSED`.
  Its paired counterpart is incomplete; an unpaired ABSENT score cannot enter a paired estimand. Its
  ledger authorization stays in the append-only history; the look stays unclaimed.
- **Ledger.** Append-only. None of the ten step-D authorizations is removed, replaced or re-timestamped.
- **Evidence.** Claim, `SCORE_STARTED` record and ledger entries are scientific provenance; never rewritten.

Enforced in code, before any claim: `_recovery_gate` in
[focal_toggle_execution.py](../src/experiment/focal_toggle_execution.py) refuses both fold-1 ids
with the frozen reason, and refuses any retained id that is not the next one lacking
`SCORE_COMPLETED` in the frozen order (no skipping, no reordering; a claimed-but-incomplete retained
look blocks the phase). Existing gates are unchanged.

## 3. Original five-fold primary analysis

**NOT FULLY ESTIMABLE DUE TO INC-1.** Fold-1 PRESENT has no valid score and can never obtain one;
fold-1 ABSENT is intentionally unused. Silently dropping fold 1 and calling the remainder the
original primary analysis is prohibited. `analyse_from_persisted_scores` now refuses to run the
five-fold analysis at all: it reads only the eight retained artifacts and labels its output.

## 4. Salvage cohort — INCIDENT-CONTINGENT SECONDARY SENSITIVITY ANALYSIS

Fixed before any score exists; not selected on any outcome.

| Pos | Label | Experiment | Fold | Arm | Checkpoint |
|--:|---|---|--:|---|---|
| 1 | 2A | `p34d-fd6e3bfd4c091e155d99` | 2 | ABSENT | `1a964dedde00…` |
| 2 | 2P | `p34d-a4e8316f62fa982ec52d` | 2 | PRESENT | `2c8029ad1a7b…` |
| 3 | 3P | `p34d-88b93ddbacbc5829f326` | 3 | PRESENT | `c6b30ee064a0…` |
| 4 | 3A | `p34d-2af84983abd170013b73` | 3 | ABSENT | `0c0a46d234d2…` |
| 5 | 4A | `p34d-652bc1e3f9d29a51a460` | 4 | ABSENT | `826035dccd5b…` |
| 6 | 4P | `p34d-406b59f08e154b284159` | 4 | PRESENT | `d756eac00a03…` |
| 7 | 5P | `p34d-66656cf0901b180cf037` | 5 | PRESENT | `7a2b9911e264…` |
| 8 | 5A | `p34d-ed8467570acf122c36cd` | 5 | ABSENT | `5a49306aae72…` |

Exactly 8 looks, 4 complete PRESENT/ABSENT pairs, **48 focal parents** (12 per retained fold,
confirmed from the frozen membership file), 158 focal normal test segments (descriptive),
8 × 464 = 3,712 expected persisted score rows.

**Frozen scoring order:** `2A 2P 3P 3A 4A 4P 5P 5A` — the parent execution matrix with fold 1
removed. Never reordered adaptively. If any retained look becomes incomplete: STOP the phase, report
as incomplete, no retry, no substitution.

## 5. Salvage estimand and estimator

*"For donor-matchable focal RoboticArm parents in intact folds 2–5, does exposure to same-parent
training siblings change parent-weighted normal-vs-anomaly concordance?"*

This is **not** the original five-fold estimand. It is never called confirmatory and never called
the original primary Phase-3.4 result.

Estimator unchanged in every definition: per-segment concordance `C_i(a)` against the fixed
232-segment anomaly reference scored by the same arm; `delta_i = C_i(PRESENT) − C_i(ABSENT)`;
parent-level mean over the parent's focal normal segments; **equal-parent mean over the 48 retained
parents**. No segment weighting, no change to the score definition, the anomaly reference cohort,
or checkpoint selection.

## 6. Salvage bootstrap (frozen before any scoring)

Fold-stratified parent bootstrap restricted to folds 2, 3, 4, 5: resample 12 focal parents with
replacement within each retained fold, carry each selected parent's complete focal-normal segment
set, one parent-level delta per selected parent, equal-parent mean of the 48 resampled deltas;
10,000 replicates; seed 27182; percentile 95 % interval. Implemented by the unchanged
`parent_concordance_delta(stratify_by_fold=True)`, which resamples within whichever folds are
present — no estimator code change.

The interval is **conditional on the eight fitted models and the four retained folds**. It does
not include training-seed uncertainty, model-pair uncertainty, site/population uncertainty, or
uncertainty from the missing fold-1 pair. N = 4 folds is never an inferential sample; fold-level
deltas are descriptive.

## 7. Limitations

One of five pre-registered fold pairs is missing for a non-random, procedural reason; the retained
cohort is 48 of 60 pre-registered focal parents. The interval is conditional on eight fitted
models; between-model variability is described by four fold-level deltas only. Every residual
confound and claim boundary of the parent config is inherited (including the residual parent gap
0–3, ABSENT larger). No confirmatory claim may be made.

## 8. Test-harness safety (retained and proven)

`tests/conftest.py` (autouse, suite-wide) refuses — before any `os.open` — claim creation, exposure
record appends and ledger appends inside the real repository paths. `tests/test_phase34_inc1_recovery.py`
proves: the guard fires before O_EXCL (an `os.open` spy records zero O_EXCL calls) for both the
consumed and an unclaimed real id; `SCORE_STARTED`/`SCORE_COMPLETED` appends and both ledger append
paths are refused with the real files byte-unchanged; isolated roots pass through; no test function
that points at the real results root calls `score_experiment`. The remaining scorer harnesses were
re-pointed from fold-1 PRESENT (now permanently refused) to the first retained look, in `tmp_path`.

## 9. What this commit contains / does not contain

Contains: this document; `configs/phase3/roboticarm_focal_toggle_inc1_recovery_v1.json`; the
recovery gate and salvage-analysis restriction in the execution library; the analysis output path
in the runner; harness tests; `reports/phase3_4_results/inc1_recovery_manifest.json`.
Does not contain: any protected score row, any aggregate performance, any inference output.

## 10. Next step — not executed

Independent review of this freeze. Only then, with explicit authorization, step E: score the eight
retained looks once each in the frozen order `2A 2P 3P 3A 4A 4P 5P 5A`.
