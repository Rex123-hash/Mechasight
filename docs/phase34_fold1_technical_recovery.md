# Phase 3.4 — Fold-1 Zero-Exposure Technical Recovery Amendment (v1)

**Classification:** technical recovery of a verified zero-exposure execution failure — not
retraining, not tuning, not a new experiment.
**Amendment:** `configs/phase3/roboticarm_focal_toggle_fold1_recovery_amendment_v1.json`
(`fold1-technical-recovery-v1`, status `FROZEN_BEFORE_RECOVERY_SCORING`)
**Safety point:** tag `phase3-pre-fold1-recovery` = `a8b3d6c` · branch `phase3-fold1-technical-recovery`
**Supersedes** (by explicit versioned decision) the `NO RETRY` / `DO NOT SCORE` decisions of
`p34r-inc1-4f86592159eadbbe276d9c6f`; that config, its four-fold result and its commit remain in
history unchanged.

---

## 1. Decision context — stated plainly

- This decision was made **after** the four-fold result was known:
  Δ = +0.00844, 95 % CI [−0.00141, +0.02038], commit `a8b3d6c`,
  artifact SHA-256 `6441ed66…1384bd`. That result **remains frozen and is reported unchanged**;
  it is never overwritten or retrospectively replaced.
- The recovery is **not** justified by the direction of that result. It is justified by the
  nature of INC-1: a test-harness execution failure that consumed the single-use claim
  **before any protected-data access** — zero forwards, zero score rows, no `SCORE_COMPLETED`,
  no artifact, no test performance observed. INC-1 produced no scientific model result.
- Both fold-1 checkpoints are the exact frozen COMMIT B objects (hash-verified on disk and in
  read-only backups): PRESENT `85386f2df404…`, ABSENT `4b4d848b74d8…`.
- Both recovered fold-1 results will be accepted regardless of value, and the recovered five-fold
  result will be reported regardless of whether it strengthens, weakens, reverses or leaves
  unchanged the four-fold conclusion. No selection between the two analyses is permitted.

## 2. INC-1 is preserved exactly

The original claim `score_claims/p34d-92fd40fc93c8c8d2e819.claim` (2026-09-11T19:50:25.721Z),
its `SCORE_STARTED` record and their timestamps are never edited, deleted, completed or reused.
They stay in the original namespace as provenance. **INC-1 was an execution/harness failure
before protected inference. It remains part of the provenance.** A test asserts byte-identity
to `576b8a1`.

## 3. Versioned recovery execution identity

| Order | Experiment | Arm | Recovery id | Original state |
|--:|---|---|---|---|
| 1 | `p34d-92fd40fc93c8c8d2e819` | PRESENT | `p34r-f1rec-78c8f486fdb7c822af87` | INC-1: claim + `[SCORE_STARTED]`, no artifact |
| 2 | `p34d-e56529a2f56f98fd4a7a` | ABSENT | `p34r-f1rec-e83624b2152b673c30c5` | never claimed |

Recovery ids derive from `{amendment, experiment_id, arm, checkpoint_sha256, original-state
binding}`. Namespace: `<experiment_id>__<recovery_id>.{claim,jsonl,csv}` — the original
`<experiment_id>` namespace is never reused. Each recovery id permits exactly one protected
evaluation; the claim and `SCORE_STARTED` records bind the original experiment id, recovery id,
fold, arm, checkpoint SHA-256, evaluator identity (`git_head` at claim time), protected cohort
hash, the original INC-1 claim identity, the zero-forward evidence, and the amendment's commit
SHA and content hash.

## 4. What changed in code — control only

Single-source scorer preserved. Against the GREEN evaluator at `64694e6`:

- `_run_protected_inference` (cohort load, checkpoint load, model construction, preprocessing,
  forward, per-window loss, segment aggregation): **byte-identical**.
- `score_experiment`: **two inserted lines** (`**_recovery_binding(exp)` in the claim and
  `SCORE_STARTED` dicts); everything from the inference call to the end — 25-column row
  assembly, atomic CSV persistence, hashing, `SCORE_COMPLETED` — **identical**.
- `acquire_score_claim`, `_append_durable`, `atomic_write_*`, `_ledger_authorisation`,
  `analyse_from_persisted_scores`, `load_persisted_scores`: identical. `focal_concordance.py`,
  `provenance.py`, `ledger.py`, `src/reproduction/`: unchanged.
- Namespace helpers (`score_path`, `score_attempt_path`, `score_claim_path`) route through
  `_artifact_stem`. New: `RecoveryExperiment`, `recovery_experiments`, `_technical_recovery_gate`
  (dispatched from `_recovery_gate` before any claim), `_recovery_binding`,
  `analyse_five_fold_recovered` (new function; the four-fold function untouched). Runner gains
  `--recovery-id` and `--five-fold-recovered`.

`_technical_recovery_gate` requires: amendment frozen and committed at HEAD byte-for-byte; the
recovery id listed for this experiment/arm/checkpoint; the original namespace in exactly its
documented state (PRESENT: INC-1 claim with `[SCORE_STARTED]`, no artifact; ABSENT: unclaimed);
frozen order 1P → 1A; at most one evaluation per recovery id.

## 5. Execution rules

- Order `1P` then `1A`; structural/provenance checks only between looks; no performance inspection.
- Code freeze: no source change after the amendment commit until both evaluations complete.
- **Failure before protected-data access** (setup) may be corrected while preserving the same
  frozen experimental object; every failed attempt remains logged.
- **Failure after protected inference begins:** STOP, preserve the failed attempt, no immediate
  retry, independent review.
- A completed result is final regardless of whether it looks good or bad.

## 6. Recovered five-fold analysis (frozen)

Inputs: the two fold-1 recovery artifacts plus the eight frozen E1 artifacts (folds 2–5,
unchanged, hash-verified). 5 pairs, 60 focal parents (12 per fold), 232-segment anomaly
reference. Original estimator: `C_i(a)`, `delta_i = C_i(PRESENT) − C_i(ABSENT)`, parent mean over
focal normal segments, equal-parent mean over 60. Original bootstrap: fold-stratified over folds
1–5, 12 parents with replacement per fold, 60 per replicate, 10,000 replicates, seed 27182,
percentile 95 %. Output `phase3_4_five_fold_recovered_analysis.json` (+ `.sha256`). The
four-fold artifact is not touched. Both results are reported side by side.

## 7. Test-isolation audit (pre-scoring)

Full suite 458 passed. Static audit (`test_no_test_can_reach_protected_scoring_outside_an_isolated_root`):
no test invokes the runner scoring stage or spawns the runner; every scorer-calling test function
is confined to a temporary root; the only child-process scorer (`_race_worker`) now refuses real
report paths (tested). Suite-wide `conftest.py` guard retained. After the suite, all 58 tracked
protected files were byte-equal to tag `a8b3d6c`; no recovery claim/attempt/artifact existed;
ledger unchanged.
