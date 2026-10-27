# Phase 3.4D — Parent-Diversity Feasibility Check and Release Gate

**Role:** skeptical experiment-release statistician.
**Question:** can the focal-parent sibling toggle remove the PRESENT-vs-ABSENT parent-diversity
imbalance **without** destroying the exact matching that makes V2 useful?
**Answer: largely yes.**

**Decision: A — upgrade to near-parent-balanced V3.**
**Release verdict: YELLOW.** · **V3 config:** `p34d-exec-e5470b025229f59a06f4c772`
**Starting HEAD:** `ee05157ca420dfd5822c38e9015c28eaf9ac71eb`

> No model trained. No inference. No test score. No checkpoint. No ledger append. The focal cohort is
> unchanged, so the primary estimand is untouched.

---

## 1. The imbalance, reconfirmed from membership metadata

| Fold | PRESENT parents | ABSENT parents | Gap | Kish ESS gap |
|---:|---:|---:|---:|---:|
| 1 | 74 | 64 | 10 | 7.18 |
| 2 | 70 | 64 | 6 | 4.57 |
| 3 | 73 | 64 | 9 | 5.85 |
| 4 | 74 | 64 | 10 | 7.35 |
| 5 | 75 | 64 | 11 | 6.25 |

Segments, windows and all 25 strata match exactly — the imbalance is purely in parent identities.

**Why V2 donors cannot fix it:** every donor parent already contributes to the common background `B`
through its non-donated segments, so returning its donated segments to ABSENT adds **segments**,
never a **parent identity**. ABSENT is `76 − 12 = 64` regardless of donor choice. Confirmed directly:
of fold 1's 47 donor parents, 45 still appear in `B`.

---

## 2. The parent-exclusive design

`B*` removes **both** the focal parents and the donor parents from the background. Then
`PRESENT = B* + focal` and `ABSENT = B* + donor`, so each arm's added parents are exclusive to it and
the gap becomes `12 − |D|`.

### Exact 12-slot matching is provably impossible

About **1.9 million** candidate donor sets were examined heuristically first; both a randomised and a
deterministic greedy/local search stalled at residual deficits. I then switched to an **exact integer
program** (HiGHS via `scipy.optimize.milp`), which settles the question rather than failing to find:

| Fold | Exact at 12 slots | Min shortfall | Binding strata |
|---:|:--:|---:|---|
| 1 | **no** | 1 | `weight25\|BckgE` |
| 2 | **no** | 4 | `weight25\|BckgE`, `weight30\|BckgB`, `weight30\|BckgE`, `weight35\|BckgG` |
| 3 | yes | 0 | — |
| 4 | **no** | 1 | `weight35\|BckgG` |
| 5 | yes | 0 | — |

**Every binding stratum is target-domain.** Each holds exactly 3 training segments across 2–3
parents; those specific holders must occupy donor slots that must also cover 16 source strata of ~23
segments each, and each parent spans only 3–4 strata. Twelve slots cannot do both.

A slot-relaxation cross-check confirms the diagnosis: folds 1 and 4 become feasible at **13** slots,
fold 2 at **15**. The binding constraint is the slot budget, not missing data.

### Re-partitioning does not rescue it

**69,653** alternative five-fold partitions of the same 60 focal-eligible parents (seed 20260909),
each fold checked by exact ILP: best **4 of 5**, no all-feasible partition. Folds 3 and 5 are
individually feasible, but cherry-picking them after seeing which are convenient was refused — every
fold uses one rule, the minimum slot count admitting an exact match.

---

## 3. The upgrade

Donor slots: **13, 15, 12, 13, 12**.

| | V2 | **V3** | |
|---|---|---|---|
| Parent gap | 10, 6, 9, 10, 11 | **−1, −3, 0, −1, 0** | ✅ |
| Kish ESS gap | 4.57 – 7.35 | **0.31 – 1.98** | ✅ |
| Segments / windows | exact | **exact** | = |
| All 25 strata | exact | **exact** | = |
| Target-domain exposure | equal | **equal** | = |
| Training volume | 79.7 – 87.6% | 76.4 – 85.0% | ▽ |
| Donor parents | 43 – 51 | 12 – 15 | ▽ |

Mean ESS gap falls **6.24 → 0.88**, an ~86% reduction, at **zero metadata cost**. The residual gap
reverses sign: ABSENT now holds 0–3 *more* parents than PRESENT.

### The approximate 12-slot variant was tested and rejected

Forcing 12 slots with minimum shortfall (1, 4, 0, 1, 0) buys parent gap 0 — but the shortfall lands
**entirely on target-domain training**, stripping up to 4 segments (~**16%** of ABSENT's target
exposure in fold 2) from one arm only. The focal normal cohort is **87 target / 116 source**, so that
is a *directional* deficit on **43%** of the primary cohort: it would depress ABSENT's target scores
for a reason unrelated to sibling presence and inflate the measured Δ. That is exactly "fixing parent
count by badly damaging metadata balance", which the gate forbids. Minimal slot relaxation avoids it
completely — which is why it wins.

---

## 4. Bootstrap finalised

**FOLD-STRATIFIED PARENT BOOTSTRAP**: 10,000 replicates, seed **27182**, percentile 95%, resampling
12 parents *within* each fold. Each fold's 12 parents share one fitted model pair, so free resampling
could change the mixture of fitted models and let between-model differences masquerade as
parent-sampling noise; stratifying holds that mixture fixed, coherent with an interval declared
conditional on those models. On synthetic data the stratified interval is slightly narrower
(0.1318 vs 0.1396) with an identical point estimate.

**Conditional on the ten fitted models.** It estimates no training-seed, model-pair or population
uncertainty. The five fold-level estimates are reported separately, and **N = 5 is never an
inferential sample** — fold is a stratification level, never a resampling level.

---

## 5. Unchanged

Primary estimand · focal cohort (60 parents, 203 normal segments, 232-anomaly reference, **identical**
to V2) · seed **1995 only** · **10** runs, **10** test looks · environment
`p34c-env-b7a5de2af4c1d9614e49` · binding score persistence · failure policy · **ledger 30 → 30** ·
V1 remains SECONDARY.

Training order (counterbalanced, unchanged): `1P, 1A, 2A, 2P, 3P, 3A, 4A, 4P, 5P, 5A`.

---

## 6. Why YELLOW

A residual parent gap of 0–3 remains, proven unreachable at 12 slots for three folds and not found in
69,653 alternative partitions; and the interval remains conditional on the ten fitted models. Both
travel with every result.

**Remaining unavoidable limitations:** residual 0–3 parent gap · interval conditional on fitted
models · donor material differs in signal content (metadata-only matching) · 9 sibling-bearing normal
segments outside the focal cohort · 11 of 60 parents contribute a single segment · training volume
~3 points below V2.

---

## 7. Next step — not executed

Execution authorisation. Then COMMIT A, training in the frozen order, COMMIT B (checkpoint hashes, no
performance), then the already-authorised inference. Nothing may be scored before the ledger
authorisation and COMMIT B exist.
