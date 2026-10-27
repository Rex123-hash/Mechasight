# Phase 3.4D — Release Gate Decision

**Decision: A — UPGRADE TO NEAR-PARENT-BALANCED V3.**
**Release verdict: YELLOW.**
**V3 execution config:** `p34d-exec-e5470b025229f59a06f4c772` (supersedes `p34c-exec-0eecc19f81e200f5c301ca25`)

No model trained, no inference, no score, no checkpoint, no ledger append. The focal cohort is
**identical** to V2, so the primary estimand is unchanged.

---

## 1. The imbalance, reconfirmed independently

| | |
|---|---:|
| ABSENT parents, every fold | **64** |
| PRESENT parents | **70–75** |
| Kish effective-parent gap | **4.57 – 7.35** |
| Segments / windows / 25 strata | matched exactly |

**Why V2 donors could never fix it:** every donor parent *already* contributes to the common
background `B` through its non-donated segments. Returning its donated segments to ABSENT adds
segments but never a **parent identity**. ABSENT is therefore `76 − 12 focal = 64` regardless of
which donors are chosen.

---

## 2. The parent-exclusive design

`B* =` official training **minus every segment of every focal parent AND every segment of every
donor parent**; `PRESENT = B* + focal`; `ABSENT = B* + donor`. Because donor parents are removed from
the background too, each arm's added parents are **exclusive to it**, and the parent gap becomes
`12 − |D|` instead of a fixed 12.

### Exact 12-slot matching is impossible — proven, not merely unfound

Roughly **1.9 million** candidate donor sets were examined by randomised and deterministic
greedy/local search before switching to an **exact integer program** (HiGHS via
`scipy.optimize.milp`):

| Fold | Exact at 12 slots | Minimum shortfall | Binding strata |
|---:|:--:|---:|---|
| 1 | **no** | 1 | `weight25\|BckgE` |
| 2 | **no** | 4 | `weight25\|BckgE`, `weight30\|BckgB`, `weight30\|BckgE`, `weight35\|BckgG` |
| 3 | yes | 0 | — |
| 4 | **no** | 1 | `weight35\|BckgG` |
| 5 | yes | 0 | — |

**Every binding stratum is target-domain.** Each holds exactly 3 training segments across 2–3
parents, and those specific holders must occupy donor slots that must *simultaneously* cover 16
source strata of ~23 segments each. With 12 slots and each parent spanning only 3–4 strata, the joint
covering problem is infeasible.

**Cross-check:** relaxing only the slot budget makes folds 1 and 4 feasible at **13** and fold 2 at
**15** — confirming the binding constraint is the slot budget, not missing data.

### Re-partitioning does not rescue it either

**69,653** alternative five-fold partitions of the same 60 focal-eligible parents were tested under
seed 20260909, each fold checked by exact ILP. Best result: **4 of 5** folds feasible. **No
all-feasible partition found.**

Folds 3 and 5 *are* individually feasible at 12 slots — but running a mixture of 12-slot and relaxed
folds selected after seeing which are convenient was **refused**. Every fold uses the same rule: the
minimum slot count admitting an exact match.

---

## 3. What the upgrade achieves

Donor slots per fold: **13, 15, 12, 13, 12**.

| | V2 | **V3** |
|---|---|---|
| Parent gap | 10, 6, 9, 10, 11 | **−1, −3, 0, −1, 0** |
| Kish ESS gap | 7.18, 4.57, 5.85, 7.35, 6.25 | **0.31, 1.98, 0.60, 0.71, 0.81** |
| Segment / window match | exact | **exact** |
| All 25 strata | exact | **exact** |
| Target-domain exposure | equal | **equal** |
| Training volume | 79.7–87.6% | 76.4–85.0% |
| Donor parents | 43–51 | 12–15 |

**The confound this phase existed to address is reduced by ~86%** (mean ESS gap 6.24 → 0.88) at
**zero metadata cost**. The residual gap is now *negative* — ABSENT holds 0–3 **more** parents than
PRESENT — the opposite sign and roughly a ninth the magnitude.

### An approximate 12-slot variant was considered and rejected

Forcing 12 slots with the minimum shortfall (1, 4, 0, 1, 0 = 6 segments) would have achieved parent
gap 0, but the shortfall falls **entirely on target-domain training**, removing up to 4 segments —
about **16%** of ABSENT's target exposure in fold 2 — from one arm only. Since the focal normal
cohort is **87 target / 116 source**, that is a *directional* deficit on 43% of the primary cohort,
depressing ABSENT's target scores for a reason unrelated to sibling presence. Trading an exactly
matched design for that is precisely the "fix parent count by badly damaging metadata balance" the
gate forbids. The minimal-slot-relaxation design avoids it entirely.

---

## 4. Bootstrap finalised

**FOLD-STRATIFIED PARENT BOOTSTRAP** replaces free resampling — see
[`bootstrap_finalization.md`](bootstrap_finalization.md). 10,000 replicates, seed 27182, percentile
95%, resampling 12 parents within each fold. Explicitly **conditional on the ten fitted models**; it
does not estimate training-seed, model-pair or population uncertainty. The five fold-level estimates
are reported separately, and **N = 5 is never an inferential sample**.

---

## 5. Unchanged

Primary estimand (parent-equal-weighted focal concordance delta, MEAN within parent, 60 parents) ·
focal cohort (60 parents, 203 normal segments, 232-anomaly reference) · seed **1995 only** · **10**
runs and **10** test looks · environment `p34c-env-b7a5de2af4c1d9614e49` · score persistence ·
failure policy · ledger 30 → 30 · V1 secondary.

---

## 6. Why YELLOW, not GREEN

A residual parent gap of 0–3 remains, and exact equality is **proven** unreachable at 12 slots for
three folds and was not found in 69,653 alternative partitions. The interval also remains conditional
on the ten fitted models. Both must travel with every result.

**Remaining unavoidable limitations:** residual 0–3 parent gap · interval conditional on the fitted
models · donor material differs in signal content (metadata-only matching) · 9 sibling-bearing
segments outside the focal cohort · 11 of 60 parents contribute one segment · training volume ~3
points below V2.
