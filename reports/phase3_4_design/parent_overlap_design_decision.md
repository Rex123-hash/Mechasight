# Phase 3.4A — Design Decision Record

**Config identity:** `p34a-ra-overlap-13dbc49b2b983d72e957a320`
**Design status:** `PREEXECUTION_DESIGN_ONLY`
**Design verdict:** **YELLOW**

**No model was trained. No test example was scored. No inference was run.**

---

## The finding that drives every decision below

RoboticArm's official split leaves almost nothing that is parent-disjoint.

| | Parents | Train segments | Train windows |
|---|---:|---:|---:|
| Crossing the official split | **62** | 1,625 | 120,250 |
| Train-only (the entire disjoint pool) | **14** | **214** | **15,836** |
| Official training total | 76 | 1,839 | 136,086 |

**Full parent-disjointness removes 88.4% of the training set.**

Worse, the disjoint pool is not a scaled-down copy of the official training set. It is missing whole
regions of the design space:

| Missing from the 14 train-only parents | Detail |
|---|---|
| **All target-domain training data** | 0 of 27 `Normal_Target_Train` segments |
| **4 of 7 operating conditions** | no `weight20`, `weight25`, `weight30`, `weight35` |
| **3 of 7 environments** | no `BckgB`, `BckgE`, `BckgG` |
| **13 of 25 metadata strata** | only 12 strata are reachable at all |

So "remove parent overlap" is not a single intervention on this dataset. Done naively it also
removes the entire target domain, most operating conditions, and 88% of the data at the same time.

---

## Design assessments

### Design A — naive official vs. disjoint: **REJECTED**

The disjoint arm would train on 214 of 1,839 segments with no target-domain data at all. Any
difference would confound parent overlap with an 88.4% data reduction *and* the disappearance of
13 strata. The target-slice AUC would be uninterpretable, since one arm never sees target material.
This design cannot answer the research question.

### Design B — size-matched overlap vs. disjoint: **partially adopted**

Size matching is feasible and unusually clean here: **every RoboticArm training segment contains
exactly 74 windows**, so matching segment counts matches window counts *exactly*, with no
approximation. The ceiling is the disjoint pool: 214 segments per arm.

But size matching alone leaves parent diversity uncontrolled — 214 segments drawn from 14 parents
against 214 drawn from up to 62. Adopted as a component, not as the primary design.

### Design C — matched replacement: **REJECTED as primary**

Replacing each overlapping segment with a metadata-matched non-test segment cannot work here,
because for 13 of 25 strata **no non-test replacement exists at any price**. Replacement would
silently change the metadata distribution in exactly the strata it was meant to preserve. Its useful
idea — per-stratum matching — is absorbed into the chosen design.

### Design D — repeated matched resampling: **adopted, with a disclosed asymmetry**

Replication is the right instinct, but it is only half-available. The disjoint arm's pool is
*exhausted*: all 14 train-only parents and all 214 of their segments are used, so **exactly one
disjoint membership exists**. Replication is possible on the overlap side only.

---

## Chosen primary design

**Parent-count-matched, size-matched, stratum-matched repeated resampling.**

| | Disjoint arm | Overlap arm |
|---|---|---|
| Membership | **FIXED** — the pool is exhausted | 5 pre-declared replicates |
| Segments | 214 | 214 (exact) |
| Windows | 15,836 | 15,836 (exact) |
| Parents | 14 | 14 (exact) |
| Per-stratum counts | reference | identical, exactly |
| Parent overlaps official test | no | yes |

Adding **parent-count matching** is what makes this stronger than Design B: without it the arms
would differ in the number of independent physical clusters as well as in overlap status.

**Construction rule (frozen).** Draw 14 crossing parents uniformly without replacement under
`numpy.default_rng(20260908)`; reject the draw unless those parents can supply the disjoint arm's
exact per-stratum counts; then draw the required number of segments per stratum from candidates
sorted by `(parent, segment_id)`. Five distinct replicates were obtained in 19 attempts.

**All five replicates verified:** 214/214 segments, 15,836/15,836 windows, 14/14 parents, and all
12 strata matching exactly.

**Runs:** 3 seeds `(1995, 20260908, 31415)` × (1 disjoint + 5 overlap memberships) = **18 runs**,
about 26 min each on CPU (each arm trains on 11.6% of the official volume), ≈ **7.8 h** total.

---

## Why the verdict is YELLOW and not GREEN

A genuinely matched contrast *is* constructible — segments, windows, parents and strata all match
exactly — so this is not RED. But three confounds are structural and no design can remove them:

1. **The disjoint arm has exactly one possible membership.** Overlap status is inseparable from the
   identity of those particular 14 parents, and they were not randomly chosen — they are train-only
   because of how the official split was built.
2. **Train-only parents are systematically shorter recordings** — median 8.5 segments per parent
   versus 33.0 for crossing parents. They may differ in recording character, not just overlap status.
3. **Both arms exclude 13 of 25 strata**, including every target-domain training segment. Neither
   arm reproduces the official baseline's training distribution.

Consequence, and it must travel with every future number: **Δ describes this restricted,
downsampled, source-domain-only regime. It does not estimate the effect on the published result.**

---

## What this design will and will not license

**Permitted:** *"Under the frozen official-baseline model and preprocessing and the matched training
design, parent-overlap presence was associated with a Δ difference in measured RoboticArm AUROC."*

**Forbidden:** that the published paper is invalid; that the published AUC is inflated by exactly Δ;
that Δ is the general real-world leakage penalty; that Δ applies to BrushlessMotor or to any other
model family; that this is causal proof of leakage inflation.

**A Δ near zero is a result**, and is to be reported with the same prominence as a large Δ.

---

## Design choices recorded for the record

- **Validation split:** upstream window-level behaviour retained **identically in both arms**.
  Changing it would alter more than parent train/test overlap. Its cost — windows from one parent in
  both train and validation — applies equally to both arms and is documented as a limitation. A
  parent-disjoint validation variant is a *separate* secondary experiment, never folded into the
  primary contrast.
- **Seeds:** declared before execution, applied identically to both arms. No seed may be chosen,
  discarded, or reweighted on the basis of observed performance.
- **Device:** one environment frozen before execution and identical across arms. A CPU arm is never
  compared against a CUDA arm. Default CPU (torch 2.14.0+cpu) unless a verified CUDA environment is
  frozen beforehand.
- **Uncertainty:** class-stratified parent-cluster bootstrap on the fixed official test cohort, with
  identical parent draws across arms, paired contrasts, and all sibling segments moving together.
  Windows are never resampled. Membership variability is estimable for the overlap arm only, and
  that asymmetry is reported rather than hidden.
- **Score persistence is binding.** Phase 3.3's audit gap must not recur: per-segment scores are
  written atomically with the evaluation output, before aggregates are finalised.
- **Ledger:** new protocol `roboticarm_parent_overlap_sensitivity_v1`. The existing 30 entries are
  untouched, and **nothing was appended in this phase**.
