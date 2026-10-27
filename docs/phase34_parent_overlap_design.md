# Phase 3.4A — RoboticArm Parent-Overlap Sensitivity Experiment: Design and Preregistration

**Status:** `PREEXECUTION_DESIGN_ONLY` — design frozen, nothing executed.
**Config:** `configs/phase3/roboticarm_parent_overlap_sensitivity_v1.json`
**Config identity:** `p34a-ra-overlap-13dbc49b2b983d72e957a320`
**Starting HEAD:** `98a20c0755d7967d636ab206dbb75aac7faf10c6`
**Design verdict:** **YELLOW**

> **No model was trained. No test example was scored. No inference was run. No ledger entry was
> added.** Everything below was derived from segment metadata and split membership alone.

---

## 1. Scientific classification — permanent

**POST-TEST CONTROLLED SENSITIVITY ANALYSIS.**

This design was created *after* Phase-3.3 Protocol-A performance was observed (RoboticArm 91.51
overall). The overlap risk itself was documented much earlier, but the experimental design is
post-test, and that distinction is preserved permanently in the config and in every artifact.

It is therefore **not**: untouched confirmatory evidence; preregistered-before-all-results evidence;
independent validation; or causal proof of leakage inflation.

---

## 2. Research question

> Under the official RoboticArm preprocessing, architecture, hyperparameters and fixed official test
> cohort, how much does measured anomaly-discrimination performance change when training examples
> from the same physical parent recordings as test examples are excluded?

Deliberately **not** shortened to "how much did leakage inflate the paper" — that framing presumes
the sign, the magnitude, and the causal transfer.

---

## 3. Structure re-verified from the data

Recomputed from the windowed HDF5 files, not taken from the prompt:

| Quantity | Value |
|---|---:|
| Parent recordings (union) | **101** |
| Parents crossing the official split | **62** |
| Normal test segments with a training sibling | **212 / 232** |
| Anomaly test segments with a normal training sibling | **0 / 232** |
| Official test segments | **464** |

All five prompt values confirmed. The overlap is **asymmetric toward the normal class** — which is
precisely the structure that would bias a normal-vs-anomaly discrimination metric optimistically.

---

## 4. The fundamental design problem, stated honestly

Removing overlapping-parent training segments **necessarily changes the training dataset**. It is
therefore impossible to vary parent overlap in a literal laboratory sense while holding every
training observation fixed. The design must separate:

- **A. the parent-overlap factor** — what we want to measure, from
- **B. training-sample composition and size** — which moves with it.

The whole design below is an attempt to pin B as tightly as the dataset permits. It does not achieve
perfect causal identification, and this document does not claim it does.

---

## 5. Feasibility — why this dataset makes it hard

| | Parents | Segments | Windows |
|---|---:|---:|---:|
| Crossing parents | 62 | 1,625 | 120,250 |
| **Train-only parents (entire disjoint pool)** | **14** | **214** | **15,836** |
| Official training total | 76 | 1,839 | 136,086 |

**Full parent-disjointness removes 1,625 segments and 120,250 windows — 88.4% of training.**

And the remainder is not representative:

| Coverage of the 14 train-only parents | |
|---|---|
| `Normal_Target_Train` segments | **0 of 27** |
| Operating conditions | 3 of 7 — no `weight20/25/30/35` |
| Environments | 4 of 7 — no `BckgB/E/G` |
| Metadata strata | **12 of 25** |

A fully parent-disjoint RoboticArm model **cannot see any target-domain training data at all**.

One piece of good luck: **every RoboticArm training segment contains exactly 74 windows**, so
matching segment counts matches window counts exactly, with no approximation anywhere.

---

## 6. Candidate designs

| | Verdict | Reason |
|---|---|---|
| **A** naive official vs. disjoint | **REJECTED** | confounds overlap with an 88.4% data cut *and* the loss of 13 strata including the whole target domain; the target-slice AUC would be meaningless |
| **B** size-matched | **partially adopted** | feasible and exact, but leaves parent diversity uncontrolled (14 parents vs. up to 62) |
| **C** matched replacement | **REJECTED as primary** | for 13 of 25 strata **no non-test replacement exists at any price**; its per-stratum matching idea is absorbed into the chosen design |
| **D** repeated matched resampling | **adopted, with a disclosed asymmetry** | replication is only available on the overlap side — see §7 |

Selection was made on identifiability, balance, available parent counts, matching feasibility,
statistical validity and computational practicality. **Not** on which design would show a larger
effect, and not with reference to the 91.51 result.

---

## 7. Chosen primary design

**Parent-count-matched, size-matched, stratum-matched repeated resampling.**

| | Disjoint arm | Overlap arm |
|---|---|---|
| Membership | **FIXED** — pool exhausted, exactly one exists | 5 pre-declared replicates |
| Segments | 214 | 214 (exact) |
| Windows | 15,836 | 15,836 (exact) |
| Parents | 14 | 14 (exact) |
| Per-stratum counts | reference | identical, exactly |
| Overlaps official test | no | yes |

Adding **parent-count matching** is the step that makes this stronger than plain size matching:
without it the arms would differ in the number of independent physical clusters, not only in overlap
status.

**Frozen construction rule.** Draw 14 crossing parents uniformly without replacement under
`numpy.default_rng(20260908)`; reject unless they can supply the disjoint arm's exact per-stratum
counts; then draw the required count per stratum from candidates sorted by `(parent, segment_id)`.
Five distinct replicates were obtained in 19 attempts and **all five verified** on segments, windows,
parents and every stratum.

Memberships are frozen in
[`proposed_arm_memberships.csv`](../reports/phase3_4_design/proposed_arm_memberships.csv) — 1,284
rows of training identities and metadata, **no scores**.

**The asymmetry, stated plainly:** the disjoint pool is exhausted, so the disjoint arm has exactly
one possible membership. Membership variability is estimable for the overlap arm only. Its spread
reflects seed variation alone.

---

## 8. Frozen components

Unchanged across every arm: dataset release; the **exact** official 464-segment RoboticArm test
cohort (hash `83cd1195…6a7a8efa`); Protocol-A preprocessing (16000/7063/7063 Hz, 100 ms,
right-inclusive slicing, zero pad/truncate, 35 ms gyroscope warm-up, `std_window`); raw standardised
5836-dimensional input; the official autoencoder `[2048,2048,2048,16]` with BatchNorm placement,
mirrored decoder and reproduced checkpoint aliasing; Adam / 1e-4 / MSE / batch 1024 / 1000 epochs /
patience 3 / valid_size 0.1; segment-median aggregation and upstream strict pairwise AUC; reporting
overall, source and target.

**No Protocol-B model, feature, split or threshold may enter this experiment.**

---

## 9. Experimental units

| Role | Unit |
|---|---|
| Training observation | 100 ms window |
| Segment | released segment (exactly 74 windows) |
| Physical cluster | original continuous parent recording |
| Evaluation unit | segment score |
| Independence / resampling unit | **parent recording** |

Training operates on windows as upstream does. **Inference never treats windows as independent.**

---

## 10. Estimands

**Primary — "overlap-associated AUC difference under the controlled matched design":**

```
Δ_AUROC = mean over (replicate, seed) AUROC_overlap  −  mean over seed AUROC_disjoint
```

on the fixed official 464-segment cohort, upstream strict pairwise AUC, segment-median aggregation.
A positive Δ means the overlap-containing arm measured higher. **This is not to be called "true
leakage inflation."**

**Secondary:** Δ source AUROC; Δ target AUROC; differences in the normal-test and anomaly-test score
distributions.

**Descriptive only:** normal test segments *with* a training sibling versus *without*. This is
observational — parent identity and operating conditions differ — and **must never** serve as the
primary causal estimate.

---

## 11. Validation split, seeds, device

**Validation:** upstream window-level split retained **identically in both arms**. Changing it would
alter more than parent train/test overlap and break comparability with the Phase-3.3 anchor.
Documented limitation: windows from one parent appear in both training and validation within every
arm, so validation loss is optimistic and early stopping is not clean model selection — equally in
both arms. A parent-disjoint validation variant is a **separate** secondary sensitivity experiment,
never folded into the primary contrast.

**Seeds:** `1995, 20260908, 31415`, declared here, applied identically to both arms. No seed may be
chosen, discarded or reweighted on the basis of observed performance; aggregation is exactly as
preregistered. Multi-seed is preferred because Phase 3.3 was one-run-only and could not separate
training stochasticity from anything else.

**Device:** one environment frozen before execution — torch version, device, backend,
deterministic-algorithm policy, seeds — identical across all arms. **A CPU arm is never compared
against a CUDA arm.** Default CPU (torch 2.14.0+cpu) unless a verified CUDA environment is frozen
first. Compute did not drive the design: 18 runs × ~26 min ≈ **7.8 h** on CPU, each arm training on
11.6% of the official volume.

---

## 12. Uncertainty

Class-stratified **parent-cluster** bootstrap over the fixed official test cohort:

- identical parent bootstrap draws across all arms;
- paired contrasts within each draw;
- all sibling segments of a parent move together;
- windows are never resampled — no pseudo-replication;
- training-set and seed variation reported **separately** from test-cohort resampling variation,
  never pooled into one interval implying more independent replication than exists.

---

## 13. Score persistence — binding requirement

Phase 3.3 could not be independently re-audited because per-segment scores were never written. That
must not recur. For **every test segment** the future evaluator must persist: `experiment_id`,
`machine`, `arm`, `replicate`, `seed`, `segment_id`, `parent_id`, `split_label`, `anomaly_label`,
`domain_shift_op`, `domain_shift_env`, `overlap_status`, `total_loss`, every primary anomaly-score
column, score provenance and checkpoint identity — **written before aggregates are finalised,
atomically with the evaluation output.**

---

## 14. Test-look policy

New protocol identity **`roboticarm_parent_overlap_sensitivity_v1`**. The Protocol-A value
`official_baseline_reproduction` must **not** be reused. A ledger entry is appended before each
first test look under the new protocol. The existing **30** entries are untouched and **nothing was
appended in this phase**.

---

## 15. Claim boundary

**Permitted:** *"Under the frozen official-baseline model and preprocessing and the matched training
design, parent-overlap presence was associated with a Δ difference in measured RoboticArm AUROC."*

**Forbidden:** the published paper is invalid; the published AUC is inflated by exactly Δ; Δ is the
general real-world leakage penalty; Δ applies to BrushlessMotor; Δ applies to other model families.

**Mandatory qualifier travelling with every future number:** both arms train on 11.6% of the
official training volume, on a source-domain-only, 12-stratum restricted subset. **Δ describes that
restricted regime, not the official baseline's training conditions.**

---

## 16. Design verdict: **YELLOW**

Not RED: a genuinely matched contrast is constructible, and segments, windows, parents and strata
match *exactly* — the factor being varied really is parent overlap, within the matched population.

Not GREEN, because three residual confounds are structural:

1. **The disjoint arm has exactly one possible membership.** Overlap status is inseparable from the
   identity of those 14 parents, which were not randomly chosen — they are train-only because of how
   the official split was constructed.
2. **Train-only parents are systematically shorter** — median 8.5 segments per parent versus 33.0
   for crossing parents — so they may differ in recording character.
3. **Both arms exclude 13 of 25 strata**, including all target-domain training data, so neither arm
   reproduces the official training distribution.

---

## 17. Unresolved design risks

- Δ may partly reflect *which* 14 parents are train-only rather than overlap status as such; this is
  not separable with this dataset.
- The target-slice secondary estimand is weak: **neither** arm trains on target-domain data, so
  target AUC measures generalisation to an unseen domain in both, not the overlap effect on the
  official target result.
- 11.6% training volume may put both arms in a different performance regime, where any overlap
  effect could be larger or smaller than at full scale. The design cannot detect that.
- With one fixed disjoint membership, a per-arm confidence statement about membership variability is
  unavailable on that side.
- The 12 retained strata are all `Normal_Source_Train`; the contrast is source-domain-only.

---

## 18. Next step — not executed

Independent design review. Execution is gated on that review, on freezing the device environment,
and on appending the first `roboticarm_parent_overlap_sensitivity_v1` ledger entries **before** any
scoring.
