# Phase 3.4B — Independent Design Review and Focal-Parent Toggle Design (v2)

**Role:** skeptical experimental-design statistician.
**Status:** `PREEXECUTION_DESIGN_ONLY`.
**Review verdict:** **B — keep V1 as secondary, adopt Design E as primary.**
**V2 config:** `configs/phase3/roboticarm_parent_overlap_sensitivity_v2.json`
**V2 config identity:** `p34b-ra-focal-2ab15394f78b560f1e8c0880`
**Starting HEAD:** `6770750e3f19b9cabcfc0ae8d9ae8e8d31d59bd0`

> **No model was trained. No inference was run. No test example was scored. No checkpoint was
> loaded. No ledger entry was added. Phase 3.4A was not executed and its config was not edited.**

---

## 1. V1 reconstructed independently

Every V1 figure was recomputed from the windowed HDF5 files rather than read from the design report,
and all of it confirms:

| | Value |
|---|---:|
| Official training | 1,839 segments / 136,086 windows / 76 parents |
| Crossing | 62 parents / 1,625 segments / 120,250 windows |
| Train-only | 14 parents / 214 segments / 15,836 windows |
| Reduction under full disjointness | **88.36%** |
| Target-domain segments in the train-only pool | **0** |
| Strata reachable by the train-only pool | **12 of 25** |
| Missing operating conditions | `weight20`, `weight25`, `weight30`, `weight35` |
| Missing environments | `BckgB`, `BckgE`, `BckgG` |

All five proposed overlap memberships verified: 214 segments, 15,836 windows, 14 parents, exact
per-stratum counts. **V1's stated matching claims are all exact.**

---

## 2. V1 scientific strengths

Worth stating plainly before the criticism: V1 is a competent design. Segment, window, parent-count
and per-stratum matching are exact, not approximate — helped by every RoboticArm training segment
holding exactly 74 windows. The construction rule is frozen and reproducible. Its verdict was
honestly set at YELLOW, and its residual confounds were disclosed rather than buried.

---

## 3. V1 structural limitations

Two the V1 freeze already recorded — the single possible disjoint membership, and the exclusion of
13 strata including all target-domain training. Two it did not.

### 3.1 Parent-contribution weight is not matched — **MATERIAL**

V1 matches *nominal* parent count. It does not match contribution weight.

| Arm | Sorted contribution vector | Gini | Top-4 | **Kish ESS** |
|---|---|---:|---:|---:|
| **disjoint** | `48, 36, 36, 36, 12, 12, 9, 8, 8, 3, 2, 2, 1, 1` | 0.533 | **72.9%** | **6.83** |
| overlap r1–r5 | see `parent_contribution_balance.csv` | 0.279–0.458 | 44.4–59.8% | **8.35–11.26** |

Four parents supply 72.9% of the disjoint arm. Because windows within a parent are highly
correlated, the arms differ in **effective sample size** — roughly a third fewer effective clusters
on the disjoint side — as well as in overlap status. A few long crossing parents do *not* dominate
the overlap arm; the domination is on the **disjoint** side, which is the opposite of the intuition
worth checking.

**Exact contribution matching is impossible.** The disjoint arm's largest parent contributes 48
segments; no crossing parent can supply more than 47 within the 12 reachable strata. Minimum
unavoidable imbalance: **1 segment**, and closing the rest would need a constrained
parent-by-stratum transportation solve — complexity that would not touch the deeper problems.

### 3.2 The primary estimand mis-states its dependence structure — **MATERIAL**

15 overlap runs against 3 disjoint runs are **not** 15 independent matched experiments: they share
three seeds and one fixed comparator. Membership × seed cells must never be pooled as N = 18.

**Recommended, and adopted:** `Δ_s = mean_r AUROC_overlap(r,s) − AUROC_disjoint(s)` per seed, then
`mean_s Δ_s`, with membership spread, seed spread and the parent-cluster bootstrap reported as three
separate components. This is preferable to V1's definition and does not require re-running anything.

---

## 4. Design E — focal-parent sibling toggle

> For normal test segments from a given physical parent, how does the anomaly score change when
> training siblings from that **same** parent are present versus absent, holding the rest of the
> training set identical?

Deliberately narrower than "global parent-disjoint training", and that narrowness is what makes it
identifiable.

Per fold: `B` = official training − focal contribution − donor reserve, **identical between arms**;
**PRESENT** = `B` + focal segments; **ABSENT** = `B` + matched donors from non-focal parents.

**The key distinction:** the ABSENT arm need not be globally parent-disjoint. It need only be
disjoint from the **focal** parents. Other training parents may keep their own test siblings. That
is legitimate for this narrower estimand — the common background is identical between arms, donor
membership is frozen before any performance is seen, and evaluation is restricted to focal normals.
It is also precisely what buys back the training volume and the target domain.

### Feasibility

| | |
|---|---:|
| Crossing parents | 62 |
| Irreplaceable even alone | **2** (shortfall exactly 1 segment each, both target strata) |
| Focal-eligible | **60** |
| Focal normal test segments covered | **203 of 232** |
| Fixed anomaly reference | 232 |
| Exact K = 5 partition | **found in 25 attempts** |

| Fold | Focal parents | Focal = donor segs | Match | Arm training segs | % official | Focal normals | Target focal→donor |
|---:|---:|---:|:--:|---:|---:|---:|:--:|
| 1 | 12 | 372 | exact | 1,467 | 79.8% | 45 | 3 → 3 |
| 2 | 12 | 374 | exact | 1,465 | 79.7% | 40 | 6 → 6 |
| 3 | 12 | 283 | exact | 1,556 | 84.6% | 46 | 6 → 6 |
| 4 | 12 | 363 | exact | 1,476 | 80.3% | 34 | 3 → 3 |
| 5 | 12 | 228 | exact | 1,611 | 87.6% | 38 | 4 → 4 |

"Exact" means segments, windows and **all 25 strata** simultaneously, with donor parents disjoint
from focal parents in every fold.

### Target-domain preservation

Design E does **not** solve this automatically — it solves it because target-domain focal
contributions can be replaced by target-domain donors from other parents. Every fold replaces its
target segments **1:1**. Both arms retain target-domain training and all **25** strata, against
V1's zero and 12.

The two excluded parents are excluded for exactly this reason: each holds 2 of the only 3 training
segments in a target stratum, so its contribution cannot be exactly replaced. Excluding them buys
exact matching everywhere at a cost of 9 focal normal test segments.

---

## 5. Estimands

**Primary — focal same-parent sibling-exposure score shift.** `d = score_PRESENT − score_ABSENT`
per focal normal test segment; parent-level median; summarised across focal parents. **Most
identifiable:** the same test segment scored under two models differing only by the toggle, so
segment-level variance cancels. Negative ⇒ sibling exposure made focal normals look more normal.

**Secondary — focal AUROC against a fixed anomaly reference** (all 232 anomaly test segments,
identical across arms). Meaningful, but weaker: the model changes between arms, so anomaly scores
move too. Reported because AUC is the unit the literature uses.

**Specificity check:** anomaly-side shift on the fixed reference — a large shift there would mean
the toggle moved the model globally rather than through same-parent familiarity.

**Uncertainty:** paired parent-cluster bootstrap over focal parents; PRESENT/ABSENT never resampled
independently; folds parent-disjoint; windows never resampled; seed variation reported separately.

---

## 6. Comparison, and which design better isolates same-parent exposure

Design E wins **11 of 12** dimensions (full table in `v1_vs_v2_design_comparison.csv`); V1 wins only
compute.

**Design E isolates same-parent exposure decisively better.** It holds parent identity fixed — each
focal parent is its own comparison — holds training size and all 25 strata exact, preserves the
target domain, and pairs *within segment*. V1 confounds overlap status with parent identity, an
88.4% volume cut, 13 missing strata, and a materially different effective cluster count.

**Neither is causal.** Design E's residual: donor segments are different physical material from the
focal segments they replace, so the toggle changes which recordings are seen as well as whether the
focal parent's own siblings are seen. Matching is on pre-test metadata only.

**Compute:** Design E is **substantially more expensive** — 10 runs × 3.07 h ≈ **31 h** vs V1's
7.8 h, about 4×. Cost was not permitted to select the design. Within a fold PRESENT and ABSENT share
the identical seed, so the contrast is seed-controlled by construction; a second seed (`20260908`)
is preregistered as an optional extension.

---

## 7. Claim boundary

**Permitted:** *"For focal normal RoboticArm parents, the presence of same-parent training siblings
was associated with a change of Δ in anomaly scores under the frozen official-baseline pipeline"*,
or the equivalent in focal-AUROC units.

**Forbidden:** the official AUC is inflated by Δ; global leakage penalty; the published paper is
invalid; Δ applies to BrushlessMotor or other model families; causal proof.

**Measures SAME-PARENT FAMILIARITY FOR FOCAL TEST PARENTS, not GLOBAL SPLIT LEAKAGE.**

---

## 8. Decisions

- **Primary:** v2 focal-parent sibling toggle, `p34b-ra-focal-2ab15394f78b560f1e8c0880`.
- **Secondary:** V1 retained unchanged as the global matched contrast, executed only if the primary
  completes and capacity allows — and if executed, summarised with the corrected two-stage estimand
  from §3.2, not V1's original pooled definition.
- **V1 config: not edited.** It stands as frozen evidence of the design evolution.
- **New protocol identity:** `roboticarm_focal_parent_sibling_toggle_v1`. Neither
  `official_baseline_reproduction` nor `roboticarm_parent_overlap_sensitivity_v1` may be reused.
- **Score persistence is binding**, extended with `fold`, `parent_is_focal` and
  `sibling_presence_status`.

---

## 9. Unresolved design risks

- Donor material differs in signal content from what it replaces; matching is metadata-only.
- 9 normal test segments with siblings fall outside the focal cohort (203 of 212 covered).
- Background composition shifts across folds because the donor reserve is withheld from `B` —
  identical between arms within a fold, but not across folds.
- The estimand is narrower than a global leakage figure and must not be reported as one.
- 31 h of CPU is a real constraint; if a verified CUDA environment is frozen first it should be used
  for **all** arms or none.

---

## 10. Next step — not executed

Review of this review. Execution remains gated on that, on freezing the device environment, and on
appending the first `roboticarm_focal_parent_sibling_toggle_v1` ledger entries **before** any
scoring.
