# Phase 3.4B — Design Review Decision

**Review verdict: B — keep V1 as secondary, adopt Design E (focal-parent sibling toggle) as primary.**

**V2 config identity:** `p34b-ra-focal-2ab15394f78b560f1e8c0880`
**Status:** `PREEXECUTION_DESIGN_ONLY` — nothing trained, nothing scored.
**V1 config was not edited.** It remains frozen as evidence of the design evolution.

---

## 1. What the V1 audit found

Every V1 fact reconstructed independently and confirmed: 1,839 / 136,086 / 76 official;
62 crossing parents holding 1,625 segments and 120,250 windows; 14 train-only parents holding 214
and 15,836; 88.36% reduction under full disjointness; zero target-domain segments in the train-only
pool; 12 of 25 strata reachable; missing `weight20/25/30/35` and `BckgB/E/G`. All five proposed
overlap memberships verified exact on segments, windows, parent count and every stratum.

Then two findings V1 did not record.

### Finding 1 — parent-contribution weight is **not** matched (MATERIAL)

V1 matches *nominal* parent count, 14 = 14. It does not match how much each parent contributes.

| Arm | Contribution vector (segments per parent, descending) | Gini | Top-4 share | **Kish effective parents** |
|---|---|---:|---:|---:|
| **disjoint** | `48, 36, 36, 36, 12, 12, 9, 8, 8, 3, 2, 2, 1, 1` | 0.533 | **72.9%** | **6.83** |
| overlap r1 | `35, 34, 30, 29, 22, 22, 15, 11, 4, 3, 3, 3, 2, 1` | 0.458 | 59.8% | 8.35 |
| overlap r2 | `29, 29, 26, 22, 20, 18, 17, 15, 14, 8, 6, 5, 3, 2` | 0.338 | 49.5% | 10.38 |
| overlap r3 | `26, 25, 23, 22, 22, 19, 18, 16, 16, 11, 6, 6, 2, 2` | 0.299 | 44.9% | 10.91 |
| overlap r4 | `34, 28, 26, 25, 22, 21, 17, 16, 9, 5, 3, 3, 3, 2` | 0.392 | 52.8% | 9.45 |
| overlap r5 | `27, 24, 23, 21, 19, 19, 19, 16, 12, 11, 11, 7, 4, 1` | 0.279 | 44.4% | 11.26 |

Windows within a parent are highly correlated, so the effective number of independent clusters is
what matters. The disjoint arm carries **6.83** against **8.35–11.26** — roughly a third fewer.
Four parents supply **72.9%** of the disjoint arm. **The arms differ in effective sample size as
well as in overlap status**, and V1's Δ would absorb that difference.

**Assessment: MATERIAL.**

### Feasibility of fixing it within V1

**Exact contribution-profile matching is impossible.** The disjoint arm's largest parent contributes
**48** segments; no crossing parent can supply more than **47** within the 12 reachable strata.
Minimum unavoidable imbalance is 1 segment on the top parent, and near-exact matching would require
solving a constrained parent-by-stratum transportation problem on top of everything else — added
complexity on a design whose deeper problems it would not touch.

### Finding 2 — the primary estimand mis-states its dependence structure (MATERIAL)

V1 defines Δ over 15 overlap runs and 3 disjoint runs. **These are not 15 independent matched
experiments.** They share three seeds and a *single fixed comparator*, so membership × seed cells
are dependent and must never be pooled as N = 18.

**Recommended two-stage summary — preferable, and adopted:**

```
for each seed s:   Δ_s = mean_r AUROC_overlap(r, s) − AUROC_disjoint(s)
point estimate:    mean_s Δ_s
```

with **three variance components reported separately**: overlap-membership spread, seed spread, and
the parent-cluster test-cohort bootstrap. This respects the seed pairing and stops the design
claiming replication it does not have.

---

## 2. Design E — the focal-parent sibling toggle

> For normal test segments from a given physical parent, how does the anomaly score change when
> training siblings from that **same** parent are present versus absent, holding the rest of the
> training set identical?

**Construction, per fold k:**

- `B` = official training − focal contribution − donor reserve (**identical between arms**)
- **PRESENT** = `B` + focal-parent training segments
- **ABSENT** = `B` + matched donor segments from non-focal parents

The ABSENT arm does **not** need to be globally parent-disjoint. It only needs to be disjoint from
the **focal** parents. Other training parents may keep their own test siblings — that does not
invalidate this narrower estimand, and it is exactly what buys back the training volume and the
target domain that V1 loses.

### Feasibility — decisive, and it passes

| | |
|---|---|
| Crossing parents | 62 |
| **Irreplaceable even alone** | **2** — each holds 2 of the only 3 training segments in a target stratum, shortfall exactly **1 segment** each |
| Focal-eligible after excluding those 2 | **60** |
| Focal normal test segments covered | **203 of 232** |
| Fixed anomaly reference cohort | 232 |

With those two parents excluded, an **exact** K = 5 fold partition was found in 25 attempts:

| Fold | Focal parents | Focal segs | Donor segs | Size / window / strata match | Donors disjoint from focal | Arm training segs | % of official | Focal normal test segs | Target focal → donor |
|---:|---:|---:|---:|:--:|:--:|---:|---:|---:|:--:|
| 1 | 12 | 372 | 372 | exact | yes | 1,467 | 79.8% | 45 | 3 → 3 |
| 2 | 12 | 374 | 374 | exact | yes | 1,465 | 79.7% | 40 | 6 → 6 |
| 3 | 12 | 283 | 283 | exact | yes | 1,556 | 84.6% | 46 | 6 → 6 |
| 4 | 12 | 363 | 363 | exact | yes | 1,476 | 80.3% | 34 | 3 → 3 |
| 5 | 12 | 228 | 228 | exact | yes | 1,611 | 87.6% | 38 | 4 → 4 |

**Target-domain preservation — the V1 killer, solved.** Design E does not solve it automatically;
it solves it because target-domain focal contributions are replaceable by target-domain donors from
*other* parents. Every fold replaces its target segments **1:1** (3→3, 6→6, 6→6, 3→3, 4→4), so
**both arms retain target-domain training and all 25 strata**, against V1's zero and 12.

---

## 3. Estimands

**Primary — focal same-parent sibling-exposure score shift.** For each focal normal test segment,
`d = score_PRESENT − score_ABSENT`; aggregate to parent by median, then summarise across focal
parents. **This is the most identifiable choice available:** the *same* test segment is scored under
two models differing only by the toggle, so segment-level variance cancels. A negative shift means
sibling exposure made focal normals reconstruct better — look more normal to the model.

**Secondary — focal AUROC against a fixed anomaly reference.** `Δ_focal_AUROC = AUROC_PRESENT −
AUROC_ABSENT`, with the normal class being focal normal test segments and the anomaly class being
all 232 anomaly test segments, **identical across arms**. Statistically meaningful but weaker: the
model changes between arms, so anomaly-side scores move too. Reported in AUC units because that is
what the literature reports.

**Specificity check:** anomaly-side score shift on the fixed reference. A large shift there would
indicate the toggle moved the model globally rather than through same-parent familiarity.

**Uncertainty:** paired parent-cluster bootstrap over focal parents; PRESENT and ABSENT are never
resampled independently; folds are parent-disjoint by construction; windows are never resampled.

---

## 4. Comparison

Full table in [`v1_vs_v2_design_comparison.csv`](v1_vs_v2_design_comparison.csv). Design E wins on
11 of 12 dimensions; V1 wins only on compute.

**Design E isolates same-parent exposure decisively better.** It holds parent identity fixed — each
focal parent is its own comparison — holds training size and all 25 strata exact, preserves the
target domain, and pairs within segment. V1 confounds overlap status with parent identity, an 88.4%
volume cut, 13 missing strata, and a materially different effective cluster count.

**Neither design is causal.** Design E's residual is that donor segments are different physical
material from the focal segments they replace, so the toggle changes *which* recordings are seen as
well as *whether* the focal parent's own siblings are seen. Matching is on pre-test metadata only.

**Compute: Design E is substantially more expensive** — 10 runs × 3.07 h ≈ **31 h** against V1's
18 × 0.43 h ≈ **7.8 h**, about 4×. Cost was not allowed to select the design; it is reported so the
trade-off is explicit. Within a fold, PRESENT and ABSENT share the identical seed, so the contrast
is seed-controlled by construction and a single primary seed is defensible; a second seed
(`20260908`) is preregistered as an optional extension.

---

## 5. Claim boundary

**Permitted:** *"For focal normal RoboticArm parents, the presence of same-parent training siblings
was associated with a change of Δ in anomaly scores under the frozen official-baseline pipeline"* —
or the same statement in focal-AUROC units.

**Forbidden:** the official AUC is inflated by Δ; this is the global split-leakage penalty; the
published paper is invalid; Δ applies to BrushlessMotor or to other model families; this is causal
proof.

**This design measures SAME-PARENT FAMILIARITY FOR FOCAL TEST PARENTS — not GLOBAL SPLIT LEAKAGE.**

---

## 6. Residual risks

- Donor material differs in signal content from the focal material it replaces; matching is on
  metadata only.
- Two crossing parents are excluded as focal, so 9 normal test segments with siblings sit outside
  the focal cohort (203 of 212 covered).
- The estimand is narrower than V1's and must not be reported as a global leakage figure.
- Background composition shifts slightly across folds because the donor reserve is withheld from
  `B` — identical between arms *within* a fold, but not across folds.
