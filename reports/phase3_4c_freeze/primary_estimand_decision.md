# Phase 3.4C — Primary Estimand Decision Record

**Status:** `PRETEST_EXECUTION_FROZEN` · **Parent design:** `p34b-ra-focal-2ab15394f78b560f1e8c0880`
**Execution config:** `p34c-exec-0eecc19f81e200f5c301ca25`

No model trained, no inference run, no score computed, no ledger entry appended.

---

## 1. Raw reconstruction error is **not** an admissible primary estimand

The v2 proposal used `score_PRESENT − score_ABSENT` on the same focal segment. That looks paired,
and it is — but the two scores come from **two independently trained autoencoders**, and their
reconstruction errors do not live on a common scale.

Five concrete reasons:

1. **Independent training** puts the two arms' errors on different absolute scales.
2. **BatchNorm running statistics differ** between arms and are applied at evaluation time.
3. **Early stopping at `patience = 3`** can halt the arms at different epochs.
4. **Reproduced upstream checkpoint aliasing** means the scored weights are the *final* epoch — an
   arbitrary point, not a selected optimum.
5. **A global scale shift moves anomaly scores too.** So a raw normal-score difference cannot
   distinguish *"focal normals look more normal"* from *"this model reconstructs everything better."*

Point 5 is decisive: the raw difference conflates the effect of interest with a nuisance shift that
has nothing to do with same-parent familiarity.

> **Verdict: raw reconstruction-error difference is demoted to SECONDARY DIAGNOSTIC (S4),
> explicitly labelled scale-dependent.**

---

## 2. Primary estimand — parent-equal-weighted focal concordance delta

For focal normal segment *i* in arm *a*, against the **fixed** 232-segment anomaly reference scored
by the **same** arm, with exact ties scoring zero exactly as upstream AUC does:

```
C_i(a) = (1 / 232) · Σ_j  I( score_i(a) < score_anomaly_j(a) )

delta_i        = C_i(PRESENT) − C_i(ABSENT)
delta_parent(p)= mean of delta_i over parent p's focal normal segments
Δ_parent_concordance = mean over the 60 focal parents of delta_parent(p)
```

**Why this is the defensible primary.** `C_i` depends only on the **rank** of segment *i* against the
anomaly reference *within its own arm*. It is therefore invariant under any strictly increasing
transform of that arm's scores — which removes precisely the global scale shift that disqualifies the
raw difference, while keeping the quantity in AUC units.

Verified numerically: under ×1000, +7 and `exp`, `max |ΔC_i| = 0`.

**Positive Δ** means same-parent sibling exposure improved normal-vs-anomaly ordering for focal
parents.

---

## 3. Parent aggregation: **MEAN**, decided on statistical grounds

Focal normal segments per parent: **min 1, median 3, mean 3.38, max 8** — and **11 of 60 parents
hold exactly one segment**.

| | |
|---|---|
| **Median** | over ~3 values it is unstable, discards magnitude, and is identical to the mean at n = 1 and n = 2 anyway |
| **Mean** | reduces exactly to the segment value at n = 1, uses all information, and makes the two-stage estimator interpretable as a parent-equal-weighted mean of within-parent mean concordance |

**MEAN is frozen.** Chosen before any result exists, on the shape of the cohort — not on outcomes.

---

## 4. Relationship to focal AUROC — the identity, and why we depart from it

The **segment-weighted** mean of `C_i` **is** the ordinary strict pairwise focal AUROC, exactly:

```
AUROC = (1/(N_n·N_a)) Σ_i Σ_j I(s_i < s_j) = (1/N_n) Σ_i C_i
```

Verified numerically to machine precision across trials.

But our physical independence unit is the **parent recording**, not the segment. Segment weighting
would let an 8-segment parent count eight times a 1-segment parent — pseudo-replication that inflates
the apparent sample size. Relative to equal parent weights, segment weighting ranges from **0.30× to
2.36×**, so these are materially different estimands, not a rescaling.

> **Frozen: PARENT-EQUAL-WEIGHTED CONCORDANCE DELTA is PRIMARY. Ordinary segment-weighted focal
> AUROC delta is SECONDARY (S1).**

---

## 5. Inference unit

| | |
|---|---:|
| **Primary inferential unit** | focal parent recording |
| **N (primary)** | **60** |
| Segment-level deltas (descriptive only) | 203 |

Inference is made at the parent level. The 203 segment-level deltas are reported descriptively and
**must never** be treated as 203 independent observations.

---

## 6. Uncertainty — and an honest limit

**Procedure.** Paired nonparametric bootstrap over the 60 focal parents; 10,000 replicates; seed
**27182** (distinct from the training seed 1995, fold seed 20260908 and donor seed 31415, so no audit
quantity shares a stream with an experimental one); percentile 95% interval. Parents are resampled
with replacement carrying all their focal segments; PRESENT and ABSENT are **never** resampled
independently, because the pairing is within segment and `delta_i` is the resampled quantity.

**Fold clustering — decision and consequence.** Each focal parent belongs to exactly one fold, and
parents within a fold share the same trained model pair, so their deltas *are* correlated through the
model. Fold is nevertheless **not** entered as an extra bootstrap level: with only **5** folds, a
cluster-level variance cannot be estimated, and resampling 5 clusters would give an unstable,
misleading interval.

> **Stated plainly: the primary interval is CONDITIONAL ON THE TEN TRAINED MODELS. It covers parent
> sampling and does NOT include training-model or seed uncertainty.**

Mitigation: the five fold-level deltas are reported separately as a descriptive spread indicating
model-level variability. Folds must **not** be presented as N = 5 independent evidence while
parent-level resampling supplies the interval.

---

## 7. Frozen secondary set — six metrics, no more

| ID | Metric |
|---|---|
| S1 | ordinary segment-weighted focal AUROC delta vs the fixed 232-segment anomaly reference |
| S2 | source-domain focal AUROC delta |
| S3 | target-domain focal concordance delta — **only if** both classes and ≥ 10 focal parents are present in the target slice; otherwise reported as *not estimable* |
| S4 | raw focal-normal reconstruction-error shift — **labelled SCALE-DEPENDENT DIAGNOSTIC** |
| S5 | anomaly-reference raw score shift — specificity check for global model-scale movement |
| S6 | distribution of the 60 parent-level deltas, with counts of Δ > 0, Δ = 0, Δ < 0 |

**No further metric may be added after any score exists.** S5 matters most for interpretation: a
large anomaly-side shift would indicate the toggle moved the model globally rather than through
same-parent familiarity.

---

## 8. Seed policy — **POLICY A, one seed**

Seed **1995**, 5 folds × 2 arms = **10 models**. Seed 20260908 is **out** of Phase 3.4 primary
execution.

Reasoning:

- The contrast is paired within fold — PRESENT and ABSENT share the identical seed — so seed effects
  largely cancel *inside* Δ rather than needing to be averaged out.
- The inferential sample is 60 parents, not seeds. A second seed would not enlarge it.
- The five folds already give five independently trained model pairs on disjoint focal parents.
- A second seed roughly doubles compute (≈31 h → ≈61 h) to estimate a quantity that cannot enter the
  primary point estimate.

**The ambiguity that actually mattered was the word "optional."** It is now removed: if seed 20260908
is ever run it becomes a **separately versioned robustness analysis** and **cannot** retroactively
alter, replace or be pooled with this primary result. No seed may be added, dropped or reweighted
after any score is observed.

---

## 9. Paired-arm audit — what is paired, and what cannot be

**Verified exact in all five folds:** `B` identical between arms; focal segments only in PRESENT;
donor segments only in ABSENT; zero focal/donor segment overlap; no donor parent is a focal parent of
that fold; segments, windows and all 25 strata match.

**Paired exactly:** `torch.manual_seed(1995)` before construction, so both arms start from identical
initial weights; identical architecture and construction order; identical validation fraction and
`random_state`; identical DataLoader configuration and dataset length, so the shuffle permutation
*sequence* is the same; identical environment and pinned thread count.

**Cannot be paired — documented, not hidden:**

- minibatch **content** differs, because the arms hold different segments at the same indices — this
  *is* the intervention;
- the stratified validation split selects different segments, because it is computed on each arm's
  own label frame even under the same seed;
- early stopping may halt the arms at different epochs.

### Residual imbalance — MATERIAL and structural

| | |
|---|---:|
| ABSENT physical parents | **64** in every fold |
| PRESENT physical parents | **70–75** |
| Kish effective-cluster gap | **4.6 – 7.4** |

ABSENT structurally excludes the 12 focal parents, and donors can only come from parents already in
`B` — **so no donor choice can add a parent back.** A load-balancing donor rule was tested: it
narrowed the ESS gap by only ~1.5 while *reducing* donor-parent diversity, so the frozen memberships
were kept rather than churned for marginal gain.

> **Interpretive consequence: the toggle changes parent diversity as well as same-parent sibling
> presence. Any observed Δ is the joint effect of both and must be reported as such.**

---

## 10. Readiness verdict — **YELLOW**

Every execution and statistical ambiguity named for this phase is resolved and frozen, so
implementation is safe to begin. Two limitations remain that no design choice can remove:

1. the ABSENT arm structurally contains 12 fewer physical parents than PRESENT;
2. no bootstrap over this experiment can include training-model uncertainty.

Neither is a reason to stop. Both are reasons the eventual claim must stay inside the frozen
boundary.
