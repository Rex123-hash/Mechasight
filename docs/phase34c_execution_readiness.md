# Phase 3.4C — Focal-Parent Sibling Toggle: Execution-Readiness Freeze

**Role:** independent senior ML statistician and experiment-release engineer.
**Status:** `PRETEST_EXECUTION_FROZEN`
**Execution config:** `configs/phase3/roboticarm_focal_toggle_execution_v1.json` —
`p34c-exec-0eecc19f81e200f5c301ca25`
**Parent design:** `p34b-ra-focal-2ab15394f78b560f1e8c0880` (unchanged, additive layer only)
**Starting HEAD:** `0f388b2ba743c29cd06bcab89e8162a09b447dcd`

**Readiness verdict: YELLOW.**

> **No model was trained. No inference was run. No test example was scored. No ledger entry was
> appended.** Everything below came from segment metadata, split membership, and a synthetic-data
> check of the estimator.

---

## 1. What this phase changed

Phase 3.4B left five things genuinely undecided. All five are now frozen — and auditing them turned
up two problems the design did not know it had.

| | Before | Now |
|---|---|---|
| Primary estimand | raw score difference | **parent-equal-weighted concordance delta**; raw difference demoted to diagnostic |
| Parent aggregation | unspecified | **MEAN**, decided from the cohort shape |
| Seed policy | "1995 primary, 20260908 optional" | **POLICY A**, one seed; the second is out and can never be pooled |
| Uncertainty | "paired parent-cluster bootstrap" | fully specified, **with the fold-clustering limit stated** |
| Device | "CPU unless CUDA is frozen first" | **forced CPU** — CUDA is not available |

**Two new findings:** raw reconstruction errors from independently trained autoencoders are not
comparable (§3), and the ABSENT arm structurally contains **12 fewer physical parents** than PRESENT
(§5).

---

## 2. V2 reconfirmed from source data

Recomputed from the windowed HDF5 rather than read from the design reports — all confirmed:

| | |
|---|---:|
| Crossing parents | 62 |
| Irreplaceable ("blocking") parents | **2** |
| Focal-eligible parents | **60** |
| Sibling-bearing normal test segments | 212 |
| …covered by the focal cohort | **203** |
| …excluded | **9** |
| Folds × focal parents per fold | **5 × 12** |
| Each focal parent appears exactly once | **yes** |
| Focal parent set == eligible set | **yes** |
| Fixed anomaly reference | **232** |

Segments, windows, all 25 strata, exact target-domain replacement and focal/donor parent
disjointness all re-verified in every fold.

---

## 3. Raw reconstruction error cannot be the primary estimand

PRESENT and ABSENT are **independently trained autoencoders**. Their reconstruction errors do not
share a scale: BatchNorm running statistics differ and are applied at evaluation; early stopping can
halt the arms at different epochs; the reproduced upstream aliasing means the scored weights are the
*final* epoch, an arbitrary point. Decisively, **a global scale shift moves anomaly scores too**, so a
raw normal-score difference cannot separate *"focal normals look more normal"* from *"this model
reconstructs everything better."*

**Demoted to secondary diagnostic S4, labelled scale-dependent.**

---

## 4. The primary estimand

```
C_i(a) = (1/232) · Σ_j I( score_i(a) < score_anomaly_j(a) )     ties score zero, as upstream
delta_i = C_i(PRESENT) − C_i(ABSENT)
delta_parent(p) = MEAN of delta_i over p's focal normal segments
Δ_parent_concordance = mean over the 60 focal parents
```

`C_i` depends only on segment *i*'s **rank** against the fixed anomaly reference **within its own
arm**, so it is invariant under any strictly increasing transform of that arm's scores — verified
numerically (`max |ΔC_i| = 0` under ×1000, +7, `exp`). That removes exactly the nuisance that
disqualifies the raw difference, while staying in AUC units.

**Aggregation: MEAN.** Focal parents hold a median of 3 normal segments (min 1, max 8; **11 of 60
hold exactly one**). A median over ~3 values is unstable and discards magnitude; the mean reduces
exactly to the segment value at n = 1. Decided on cohort shape, before any result.

**Relation to focal AUROC.** The *segment*-weighted mean of `C_i` **is** the strict pairwise focal
AUROC exactly — verified to machine precision. We depart from it deliberately: the parent is the
physical independence unit, and segment weighting would let an 8-segment parent count eight times a
1-segment one. Relative to equal parent weights, segment weighting spans **0.30×–2.36×**, so these
are different estimands, not a rescaling.

> **PARENT-EQUAL-WEIGHTED is PRIMARY; segment-weighted focal AUROC is SECONDARY (S1).**
> **Inference unit: 60 parents.** The 203 segment deltas are descriptive only.

---

## 5. Paired-arm audit — and a structural imbalance

**Verified exact in all five folds:** `B` identical between arms; focal only in PRESENT; donors only
in ABSENT; zero focal/donor overlap; no donor parent is that fold's focal parent; segments, windows
and all 25 strata match.

**But the arms differ in parent count.**

| | |
|---|---:|
| ABSENT physical parents | **64** — every fold |
| PRESENT physical parents | **70–75** |
| Kish effective-cluster gap | **4.6 – 7.4** |

ABSENT structurally excludes the 12 focal parents, and donors can only come from parents already in
`B`, **so no donor choice can add a parent back**. A load-balancing donor rule was tested and
narrowed the gap by only ~1.5 while reducing donor diversity — so the frozen memberships were kept
rather than churned for marginal gain.

> **The toggle moves parent diversity as well as sibling presence. Any Δ is the joint effect of both
> and must be reported as such.** This is severity MATERIAL and is now in the claim boundary.

---

## 6. Uncertainty, with its limit stated

Paired nonparametric bootstrap over the **60 focal parents**; **10,000** replicates; seed **27182**
(distinct from training 1995, fold 20260908, donor 31415); percentile 95%. Parents resampled with
replacement carrying all their segments; PRESENT and ABSENT never resampled independently.

**Fold clustering.** Parents within a fold share a trained model pair, so their deltas are
correlated. Fold is **not** added as a bootstrap level: 5 clusters is far too few to estimate a
cluster variance, and resampling them would give an unstable interval.

> **The primary interval is CONDITIONAL ON THE TEN TRAINED MODELS. It covers parent sampling and
> does NOT include training-model or seed uncertainty.**

The five fold-level deltas are reported separately as a descriptive spread. Folds must **not** be
presented as N = 5 independent evidence while parent resampling supplies the interval.

---

## 7. Execution policy

**Environment — forced, not chosen.** `torch.cuda.is_available()` is **False** and torch is a `+cpu`
build, so CPU is the only option. No new framework build was installed.
**`execution_environment_id = p34c-env-b7a5de2af4c1d9614e49`** — Python 3.14.4, torch 2.14.0+cpu,
numpy 2.4.4, pandas 3.0.2, sklearn 1.9.0, h5py 3.16.0, MKL 2026.1, threads pinned at 8. Determinism
policy is upstream-faithful: `torch.manual_seed` only, **not** `use_deterministic_algorithms`, since
enabling it would depart from the frozen pipeline. **No CPU↔CUDA switch once training begins; a
material environment failure stops the phase.**

**Seed — POLICY A.** Seed 1995, **10 models**. Seed 20260908 is out; if ever run it is a separately
versioned robustness analysis that cannot alter or be pooled with this result.

**Training order — counterbalanced, frozen:** `1P, 1A, 2A, 2P, 3P, 3A, 4A, 4P, 5P, 5A`. Pairs run
back to back so drift hits both arms of a pair nearly equally, and the leading arm alternates by fold
parity so neither arm systematically runs first.

**Paired randomness.** Identical initial weights (same `manual_seed` before construction), identical
architecture, validation fraction and `random_state`, identical DataLoader config and dataset length
(so the same permutation *sequence*), identical environment and thread count. **Cannot be paired:**
minibatch *content* (that is the intervention), the stratified validation split (computed on each
arm's own labels), and the early-stopping epoch.

**Checkpoints.** All 10 retained until the experiment *and its independent audit* are complete —
~1.63 GB. Each records experiment id, fold, arm, seed, SHA-256, path, environment id, epochs run,
best validation epoch, **final epoch actually scored**, config identity and training-membership hash.
Working copies stay under `data/interim/phase3_4/` with hashes in a committed manifest; a durable
out-of-tree copy is proposed but **nothing is moved or uploaded in this phase**. Phase 3.3 left its
checkpoints local-only without saying so; here the status is stated up front.

**Score persistence — binding.** 24 columns × 464 segments × 10 experiments = **4,640 rows**,
written atomically and hashed **before** any aggregate exists. Aggregates are computed *from* the
persisted artifact, so every metric is re-derivable later with no model inference. Schema frozen in
`score_schema.json`.

**Ledger.** New protocol `roboticarm_focal_parent_sibling_toggle_v1`. **All 10 entries authorised
atomically before the first test inference** — not one per model — which removes any possibility of
adaptive stopping after seeing early folds. The existing 30 entries are immutable and **nothing is
appended in this phase**.

**Commit structure.** COMMIT A (implementation, evaluator, authorisation machinery, environment
freeze, membership and experiment identities) → training → COMMIT B (checkpoint hashes, still no test
performance) → only then the already-authorised inference. **No model may be scored whose checkpoint
hash is not already frozen in COMMIT B.**

**Failure policy** — see [`failure_policy.md`](../reports/phase3_4c_freeze/failure_policy.md).
Restart is permitted only *before* test exposure and only for mechanical failure; re-scoring after
test exposure is prohibited without exception; NaN or an environment change stops the phase.

---

## 8. Cohort boundaries

The full fixed 464-segment cohort is scored **once per authorised experiment** — focal normals for
the primary, the 232 anomalies for concordance, non-focal normals for the global diagnostic. But the
**primary analysis uses only** the 203 focal normal segments, their 60 parents, and the fixed anomaly
reference. **Non-focal normal performance must not influence any primary design choice, threshold,
aggregation rule or reporting decision.**

**Excluded parents:** `20240222_11_14_08_dataLog` and `20240405_13_09_52_dataLog` — each holds 2 of
the only 3 training segments in a rare target-domain stratum (`weight35|BckgE` and `weight30|BckgG`),
a shortfall of exactly 1 segment each. Together they account for the **9** excluded normal test
segments. They stay outside the primary cohort; future descriptive reporting may mention them but
must not score them under an improvised unmatched design.

---

## 9. Claim boundary

**If positive:** *"For the 60 donor-matchable focal RoboticArm parents, exposing the autoencoder to
same-parent training siblings was associated with a [Δ] change in parent-weighted
normal-versus-anomaly concordance under the frozen official-baseline pipeline."*
**If negative:** state the negative value in the same words. **If near zero:** state it with equal
prominence — a null is a result.

**Mandatory qualifiers:** donor-matchable focal cohort of 60 parents / 203 segments, not the full
test set · post-test controlled sensitivity analysis · **not** global split leakage · **not**
published-AUC inflation · **not** causal proof · does not generalise to BrushlessMotor, other models
or other sites · the interval is conditional on the ten trained models · **the ABSENT arm contains 12
fewer physical parents by construction.**

**V1 remains SECONDARY**, not to be executed until V2 is complete and audited, and its results must
not influence V2 execution.

---

## 10. Why YELLOW rather than GREEN

Every ambiguity this phase was convened to resolve is resolved and frozen; implementation is safe to
begin. Two limitations remain that no design choice can remove:

1. **ABSENT structurally contains 12 fewer physical parents than PRESENT**, so the toggle moves
   parent diversity alongside sibling presence.
2. **No bootstrap here can include training-model uncertainty** — 5 folds is too few, so the interval
   is conditional on the ten models.

Neither is a reason to stop. Both are reasons the eventual claim must stay inside the boundary above,
and forcing GREEN would misdescribe what the experiment can support.

---

## 11. Next step — not executed

Final review of this freeze. Then COMMIT A, then training under the frozen order, then COMMIT B, then
the already-authorised inference. **Nothing may be scored before the ledger authorisation and
COMMIT B exist.**
