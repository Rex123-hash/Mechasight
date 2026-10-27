# MechaSight — Evaluation Protocol (Phase 2.2 Freeze, Phase 3.2.1 Reporting Correction)

**Version:** 2.2 protocol freeze · **Reporting correction:** Phase 3.2.1
(`p321-correction-75e1479967f120aff5047360`) · **Status:** the matched evaluation cohort, released
adaptation budget, preprocessing, validation, threshold, aggregation and uncertainty rules remain
frozen and **unchanged**. Phase 3.1 fitted 28 pre-registered classical configurations and Phase 3.2
opened the locked common test once, in a single 28-entry batch; the ledger stands at 28 and no
further test look has occurred.

Phase 3.2.1 corrected **reporting conventions only** — see
`reports/phase3_2_1/phase321_reporting_correction.md` and the policy at
`configs/phase3/phase321_reporting_correction_policy_v1.json`. Three rules now apply to every
result table: F1 uses the count identity `2TP/(2TP+FP+FN)`; precision with no positive predictions
is reported as undefined and never as 0; and every Average Precision carries the anomaly prevalence
of its own slice.

This document exists so that two things can never be silently mixed:

- **`official_baseline_reproduction`** — reproducing the published IMAD-DS benchmark on its own
  terms, for comparability with the authors' numbers.
- **`mechasight_experimental_protocol`** — our leakage-controlled protocol, which deliberately
  differs and is therefore **not** comparable to published numbers.

Every reported result must name which protocol produced it. A result without that label is not
reportable.

---

## 1. Why two protocols

Phase 1 established two structural properties of the release
(`reports/dataset_audit/dataset_summary.md`):

- **RoboticArm's official split is not parent-recording-disjoint.** Its 2,303 segments are
  consecutive excerpts of 101 continuous recordings; 62 recordings straddle train/test and 212 of
  232 normal test segments have a sibling excerpt in training, while no anomalous test segment can.
- **BrushlessMotor's classes are disjoint in recording time** (all normals before all anomalies).

Dropping the official split would make our numbers incomparable to the literature. Keeping only the
official split would mean reporting RoboticArm numbers carrying a known, unquantified bias risk.
So we keep both, clearly separated.

---

## 2. Protocol A — official benchmark reproduction `[FROZEN]`

**Purpose:** reproduce the published IMAD-DS benchmark as faithfully as the released code permits,
to check we can hit the published 91.62 (RA) / 58.95 (BM) overall AUC before claiming anything.

**Source of truth:** the authors' released code, read directly — `preprocess.ipynb`,
`baseline.ipynb`, `model.py`, `utilities.py` in `augustif/IMAD-DS_baseline`. Details are taken from
that code, not inferred. Where the code and the paper disagree, both are recorded.

| Element | Value | Evidence |
|---|---|---|
| Split | The released train/test folders, unmodified | `preprocess.ipynb` |
| Training data | **normal source train + normal target train, concatenated** | `preprocess.ipynb`: `Train_Metadata = pd.concat([normal_source_train, normal_target_train])` |
| Test data | all four test CSVs concatenated | same |
| Window | 100 ms, non-overlapping, timestamp-aligned | `WINDOW_SIZE_TS = 100ms` |
| Window start | `max(all sensors' first timestamp, gyro_first + 35 ms)` | `GYROSCOPE_WARM_UP_TIME` |
| Window end | `min(all sensors' last timestamp)` | `preprocess.ipynb` |
| Per-sensor length | `int(fs × 0.1)` with **fs = 16000 / 7063 / 7063**, zero-pad or truncate | `sensor_dict` |
| Resampling | none — native rates, concatenated | `preprocess.ipynb` |
| Normalisation | **`std_window`** — each window standardised by its own per-channel mean/std | `baseline.ipynb` `PARAMS['normalisation']` |
| Model | dense AE `[2048, 2048, 2048, 16]`, mirrored decoder, ReLU | `model.py` |
| Optimiser | Adam, lr 1e-4, batch 1024, MSE | `baseline.ipynb` |
| Validation | 10% of **windows**, stratified on `anomaly+op+env` | `baseline.ipynb` |
| Early stopping | patience 3, best = lowest validation loss | `baseline.ipynb` |
| Segment score | **median** of per-window reconstruction errors | `baseline.test(..., 'median')` |
| Metric | ROC AUC, reported for Source / Target / Source+Target | `utilities.calculate_single_auc` |
| Seed | 1995 | `baseline.ipynb` |

**Known limitations of Protocol A, to be printed alongside every Protocol A result:**

1. **RoboticArm:** the split is not parent-recording-disjoint (§4). Results carry a credible risk of
   optimistic bias of unmeasured magnitude.
2. **Both machines:** the authors' validation split is at *window* level, so windows from one
   segment appear on both sides. This affects early stopping only, not the test metric, but it is
   inherited by any faithful reproduction and must be disclosed.
3. **BrushlessMotor:** `fs = 7063` is applied although the released BM timestamps imply ~7135.7 Hz,
   so each 100 ms IMU window is truncated by a median of ~5 samples. This is part of what we are
   reproducing and **must not be "fixed" inside Protocol A** — fixing it would make the
   reproduction no longer a reproduction.
4. **BrushlessMotor:** the class/recording-time confound (§6) applies to the published numbers too.

**Rule:** Protocol A is a *reproduction target*, not a MechaSight result. No MechaSight method is
tuned under Protocol A, and no Protocol A number is presented as our own contribution.

---

## 3. Domain-generalisation training regimes `[FROZEN]`

The dataset offers abundant source-domain normals and few target-domain normals (RA 1812/27,
BM 1263/18). Two regimes exist and must never be pooled or compared as if equivalent.

**Regime S — SOURCE-ONLY GENERALISATION.**
Train on source-domain normal segments only. Target-domain normals are never seen in training,
validation, normalisation fitting, or threshold selection. Evaluate on source and target test sets
separately and pooled. This is domain *generalisation*.

**Regime T — TARGET-ADAPTED (FEW-SHOT).**
Train on the same source-domain normals **plus** the **released** target-normal training
allocation: RoboticArm 27 segments, BrushlessMotor 18 segments. This is few-shot domain
*adaptation*, and the budget is the one IMAD-DS actually exposes.

**Adaptation-budget invariant `[PHASE 2.2]`:** four quantities are recorded separately and must
never be collapsed into one another.

| Quantity | RoboticArm | BrushlessMotor | Meaning |
|---|---:|---:|---|
| `target_adaptation_observed` | 27 | 18 | released target normals **actually consumed** by Regime T |
| `target_adaptation_parent_groups` | 18 | 18 | distinct physical parent recordings holding them |
| `target_adaptation_blocked_siblings` | 96 | 0 | further normal segments of those parents, **consumed by nothing** |
| `target_adaptation_parent_closure` | 123 | 18 | observed + blocked; leakage bookkeeping, **never a training budget** |

Parent independence requires the 96 RoboticArm siblings to be **excluded from evaluation**. It does
not require them to become adaptation training data. They are barred from Regime-S training,
Regime-T training, validation, threshold calibration, hyperparameter selection and common test;
their sole purpose is to keep the 18 adaptation parents out of the evaluation cohort.

**Primary scientific question `[FROZEN PHASE 2.2]`:** *does access to the released few-shot
target-normal training allocation improve anomaly detection under domain shift?* The intended
training difference is therefore RA `S = source normals` vs `T = same source normals + released 27`,
and BM `S = source normals` vs `T = same source normals + released 18`.

**Matched-test invariant `[PHASE 2.1, UNCHANGED]`:** Regime S reserves the released adaptation
allocation rather than moving it into test. Both regimes use identical source validation and
identical common test segment/parent sets. The only intended experimental difference is access to
the released target-normal adaptation data.

**Secondary regime, documented and NOT authorised `[PHASE 2.2]`:** a future *T-parent-complete*
experiment may adapt on all 123 normal RoboticArm segments of the 18 adaptation parents. If it is
ever run it must be reported as a **larger adaptation-budget** experiment and never conflated with
the primary few-shot Regime T. It could support a budget study of 0 vs 27 vs 123 target segments.
No such model experiment is authorised now, and none has been run.

**The published benchmark uses Regime T.** Verified from `preprocess.ipynb`, which concatenates
`attributes_normal_source_train.csv` and `attributes_normal_target_train.csv` into one training
frame. Any comparison against the published 91.62 / 58.95 must therefore also use Regime T.

**Reporting rule:** every experiment records its regime in its config and in its results table.
`S` and `T` numbers never appear in the same column. A table mixing them is not reportable.

---

## 4. Protocol B — MechaSight leakage-controlled `[FROZEN AND IMPLEMENTED IN PHASE 2]`

Applies to **RoboticArm only**. BrushlessMotor already has one segment per recording and needs no
regrouping (but see §6).

### 4.1 Parent-recording grouping key, justified from acquisition structure

The grouping key is the **wall-clock stamp prefix of the segment id**, e.g. `20240418_14_01_48`.

Justification, measured rather than assumed:

- Segments sharing a stamp are **consecutive excerpts of one continuous acquisition**. Their `Time`
  columns run on one continuous clock: within recording `20240221_11_10_55`, `dataLog_0` spans
  2.486–9.986 s, `dataLog_1` spans 12.514–20.014 s, and so on to `dataLog_47` (~490 s) — a fixed
  7.5 s kept out of every ~10 s stride.
- 101 stamps cover all 2,303 segments; median 16 excerpts per stamp, max 48.
- **No stamp spans more than one operating condition, domain, or class** (measured: 0 of 101 for
  each). The grouping key is therefore consistent with the design factors.

### 4.2 Does recording-level grouping conflict with the source/target structure?

**Answered from data: no conflict for domain or operating condition; a real constraint on the
environmental factor.**

- **Domain is fully determined by the operating condition.** `weight00/10/15/20` → source (511, 510,
  511, 512 segments); `weight25/30/35` → target (87, 85, 87). No overlap. Since no recording spans
  two weights, no recording spans two domains. Grouping by recording therefore preserves the
  source/target structure exactly.
- **The environmental factor cannot be independently controlled.** 93 of 101 recordings carry
  **several** `Bckg` labels, because the backgrounds were mixed in digitally *after* recording
  (Phase 1 Q7). One physical recording is reused under multiple simulated environments. So a
  recording-disjoint split cannot freely balance `Bckg`; the background distribution follows
  whichever recordings land on each side.
- **Consequence, stated rather than hidden:** Protocol B can stratify on domain, operating condition
  and class, but only *report* the resulting background distribution rather than fix it. Because
  source and target use disjoint background sets (A,C,D,F vs B,E,G) and domain is preserved exactly,
  the background *set* is still correct per domain; only the proportions within a domain drift.

### 4.3 Feasibility

Anomalous segments are test-only by protocol, so **only normal recordings need partitioning**.
Parent recordings per cell:

| Domain | Condition | normal recordings | anomaly recordings |
|---|---|---|---|
| source | weight00 / 10 / 15 / 20 | 16 / 13 / 16 / 13 | 3 / 3 / 3 / 4 |
| target | weight25 / 30 / 35 | 9 / 7 / 8 | 2 / 2 / 2 |

**No cell has only one recording**, so a recording-disjoint stratified partition is feasible in all
14 cells. Margins are thin in the target domain (7–9 normal recordings per weight).

### 4.4 Final matched-cohort algorithm `[FROZEN PHASE 2.2, 2026-08-23]`

```
INPUT : metadata, parent key, released split, seed 1995
1. Preserve the Phase-2 source train/validation/common-test parent assignment.
2. Define adaptation_parents as every parent containing a released target-normal
   training example.
3. Role target_adaptation  := the released target-normal training segments only
                              (RA 27, BM 18).
4. Role adaptation_sibling_blocked
                           := the remaining normal segments of adaptation_parents
                              (RA 96, BM 0).  Consumed by nothing, in any regime.
5. Role common_evaluation  := source test parents, target-normal parents outside
                              adaptation_parents, and every anomalous test parent.
6. Regime S: source train -> train; target_adaptation -> reserved_adaptation.
7. Regime T: same source train + target_adaptation -> train.
8. Both regimes: identical source validation; common_evaluation -> test;
   adaptation_sibling_blocked -> blocked_sibling.
9. Assert exact equality of S/T test segment ids, parent ids and anomaly ids, and
   exact equality with the Phase-2.1 common-test cohort.
```

Every segment carries exactly one primary protocol role. Parent-level leakage is checked over
**consuming** splits (`train`, `validation`, `test`); `reserved_adaptation` and `blocked_sibling`
are excluded from that check because they enter no fit, no calibration and no metric, so they
cannot carry information across a parent boundary.

**Two superseded designs, both preserved:**

1. `reports/phase2/protocol_b_split_v1.csv` (Phase 2) assigned unused target adaptation examples to
   Regime-S test, producing **unequal S/T test populations**. Reason for correction: *Regime S and
   Regime T must be compared on an identical held-out evaluation cohort so that performance
   differences are attributable to target-normal adaptation rather than test-set composition.*
2. `reports/phase2_1/protocol_b_common_test_v2.csv` (Phase 2.1) fixed that, but made the whole
   **parent closure** the Regime-T training pool, so RoboticArm would have adapted on 123 segments
   when the release offers 27. Reason for correction: *parent independence requires sibling
   segments to be excluded from evaluation; it does not require them to become adaptation training
   data.* Phase 2.1 conflated "target examples actually consumed" with "same-parent siblings
   blocked to prevent leakage"; Phase 2.2 separates them.

Both remain versioned as historical evidence and are forbidden for future S-vs-T comparisons. The
current freeze is `reports/phase2_2/protocol_b_few_shot_adaptation_v3.csv`. The Phase-2.2
correction changes **training/adaptation role labels only**: the common evaluation cohort is
bit-for-bit the Phase-2.1 cohort, proved in `blocked_sibling_isolation_proof.csv`.

**Explicitly forbidden:** choosing among these options by looking at which yields better model
results. The choice is made before any model is trained, and recorded.

### 4.5 What Protocol B is for

Protocol B produces MechaSight's **primary** RoboticArm results. Protocol A results are reported
beside them for comparability with the literature. The difference between them is itself a
measurement — see §5.

---

## 5. The leakage-bias question is a hypothesis, not a finding `[OPEN]`

**Established by Phase 1:** parent-recording independence is violated in the official RoboticArm
split; 212 of 232 normal test segments have a training sibling; no anomalous test segment does.

**NOT established:** that this inflates AUC, or by how much. The direction is plausible — normal
test segments with near-twins in training should reconstruct more easily and score lower, while
anomalous segments have no such advantage — but plausibility is not measurement.

**Correct wording, to be used everywhere:** *the released RoboticArm split creates a credible risk
of optimistic bias, of unmeasured magnitude and direction.*

**How it will be tested (Phase 3+, pre-registered here):** train the *same* baseline under Protocol
A and Protocol B, changing nothing but the split, and report paired differences in AUROC/AUPRC
(with pAUROC secondary) with
its interval. Both outcomes are publishable, including "no measurable difference", which would mean
the leakage does not matter in practice for this model.

**Anti-goal:** no experiment may be designed, selected or reported to make the leakage look larger.
The comparison uses one model, one seed set, one metric, declared in advance.

---

## 6. BrushlessMotor temporal confound `[STRUCTURAL LIMITATION — NOT SOLVABLE]`

All 1,437 BM normal segments were recorded 10:20:21–16:40:50 and all 156 anomalous segments
16:48:07–18:22:40 on 2024-04-23 — zero overlap, 7.3-minute gap. Recording time therefore predicts
the class perfectly.

**This cannot be removed by any analysis of this dataset.** Fault effects and acquisition-session
effects are **not identifiable** for BrushlessMotor: any difference between the normal and anomalous
populations may be caused by the induced fault, by session drift, or by any mixture of the two, and
the data contain no interleaved normal/fault blocks that would let the two be separated.

**A temporal-drift diagnostic is not deconfounding.** Training on early-session normals and
evaluating on late-session normals would measure only whether *detectable drift exists inside the
normal training data*. Even a null result would not establish that the later anomaly session is free
of independent session effects — the anomaly session is a different, later period whose drift is
unobservable because no healthy recordings exist within it. The diagnostic is worth running as a
lower bound on drift, and it is explicitly **not** a fix.

**Required wording wherever a BM result appears:** *for BrushlessMotor, fault effects and
acquisition-session effects are not fully identifiable; reported detection performance may include
an unquantified session-drift component.*

The same applies between the two anomaly types (belt 16:48–17:33, magnetic 17:38–18:22, disjoint),
so per-type comparisons on BM carry the same limitation.

---

## 7. Train / validation / test roles `[FROZEN]`

| Primary role | Split label | May be used for |
|---|---|---|
| `source_training` | `train` | model fitting and fitting learned preprocessing/normalisation statistics |
| `source_validation` | `validation` | early stopping, model/window/feature selection and threshold calibration |
| `target_adaptation` | `train` (T) / `reserved_adaptation` (S) | Regime-T model fitting; in Regime S nothing, membership is recorded solely to define the S/T counterfactual |
| `adaptation_sibling_blocked` | `blocked_sibling` | **nothing, in either regime**; withheld solely to keep adaptation parents out of the evaluation cohort |
| `common_evaluation` | `test` | computing final reported metrics — **nothing else** |

Every segment carries exactly **one** primary protocol role. `train`, `validation` and `test` are
the *consuming* splits; `reserved_adaptation` and `blocked_sibling` are consumed by nothing and the
preprocessing pipeline refuses to materialise them into a cache, feature file or manifest.

- Anomaly labels exist only in test and are used **only** to compute metrics. Training data objects
  expose no anomaly-label field at all.
- Protocol B uses the **same frozen source-normal validation parents** in S and T. The released
  target-adaptation examples are train-only in T and reserved in S; none are validation. Blocked
  siblings are neither. The fixed RA validation set
  has 6 parents/186 segments covering all source conditions. Grouped CV was considered but rejected
  here because it would multiply every model fit and change the calibration cohort; the fixed,
  paired cohort is simpler and was frozen before results. Its small parent count is disclosed.
- No test-set tuning of any kind. Every test-set evaluation is logged with a timestamp and config
  hash in `reports/experiment_results/` so the number of test looks is auditable.

---

## 8. Preprocessing and normalisation fitting `[FROZEN]`

- Any statistic used by a model — mean, std, min/max, PCA basis, scaler, resampling reference rate —
  is fitted on **training data only** and stored with provenance (which split, which segments,
  which config hash).
- Per-window normalisation (`std_window`) needs no fitting and is leak-free, but it removes
  within-window amplitude. Both it and train-fitted per-channel z-scoring are evaluated as a
  declared experimental variable; neither is inherited without justification.
- Under Regime S, target-domain normals are not training data and contribute to no statistic.
- **Window-sibling leakage prevention:** windows inherit their parent segment's split, and segments
  inherit their parent recording's split. No window, and no segment, ever crosses a boundary from
  its parent.

---

## 9. Metrics and reporting `[FROZEN]`

- **Unit of evaluation:** the segment. Window-level numbers are diagnostics only, always labelled.
- **Aggregation:** Protocol A exactly reproduces upstream median reconstruction error. Protocol B's
  primary rule is frozen to the **median window anomaly score**. Every segment score stores its
  contributing window ids. Test-set aggregation search is forbidden.
- **Primary threshold-free metrics:** segment-level **AUROC and AUPRC** in Source+Target, per machine.
- **Secondary threshold-free:** pAUROC@FPR≤0.1, domain-specific metrics and source→target degradation.
- **Threshold-dependent:** F1, recall/sensitivity, specificity and false-positive rate.
- **Threshold-free vs threshold-dependent:** AUROC, AUPRC and pAUROC require no operating threshold.
  F1 and FPR are threshold-dependent and may be reported **only** with the frozen threshold rule
  below. "Best F1 over thresholds on the test set" is forbidden.
- **Threshold rule:** empirical 90th percentile of **source-validation normal segment scores** using
  quantile method `higher`; classify anomalous only when `segment_score > threshold`. The cohort ids
  are identical across S/T, though model scores and hence thresholds may differ. No anomaly or test
  score participates.
- **Balance caveat:** Protocol-B v2 has RA 234 normal/232 anomalous and BM 156/156 pooled, still
  unlike plant prevalence; target-only RA is especially imbalanced (20 normal/116 anomalous).
- **Per machine, always.** No cross-machine average is presented as *the* result.
- **Head-to-head claims** are decided by the paired parent-cluster bootstrap frozen in
  `docs/leakage_risks.md` R8 — never by comparing marginal confidence intervals.

### 9.1 Statistical independence and uncertainty `[FROZEN PHASE 2.1]`

- Windows are never inferential units.
- RoboticArm uses a 10,000-replicate, seed-1995 **parent-recording cluster bootstrap**, stratified by
  domain × anomaly status × operating condition. A selected parent contributes all child segments.
- Paired model comparisons reuse identical parent draws and report percentile intervals on metric
  differences. Marginal metric intervals are descriptive.
- BrushlessMotor uses the same stratified procedure at recording/segment level because its mapping
  is one-to-one, while retaining the class/session-confound limitation.
- Every model in a paired comparison receives the same frozen test observations, window config,
  feature config, score aggregation and metric implementation.

---

## 10. Terminology `[FROZEN]`

- **Anomaly score.** Model outputs are anomaly scores: higher means more anomalous, on an arbitrary
  monotone scale with no probabilistic meaning. This is the term used in code, tables, figures and
  any future interface.
- **"Confidence" and "probability" are banned** for raw model outputs. They may be used only for a
  quantity produced by an explicit calibration fitted on held-out training normals whose reliability
  is itself evaluated (e.g. ECE or a reliability diagram). Until such a procedure exists and is
  reported, neither word appears in any output.
- **Cohorts, not datasets.** RoboticArm and BrushlessMotor are two **experimental cohorts within
  the same dataset (IMAD-DS)**. They are analysed and reported separately because their acquisition
  and anomaly characteristics differ substantially — not because they are different datasets.

---

## 11. Architectural requirement — keep the two paths separable `[FROZEN]`

The codebase must make it impossible to mix the two protocols by accident:

- Protocol choice is an explicit, required config field (`protocol: official_reproduction` or
  `protocol: mechasight`), recorded in every results artefact and every log line.
- The official 100 ms window, `fs = 16000/7063/7063`, `std_window` normalisation and segment-level
  median aggregation remain permanently available as a named configuration, even if MechaSight's own
  experiments adopt different values.
- MechaSight's window length, rate handling and normalisation remain validation-controlled
  (`docs/research_definition.md` §16); Protocol-B primary aggregation is frozen to median.
  Changing an experimental setting must not alter the reproduction path.
- Results tables carry the protocol and regime as columns. Any table lacking them is not reportable.

---

## 12. Decisions resolved through Phase 2.1 / 2.2

0. `[PHASE 2.2]` The primary Regime-T adaptation budget is the **released** allocation (RA 27,
   BM 18), not the parent closure (RA 123). Blocked siblings (RA 96) are reserved solely to enforce
   parent independence. A *T-parent-complete* budget experiment is documented but unauthorised.
1. Phase-2 split v1 and Phase-2.1 split v2 are historical/superseded; few-shot v3 is frozen in §4.4
   and `reports/phase2_2/`. v1 fixed nothing; v2 fixed unequal S/T test cohorts; v3 fixes the
   adaptation budget without moving a single evaluation segment.
2. BrushlessMotor IMU interpolation is enabled only for operations requiring a regular grid
   (currently frequency features), never globally.
3. Window length and stride remain explicit research hyperparameters. The Phase-2 smoke config uses
   100 ms / 100 ms only as an initial pipeline value, not as an optimality claim.
4. Both per-window deterministic and train-fitted channel standardisation are implemented; the
   experimental choice remains for Phase 3 validation, while Protocol A is fixed to `std_window`.
5. The first-accelerometer-sample rule is machine-aware: BrushlessMotor 1, RoboticArm 0.
6. The meaning of the `_corr` marker remains unresolved and no transformation depends on it.
7. Validation is the shared fixed source-normal cohort; target adaptation is train-only in T.
8. Primary metrics are AUROC/AUPRC; threshold is source-validation-normal q90 (`higher`).
9. Segment aggregation is median; RoboticArm uncertainty is parent-clustered and paired.

---

## 13. Two acquisition artefacts and what they do (and do not) license

**Accelerometer first-sample rail.** Measured across **all 3,896 segments**: present in
**1,430 of 1,593 BrushlessMotor segments (89.8%)** and **0 of 2,303 RoboticArm segments (0.0%)**.
In every affected file the railed row is the **first row only** — the highest railed row index
anywhere in the release is 0. Prevalence is **not associated with any label**: split p = 1.00,
domain p = 1.00, class p = 0.43, operating condition p = 0.38, background p = 0.86.

Because the rule would be *"discard sample index 0 of the accelerometer stream"*, it depends only on
sample position — information available at inference time — and on no label. It is therefore
deployment-safe. **Proposed, not yet frozen:** apply it, per machine, as a declared preprocessing
step; note that Protocol A already removes these samples incidentally via the 35 ms gyroscope crop,
so it changes nothing there.

**Non-uniform IMU sampling (BrushlessMotor only).** 2.7% of BM IMU intervals sit at ~122.5 µs
against 97.3% at ~140.6 µs — an 18 µs difference far beyond the 1 µs timestamp grid. RoboticArm's
IMU is uniform. **No resampling decision is taken here.** The requirement differs by feature family:

| Feature family | Sensitivity to non-uniform sampling |
|---|---|
| Raw/time-domain sequence models (CNN-AE, TCN) | Low — they consume ordered samples; irregular spacing is a mild time-warp |
| Classical time-domain statistics (RMS, kurtosis, crest factor) | Low — order-independent or nearly so |
| **FFT / PSD / spectral features** | **High** — the DFT assumes a uniform grid; 13% jitter on 2.7% of intervals smears high-frequency content |

**Options to be decided in Phase 2 review, with a sensitivity test rather than an assumption:**
(a) leave raw, accept the approximation, and quantify the error; (b) resample the BM IMU to a
uniform grid for spectral features only, leaving time-domain paths on raw samples; (c) resample
everything uniformly; (d) use methods that tolerate non-uniform sampling (e.g. Lomb–Scargle) for the
spectral path. Any resampling reference rate is a fitted parameter and must be derived from
**training data only**.

---

## 14. What this document does not freeze

Model families, architectures, hyperparameters, feature sets, the reliability-aware fusion
mechanism, and whether any of it works. Those remain in `docs/research_definition.md` §16.
