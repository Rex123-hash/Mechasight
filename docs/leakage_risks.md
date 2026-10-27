# Leakage and Validity Risk Register — Phase 0

The purpose of this register is to make cheating **structurally difficult** rather than a matter
of good intentions. Each risk states the failure, why it matters on IMAD-DS specifically, the
mitigation, and how the mitigation will be *enforced* (test, code structure, or convention).

**Status update.** This register was written in Phase 0 before any enforcement existed. The
enforcement column is now implemented and exercised by the test suite: parent-level splits,
normal-only fitting guards, blocked-sibling isolation, train-only normalisation, frozen thresholds,
the test-look ledger and the class-stratified cluster bootstrap are all in place, and the first
locked common test has been opened once under them. Individual entries below retain their original
Phase-0 wording; treat the enforcement column as delivered rather than planned.

Note also that the "median of window scores" aggregation described below is the Phase-2 wording; the
Phase-3 classical baselines instead take the median of **each feature** before scoring one vector
per segment.

Severity: **CRITICAL** = would invalidate the project's conclusions · **HIGH** = would materially
inflate reported results · **MEDIUM** = would distort interpretation.

---

## R1 — Windows from one recording split across train / validation — **CRITICAL**

**Failure.** A segment yields many 100 ms windows. Windows from the same recording are highly
correlated (same machine, same load, same noise realisation, adjacent in time). Splitting at
window level puts near-duplicates on both sides, so validation loss no longer measures
generalisation and every early-stopping and model-selection decision is made on a leaked signal.

**Why it matters here.** This is not hypothetical: the authors' released `baseline.ipynb` does
exactly this — `train_test_split(range(len(Y_train_raw)), stratify=..., test_size=0.1)` over
**windows**. It affects only their validation split (the official test split is separate and
segment-defined), so their published test AUCs remain meaningful, but it means the reference
implementation cannot be copied as-is.

**Mitigation.** All splitting is grouped by **`recording_stamp`** (`GroupShuffleSplit` /
`GroupKFold`), not by `segment_id`. Phase 1 showed segment-level grouping is insufficient on
RoboticArm, where many segments are excerpts of one recording (see R9). No window ever crosses a
split boundary from its parent segment, and no segment from a parent recording crosses one either. Splits are computed once from
metadata, written to `configs/splits/`, versioned, and loaded — never recomputed ad hoc.

**Enforcement.** `tests/test_splits.py`: assert
`set(train.recording_stamp) ∩ set(valid.recording_stamp) == ∅` for every generated split, assert
the same for `segment_id`, and assert every window's `segment_id` lies in exactly one split.

---

## R2 — Normalisation / feature statistics fitted on non-training data — **CRITICAL**

**Failure.** Computing means, standard deviations, min/max, PCA bases, scaler parameters or
feature-selection statistics over data that includes validation or test segments.

**Why it matters here.** There is a subtlety specific to this dataset. The reference baseline
used `normalisation: 'std_window'` — each 100 ms window standardised by **its own** mean and
std per channel. That is leak-free by construction, but it **removes absolute amplitude and
energy within the window**, and amplitude is exactly what a loosened bolt or a tightened belt
changes. This is a plausible partial explanation for the weak microphone-subvector results in the
published masked-input baseline — though that baseline trained no microphone-only model, so the
explanation is a hypothesis to test in Phase 3, not an established cause. So the
naive "safer" choice is also potentially the crippling one, and the trade-off must be measured
rather than assumed.

**Mitigation.**
- Any statistic used by the model is fitted on **training segments only** and stored as a fitted
  artefact with provenance metadata (which split, which segments, which config hash).
- Both schemes are evaluated as a documented experimental variable: (a) train-fitted per-channel
  z-score, (b) per-window standardisation. Neither is inherited without justification.
- Target-domain training normals (27 / 18 segments) count as training data when used, and the
  experiment variant that uses them is labelled distinctly (§7 of the research definition).

**Enforcement.** Scalers are fitted only inside a `fit(train_split)` call that physically cannot
see other splits; `tests/test_no_leakage.py` asserts a fitted scaler's recorded segment-id set is
disjoint from validation and test.

---

## R3 — Model selection or hyperparameter tuning against test AUC — **CRITICAL**

**Failure.** Choosing epochs, architecture, learning rate, window size, feature set, aggregation
function or anomaly-score definition by looking at test-set AUC. This is the most likely way this
project would produce inflated results, because it does not feel like cheating — it feels like
iterating.

**Why it matters here.** It is *structurally* tempting on IMAD-DS: training data is normal-only,
so the validation signal is a reconstruction/density loss on normals, which does **not** correlate
reliably with anomaly-detection AUC. The honest position is that model selection is genuinely
hard under this protocol, and that difficulty must appear in the report rather than being
resolved by peeking.

**Mitigation.**
- Selection criterion is declared in the config **before** the run and is computed on held-out
  **training normals** only (validation loss, or a normal-only proxy such as validation-score
  dispersion). Never test AUC.
- The test set is evaluated **once per finalised configuration**. Re-running a configuration on
  test after seeing its test result, with any change, converts it into a tuning run and must be
  reported as such.
- An experiment log (`reports/experiment_results/`) records every test-set evaluation with a
  timestamp and config hash, so the number of test looks is auditable rather than forgotten.
- Where a decision genuinely cannot be made without test feedback, it is made once, documented as
  a *selection on test*, and the affected result is explicitly flagged as optimistic.

**Enforcement.** Convention plus the append-only evaluation log; the reviewer can count test looks.

---

## R4 — Anomaly threshold chosen on test data — **HIGH**

**Failure.** Reporting "best F1" or "best accuracy" by sweeping the decision threshold on the test
set. This can move F1 by tens of points and is indistinguishable from a real improvement in a table.

**Why it matters here.** The designed test prevalence—RA 234 normal/232 anomaly and BM 156/156
pooled, with RA target-only 20/116—is completely unrepresentative of a plant where anomalies are
rare. Swept-threshold F1 would therefore be optimistic and prevalence-specific.

**Mitigation.** The threshold is the q90 (`higher`) of **source-validation normal segment scores**;
anomaly is strict `score > threshold`. Any change must precede all test results. AUROC and AUPRC are
primary because they are threshold-free; pAUROC is secondary.

**Enforcement.** The low-level quantile primitive remains a pure numeric helper. The official
Phase-3 threshold entry point accepts a structured score batch and verifies experiment identity,
source domain, `source_validation` role, normal labels, unique segment ids and the
`higher_is_more_anomalous` convention before calling it. Phase-3.0 planted-failure tests reject
test, target-adaptation, anomalous, mixed-role and stale-provenance batches.

---

## R5 — Anomaly labels used during training — **CRITICAL**

**Failure.** Training on, or selecting with, the ground-truth `anomaly_label`. This breaks the
dataset's stated protocol and makes results incomparable to every published number.

**Why it matters here.** The risk enters through the model ladder, not through carelessness:
XGBoost, LightGBM and any "1D CNN classifier" need labels, and the only labels in IMAD-DS are in
the test set. See §11 of the research definition, where the tree-based family is **deferred to
Phase 1** precisely because the only label-free way to use it rests on unverified metadata
assumptions.

**Mitigation.**
- The canonical manifest retains anomaly labels for audit/evaluation. The validated model-facing
  `EstimatorTrainingData` exposes explicitly selected feature matrix `X` and reproducibility ids,
  but no anomaly-label or target field, so labels are not supplied to the estimator by the official
  path.
- Self-supervised targets may use `domain_shift_op` / `domain_shift_env` for **training** segments
  only, and those are condition labels, not anomaly labels. Note this is permission for a *method*
  that Phase 0 has not adopted — no self-supervised entry is approved until Phase 1 resolves Q3/Q4.
- If a supervised oracle is ever reported, it lives in a separate table under an explicit
  "not comparable to the unsupervised protocol" heading.

**Enforcement.** `tests/test_phase30_data_contract.py` asserts the label-free estimator interface
and plants test, anomalous, blocked, stale, wrong-protocol and wrong-regime rows. The contract also
binds the processed cohort to the accepted v3 segment identities.

---

## R6 — Operating-condition metadata acting as a shortcut at test time — **HIGH**

**Failure.** Feeding `domain_shift_op` / `domain_shift_env` (or anything derived from them, such
as a one-hot condition vector or a domain id) into the model at test time. If anomaly and normal
test segments are not perfectly balanced across conditions, the model can partly read the answer
off the metadata instead of the signal.

**Why it matters here.** Balance across conditions within the test set is **unverified** —
open question Q4. It cannot be assumed just because the normal/anomaly *counts* are balanced.

**Mitigation.**
- Condition metadata is never a model input. It is used for (a) self-supervised targets on
  training data, (b) stratification, (c) result slicing.
- Phase 1 explicitly cross-tabulates `anomaly_label × domain_shift_op × domain_shift_env` on the
  test set and reports whether condition predicts the label. If it does, every condition-aware
  method must be re-examined.

**Enforcement.** Feature-builder signatures exclude metadata columns; Phase 1 audit report
includes the cross-tabulation as a required output.

---

## R7 — Window-level metrics presented as segment-level — **HIGH**

**Failure.** Computing AUC over windows rather than segments. Each segment contributes many
windows, so window-level AUC is computed over a far larger set of non-independent samples. It
produces tighter-looking, more impressive, and incomparable numbers.

**Mitigation.** Segment-level is the **only** headline metric, aggregated per the protocol
(median of window scores, with mean/trimmed variants as a documented experimental choice).
Window-level results may appear as diagnostics, always labelled "window-level, diagnostic only,
not comparable to published AUC".

**Enforcement.** The evaluation module returns segment-level metrics by default; the window-level
entry point is named so that a window-level number cannot be mistaken for the headline one.

---

## R8 — Small test sets, multiple comparisons, and seed cherry-picking — **HIGH**

**Failure.** Declaring a winner on a difference that is inside the noise. A comparison grid of
~8 model families × 2 machines × 3 domain conditions × several sensor configurations × several
corruption levels yields hundreds of numbers; several will look good by chance alone. Re-running
with different seeds and reporting the best is the same error in a different costume.

### The inferential decision rule — paired, not marginal

Candidate systems are evaluated on **the same test segments**. Their errors are therefore
correlated, often strongly, and any procedure that compares two systems through their *individual*
(marginal) confidence intervals is the wrong test and is not used anywhere in this project.

> **Historical note.** The first draft of this register contained two marginal-CI decision rules.
> Both were withdrawn on review as statistically incorrect, and no marginal-CI decision language
> survives in this project. The reasoning is kept here so the error is not reintroduced: shared
> segment-level difficulty cancels in a paired difference, so two systems can differ significantly
> in the paired sense while their marginal intervals overlap substantially. A marginal-interval
> rule would therefore have discarded real effects.

**Pre-registered rule (frozen). This is the sole head-to-head decision procedure.**

1. For a head-to-head comparison of systems A and B, compute a **paired recording-cluster
   bootstrap** of metric differences (primary: ΔAUROC and ΔAUPRC; secondary: ΔpAUROC).
2. Each RoboticArm replicate resamples **parent-recording IDs** with replacement and retains every
   child segment from a selected parent. Both systems receive identical parent draws. Windows are
   never resampling units, and sibling segments are never treated as independent. BrushlessMotor
   uses the same rule at recording/segment level because that mapping is one-to-one.
3. **Stratification, by analysis:**
   - **S+T primary analysis** — resample parent clusters within
     **domain × anomaly status × operating condition**. Each stratum retains its parent count.
   - **Source-only or Target-only analyses** — use anomaly status × operating condition within
     that domain.
4. **Superiority is declared only when the bootstrap CI of the paired difference excludes zero in
   the favourable direction.** Report the point estimate of Δ with its interval.
5. Marginal per-system CIs are still reported, because a reader needs the absolute uncertainty of
   each system — but they are **descriptive only**, and no decision is ever taken from them.
6. **Seed handling (frozen algorithm, see below).**

### Seed and bootstrap algorithm — frozen

Exactly one procedure, pre-registered before any result exists. No alternatives are retained.

```
Fixed in config before running:  K = 5 seeds (declared explicitly, e.g. 0,1,2,3,4)
                                 B = 10,000 bootstrap replicates
                                 stratification per rule 3 above

for b in 1..B:
    draw one stratified resample of parent IDs         # shared by BOTH systems
    expand every selected parent to all child segments
    for each system S in {A, B}:
        for each seed k in 1..K:
            m[S,k] = metric(S, seed k, resampled segment IDs)
        M[S] = mean over k of m[S,k]                   # seed-averaged system metric
    delta[b] = M[A] - M[B]

report: point estimate of delta, and the percentile CI of delta over b
```

For BrushlessMotor, parent IDs and segment IDs are identical. The estimand is therefore **the
seed-averaged system**, which is the right target: the claim is
about the method, not about one lucky initialisation.

**Stated limitation of this choice.** The interval propagates *segment-sampling* uncertainty while
treating the seed set as fixed; it is a CI conditional on those K seeds and does **not** include
seed-sampling uncertainty. This is accepted deliberately, because with K = 5 any attempt to model
the seed distribution would rest on five points. The mitigation is transparency, not a second
procedure: the ordinary **across-seed mean ± SD** for each system is reported separately in every
results table, and where that SD is large relative to the observed Δ, the comparison is explicitly
flagged as seed-sensitive in the text. A reader can then see exactly what the interval does and
does not cover.

**No stronger hierarchy is adopted.** A hierarchical / mixed-effects model crossing seed and
segment variance would be the statistically richer option, but with K = 5 seeds its variance
components would be poorly identified, and it would add distributional assumptions to a comparison
that the paired bootstrap already handles assumption-lightly. The simpler frozen procedure above,
plus separate reporting of across-seed SD, is the deliberate choice.

### Sample sizes actually in play

The pre-declared primary metric is segment-level AUC in the **S+T** condition, per machine.
S+T pools both domains, so its sample size is the sum:

| Reporting condition | Normal segments | Anomalous segments |
|---|---|---|
| **Robotic Arm, S+T (primary)** | **234** (214 source + 20 target) | **232** (116 + 116) |
| **Brushless Motor, S+T (primary)** | **156** (78 + 78) | **156** |
| Robotic Arm, Source / Target (secondary) | 214 / 20 | 116 / 116 |
| Brushless Motor, Source *or* Target (secondary) | 78 | 78 |

The target-normal test set is small because 96 released-test sibling segments are withheld to keep
the 18 adaptation parents out of evaluation. Target-only intervals will therefore be wide and AUPRC
reflects a 20-normal/116-anomaly evaluation prevalence that is not operational prevalence.

`[PHASE 2.2]` Those 96 siblings are **blocked**, not trained on. Phase 2.1 read parent closure as
the Regime-T training budget; Phase 2.2 separates *target examples actually consumed* (RA 27,
BM 18) from *same-parent siblings blocked to prevent leakage* (RA 96, BM 0). The blocked segments
enter no fit, no calibration, no validation and no metric in either regime, so the evaluation
cohort above is unchanged — only the training-role labels moved.

### Hanley–McNeil figures — planning illustration only

These are **not** the decision rule and must not be used as one. They exist to give a rough sense
of how much resolving power the test sets have, for planning purposes:

| Condition | AUC ≈ 0.90 | AUC ≈ 0.75 | AUC ≈ 0.60 |
|---|---|---|---|
| RA S+T (234 vs 232; independence approximation only) | ±2.9 | ±4.4 | ±5.1 |
| BM S+T (156 vs 156) | ±3.5 | ±5.4 | ±6.3 |
| RA per-domain (116 vs 116) | ±4.1 | ±6.3 | ±7.3 |
| BM per-domain (78 vs 78) | ±5.0 | ±7.6 | ±8.9 |

(95% CI half-widths in AUC points, single-system marginal, assuming independent samples.)

**These are planning illustrations only.** They say something about how precisely a *single*
system's absolute AUC can be pinned down, which is useful for anticipating how large an experiment
grid is worth running. They say **nothing** about whether two systems differ, they set **no**
threshold on a difference, and **no number in this table may be quoted as a minimum detectable
effect.** Because shared segment-level difficulty cancels in a paired difference, the paired
procedure can resolve differences considerably smaller than these widths — how much smaller is an
empirical question that the paired bootstrap answers directly, per comparison. The only planning
message here is that **domain-specific estimates are less precise; target-only RA is particularly
weak with only 20 normal segments, and cluster-bootstrap intervals—not this table—govern reporting.**

**Mitigation.**
- **Pre-declared primary metrics**: segment-level AUROC and AUPRC in S+T, per machine. pAUROC@0.1
  is secondary.
- **Paired bootstrap** for every head-to-head claim, per the frozen rule and algorithm above.
- **K = 5 fixed seeds** per configuration; across-seed mean ± SD reported separately in every
  table. Never report a single run as the result.
- Where a paired interval contains zero, the correct statement is "the paired difference does not
  exclude zero", and that is an acceptable project outcome.

**Enforcement.** The comparison function takes two systems' per-segment scores plus shared
parent-cluster draws and returns a paired Δ interval; there is no code path that compares systems by
their marginal intervals. `tests/` asserts identical parent draws and forbids windows or RoboticArm
sibling segments as independent resampling units.

---

## R9 — Near-duplicate recordings across splits — **CRITICAL, CONFIRMED REALISED (Phase 1)**

> **PHASE 1 UPDATE — this risk is no longer hypothetical, and it is worse than "medium/high".**
> RoboticArm's 2,303 segments are consecutive 7.5 s excerpts of only **101 continuous recordings**
> (median 16, max 48 per recording). **62 of 101 recordings straddle the official train/test split**,
> and **212 of 232 normal test segments (91.4%) have a sibling excerpt from the same recording in
> the training set.** Because training is normal-only, **0 of 232 anomalous test segments can** —
> so the violation is asymmetric and falls on the normal class only, creating a **credible risk of
> optimistic bias whose magnitude has not been measured**. Verified from timestamps: test
> excerpt `dataLog_8` (82.7–90.2 s) lies ~2.5 s from training excerpts `dataLog_7` and `dataLog_9`
> in one continuous stream at the same load.
>
> **Consequences, all binding on Phase 2:**
> 1. Any RoboticArm metric computed on the **official** split carries this risk and must be
>    labelled as such. Whether the published 91.62 AUC is inflated, and by how much, is a
>    **hypothesis to be tested** by running one baseline under both splits
>    (`docs/evaluation_protocol.md` §5), not an established fact.
> 2. MechaSight must construct a **recording-disjoint** split for RoboticArm and report it as the
>    primary result, with the official split reported alongside only for comparability.
> 3. Grouping by `segment_id` is **not sufficient**; grouping must be by `recording_stamp`.
> 4. BrushlessMotor is unaffected (1,593 segments from 1,593 recordings, zero straddling), so the
>    two machines need different split handling — another reason never to pool them.
>
> Evidence: `reports/dataset_audit/dataset_summary.md` (Q8), `recording_times.csv`,
> `audit_findings.json` temporal_structure. Detector and tests in `src/data/temporal.py` and
> `tests/test_temporal.py`.

### Original Phase 0 text (preserved)

**Failure.** If the released train and test sets contain segments cut from the same underlying
continuous recording, or repeated runs of an identical condition, the test set is partly seen.

**Why it matters here.** The split was made by the dataset authors and is respected as given —
but its internal structure is unverified. Segment ids may encode a parent recording (the id is
derived from the microphone filename), which would make this checkable cheaply.

**Mitigation / audit task for Phase 1.** Parse `segment_id` structure; check for a parent-recording
component; check timestamp ranges for overlap between train and test segments; report duplicate
or contiguous-in-time segments straddling the split. Findings go in the Phase 1 audit, and if
overlap exists it is reported as a dataset property, not silently absorbed. This is open question Q8.

---

## R13 — Class/recording-time confound (BrushlessMotor) — **HIGH, CONFIRMED (Phase 1)**

**Failure.** The class label is perfectly predicted by recording time, so a detector can score well
by responding to session drift rather than to the mechanical fault.

**Why it matters here.** All 1,437 BrushlessMotor normal segments were recorded 10:20:21–16:40:50
and all 156 anomalous segments 16:48:07–18:22:40 on a single day — **zero overlap, 7.3-minute gap**.
No normal was recorded after the first anomaly. The two anomaly types are also blocked from each
other (belt 16:48–17:33, magnetic 17:38–18:22), so per-type comparisons carry the same confound.
Thermal state, mechanical settling, sensor bias drift and ambient conditions all vary over an
8-hour session and are inseparable from the fault effect on this machine.

**This is not a criticism the authors could easily have avoided** — inducing a physical fault
requires stopping and modifying the machine. But the complete absence of interleaved normal/fault
blocks means the confound cannot be removed after the fact.

**Mitigation — bounding and disclosure only. This confound cannot be removed.**

Fault effects and acquisition-session effects are **not identifiable** for BrushlessMotor: any
difference between the normal and anomalous populations may be caused by the induced fault, by
session drift, or by any mixture, and the dataset contains no interleaved normal/fault blocks that
would separate them.

- Report it as a limitation wherever a BrushlessMotor result is presented, using the wording frozen
  in `docs/evaluation_protocol.md` §6.
- **Temporal-drift diagnostic (proposed, not implemented) — explicitly NOT a deconfounding.** Using
  training normals only, train on early-session normals and evaluate on late-session normals. This
  measures whether *detectable drift exists inside the normal training data*, which is a useful
  lower bound. It **cannot** establish that the later anomaly session is free of independent session
  effects, because the anomaly session is a different, later period containing no healthy recordings
  against which drift could be observed. A null diagnostic result would therefore prove nothing
  about the anomaly session.
- Never claim that a BrushlessMotor detector responds to the fault rather than to session state.
  The honest claim is that it separates the two populations, with the cause unidentifiable.

**Enforcement.** `src/data/temporal.py` measures the confound; `tests/test_temporal.py` checks the
detector fires on blocked classes and stays quiet on interleaved ones.

---

## R10 — Corruption experiments tuned after seeing their results — **MEDIUM**

**Failure.** Choosing which corruption types, which modality, and which severity levels to report
after seeing which ones make reliability-aware fusion look best.

**Why it matters here.** This project's central hypothesis is *about* corruption robustness, so
this is the specific place where motivated reasoning would do the most damage.

**Mitigation.** The corruption grid (types, target modalities, severity sweep) is written into a
config file and committed **before** the corresponding experiments run. Every level in the sweep
is reported, including the ones where the proposed method loses. Adding a corruption type later
is allowed but must be committed as a dated amendment.

**Enforcement.** Config committed ahead of results; git history makes the ordering checkable.

---

## R11 — Preprocessing choices informed by test-set inspection — **MEDIUM**

**Failure.** Looking at test segments during exploratory data analysis and letting that shape
window length, filtering, band selection, or feature design.

**Mitigation.** Phase 1's exploratory audit is performed on **training** segments plus test
*metadata only* (counts, labels, condition distributions). Test signal content is not plotted or
inspected until the protocol is frozen. Where a test-set property must be known for validity
checking (e.g. R6's cross-tabulation), only metadata is used — never waveform content.

---

## R12 — Reporting failures that are not leakage but mislead — **MEDIUM**

Grouped, because each would distort the reader's conclusion:

- **Averaging the two machines** into one headline number. RA 91.62 vs BM 58.95 in the reference
  baseline; the mean describes neither. → Always report per machine.
- **Calling an anomaly score a "confidence" or a "probability"** without calibration. → Banned
  in code, tables, figures and any future UI unless an explicit calibration is fitted on training
  normals and its reliability evaluated.
- **Comparing source-only training against source+target training** as though they were the same
  experiment. → Labelled distinctly wherever reported.
- **Comparing our numbers to the paper's Table 4** without matching the protocol (same windowing,
  same aggregation, same segment-level scoring). → Any comparison states which protocol details
  match and which differ.
- **Presenting masked-input single-sensor results as single-sensor models.** This is the error
  latent in the reference paper's `S-x` rows. → MechaSight trains genuine single-modality models
  and labels masked-input results as masked-input results.

---

## Enforcement summary — what Phase 2 must actually build

| Artefact | Purpose |
|---|---|
| `configs/splits/` (versioned, generated once from metadata) | R1, R2 |
| `src/data/splits.py` with **recording-grouped** splitting only | R1, **R9** |
| A recording-disjoint RoboticArm split, reported as primary alongside the official one | **R9** |
| Early/late-session drift *diagnostic* from training normals (bounds drift; does not deconfound) | **R13** |
| `src/preprocessing/` scalers carrying fitted-on provenance | R2 |
| `src/experiment/data.py` label-free estimator-facing contract | R1, R2, R5 |
| `src/experiment/metrics.py` and `bootstrap.py` segment metrics / cluster CIs | R7, R8 |
| Provenance-aware `select_protocol_b_threshold` wrapping the numeric q90 helper | R4 |
| Append-only `TestLookLedger`, required by the official test `EvaluationGate` | R3 |
| Phase-2 split tests plus `tests/test_phase30_*.py` planted safety failures | R1, R2, R4, R5, R8 |
| Corruption grid committed to `configs/` before robustness runs | R10 |
