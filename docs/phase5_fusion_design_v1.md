# Multi-Sensor Fusion Development — Scientific Design v1 (Phase 5.1)

<!-- Legacy roadmap name: Phase 5.1 -->

**Status:** PROPOSED — NOT AUTHORIZED FOR EXPERIMENTAL EXECUTION.
**Machine-readable design:** `configs/phase5/fusion_design_v1.json` (same status; it holds the
exact unit registry and counts).
**Base commit:** `f6a33af4db0a610947ad8f7de7fa5d9081873232` on `phase5-multisensor-fusion`.
**Scope of 5.1:** design and planning only. No fusion code, no model fit, no protected read,
no score, no ledger change.

---

## 1. Starting point: what exists and what does not

All facts below were read from non-protected artifacts (pre-test bundles, configs, ledger
metadata, the dataset audit). No protected score, label or common-evaluation cache was read.
The aggregate Phase-4 conclusions in `docs/phase4_result_report.md` are cited only as context;
they choose nothing in this design (section 13 lists every place they could have and did not).

### 1.1 Reusable without new work

| Asset | Location | What it gives Phase 5 |
|---|---|---|
| 156 frozen single-modality detectors | `reports/phase4_pretest/experiments/<id>/`, models in `data/interim/phase4/models/` | L0 comparators; the inputs to L2 |
| Source-validation segment scores for all 156 | `validation_scores.csv` in each bundle | L2 calibration and fused thresholds **without any new inference** |
| Common validation cohort | verified: for every one of the 52 (machine, regime, family, seed) units, mic/acc/gyro scores cover one identical set of normal source-validation segments (RoboticArm 186, BrushlessMotor 126), identical for S and T | segment-level score fusion is well defined |
| Score orientation | all 156 thresholds declare `higher_is_more_anomalous` | no sign flips needed |
| 156 sealed Phase-4 test score tables | `reports/phase4_protected/staging/scores/` (hash-verified) | L2 test tables derive from these; no re-inference |
| 28 all-channel classical detectors | Phase 3.1 validation bundles `reports/phase3_1/validation_only/`, Phase 3.2 sealed test scores `reports/phase3_2/experiments/` | **these are L1** for statistical_reference, Isolation Forest and OCSVM (section 3.2) |
| Row-aligned spectral caches | `data/interim/phase4/spectral/<machine>/<regime>/{training,source_validation}.spectral.h5` | L1 autoencoder input: row *i* of mic/acc/gyro is one `window_id` |
| 98-feature segment artifacts | `data/interim/phase31_primary/` (training 1137–1555 segments per unit) | L1 classical input |
| Analysis machinery | `src/experiment/phase4_analysis.py`, `src/experiment/bootstrap.py` | R8 seed averaging, shared paired cluster bootstrap |
| Provenance and protection | Phase-4 bundle/manifest conventions; Phase-5.0 `verify_ledger_history`, `authorized_protected_read`, tripwires | protected execution with no second framework |

### 1.2 Must be built or fitted

- **New fits (24):** an all-channel Mahalanobis detector (4) and an all-channel dense autoencoder
  (20) — the two families with no all-channel detector.
- **L2 machinery:** source-validation calibration, fusion of the 4 modality subsets, fused q90
  thresholds (208 validation operations, no inference).
- **Integrity gate** and its development falsification suite.
- **Paired bootstrap for threshold metrics** (healthy-target FPR, recall): Phase 4 reported these
  as point estimates only.
- **A read guard for sealed historical test score tables.** The Phase-5.0 guard covers test
  inputs but not `reports/phase4_protected/staging/scores/*.csv` or
  `reports/phase3_2/experiments/*/scores.parquet`. L2 development must not be able to read them
  by accident, so this guard comes first (5.2).
- One protected script (score-all / analyse-once) on the Phase-4 pattern.

### 1.3 What the dataset cannot support

The dataset audit (`reports/dataset_audit/dataset_summary.md`) found, across all 11,688 files
and 27,272 channel instances: **0 NaN, 0 ±Inf, 0 constant channels, 0 duplicate timestamps,
0 non-monotonic timestamps, 0 gaps**. There are **no naturally occurring or labelled sensor
faults**. The only injected degradation is environmental noise mixed into audio, which is a
benign *operating/environmental* shift, not a sensor failure. Training and source validation
contain normals only, so **no anomaly is available outside the protected cohort**.

Consequences that drive the design:
- graded sensor-quality estimation cannot be learned or validated on this data;
- any reliability mechanism can only be exercised under **controlled synthetic degradation**;
- whether a mechanism suppresses genuine faults cannot be tested on real faults before the
  protected evaluation; it can only be (a) excluded by construction, (b) checked on synthetic
  fault-like signals in development, and (c) checked as a pre-registered protected diagnostic.

---

## 2. Scientific question, hypotheses, estimand

**Question.** Under leakage-controlled Protocol B (few-shot split v3, Regimes S and T, frozen
common test), and separately for RoboticArm and BrushlessMotor:

1. Does combining microphone, accelerometer and gyroscope evidence — by feature-level fusion
   (L1) or by source-validation-calibrated score averaging (L2) — change **ranking
   discrimination**, **healthy-target false-alarm rate** and **anomaly recall at a frozen
   source-validation threshold** relative to each matched single-modality detector?
2. When one modality is lost, how much of that capability does an **integrity-gated** late
   fusion (L3) retain?

**Hypotheses** (all two-sided; no prior for or against fusion):
- **H1** L2 (all three modalities) differs from each matched single modality — mic, acc and gyro
  individually — on the ranking axis and/or the operating axis.
- **H2** L1 (early fusion) differs from L2 (late fusion) at matched family and seed.
- **H3** Under loss of one modality, gated fusion of the remaining two retains or loses
  capability relative to intact L2.

**Estimand.** Δ(metric) = metric(fusion arm) − metric(comparator) on the common test cohort,
seed-averaged per bootstrap replicate (frozen R8 rule), conditional on the observed evaluation
cohort; 95 % percentile interval from the shared paired class-stratified cluster bootstrap
(seed 1995, 10,000 replicates; parent recording is the unit for RoboticArm, recording=segment
for BrushlessMotor).

**Population.** Common test: RoboticArm 466 segments (234 normal / 232 anomalous; healthy
target slice 20 segments from 6 parents); BrushlessMotor 312 (156 / 156; healthy target 78).
Calibration population: normal source-validation segments only (RoboticArm 186,
BrushlessMotor 126).

**What "shift" means here.** Phase 5 adds no shift-adaptation mechanism. It asks whether
*combining sensors* changes behaviour under the operating/environmental shift already in the
benchmark (target slice, healthy-target false alarms) and under a sensor-loss shift. The only
adaptation lever remains Regime T's released target normals, unchanged from Phase 4.

---

## 3. The baseline ladder

### 3.1 L0 — single modality (exists; not retrained, not rescored)

The 156 Phase-4 detectors. Every fusion arm is compared with **mic, acc and gyro individually**
at matched machine, regime, family and seed. The family-matched best Phase-4 modality is
reported only as a labelled **post-hoc historical benchmark**: it was selected with previously
observed protected results and is never the sole comparator.

### 3.2 L1 — feature-level (early) fusion

One detector per unit trained on all three modalities jointly, with the same recipe as its
single-modality counterparts.

**Classical families (98 features: mic 14 + acc 42 + gyro 42; train-only z-score).**
- *statistical_reference, Isolation Forest, OCSVM — reuse, no new fit, no new look.* The 28
  Phase-3.2 all-channel detectors are exactly these L1 detectors. A field-by-field comparison of
  their configs with the Phase-4 classical configs shows **zero differences** in split version,
  split-manifest hash, preprocessing config id and hash, normalisation, threshold policy,
  aggregation, score direction, window/stride and every model hyperparameter; the ledger shows
  the **same test-cohort hash** for both stages; Phase-4 feature subsets are the 98 Phase-3.1
  columns restricted by prefix. Refitting would reproduce the same detectors and spend 28
  duplicate protected looks. They were already looked at (Phase 3.2), so they enter Phase 5 as
  re-analysed comparators, not as new looks (section 9).
- *Mahalanobis (new fit, 4 units).* LedoitWolf (`assume_centered=false`) on the 98 standardised
  features, squared Mahalanobis distance. With 1137–1555 training segments per unit, n/p ≥ 11.6,
  and shrinkage keeps the covariance well conditioned.

**Dense autoencoder (new fit, 20 = 2 machines × 2 regimes × 5 seeds).**

| Check | Finding |
|---|---|
| Correspondence | one `window_id` per row across mic/acc/gyro in the existing caches (verified at entry review; asserted again in 5.2) |
| Dimensionality | RoboticArm 800 + 1059 + 1059 = **2918**; BrushlessMotor 800 + 1068 + 1068 = **2936** |
| Scaling | train-only per-bin z-score; identical to concatenating the three Phase-4 standardisers (asserted in 5.2) |
| Loss weighting | plain MSE over all bins — the naive choice. The microphone supplies 27.4 % / 27.2 % of bins, the IMU channels 36 % each. Not re-weighted: a re-weighted loss would be a second, non-naive arm |
| Capacity | frozen Phase-4 template `d→256→64→16→64→256→d`. One 16-d bottleneck for three modalities versus three 16-d bottlenecks behind L2. Accepted as the frozen recipe, stated as a limitation, **not tuned** (tuning on validation reconstruction error is not tuning on detection, but it would add an unmatched search) |
| Training | Adam 1e-3, batch 256, ≤ 200 epochs, patience 10 on source-validation window MSE, best checkpoint, seeds 0–4 |
| Scores | window = MSE; segment = median of windows; threshold = q90 of own source-validation scores |
| Compute | ~115k × 2918 float32 training windows ≈ 1.35 GB in memory; time measured in 5.3 (U4) |

**No family is excluded from L1** — every family can consume the joint input, so the C8 question
(selection by performance) does not arise.

### 3.3 L2 — score-level (late) fusion

A fixed rule over the frozen Phase-4 per-modality segment scores of one unit; nothing is
trained.

1. **Pairing.** Seed *s* of mic, acc and gyro are fused together (deterministic families: seed 0).
2. **Calibration (source validation only).** For each modality, robust
   z = (score − median_v) / IQR_v with median_v and IQR_v from that modality's frozen
   source-validation segment scores. All 156 bundles hold these scores, so calibration needs no
   new inference and leaves every Phase-4 artifact unchanged.
3. **Pooling.** Arithmetic mean of the calibrated scores over the subset.
4. **Subsets.** {mic, acc, gyro}, {acc, gyro}, {mic, gyro}, {mic, acc}.
5. **Threshold.** q90 (method `higher`) of that subset's **fused source-validation scores** —
   never inherited from a single modality. Calibration and threshold use the same validation
   cohort, exactly as Phase 4 fitted thresholds in-sample; validation FPR is ≈ 10 % by
   construction.
6. **Orientation.** All inputs are `higher_is_more_anomalous`; so is the fused score.
7. **Missing / degenerate.** No missing scores are expected. A missing or non-finite score, or
   IQR_v = 0, is a technical defect → STOP; never imputed or repaired.

*Why robust z and not an empirical CDF:* an ECDF fitted on 126–186 validation segments saturates
at 1 for every test score above the validation maximum, tying exactly the segments that matter
most. Robust z is monotone and unbounded, so each modality's own ordering survives. *Why equal
weights:* learning weights would need anomalies, and none exist outside the protected cohort.

**Max pooling is excluded.** L3 is a gated *mean*, so L2-mean is its exact ablation. Max pooling
lets any single failed or fooled modality decide by construction and collapses completely under
modality loss. It would add 208 looks for a rule whose failure mode is known a priori. This is an
a-priori property, not a Phase-4 observation.

### 3.4 L3 — integrity-gated score fusion (the reliability-aware arm)

**Operational definition of reliability used here:** a modality is reliable in a segment if its
raw signal is a *valid measurement* — finite, not dead or stuck, correctly timed. Reliability is
decided from signal properties that a mechanical fault does not produce. Nothing weaker is
supported by the data (section 1.3), and nothing stronger is claimed.

- **Weight.** w_m(x) = 1 if modality m passes the integrity gate on segment x, else 0.
- **Fused score.** Mean of calibrated scores over modalities with w_m = 1.
- **Threshold.** q90 of the source-validation fused scores **of that available subset**
  (per-availability thresholds, all fitted on validation).
- **All modalities gated.** The segment is reported as SENSOR-INVALID with no anomaly decision
  and counted separately; it is never silently scored.

**Integrity gate v1** (on the modality's raw post-startup-transform signal over the segment's
common coverage; parameters fixed now from the physics of a dead or stuck sensor, not fitted):

| Rule | Invalid when |
|---|---|
| G1 non-finite | any non-finite sample or timestamp |
| G2 flatline | ≥ 50 % of samples, in any channel, lie in runs of identical consecutive values lasting ≥ 20 ms (dead, stuck-at, zero dropout, saturation plateau) |
| G3 timing | non-monotonic or duplicate timestamps, or the modality's own stream covers < 90 % of the segment's nominal duration |

**Transduction boundary (C3).** The gate reads only the segment being scored, and every
parameter (20 ms, 0.50, 0.90, calibration medians/IQRs, subset thresholds) is fixed on
training/validation before evaluation. That is frozen per-sample inference. It would become
transductive if any protected-cohort quantity — a test-set median, a test-set rate of gating, a
threshold re-fitted on test scores, a normalisation over the test batch — influenced a weight,
score or threshold. None does, and the protected script will hold no code path that could.

**How L3 reaches the protected evaluation without new inference.**
- *Clean condition.* On clean data every modality passes, so L3 = L2(all three). Before any ledger
  append, score-all scans the raw common-test inputs with the gate. The matrix assumes zero gated
  segment-modalities, as the audit implies. If any segment-modality is gated, score-all **STOPS
  before the ledger append** and the design is amended; no table is added silently.
- *Loss condition.* Complete loss of modality m (dead, disconnected or stuck sensor) is gated with
  certainty (verified on synthetic inputs in 5.4). L3 under loss of m is therefore exactly the L2
  detector of the remaining pair with its own validation threshold. **The three L2 pair tables
  are the L3 loss-condition detectors.** No corrupted protected input is created.

The protected evidence for L3 is therefore the measured capability of each gated subset; the
evidence that the gate selects the right subset is developmental (section 5).

---

## 4. Reliability versus anomaly novelty

The entry review's candidate R2 weighted each modality by its input-space novelty relative to
frozen training references. It was examined against the six required questions.

| Question | Answer for R2 (novelty) | Answer for the integrity gate |
|---|---|---|
| 1. Why is the statistic reliability and not anomaly evidence? | It is not. Distance from the training-normal manifold is exactly what each per-modality detector already scores; R2 is the anomaly score in another form. | G1–G3 measure finiteness, dead/stuck runs and timing — properties a loose screw, imbalance or bearing fault does not create in a working sensor. |
| 2. Could it suppress a sensor that is correctly detecting a fault? | **Yes, by construction**: the modality that sees the fault most clearly is the most novel and gets the least weight. | Only in one case: a fault violent enough to **saturate** the sensor for ≥ 50 % of the segment. The audit rules out fully constant channels but not partial plateaus; this is measured in 5.4 (U1) and checked as a protected diagnostic. |
| 3. How would the risk be tested on train/validation? | It cannot: there are no anomalies outside the protected cohort. Synthetic "faults" would test the synthetic generator, not R2. | Negative controls: fault-like but valid perturbations (gain ×0.5/×2/×3, offset, noise at 20/10/0 dB, impulses, tones, natural target-domain audio) must never trigger the gate. |
| 4. Per segment with frozen references? | Yes. | Yes; it uses no reference distribution at all. |
| 5. Test-cohort adaptation? | Not needed. | None. |
| 6. Computational/calibration cost? | Cheap, but needs a novelty reference and a weighting curve, and that curve is a free parameter with no data to set it. | One linear pass over raw samples; three fixed parameters. |

**Verdict.** R2 is rejected as a reliability mechanism. It would suppress true positives by
design and cannot be falsified before the protected test. Inter-modality disagreement weighting
(former R3) is rejected for the same reason: when a fault is visible in one modality only, the
correct modality is the disagreeing one. Down-weighting audio under shift is rejected because it
was suggested by observed Phase-4 protected results (section 13). The integrity gate is the
strongest mechanism this dataset can support without inventing quality labels. It is reported as
**integrity gating validated on synthetic failures**, not as sensor-quality estimation.

---

## 5. Development falsification (5.4; training/validation and synthetic inputs only)

| Test | Data | Pass criterion | On failure |
|---|---|---|---|
| Clean false-trigger | every training and source-validation segment-modality, both machines, including the released target-normal training segments and target-domain audio | 0 triggers; report the maximum flatline fraction (margin to 0.50) | STOP; parameters are not retuned |
| Positive controls | one modality of source-validation segments replaced by constant (dead), zero dropout, stuck-at-last-value, saturation plateau, non-finite samples or a truncated stream, each at 10/25/50/75/100 % coverage | gate fires at ≥ 50 % flatline coverage and for any non-finite or truncated stream; does not fire below 50 % | STOP |
| Negative controls (no fault suppression) | gain ×0.5/×2/×3, bias offset, additive Gaussian noise 20/10/0 dB, impulse train, added tone, natural target-domain audio | gate never fires | STOP: the gate would be reading signal content |
| Ungated failure demonstration | each frozen Phase-4 model scores one synthetic constant input (156 scorings, no data); the constant is calibrated and inserted for the lost modality | report source-validation FPR of ungated L2(all three) versus gated L3 (the pair) under loss of each modality | descriptive; motivates the gate, decides nothing |
| Pre-registered protected diagnostic | analyse-once: gated segment-modalities among test anomalies and test normals per machine | expected 0 / 0; every gated anomaly listed individually | reported, never used to change a decision |

Partial degradations (noise, gain, partial dropout below 50 %) that the gate does not catch leave
L3 = L2. Phase 5 makes no claim about them on the protected test.

---

## 6. Outcomes and multi-objective interpretation (C1)

**Ranking axis.** Primary: AUROC on the complete common test. Secondary: AUPRC (with prevalence),
standardised pAUROC at FPR ≤ 0.10, target-slice and source-slice AUROC.

**Operating axis** at the frozen q90 threshold. Primary: healthy-target FPR (lower is better) and
anomaly recall (higher is better). Secondary: healthy source-slice FPR, target-slice recall,
precision when defined, F1 by count identity.

**Practical margins** (fixed by engineering judgement now; not derived from any Phase-4 effect
size or variance): AUROC / AUPRC **0.02**; healthy-target FPR **0.10** (two of the 20 RoboticArm
healthy target segments); recall **0.05** (≈ 12 RoboticArm / 8 BrushlessMotor anomalies).

**Per-metric category**
- **IMPROVED** — the 95 % CI excludes 0 in the favourable direction and |point| ≥ margin.
- **DEGRADED** — the 95 % CI excludes 0 in the unfavourable direction and |point| ≥ margin.
- **EQUIVALENT** — the CI lies inside (−margin, +margin), or excludes 0 with |point| < margin
  (detectable but practically negligible).
- **INCONCLUSIVE** — everything else, including fewer than 9,000 valid bootstrap replicates.

**Axes.** R is the Δ-AUROC category; AUPRC and pAUROC are reported, and a disagreement is flagged,
never substituted. O combines healthy-target FPR and recall:

| O | Definition |
|---|---|
| O+ | one IMPROVED, the other IMPROVED or EQUIVALENT |
| O− | one DEGRADED, the other DEGRADED or EQUIVALENT |
| O_tradeoff | one IMPROVED, the other DEGRADED |
| O0 | both EQUIVALENT |
| O? | otherwise (an INCONCLUSIVE component without a decisive conflict) |

**Joint categories**

| Category | Rule |
|---|---|
| BOTH_IMPROVED | R IMPROVED and O+ |
| RANKING_GAIN_OPERATION_PRESERVED | R IMPROVED and O0 |
| OPERATING_GAIN_RANKING_PRESERVED | R EQUIVALENT and O+ |
| TRADE_OFF | (R IMPROVED with O− or O_tradeoff) or (R DEGRADED with O+) or (R EQUIVALENT with O_tradeoff) |
| EQUIVALENT | R EQUIVALENT and O0 |
| UNDERPERFORMANCE | (R DEGRADED with O−, O0 or O_tradeoff) or (R EQUIVALENT with O−) |
| INCONCLUSIVE | R INCONCLUSIVE or O?, reported with whichever axis is decisive |

A TRADE_OFF is a finding reported with both components. It is neither a success nor a failure.
Note that at a fixed threshold, "fewer false alarms and fewer detections together" can simply
reflect a shift in score scale; the report will say so rather than read it as a property of
fusion.

**Across families.** The five family categories are reported side by side for every
(machine, regime, contrast). A machine-level statement requires all five families to be
non-contradicting on the claimed axis (the Phase-4 dominance convention); otherwise the result
is **FAMILY-DEPENDENT**.

**Small-slice rule.** RoboticArm operating results rest on 20 healthy target segments from 6
parents. They are labelled LOW-PRECISION, and no RoboticArm operating claim is made from a point
estimate.

---

## 7. Contrasts and statistics

**Primary (per machine × regime × family; AUROC, healthy-target FPR, recall):**
- **P1** L2(all three) − mic, − acc, − gyro (each individually).
- **P2** L1 − L2(all three).
- **P3** L2(pair without m) − L2(all three), for m ∈ {mic, acc, gyro}: the cost of losing m under
  gating.

**Secondary:** S1 L1 − each single modality; S2 each pair − each of its two members; S3 fusion
arms versus the post-hoc historical best single modality (labelled post-hoc); S4 source-slice and
target-slice versions of the primaries; S5 AUPRC and pAUROC versions; S6 T − S for every fusion
arm.

**Statistics.** Ranking intervals reuse the Phase-4 bootstrap and paired-difference machinery
with the shared draws. Threshold-metric intervals (healthy-target FPR, recall) are **new in 5.3**:
same draws, same R8 averaging. Valid-replicate counts are reported for every interval.

**Multiplicity.** There is no family-wise correction. Inference is conditional on the observed
cohort, and claims require cross-family consistency (5/5 non-contradicting). The report states
the number of contrasts so that chance findings at the 5 % level are expected and read
accordingly.

---

## 8. Leakage and matching rules

- **Protocol B is inherited unchanged:** parent-aware split v3, blocked siblings unused, Regime S
  never touches the released target allocation, the common test is identical for S and T.
- **Calibration and thresholds** use normal source-validation segments only, in both regimes.
  Regime T's target normals train models but never calibrate or threshold, as in Phase 4.
- **No label** is read before analyse-once. **No test statistic** enters any weight, calibration,
  threshold or architecture choice (section 3.4).
- **Matched comparator policy (C5).** A contrast is admissible only between detectors with the
  same machine, regime, family and seed; the same test cohort and bootstrap draws; the same
  threshold rule (q90, `higher`, source validation); the same score orientation; and identical
  segment ids. L1 uses the single-modality recipe unchanged except for input width. L2/L3 use the
  frozen Phase-4 detectors themselves. The post-hoc best modality is never the sole comparator.
- **Development reads.** Before L2 is implemented, sealed historical test score tables get the
  same read guard as test inputs (5.2), so validation-only development cannot see a test score.

---

## 9. Exact experiment matrix

**Unit** = machine × regime × family × seed:
2 × 2 × (1 + 5 + 1 + 1 + 5) = **52** (13 per machine-regime cell).

| Arm | Tables | New fit | New protected inference | New look |
|---|---|---|---|---|
| L0 single modality (Phase 4) | 156 | 0 | 0 | 0 (re-analysed) |
| L1 classical, reused (Phase 3.2) | 28 | 0 | 0 | 0 (re-analysed) |
| L1 Mahalanobis, new | 4 | 4 | 4 | 4 |
| L1 dense AE, new | 20 | 20 | 20 | 20 |
| L2 all three | 52 | 0 | 0 (derived from sealed Phase-4 tables) | 52 |
| L2 pairs = L3 under single-modality loss | 156 | 0 | 0 (derived) | 156 |
| L3 clean | 0 separate tables (= L2 all three; otherwise STOP before the append) | 0 | gate scan only | 0 |

| Operation class | Count |
|---|---|
| Model-fitting jobs | **24** (Mahalanobis 4, dense AE 20) |
| Validation-only operations | 24 new L1 validation scorings + q90; 28 reused L1 validation bundles verified; **208** L2 calibrations + fused q90 thresholds; gate clean scans and controls; 156 dead-input constant scorings (synthetic input, no data) |
| Protected inference operations | **24** L1-new scorings + **2** gate scans of raw common-test inputs (one per machine; label-free, no score, no metric) |
| Derived protected tables from sealed Phase-4 scores | **208** |
| Historical tables re-analysed, no new look | **184** (156 Phase 4 + 28 Phase 3.2) |
| **New protected looks** | **232** = 24 + 52 + 156 |
| Authorization units | **1** batch |
| Ledger | 196 → **428** |

**Why derived tables are looks (C6).** An L2 table is a new detector whose test performance has
never been observed. Producing it needs no inference, but scoring it against test labels is a new
exposure, so it is authorised, ledgered and sealed like any other. Re-analysing the 156 Phase-4
and 28 Phase-3.2 tables adds no look: no new detector meets the test labels. Their reuse is
recorded with the hashes of the sealed tables consumed.

**Is the matrix larger than the question needs?** The 156 pair looks are the only evidence for
H3 and also answer which sensor subsets carry the detection — the modality ablation planned in
`docs/research_definition.md` §13.1 — using genuinely trained single-modality detectors. Dropping
families would break the matched, cross-family reading inherited from Phase 4 and saves no
compute (L2/L3 need no inference). The real reductions are already taken:
- L1 classical is reused (−28 looks);
- max pooling is excluded (−208);
- L3-clean is not duplicated (−52);
- L1 pairs are excluded (−156 fits and looks);
- protected corruption sweeps are deferred (several hundred inference looks).

**Deviation from the research definition.** §13 planned test-time additive-noise and sensor-fault
sweeps and retrained pair models. Phase 5 v1 delivers complete-modality loss on the protected
test and partial degradations in development only. Protected sweeps and L1 pair models are
deferred to a possible separately designed later stage. This is stated here so the deviation is
not silent.

---

## 10. Protected execution, ledger and provenance

This inherits the Phase-4 machinery and conventions; there is no second provenance framework.

- **Human authorisation.** After the 5.5 freeze and an independent read-only audit, the human
  states an explicit authorisation token in chat. It is recorded in a pre-scoring identity
  record. `authorized_protected_read` is an accident guard: a non-empty authority string is
  **not** an authorisation.
- **Execution identity.** Clean tracked tree; the scoring HEAD is bound into the batch id; the
  proposed batch id is recorded at the freeze; authorised ids and cohort hashes must equal the
  proposed ones.
- **Development reads prevented.** The Phase-5.0 guard on test inputs, plus the 5.2 guard on
  sealed historical score tables. One new protected script becomes the only additional opener of
  `authorized_protected_read` and is added to the `AUTHORIZED_SCOPES` tripwire in the 5.5 freeze
  commit.
- **Order of score-all:**
  1. verify the token, clean tree, ledger = 196, every bundle/model/manifest hash, the Phase-4
     score-manifest payload and table hashes, and the Phase-3.2 artifact hashes;
  2. open the scope and run the gate scan of raw common-test inputs — **STOP before any append if
     anything is gated**;
  3. materialise L1-new label-free inputs from the existing hash-verified caches;
  4. append **exactly 232** rows atomically, each with `batch=<id>` in its reason, and write
     `batch_authorization.json` (`batch_id`, `ledger_start_count` 196, `ledger_end_count` 428,
     `experiment_count` 232, `code_git_sha`) so `verify_ledger_history` accepts it without edits;
  5. score 24 L1-new tables and derive 208 L2 tables into a sealed staging area — no label read,
     no metric;
  6. completion gate: 232 tables present, hashed, finite, label-free.
- **Analyse-once.** All contrasts, categories and diagnostics in one pass. It refuses if an
  analysis directory exists; a completed analysis is final.
- **Manifest hashes.** `pre_analysis_score_manifest_sha256`,
  `post_analysis_score_manifest_sha256` and `score_payload_sha256` (canonical JSON without
  mutable status fields), per `docs/phase5_entry_hygiene.md` §5.
- **Derived-table provenance.** Each L2 table records the ids and verified hashes of the Phase-4
  tables it consumed, its calibration parameters and threshold, and the rule version.
- **Failure policy (Phase 4, inherited).** A refusal before the append consumes no look. After
  the append the looks are consumed and the same identity is never rerun; recovery needs a new
  versioned batch identity and an incident record. No metric is computed before all 232 tables
  verify.
- **Closure.** The result commit registers the batch as a closed segment in
  `src/experiment/ledger_history.py`.

---

## 11. Claim boundary and limitations (C4)

- **Prior exposure.** Phase-4 aggregate protected results were observed before this design.
  Phase 5 is prospective in its frozen implementation, with controlled look accounting and no
  tuning on Phase-5 outputs, but it is **not a pristine confirmatory test** of fusion in general.
  The Phase-3.2 all-channel classical results were also observed before (descriptively, in the
  Phase-4 report §12); their paired contrasts are new analyses of old detectors.
- **Per machine and per regime only;** no universal fusion claim.
- **BrushlessMotor** is benchmark discrimination under
  `brushless_motor_acquisition_time_class_confound_v1`. More sensors can mean better *session*
  discrimination, so a fusion gain there is not evidence of fault sensitivity.
- **RoboticArm operating results** rest on 20 healthy target segments from 6 parents
  (LOW-PRECISION).
- **L3 is integrity gating** validated on synthetic failures. It is not quality estimation, says
  nothing about partial real-world degradation, and the loss condition models complete loss of
  one modality only.
- **L1 autoencoder capacity** differs from three single-modality autoencoders; this is the frozen
  recipe, reported, not tuned.
- **Calibration and thresholds share one validation cohort** (in-sample q90, as in Phase 4).
- **No shift-adaptation mechanism** is introduced.

---

## 12. Implementation plan (each sub-phase ends in a STOP)

| Stage | Deliverables | Acceptance | Data |
|---|---|---|---|
| **5.2** Data assembly and hygiene | guarded loaders and a path tripwire for sealed historical score tables; per-unit validation-score assembly (identical cohorts asserted); L1 classical 98-column assembly; L1 spectral concatenation with `window_id` equality and dims 2918/2936 asserted; concatenated-standardiser equality; Phase-3.1 thresholds verified against Phase 3.2 (U2); determinism tests | suite green; no protected read; protected-state snapshot unchanged | train/validation |
| **5.3** Matched naive fusion | 4 Mahalanobis + 20 AE L1 fits with validation q90 bundles and a reproduction check; 208 L2 calibrations and thresholds (IQR > 0 checked, U3); threshold-metric bootstrap implemented and unit-tested | bundles hash-manifested; reproduction exact within tolerance; validation FPR ≈ 0.10 per fused subset | train/validation |
| **5.4** Integrity gate and falsification | gate v1; clean, positive and negative control suites (section 5); dead-input constants and the ungated-vs-gated validation demonstration | every section-5 criterion met, otherwise STOP and report | train/validation, synthetic |
| **5.5** Pre-test freeze | frozen config v2 (FROZEN_FOR_PRETEST; batch not authorised); 232 experiment ids; proposed batch id; protected script tested on fixtures; failure-policy note; `AUTHORIZED_SCOPES` updated; **independent read-only audit** | audit passed | none protected |
| **5.6** Authorisation | human token, pre-scoring identity record | explicit approval | — |
| **5.7** Protected evaluation | score-all (gate scan, 232-row append, 24 inferences, 208 derivations), completion gate, analyse-once | gate GREEN; analysis final | one authorised batch |
| **5.8** Result report | categories per contrast, trade-offs, limitations; closed ledger segment registered | review | documentation |

---

## 13. Self-review

- **Could the method suppress true faults?** L2 cannot: it has no weights. L3 can only through
  G2 saturation plateaus covering ≥ 50 % of a segment; this is tested by negative controls and
  the clean-data margin (U1) and reported as a protected diagnostic. R2, which would suppress
  faults by construction, was rejected.
- **Could improvement be confounding?** On BrushlessMotor, yes; this is stated as a claim
  boundary (more sensors can mean better session discrimination). On RoboticArm the
  parent-aware split and cluster bootstrap carry over unchanged.
- **Are comparisons matched?** Yes — same unit, cohort, draws, threshold rule and orientation;
  L1 classical is verified field-by-field. The L1 autoencoder's capacity asymmetry is the one
  known mismatch and is disclosed.
- **Were choices influenced by historical protected results?** Checked item by item:
  - families, seeds and recipes are Phase 4's own;
  - the subsets are all non-trivial subsets;
  - pooling is chosen by its relation to L3 and an a-priori property of max;
  - margins are judgement-based (an earlier draft rationale citing Phase-4 seed SDs was removed);
  - the audio down-weighting idea from the Phase-4 report was rejected because it is
    outcome-driven;
  - the dense autoencoder's "fusion builds on it" role was declared in the Phase-4 config before
    any protected result, and gives it no extra arm here.
- **Does the operating interpretation overstate the RoboticArm slice?** The LOW-PRECISION label
  and the CI-based categories (wide intervals → INCONCLUSIVE) prevent it.
- **Is the count larger than needed?** Section 9 justifies each table class and lists the
  reductions already taken.
- **Does it address shift or merely combine scores?** It measures fusion under the benchmark's
  operating shift and under sensor loss. It adds no shift adaptation, and says so.
- **Can a simpler method answer it?** L2-mean *is* the simple method. L3 adds one binary gate and
  per-subset thresholds, the minimum needed to survive a dead sensor.
- **Industrial plausibility.** Per-sensor detectors, sensor-validity flags, exclusion of invalid
  channels and per-configuration alarm thresholds are standard condition-monitoring practice. The
  design tests exactly that pattern.
