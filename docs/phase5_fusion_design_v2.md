# Multi-Sensor Fusion Development — Scientific Design v2 (Phase 5.1, review-corrected)

<!-- Legacy roadmap name: Phase 5.1 -->

**Status:** PROPOSED — NOT AUTHORIZED FOR EXPERIMENTAL EXECUTION.
**Machine-readable design:** `configs/phase5/fusion_design_v2.json` (same status; it holds the
exact unit registry, counts and the review-correction record).

**Supersedes:** `docs/phase5_fusion_design_v1.md` and `configs/phase5/fusion_design_v1.json`,
reviewed at `803d5e2` (verdict YELLOW). Both v1 files are left byte-identical; the v2 config pins
their SHA-256. v1's own change-control rule forbids editing a reviewed design in place, so the
corrections are a new version. Everything the review accepted is carried over unchanged;
section 16 maps each correction C1–C8 to where it is resolved.

**Scope of 5.1:** design and planning only. No fusion code, no model fit, no protected read, no
score, no ledger change.

---

## 1. Starting point: what exists and what does not

All facts were read from non-protected artifacts: pre-test and validation-only bundles, configs,
provenance, ledger metadata, the dataset audit and source code. No protected score, metric, label
or common-evaluation content was read. Aggregate Phase-4 conclusions are cited only as context
and choose nothing (section 15).

### 1.1 Reusable without new work

| Asset | Location | What it gives Phase 5 |
|---|---|---|
| 156 frozen single-modality detectors | `reports/phase4_pretest/experiments/<id>/`, models in `data/interim/phase4/models/` | L0 comparators; the inputs to L2 |
| Source-validation segment scores for all 156 | `validation_scores.csv` in each bundle | L2 calibration and fused thresholds with **no new inference** |
| Common validation cohort | for each of the 52 (machine, regime, family, seed) units, mic/acc/gyro cover one identical set of normal source-validation segments (RoboticArm 186, BrushlessMotor 126), identical for S and T | segment-level score fusion is well defined |
| Score orientation | all 156 thresholds declare `higher_is_more_anomalous` | no sign flips |
| 156 sealed Phase-4 test score tables | `reports/phase4_protected/staging/scores/` | L2 test tables derive from these in the protected batch only (section 12) |
| 28 all-channel classical detectors | `reports/phase3_1/validation_only/`, `reports/phase3_2/experiments/` | L1 for statistical_reference, Isolation Forest and OCSVM (section 9.2) |
| Row-aligned spectral caches (train/validation) | `data/interim/phase4/spectral/<machine>/<regime>/` | L1 autoencoder input; one `window_id` per row across modalities |
| 98-feature segment artifacts | `data/interim/phase31_primary/` (1137–1555 training segments per unit) | L1 classical input |
| Analysis machinery | `phase4_analysis.py`, `bootstrap.py`, `phase32.make_common_bootstrap_draws` | R8 seed averaging and shared paired draws |
| Protection and provenance | Phase-4 conventions; Phase-5.0 `verify_ledger_history`, `authorized_protected_read`, tripwires | protected execution with no second framework |

### 1.2 Must be built or fitted

1. **First:** a read guard for sealed historical test score tables (section 12.2).
2. **New fits (24):** an all-channel Mahalanobis detector (4) and an all-channel dense
   autoencoder (20).
3. **L2 machinery:** calibration, four modality subsets, fused q90 thresholds (208 validation
   operations).
4. **L3 runner:** a development-only, end-to-end raw-input path with the integrity gate.
5. **Clean scan and end-to-end corruption suite** (section 6).
6. **Paired bootstrap for threshold metrics** (Phase 4 reported healthy-target FPR and recall as
   point estimates only).
7. **One protected script** (score-all / analyse-once) on the Phase-4 pattern, with the
   exposure-first ordering of section 12.

### 1.3 What the dataset cannot support

The audit found, across all 11,688 files and 27,272 channel instances: **0 NaN, 0 ±Inf, 0 constant
channels, 0 duplicate timestamps, 0 non-monotonic timestamps, 0 gaps**. That establishes the
absence of *gross integrity failures*. It does **not** show that every reading is physically
correct, calibrated or free of subtle corruption.

- There are no naturally occurring or labelled sensor faults.
- Training and validation contain normals only, so **no physical anomaly exists outside the
  protected cohort**.
- The audio environmental noise is a benign operating/environmental shift, not a sensor failure.

Therefore graded sensor-reliability estimation cannot be learned or validated here. Any
reliability mechanism can only be exercised under **synthetic** degradation, and synthetic
fault-like signals are negative controls, not physical fault evidence.

---

## 2. Scientific question, hypotheses, estimand

**Question.** Under leakage-controlled Protocol B (few-shot split v3, Regimes S and T, frozen
common test), and separately for RoboticArm and BrushlessMotor:

1. Does combining microphone, accelerometer and gyroscope evidence — by feature-level fusion
   (L1) or by source-validation-calibrated score averaging (L2) — change **ranking
   discrimination**, **healthy-target false-alarm rate** and **anomaly recall at a frozen
   source-validation threshold**, relative to each matched single-modality detector?
2. What detection capability does each **two-sensor subset** retain on clean data, and — under
   explicit development-verified conditions — what does that imply for a **sensor-integrity-aware
   fallback** (L3) after one sensor fails completely?

**Hypotheses** (two-sided; no prior for or against fusion):
- **H1** L2 (all three) differs from each matched single modality — mic, acc and gyro
  individually — on the ranking and/or operating axis.
- **H2** L1 differs from L2 at matched family and seed.
- **H3** Each L2 pair differs from intact L2 (all three): the clean-data cost of not using one
  modality.

**Estimand.** Δ(metric) = metric(fusion arm) − metric(comparator) on the common test cohort,
seed-averaged per replicate (R8), conditional on the observed evaluation cohort. The interval is
the 95 % percentile interval from the shared paired class-stratified cluster bootstrap (seed 1995,
10,000 replicates; parent recording is the unit for RoboticArm, recording = segment for
BrushlessMotor).

**Population.** RoboticArm 466 test segments (234 normal / 232 anomalous; healthy target slice 20
segments from 6 parents). BrushlessMotor 312 (156 / 156; healthy target 78). Calibration uses
normal source-validation only (RoboticArm 186, BrushlessMotor 126).

**Shift.** Phase 5 adds no shift-adaptation mechanism. It measures whether combining sensors
changes behaviour under the benchmark's operating/environmental shift. The only adaptation lever
remains Regime T's released target normals.

---

## 3. The baseline ladder

### 3.1 L0 — single modality

The 156 Phase-4 detectors. They are not retrained and not rescored. Every fusion arm is compared
with **mic, acc and gyro individually** at matched machine, regime, family and seed. The best
Phase-4 modality appears only as a labelled post-hoc historical benchmark and is never the sole
comparator.

### 3.2 L1 — feature-level (early) fusion

One detector per unit, trained on all three modalities with the single-modality recipe.

- **Statistical reference, Isolation Forest, OCSVM — reuse the 28 Phase-3.2 all-channel
  detectors.** No new fit and no new look; the evidence is in section 9.2.
- **Mahalanobis — new fit (4).** LedoitWolf on the 98 standardised features (n/p ≥ 11.6).
- **Dense autoencoder — new fit (20).**
  - Input: per-window concatenation [mic | acc | gyro] from one cache row (`window_id`).
  - Input dims: RoboticArm **2918**, BrushlessMotor **2936**.
  - Standardisation: train-only per-bin z-score (equal to the three Phase-4 standardisers
    concatenated; asserted in 5.2).
  - Architecture: the frozen template `d→256→64→16→64→256→d`.
  - Loss: plain MSE (naive; the microphone supplies about 27 % of bins).
  - Training: Adam 1e-3, batch 256, ≤ 200 epochs, patience 10 on source-validation window MSE,
    seeds 0–4.
  - Scores: window MSE; segment = median of windows; own validation q90.
  - The capacity asymmetry (one 16-d bottleneck versus three behind L2) is disclosed, not tuned.

No family is excluded from L1, so the question of selecting families by performance does not
arise.

### 3.3 L2 — score-level (late) fusion

A fixed rule over the frozen Phase-4 per-modality segment scores; nothing is trained.

1. **Pairing:** seed *s* of mic, acc and gyro are fused together (deterministic families use seed 0).
2. **Calibration:** robust z = (score − median_v)/IQR_v, using that modality's frozen
   source-validation scores only.
3. **Pooling:** mean over the subset.
4. **Subsets:** {mic, acc, gyro}, {acc, gyro}, {mic, gyro}, {mic, acc}.
5. **Threshold:** q90 (`higher`) of the fused source-validation scores of that subset, never
   inherited from a single modality. This is in-sample, as Phase 4 was.
6. **Defects:** a missing or non-finite score, or IQR_v = 0, is a technical defect → STOP; never
   imputed.

Robust z is preferred to an ECDF because an ECDF fitted on 126–186 segments ties every test score
above the validation maximum. Weights are equal because learning weights needs anomalies.

**Max pooling is excluded a priori:** one failed or fooled sensor decides by construction, it
collapses under modality loss, and L2-mean is L3's exact ablation.

### 3.4 L3 — sensor-integrity-aware fallback

**What it is.** An engineering safety/integrity rule. It excludes a modality whose raw signal
fails a frozen integrity gate, and decides with the validation threshold of the surviving subset.
It is **not** learned sensor-reliability estimation, not novelty weighting, and not a classifier
of fault versus sensor error.

**When every sensor passes the gate, L3 is identical to L2 (all three)** in score and decision.

**Decision policy**

| Integrity outcome in a segment | L3 output |
|---|---|
| all three valid | L2 (all three) score and threshold |
| one modality invalid by a value failure (A or B below) | L2 of the surviving pair, with that pair's validation q90 |
| two modalities invalid by value failures | surviving single modality's frozen Phase-4 detector and threshold (decision-identical to its robust-z threshold, a monotone transform) |
| all three invalid | `ABSTAIN_SENSOR_INVALID` — no anomaly decision; the segment is retained and counted |
| any timestamp failure or absent modality (C or D) | `ABSTAIN_ALIGNMENT_FAILURE` in v1; the segment is retained and counted |

Every segment carries its per-modality gate verdicts. Any invalid modality raises an
**INTEGRITY_FLAG** maintenance event independent of the anomaly decision, so a gated segment is
never silently passed as healthy.

**Transduction boundary.** The gate reads only the segment being scored, and every parameter is
fixed on training/validation before evaluation. That is frozen per-sample inference. It would
become transductive if any protected-cohort quantity — a test median, a test gating rate, a
threshold refitted on test scores, a batch normalisation — influenced a weight, score or
threshold.

---

## 4. End-to-end sensor-loss feasibility (from the repository code)

The current pipeline (`src/preprocessing/signals.py`, `pipeline.py`) does the following:

- `read_sensor_parquet` loads each modality and **raises on non-increasing timestamps**. It does
  not check values.
- `apply_startup_transformations` is **value-independent**: it drops a fixed number of initial
  accelerometer samples and crops the gyroscope warm-up by time.
- `common_coverage` takes the window start and end from **timestamps only**, over **all three
  modalities**, and raises if one is missing.
- Normalisation, feature extraction and spectral transforms are computed **per modality**.
- The existing cache builders write all three modalities per window and refuse non-finite
  vectors, so a fallback needs a per-modality runner (section 6) rather than the caches.

| Failure | Gate position | Detection | Survivors processed? | Survivor scores | Threshold | v1 status |
|---|---|---|---|---|---|---|
| **A** non-finite values | after loading, startup transforms and shared alignment; before per-modality processing | G1 | **Yes** — timestamps are valid, so the window grid is exactly the clean grid | frozen Phase-4 per-modality transforms and detectors on survivors only; must equal their clean scores | survivor-subset q90 | **SUPPORTED (fallback)** |
| **B** flatline / stuck values | as A | G2 | **Yes**, as A | as A | survivor-subset q90 | **SUPPORTED**; partial flatlines < 0.50 are not gated and degrade the input silently |
| **C** invalid timestamps | before shared alignment | G3 (the reader's exception becomes a gate verdict, not an unhandled error) | **No** — a timestamp failure in one modality corrupts or blocks the shared window grid of all modalities; recovery needs alignment over survivors only, which changes the grid relative to frozen three-sensor processing, so survivor scores would no longer equal the protected pair scores | not produced | none | **ABSTAIN_ALIGNMENT_FAILURE** |
| **D** absent modality | at loading | missing or unreadable stream | **No** — `common_coverage` requires all three modalities | not produced | none | **ABSTAIN_ALIGNMENT_FAILURE** |
| two value failures | — | — | one survivor | its frozen Phase-4 detector | its Phase-4 q90 | supported |
| three value failures | — | — | none | — | — | ABSTAIN_SENSOR_INVALID |

Recovery from C and D needs subset-aware alignment, an architectural change to the frozen
preprocessing. It is **outside first-stage scope**, so the claim is narrowed instead.

### 4.1 What the pair tables measure, and the bridging conditions

The protected pair tables measure **the detection capability, on clean protected data, of each
two-sensor subset combined by the frozen L2 rule with its own validation threshold.** A comparison
between clean three-sensor and clean two-sensor scores does **not**, by itself, demonstrate
resilience to a failed physical sensor.

The pair tables equal L3's output after a failure only if:

- **B1** the gate classifies the failed modality as invalid (development suite, scenarios A/B);
- **B2** the surviving modalities' segment scores under the failure equal their clean scores
  (≤ 1e-9 absolute, every frozen Phase-4 detector);
- **B3** the runner selects the surviving subset's frozen validation threshold;
- **B4** the failure is of type A or B (C and D abstain and are not represented);
- **B5** the surviving physical sensors are unaffected by whatever failed the other one. This is
  assumed; it cannot be tested on this dataset.

**Supported claim.** If B1–B4 hold in development, the protected pair results estimate L3's
detection capability after a complete value failure (non-finite or flatline) of one modality,
conditional on B5. They do not show resilience to:
- timestamp failures or absent modalities;
- partial or subtle degradation;
- real physical sensor failures.

**L3 is not executed on protected inputs, and no gate runs on protected data.**

---

## 5. Integrity gate v2 — exact semantics and false-suppression analysis

**Order of operations per segment**
1. Load each modality; a missing or unreadable stream is a D failure.
2. G3 per modality on timestamps: finite, strictly increasing, and own-stream coverage ≥ 0.90 of
   the machine's nominal segment duration (RoboticArm 7.5 s, BrushlessMotor 5.2 s). Any failure
   is a C failure.
3. Apply the frozen, value-independent startup transforms.
4. Shared alignment over all three modalities (the unchanged frozen rule).
5. G1 and G2 per modality on values inside the common coverage [start, end).
6. Process and score the surviving modalities with the frozen Phase-4 transforms and detectors.
7. Robust-z calibration, mean over survivors, survivor-subset threshold, then decision and
   integrity flag.

**G1 — non-finite.** Any NaN or ±Inf value in any channel inside the coverage makes the modality
invalid.

**G2 — flatline**, evaluated per channel on raw stored values (pre-resampling, pre-normalisation):
- **Run:** a maximal sequence of consecutive samples whose stored float64 values are exactly
  equal.
- **Duration:** `t_last − t_first`, using that stream's own timestamps. Sample count × nominal
  period is not used, because effective IMU rates vary by session.
- **Qualifying run:** duration ≥ 0.020 s.
- **Channel flatline fraction:** samples in qualifying runs ÷ samples of that channel inside the
  coverage.
- **Modality rule:** invalid if **any** channel's fraction is ≥ 0.50. The microphone has one
  channel; accelerometer and gyroscope have three axes, and the frozen detectors consume all axes
  jointly, so one dead axis corrupts the modality input.
- **Quantisation and quiet intervals:** there is **no exemption**. The rule assumes a live sensor
  never holds an exactly equal stored value for 20 ms. The assumption is measured on every clean
  training and source-validation segment-channel rather than special-cased. (Quantisation
  context: at the documented ±16 g accelerometer range one 16-bit LSB is about 0.49 mg, and
  stationary robotic-arm phases are where long equal-value runs are most likely. That is exactly
  why the scan includes all clean segments of both machines.)

**G3 — timing.** Non-finite, non-increasing or duplicate timestamps, or own-stream coverage < 0.90,
is an alignment failure → ABSTAIN in v1.

**Fixed a priori:** 0.020 s, 0.50 and 0.90 come from the physical definition of a dead or stuck
sensor; they are not fitted.

**Clean-data acceptance (5.4)**
- Zero G1/G2/G3 triggers on every clean training and source-validation segment-channel, including
  the released target-normal training segments and target-domain audio. Any trigger → STOP and
  report; no retuning.
- The full distribution of flatline fraction and longest run per machine × sensor × channel is
  reported.
- A sensor whose maximum clean flatline fraction exceeds 0.25 is labelled **LOW-MARGIN** in every
  later claim.

**False-suppression analysis.** A genuine fault can drive a channel into saturation. A saturation
plateau is a run of equal values, so if it lasts ≥ 20 ms per run and covers ≥ 50 % of the
segment, G2 gates the modality and its anomaly evidence leaves the decision. One saturated axis
gates the whole modality.

The gate cannot distinguish a failed sensor from such a fault. The policy for this ambiguity is:
- the modality is excluded from the anomaly decision;
- the **INTEGRITY_FLAG** is raised, so the segment is surfaced as a maintenance event and never
  passed silently as healthy;
- the survivors still decide on the fault.

This is defended as an engineering integrity rule with known limits, not as fault preservation.
The development suite **characterises** the suppression envelope (gain × 1.5/2/3/5 then clipping
at full scale) but does not, and cannot, prove that genuine faults are never suppressed. There is
no physical fault outside the protected cohort to test with.

**Not detected by the gate:** partial flatlines < 0.50, gain error, bias drift, added noise,
timing jitter within tolerance, and miscalibration.

---

## 6. Development falsification (5.4; permitted train/validation and synthetic inputs only)

### 6.1 Clean scan

G1–G3 on every training and source-validation segment-channel of both machines. Acceptance is as
in section 5.

### 6.2 End-to-end corruption suite

This exercises, on corrupted copies of permitted raw recordings written to a development
directory: raw loading → integrity detection → shared alignment → surviving-modality processing
with the frozen Phase-4 transforms → scoring with **all 156** frozen Phase-4 detectors → robust-z
calibration → subset threshold → decision, abstention and flag.

- **Segments:** 12 source-validation segments per machine. Sort segment ids and take indices
  round(k·(n−1)/11), k = 0…11.
- **Single-modality failure scenarios** (each applied to mic, acc and gyro in turn):

| Id | Corruption |
|---|---|
| A1 | NaN, all channels, 100 % |
| A2 | NaN, one channel, 100 % |
| A3 | NaN, all channels, 10 % |
| B1 | constant (dead), 100 % |
| B2 | stuck at last value from 25 % of coverage (75 % flat) |
| B3 | zero dropout, 60 % |
| B4 | flatline, 40 % (below the trigger; expected valid) |
| B5 | plateau at the channel extreme, 60 % |
| C1 | two timestamps swapped |
| C2 | duplicated timestamp |
| C3 | NaN timestamp |
| C4 | stream truncated to 50 % of duration |
| D1 | modality file absent |

- **Multi-failure:** M1 two modalities dead (3 combinations); M2 all three dead (1).
- **Negative controls** (per modality, valid fault-like changes): N1 gain ×0.5, N2 ×2, N3 ×3,
  N4 offset +3 training SD, N5 Gaussian noise 20 dB, N6 10 dB, N7 0 dB, N8 impulse train, N9 added
  tone. These are controls for gate behaviour, **not physical fault evidence**.
- **Saturation characterisation** (per modality, not pass/fail): S1–S4, gain ×1.5/2/3/5 then clip
  at full scale. Accelerometer: the documented ±15.988832 g rail. Gyroscope and microphone: the
  clean-training maximum |value| of that machine-channel, an *emulated* full scale because their
  physical range is not documented.
- **Clean baseline:** 1.
- **Runs per segment:** 13 × 3 + 4 + 9 × 3 + 4 × 3 + 1 = **83**; total **1,992** segment runs.

**Pass criteria (all required; any failure → STOP, no retuning):**

| Id | Criterion |
|---|---|
| E0 | the clean runner reproduces every Phase-4 bundle's recorded validation scores within 1e-9 (Phase-4 precedent 2.8e-14) |
| E1 | gate verdicts equal expectations: A1–A3, B1–B3, B5 → invalid; B4 → valid; C1–C4, D1 → alignment failure; M1/M2 as defined; N1–N9 → all valid |
| E2 | for every A/B scenario, surviving-modality scores equal their clean scores (≤ 1e-9) for every frozen detector — bridging condition B2 |
| E3 | the selected threshold is the surviving subset's frozen q90; two-failure decisions equal the surviving modality's Phase-4 decision |
| E4 | C, D and M2 produce the defined ABSTAIN status, the segment is retained and counted, and no exception escapes |
| E5 | clean and N1–N9 runs give L3 score and decision identical to L2 (all three) |
| E6 | any unexpected exception is a software defect and fails the suite |

**Descriptive:** B4 partial-degradation behaviour, the S1–S4 suppression envelope, and the
source-validation FPR of ungated L2 (dead-input score inserted) versus L3 (the pair) under loss of
each modality.

---

## 7. Outcomes and multi-objective interpretation

**Ranking:** primary AUROC; secondary AUPRC (with prevalence), pAUROC@0.10, target- and
source-slice AUROC.
**Operating** at the frozen q90: primary healthy-target FPR (lower is better) and recall (higher
is better); secondary source FPR, target-slice recall, precision and F1.

**Practical margins**, fixed by engineering judgement and not derived from any Phase-4 effect
size or variance:
- **AUROC / AUPRC 0.02** — the smallest ranking change treated as practically relevant.
- **Healthy-target FPR 0.10** — two of the 20 RoboticArm healthy target segments, the smallest
  change that is not a single segment.
- **Recall 0.05** — about 12 RoboticArm or 8 BrushlessMotor anomalies.

**Per-metric category** (EQUIVALENT is tested first):

| Category | Rule |
|---|---|
| **EQUIVALENT** | the **entire** 95 % CI lies inside (−margin, +margin) — two one-sided tests at 2.5 % each. Annotated DETECTABLE_BUT_NEGLIGIBLE if the CI also excludes 0 |
| **IMPROVEMENT** | the 95 % CI excludes 0 favourably and is not inside the margin; tagged **MEANINGFUL** if \|point\| ≥ margin, else **BELOW_MARGIN** |
| **DEGRADATION** | the 95 % CI excludes 0 unfavourably and is not inside the margin; tagged as above |
| **INDETERMINATE** | otherwise: the CI contains 0 and crosses a margin, or there are fewer than 9,000 valid replicates. **A small non-significant point estimate is INDETERMINATE, never EQUIVALENT** |

**Axes.** R is the Δ-AUROC category; AUPRC and pAUROC disagreements are flagged, never
substituted. O combines healthy-target FPR and recall:
- **O+** one IMPROVEMENT, the other IMPROVEMENT or EQUIVALENT;
- **O−** one DEGRADATION, the other DEGRADATION or EQUIVALENT;
- **O_tradeoff** one IMPROVEMENT and one DEGRADATION;
- **O0** both EQUIVALENT;
- **O?** otherwise.

**Joint categories**

| Category | Rule |
|---|---|
| BOTH_IMPROVED | R IMPROVEMENT and O+ |
| RANKING_GAIN_OPERATION_PRESERVED | R IMPROVEMENT and O0 |
| OPERATING_GAIN_RANKING_PRESERVED | R EQUIVALENT and O+ |
| TRADE_OFF | (R IMPROVEMENT with O− or O_tradeoff) or (R DEGRADATION with O+) or (R EQUIVALENT with O_tradeoff) |
| EQUIVALENT | R EQUIVALENT and O0 |
| UNDERPERFORMANCE | (R DEGRADATION with O−, O0 or O_tradeoff) or (R EQUIVALENT with O−) |
| INDETERMINATE | R INDETERMINATE or O?, reported with whichever axis is decisive |

Size tags (MEANINGFUL / BELOW_MARGIN) are carried into the joint category. A TRADE_OFF is a finding
reported with both components, neither success nor failure. At a fixed threshold, "fewer false
alarms and fewer detections" can reflect a score-scale shift; the report will say so.

**RoboticArm healthy-target slice**
- **Discreteness:** hFPR moves in steps of 1/20 = 0.05 per seed (0.01 after five-seed averaging),
  so the 0.10 margin is two segments. Per-seed counts k/20 are printed beside every interval.
- **Parent uncertainty:** the shared draws stratify by class only, so the number of the 6
  healthy-target parents drawn varies, and some replicates may contain none. Those are invalid for
  hFPR and counted.
- **Rule:** RoboticArm operating results are **LOW-PRECISION**, and no RoboticArm operating claim
  is made from a point estimate. EQUIVALENT would need the full CI inside ±0.10, which is not
  expected with 6 parents, so **INDETERMINATE is the anticipated honest outcome**.

**Across families.** This is a **descriptive robustness label only**: CONSISTENT_ACROSS_FAMILIES
when all five families are non-contradicting on an axis, otherwise FAMILY_DEPENDENT. A
family-dependent result is reported with each family's category and is never erased or averaged
away.

---

## 8. Contrasts, statistics, multiplicity

**Primary** (per machine × regime × family; AUROC, healthy-target FPR, recall):
- **P1:** L2 (all three) − mic, − acc, − gyro.
- **P2:** L1 − L2 (all three).
- **P3:** L2 (pair without m) − L2 (all three).

**Secondary:**
- S1: L1 − each single modality.
- S2: each pair − its two members.
- S3: versus the post-hoc best modality (labelled post-hoc).
- S4: source- and target-slice versions.
- S5: AUPRC and pAUROC versions.
- S6: T − S.

**Statistics.** Ranking intervals reuse the Phase-4 machinery and the shared draws. Threshold-metric
intervals are new in 5.3, with the same draws and the same R8 averaging. Valid-replicate counts
are reported everywhere.

**Multiplicity — no adjustment.** Percentile tails from 10,000 replicates cannot support
Bonferroni-level quantiles. Instead:
- **Exact primary counts:** P1 60 + P2 20 + P3 60 = **140 primary contrasts × 3 metrics = 420
  primary intervals**.
- **Yardstick:** for each contrast family and metric, the report states how many intervals exclude
  0 next to the ≈ 5 % expected by chance with no true difference. Intervals are correlated, so this
  is a reading aid, not a test.
- **Language:** every interval is labelled nominal. An isolated nominally supported contrast is
  never presented as a discovery. Statements beyond one contrast rest on the descriptive cross-family
  and cross-regime pattern.

---

## 9. Leakage, matching and historical L1 evidence

### 9.1 Leakage and matching rules (unchanged from v1)

- Protocol B unchanged; calibration and thresholds use normal source validation only.
- No label is read before analyse-once, and no test statistic enters any weight, calibration,
  threshold or design choice.
- Contrasts are admissible only at matched machine, regime, family and seed, with the same cohort
  and draws, the same q90 rule and orientation, and identical segment ids.
- Development cannot read sealed test score tables (section 12.2).

### 9.2 Historical L1 matching evidence (C6)

**Method.** For each of the 28 Phase-3.2 all-channel units and each of its three Phase-4
single-modality counterparts, I compared frozen configs, Phase-3.1 validation-only bundles,
Phase-4 pre-test bundles, Phase-3.2 provenance and ledger metadata. **No historical score, metric
or label was opened.**

**Result: 28/28 units pass all 17 checks, 0 failures.**

| Requirement | Evidence |
|---|---|
| machine, regime, family, seed | unit key; Phase-3.1 bundle id = Phase-3.2 bundle id (28/28) |
| segment/parent split | equal split version and split-manifest hash (v1 check); equal training and validation **cohort** and **contract** hashes (28/28) |
| source/target allocation | equal adaptation policy version (v1); equal training cohort hashes and segment/window counts per regime (28/28) |
| preprocessing and feature definitions | equal preprocessing config id and hash (v1); identical feature representation (median_by_segment of the same 100 ms features) (28/28) |
| training and normalisation rules | equal normalisation strategy, epsilon and all hyperparameters; same implementation version `phase3.1-classical-baselines-v1`; **empty `git diff 99f6055..85df48d`** for classical models, feature extraction, normalisation and the Phase-3.1 pipeline |
| input dimensionality | 98 = 14 + 42 + 42 (28/28); this is the intended fusion difference |
| threshold semantics | equal policy, quantile 0.9/`higher`, source cohort hash and source segment ids; **the Phase-3.1 threshold equals the threshold Phase 3.2 applied** (28/28; resolves v1's U2) |
| evaluation cohort and segment identity | identical ledger test-cohort hash on both machines; same common-test feature artifacts |
| score orientation | `higher_is_more_anomalous` in thresholds and configs (28/28) |
| statistical procedure | equal bootstrap configuration (28/28); `make_common_bootstrap_draws` is a deterministic function of seed, segment ids, parents and classes, imported unchanged by Phase-4 analysis |

**Label: MATCHED_BY_RECORDED_IDENTITY.** Equivalence rests on recorded configuration, cohort,
contract and threshold identities and unchanged code. It does not rest on re-deriving or comparing
fitted model objects. No material incompatibility was found, so the 28 looks are not duplicated.

---

## 10. Exact experiment matrix

**Unit** = machine × regime × family × seed = 2 × 2 × (1 + 5 + 1 + 1 + 5) = **52**.

| Arm | Tables | New fit | New protected inference | New look |
|---|---|---|---|---|
| L0 single modality (Phase 4) | 156 | 0 | 0 | 0 (re-analysed) |
| L1 classical reused (Phase 3.2) | 28 | 0 | 0 | 0 (re-analysed) |
| L1 Mahalanobis new | 4 | 4 | 4 | 4 |
| L1 dense AE new | 20 | 20 | 20 | 20 |
| L2 all three | 52 | 0 | 0 (derived from sealed tables) | 52 |
| L2 pairs | 156 | 0 | 0 (derived) | 156 |
| L3 | 0 (identical to L2 when sensors pass; not executed on protected inputs) | 0 | 0 | 0 |

**Accounting definitions**

| Term | Count and meaning |
|---|---|
| New model inference | **24** — a frozen L1-new model applied to protected inputs |
| Derived fusion scoring | **208** — a frozen L2 rule applied to sealed Phase-4 score tables; no raw-sensor inference |
| Protected-data reads | inputs: common-evaluation spectral caches and common-test feature artifacts (24 L1-new units only); sealed outputs: 156 Phase-4 tables (to derive L2), plus the 28 Phase-3.2 tables in analyse-once only; labels: analyse-once only; **raw common-test recordings: not read** |
| Re-analysis of existing protected outputs | **184** tables (156 + 28); no new detector meets the labels → no new look |
| New protected evaluation units | **232** = one ledger row each = one new detector evaluated against protected outcomes |
| Authorisation | **1** human authorisation for **1** batch of 232 units |
| Completed result | only when all 232 tables are persisted and hash-verified (completion gate GREEN) **and** analyse-once has finished. A ledger row records authorised exposure, not completion |
| Model-fitting jobs | **24** |
| Validation-only operations | 24 L1-new validations; 28 reused validations verified; 208 L2 calibrations/thresholds; clean scan; **1,992** end-to-end suite segment runs; 156 dead-input scorings (synthetic) |
| Protected inference operations | **24** (the v1 protected gate scans are removed) |
| Ledger | 196 → **428** |

**Change from v1:** none to the look count (232). The two protected gate scans (v1) are removed,
so protected inference operations fall from 26 to 24. Development operations gain the 1,992-run
end-to-end suite.

**Is every entry justified?**
- **The 24 L1-new units** are the only early-fusion detectors for two families.
- **The 52 L2 all-three units** are the primary late-fusion arm.
- **The 156 pair units** carry H3 (clean subset capability, which is also the §13.1
  modality-ablation question of the research definition). Under B1–B5 they are the only protected
  evidence for the fallback. Dropping families would break the matched cross-family reading and
  save no compute.
- **Reductions already taken:** L1 classical reuse (−28), no max pooling (−208), no separate L3
  tables (−52), no L1 pairs (−156), no protected corruption sweeps.
- **Deviation from research definition §13:** the protected noise and sensor-fault sweeps and the
  retrained pair models are deferred; partial degradations are studied only in development.

---

## 11. Claim boundary and limitations

- **Prior exposure.** Phase-4 aggregate protected results were observed before this design.
  Phase 5 is prospective in its frozen implementation, but it is **not a pristine confirmatory
  test**. Phase-3.2 L1 results were also observed before, descriptively; their paired contrasts
  are new analyses of old detectors.
- **Scope.** Results are per machine and per regime; there is no universal fusion claim.
- **BrushlessMotor confound.** Benchmark discrimination only
  (`brushless_motor_acquisition_time_class_confound_v1`); more sensors can mean better session
  discrimination.
- **RoboticArm operating results** are LOW-PRECISION (20 healthy target segments, 6 parents).
- **L3 is a sensor-integrity-aware fallback for gross integrity failures,** identical to L2
  whenever sensors pass. It is not learned reliability estimation, and the clean audit does not
  certify the physical correctness of readings.
- **The pair tables measure clean subset capability.** Only under B1–B5 do they estimate
  post-failure capability, and only for complete value failures.
- **Not covered:**
  - timestamp failures and absent modalities (abstain in v1);
  - partial flatlines, gain error, drift, noise, jitter and miscalibration;
  - real physical sensor failures.
- **Synthetic controls** do not prove that genuine faults are never suppressed; saturating faults
  can be gated (flagged, not hidden).
- **Recipe and calibration limits.** The L1 AE capacity asymmetry is disclosed, and calibration
  and thresholds share one validation cohort (in-sample, as in Phase 4).
- **No shift adaptation** is introduced.

---

## 12. Protected execution, exposure accounting and score-table access

### 12.1 Corrected order

Phase-4 machinery and conventions are inherited; there is no second provenance framework. **No
gate runs on protected data and no raw test recording is read.**

1. **Pre-exposure verification on non-protected artifacts only:**
   - the human authorisation token;
   - a clean tracked tree;
   - ledger = 196 with `verify_ledger_history` GREEN;
   - every Phase-5 bundle, model and manifest hash, and the frozen design hash.

   No protected file is opened. A refusal here consumes no look.
2. **Record the exposure, before any protected read:**
   - append exactly 232 ledger rows atomically, with `batch=<id>` in every reason;
   - write `batch_authorization.json` (`batch_id`, start 196, end 428, `experiment_count` 232,
     `code_git_sha`);
   - write the attempt-log entry `EXPOSURE_RECORDED`.
3. **Open the authorised scope and verify protected integrity:** common-evaluation cache and
   common-test artifact hashes; Phase-4 score-manifest payload and table hashes. A mismatch is a
   TECHNICAL failure after exposure, fully attributable.
4. **Score and derive:** score the 24 L1-new tables from label-free inputs and derive the 208 L2
   tables from the verified sealed tables. Persist each table to sealed staging with its hash, and
   log each unit's PERSISTED state.
5. **Completion gate:** 232 tables present, hashed, finite, label-free, with ids equal to the
   authorised set.
6. **Analyse-once:** labels and the 28 Phase-3.2 comparator tables are read inside the scope; every
   contrast, category and diagnostic is computed once, and the result is final. It refuses if an
   analysis directory exists.

**Ordering change from Phase 4.** Phase 4 materialised test inputs before its ledger append.
Phase 5 records the exposure first, so every protected read is attributable to a recorded
authorisation.

**Distinctions kept separate**

| Record | Meaning |
|---|---|
| Batch authorisation | human token + pre-scoring identity record (no protected read) |
| Protected-data exposure | ledger rows + `batch_authorization.json`, written before the first protected read |
| Persisted score output | sealed staging tables + manifest with `pre_analysis_score_manifest_sha256` |
| Derived analysis | analyse-once outputs + analysis manifest, `post_analysis_score_manifest_sha256` and `score_payload_sha256` |
| Completed accounting | completion gate GREEN + finished analysis + result commit registering the closed ledger segment |

**Unit states**
- PLANNED → AUTHORISED → EXPOSED (ledger row) → PERSISTED → COMPLETED;
- or EXPOSED → FAILED_TECHNICAL.

An append-only attempt log records every step and state transition with UTC time and code SHA,
including failures.

**Error semantics.** With no gate on protected data, **every exception in the protected batch is a
TECHNICAL (software/infrastructure) failure**. There is no protected outcome called sensor
degradation in this design, so a gate activation can never become an occasion to redesign, change
eligibility or change thresholds after seeing protected inputs.

**Recovery and duplicate prevention**
- **Before exposure:** a refusal at step 1 consumes no look and needs no recovery record.
- **After exposure:**
  - looks are consumed, and analyse-once refuses unless every planned unit is PERSISTED;
  - an incident record names the failed units and the cause;
  - PERSISTED units with verified hashes are retained and never re-scored;
  - only FAILED_TECHNICAL units are re-attempted, under a new human authorisation, a new
    versioned batch id and new experiment ids (`<id>-r<k>`), as a new authorised ledger segment,
    with a mapping file to the planned units.
- **Duplicates:** `verify_ledger_history` refuses a repeated experiment id, and analyse-once refuses
  if any planned unit has zero or more than one PERSISTED table. The same id is never scored twice.

### 12.2 Sealed score-table access (first prerequisite of 5.2)

- **Gap.** Sealed historical test score tables (`reports/phase4_protected/staging/scores/*.csv`,
  `reports/phase3_2/experiments/*/scores.parquet`) are readable by ordinary code today; the
  Phase-5.0 guard covers test inputs only.
- **First 5.2 deliverable:**
  - guarded loaders for both table families that require an `authorized_protected_read` scope and
    refuse **before** opening the file;
  - a repository tripwire that fails if any `src/` or `scripts/` file other than the sanctioned
    loader module references those paths or reads them directly (including `read_csv`).
- **Development rule.** L2 is developed and tested on source-validation scores and **synthetic
  fixture tables only**. No development path may open a sealed test score table.
- **Authorised path.** The tables are opened only by the Phase-5 protected script inside its scope,
  after the exposure is recorded (step 3 onward). The scope is an accident guard; the human token
  and recorded exposure are the authorisation.
- **Acceptance.** Refusal outside a scope with the real files present, no mutation, and synthetic
  fixtures loading normally.

---

## 13. Implementation plan (each sub-phase ends in a STOP)

| Stage | Deliverables | Acceptance | Data |
|---|---|---|---|
| **5.2** | **sealed-score guard (first)**; validation-score assembly with identical cohorts asserted; L1 classical 98-column and spectral concatenation assembly (`window_id` equality, dims 2918/2936, standardiser equality); determinism tests | suite green; no protected read; protected-state snapshot unchanged | train/validation |
| **5.3** | 4 + 20 L1 fits with validation q90 bundles and a reproduction check; 208 L2 calibrations and thresholds (IQR > 0); threshold-metric bootstrap, unit-tested on synthetic data | bundles hash-manifested; reproduction within tolerance; validation FPR ≈ 0.10 per fused subset | train/validation |
| **5.4** | L3 runner; clean scan; end-to-end suite (1,992 runs); saturation characterisation; ungated-vs-gated validation demonstration | clean-scan acceptance and E0–E6, otherwise STOP | train/validation, synthetic |
| **5.5** | frozen config (FROZEN_FOR_PRETEST; batch not authorised); 232 ids; proposed batch id; protected script tested on fixtures; failure/recovery note; `AUTHORIZED_SCOPES` updated; **independent read-only audit** | audit passed | none protected |
| **5.6** | human token; pre-scoring identity record | explicit approval | — |
| **5.7** | score-all in section-12 order; completion gate; analyse-once | gate GREEN; analysis final | one authorised batch |
| **5.8** | result report; closed ledger segment registered | review | documentation |

---

## 14. Unresolved until measured

- **U1:** clean flatline-fraction distribution and margin per machine × sensor × channel (the audit
  rules out only fully constant channels) — 5.4.
- **U2:** **RESOLVED** — the Phase-3.1 thresholds equal the thresholds Phase 3.2 applied (28/28).
- **U3:** IQR_v > 0 for every modality, family and seed — 5.3.
- **U4:** L1 dense-AE training time and memory (about 1.35 GB of float32 training windows) — 5.3.
- **U5:** valid-replicate counts for RoboticArm hFPR intervals. These cannot be measured beforehand
  because the class-stratified draws depend on test labels; analyse-once reports them, and fewer
  than 9,000 valid replicates gives INDETERMINATE.
- **U6:** the runner reproduces every Phase-4 validation score within 1e-9 (E0) — 5.4.

---

## 15. Self-review (updated)

- **Suppression of true faults.** L2 has no weights. L3 can suppress only through G2 saturation
  plateaus (≥ 50 %, ≥ 20 ms runs); such segments are flagged, not hidden, the envelope is
  characterised, and no claim of universal preservation is made.
- **Confounding.** On BrushlessMotor, possibly; this is stated as a claim boundary.
- **Matched comparisons.** L1 classical is verified on 17 recorded identities. The AE capacity
  asymmetry is disclosed.
- **Influence of historical results.** Families, seeds and recipes are Phase 4's own; the subsets
  are exhaustive; max pooling is excluded a priori; the margins are judgement-based; audio
  down-weighting was rejected as outcome-driven.
- **Overstating the RoboticArm slice.** Prevented by LOW-PRECISION, the full-CI equivalence rule,
  per-seed counts and the anticipated INDETERMINATE.
- **Count larger than needed.** Every class of table is justified in section 10, and the v1 gate
  scans are removed.
- **Shift versus combination.** Phase 5 measures fusion under the benchmark's shift and clean
  subset capability. It adds no adaptation, and says so.
- **A simpler method.** L2-mean *is* the simple method. L3 adds one binary gate and per-subset
  thresholds, scoped to failures the pipeline can actually survive.
- **Industrial plausibility.** Per-sensor detectors, validity flags, exclusion with maintenance
  alerts and per-configuration thresholds are standard practice. The C/D gap (subset-aware
  alignment) is named as the next engineering step rather than assumed away.

---

## 16. Review corrections record (v1 → v2)

| Correction | Resolution | Where |
|---|---|---|
| **C1** L3 meaning | renamed sensor-integrity-aware fallback; L3 = L2 when sensors pass; pair tables = clean subset capability; sensor-loss reading only under B1–B5 | §3.4, §4.1, §11 |
| **C2** end-to-end feasibility | per-failure-type analysis from the code: A/B recoverable on the unchanged grid; C/D abstain (subset-aware alignment out of scope); development end-to-end suite with E0–E6 | §4, §6 |
| **C3** test-dependent gate | protected gate scan removed; exposure recorded before any protected read; no gate on protected data, so every protected error is TECHNICAL; records kept distinct | §10, §12.1 |
| **C4** false suppression | exact G2 semantics; no quantisation exemption, measured instead; any-channel rule; LOW-MARGIN flag; integrity flag on ambiguity; saturation characterisation; no physical-fault claim | §5, §6 |
| **C5** statistics | full-CI equivalence; IMPROVEMENT/DEGRADATION with size tags; INDETERMINATE; RoboticArm discreteness and parent rules; exact contrast counts, yardstick and nominal language; five-family rule descriptive only | §7, §8 |
| **C6** L1 matching | 28/28 × 17 checks, no code drift, deterministic draws; label MATCHED_BY_RECORDED_IDENTITY | §9.2 |
| **C7** look accounting | definitions, unit states, recovery and duplicate prevention; counts unchanged (232; 196 → 428), protected inference 26 → 24 | §10, §12.1 |
| **C8** score-table guard | first 5.2 deliverable; fixture-only development; separate authorised path after exposure | §12.2 |
