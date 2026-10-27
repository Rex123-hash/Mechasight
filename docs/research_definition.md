# MechaSight — Research Scope and Experimental Definition

<!-- Legacy roadmap name: Phase 0 -->

**Current status:** Benchmark Integrity and Evaluation Validation — COMPLETE.
The [final evaluation report](phase34_final_report.md) records the corrected sibling-exposure
result and closes this work. Single-Modality Model Benchmarking is next and has not started;
the [project roadmap](../README.md#roadmap) gives the current descriptive stage names.

This document preserves the original research rationale and its earlier amendments. Internal
stage numbers, candidate-model decisions and prospective statements in the historical sections
describe planning at that time, not the current execution status. Later frozen study records
and the final evaluation report establish the completed methods and results.

Every *performance* number quoted in **this document** remains a **published result by other
authors**, explicitly attributed. MechaSight's own first results are **not** reproduced here — they
live in `reports/phase3_2/` with the corrected reporting layer in `reports/phase3_2_1/`, and must
always be quoted per machine with their cluster counts and limitation ids. Dataset *measurements*
in §5 are our own, from `reports/dataset_audit/`.

> **PHASE 1 CHANGED THIS DOCUMENT.** Sampling rates, segment durations and the split structure
> stated in Phase 0 were wrong or incomplete. See `reports/dataset_audit/dataset_summary.md` and
> §5.11 below. The two most consequential findings: **the two machines differ in sampling rate,
> duration and recording structure**, and **RoboticArm's official train/test split is not
> recording-disjoint** — 91.4% of its normal test segments have a sibling from the same continuous
> recording in the training set.

**Document version:** 1.4 (2026-09-01) — Phase-2.2 acceptance recorded; Phase-3.0 safety preflight.

> **PHASE-2.1 OVERRIDE.** Where older sections below still describe equal released counts,
> segment-level bootstrap, pAUC as co-primary, or unequal Regime-S/Regime-T test populations,
> `docs/evaluation_protocol.md` v2.1 and `configs/evaluation/protocol_b_common_test_v2.json` are the
> authoritative replacements. The historical wording remains visible rather than silently erased.

*Revision 1.1:* paired bootstrap replaced marginal-CI comparison (§9, §12, §14); S+T sample sizes
corrected to 232/232 and 156/156; microphone conclusions restricted to what masked-input evaluation
supports and the vibration-primary framing withdrawn (§5.10, D2); XGBoost/LightGBM **deferred**
rather than recast (§11, D5); k-NN/Mahalanobis justification marked unverified (§11, D7); novelty
assessment upgraded to two fully-read primary papers (§8, D11); `standardize()` defect scoped to
the code path that contains it (D10).

*Revision 1.2:* all remaining marginal-CI decision language removed and the "differences below
~9 AUC points are unresolvable" statement withdrawn — Hanley–McNeil widths are planning
illustrations only and set no threshold (§14, §17); bootstrap stratification specified as the four
domain × class strata for S+T and class-only for single-domain analyses (§12, §14); **one** seed /
bootstrap algorithm frozen, with no alternatives retained (§12); H1–H3 split into independently
reported per-machine hypotheses with no project-wide success declaration from a single machine
(§9, §15); Chu & Zeng characterisation corrected — the "contains no fault data" claim is
**withdrawn** as the article is internally inconsistent and its §4.4 reports fault experiments, so
unseen-fault detection is no longer claimed as a distinguishing element (§8, D11).
**Evidence tags used throughout:** `[VERIFIED]` `[LIKELY / NEEDS DATASET CONFIRMATION]` `[UNVERIFIED]` `[CONTRADICTED]`

---

## 1. Project title

**MechaSight — Robust Multi-Sensor Machine Anomaly Detection Under Real-World Operating Shifts**

Minor project, Mechanical & Automation Engineering. Research-first: the deliverable is
reproducible evidence about what works, not a product demo.

---

## 2. Problem statement

A change in a machine's measured signal does not by itself mean the machine is faulty.
A healthy machine emits different sound, vibration and rotational patterns when its load
changes, its speed changes, or the acoustic environment around it changes. A genuine
mechanical fault also changes those signals.

An anomaly detector trained on one set of operating conditions therefore has two ways to
fail, and they pull in opposite directions:

- **False alarm under shift** — it flags a healthy machine running under a new load/speed/noise
  condition as anomalous, because that condition is unfamiliar rather than faulty.
- **Missed fault** — it is made so tolerant of variation that a real fault no longer stands out.

This is not a hypothetical framing. The IMAD-DS authors observed exactly this failure mode
and named its mechanism: some domain-combined AUCs fall *below* the AUCs of either domain
alone because "normal samples in the target domain have higher anomaly scores than anomalous
samples in the source domain, leading the model to mistake domain changes for anomalies"
(Albertini et al., DCASE 2024). `[VERIFIED — primary source]`

Multi-sensor measurement is the usual remedy, on the reasoning that a fault visible to one
sensor may be invisible to another. But the same primary source shows that naive fusion is
not automatically the answer either: for the Brushless Motor, feeding the autoencoder all
three sensors gave AUC 58.95, **worse** than the accelerometer-derived score at 69.30.
The authors' own conclusion is that "using [the reconstruction loss] as an anomaly score does
not guarantee that using all sensors always yield optimal performance. Therefore, exploring
alternative multi-sensor methods is key to fully exploiting the potential of multi-sensor
data." `[VERIFIED — primary source]`

That sentence is the load-bearing justification for this project.

---

## 3. Research question

**Primary (frozen):**

> On a multi-rate, multi-sensor industrial anomaly-detection dataset with documented
> operational and environmental domain shift, can an anomaly-detection system that weights
> each sensing modality by an estimate of that modality's current reliability detect genuine
> machine anomalies **more robustly** — under domain shift and under degraded/missing sensors —
> than (a) the best single modality, (b) naive equal-weight fusion of all modalities, and
> (c) established baselines, when all are trained under the same normal-only protocol and
> compared with uncertainty quantified?

**Secondary questions:**

- **RQ2.** How much anomaly-detection performance is lost when moving from source-domain to
  target-domain operating conditions, per modality and per fusion strategy?
- **RQ3.** Which modality carries the anomaly evidence for each machine and each anomaly type,
  and does that ranking change between source and target domains?
- **RQ4.** How gracefully does each system degrade when a sensor is removed, saturated, biased,
  dropped-out, or swamped by injected noise at test time?

**Explicitly not claimed:** that reliability-aware fusion is novel in general (it is not — see
§8), or that it will win (unknown — see §9).

---

## 4. Dataset

**Primary dataset (frozen): IMAD-DS** — *Industrial Multi-sensor Anomaly Detection under
Domain Shift Conditions.*

| Field | Value | Evidence |
|---|---|---|
| Authors | Augusti F., Albertini D., Esmer K., Sannino R., Bernardini A. (STMicroelectronics + Politecnico di Milano) | `[VERIFIED]` Zenodo API + paper |
| Paper venue | DCASE 2024 Workshop, Tokyo, 23–25 Oct 2024 | `[VERIFIED]` |
| Record DOI | `10.5281/zenodo.12665499` (concept DOI `10.5281/zenodo.12636236`) | `[VERIFIED]` Zenodo API |
| Published | 2024-07-05 | `[VERIFIED]` Zenodo API |
| Licence | CC BY-SA 4.0 | `[VERIFIED]` Zenodo API |
| Files | `RoboticArm.7z` (3,545,453,763 B ≈ 3.30 GiB), `BrushlessMotor.7z` (1,577,018,050 B ≈ 1.47 GiB) | `[VERIFIED]` Zenodo API |
| Acquisition hardware | STEVAL-STWINBX1 (STWIN.box) IoT industrial sensor node | `[VERIFIED]` |
| Format | per-sensor `.parquet` files + `attributes_*.csv` metadata, split into `train/` and `test/` folders | `[VERIFIED]` Zenodo + baseline code |

**Optional, deferred, secondary datasets:** MIMII DUE and ToyADMOS2 are **audio-only** and
therefore cannot test the multi-sensor hypothesis. They are retained only as a possible
*scoped, separately-reported* external check of the audio branch in Phase 9. They must never be
merged with IMAD-DS into a single training pool. `[FROZEN decision]`

---

## 5. Verified dataset facts

All facts below are from the DCASE 2024 paper, the Zenodo record, or the authors' released
baseline code. Nothing here comes from the downloaded dataset, because **the dataset has not
been downloaded** — see §18.

### 5.1 Machines and anomalies `[VERIFIED]`

| Machine | What it is | Anomaly mechanism |
|---|---|---|
| **Robotic Arm** | Scaled version of a robotic arm used to move silicon wafers in a factory, reproducing actual factory movements | Screws at the arm's nodes are loosened / bolts removed, causing imbalance and the spatial miscalibration typical of such machines |
| **Brushless Motor** | Scaled representation of an industrial brushless motor | (1) a magnet is moved closer to the motor load, causing oscillations by interacting with two symmetrical magnets on the load; (2) a belt rotating in unison with the motor shaft is tightened, creating mechanical stress |

> Wording discrepancy: the paper says "loosening the screws at the arm's nodes"; the Zenodo
> description says "removing bolts at the nodes of the arm". Same physical fault class,
> inconsistent wording between the two primary sources. Recorded, not resolved.

### 5.2 Sensors `[VERIFIED — with an important caveat]`

| Sensor | Part | Published | Reference code | **MEASURED — RoboticArm** | **MEASURED — BrushlessMotor** |
|---|---|---|---|---|---|
| Analog microphone | IMP23ABSU | 16 kHz | 16000 Hz | **16000.000 Hz** | **16003.077 Hz** |
| 3-axis accelerometer | ISM330DHCX | 6.7 kHz | 7063 Hz | **7063.294 Hz** | **7135.716 Hz** |
| 3-axis gyroscope | ISM330DHCX | 6.7 kHz | 7063 Hz | **7063.294 Hz** | **7135.331 Hz** |

`[MEASURED — Phase 1]` Span-estimator rates over all 3,896 segments; within-machine range ≤0.02%.
**These are timestamp-derived effective rates, not a correction to the documented ODR** — see §5.12.
The two machines differ from each other, which no primary source states. The reference code's
7063 Hz matches RoboticArm's effective rate but is applied to both machines. Per-sample estimators
(`1/median(dt)`) are unusable because timestamps are quantised to 1 µs.

The authors' `preprocess.ipynb` sets the IMU rate to `7063` with the comment *"Estimated
sampling rate calculated by averaging time deltas across all files."* The published "6.7 kHz"
is the nominal ODR; the effective rate measured from timestamps is ~7063 Hz, ~5.4% higher.
**MechaSight must not hard-code either value.** The rate must be re-derived from the actual
timestamps in Phase 1. See `docs/assumptions_and_open_questions.md` Q1.

### 5.3 Domain-shift design `[VERIFIED]`

| Machine | Shift axis | Source domain | Target domain |
|---|---|---|---|
| Robotic Arm | Attached load (weight index) | 00, 10, 15, 20 | 25, 30, 35 |
| Robotic Arm | Factory background noise @ SNR −4 dB | A, C, D, F | B, E, G |
| Brushless Motor | Rotation speed [rpm] | 1500, 1600, 1700, 1800, 1900, 2000, 2400, 2800, 3000 | 1000, 1100, 1200, 1300, 1400 |
| Brushless Motor | Factory background noise @ SNR −4 dB | A, C, D, F | B, E, G |

Seven background-noise recordings from different real factories, all scaled to SNR −4 dB.
(The paper's table caption says "Letters A to F index 7 background noise recordings", which
is internally inconsistent — A–G is seven letters. Treated as a typo in the caption; the table
body listing A,C,D,F | B,E,G is taken as authoritative.)

**Two consequences that materially constrain this project:**

1. **The SNR is the same (−4 dB) in source and target.** The environmental shift is a change of
   *which* factory-noise recording is mixed in, not a change in noise *level*. The microphone is
   heavily corrupted in **both** domains. Any project narrative of the form "clean audio in
   training, noisy audio in deployment" is **wrong for this dataset**.
2. **Background noise is added to audio only.** The paper states: *"we assume that the coupling
   of the machine with its surrounding environment is reflected only in the audio signals, as
   the acoustic coupling is more relevant than the others."* The accelerometer and gyroscope
   are **not** environmentally corrupted by construction. This is discussed as a threat to
   validity in §17.

### 5.4 Audio simulation chain `[VERIFIED]`

Recordings are made in a fully anechoic chamber. Environmental noise is then simulated:
image source method (Allen & Berkley) via **pyroomacoustics**, shoebox room 10 × 7.5 × 4 m,
fixed **T60 = 0.5 s**, RIRs computed once for the whole dataset. Background noise is convolved
with the emitter RIRs, machine sound with its own RIR, the noise is scaled to hit the target
SNR, both are cropped back to the original machine-sample length (removing the reverberation
tail) and mixed. The same setup was used to generate audio for the DCASE 2024 Task 2 challenge.

### 5.5 Segment counts `[VERIFIED]`

Paper Table 2 — number of **segments** per class:

| Machine | Split | Source normal | Source anomaly | Target normal | Target anomaly |
|---|---|---|---|---|---|
| Robotic Arm | Train | 1812 | 0 | 27 | 0 |
| Robotic Arm | Test | 116 | 116 | 116 | 116 |
| Brushless Motor | Train | 1263 | 0 | 18 | 0 |
| Brushless Motor | Test | 78 | 78 | 78 | 78 |

Totals: Robotic Arm 2303, Brushless Motor 1593, **grand total 3896 segments.**
This confirms the "≈3,896 segments" figure carried in the project planning documents, and it is
now sourced rather than assumed. Note the shape of it: **3120 of the 3896 segments (80%) are
training normals; only 776 segments (20%) are test segments**, and the test set is what every
reported metric is computed on.

### 5.6 What one "segment" is `[VERIFIED structure / UNVERIFIED duration]`

One row of an `attributes_*.csv` = one segment = one simultaneously-recorded triple of
`.parquet` files (mic, acc, gyro) sharing a unique ID. The baseline derives `segment_id` by
stripping the `imp23absu_mic_` prefix from the microphone filename.

~~**Segment duration is not stated in any primary source.** DCASE clips are 6–10 s, which makes a
segment length of that order plausible but not established.~~

`[MEASURED — Phase 1: the 6–10 s estimate was WRONG]` Durations are fixed to the microsecond and
**differ by machine**: **BrushlessMotor 5.198938 s**, **RoboticArm 7.499937 s** (std 1.0 µs, and
identical across split, domain and class, so duration leaks nothing). At 100 ms this gives ~51
windows per BrushlessMotor segment and ~74 per RoboticArm segment.

**A RoboticArm "segment" is not an independent recording.** Its 2,303 segments are consecutive
7.5 s excerpts of only 101 continuous recordings — median 16, max 48 excerpts per recording, sharing
one continuous clock. BrushlessMotor is one segment per recording. See §5.11.

### 5.7 Metadata fields `[VERIFIED from baseline code]`

Per-segment metadata columns used by the authors:
`segment_id`, `split_label`, `anomaly_label`, `domain_shift_op`, `domain_shift_env`,
plus one filename column per sensor (`imp23absu_mic`, `ism330dhcx_acc`, `ism330dhcx_gyro`).

`split_label` takes values of the form `Normal_Source_Test`, `Anomaly_Source_Test`,
`Normal_Target_Test`, `Anomaly_Target_Test` (and the corresponding `_Train` forms).
`anomaly_label` is tested as `== 'normal'` vs `!= 'normal'`, i.e. it holds **named anomaly
types**, not a binary flag — consistent with the Brushless Motor having two distinct anomaly
mechanisms. Exact value vocabulary `[NEEDS DATASET CONFIRMATION]`.

The six metadata files are:
`attributes_normal_source_train.csv`, `attributes_normal_target_train.csv`,
`attributes_normal_source_test.csv`, `attributes_anomaly_source_test.csv`,
`attributes_normal_target_test.csv`, `attributes_anomaly_target_test.csv`.

### 5.8 Sensor synchronisation and multi-rate handling `[VERIFIED from baseline code]`

The parquet files carry a `Time` column in seconds. The authors align sensors **by timestamp,
not by sample index**:

- Each sensor's frame is indexed by `Time` and sorted.
- Segment start = `max(all sensors' first timestamp, gyro_first_timestamp + 35 ms)`.
- Segment end = `min(all sensors' last timestamp)`.
- The **first 35 ms of gyroscope data is discarded** for a documented
  `GYROSCOPE_WARM_UP_TIME` — a real sensor artefact that must be respected.
- The common interval is cut into **non-overlapping 100 ms windows**; window count =
  `floor((end − start) / 100 ms)`.
- Each sensor keeps its **native rate** — no resampling. A window is 1600 mic samples and
  706 IMU samples (`int(7063 × 0.1)`). Windows shorter than expected are zero-padded, longer
  ones truncated.

This validates the project instruction that sensors should be aligned by justified temporal
logic rather than forced to an identical raw sample count.

### 5.9 The authors' baseline `[VERIFIED]`

Fully-connected autoencoder over the concatenation of all three sensors' 100 ms windows:
`FC(ΣL_s·C_s → 2048) → FC(2048 → 2048) → FC(2048 → 2048) → FC(2048 → 16)` (ReLU), mirrored
decoder. Adam, lr 1e-4, batch 1024, MSE. Trained on **normal data only**; 10% of training
windows held out for validation; best model = lowest validation loss.

Segment anomaly score = **median** of the per-window reconstruction errors in that segment.
Metric = ROC AUC, computed separately for Source, Target and Source+Target.

### 5.10 Published baseline results `[VERIFIED — these are the authors' numbers, not ours]`

AUC (%), from paper Table 4. `F-x` = per-sensor reconstruction error while all sensors are
present; `S-x` = per-sensor error while the other sensors are **zeroed out** at the input.

| | RA S+T | RA Source | RA Target | BM S+T | BM Source | BM Target |
|---|---|---|---|---|---|---|
| Overall | 91.62 | 93.28 | 90.48 | 58.95 | 73.63 | 55.59 |
| F-acc | 90.49 | 98.98 | 94.00 | 69.30 | 77.80 | 59.62 |
| S-acc | 88.96 | 98.40 | 84.24 | 67.38 | 77.17 | 56.03 |
| F-gyr | 87.88 | 93.91 | 93.37 | 57.27 | 68.28 | 55.70 |
| S-gyr | 46.79 | 44.99 | 48.54 | 57.38 | 68.11 | 56.49 |
| F-mic | 66.31 | 73.27 | 63.18 | 54.19 | 58.83 | 49.27 |
| S-mic | 50.92 | 52.11 | 49.69 | 50.71 | 53.13 | 46.10 |

Four things MechaSight must carry forward from this table:

1. **The two machines are not equally hard.** Robotic Arm is largely solved by the baseline
   (91.6 overall); Brushless Motor is not (58.95, barely above chance). Reporting a single
   averaged headline number across both machines would hide this and must not be done.
2. **Under masked-input evaluation of the joint autoencoder, the microphone subvector carries
   almost no separating information** (S-mic 50.92 / 50.71 ≈ chance) at −4 dB SNR. This is a
   statement about *that* evaluation, not about audio as a modality: `S-mic` is the joint
   three-sensor AE fed an input whose accelerometer and gyroscope blocks are zeroed, which is
   off-distribution for a model trained with all three present. **No microphone-only model was
   ever trained by the authors**, so no conclusion about a genuinely audio-only detector can be
   drawn from this row. Establishing what audio alone can do requires training it — Phase 3.
3. **The accelerometer subvector carries the most separating information in this baseline**, and
   domain shift costs it a lot on the Brushless Motor (77.80 → 59.62).
4. **Naive all-sensor fusion loses to the accelerometer on the Brushless Motor**
   (58.95 vs 69.30). This is the opening the project aims at.

### 5.11 Phase 1 audit findings `[MEASURED — our own results]`

Full evidence in `reports/dataset_audit/dataset_summary.md`. Summary of what the real release
shows, and how it differs from what Phase 0 assumed.

**Confirmed.** All 3,896 segments present, complete and readable; counts match Table 2 cell by cell.
Zero NaN, Inf, constant channels, duplicate ids or label contradictions across 27,272 channel
instances. Table 1's operating and background conditions match exactly. Sensors are genuinely
co-clocked. The released audio **is** pre-mixed, and the environmental noise **is** audio-only —
both now verified from data rather than taken from the paper's text. IMU units are `g` and `mdps`.

**Contradicted.** Sampling rates (§5.2), segment durations (§5.6), and the assumption that no test
segment shares a parent recording with a training segment (below).

**RoboticArm and BrushlessMotor should be treated as two separate experimental cohorts within the
same dataset**, because their acquisition and anomaly characteristics differ substantially — a fact
no primary source mentions. Each carries a distinct validity problem:

| | BrushlessMotor | RoboticArm |
|---|---|---|
| Segments per parent recording | 1 | median 16, max 48 |
| Normal test segments with a training sibling from the same recording | **0 / 156 (0%)** | **212 / 232 (91.4%)** |
| Normal vs anomalous recording windows | **disjoint** (7.3 min gap) | overlapping |
| Duration / mic rate / IMU rate | 5.199 s / 16003.08 / 7135.7 Hz | 7.500 s / 16000.00 / 7063.3 Hz |
| Sensor start offset | 16.66 ms | 0.036 ms |
| IMU sampling | 2.7% irregular intervals | uniform |

**R9 realised on RoboticArm.** 62 of 101 recordings straddle the train/test split, affecting 1,837
of 2,303 segments (79.8%). Test excerpt `dataLog_8` (82.7–90.2 s) sits ~2.5 s from training
excerpts `dataLog_7` and `dataLog_9` in the same continuous recording at the same load. Because
training is normal-only, the violation is **asymmetric**: normal test segments have near-twins in
training while anomalous ones cannot.

**What this establishes and what it does not.** Established: parent-recording independence is
violated, and the violation falls on one class only. **Not established:** that AUC is inflated, or
by how much. The correct statement is that **the released RoboticArm split creates a credible risk
of optimistic bias, of unmeasured magnitude.** It is a *candidate* explanation for the published
RA-vs-BM gap (91.62 vs 58.95), competing with genuine differences in machine difficulty. The
hypothesis will be tested in Phase 3+ by running the same baseline under the official and
recording-disjoint splits, changing nothing else — see `docs/evaluation_protocol.md` §5. Both
outcomes are reportable, including no measurable difference.

**New confound on BrushlessMotor.** All normals were recorded before all anomalies, with zero
overlap, so recording time predicts the class perfectly. Fault effects and acquisition-session
effects are consequently **not identifiable** for this cohort, and no analysis of this dataset can
separate them — an early-vs-late normal comparison is a drift *diagnostic*, not a deconfounding.
The two anomaly types are also temporally blocked from each other. See
`docs/evaluation_protocol.md` §6.

**Other findings:** timestamps quantised to 1 µs; BrushlessMotor's IMU is bimodally sampled (2.7% of
intervals ~18 µs shorter), so it is not on a uniform grid; an undocumented accelerometer
first-sample rail artefact at ±16 g; 85 RoboticArm ids carry an undocumented `corr` marker; the
microphone unit is genuinely UNKNOWN and its scale differs between machines.

### 5.12 
Two different quantities were being conflated. They are now kept apart:

**A. Nominal / documented acquisition rate.** The DCASE paper and the Zenodo description both state
16 kHz for the microphone and **6.7 kHz** for the accelerometer and gyroscope. This is the
configured ISM330DHCX output data rate. `[VERIFIED from S1, S2]`

**B. Timestamp-derived effective rate in the released parquet files.** Computed by the span
estimator over all 3,896 segments:

| | mic | accelerometer | gyroscope |
|---|---|---|---|
| RoboticArm | 16000.001 Hz | 7063.28 Hz | 7063.28 Hz |
| BrushlessMotor | 16003.077 Hz | 7135.71 Hz | 7135.33 Hz |

**Correct statement:** *the released parquet timestamps imply machine-dependent effective sample
rates that differ from the nominal acquisition rates documented in the paper.* The earlier wording
"the published 6.7 kHz is wrong" is **withdrawn** — it conflated a documented configuration setting
with a measured delivery rate, and the audit has no evidence that the paper misreports the setting.

**What the evidence supports.** The deviation is not a uniform clock error, and its pattern is
diagnostic:

- The **microphone** is accurate and session-invariant: RoboticArm 16000.001 Hz (**+0 ppm** from
  nominal), BrushlessMotor 16003.077 Hz (**+192 ppm**). Across RoboticArm's 101 recording sessions
  the mic rate varies by **0 ppm** (between-session SD 0.0009 Hz, *smaller* than within-session SD).
- The **IMU** is offset by percent and varies by session: **+5.94%** (RA) and **+7.03%** (BM) from
  6667 Hz, differing from each other by **1.03%**. Across RoboticArm's sessions the IMU rate varies
  **19× more between sessions than within** one (between-session SD 0.359 Hz vs within 0.019 Hz;
  182 ppm spread).
- The sample **count** corroborates the timestamps: a 5.1999 s BrushlessMotor segment contains
  37,106 IMU samples where 6667 Hz would give 34,668. There are genuinely more samples than the
  nominal rate over that duration, and the microphone independently confirms the duration.

**Documented mechanism, register value unavailable.** The ISM330DHCX datasheet documents
`INTERNAL_FREQ_FINE`, an 8-bit two's-complement register describing effective ODR/timestamp-rate
offset in 0.15% steps. For selected ODR 6667 Hz,
`ODR_actual = 6667 × (1 + 0.0015 × INTERNAL_FREQ_FINE)`. The observed rates imply approximately
+39.6 (~+40) for RoboticArm and +46.9 (~+47) for BrushlessMotor, which reproduces them closely.
The audio and IMU paths using different clock sources remain consistent with the wider pattern.

**The released timestamp-derived rates are quantitatively consistent with the ISM330DHCX
documented internal-frequency variation mechanism. Because the acquisition-time sensor-register
values are not present in the archived dataset, this explanation is strongly consistent with, but
not proven by, the released data.**

**Alternative explanations not excluded:** a different ODR configuration than the one documented; a
timestamp-reconstruction step in the acquisition firmware; or post-hoc timestamp interpolation.
Distinguishing them would need the acquisition configuration or firmware, which the release does not
include.

**What is safe to conclude, and what MechaSight does:** whatever the cause, the released timestamps
are the only timing information available, and they are internally consistent and highly stable
within a machine (range ≤0.02%). Rates are therefore **derived per file from its own timestamps**,
never hard-coded, and the nominal and effective figures are always reported as distinct quantities.
---

## 6. Source / target / domain definitions

- **Source domain** — the operating and environmental conditions under which abundant normal
  training data exists (RA loads 00–20 with noises A,C,D,F; BM speeds 1500–3000 rpm with
  noises A,C,D,F). Analogue: the controlled commissioning environment.
- **Target domain** — conditions in which normal training data is severely limited
  (RA: **27** training segments; BM: **18**) but which are equally represented at test time
  (RA 116+116; BM 78+78). Analogue: the actual deployment environment.
- **Operational domain shift** — a change in an allowed machine working configuration:
  attached load (RA) or rotation speed (BM). The machine is healthy; only its duty point moved.
- **Environmental domain shift** — a change in which factory background-noise recording is
  mixed into the audio channel, at fixed SNR −4 dB. Affects the microphone only.
- **Source + Target ("S+T")** — the pooled test set. This is the hardest and most realistic
  condition, because the scoring must separate anomalies from normals *across* domains with a
  single score scale; it is where "domain change mistaken for anomaly" shows up.

---

## 7. Normal / anomaly definition, and the learning problem

**Normal** = the machine operating without an introduced physical fault, under any of its
allowed load/speed settings, with any of the background-noise conditions. Normality therefore
*includes* wide legitimate variation. This is the crux of the whole project.

**Anomalous** = a physically introduced fault: loosened node screws (RA); displaced magnet or
tightened belt (BM). Anomalies are induced, not naturally occurring, and are of a small number
of known types.

**Task type — frozen: unsupervised / one-class (normal-only) anomaly detection.**

The dataset is built for it: *"Unsupervised AD systems exclusively use normal data for
training... Anomalous samples are included only in the test set to assess the system's
capability to detect unknown anomalies."* `[VERIFIED]`

Concretely, the contract MechaSight binds itself to:

| Available at training time | Available at test time (for scoring only) |
|---|---|
| Normal segments only (source 1812/1263, target 27/18) | Normal + anomalous segments |
| `domain_shift_op`, `domain_shift_env` metadata for those normal segments | Ground-truth `anomaly_label` — **used only to compute metrics, never as model input** |
| Sensor signals | Sensor signals |

**Anomaly labels are never a training target.** Any model family that requires them is either
excluded or re-cast (see §11).

Two sanctioned variants, to be labelled distinctly wherever reported:

- **(a) Source-only training** — train on source normals only; target normals are unseen.
  This is the strict domain-*generalisation* setting and is the more honest test of robustness.
- **(b) Source + few target normals** — the dataset's own setting, adding the 27/18 target
  normals. This is few-shot domain *adaptation*.

Both are legitimate. They must never be compared against each other as if they were the same
experiment.

---

## 8. Research gap — honest assessment

The proposed gap, restated from the project plan:

> *Existing industrial anomaly-detection systems may lose reliability under operational/
> environmental domain shift. Multi-sensor information can help, but blindly trusting every
> sensor may not always produce the most robust result. MechaSight therefore investigates
> reliability-aware sensor fusion and robustness to sensor degradation.*

**Directly supported by primary sources:**

- That AD performance degrades under domain shift on this dataset — the authors demonstrate it
  and say so in the abstract. `[VERIFIED]`
- That multi-sensor data generally helps relative to a masked single sensor. `[VERIFIED]`
- That *blindly* fusing all sensors is **not** always optimal, and that alternative multi-sensor
  methods are needed — the authors state this explicitly and their Brushless Motor numbers
  demonstrate it. `[VERIFIED]` This is the strongest single piece of support the project has.
- That domain shift is a recognised, actively-worked problem in machine condition monitoring —
  DCASE Task 2 has been organised around it since 2020, with MIMII DUE, MIMII DG and
  ToyADMOS2 built for it. `[VERIFIED]`

**Our proposed contribution / hypothesis (not established fact):**

- That an explicit *per-modality reliability estimate*, computed at inference time without
  anomaly labels, can be used to weight modality contributions and improve robustness on
  IMAD-DS relative to the best single modality and to equal-weight fusion.
- A like-for-like comparison of model families under one leakage-controlled protocol on this
  dataset, with uncertainty quantified.
- Genuinely single-modality models as baselines (see the disagreement below).

**Overstated, and corrected here:**

- **"Reliability-aware / uncertainty-weighted fusion is novel."** It is not, and this is now
  established from primary sources read in full rather than from search snippets — see
  `docs/source_verification.md` S5.
  - **Chu & Zeng (2025, PLoS ONE)** already apply *adaptive confidence-weighted Dempster–Shafer
    evidence fusion* to multi-sensor **industrial motor condition monitoring**, weighting sensors
    by instantaneous classification confidence to resist noisy or partially missing signals.
  - **Jeong et al. (2025, arXiv:2505.02393)** already use *inverse-variance (precision) weighting*
    of modality latents for anomaly detection, and explicitly frame it as countering **modality
    dominance** — the same failure mode we observe in the IMAD-DS Brushless Motor result.

  Chu & Zeng go further than the mechanism alone: their Section 4.4 reports zero-shot flagging of
  bearing-wear and rotor-imbalance faults via Dempster–Shafer conflict and entropy. **So
  "detecting unseen faults by reliability-weighted multi-sensor fusion" is itself not a
  distinguishing claim, and MechaSight does not make one.** (That article is internally
  inconsistent about whether it contains fault data — its Introduction says it does not, its
  Section 4.4 says it does. See `source_verification.md` S5a; an earlier draft of this document
  repeated the Introduction's claim and has been corrected.)

  What remains open — and is therefore the contribution — is the **combination of setting
  conditions**: **normal-only (one-class) training**, with no supervised target from which a
  confidence signal could be derived; **multi-rate heterogeneous physical sensors**; and a
  **documented operational *and* environmental domain-shift protocol** with a defined source/target
  split. Chu & Zeng's primary task is supervised operating-state classification with no domain-shift
  protocol and a softmax-derived confidence signal; Jeong et al. are weakly supervised on
  video-level labels with image+event modalities and no domain-shift protocol. The claim MechaSight
  may make is that *combination*, plus honest evaluation against genuinely single-modality
  baselines. The phrase "nobody has done this before" must not appear in any project document.
- **"Existing systems are single-sensor."** Overstated. The IMAD-DS paper itself cites TEP, HAI,
  CWRU and SKAB as multi-sensor industrial AD datasets. The accurate claim is narrower: most
  industrial AD datasets that model **domain shift** are single-sensor (audio), and multi-rate
  multi-sensor + domain shift together is what IMAD-DS was built to fill.
- **"Audio corrupted by factory noise while vibration stays clean is a real-world finding."**
  On this dataset it is a **construction decision** by the dataset authors (§5.3), not an
  empirical discovery. Must be stated as such every time.

**Refined research question:** see §3. The refinement versus the synopsis is that the
comparison targets are named and pre-committed (best single modality, equal-weight fusion,
established baselines), so that "improvement" cannot be declared against a straw man.

---

## 9. Proposed hypothesis

**The two machines are separate experiments and are hypothesised separately.** The Robotic Arm and
the Brushless Motor differ so much in difficulty under the reference baseline (91.62 vs 58.95
overall AUC) that a hypothesis spanning both would be answerable by whichever machine happened to
cooperate. Every hypothesis below is therefore stated, tested and reported **once per machine**,
with identical methodology and no pooling.

**H1-RA.** On the **Robotic Arm**, weighting each modality's contribution to the anomaly score by
an inference-time reliability estimate yields higher segment-level AUC and pAUC in the Source+Target
condition than (i) equal-weight fusion of all modalities and (ii) the best single modality.

**H1-BM.** The same statement for the **Brushless Motor**, tested independently.

**H2-RA / H2-BM.** Reliability-aware fusion degrades more gracefully than equal-weight fusion when
a modality is removed or corrupted at test time — tested and reported per machine.

**H3-RA / H3-BM.** The source→target AUC drop is smaller for reliability-aware fusion than for
equal-weight fusion — tested and reported per machine.

**Pre-committed falsification conditions.** Because all systems are scored on the *same* test
segments, superiority is decided by the **paired parent-recording cluster bootstrap** frozen in
`docs/leakage_risks.md` R8 — the sole decision procedure in this project. A per-machine hypothesis
H*x*-RA or H*x*-BM is supported only when the bootstrap CI of **ΔAUROC** *and* **ΔAUPRC** against
**each** comparator **excludes zero in the favourable direction**, with both systems scored on
identical stratified parent-cluster draws in every replicate. It is rejected otherwise. pAUROC is
secondary under the Phase-2.1 freeze.

**No project-wide success declaration from a single machine.** If H1-RA is supported and H1-BM is
not — or the reverse — the finding is reported exactly that way: *supported on one machine, not on
the other*, with both results given equal prominence and the difference discussed. There is no
combined "MechaSight works" claim, no cross-machine pooling of the paired test, and no presentation
of the successful machine as the headline with the other in a footnote. Two machines are not enough
to generalise from even when both agree; they are certainly not enough when they disagree.

**A negative result on both machines is a valid project outcome and will be reported as the
headline finding if that is what the evidence says.**

**Anti-goal, stated so it can be checked against later:** no experiment may be designed,
selected, or reported for the purpose of making reliability-aware fusion look good. Comparators
get the same tuning budget, the same protocol, and the same reporting detail.

---

## 10. Objectives

Mapped to the three synopsis objectives, made concrete and kept within the frozen scope.

**Objective 1 — Compare model families.** Under one leakage-controlled, normal-only protocol,
compare statistical, classical one-class, reconstruction-based and temporal model families
across microphone, accelerometer and gyroscope, on both machines, reporting source, target and
S+T performance with uncertainty.

**Objective 2 — Improve robustness under shift.** Quantify the source→target degradation of
every system; quantify false positives on *healthy target-domain* segments specifically; then
develop and evaluate reliability-aware fusion against the comparators in §9.

**Objective 3 — Stress-test and explain.** Evaluate every surviving system under injected
sensor degradation (missing, dropped-out, biased, saturated, additionally noised) and attribute
anomaly decisions to modalities, per machine and per anomaly type.

---

## 11. Candidate model families and their roles

Assessed against one question: **does it fit a protocol where anomaly labels exist only in the
test set?**

| Family | Role | Fits normal-only protocol? | Verdict |
|---|---|---|---|
| **Statistical baseline** (per-channel RMS/kurtosis/crest-factor/band energies → Gaussian or Mahalanobis distance) | The floor everything must beat; also the mechanical-engineering-literate baseline (these are standard vibration condition-monitoring indicators) | Yes — fit on normal training data only | **KEEP — build first** |
| **GMM / KDE density on features** | Slightly richer density baseline; handles multi-modal normality from multiple load/speed settings | Yes | **KEEP** |
| **k-NN / Mahalanobis on embeddings** | Not in the synopsis. A strong embedding-distance comparator: score a test segment by its distance to the training-normal embedding distribution, rather than by reconstruction error. Included because modern DCASE-style anomaly detection commonly scores in an embedding space rather than by reconstruction, so omitting it risks a comparison that includes no such method at all. | Yes | **ADD — see evidence note below** |
| **Isolation Forest** | Classical unsupervised outlier detector on engineered features | Yes | **KEEP** |
| **One-Class SVM** | Classical one-class boundary method; the textbook fit for this protocol | Yes, but scales poorly in the number of training windows — subsample | **KEEP with caveat** |
| **XGBoost / LightGBM** | ⚠️ **Does not fit as an anomaly classifier.** There are no anomaly labels in training. Training them on test anomalies would be a protocol violation and test leakage. | **No, not as-is** | **DEFER pending Phase 1 — see below** |
| **Autoencoder (dense)** | Reconstruction-error scorer; also the exact reference baseline, so it is our reproducibility anchor | Yes | **KEEP — must reproduce the published numbers first** |
| **1D CNN** | ⚠️ Not a model family by itself — needs an objective. Valid as a **CNN autoencoder** or as a self-supervised attribute classifier. Not valid as a normal-vs-anomaly classifier. | Yes, once re-cast | **KEEP, re-cast** |
| **TCN** | Same caveat as 1D CNN. Justify by dilated receptive field over 100 ms windows; only worth it if it beats the CNN-AE. | Yes, once re-cast | **DEFER to Phase 5, drop if it earns nothing** |
| **Reliability-aware fusion** | The proposed contribution | Yes, if the reliability estimate uses no anomaly labels | **KEEP — Phase 6, and it must be built on whichever encoder won Phases 3–5, not on a favourite** |

### The XGBoost / LightGBM problem — DEFERRED, not solved

Gradient-boosted trees are supervised. The synopsis lists them as "feature/tree-based
benchmarks" for anomaly detection. Under IMAD-DS's official protocol there is no label to boost
on: training is 100% normal. **Status: DEFERRED pending Phase 1. No decision is taken here.**

The obvious escape route is to re-cast the GBM as a self-supervised *operating-condition
classifier* — train on normal training segments to predict `domain_shift_op` / `domain_shift_env`,
then score a test segment by the negative log-likelihood of its predicted condition. That is a
real family of approach and it uses no anomaly labels. But it **cannot be adopted yet**, because
it rests on assumptions about the metadata that Phase 0 has no way to check:

1. **It needs a target for test segments.** Negative log-likelihood is computed against *some*
   condition label. If that label is taken from the test segment's own metadata, the method
   assumes operating condition is known at inference time — a deployment assumption (that a plant
   knows its current load/speed setpoint) that IMAD-DS does not itself justify and that the
   project has not argued for. If instead the score is a label-free quantity such as predictive
   entropy or maximum softmax, that is a *different method* with different properties and needs
   its own justification.
2. **The condition vocabulary is unknown.** Whether `domain_shift_op` values are the raw load
   indices and rpm figures, coarser bins, or something else is open question Q3/Q4.
3. **Class balance is unknown.** The target domain has 27 (RA) and 18 (BM) training segments
   spread over several conditions. A classifier over conditions may have classes with a handful of
   examples, making its likelihood estimates unstable exactly where robustness matters most.
4. **Condition may partly predict the anomaly label** (risk R6, open question Q4). If it does,
   a condition-scored method has a shortcut, and this recast is the most exposed of all our
   candidates to it.

**Decision deferred to Phase 1**, which must establish the condition vocabulary, its availability
and its balance across normal and anomalous test segments. Only then can this be assessed.
If Phase 1 shows the assumptions do not hold, **the tree-based family is dropped**. It is not
preserved merely because the synopsis listed it — a benchmark entry that does not fit the protocol
is worse than no entry, and "the synopsis promised one" is not a scientific reason.

A clearly-fenced supervised oracle (trained with anomaly labels under nested cross-validation
*within the test set*, reported in a separate table headed "supervised upper bound — not
comparable to the unsupervised protocol") remains a possible, separate use for answering "how much
signal is present at all?". It is optional and carries a real risk of being quoted out of context.

What must *not* happen under any option is training a GBM on test-set anomalies and reporting its
accuracy alongside the unsupervised models.

### Evidence note on the k-NN / Mahalanobis addition

This comparator is added on the strength of general familiarity with DCASE-style anomaly
detection, **not** on a primary source read during Phase 0. The specific claim that
embedding-distance scoring outperforms reconstruction-error scoring on this class of task is
therefore currently `[UNVERIFIED]` in this project. **Action required before implementation
(Phase 3):** read the DCASE Task 2 challenge-results papers and at least one top-ranked system
description in full, record them in `docs/source_verification.md`, and either substantiate the
justification or drop the comparator. It must not be introduced into the report with an
unsourced claim about leaderboards attached to it.

---

## 12. Evaluation strategy

**Unit of evaluation: the segment.** Windows within a segment are not independent samples;
scoring at window level would inflate the apparent sample count by roughly the number of windows
per segment and would not match the official protocol. Segment score = an aggregation (median,
per the paper) of that segment's window scores. Window-level numbers may be reported as
diagnostics only, always labelled as such.

**Metrics, and why each is here:**

| Metric | Why it answers the research question |
|---|---|
| **ROC AUC, segment level, reported separately for Source / Target / S+T** | The official IMAD-DS metric; required for any comparison to published numbers. S+T is the one that exposes "domain shift mistaken for anomaly". |
| **pAUC @ FPR ≤ 0.1** | The DCASE Task 2 standard. Directly measures usable performance in the low-false-alarm regime a plant would actually accept. This is the principled version of "don't optimise for accuracy". |
| **Source → Target AUC degradation (Δ)** | The direct numeric answer to RQ2 and objective 2. |
| **False-positive rate on healthy *target-domain* segments at a threshold fixed from training normals** | The concrete operational cost of the confusion the project is about. See threshold rule below. |
| **AUPRC** | Reported because it is more informative than AUC under class imbalance. Caveat to state: the IMAD-DS test set is balanced 50/50 by design, so AUPRC adds little here and must not be presented as if it reflected real plant prevalence. |
| **F1** | Only with a **pre-declared, training-derived threshold** (below). "Best F1 over thresholds on the test set" is forbidden. |
| **Paired bootstrap CIs on every head-to-head comparison** | Systems are scored on identical test segments, so the valid test is the CI of the *paired difference* (ΔAUC, ΔpAUC). Resampling is stratified by the four domain × class strata for S+T, and by class alone for single-domain analyses; both systems always receive identical sampled segment IDs. Marginal CIs are reported descriptively only. See §14 and the frozen algorithm in `docs/leakage_risks.md` R8. |
| **Per-machine and per-anomaly-type breakdowns** | Robotic Arm and Brushless Motor behave completely differently (§5.10); averaging them hides the finding. |

**Threshold rule (pre-declared, frozen):** where a hard decision is needed, the threshold is the
*q*-th percentile of anomaly scores over held-out **training normal** segments, with *q* fixed
in advance (q = 90 as default) and identical for every model. No test data touches it.

**Calibration / "confidence":** model outputs are **anomaly scores**, not probabilities, and
will be called anomaly scores everywhere — in code, tables, figures and any future UI. A
calibrated probability may only be reported if an explicit calibration procedure is fitted on
held-out *training normals* and its reliability is itself evaluated (e.g. ECE / reliability
diagram). Until then, the word "confidence" is banned from outputs.

**Reproducibility anchor:** before any new model is claimed to be better, the published dense-AE
baseline must be re-implemented and its Table 4 numbers approximately reproduced. If we cannot
reproduce the reference, we have no basis for claiming to beat it. Any gap is reported, not hidden.

**Seeds and paired inference (frozen procedure).** Every configuration is run over **K = 5 fixed
seeds declared in config before the run**. Head-to-head comparison uses the single pre-registered
algorithm in `docs/leakage_risks.md` R8: within each of B = 10,000 stratified bootstrap resamples
of segment IDs — shared identically by both systems — each system's metric is computed for every
one of the K seeds and averaged across them, and the paired difference of those seed-averaged
metrics forms the bootstrap distribution of Δ. The estimand is the seed-averaged system, so the
claim is about the method rather than one lucky initialisation. The resulting interval is
conditional on the fixed seed set and does not carry seed-sampling uncertainty; the ordinary
across-seed **mean ± SD** is therefore reported separately in every table, and comparisons where
that SD is large relative to Δ are flagged as seed-sensitive. No alternative aggregation is
retained.

---

## 13. Robustness experiments

All applied **at test time only**, to a model trained normally. Corruption strength is swept,
not cherry-picked, and the sweep is fixed before results are seen.

1. **Modality ablation** — every subset of {mic, acc, gyro}, with **genuinely retrained
   single-modality and pair models**, not just masked inputs (see the disagreement in §20).
2. **Missing modality at inference** — a modality trained-with but absent at test. Imputation
   rule (zeros post-normalisation = mean imputation, as the authors do) stated explicitly,
   since the choice materially affects the result.
3. **Additive noise** — inject noise into a chosen modality at a swept SNR. Necessary because
   the dataset's own noise is fixed at −4 dB in both domains and is audio-only.
4. **Sensor faults** — bias/offset drift, gain error, clipping/saturation, sample dropout,
   stuck-at values. These are the failure modes real MEMS sensors on a machine actually exhibit,
   which keeps the study grounded in Mechanical & Automation practice.
5. **Cross-domain generalisation** — train on source only, test on target (the strict setting
   from §7a).

**Reporting rule:** for every corruption, report both the corrupted-modality condition *and*
the uncorrupted control, so degradation is measured rather than asserted.

---

## 14. Leakage risks

Summarised here; the full analysis with mitigations is in **`docs/leakage_risks.md`**.
The most serious ones:

- **R1 — Window/segment leakage across splits.** The authors' own baseline splits
  train/validation at the *window* level with `train_test_split(...)`, so windows from one
  recording land in both. MechaSight must split by `segment_id` with grouped splitting.
- **R2 — Normalisation fitted on anything but training data.** Note that the reference baseline
  actually used per-window standardisation (`std_window`), which is leak-free but destroys
  amplitude information — plausibly part of why S-mic ≈ chance. Both options must be evaluated,
  and the choice justified, not inherited.
- **R3 — Model selection or hyperparameter tuning on test AUC.** Selection must use validation
  loss on held-out training normals. This is genuinely hard (validation loss ≠ AUC) and the
  difficulty must be reported rather than quietly resolved by peeking.
- **R4 — Threshold tuned on test.** Prevented by the frozen percentile rule in §12.
- **R5 — Anomaly labels used in training.** Prevented by §7; enforced by an automated test.
- **R6 — Condition metadata as a model input at test time.** Especially in the re-cast GBM.
- **R7 — Window-level metrics presented as if they were segment-level.**
- **R8 — Small test sets, multiple comparisons, and the choice of statistical test.** The primary
  metric (S+T, per machine) has 232 normal vs 232 anomalous segments for the Robotic Arm and
  156 vs 156 for the Brushless Motor; the per-domain breakdowns carry roughly half that. A
  comparison grid of families × machines × domains × sensor configs produces many numbers, some of
  which will look good by chance. Mitigation: one pre-declared primary metric, K = 5 fixed seeds,
  and — critically — **every head-to-head claim decided by the single frozen paired bootstrap of
  ΔAUC/ΔpAUC**, resampling within the four domain × class strata (Source-Normal, Source-Anomaly,
  Target-Normal, Target-Anomaly) for S+T and by class alone for single-domain analyses, with both
  systems scored on identical resampled segment IDs and superiority requiring the paired interval
  to exclude zero. Marginal per-system CIs are reported descriptively and are never used to decide
  anything.

---

## 15. Frozen decisions

Not to be changed without an explicit, evidence-backed review:

1. MechaSight is a **research project**, judged on reproducible evidence, not on headline numbers.
2. Problem domain: **multi-sensor industrial machine anomaly detection under operational and
   environmental domain shift.** Relevant to Mechanical & Automation Engineering through
   condition monitoring, vibration analysis and fault diagnosis.
3. **IMAD-DS first.** MIMII DUE / ToyADMOS2 only as scoped, separately-reported audio-side
   external validation in Phase 9, never merged into IMAD-DS training.
4. **Normal-only (unsupervised/one-class) training protocol.** Anomaly labels are evaluation-only.
5. **Segment-level evaluation**, with source / target / S+T reported separately, per machine.
6. **Baseline-first.** Simple statistical and classical models are built and reported before any
   deep model, and are never omitted from the final comparison because they were beaten or
   because they won.
7. **Reproduce the published baseline before claiming to beat anything.**
8. **Anomaly score ≠ confidence** unless explicitly calibrated and evaluated.
9. **Uncertainty is reported** on every headline comparison, and head-to-head superiority is
   decided by the single frozen **paired bootstrap of the difference** on shared stratified
   resamples of segment IDs. Marginal per-system intervals are descriptive only and decide nothing.
10. **Hypotheses, tests and conclusions are per machine.** Support on one machine is never
    presented as project-wide success.
10. **Negative results are publishable within this project** and will not be buried.
11. Research-critical logic lives in `src/`, not in notebooks.
12. No UI until the research is done; the eventual `app/` is a replay of real saved outputs,
    never a simulated live feed.

## 16. Experimental decisions (open, to be settled by evidence)

Window length (the 100 ms of the baseline is a starting point, not a law) · window overlap ·
sampling-rate handling (native vs resampled) · normalisation scheme · feature set (time-domain
vs FFT/spectral vs learned) · whether to use envelope/order analysis keyed to shaft speed ·
architecture depth and bottleneck size · all hyperparameters · the score-aggregation function
(median vs mean vs trimmed) · which model family wins · the anomaly threshold percentile *q*
(default 90, may be revised **before** looking at test results) · calibration method ·
**the exact reliability-aware mechanism** (candidates: per-modality reconstruction-uncertainty
weighting, learned gating, evidential/Dempster–Shafer combination, modality-dropout-trained
ensembles) · the final fusion architecture · **whether reliability-aware fusion improves anything
at all.**

---

## 17. Limitations

1. **Scaled laboratory machines, induced faults.** Two small machines with a handful of
   deliberately-introduced fault types. Findings do not transfer to arbitrary industrial plant
   without further evidence.
2. **Environmental shift is audio-only by construction** (§5.3). Any conclusion of the form
   "the system learns to distrust the microphone when the environment is noisy" is partly a
   restatement of how the dataset was built. Vibration-side environmental corruption has to be
   injected by us, and injected corruption is not real corruption.
3. **Anechoic-chamber recordings plus a simulated room** (single shoebox geometry, fixed
   T60 = 0.5 s, RIRs computed once). Real reverberation and real machine–environment coupling
   are more varied.
4. **Small test sets** — 464 and 312 segments in total, giving 232/232 (RA) and 156/156 (BM) for
   the S+T primary metric and about half that per domain. Absolute per-system AUC estimates are
   correspondingly imprecise. How small a *difference* between two systems can be resolved is not
   fixed in advance and is not read off any marginal-width table: it is determined per comparison
   by the paired bootstrap, which benefits from the cancellation of shared segment difficulty.
5. **Tiny target-domain training data** (27 and 18 segments) — few-shot adaptation results will
   have high variance.
6. **Only two machines**, with very different difficulty, so per-machine conclusions may not
   generalise even between them.
7. **Anomaly types are few and known.** "Detecting unknown anomalies" is tested only against
   these specific mechanisms.
8. **Fixed SNR −4 dB.** No SNR sweep exists in the dataset; audio robustness across SNR must be
   simulated.
9. **Compute.** Deep temporal models over a large number of multi-rate windows on student
   hardware may constrain model sizes and seed counts; any such constraint will be reported
   rather than hidden.
10. ~~Segment duration and per-window counts unverified.~~ **Resolved in Phase 1** (§5.6).
11. **RoboticArm's official split is not recording-disjoint** — 91.4% of its normal test segments
    have a sibling from the same continuous recording in training, creating a credible risk of
    optimistic bias of unmeasured magnitude. Every RoboticArm result must state which split it used.
12. **BrushlessMotor's classes are perfectly separated in recording time**, so fault effects and
    acquisition-session effects are **not identifiable** for that cohort. Reported detection
    performance may include an unquantified session-drift component, and no available control can
    remove it.
13. **BrushlessMotor's IMU is not uniformly sampled** (2.7% of intervals ~13% shorter), so
    DFT-based spectral methods assuming a uniform grid are approximations for that cohort. Whether
    to resample is an open Phase 2 decision, not a foregone one.
14. **Microphone units are unknown and differently scaled per machine**, so absolute microphone
    values are not comparable across machines.

---

## 18. Open questions

Tracked in full in **`docs/assumptions_and_open_questions.md`**. The blocking ones for Phase 1:

- **Q1.** What is the *actual* IMU sampling rate in the released parquet files — nominal 6.7 kHz,
  the authors' estimated 7063 Hz, or something else? Is it stable across files?
- **Q2.** What is the duration of one segment, and how many 100 ms windows does a segment yield?
- **Q3.** What is the exact vocabulary of `anomaly_label`, and how many test segments does each
  anomaly type have (especially the Brushless Motor's two mechanisms)?
- **Q4.** Are `domain_shift_op` / `domain_shift_env` balanced across normal and anomalous test
  segments? If not, condition metadata partly predicts the label and is a shortcut risk (R6).
- **Q5.** Do the released files actually contain 3896 segments, matching Table 2?
- **Q6.** Are the three sensors' timestamps genuinely overlapping per segment, and how much data
  is lost to the `max(start)/min(end)` intersection and the 35 ms gyro warm-up?
- **Q7.** Are the microphone signals in the released files the noise-mixed versions, or anechoic
  with noise mixing left to the user?
- **Q8.** Does any test segment share an underlying recording with any training segment?

---

## 19. Primary references

Full URLs, access dates and verification notes are in **`docs/source_verification.md`**.

1. Albertini D., Augusti F., Esmer K., Bernardini A., Sannino R. — *IMAD-DS: A Dataset for
   Industrial Multi-Sensor Anomaly Detection Under Domain Shift Conditions*, DCASE 2024
   Workshop, Tokyo, Oct 2024. **Primary source for everything in §5.**
2. Augusti F. et al. — IMAD-DS dataset, Zenodo, DOI `10.5281/zenodo.12665499`
   (concept DOI `10.5281/zenodo.12636236`), CC BY-SA 4.0.
3. `augustif/IMAD-DS_baseline` — the authors' released baseline code (`preprocess.ipynb`,
   `baseline.ipynb`, `model.py`, `utilities.py`). **Primary source for §5.2, §5.7, §5.8, §5.9.**
4. Nishida T. et al. — *Description and Discussion on DCASE 2024 Challenge Task 2: First-Shot
   Unsupervised Anomalous Sound Detection for Machine Condition Monitoring*, arXiv:2406.07250.
   Source for the pAUC@0.1 convention and for RoboticArm/BrushlessMotor appearing in the
   DCASE 2024 Task 2 evaluation set.
5. Tanabe R. et al. — *MIMII DUE*, 2021. Deferred external validation candidate.
6. Harada N. et al. — *ToyADMOS2*, 2021. Deferred external validation candidate.
7. Allen J. B., Berkley D. A. — *Image method for efficiently simulating small-room acoustics*,
   JASA 65, 1979. The RIR method behind the environmental shift.
8. Scheibler R., Bezzam E., Dokmanić I. — *Pyroomacoustics*, ICASSP 2018. Implementation used.
9. Chu G., Zeng P. — *Real-time motor operating state recognition via multi-sensor fusion: A
   wavelet–neural–evidence framework for industrial condition monitoring*, PLoS ONE 20(11), 2025,
   DOI 10.1371/journal.pone.0335360. **Closest prior art** on confidence-weighted multi-sensor
   fusion for industrial motors, including zero-shot fault flagging in its §4.4; read in full.
   Note the article's internal inconsistency about fault-state data, logged in
   `source_verification.md` S5a. See §8 for how our setting differs.
10. Jeong S., Park J., Imani M. — *Uncertainty-Weighted Image-Event Multimodal Fusion for Video
    Anomaly Detection*, arXiv:2505.02393v2, 2025. Inverse-variance modality weighting and the
    modality-dominance framing; read in full.

---

## 20. Claims that differ from the existing project plan

Each item states the existing assumption, the evidence, the assessment, and the recommended change.

### D1 — "Accelerometer/gyroscope sample at 6.7 kHz"

- **Existing assumption:** IMU rate is 6.7 kHz.
- **Evidence:** paper and Zenodo both say 6.7 kHz (nominal ODR), but the authors' own
  `preprocess.ipynb` uses `'fs': 7063` with the comment *"Estimated sampling rate calculated by
  averaging time deltas across all files"*, and derives window lengths from it.
- **Assessment:** partially contradicted. 6.7 kHz is the nominal setting; ~7063 Hz is the
  measured effective rate. Hard-coding either without checking is a reproducibility hazard.
- **Recommended change:** state both in documentation; derive the rate empirically from
  timestamps in Phase 1; never hard-code.

### D2 — "The microphone is a primary sensing modality"

- **Existing assumption:** audio is a co-equal modality, and the audio-side story (factory noise
  swamping the mic) is central to the project narrative.
- **Evidence:** in the published baseline, the microphone subvector scores AUC 66.31 (RA) / 54.19
  (BM) with all sensors present, and 50.92 / 50.71 under masked input. The accelerometer subvector
  scores far higher throughout.
- **Assessment:** an *audio-led* framing is not supported by the published evidence. But the
  published evidence also cannot settle the question, for three reasons that all point the same
  way: (i) `S-mic` is a masked-input probe of a jointly-trained AE, not a microphone-only model —
  no such model exists in the literature for this dataset; (ii) the baseline's per-window
  standardisation strips within-window amplitude, which plausibly handicaps audio badly; (iii) a
  raw 100 ms waveform is a weak audio representation compared with log-mel or spectral features,
  which is what audio anomaly detection normally uses.
- **Recommended change:** treat modality importance as an **open empirical question, not a frozen
  framing**. Stop describing MechaSight as audio-led; do **not** replace that with a frozen
  "vibration-primary" framing either, since that would bake in a conclusion drawn from a masked-
  input probe. Train genuine single-modality models — microphone-only with a fair audio
  representation included — in Phase 3, and let the modality ranking be a *result*. "Can a better
  audio representation make the microphone useful at −4 dB?" becomes an explicit sub-question.

### D3 — "Source domain is clean, target domain is noisy"

- **Existing assumption:** implicit in the project narrative about environmental shift.
- **Evidence:** paper Table 1 — **SNR is −4 dB in both domains**; the shift is *which* factory
  noise recording (A,C,D,F vs B,E,G), not how much.
- **Assessment:** contradicted.
- **Recommended change:** rewrite the environmental-shift description in the synopsis and slides
  as "a change of acoustic environment at constant, severe SNR". Any SNR-sweep robustness result
  must be labelled as our own injected corruption.

### D4 — "The published paper shows multi-sensor beats single-sensor"

- **Existing assumption:** taken from the paper's conclusion.
- **Evidence:** the `S-x` rows are **not single-sensor models**. `utilities.get_masked_batch`
  zeroes the other sensors' inputs into the *same multi-sensor autoencoder*. No genuinely
  single-modality model was ever trained. Zeroing after standardisation is mean-imputation, and
  an AE trained on all three sensors is out-of-distribution when two are constant.
- **Assessment:** the authors' claim is about masked-input degradation, not about single-sensor
  models. It is weaker than it reads. This is an opportunity, not just a criticism.
- **Recommended change:** MechaSight trains **genuine** single-modality and pair models as
  baselines. Doing so is a real, cheap, defensible contribution, and it is required before any
  claim that fusion helps.

### D5 — "XGBoost/LightGBM belong in the anomaly-detection comparison"

- **Existing assumption:** listed as feature/tree-based benchmarks in the synopsis.
- **Evidence:** the dataset's official protocol has zero anomalous training segments; anomaly
  labels exist only in the test set.
- **Assessment:** as anomaly classifiers they cannot be trained without violating the protocol.
  The self-supervised condition-classifier recast is a candidate escape route but rests on
  unverified assumptions about condition-label vocabulary, availability at inference time, and
  balance across normal/anomalous test segments (§11).
- **Recommended change: DEFER to Phase 1.** Do not adopt the recast now and do not train them on
  test anomalies. If the Phase 1 metadata audit does not support the recast, drop the family
  outright rather than preserving it because the synopsis listed it.

### D6 — "1D CNN and TCN are model families to compare"

- **Existing assumption:** listed as temporal models alongside the AE.
- **Evidence:** they are *architectures*; without a stated training objective they are not
  comparable entries. Under normal-only training the objective must be reconstruction or
  self-supervision.
- **Assessment:** underspecified rather than wrong.
- **Recommended change:** specify them as CNN-AE / TCN-AE or as self-supervised attribute
  classifiers, and defer TCN until a CNN-AE has shown temporal modelling is worth the cost.

### D7 — "The model ladder contains no embedding-distance method"

- **Existing assumption:** the ladder ends at AE → CNN/TCN → fusion, i.e. every deep entry is
  scored by reconstruction error.
- **Evidence:** modern DCASE-style anomaly detection commonly scores in an embedding space
  (distance of a test embedding to the training-normal distribution) rather than by
  reconstruction. **This is stated from general familiarity, not from a primary source read in
  Phase 0** — see the evidence note in §11.
- **Assessment:** a comparison containing no embedding-distance method at all would be narrow, and
  would risk our proposal beating only reconstruction-based entries.
- **Recommended change:** add a k-NN / Mahalanobis embedding-distance comparator, **conditional on
  substantiating the justification from primary sources before Phase 3 implementation.** No claim
  about DCASE leaderboard rankings is made in this project until such a source is read and logged.

### D8 — "Accuracy/F1-style reporting on a balanced test set"

- **Existing assumption:** general ML metric habits.
- **Evidence:** the IMAD-DS test set is exactly balanced by design at every level of reporting
  (232/232 RA S+T, 156/156 BM S+T, 116/116 and 78/78 per domain), which is
  nothing like plant reality, where anomalies are rare.
- **Assessment:** balanced-set accuracy and AUPRC would be read as if they meant real-world
  performance. They do not.
- **Recommended change:** lead with AUC and pAUC@0.1 plus target-domain false-positive rate at a
  fixed training-derived threshold; state the balance caveat wherever AUPRC or F1 appears.

### D9 — "A single headline number can summarise MechaSight's performance"

- **Existing assumption:** implicit in presentation-oriented planning.
- **Evidence:** the baseline gets 91.62 on the Robotic Arm and 58.95 on the Brushless Motor.
  An average of ~75 would describe neither machine.
- **Assessment:** averaging across machines would be actively misleading.
- **Recommended change:** all headline reporting is per-machine. No cross-machine mean is
  presented as *the* result.

### D10 — Reference-code defects not to inherit

- **Evidence:** in the authors' released `utilities.py`, `standardize()` is
  `return (data - mean) / std + 1e-5` — the epsilon is added *after* the division, so it does not
  guard against division by zero, contrary to its docstring. Separately, the train/validation
  split in `baseline.ipynb` is window-level, not segment-level (R1), and pandas timestamp slicing
  `sensor_df[start:end]` is right-inclusive, so consecutive 100 ms windows can share a boundary
  sample.
- **Scope limit on the epsilon defect — important.** `standardize()` is reached only by the `'std'`
  normalisation mode. The published notebook sets `normalisation: 'std_window'`, which calls
  `standardize_window()`, where the epsilon **is** correctly placed in the denominator
  (`data /= std_ + 1e-5`). **The defect therefore should not be claimed to affect the reference
  Table 4 run**, and no such claim is made here. It matters to us only because we intend to
  evaluate the train-fitted `'std'` path (R2), which is the path that contains it.
- **Assessment:** minor individually, and the epsilon one is off the published code path, but
  together they show the reference implementation should be read rather than copied.
- **Recommended change:** re-implement preprocessing from the documented *logic*, with unit tests
  (`tests/`) for window alignment, segment-grouped splitting and normalisation-fit provenance.

### D11 — Novelty framing

- **Existing assumption:** reliability-aware fusion is the novel contribution.
- **Evidence:** two primary papers read in full (`source_verification.md` S5). Chu & Zeng (2025,
  PLoS ONE) apply adaptive confidence-weighted Dempster–Shafer fusion to multi-sensor industrial
  motor condition monitoring; Jeong et al. (2025, arXiv:2505.02393) apply inverse-variance
  uncertainty weighting to anomaly detection and name modality dominance as the target problem.
- **Assessment:** the mechanism is established prior art, and Chu & Zeng's §4.4 shows that
  fused-evidence flagging of unseen faults on an industrial motor also already exists — so novelty
  cannot be claimed there either. (That article contradicts itself on whether it contains fault
  data; see `source_verification.md` S5a. An earlier draft of this document asserted it contains
  none, which was wrong and is withdrawn.) What remains genuinely distinct is the *setting*: their
  primary task is supervised operating-state classification, with no domain-shift protocol and a
  confidence signal taken from a supervised classifier's softmax, which does not exist under a
  normal-only protocol.
- **Recommended change:** claim the **combination of setting conditions** — reliability-aware
  fusion under normal-only training, with multi-rate physical sensors, evaluated against a
  documented operational and environmental domain-shift protocol, compared against genuinely
  single-modality and equal-weight baselines with paired uncertainty. Never claim priority over the
  mechanism, and never claim unseen-fault detection as the novel element.

### D12 — Public audio-only results already exist for these exact machines

- **Existing assumption:** not previously accounted for.
- **Evidence:** RoboticArm and BrushlessMotor are machine types in the **DCASE 2024 Task 2
  evaluation dataset** (audio-only, 6–10 s clips, 16 kHz), so published audio-only anomaly
  detection results for these machines exist.
- **Assessment:** an opportunity and a caution — it gives an external audio-side reference point,
  but it also means any claim of the form "no prior results exist for these machines" is false.
- **Recommended change:** cite the DCASE 2024 Task 2 context in the report; treat DCASE audio
  results as *related* rather than directly comparable (different packaging and protocol).

---

*The original research definition preceded dataset acquisition, preprocessing and modelling.
Those activities subsequently progressed through the completed evaluation validation work;
the current status and roadmap are linked at the beginning of this document.*
