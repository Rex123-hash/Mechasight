# IMAD-DS Dataset Audit — Phase 1 Findings

**Generated:** 2026-08-21 · **Auditor:** `scripts/audit_imadds.py` (reproducible)
**Data:** Zenodo record 12665499, both archives MD5-verified against the official checksums.

Reproduce with:

```bash
python scripts/verify_download.py --data-root data/raw
python scripts/extract_imadds.py --data-root data/raw
python scripts/audit_imadds.py --data-root data/raw/extracted --jobs 6
python scripts/audio_noise_probe.py --data-root data/raw/extracted
```

---

## Headline

The release has **pristine file/schema integrity**, **real acquisition artefacts**, and
**materially more problematic split design than expected**. Those are three separate things and are
reported separately throughout.

- **File / schema integrity: perfect.** All **3,896** segments present, complete and readable; no
  NaN, no Inf, no constant channels, no missing or unreadable files, no duplicate ids, no
  metadata/label contradictions. Counts match the published Table 2 exactly, cell by cell.
- **Signal / acquisition characteristics: several real artefacts.** Non-uniform BrushlessMotor IMU
  timing (2.7% of intervals), an accelerometer first-sample rail on 89.8% of BrushlessMotor
  segments, the documented gyroscope warm-up, 1 µs timestamp quantisation, and unknown microphone
  scaling. None of these is a file defect; all of them constrain preprocessing.
- **Split design: two distinct problems**, one per cohort, structurally different in ways the paper
  does not mention:

| | **BrushlessMotor** | **RoboticArm** |
|---|---|---|
| Segments per parent recording | 1 (clean) | **median 16, max 48** |
| Normal test segments with a training sibling from the *same recording* | **0 / 156 (0%)** | **212 / 232 (91.4%)** |
| Normal vs anomalous recording windows | **temporally disjoint** (7.3 min gap) | overlapping (interleaved over 57 days) |
| Segment duration | 5.199 s | 7.500 s |
| Microphone rate | 16003.08 Hz | 16000.00 Hz |
| IMU rate | **7135.7 Hz** | **7063.3 Hz** |
| Sensor start offset (mic→IMU) | 16.66 ms | 0.036 ms |
| IMU sampling regularity | 2.7% irregular intervals | uniform |

**Neither cohort is clean.** RoboticArm's official split violates parent-recording independence;
BrushlessMotor has a strict class/recording-time confound. They are different problems, so
**RoboticArm and BrushlessMotor should be treated as two separate experimental cohorts within the
same dataset**, and the published difficulty gap (91.62 vs 58.95 AUC) now has a candidate structural
explanation competing with genuine machine difficulty.

---

## Q5 — Counts: published vs metadata vs readable

All three agree exactly. `count_comparison.csv`

| Machine | Cell | Published | Metadata rows | Readable triplets |
|---|---|---|---|---|
| RoboticArm | normal/source/train | 1812 | 1812 | 1812 |
| RoboticArm | normal/target/train | 27 | 27 | 27 |
| RoboticArm | normal & anomaly ×{source,target}/test | 116 each | 116 each | 116 each |
| BrushlessMotor | normal/source/train | 1263 | 1263 | 1263 |
| BrushlessMotor | normal/target/train | 18 | 18 | 18 |
| BrushlessMotor | normal & anomaly ×{source,target}/test | 78 each | 78 each | 78 each |
| **Total** | | **3896** | **3896** | **3896** |

Files: 6,909 + 4,779 = 11,688 parquet, 12 metadata CSVs, **zero** other files, zero zero-byte
files, zero orphan parquet files, zero metadata rows without a sensor file.

**Q5: CONFIRMED.** The published total is exactly reproducible.

---

## Q1 — Sampling rates

`sampling_rate_summary.csv`. Two estimators are reported because they disagree, and the reason
matters.

| Machine | Sensor | 1/median(dt) | **(n−1)/span** | Published | Reference code |
|---|---|---|---|---|---|
| BrushlessMotor | mic | 16129.03 | **16003.077** | 16 kHz | 16000 |
| BrushlessMotor | acc | 7092.20 | **7135.716** | 6.7 kHz | 7063 |
| BrushlessMotor | gyro | 7092.20 | **7135.331** | 6.7 kHz | 7063 |
| RoboticArm | mic | 16004.30 | **16000.000** | 16 kHz | 16000 |
| RoboticArm | acc | 7042.25 | **7063.294** | 6.7 kHz | 7063 |
| RoboticArm | gyro | 7042.25 | **7063.294** | 6.7 kHz | 7063 |

Rates are extremely stable within a machine — the span estimator's full range is 0.0002% (mic) to
0.02% (IMU) across all segments. No drift, no gaps (`n_gaps = 0` everywhere), no duplicate or
non-monotonic timestamps anywhere in 11,688 files.

**Why the two estimators differ:** timestamps are stored on a **1 µs grid**. A true mic interval of
62.488 µs cannot be represented, so every dt rounds to 62 or 63 µs and `1/median(dt)` snaps to
16129 Hz. Simulating a 16003.076 Hz stream rounded to 1 µs reproduces the observed 16129.03
exactly (`tests/test_timestamps.py`). The span estimator divides by a 5–7 s elapsed time and is
unaffected. **Use the span estimator; the per-sample estimators are artefacts.**

**Q1: the documented nominal rate and the timestamp-derived effective rate differ.**

The earlier phrasing "the published 6.7 kHz is wrong" is **withdrawn**. It conflated two distinct
quantities:

- **Nominal / documented ODR** — 16 kHz and 6.7 kHz, as stated by the paper and Zenodo. This is a
  configuration setting, and the audit has no evidence it is misreported.
- **Timestamp-derived effective rate** — what the released parquet timestamps actually imply.

Correct statement: *the released parquet timestamps imply machine-dependent effective sample rates
that differ from the nominal acquisition rates documented in the paper.*

Two consequences remain, unchanged:

1. **The two machines differ from each other**, which no primary source states.
2. The reference `preprocess.ipynb` hard-codes 7063 Hz for whichever machine is selected. That
   matches RoboticArm's effective rate but is **1.03% below BrushlessMotor's**. At 7135.7 Hz a
   100 ms window holds ~713.6 samples, but the reference truncates to `int(7063×0.1) = 706`,
   discarding a median of 5 samples (0.70%) from every BrushlessMotor IMU window. Using the nominal
   6.7 kHz would discard 41 samples (5.8%) per window.

**Diagnostic pattern and documented mechanism.** The microphone is accurate and session-invariant (RA
+0 ppm, BM +192 ppm; 0 ppm variation across RA's 101 sessions) while the IMU is offset by percent
and session-dependent (+5.94% / +7.03% from 6667 Hz; 19× more variation between RA sessions than
within one). Sample counts corroborate the timestamps: a 5.1999 s BM segment holds 37,106 IMU
samples where 6667 Hz would give 34,668. A plausible but **UNVERIFIED** explanation is that audio is
clocked from an accurate reference while the IMU ODR comes from the sensor's internal oscillator,
which carries a per-device tolerance. The official ISM330DHCX datasheet documents
`INTERNAL_FREQ_FINE`, an 8-bit two's-complement value that expresses the effective ODR/timestamp
rate offset in 0.15% steps. For selected ODR 6667 Hz, the documented relationship
`ODR_actual = 6667 × (1 + 0.0015 × INTERNAL_FREQ_FINE)` gives approximately +39.6 (~+40) for
RoboticArm and +46.9 (~+47) for BrushlessMotor, quantitatively reproducing the observations.

**The released timestamp-derived rates are quantitatively consistent with the ISM330DHCX
documented internal-frequency variation mechanism. Because the acquisition-time sensor-register
values are not present in the archived dataset, this explanation is strongly consistent with, but
not proven by, the released data.** Alternatives not excluded are a different ODR configuration or
timestamp reconstruction in the acquisition firmware. The nominal 6.7 kHz specification is not
called wrong.

**Additional finding — BrushlessMotor IMU is not uniformly sampled.** Its intervals are bimodal:
97.3% at ~140.6 µs and **2.7% at ~122.5 µs**, an 18 µs difference far beyond 1 µs rounding
(`frac_dt_irregular = 0.027`). This looks like FIFO batching. RoboticArm's IMU is uniform
(`frac_dt_irregular = 0.000`). Any spectral analysis of BrushlessMotor IMU data assumes a uniform
grid that does not hold. **Phase 2 must decide whether to resample; Phase 1 only measures it.**

---

## Q2 — Segment duration

`duration_summary.csv`

| Machine | mic duration | acc/gyro duration | std |
|---|---|---|---|
| BrushlessMotor | **5.198938 s** | 5.199911 s | 1.0 µs |
| RoboticArm | **7.499937 s** | 7.499930 s | 1.0 µs |

Durations are fixed to the microsecond and identical across train/test, source/target and
normal/anomaly — there is no duration cue that could leak the label.

**Q2: CONTRADICTED.** Phase 0 assumed 6–10 s inferred from DCASE clip lengths. BrushlessMotor is
**5.2 s**, outside that range; RoboticArm is 7.5 s. The two machines differ, which was not
anticipated. Windows per segment at 100 ms: ~51 (BM) and ~74 (RA), not the ~60–100 estimated.

---

## Q3 — Anomaly label vocabulary

`label_condition_counts.csv`

| Machine | Values | Test counts (source / target) |
|---|---|---|
| BrushlessMotor | `normal`, `belt`, `magnetic` | belt 39/42, magnetic 39/36, normal 78/78 |
| RoboticArm | `normal`, `loosescrewsA` | loosescrewsA 116/116, normal 116/116 |

**Q3: CONFIRMED and refined.** BrushlessMotor's two mechanisms match the paper (magnet, belt) and
are near-evenly split. RoboticArm has a **single** anomaly type. The `A` suffix in `loosescrewsA`
suggests a variant naming scheme with no second variant present — no evidence of another type.

Per-type sample sizes are small (36–42 per type per domain for BM), so per-type reporting will
carry wide intervals.

---

## Q4 — Condition/label balance (shortcut risk R6)

Operating conditions match paper Table 1 **exactly** for both machines:

- RoboticArm loads: source `weight00,10,15,20`, target `weight25,30,35`
- BrushlessMotor speeds: source `spd1500…3000` (9), target `spd1000…1400` (5)
- Backgrounds: source `BckgA,C,D,F`, target `BckgB,E,G` — both machines

Mutual information between condition and anomaly status, against a 2000-permutation null:

| Machine | Condition | MI (bits) | Null mean | Excess | p |
|---|---|---|---|---|---|
| BrushlessMotor | operational | 0.00121 | 0.03083 | **−0.0296** | 1.000 |
| BrushlessMotor | environmental | 0.00047 | 0.01427 | **−0.0138** | 1.000 |
| BrushlessMotor | joint | 0.00499 | 0.13459 | **−0.1296** | 1.000 |
| RoboticArm | operational | 0.00000 | 0.00942 | **−0.0094** | 1.000 |
| RoboticArm | environmental | 0.00000 | 0.00927 | **−0.0093** | 1.000 |
| RoboticArm | joint | 0.00000 | 0.03797 | **−0.0380** | 1.000 |

Every observed MI is **below** its permutation null. No condition value is class-exclusive
(`condition_class_balance.csv`).

**Q4: CONFIRMED — no shortcut.** Operating and environmental condition carry no information about
anomaly status. Risk R6 is empirically negligible for both machines.

Note the raw MI figures would have looked alarming without the null: BrushlessMotor's joint
condition has 51 values, and 51 categories over 312 samples generate 0.135 bits by chance alone.

---

## Q6 — Sensor synchronisation

`timestamp_alignment_summary.csv`

| Metric | BrushlessMotor | RoboticArm |
|---|---|---|
| start spread (mic→IMU) | 16.660 ms (range 16.652–16.671) | 0.036 ms (range 0–0.102) |
| end spread | 17.635 ms | 0.043 ms |
| intersection | 5.1823 s | 7.4999 s |
| coverage ratio | 0.99343 | 0.99999 |
| usable after 35 ms gyro cut | 5.1473 s | 7.4649 s |
| data discarded per sensor | ~1.01% | ~0.47% |

**Q6: CONFIRMED — genuinely co-clocked, not merely timestamp-aligned.** The BrushlessMotor offset
is deterministic to within ±10 µs across all 1,593 segments; a spread that tight cannot arise from
independent clocks and indicates a fixed firmware start latency on one acquisition time base.
RoboticArm's sensors start within 36 µs of each other.

**Gyroscope warm-up empirically justified.** In the first 35 ms the gyro standard deviation is
64000/71679/183009 mdps versus 5097/4345/3891 mdps afterwards — **12–47× larger**. The documented
cut is real and necessary.

**New finding — undocumented accelerometer startup artefact.** In 37 of 40 sampled BrushlessMotor
files the **first accelerometer sample is railed at ±15.988832 g** (the ±16 g full scale), and it
is typically the only railed row in the file. The reference pipeline removes it only incidentally,
because the 35 ms gyro crop happens to start later. Any pipeline that does not apply that crop
would ingest a full-scale spike as the first sample of every recording.

---

## Q7 — Is the released audio already noise-mixed?

The archives contain **only** parquet and CSV — no noise-source files, no impulse responses, no
alternative anechoic versions (verified: 11,688 parquet + 12 CSV + 0 other files).

Decisive test (`scripts/audio_noise_probe.py`): hold the operating condition fixed, vary only the
background label, and ask whether sensor statistics change. Run on **training-normal segments
only**, so no test signal informed it.

Total RMS is uninformative here **by construction** — every background is scaled to the same
−4 dB SNR, which equalises total power. Spectral *shape* is the discriminating statistic:

| | mic 20–200 Hz | mic 1–3 kHz | acc (3 bands) | gyro (3 bands) |
|---|---|---|---|---|
| BrushlessMotor | **9/9** conditions, p=2.2e−05 | **9/9**, p=5.1e−07 | 0–1/9, p≈0.4 | 0/9, p≈0.5 |
| RoboticArm | **4/7**, p=4.0e−06 | **6/7**, p=5.4e−07 | 0–1/7, p≈0.3 | 0/7, p≈0.2 |

**Q7: CONFIRMED (pre-mixed), and the audio-only claim CONFIRMED empirically.** The background label
systematically changes the microphone spectrum and leaves the accelerometer and gyroscope
statistically untouched. Phase 0 took "noise applies to audio only" from the paper's text; it is
now verified against the data.

Corroborating structural evidence: within a single RoboticArm parent recording, the **environmental
label varies between segments while the operational label cannot** (93 of 101 recordings carry
multiple `Bckg` values, 0 carry multiple `weight` values). Background is assigned in
post-processing to excerpts of one physical recording — exactly what digital mixing implies.

---

## Q8 — Parent-recording structure and train/test leakage — **THE CRITICAL FINDING**

`recording_times.csv`, `audit_findings.json → temporal_structure`

Segment ids embed the recording's wall-clock start, which makes the release's recording structure
recoverable. The two machines could hardly be more different.

### BrushlessMotor — clean splits, confounded time

- 1,593 segments from **1,593 distinct recordings** — exactly one segment each.
- **0 recordings span train and test.** No parent-recording leakage.
- Single session: 2024-04-23, 10:20:21 → 18:22:40 (8.04 h), recordings ~10 s apart.
- **But: all 1,437 normal segments were recorded 10:20:21–16:40:50 and all 156 anomalous segments
  16:48:07–18:22:40 — zero overlap, a 7.3-minute gap.** No normal was recorded after the first
  anomaly; no anomaly before the last normal.
- The two anomaly types are also temporally blocked: belt 16:48–17:33, magnetic 17:38–18:22.

**Consequence:** recording time predicts the class perfectly. Any session-drift effect — thermal,
mechanical settling, sensor bias, ambient change — is fully confounded with the fault. A detector
cannot be shown, on this machine alone, to be detecting a mechanical fault rather than "later in
the session". This is physically hard to avoid (inducing a fault requires stopping the machine),
but the complete absence of interleaved blocks means it cannot be separated after the fact.

### RoboticArm — interleaved time, leaky splits

- 2,303 segments from only **101 parent recordings** — median 16 segments per recording, max 48.
- 5 recording dates spanning 57 days (2024-02-21 → 2024-04-18).
- Normal and anomalous recordings **overlap** in time (1,211 normals after the first anomaly) — no
  temporal confound.
- **62 of 101 recordings contain segments on both sides of the train/test split, affecting 1,837 of
  2,303 segments (79.8%).**
- At the level that matters: **212 of 232 normal test segments (91.4%) have a sibling segment from
  the same parent recording in the training set.** By construction 0 of 232 anomalous test segments
  can, since training is normal-only.

Segments are **consecutive excerpts of one continuous stream**, confirmed from the timestamps:

| dataLog index | split | t_start (s) | t_end (s) |
|---|---|---|---|
| 7 | train | 72.679 | 80.179 |
| **8** | **test** | **82.707** | **90.207** |
| 9 | train | 92.738 | 100.238 |
| **10** | **test** | **102.770** | **110.270** |

Test segment 8 sits ~2.5 s from train segments 7 and 9 in the same recording, same machine, same
load. All 48 excerpts share one continuous clock (2.486 s → ~490 s).

**Consequence — a credible risk of optimistic bias, not a measured one.** Established: parent-
recording independence is violated, and the violation falls on the normal class only. *Not*
established: that AUC is inflated, or by how much. The plausible mechanism is that normal test
segments with near-twins in training reconstruct easily while anomalous segments have no such
advantage — but plausibility is not measurement. This is a **candidate** structural explanation for
the published RA-vs-BM gap (91.62 vs 58.95), competing with genuine machine-difficulty differences.
It will be tested by running one baseline under both splits and reporting the paired difference
(`docs/evaluation_protocol.md` §5); "no measurable difference" is an acceptable outcome.

**Q8: CONTRADICTED.** Phase 0 assumed no test segment shares an underlying recording with a
training segment. That holds for BrushlessMotor and fails badly for RoboticArm.

---

## Q9 — Size and compute

| | Compressed | Extracted | Files | Ratio |
|---|---|---|---|---|
| BrushlessMotor | 1.47 GiB | 3.83 GiB | 4,785 | 0.383 |
| RoboticArm | 3.30 GiB | 6.70 GiB | 6,915 | 0.493 |
| **Total** | **4.77 GiB** | **10.53 GiB** | **11,700** | |

Windows at 100 ms after cropping: ~51/segment (BM) and ~74/segment (RA) → roughly **81k + 170k ≈
251k windows**. As float32 at 1,600 mic + ~714×3 IMU values per window that is on the order of
3.7 GB — too large for comfortable in-memory use alongside models on typical student hardware.

Full audit of all 3,896 segments takes **~2 minutes** with 6 worker processes.

**Recommendation for Phase 2 (not implemented): cached, memory-mapped storage** (HDF5 or npy
memmap) written once and streamed. Not in-memory; not recomputed per epoch.

---

## Q10 — Channel semantics, units and ranges

**Units are encoded in the column names** — no guessing required.

| Sensor | Columns | Unit |
|---|---|---|
| mic | `Time`, `MIC [Waveform]` | **UNKNOWN** — "Waveform" is not a physical unit; scaling differs between machines (BM ±19.2, RA ±1.65), so values are not comparable across machines without normalisation |
| acc | `Time`, `A_x/A_y/A_z [g]` | **g** (standard gravity) |
| gyro | `Time`, `G_x/G_y/G_z [mdps]` | **millidegrees per second** |

Schema is identical across all 3,896 segments — one column set per sensor, `Time` always float64.

| Machine | acc range (g) | gyro range (mdps) | Notes |
|---|---|---|---|
| BrushlessMotor | −15.989 … +15.989 | −457,940 … +649,180 | hits the ±16 g rail (first sample only) |
| RoboticArm | −0.460 … +1.497 | −24,920 … +24,640 | A_z centred near +1 g — gravity on the vertical axis, as expected |

**File / schema integrity** across **11,688 files / 27,272 channel instances**: **0 NaN, 0 ±Inf,
0 constant channels, 0 duplicate timestamps, 0 non-monotonic timestamps, 0 gaps, 0 empty or corrupt
files, 0 missing files, 0 duplicate ids.** Integrity is pristine.

**Signal / acquisition characteristics** are a separate matter, and the release does contain
real artefacts: non-uniform BrushlessMotor IMU timing, the documented gyroscope warm-up, the
accelerometer first-sample rail, 1 µs timestamp quantisation, and unknown microphone scaling. A
dataset can have perfect file integrity and still carry acquisition artefacts; the two must not be
reported as one number. See the split summary below.

**Q10: RESOLVED for the IMU** (g and mdps), **UNKNOWN for the microphone** — recorded as UNKNOWN
rather than guessed.

---

## Consolidated Phase 0 verdicts

| Phase 0 assumption | Verdict |
|---|---|
| Q5 — 3,896 segments as published | **CONFIRMED** exactly, cell by cell |
| Q1 — IMU at 6.7 kHz / 7063 Hz | **CONTRADICTED** — 7063 Hz (RA) and 7135.7 Hz (BM); machines differ |
| Q2 — segments ~6–10 s | **CONTRADICTED** — 5.199 s (BM), 7.500 s (RA) |
| Q3 — named anomaly types, 2 for BM | **CONFIRMED** — BM {belt, magnetic}, RA {loosescrewsA} only |
| Q4 — conditions balanced across classes | **CONFIRMED** — MI below the permutation null for both |
| Q6 — sensors overlap, small loss from cropping | **CONFIRMED** — co-clocked; ~0.5–1.0% discarded |
| Q7 — audio pre-mixed, noise audio-only | **CONFIRMED** empirically for both machines |
| Q8 — no shared parent recording across splits | **CONTRADICTED for RoboticArm** (91.4% of normal test segments); confirmed for BrushlessMotor |
| Q10 — units unknown | **PARTIALLY CONFIRMED** — IMU units encoded; mic unit genuinely unknown |
| Data-quality risk | **CONFIRMED CLEAN** — zero defects in 3,896 segments |

---

## Phase 1.5 — full-dataset follow-up on two artefacts

Reproduce with `python scripts/investigate_artifacts.py --data-root data/raw/extracted`
(outputs `accelerometer_artefact_scan.csv`, `artefact_investigation.json`).

### Accelerometer first-sample rail — scanned across all 3,896 segments

Phase 1 saw this in 37 of 40 sampled files and could not say how general it was.

| | BrushlessMotor | RoboticArm |
|---|---|---|
| Segments scanned | 1,593 | 2,303 |
| First row railed at ±15.988832 g | **1,430 (89.8%)** | **0 (0.0%)** |
| Any railed row anywhere | 1,430 (89.8%) | 0 |
| Railed row **only** at index 0 | 1,430 (100% of affected) | — |
| Highest railed row index in the entire release | **0** | none |

Axes railed in the first row vary: 1 axis in 597 files, 2 in 606, 3 in 227, 0 in 163.

**Association with labels — the question that decides whether a cleaning rule is legitimate:**

| Field | χ² p | Rail rate by value |
|---|---|---|
| train/test | 1.00 | 0.897 / 0.898 |
| source/target | 1.00 | 0.898 / 0.897 |
| anomaly label | 0.43 | normal 0.900, belt 0.901, magnetic 0.853 |
| operating condition | 0.38 | 0.829–0.943 across 14 speeds |
| background | 0.86 | 0.860–0.932 across 7 backgrounds |

**Conclusion.** The artefact is confined to sample index 0, is absent from RoboticArm entirely, and
is **independent of every label**. A rule of the form *"discard accelerometer sample index 0"*
therefore depends only on sample position — available at inference time — and on no label, so it is
deployment-safe and carries no leakage. It is **proposed, not frozen** (see
`docs/evaluation_protocol.md` §13). Note Protocol A already removes these samples incidentally via
the 35 ms gyroscope crop, so it changes nothing there.

### The `_corr` marker — investigated, still unresolved

85 of 2,303 RoboticArm segment ids carry a `corr` token (77 as `corr<N>`, 8 as `corr_<N>`).

- **Not documented anywhere.** Absent from the DCASE paper, the Zenodo description, and every file
  in the authors' released repository (`README.md`, `preprocess.ipynb`, `baseline.ipynb`,
  `model.py`, `utilities.py`) — searched directly.
- **It is a whole-recording property, not a per-segment one.** The 85 segments come from exactly
  **6 of 101 parent recordings**, and **no recording mixes marked and unmarked segments**.
- The 6 marked recordings: `20240405_10_54_03`, `20240405_11_02_26`, `20240405_11_27_37`,
  `20240405_11_58_13`, `20240405_13_01_28`, `20240416_13_35_37` — 3 source (weight15) and 3 target
  (weight25/30), all **normal**, all from 5 and 16 April 2024.

**Association testing, and a correction to the naive analysis.** A segment-level χ² suggested strong
associations (operating condition p < 1e-4, class p = 0.003). **Those p-values are invalid**:
segments are clustered within recordings, so 85 clustered segments were being treated as 85
independent observations. Repeating the test at the **recording** level, where the marker actually
lives, gives Fisher's exact **p = 0.59** for class — **not significant**. With 6 marked recordings
there is no power to detect much of anything.

The one real difference: marked segments have marginally fewer accelerometer samples (median 52,972
vs 52,976) and a marginally lower IMU rate (7062.88 vs 7063.34 Hz, Mann–Whitney p < 1e-4), while the
microphone rate is identical (p = 0.63). That is a 0.007% difference and is most economically
explained by these being different acquisition sessions — which independently corroborates the
session-dependent-IMU-clock hypothesis in Q1 rather than indicating any processing difference.

**Status: UNRESOLVED.** The marker is retained as an explicit metadata flag
(`has_corr_marker` in `src/data/temporal.py`), carried through the manifest, and **not** used to
include, exclude or weight any segment. If its meaning is later established, the flag is already
there to act on. Three of the six marked recordings straddle the official train/test split, so they
are handled by the parent-recording grouping like any other recording.

## New findings not anticipated in Phase 0

1. **The two machines differ in sampling rate, duration, alignment offset, sampling regularity,
   recording structure and signal scaling.** No primary source mentions any of this. **RoboticArm
   and BrushlessMotor should be treated as two separate experimental cohorts within the same
   dataset**, because their acquisition and anomaly characteristics differ substantially.
2. **RoboticArm's official split is not recording-disjoint** (Q8) — the single most consequential
   finding of Phase 1.
3. **BrushlessMotor's classes are perfectly separated in recording time** (Q8).
4. **Timestamps are quantised to 1 µs**, making per-sample rate estimators unusable.
5. **BrushlessMotor's IMU has a 2.7% bimodal sampling irregularity.**
6. **Undocumented accelerometer first-sample rail artefact.**
7. **85 RoboticArm segment ids carry an undocumented `corr` marker** (`_corr103`, `_corr_377`, …).
   Phase 1.5 established it is a whole-recording property covering 6 of 101 recordings, absent from
   all primary sources, and not significantly associated with class at recording level (Fisher
   p = 0.59). Meaning remains **unresolved**; the flag is retained, not discarded.
