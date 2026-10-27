# Assumptions and Open Questions — Phase 0

Two lists, deliberately separated.

**Part A** is what MechaSight currently *believes but has not confirmed against the actual data*.
Every item here is a candidate for being wrong. None of it may be stated as fact in the report,
the synopsis, or the presentation until Phase 1 confirms it.

**Part B** is what MechaSight has *decided* without external evidence — judgement calls that a
reviewer may reasonably disagree with.

Verified facts live in `docs/research_definition.md` §5 and are traced in
`docs/source_verification.md`. If something appears in both places, the verified version wins.

---

# Part A — Assumptions requiring real-dataset confirmation

**STATUS: Phase 1 complete (2026-08-21).** Every question below now carries a verdict from the real
release. Original Phase 0 text is preserved verbatim; verdicts are appended. Nothing contradicted
has been deleted — see `reports/dataset_audit/dataset_summary.md` for the full evidence and
`reports/dataset_audit/*.csv` for the machine-readable outputs.

| | Verdict |
|---|---|
| Q1 effective IMU rate | **CONTRADICTED** |
| Q2 segment duration | **CONTRADICTED** |
| Q3 anomaly_label vocabulary | **CONFIRMED** |
| Q4 condition/label balance | **CONFIRMED** (no shortcut) |
| Q5 segment counts | **CONFIRMED** exactly |
| Q6 sensor alignment | **CONFIRMED** |
| Q7 audio pre-mixed / audio-only noise | **CONFIRMED** |
| Q8 no shared parent recording across splits | **CONTRADICTED for RoboticArm** |
| Q9 size and compute | **RESOLVED** |
| Q10 channel units | **PARTIALLY CONFIRMED** (mic still UNKNOWN) |

Each is a Phase 1 audit task. "Blocking" means Phase 2 preprocessing cannot be designed
responsibly without the answer.

### Q1 — Effective IMU sampling rate — **BLOCKING**

**Assumption:** accelerometer and gyroscope sample at ~7063 Hz effective (nominal 6.7 kHz).
**Basis:** the authors' `preprocess.ipynb` hard-codes 7063 with the comment that it was estimated
by averaging time deltas across all files; the paper and Zenodo both say 6.7 kHz.
**Why it matters:** it sets window length in samples (706 vs 670 for a 100 ms window), and it sets
the frequency axis of every spectral feature. A ~5% frequency-axis error would quietly shift every
harmonic and order-analysis result.
**Audit:** compute the empirical rate from the `Time` column per file; report the distribution
across files, per machine and per split; check for drift within a file and for dropped samples.
**Failure mode to watch for:** a *variable* rate, which would break the fixed-length window
assumption entirely and force resampling.

> **VERDICT: PARTIALLY CONFIRMED — corrected in Phase 1.5.** Measured span-based rates: RoboticArm
> mic 16000.000 Hz, IMU **7063.294 Hz**; BrushlessMotor mic 16003.077 Hz, IMU **7135.716 /
> 7135.331 Hz**. The Phase 1 wording "the published 6.7 kHz is wrong" is **withdrawn**: it conflated
> the *documented nominal ODR* with the *timestamp-derived effective rate*. Correct statement — the
> released parquet timestamps imply machine-dependent effective sample rates that differ from the
> nominal acquisition rates documented in the paper. See `research_definition.md` §5.12 for the
> diagnostic evidence (audio ppm-accurate and session-invariant; IMU percent-offset and
> session-dependent). The official ISM330DHCX datasheet documents `INTERNAL_FREQ_FINE`, an 8-bit
> two's-complement correction in 0.15% steps. At selected ODR 6667 Hz,
> `ODR_actual = 6667 × (1 + 0.0015 × INTERNAL_FREQ_FINE)`; the observed effective rates imply
> approximately +40 (RoboticArm) and +47 (BrushlessMotor). **The released timestamp-derived rates
> are quantitatively consistent with the ISM330DHCX documented internal-frequency variation
> mechanism. Because the acquisition-time sensor-register values are not present in the archived
> dataset, this explanation is strongly consistent with, but not proven by, the released data.**
> The 7063 Hz in the reference
> code matches RoboticArm's effective rate, yet that code applies it to both machines — a 1.03% error on BrushlessMotor that silently
> truncates ~5 samples (0.70%) from every 100 ms IMU window. Rates are stable within a machine
> (range ≤0.02%); no drift, no gaps. Two further findings: timestamps are quantised to **1 µs**,
> which makes `1/median(dt)` an artefact (use the span estimator), and BrushlessMotor's IMU is
> **bimodal** — 2.7% of intervals at ~122.5 µs against 97.3% at ~140.6 µs — so it is not uniformly
> sampled. RoboticArm's IMU is uniform.

### Q2 — Segment duration and windows per segment — **BLOCKING**

**Assumption:** a segment is on the order of 6–10 s, giving roughly 60–100 windows of 100 ms.
**Basis:** inference only — DCASE 2024 Task 2 clips for these machines are 6–10 s and were
generated with the same simulation setup. **No primary source states the IMAD-DS segment length.**
**Why it matters:** it determines total window count (and hence training time and memory), the
statistical weight of a segment-level median, and whether 100 ms is a sensible window at all.
**Audit:** measure duration per segment; report distribution per machine, split and domain; check
whether all segments are equal length; count windows per segment after the 35 ms gyro warm-up cut
and the sensor-intersection cut.

> **VERDICT: CONTRADICTED (Phase 1).** Durations are fixed to the microsecond but differ by
> machine: **BrushlessMotor 5.198938 s**, **RoboticArm 7.499937 s** (std 1.0 µs). The 6–10 s
> estimate inferred from DCASE clip lengths is wrong for BrushlessMotor and coincidental for
> RoboticArm. Windows per segment at 100 ms after cropping: ~51 (BM), ~74 (RA), not ~60–100.
> Duration is identical across split, domain and class, so it leaks nothing.

### Q3 — `anomaly_label` vocabulary and per-type counts — **BLOCKING for Objective 3**

**Assumption:** `anomaly_label` holds `normal` plus named fault types, with the Brushless Motor
having two (magnet displacement, belt tightening).
**Basis:** the paper describes two BM anomaly mechanisms; the baseline code tests
`anomaly_label != 'normal'`, implying named values.
**Why it matters:** per-anomaly-type reporting is an Objective 3 deliverable. If the 78 BM
anomalous test segments split unevenly between the two mechanisms, per-type AUCs will have very
wide CIs and must be reported as such.
**Audit:** enumerate unique values and counts, per machine, split and domain.

> **VERDICT: CONFIRMED (Phase 1).** BrushlessMotor: `normal`, `belt`, `magnetic` — test counts
> belt 39/42 and magnetic 39/36 (source/target). RoboticArm: `normal`, `loosescrewsA` — a
> **single** anomaly type, 116 per domain. Per-type sample sizes of 36–42 mean per-type reporting
> will carry wide intervals.

### Q4 — Condition balance across normal and anomalous test segments — **BLOCKING for R6**

**Assumption:** operating conditions are distributed similarly among normal and anomalous test
segments, so condition metadata does not predict the label.
**Basis:** none. Purely assumed. The balanced 116/116 and 78/78 counts say nothing about the
*conditions* within each group.
**Why it matters:** if certain loads or speeds appear only among anomalous test segments, any
condition-aware model has a shortcut. This is one of the reasons the tree-based condition-classifier
recast is deferred rather than adopted (research definition §11, J5).
**Audit:** cross-tabulate `anomaly_label × domain_shift_op × domain_shift_env` for the test set;
report mutual information between condition and label. This is a required Phase 1 output.

> **VERDICT: CONFIRMED — no shortcut (Phase 1).** Mutual information sits *below* its
> 2000-permutation null for every condition on both machines (p = 1.000 throughout; RoboticArm's
> MI is exactly 0.000). No condition value is class-exclusive. Operating and environmental
> conditions also match paper Table 1 exactly. **Risk R6 is empirically negligible.** Note the raw
> MI would have misled without the null: BrushlessMotor's 51 joint condition values generate
> 0.135 bits by chance over 312 samples.

### Q5 — Actual segment counts match Table 2 — **BLOCKING for reproducibility**

**Assumption:** the release contains 3896 segments distributed exactly as the paper's Table 2.
**Basis:** the paper's table. The dataset has not been downloaded.
**Why it matters:** if counts differ, every derived number changes, and the reference baseline's
`try/except` around segment reading means some segments could be unreadable in practice.
**Audit:** count rows in all six `attributes_*.csv` files, count actual `.parquet` files, and
verify each metadata row's three sensor files exist and are readable. Report any mismatch against
Table 2 rather than adopting the paper's numbers.

> **VERDICT: CONFIRMED exactly (Phase 1).** Published = metadata rows = readable triplets = **3896**,
> matching Table 2 cell by cell. 11,688 parquet + 12 CSV, zero other files, zero orphans, zero
> unreadable files, zero duplicate segment ids, zero label/filename contradictions.

### Q6 — Sensor timestamp overlap and data loss from alignment — **BLOCKING**

**Assumption:** the three sensing modalities' recordings (from two physical sensor devices)
overlap substantially within a segment, so the
intersection interval (plus the 35 ms gyro warm-up cut) loses only a small fraction of each segment.
**Basis:** implied by the baseline working at all; not quantified anywhere.
**Why it matters:** if start-time offsets are large, usable duration per segment shrinks and
window counts drop. Also determines whether the sensors are truly hardware-synchronised or merely
timestamped on a shared clock.
**Audit:** per segment, report each sensor's start/end timestamps, the intersection length, and
the fraction of each sensor's data discarded.

> **VERDICT: CONFIRMED — genuinely co-clocked (Phase 1).** BrushlessMotor's mic-to-IMU start offset
> is 16.660 ms with a range of only ±10 µs across all 1,593 segments — far too tight for
> independent clocks. RoboticArm's sensors start within 0.036 ms. Coverage ratios 0.9934 (BM) and
> 0.99999 (RA); the intersection plus 35 ms gyro cut discards ~1.0% (BM) and ~0.5% (RA) per sensor.
> The gyro warm-up is empirically justified: standard deviation in the first 35 ms is **12–47×**
> the post-warm-up value. **New:** an undocumented accelerometer artefact — the first sample is
> railed at ±15.988832 g (±16 g full scale) in 37 of 40 sampled BrushlessMotor files.

### Q7 — Are released microphone signals already noise-mixed? — **BLOCKING**

**Assumption:** the released audio is the final noise-mixed version (anechoic machine sound
convolved with its RIR, plus scaled reverberant factory noise at −4 dB).
**Basis:** the paper describes the mixing pipeline as producing "the final audio sample used in
the dataset", which implies pre-mixed, but does not state what the Zenodo release contains.
**Why it matters:** if the release were anechoic with mixing left to the user, then the
environmental domain shift would not exist in the files at all and would have to be reproduced —
which would change Phase 2 entirely, and would also make an SNR sweep trivially available.
**Audit:** inspect audio for reverberation and noise floor; check for any additional files or
noise-source material in the archives; compare a source-noise segment against a target-noise
segment of the same operating condition.

> **VERDICT: CONFIRMED — pre-mixed, and audio-only (Phase 1).** The archives contain only parquet
> and CSV: no noise sources, no impulse responses, no anechoic alternatives. Holding the operating
> condition fixed and varying only the background label changes the **microphone spectrum**
> significantly (BM 9/9 conditions, p = 5e-07; RA 6/7, p = 5e-07) while leaving the accelerometer
> and gyroscope statistically untouched (0–1 of 7–9 conditions, p ≈ 0.2–0.5). Total RMS is
> uninformative here by construction, since every background is scaled to the same −4 dB SNR.
> Corroborating: within one RoboticArm recording the environmental label varies across segments
> (93 of 101 recordings) while the operational label never does — background is assigned in
> post-processing. Phase 0 took "audio-only" from the paper's text; it is now verified from data.

### Q8 — Train/test overlap at the level of underlying recordings — **NON-BLOCKING, important**

**Assumption:** no test segment shares an underlying continuous recording with a training segment.
**Basis:** assumed from the authors' split design; unverified.
**Why it matters:** risk R9. Would inflate all reported results, including the published ones.
**Audit:** parse `segment_id` for a parent-recording component; look for contiguous timestamp
ranges spanning the train/test boundary.

> **VERDICT: CONTRADICTED for RoboticArm — the most consequential Phase 1 finding.**
> RoboticArm's 2,303 segments come from only **101 parent recordings** (median 16 per recording,
> max 48), and **62 of those 101 span the train/test split, affecting 1,837 segments (79.8%)**.
> At the level that matters: **212 of 232 normal test segments (91.4%) have a sibling from the same
> parent recording in training**, while 0 of 232 anomalous ones can, since training is normal-only.
> Segments are consecutive excerpts of one continuous stream — test `dataLog_8` (82.7–90.2 s) sits
> ~2.5 s from train `dataLog_7` and `dataLog_9` in the same recording at the same load. The
> violation is asymmetric and falls on the normal class only, creating a **credible risk of
> optimistic bias whose magnitude has not been measured** (Phase 1.5 correction: the earlier
> "asymmetrically inflates AUC" overstated an untested hypothesis). It is a *candidate* explanation
> for the published RA-vs-BM gap (91.62 vs 58.95), and will be tested by running one baseline under
> both splits — `docs/evaluation_protocol.md` §5.
>
> **BrushlessMotor is clean** on this axis — 1,593 segments from 1,593 recordings, zero straddling.
> But it carries the opposite defect: **all normal segments were recorded before all anomalous
> ones**, with a 7.3-minute gap and zero overlap, so recording time predicts the class perfectly
> (see the new Q11 below).

### Q9 — File size and compute feasibility — **NON-BLOCKING, planning**

**Assumption:** ~5 GB compressed expands to a workable size, and the resulting window tensors fit
in available memory or can be streamed from HDF5.
**Basis:** Zenodo file sizes are verified (3.30 + 1.47 GiB); the expansion ratio for parquet in a
7z archive is not known, and total window count depends on Q2.
**Why it matters:** determines whether Phase 2 needs streaming/caching rather than in-memory
arrays, and constrains seed counts in Phase 3+.
**Audit:** measure extracted size; estimate window-tensor size from Q1/Q2; decide storage strategy.

> **VERDICT: RESOLVED (Phase 1).** 4.77 GiB compressed becomes **10.53 GiB extracted**, 11,700
> files. About 251k windows at 100 ms (~81k BM + ~170k RA), on the order of 3.7 GB as float32 — too
> large for comfortable in-memory use alongside models. Full audit of all 3,896 segments runs in ~2
> minutes with 6 workers. **Recommendation carried to Phase 2 review (not implemented): cached,
> memory-mapped storage, written once and streamed.**

### Q10 — Channel semantics and units — **NON-BLOCKING, needed for physical interpretation**

**Assumption:** accelerometer channels are X/Y/Z in g or m/s², gyroscope in dps or rad/s,
microphone in normalised amplitude or raw ADC counts.
**Basis:** none for the released files. The ISM330DHCX and IMP23ABSU datasheets give ranges, but
what the release stores is unverified.
**Why it matters:** any physically-interpretable feature (RMS acceleration, shaft order, angular
velocity) needs correct units. Getting this wrong would not hurt AUC but would make the
mechanical-engineering interpretation wrong — which is much of this project's academic point.
**Audit:** inspect column names, value ranges and plausibility against the datasheet full-scale
settings; report units as verified or as unknown.

> **VERDICT: PARTIALLY CONFIRMED (Phase 1).** Units are encoded in the column names:
> accelerometer `A_x/A_y/A_z [g]` in **standard gravity**, gyroscope `G_x/G_y/G_z [mdps]` in
> **millidegrees per second**. RoboticArm's A_z is centred near +1 g, consistent with gravity on
> the vertical axis. The microphone column is `MIC [Waveform]` — **unit UNKNOWN**, and its scale
> differs between machines (BM ±19.2, RA ±1.65), so microphone values are not comparable across
> machines without normalisation. Recorded as UNKNOWN rather than guessed.
> Data quality across 27,272 channel instances: 0 NaN, 0 Inf, 0 constant channels.

### Q11 — Class/recording-time confound (BrushlessMotor) — **NEW, found in Phase 1**

**Finding:** all 1,437 BrushlessMotor normal segments were recorded 10:20:21–16:40:50 and all 156
anomalous segments 16:48:07–18:22:40 on 2024-04-23 — **zero temporal overlap, 7.3-minute gap**. The
two anomaly types are likewise blocked (belt 16:48–17:33, magnetic 17:38–18:22).
**Why it matters:** recording time predicts the class perfectly, so any session drift — thermal,
mechanical settling, sensor bias, ambient change — is fully confounded with the fault. On this
machine alone we cannot demonstrate that a detector responds to the mechanical fault rather than to
"later in the session". Physically hard for the authors to avoid, but the complete absence of
interleaving means it cannot be separated after the fact.
**Status after Phase 1.5: NOT SOLVABLE — structural limitation.** Fault and acquisition-session
effects are **not identifiable** for this cohort. An early-vs-late normal experiment is a
temporal-drift **diagnostic**, not a deconfounding: it bounds drift inside the normal training data
but cannot establish that the later anomaly session is free of independent session effects, because
no healthy recordings exist within that session. Required reporting wording is frozen in
`docs/evaluation_protocol.md` §6.

### Q12 — Undocumented `corr` marker in RoboticArm ids — **NEW, found in Phase 1**

85 of 2,303 RoboticArm segment ids carry a `corr` token (`_corr103`, `_corr300`, `_corr_377`, ...):
77 in the form `corr<N>` and 8 as `corr_<N>`.

> **Phase 1.5 investigation — still UNRESOLVED, but much better characterised.** Absent from the
> paper, the Zenodo description and every file in the authors' released repository (searched
> directly). It is a **whole-recording property**: the 85 segments come from exactly **6 of 101
> parent recordings**, and no recording mixes marked with unmarked segments. All 6 are `normal`,
> 3 source and 3 target, recorded 5 and 16 April 2024. A segment-level χ² appeared to show strong
> associations, but that test is **invalid** because segments are clustered within recordings;
> at recording level Fisher's exact gives **p = 0.59** for class — not significant, and with 6
> recordings there is little power. The only real difference is a 0.007% lower IMU sample rate,
> which corroborates the session-dependent IMU clock hypothesis (Q1) rather than indicating
> processing.
>
> **Handling:** retained as an explicit `has_corr_marker` flag through the manifest, used to
> include, exclude or weight **nothing**. Not silently discarded.

---

# Part B — Judgement calls made without external evidence

A reviewer may reasonably disagree with any of these. They are decisions, not findings.

### J1 — Segment-level evaluation as the only headline metric
Follows the dataset's own protocol and keeps comparability with published numbers. Cost: fewer
effective samples, wider CIs. Alternative rejected: window-level headline metrics (R7).

### J2 — pAUC@0.1 retained as a secondary threshold-free metric
Phase 2.1 freezes AUROC and AUPRC as primary. pAUROC remains a secondary low-FPR diagnostic imported
from DCASE Task 2 and is always distinguished from the authors' AUROC-only Table 4.

### J3 — Threshold at the 90th percentile of source-validation normal segment scores
The value 90 is a convention, not a derivation. It is fixed in advance specifically so it cannot
be tuned later (R4). May be revised once, before any test result is inspected, with the change
recorded.

### J4 — Adding an embedding + k-NN/Mahalanobis comparator not in the synopsis
Judgement: a comparison in which every deep entry is scored by reconstruction error would be
narrow. **The supporting claim is not yet sourced** — see the evidence note in
`research_definition.md` §11. Primary sources must be read and logged before Phase 3
implementation, and if the justification does not hold up, the comparator is dropped rather than
carried on an unsourced assertion.

### J5 — XGBoost/LightGBM deferred rather than recast — *revised*
The first draft of this document recommended re-casting them as self-supervised condition
classifiers. That was premature: the recast needs a condition target for test segments, which
either assumes operating condition is known at inference time (a deployment assumption this
project has not argued for) or requires a different, label-free score that would need separate
justification. It also needs the condition vocabulary, its availability and its balance across
normal/anomalous test segments — all unknown (Q3, Q4). **Decision deferred to Phase 1.** If the
audit does not support it, the family is dropped. "The synopsis listed a tree model" is explicitly
rejected as a reason to keep one.

### J6 — Deferring TCN, and requiring CNN-AE to justify temporal modelling first
Judgement about effort allocation, not about TCN's merit.

### J7 — Both source-only and source+target training variants retained
Doubles the experimental grid. Justification: source-only is the stricter and more honest test of
robustness, while source+target is the dataset's own setting and the one comparable to published
numbers. Dropping either would weaken a different part of the argument.

### J8 — Corruption types chosen for mechanical plausibility
Bias/offset drift, gain error, clipping, dropout, stuck-at values were chosen because they are how
MEMS sensors on machinery actually fail, not because of any benchmark convention. Severity levels
are not yet chosen and must be committed before the runs (R10).

### J9 — Treating the paper's Table 1 caption inconsistency as a typo
"Letters A to F index 7 background noise recordings" is internally inconsistent; the table body
(A,C,D,F | B,E,G) was taken as authoritative. Low risk, but it is an interpretation.

### J10 — Keeping the reliability-aware fusion hypothesis at all
Given that the accelerometer subvector carries most of the separating signal in the published
baseline, one could argue the honest project is simply "use the accelerometer". Note this argument
rests on masked-input probes of a joint autoencoder, not on trained single-modality models, so it
is weaker than it first appears. The hypothesis is retained because the
IMAD-DS authors' own Brushless Motor result (all-sensor fusion 58.95 < accelerometer 69.30) is
direct evidence that naive fusion misallocates trust, and because "which sensor to trust" is not
knowable in advance in deployment. If Phase 3 baselines show accelerometer-only dominating every
fusion variant on both machines, that finding is itself the answer and the project reports it.

---

## How this file is maintained

- When an assumption is confirmed, it moves to `research_definition.md` §5 with a `[VERIFIED]`
  tag and a pointer to the audit output that confirmed it.
- When an assumption is **contradicted**, it stays here, marked `[CONTRADICTED]`, with the
  evidence and every downstream document it forces a change to. Contradicted assumptions are
  never deleted — the record of having been wrong is part of the audit trail.
- New assumptions discovered mid-phase are appended, never silently absorbed into the definition.
