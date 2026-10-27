# Single-Modality Model Benchmarking — Result Report

<!-- Legacy roadmap name: Phase 4 -->

**Status:** protected evaluation COMPLETE and FINAL. One 156-look batch; score-all then
analyse-once; no metric existed before all 156 tables verified.
**Design freeze:** `docs/phase4_single_modality_freeze.md` · **Config:**
`configs/phase4/single_modality_benchmark_v1.json` · **Artifacts:** `reports/phase4_protected/`.

## 1. Protected execution record

| Item | Value |
|---|---|
| Branch | `phase4-single-modality-benchmark` |
| Fit HEAD (all 156 bundles) | `85df48d819c25679b596286adda9df21720c93e9` |
| Scoring HEAD (bound by `batch_authorization.json`) | `0021c93135a4d31800703520d5026bbed57f9b2e` (source identical to `61aa1f8`; the commit adds only the pre-scoring identity record) |
| Source snapshot hash | `7296665253064b97b8b67c3b88714a29dc3b3df79d0b96a439f5eefc909fbc49` |
| Proposed batch id (fit HEAD) | `p4-test-batch-f3fa02a57fc7adc6650e7522` |
| Authorised batch id (scoring HEAD; identical 156 experiment ids and cohort hashes) | `p4-test-batch-0884109383d3695de7a07121` |
| Ledger | **40 → 196**, one atomic append |
| Experiments scored | **156 / 156**; completion gate GREEN (`completion_gate.json`, 14 checks) |
| Score-manifest SHA-256 | `d0874952a5685ea8eb474b6fa0095e8bf1bf9d6ace4e71e58ef5477633ca5504` |
| Analysis-manifest SHA-256 | `a2287f7dccfe7579ea2a3e10dd443a07a0ce344b48347c1b74de95e523f847d0` (per-table hashes inside) |
| Failures / recoveries | None during protected inference. One precondition refusal *before* any protected action: the first `score-all` invocation refused because its own log redirect created an untracked file (clean-tree check); ledger unchanged at 40, no cache, no score. Re-invoked with the log outside the tree. Not a failure-policy event. |

Test cohorts: RoboticArm 466 segments (234 normal / 232 anomalous; target slice 20 normal / 116
anomalous, prevalence 0.853, 6 healthy-target parents); BrushlessMotor 312 (156 / 156; target 78 /
78, prevalence 0.50). Bootstrap: class-stratified parent-cluster, seed 1995, 10 000 replicates,
seed-averaged per replicate (R8).

## 2. Headline tables (complete common test, seed-averaged point, 95 % CI)

AUROC by machine × regime × modality × family. `*` marks the modality with the highest
AUROC within that family; healthy-target FPR (hFPR) is at the frozen q90.

### RoboticArm (limitations: six-parent validation; 20 healthy target segments from 6 parents)

| Regime | Family | mic AUROC | acc AUROC | gyro AUROC | hFPR mic / acc / gyro |
|---|---|---|---|---|---|
| S | statistical_reference | 0.592 [0.458, 0.695] | **0.909** [0.751, 0.980] | 0.701 [0.480, 0.862] | 0.55 / 0.35 / 0.00† |
| S | isolation_forest | 0.612 [0.482, 0.719] | **0.952** [0.847, 0.991] | 0.765 [0.572, 0.903] | 0.35 / 0.05 / 0.00† |
| S | one_class_svm | 0.659 [0.509, 0.769] | **0.959** [0.859, 0.994] | 0.718 [0.589, 0.822] | 0.60 / 0.55 / 0.15 |
| S | mahalanobis_shrinkage | 0.691 [0.566, 0.778] | **0.961** [0.860, 0.997] | 0.850 [0.750, 0.915] | 0.60 / 0.55 / 0.10 |
| S | dense_autoencoder | 0.670 [0.479, 0.811] | **0.807** [0.707, 0.880] | 0.577 [0.487, 0.644] | 0.92 / 0.23 / 0.07 |
| T | statistical_reference | 0.590 [0.457, 0.692] | **0.904** [0.741, 0.978] | 0.696 [0.477, 0.857] | 0.50 / 0.30 / 0.00† |
| T | isolation_forest | 0.617 [0.484, 0.723] | **0.946** [0.831, 0.989] | 0.756 [0.560, 0.896] | 0.40 / 0.33 / 0.00† |
| T | one_class_svm | 0.638 [0.510, 0.735] | **0.980** [0.930, 0.996] | 0.696 [0.562, 0.806] | 0.60 / 0.40 / 0.10 |
| T | mahalanobis_shrinkage | 0.672 [0.556, 0.755] | **0.969** [0.894, 0.995] | 0.849 [0.754, 0.914] | 0.55 / 0.50 / 0.00† |
| T | dense_autoencoder | 0.545 [0.458, 0.615] | **0.799** [0.705, 0.873] | 0.578 [0.492, 0.642] | 0.28 / 0.16 / 0.04 |

† produced no positive predictions on the complete cohort in every seed (threshold-transfer
artefact: zero false alarms coexist with zero detections; recall 0.000) — except IF-gyro-T, where
four of five seeds raised no alarm and one seed detected a single anomaly (1 TP, 0 FP; seed-mean
recall 0.001). Same mechanism as Phase 3.2.1 §5.

### BrushlessMotor (limitation `brushless_motor_acquisition_time_class_confound_v1`: benchmark discrimination only, never fault sensitivity)

| Regime | Family | mic AUROC | acc AUROC | gyro AUROC | hFPR mic / acc / gyro |
|---|---|---|---|---|---|
| S | statistical_reference | 0.486 [0.420, 0.550] | 0.572 [0.505, 0.635] | **0.684** [0.622, 0.743] | 0.37 / 0.71 / 0.88 |
| S | isolation_forest | 0.465 [0.400, 0.529] | 0.645 [0.581, 0.705] | **0.729** [0.671, 0.782] | 0.81 / 1.00 / 0.92 |
| S | one_class_svm | 0.493 [0.428, 0.558] | **0.732** [0.674, 0.787] | 0.721 [0.662, 0.776] | 1.00 / 1.00 / 0.99 |
| S | mahalanobis_shrinkage | 0.474 [0.410, 0.538] | **0.835** [0.789, 0.878] | 0.694 [0.633, 0.751] | 0.47 / 0.81 / 1.00 |
| S | dense_autoencoder | 0.664 [0.603, 0.724] | **0.882** [0.841, 0.919] | 0.780 [0.728, 0.828] | 0.70 / 0.72 / 0.71 |
| T | statistical_reference | 0.486 [0.421, 0.550] | 0.563 [0.496, 0.627] | **0.680** [0.619, 0.739] | 0.35 / 0.60 / 0.74 |
| T | isolation_forest | 0.462 [0.398, 0.526] | 0.570 [0.501, 0.637] | **0.633** [0.568, 0.695] | 0.76 / 0.99 / 0.87 |
| T | one_class_svm | 0.513 [0.447, 0.576] | **0.908** [0.874, 0.940] | 0.785 [0.731, 0.834] | 0.71 / 0.83 / 0.76 |
| T | mahalanobis_shrinkage | 0.477 [0.413, 0.542] | **0.896** [0.861, 0.927] | 0.750 [0.694, 0.801] | 0.27 / 0.72 / 1.00 |
| T | dense_autoencoder | 0.820 [0.774, 0.863] | **0.982** [0.971, 0.991] | 0.933 [0.907, 0.957] | 0.07 / 0.09 / 0.02 |

Full AUPRC (with prevalence context), pAUROC@0.10, source/target slices, per-seed values and
per-experiment JSON are in `reports/phase4_protected/analysis/`.

## 3. AUPRC and pAUROC

AUPRC orders the modalities identically to AUROC in every (machine, regime, family) cell.
pAUROC@0.10 agrees except BrushlessMotor `statistical_reference` (AUROC/AUPRC: gyro; pAUROC:
acc, by 0.559 vs 0.512 — both close to chance). Low-FPR performance is weak nearly everywhere:
the only pAUROC values above 0.80 are RoboticArm accelerometer with OCSVM-T (0.902) and
Mahalanobis-T (0.843), and BrushlessMotor dense-AE-T accelerometer (0.942) and gyroscope
(0.846). RoboticArm AUPRC must be read against a 0.853 target-slice prevalence baseline:
several RoboticArm target-slice AP values (0.81–0.90) sit at or below that baseline (AP can fall
below prevalence).

## 4. Healthy-target false alarms (shift robustness at the frozen q90)

- **RoboticArm.** The microphone is the modality most often fooled by a healthy shifted
  machine: hFPR 0.28–0.92 across families (0.28 is dense-AE-T; 0.35–0.92 for the other
  configurations), versus 0.05–0.55 for the accelerometer and
  0.00–0.15 for the gyroscope. The gyroscope's low hFPR is partly an artefact: four gyroscope
  configurations raise **no alarm at all** in any seed (statistical reference S and T, Isolation
  Forest S, Mahalanobis T; recall 0), and Isolation Forest T raises none in four of five seeds
  (one seed: a single detection). Only 20 segments from 6 parents; direction informative,
  magnitude imprecise.
- **BrushlessMotor.** Every classical configuration flags 27–100 % of healthy target
  recordings; in Regime S all five families flag ≥ 70 % for the IMU channels. The single
  exception is the dense AE in Regime T: hFPR 0.09 (acc), 0.02 (gyro), 0.07 (mic) with
  recall 0.94 / 0.75 / 0.32. That is the only configuration in the benchmark that is
  simultaneously discriminative and quiet on healthy shifted data — on a cohort where class
  and session are confounded (§9).

## 5. Source-to-target degradation

Target-slice AUROC (seed mean) versus source-slice AUROC:

- **RoboticArm accelerometer collapses under shift** for four of five families: source
  0.88–0.998 → target 0.52–0.66 (S) and 0.54–0.61 (T). The exceptions in Regime T are
  OCSVM (0.991 → **0.972**) and Mahalanobis (0.994 → 0.765): access to the 27 released target
  normals is what lets OCSVM-acc-T preserve strong target-slice ranking under few-shot
  adaptation; its healthy-target false-alarm rate nevertheless remains imperfect (hFPR 0.40).
  RoboticArm
  microphone is at chance in *both* slices (0.45–0.57), so it cannot "degrade"; RoboticArm
  gyroscope drops 0.68–0.87 → 0.50–0.62 for the classical families, while the dense AE is
  near chance in both slices (0.57 → 0.54).
- **BrushlessMotor** shows the same pattern with a sharper failure: Isolation-Forest
  accelerometer scores target *normals* as more anomalous than target *anomalies*
  (target-slice AUROC 0.315 in S, 0.224 in T) — a speed change is read as a fault. The dense
  AE in Regime T is nearly shift-invariant (acc 0.985 → 0.980; gyro 0.929 → 0.941; mic
  0.796 → 0.847).
- **Regime T − S (paired, complete cohort).** RoboticArm: within ±0.02 for classical
  families; dense-AE microphone −0.126 [−0.213, −0.007]. BrushlessMotor: the 18 target normals
  help the AE strongly (+0.10 to +0.16, CIs exclude 0, all three modalities), help OCSVM/Mahalanobis
  on the IMU (+0.06 to +0.18), and *hurt* Isolation Forest (−0.075 acc, −0.096 gyro).

## 6. Paired modality contrasts and the dominance rule

The frozen rule requires the paired CI to exclude zero with the same sign in **all five**
families.

| Machine | Regime | Contrast | AUROC | AUPRC | pAUROC |
|---|---|---|---|---|---|
| RoboticArm | S | acc − mic | 4/5 (AE CI includes 0) — no verdict | 4/5 | 0/5 |
| RoboticArm | S | acc − gyro | **acc dominates** (5/5, Δ +0.11 to +0.24) | 4/5 | 1/5 |
| RoboticArm | S | gyro − mic | 1/5 | 1/5 | 0/5 |
| RoboticArm | T | acc − mic | **acc dominates** (5/5, Δ +0.26 to +0.34) | **acc dominates** | 3/5 |
| RoboticArm | T | acc − gyro | **acc dominates** (5/5, Δ +0.12 to +0.28) | 4/5 | 3/5 |
| RoboticArm | T | gyro − mic | 1/5 | 1/5 | 1/5 |
| BrushlessMotor | S | acc − mic | **acc dominates** (5/5) | **acc dominates** | **acc dominates** |
| BrushlessMotor | S | gyro − mic | **gyro dominates** (5/5) | **gyro dominates** | 2/5 |
| BrushlessMotor | S | acc − gyro | split: stat_ref, IF favour gyro; Mahalanobis, AE favour acc; OCSVM tie | split | split |
| BrushlessMotor | T | acc − mic | **acc dominates** (5/5) | 4/5 | five significant, opposite signs (4 positive; Isolation Forest −0.011 [−0.024, −0.002]) — no verdict |
| BrushlessMotor | T | gyro − mic | **gyro dominates** (5/5) | **gyro dominates** | 4/5 |
| BrushlessMotor | T | acc − gyro | split (2 vs 3, all CIs exclude 0) — **model-dependent** | split | split |

## 7. Seed stability

Across-seed SD of AUROC: dense AE 0.002–0.017; Isolation Forest 0.003–0.028 (largest:
BrushlessMotor-S accelerometer 0.028). Every SD is far below the bootstrap CI width and below
every dominance-relevant Δ, so no comparison is seed-sensitive.

## 8. Strongest modality by family, and the universal-winner question

- **RoboticArm:** accelerometer, in every family, every regime, every metric.
- **BrushlessMotor:** accelerometer for OCSVM, Mahalanobis and dense AE; gyroscope for
  statistical reference and Isolation Forest (both near chance on acc).
- **Universal winner: NO.** The frozen rule supports "accelerometer > gyroscope" and (Regime T)
  "accelerometer > microphone" on RoboticArm, and "IMU > microphone" on BrushlessMotor. It does
  **not** support a single modality across machines: on BrushlessMotor the accelerometer-vs-
  gyroscope ranking is **MODEL-DEPENDENT**, and on RoboticArm in Regime S the dense AE's wide
  microphone CI blocks the acc − mic verdict.
- Family-matched best single modality for future fusion comparisons is recorded per (machine,
  regime, family) in `family_bootstrap_metrics.csv`.

## 9. Important disagreements between metrics

1. **AUROC vs healthy-target FPR.** RoboticArm Mahalanobis/OCSVM accelerometer rank anomalies
   almost perfectly on the source slice yet flag 40–55 % of healthy target segments; RoboticArm
   Isolation-Forest/statistical-reference gyroscope report hFPR 0.00 only because they flag
   nothing (IF-gyro-T: one detection in one seed). AUROC alone would pick a very different
   "winner" than the false-alarm axis.
2. **Complete-cohort AUROC vs target-slice AUROC.** RoboticArm accelerometer AUROC ≈ 0.95 on the
   pooled cohort hides a target-slice AUROC ≈ 0.6: the pooled number is carried by the source
   slice (330 of 466 segments).
3. **AP vs prevalence.** RoboticArm target-slice AP of 0.81–0.90 is at or below the 0.853
   prevalence baseline for most configurations.
4. BrushlessMotor `statistical_reference`: pAUROC picks acc, AUROC/AUPRC pick gyro (all ≈ chance).

## 10. MechaSight interpretation

**A. Discrimination.** Vibration carries the anomaly evidence. On RoboticArm the accelerometer
separates loose-screw segments from normal ones with AUROC 0.90–0.98 under Protocol B; the
microphone is at 0.55–0.69 and near chance within each domain slice. On BrushlessMotor the IMU
channels separate the released classes (acc 0.83–0.98 for the density/boundary/AE families),
the microphone does not (0.46–0.51) except through the dense AE (0.66 S, 0.82 T).

**B. Robustness to benign shift.** The central MechaSight failure mode is present and large:
a healthy machine at a new load or speed is scored as anomalous. RoboticArm accelerometer
target-slice AUROC ≈ 0.6 and hFPR up to 0.55; BrushlessMotor classical models flag most healthy
target recordings, and Isolation Forest inverts (target normals scored above target
anomalies). The two configurations whose target-slice ranking survives the shift both used
the released target normals in training: OCSVM-acc-T (RoboticArm; target-slice AUROC 0.972 but
hFPR 0.40, so false-alarm performance remains imperfect) and dense-AE-T (BrushlessMotor, all
modalities; hFPR ≤ 0.09). Under source-only generalisation (Regime S), **no single modality is
shift-robust** on either machine.

**Pre-registered audio expectation.** The frozen, verified evidence is that the environmental
perturbation is applied to audio only (`research_definition.md` §5.3 [VERIFIED];
`source_verification.md` S1 §3.1 VERIFIED, S6 CONFIRMED). The expectation is **partly
supported**: on RoboticArm the microphone has the highest healthy-target FPR of the three
modalities in every family. It is **not** supported on the ranking axis: the microphone cannot
degrade from source to target because it is already at chance in the source slice, and the
largest source→target *drops* occur in the IMU channels, driven by the operational (load /
speed) shift rather than the acoustic one.

## 11. BrushlessMotor confound

All released BrushlessMotor normal recordings precede all anomalous ones. Every BrushlessMotor
number above is *benchmark discrimination on the released cohort*. The dense-AE-T result
(AUROC 0.98, hFPR 0.09) is exactly the kind of result a session-drift detector would also
produce; it must not be presented as fault sensitivity.

## 12. Unexpected findings

1. **Single-modality accelerometer scores above the seven-channel concatenation of Phase 3.2**
   (descriptive, cross-stage, not paired-bootstrapped): RoboticArm OCSVM-T 0.980 vs 0.893
   all-channel; Isolation-Forest-S 0.952 vs 0.914; BrushlessMotor dense-AE-T 0.982 vs the best
   classical all-channel 0.865. This is descriptive evidence that simple concatenation *can*
   underperform the best single modality; matched fusion baselines under one batch are still
   required before any such claim is made.
2. **Isolation Forest is hurt by the 18 BrushlessMotor target normals** (T − S −0.08 / −0.10)
   while every other family gains — the added normals shift its partition statistics without
   reshaping the boundary.
3. **The dense AE is the only family that turns few-shot target normals into shift robustness**
   on BrushlessMotor (target-slice AUROC ≈ source-slice; hFPR ≤ 0.09), and the only family on
   which the BrushlessMotor microphone carries usable evidence.
4. On RoboticArm the dense AE underperforms the classical accelerometer models by ~0.15 AUROC
   while three of its microphone runs hit the frozen 200-epoch cap; accepted as frozen, not
   revisited.

## 13. Conclusion

Under the leakage-controlled Protocol B, the **accelerometer is the strongest single modality on
RoboticArm for every model family and metric**, and **inertial sensing dominates the microphone
on BrushlessMotor for every family**; whether accelerometer or gyroscope is better on
BrushlessMotor is model-dependent. There is **no universal single-modality winner**, and no
modality is robust to benign operating shift under source-only training. Few-shot access to
target normals preserves strong target-slice ranking for specific model–modality pairs
(OCSVM-acc on RoboticArm, whose healthy-target false-alarm rate remains imperfect at 0.40; dense AE
on BrushlessMotor, which is also quiet on healthy target data), not in general.

## 14. Implications for Multi-Sensor Fusion Development (not started)

- The comparator "(a) best single modality" is now defined per (machine, regime, family) with
  persisted per-segment scores, so fusion can be tested against its family-matched best
  modality with the frozen paired bootstrap.
- Existing cross-stage results give descriptive evidence that simple feature concatenation can
  underperform the best single modality (Phase 3.2 vs this stage); the fusion stage must include
  a matched concatenation arm under the same batch and paired bootstrap before drawing that
  conclusion.
- The microphone contributes little ranking evidence but is the dominant false-alarm source on
  RoboticArm; a reliability-weighting scheme that *down-weights* audio under shift is the
  natural first hypothesis, and the RoboticArm target slice (20 healthy segments) is the
  right place to look for a false-alarm reduction.
- The dense-AE-T BrushlessMotor results set a high, confound-limited ceiling; fusion claims on
  BrushlessMotor should be made on the false-alarm axis and worded as benchmark discrimination.
