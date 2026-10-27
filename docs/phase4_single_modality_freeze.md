# Single-Modality Model Benchmarking — Design Freeze and Pre-Test Record

<!-- Legacy roadmap name: Phase 4 -->

**Status:** pre-test fitting complete; protected batch **proposed, not authorised**.
**Frozen configuration:** `configs/phase4/single_modality_benchmark_v1.json`.
**Failure policy:** `docs/phase4_failure_policy_note.md`.

## 1. Question

How well can each sensing modality — microphone, accelerometer, gyroscope — detect machine
anomalies **on its own**, per machine, under the leakage-controlled Protocol B (few-shot
adaptation split v3, Regimes S and T, common test unchanged)? The stage also produces the
"best single modality" comparator and the per-modality scores that Multi-Sensor Fusion
Development needs. No cross-modality cell is evaluated here.

## 2. Model set (five families, no additions)

| Family | Input | Implementation | Seeds |
|---|---|---|---|
| `statistical_reference` | modality subset of the frozen 98 segment-median features, train-only z-score | `src/models/classical.py` (unchanged Phase-3.1 code) | 0 |
| `isolation_forest` | same | same | 0–4 |
| `one_class_svm` | same | same | 0 |
| `mahalanobis_shrinkage` | same | `src/models/mahalanobis.py`: `LedoitWolf`, `assume_centered=False`; score = squared Mahalanobis distance | 0 |
| `dense_autoencoder` | per-window log-power spectrum of one modality, train-only per-bin z-score | `src/models/dense_autoencoder.py`: `d→256→64→16→64→256→d`, ReLU hidden, linear output, MSE, Adam 1e-3, batch 256, ≤200 epochs, patience 10 on source-validation window MSE, best-validation checkpoint | 0–4 |

Deterministic families carry seed 0 as a ledger field only. Segment score for the AE is the
**median** of its window scores (frozen Protocol-B rule). Threshold for every configuration:
q90 of source-validation segment scores, quantile method `higher`.

Not in this benchmark: Random Forest, XGBoost/LightGBM, CNN, TCN, k-NN/embedding distance,
all-channel cells, raw-waveform AE (not selected; not claimed invalid).

## 3. Modality inputs

**Classical:** columns of the frozen Phase-3.1 artifact whose name starts with `mic__` (14),
`acc__` (42) or `gyro__` (42). Feature-wise z-scoring on a subset is identical to the
98-dimensional fit restricted to those columns.

**Autoencoder:** frozen `resample_regular` → truncate to a fixed length → frozen
`one_sided_power` → `log(power + 1e-12)` → concatenate that modality's channels (channel-major).

| | mic | RoboticArm acc / gyro | BrushlessMotor acc / gyro |
|---|---:|---:|---:|
| fixed samples | 1598 | 704 | 711 |
| bins | 800 | 353 | 356 |
| **input dim** | **800** | **1059** | **1068** |

### 3.1 Fixed-length amendment (recorded before any model was fit)

The approved rule was `floor(0.1·rate) − 1` (mic 1599; RA IMU 705; BM IMU 712; BM IMU dim
1071). Measured over all 194,165 authorised training and source-validation windows on
2026-09-13, the frozen resampler produces:

| cohort | mic | acc | gyro |
|---|---|---|---|
| RoboticArm train (115,070) | 1598: 118 · 1599: 58,984 · 1600: 55,968 | 705 / 706 / 707 | 705 / 706 / 707 |
| RoboticArm validation (13,764) | 1598: 3 · 1599: 6,570 · 1600: 7,191 | 705 / 706 / 707 | 705 / 706 / 707 |
| BrushlessMotor train (58,905) | 1599: 669 · 1600: 58,236 | 713 / 714 | 712: 1,115 · 713 · 714 |
| BrushlessMotor validation (6,426) | 1599: 79 · 1600: 6,347 | 713 / 714 | 712: 120 · 713 · 714 |

121 RoboticArm mic windows fall two below the floor, so the `−1` rule was falsified by
training data before fitting. The frozen rule is `floor(0.1·rate) − 2`, which covers every
observed window with one sample of headroom. Only the BrushlessMotor IMU dimension changes
(1071 → 1068). A protected-test window shorter than the fixed length is a technical defect
under the failure policy, not data.

## 4. Experiment count

2 machines × 2 regimes × 3 modalities = 12 cells; per cell 1 + 5 + 1 + 1 + 5 = 13
configurations; **156 experiments**, each one protected look. Ledger expected start: 40.

## 5. Train / validation / test flow

1. **Pre-test (done):** spectral caches for `train` and `validation` only (the pipeline
   refuses reserved/blocked roles; the cache builder refuses `common_evaluation`). Per
   configuration: standardizer fit on the frozen training cohort → fit → source-validation
   scores → q90 → immutable bundle (`reports/phase4_pretest/experiments/<id>/`) plus a
   hash-manifested model artifact (`data/interim/phase4/models/<id>/`, git-ignored; its
   manifest hash is recorded in the committed bundle).
2. **Protected, one invocation (`scripts/run_phase4_protected.py score-all`):** verify every
   bundle and model hash → build the common-evaluation spectral cache → load label-free test
   inputs → append **156 looks atomically** to the ledger → score every bundle into a sealed
   staging area. The scoring loop reads no anomaly label and computes no metric.
3. **Analyse once (`analyse-once`):** refuses unless exactly 156 hashed score tables verify;
   then computes AUROC, AUPRC (+ prevalence), pAUROC@0.10, source/target slices,
   healthy-target FPR at the frozen q90, seed aggregation, paired modality contrasts and
   T − S contrasts in one pass. A completed analysis is final.

## 6. Seed aggregation and inference

Frozen R8 rule: class-stratified parent-cluster bootstrap, seed 1995, 10,000 replicates,
shared draws; per replicate each seed's metric is computed and averaged; the paired
difference of seed-averaged metrics is the inference. Across-seed mean ± SD reported
alongside. No best-seed selection.

## 7. Pre-registered contrasts and reading rule

Within each (machine, regime, family): `acc − mic`, `gyro − mic`, `acc − gyro` for AUROC,
AUPRC, pAUROC@0.10. Within each (machine, modality, family): `T − S`.

**Dominance rule:** a modality dominates another on a machine and regime only when the
paired CI excludes zero with the same sign in **every one of the five families**; otherwise
the ranking is reported as model-dependent.

**Expectations stated before results:** the microphone is the modality whose target-slice
metrics and healthy-target FPR degrade most (the environmental shift is injected into audio
only). BrushlessMotor results are benchmark discrimination on the released cohort
(`brushless_motor_acquisition_time_class_confound_v1`), never fault sensitivity. RoboticArm
target-slice values rest on 20 healthy segments from 6 parents.

## 8. Reproduction check before scoring

Bundle files (config, provenance, threshold, validation scores, summary) and model artifacts
are hash-verified against their manifests. The frozen checkpoint or estimator is scored;
nothing is retrained. The torch intra-op thread count is recorded in every AE training
history because it affects floating-point summation order.

## 9. Pre-test record (2026-09-13)

| Item | Value |
|---|---|
| Bundles | **156 / 156** under `reports/phase4_pretest/experiments/` |
| Fit code HEAD (every bundle) | `85df48d819c25679b596286adda9df21720c93e9` |
| Source snapshot hash (every bundle) | `7296665253064b97b8b67c3b88714a29dc3b3df79d0b96a439f5eefc909fbc49` |
| Frozen config | `configs/phase4/single_modality_benchmark_v1.json` |
| Proposed protected batch id | `p4-test-batch-f3fa02a57fc7adc6650e7522` — **PROPOSED, NOT AUTHORISED** |
| Test-look ledger | 40 entries before and after; untouched |
| Reproduction check | all 156 bundles re-score source validation from the frozen artifact with max abs difference 2.8 × 10⁻¹⁴ (`bundle_reproduction_check.json`) |
| Protected data | no common-evaluation cache built; no test feature loaded; no test score or metric computed |

**Compute observed (CPU, AMD Ryzen 7 7840HS, two to three 4-thread workers):** spectral
caches 16 min; 96 classical fits 10 s; 60 AE trainings 4.0 h of fit time, wall ≈ 3 h.

**AE training behaviour (source-validation window MSE in standardised units; descriptive,
not a ranking):** BrushlessMotor converges in 30–60 epochs (mic 0.67, acc 0.79, gyro 0.85);
RoboticArm acc/gyro in 20–100 epochs (0.81 / 0.95); RoboticArm mic keeps improving slowly,
using 105–200 epochs (0.43). Three RoboticArm mic runs (S seed 4; T seeds 2 and 3) reached the
frozen 200-epoch cap with best epochs 196–199 and a last-vs-best gap ≤ 2.4 × 10⁻⁴; the cap is
frozen and is not revisited. Across-seed SD of best validation MSE is ≤ 0.001 everywhere.
The torch intra-op thread count (4) is recorded in every history.

Reproduction-check and `--regimes` worker split were added after the fits at a later commit;
the bundles remain bound to the fit HEAD above, and the protected batch id binds the
scoring-time HEAD separately.
