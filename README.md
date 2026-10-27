# MechaSight

**A leakage-aware multi-sensor framework for machine anomaly detection under operating-condition shifts**

Minor project, Mechanical & Automation Engineering.
**Team:** Amaan Khan (student researcher), Vikas Sharma (supervisor), Surbhi Upadhyay (co-supervisor).

> **One-line idea:** a machine that sounds or vibrates *differently* is not necessarily *faulty*.
> MechaSight asks whether combining a microphone, an accelerometer and a gyroscope helps tell genuine
> faults apart from harmless changes in load, speed or background noise. It evaluates that question
> under a protocol strict enough that the answer can be trusted.

---

## Contents

1. [What the project does](#1-what-the-project-does)
2. [Current status: what is done and what is pending](#2-current-status)
3. [Where is the code?](#3-where-is-the-code)
4. [Repository layout](#4-repository-layout)
5. [Installation](#5-installation)
6. [Running the pipeline, step by step](#6-running-the-pipeline-step-by-step)
7. [Key results so far](#7-key-results-so-far)
8. [Evaluation rules (why the results can be trusted)](#8-evaluation-rules)
9. [Documentation index](#9-documentation-index)
10. [Limitations](#10-limitations)
11. [About this repository, licence and attribution](#11-about-this-repository)

---

## 1. What the project does

Factories need to detect machine faults early, but they rarely have labelled fault data. The usual
approach is **unsupervised anomaly detection**: learn what *healthy* looks like and flag anything
different. That approach breaks when a healthy machine changes its operating condition. A new load,
a new speed or a noisier hall changes the signals, and the detector raises false alarms.

MechaSight studies this on the public **IMAD-DS** dataset (Albertini et al., DCASE 2024 Workshop;
Zenodo DOI [10.5281/zenodo.12665499](https://doi.org/10.5281/zenodo.12665499)). It covers two scaled
industrial machines, a **robotic arm** and a **brushless motor**. Each segment is recorded with three
sensors:

| Sensor | Signal | Channels | Effective rate |
|---|---|---|---|
| Microphone (IMP23ABSU) | sound | 1 | ≈ 16 kHz |
| Accelerometer (ISM330DHCX) | vibration | 3 axes | ≈ 7.1 kHz |
| Gyroscope (ISM330DHCX) | rotation | 3 axes | ≈ 7.1 kHz |

The project builds a **four-level ladder** of detectors and compares them under identical, frozen
rules:

| Level | Name | What it does |
|---|---|---|
| **L0** | single modality | one detector per sensor (five model families) |
| **L1** | early fusion | one detector trained on all sensors' features or spectra concatenated |
| **L2** | score fusion | each sensor's score converted to a robust z-score and averaged (all three sensors, or any pair) |
| **L3** | integrity fallback | rule-based sensor checks; if a sensor is dead or stuck it is excluded and the remaining sensors decide; if timing is broken the system abstains and raises a flag |

Model families: a statistical reference (RMS of z-scored features), Isolation Forest, one-class SVM,
Mahalanobis distance with Ledoit–Wolf shrinkage, and a dense autoencoder on log-power spectra.

## 2. Current status

**Multi-Sensor Fusion Development: development complete. Protected fusion evaluation pending
authorization.**

| Stage | Status |
|---|---|
| Dataset audit and integrity review | ✅ complete |
| Leakage-controlled evaluation protocol (Protocol B) and splits | ✅ complete |
| Classical baselines and first locked test | ✅ complete |
| Reproduction of the published IMAD-DS baseline (Protocol A) | ✅ complete (within 0.33 AUC points) |
| Sibling-exposure sensitivity study | ✅ complete |
| Single-modality benchmark: 156 detectors, protected evaluation | ✅ complete |
| Fusion development: 24 L1 models, 208 L2 calibrations, L3 integrity fallback | ✅ complete; reproduced exactly on development data |
| L3 synthetic falsification (1,992 corruption runs) | ✅ all 7 pre-registered criteria passed |
| Protected-evaluation executor (exposure-first, append-only test-look ledger) | ✅ implemented; tested end to end on a development mirror |
| **Protected evaluation of the fusion arms (232 planned test looks)** | ⏳ **not executed**: awaits independent review and human authorization |

> **No fusion accuracy, improvement or statistical significance is claimed anywhere in this
> repository.** Exact reproduction of development scores shows the implementation is consistent; it
> does not measure detection performance.

## 3. Where is the code?

All research logic is ordinary, importable Python in `src/`. The runnable entry points are in
`scripts/`, the frozen settings in `configs/`, and the automated checks in `tests/` (60 files).

### 3.1 Model definitions and training code

| What | File | Notes |
|---|---|---|
| **Statistical reference, Isolation Forest, one-class SVM** | [`src/models/classical.py`](src/models/classical.py) | fit/score wrappers around scikit-learn; hyperparameters frozen in code |
| **Mahalanobis (Ledoit–Wolf) detector** | [`src/models/mahalanobis.py`](src/models/mahalanobis.py) | squared Mahalanobis distance |
| **Dense autoencoder (PyTorch)** | [`src/models/dense_autoencoder.py`](src/models/dense_autoencoder.py) | architecture `d→256→64→16→64→256→d`; the **training loop** (Adam 1e-3, batch 256, early stopping), window scoring and save/load |
| Classical training on engineered features | [`src/experiment/phase31.py`](src/experiment/phase31.py) | standardizer fit on training data only, model fit, validation scores, q90 threshold |
| **Single-modality training (all 156 L0 detectors)** | [`src/experiment/phase4.py`](src/experiment/phase4.py) | trains each family per machine × regime × sensor and writes hash-manifested bundles |
| **Early-fusion training (24 L1 models)** | [`src/fusion/l1.py`](src/fusion/l1.py) | Mahalanobis on 98 features, autoencoder on concatenated spectra |
| Input assembly for fusion | [`src/fusion/assembly.py`](src/fusion/assembly.py) | builds the all-sensor feature and spectrum matrices |
| **Score fusion (208 L2 calibrations)** | [`src/fusion/l2.py`](src/fusion/l2.py) | robust z-score (median/IQR), subset mean, fused q90 threshold |
| **Integrity fallback (L3)** | [`src/fusion/l3.py`](src/fusion/l3.py), [`src/fusion/l3_falsification.py`](src/fusion/l3_falsification.py) | gates G1–G3, fallback decision, synthetic corruption suite |

### 3.2 Data processing

| What | File |
|---|---|
| Reading sensor files, timestamp checks, alignment, 100-ms windows | [`src/preprocessing/signals.py`](src/preprocessing/signals.py), [`src/preprocessing/pipeline.py`](src/preprocessing/pipeline.py) |
| Recording-parent-aware train/validation/test splits | [`src/preprocessing/splits.py`](src/preprocessing/splits.py) |
| 14 time/frequency features per channel | [`src/features/extraction.py`](src/features/extraction.py) |
| Log-power spectra for the autoencoders | [`src/preprocessing/spectral_windows.py`](src/preprocessing/spectral_windows.py), [`src/preprocessing/spectral_cache.py`](src/preprocessing/spectral_cache.py) |
| Normalisation (train-only z-score) | [`src/preprocessing/normalization.py`](src/preprocessing/normalization.py) |
| Dataset audit | [`src/data/audit.py`](src/data/audit.py) |

### 3.3 Evaluation and statistics

| What | File |
|---|---|
| AUROC, average precision, partial AUROC, threshold metrics | [`src/experiment/metrics.py`](src/experiment/metrics.py) |
| Parent-aware paired bootstrap | [`src/experiment/bootstrap.py`](src/experiment/bootstrap.py), [`src/experiment/phase32.py`](src/experiment/phase32.py) |
| Single-modality analysis and modality contrasts | [`src/experiment/phase4_analysis.py`](src/experiment/phase4_analysis.py) |
| Fusion contrasts, practical margins and categories | [`src/fusion/statistics.py`](src/fusion/statistics.py), [`src/experiment/phase5_analysis.py`](src/experiment/phase5_analysis.py) |
| Append-only test-look ledger (every test evaluation is logged) | [`src/experiment/ledger.py`](src/experiment/ledger.py), [`src/experiment/ledger_history.py`](src/experiment/ledger_history.py) |
| Protected-evaluation executor (exposure recorded before any test read) | [`src/experiment/phase5_execution.py`](src/experiment/phase5_execution.py) and the other `src/experiment/phase5_*.py` modules |
| Official baseline reproduction (Protocol A) | [`src/reproduction/protocol_a.py`](src/reproduction/protocol_a.py) |

### 3.4 Entry-point scripts

| Script | What it runs |
|---|---|
| `scripts/verify_download.py`, `scripts/extract_imadds.py` | verify the IMAD-DS download against official checksums, then extract it |
| `scripts/audit_imadds.py` | full dataset audit → `reports/dataset_audit/` |
| `scripts/build_phase22_splits.py` | the current Protocol-B split (v3) |
| `scripts/prepare_phase31_features.py` | the 98 engineered features per segment |
| `scripts/run_phase31_baselines.py` | fit the classical baselines (validation only) |
| `scripts/build_phase4_spectral_cache.py` | spectral caches for the autoencoders |
| `scripts/run_phase4_pretest.py` | **train all 156 single-modality detectors** (validation only) |
| `scripts/verify_phase4_bundles.py` | re-score every saved model and check it reproduces exactly |
| `scripts/run_phase5_development.py` | **train the 24 L1 fusion models** (`l1-fit`), verify them (`l1-verify`), compute and verify the 208 L2 calibrations (`l2-calibrate`, `l2-verify`) |
| `scripts/run_phase5_l3.py` | L3 clean scan (`clean-scan`) and the 1,992-run falsification suite (`falsify`) |
| `scripts/run_phase5_mirror.py` | end-to-end development-mirror run of the protected-evaluation path |
| `scripts/run_protocol_a_reproduction.py` | reproduce the published baseline (`preprocess`, `train`, `evaluate`) |
| `scripts/run_phase4_protected.py`, `scripts/run_phase32_first_test.py` | historical, already-completed protected evaluations (kept for traceability; they refuse to run twice) |

## 4. Repository layout

```
configs/      frozen experiment configurations (preprocessing, splits, model ladder, fusion design)
src/          all research code: data, preprocessing, features, models, fusion, experiment, evaluation
scripts/      runnable entry points (see section 3.4)
tests/        protocol, leakage-prevention, signal-processing and executor tests (pytest)
reports/      committed results: dataset audit, splits, model bundles (configs, scores, thresholds),
              historical protected results, development evidence
docs/         research definition, frozen protocols, stage reports and audits
defence/      minor-project presentation slides (28 October 2026)
data/         empty placeholders; the dataset and model weights are NOT committed (see .gitignore)
app/, notebooks/   placeholders
```

Trained model weights (`*.pth`, `*.joblib`) are not committed. Every model's configuration,
validation scores, threshold and manifest hash are committed under `reports/`, so a retrained model
can be checked against them.

## 5. Installation

Python 3.14 was used; `requirements.txt` pins every package.

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

On Linux or macOS, activate with `source .venv/bin/activate` instead.

```bash
pip install -r requirements.txt
```

Main dependencies: NumPy, pandas, PyArrow, h5py, scikit-learn, PyTorch (the CPU build is enough),
joblib, matplotlib, pytest and psutil.

Download the two archives of the dataset (`RoboticArm.7z`, `BrushlessMotor.7z`) from
[Zenodo record 12665499](https://zenodo.org/records/12665499) into `data/raw/`.

## 6. Running the pipeline, step by step

Every step runs on training and validation data only. The test cohort is touched only by the
dedicated protected stages, which require a recorded authorization.

```bash
python scripts/verify_download.py --data-root data/raw
```

```bash
python scripts/extract_imadds.py --data-root data/raw
```

```bash
python scripts/audit_imadds.py --data-root data/raw/extracted
```

```bash
python scripts/build_phase22_splits.py
```

```bash
python scripts/prepare_phase31_features.py
```

```bash
python scripts/run_phase31_baselines.py
```

```bash
python scripts/build_phase4_spectral_cache.py
```

```bash
python scripts/run_phase4_pretest.py
```

```bash
python scripts/verify_phase4_bundles.py
```

```bash
python scripts/run_phase5_development.py l1-fit
```

```bash
python scripts/run_phase5_development.py l1-verify
```

```bash
python scripts/run_phase5_development.py l2-calibrate
```

```bash
python scripts/run_phase5_development.py l2-verify
```

```bash
python scripts/run_phase5_l3.py clean-scan
```

```bash
python scripts/run_phase5_l3.py falsify
```

Run the test suite:

```bash
python -m pytest -q
```

Compute notes, measured on a laptop CPU (AMD Ryzen 7 7840HS): spectral caches take about 16 min,
the 96 classical fits about 10 s, and the 60 single-modality autoencoder trainings about 3 h of wall
time. Some suites read the dataset and are slow.

## 7. Key results so far

**Protocol A, the published baseline on the released split, reproduced.** Historical, completed.

| Machine | Slice | Published AUC | Reproduced AUC |
|---|---|---:|---:|
| RoboticArm | overall | 91.62 | 91.51 |
| BrushlessMotor | overall | 58.95 | 58.98 |

**Dataset audit findings that changed the design:**
- On RoboticArm, **91.4%** of normal test segments in the official split have a sibling excerpt
  from the *same continuous recording* in training. That is a leakage risk, so MechaSight's
  Protocol B splits by **parent recording**.
- On BrushlessMotor, every normal recording precedes every anomalous one in time. Class and
  recording session are confounded, so those results are reported only as benchmark
  discrimination.

**Single-modality benchmark: 156 detectors, protected evaluation complete.** See
`docs/phase4_result_report.md`.
- The accelerometer is the strongest sensor on RoboticArm in every model family. On
  BrushlessMotor, inertial sensors beat the microphone. There is no universal best sensor.
- Benign operating shift is a major failure mode. Under source-only training, no single sensor is
  robust; RoboticArm accelerometer AUROC ≈ 0.95 overall falls to ≈ 0.6 on the shifted (target)
  slice.
- On RoboticArm the microphone causes the most false alarms on healthy machines in new
  conditions: a healthy-target false-alarm rate of 0.28–0.92.

**Fusion development (development data only; not performance):**
- All 24 L1 models and all 208 L2 calibrations reproduce exactly: 3,744 L1 segment scores and
  32,448 L2 scores, maximum difference 0.0.
- The L3 integrity gates gave 0 false triggers on 21,154 clean development segment-channels and
  passed all 7 pre-registered criteria over 1,992 synthetic corruption runs.

## 8. Evaluation rules

These rules are why the results can be trusted:

- **Normal-only training.** Anomaly labels are used for evaluation only, never for training.
- **Splits by parent recording**, so near-duplicate excerpts never sit on both sides of a split.
- **Frozen thresholds:** the 90th percentile of normal validation scores, a strict
  `score > threshold`, fixed before any test data is seen.
- **Every test evaluation is logged** in an append-only *test-look ledger*. New evaluations must
  be authorized and recorded **before** test data is read.
- **Paired, parent-aware bootstrap** (10,000 replicates) with pre-registered practical margins.
  Results are reported per machine, never averaged.
- **Two protocols kept apart.** Protocol A reproduces the published benchmark; Protocol B is
  MechaSight's leakage-controlled protocol.

## 9. Documentation index

Start with [`docs/research_definition.md`](docs/research_definition.md).

| Document | Content |
|---|---|
| [`reports/dataset_audit/dataset_summary.md`](reports/dataset_audit/dataset_summary.md) | dataset audit findings |
| [`docs/evaluation_protocol.md`](docs/evaluation_protocol.md) | frozen evaluation protocol (A vs B, regimes, metrics, thresholds) |
| [`docs/leakage_risks.md`](docs/leakage_risks.md) | leakage and validity risk register |
| [`docs/preprocessing_phase2.md`](docs/preprocessing_phase2.md) | preprocessing design |
| [`docs/phase4_single_modality_freeze.md`](docs/phase4_single_modality_freeze.md) | single-modality model design |
| [`docs/phase4_result_report.md`](docs/phase4_result_report.md) | single-modality benchmark results |
| [`docs/phase5_fusion_design_v2.md`](docs/phase5_fusion_design_v2.md) | fusion design (L0–L3, hypotheses, statistics) |
| [`docs/phase5_3_completion.md`](docs/phase5_3_completion.md) | L1/L2 development completion |
| [`docs/phase5_4_sensor_integrity_fusion.md`](docs/phase5_4_sensor_integrity_fusion.md) | L3 integrity fallback and falsification |
| [`docs/phase5_6_protected_evaluation_readiness.md`](docs/phase5_6_protected_evaluation_readiness.md) | readiness plan for the protected fusion evaluation |
| [`docs/phase5_6c_final_candidate.md`](docs/phase5_6c_final_candidate.md) | final executable candidate and integrity report |
| [`docs/phase34_final_report.md`](docs/phase34_final_report.md) | sibling-exposure sensitivity study |
| [`reports/phase3_3_protocol_a/protocol_a_reproduction_report.md`](reports/phase3_3_protocol_a/protocol_a_reproduction_report.md) | Protocol-A reproduction |

The `docs/phase*` files form a chronological record of each stage, including independent audits
and corrections.

## 10. Limitations

- Protected fusion results are **pending**. No fusion performance is claimed.
- BrushlessMotor has a class/acquisition-time confound.
- RoboticArm's healthy shifted cohort is small: 20 segments from 6 recordings.
- The sensor-integrity fallback is validated only on **synthetic** corruptions, not real sensor
  failures.
- The data are recorded, laboratory-scale machines, not live industrial monitoring.

## 11. About this repository

This repository is a curated public snapshot of the MechaSight development repository. Raw data,
model weights and local working files are excluded. Documentation was lightly edited: wording, and
the redaction of local paths.

Several integrity scripts (for example the frozen-package and candidate verifiers) check files
against hashes and commit identifiers from the original development history. In this snapshot
those checks are expected to report mismatches. Scientific results and configurations are
otherwise unchanged.

**Licence and attribution.** The IMAD-DS dataset is licensed CC BY-SA 4.0 by its authors and is not
redistributed here. Please cite: D. Albertini, F. Augusti, K. Esmer, A. Bernardini and R. Sannino,
*IMAD-DS: A Dataset for Industrial Multi-Sensor Anomaly Detection Under Domain Shift Conditions*,
DCASE 2024 Workshop. Third-party papers are not committed; `docs/source_verification.md` lists
where to obtain them.
