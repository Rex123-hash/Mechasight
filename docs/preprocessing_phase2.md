# MechaSight Phase 2/2.1/2.2 preprocessing design

**Status:** Phase-2.2 adaptation-budget correction implemented and tested 2026-08-23, on top of the
Phase-2.1 matched-cohort correction (2026-08-22).

**Superseded for the Phase-3 classical path.** This document describes the Phase-2 window-level
preprocessing. Models have since been trained: Phase 3.1 fitted 28 pre-registered classical
configurations and Phase 3.2 recorded the first locked common-test results. The Phase-3 classical
representation is *median of **each feature** across a segment's windows*, producing one
98-dimensional vector per segment — **not** median of per-window anomaly scores. Where this document
or the Protocol-B evaluation config refers to a "median window anomaly score" aggregation, that
wording predates the Phase-3.1 representation freeze in
`configs/phase3/preprocessing_primary_v1.json` and is superseded for the classical baseline path.
The historical configs are deliberately left unedited so earlier artifacts stay reproducible.

## Two non-interchangeable paths

The required `protocol` and `regime` fields are validated and hashed into a configuration id.

- `official_baseline_reproduction`, Regime T: released folders; 100 ms non-overlapping windows;
  common timestamp coverage after 35 ms gyro warm-up; the upstream right-inclusive pandas slicing
  behavior; fixed lengths `int(fs × 0.1)` at 16000/7063/7063 Hz; zero-pad/truncate; and exact
  per-window/per-channel `std_window` normalisation with epsilon `1e-5` in the denominator.
- `mechasight_experimental_protocol`, Regime S or T: frozen parent-recording split; configurable
  window length and stride; half-open physical windows; machine-aware startup handling; native
  multi-rate temporal arrays; optional regular grids only for consumers that require them; and
  deterministic or train-fitted normalisation.

Changing protocol, regime, split version, window length, stride, resampling, crop, normalisation,
feature family, or cache format changes the configuration id. Machine and split are additionally
encoded in the cache path and HDF5 header. Existing caches are exclusively created and never
silently overwritten.

Phase-2.1 used split version `protocol_b_common_test_v2` and config version `phase2.1-v2`
(`phase2.1-v2-8688f5639d9fa7c1` / `phase2.1-v2-f497d27caee67b4a`). Phase-2.2 supersedes it with
split version `protocol_b_few_shot_adaptation_v3` and config version `phase2.2-v3`; its S/T ids are
`phase2.2-v3-dcef3e9c758c24ae` and `phase2.2-v3-0a0285c9afd6ad6d`. None of the three generations can
validate against another's split manifests or caches, so a Phase-2.1 training cache can never be
mistaken for the corrected primary Regime-T definition. Historical v1/v2 artifacts are intact.

## Split before sampling

`reports/phase2_2/protocol_b_few_shot_adaptation_v3.csv` is the current Protocol-B assignment
source. Phase-2 v1 (`reports/phase2/`) and Phase-2.1 v2 (`reports/phase2_1/`) remain preserved and
superseded: v1 gave S and T different test populations; v2 fixed that but used the whole parent
closure as the Regime-T training pool.

Five primary protocol roles now exist, and every segment carries exactly one:

| Role | Split label | RA segments | BM segments |
|---|---|---:|---:|
| `source_training` | `train` | 1528 | 1137 |
| `source_validation` | `validation` | 186 | 126 |
| `target_adaptation` | `train` (T) / `reserved_adaptation` (S) | 27 | 18 |
| `adaptation_sibling_blocked` | `blocked_sibling` | 96 | 0 |
| `common_evaluation` | `test` | 466 | 312 |

RoboticArm's 27 released target-normal training segments sit in 18 parent recordings whose parent
closure is 123 normal segments. Phase 2.2 keeps those two facts apart: Regime T consumes the 27,
and the 96 siblings are blocked from Regime-S training, Regime-T training, validation, threshold
calibration, hyperparameter selection and common test. Their only job is to keep the 18 adaptation
parents out of the evaluation cohort. BrushlessMotor has one segment per recording, so its 18
released examples have no siblings and the closure equals the allocation.

Parent-leakage assertions run over the consuming splits (`train`/`validation`/`test`); the two
non-consuming labels are excluded because they enter no computation. Source train/validation and
the complete common test are identical across regimes, and the common test is additionally asserted
identical to the Phase-2.1 freeze. Automated assertions compare segment, parent, anomaly,
validation and blocked-sibling ids exactly.

## Timestamp alignment and windows

Parquet streams are sorted and required to have strictly increasing physical timestamps. Alignment
uses `[max(first timestamps), min(last timestamps)]` after configured startup transforms; raw sample
index is never used to align modalities. MechaSight windows are half-open `[start, end)`, preventing
boundary samples from becoming siblings in adjacent non-overlapping windows. Every manifest row
retains physical start/end, raw file references, segment, parent, and window index.

The experimental 100 ms / 100 ms configuration is a smoke-build value and an initial Phase-3
candidate, not a claim of optimality. Both fields are configurable; a window-size search was not
performed in Phase 2.

## Sampling and resampling

Native timestamps and modality-specific lengths are always retained. There is no global sample
rate. The three consumer routes are:

1. Time-domain features: native samples, no interpolation.
2. Frequency-domain features: a modality- and machine-specific regular grid, linear interpolation,
   no extrapolation beyond the physical intersection. Targets are the Phase-1 median span rates:
   RA 16000.0/7063.280051/7063.280051 Hz and BM
   16003.076013/7135.714148/7135.329526 Hz (mic/acc/gyro). These targets are near-native, not
   downsampling targets, so an anti-alias low-pass is not invoked. Any future lower target rate must
   add a justified anti-alias filter.
3. Raw temporal models: native modality-specific ragged arrays with timestamps, offsets, and
   lengths. Fusion is deliberately deferred.

Linear interpolation is a coordinate transform for operations needing a regular grid; it does not
create new physical information. Original and transformed sample counts and the transformation
description are stored per modality in every manifest row.

## Startup and warm-up

- Protocol A: no accelerometer correction; 35 ms gyro warm-up retained exactly.
- MechaSight: `drop_initial_acc_samples = {BrushlessMotor: 1, RoboticArm: 0}`. The deterministic
  machine rule is applied to every BM stream rather than detecting rails from label- or
  signal-dependent logic. Counts are logged. The 35 ms gyro warm-up remains configurable and is
  applied before common-coverage calculation.

## Normalisation

- `std_window` is a deterministic inference-time transformation and uses the current window only.
- `train_zscore` has explicit `fit(..., split="train")` / `transform(...)` semantics, records the
  contributing segment ids and refuses validation/test fitting.
- `none` preserves amplitude and is the Phase-2 experimental smoke configuration. Normalisation
  selection belongs to Phase 3 validation; no test population statistic is implemented.

## Classical features

All features are per channel and compatible with mic/accelerometer/gyro channels. Time features are
mean, population standard deviation, RMS, population variance, absolute peak, peak-to-peak,
skewness, excess kurtosis, crest factor, and sample energy. Frequency features are dominant non-DC
frequency, one-sided Parseval spectral energy, spectral centroid, spectral spread, and configurable
band energy. Units/definitions are in `src/features/extraction.py`. Synthetic tests cover constants,
known RMS/energy, sine frequency and PSD peak, centroid, band energy, and Parseval equality.

## Cache, temporal representation, and manifest

The raw release is read-only. The HDF5 `hdf5_ragged_v1` cache stores, per modality, a contiguous
float32 value array, float64 physical timestamps, `(start, stop)` offsets, and lengths. It is
append-only during a streaming build, so only one segment/window needs to be resident. Modalities
are never padded into a misleading equal-length fusion tensor in the MechaSight path. Protocol A
does retain its required fixed shapes.

The JSONL processed manifest contains machine, modalities, all raw references, original segment,
parent recording, globally unique window id, physical bounds, protocol, regime, split, domain,
operating/background conditions, anomaly label, original/transformed sampling metadata, and
configuration id. Training data consumers must select their split before loading arrays; labels in
the traceability artifact do not authorise supervised fitting.

## Official AUC compatibility

`official_pairwise_auc` implements the pinned upstream rule exactly: a pair is correct only when
`normal_score < anomaly_score`; exact ties receive zero. No AUC was calculated in Phase 2.

## Phase-2.1 evaluation freeze

Protocol B uses shared source-normal validation parents for both regimes; the released
target-adaptation allocation is train-only in T and unused in S, and blocked siblings are unused in
both. Primary threshold-free metrics are AUROC and AUPRC. The operating
threshold is the source-validation normal segment-score q90 with quantile method `higher`, and the
decision rule is strict `score > threshold`. Protocol B segment scores are medians and retain every
contributing window id. RoboticArm uncertainty uses a paired parent-recording cluster bootstrap;
BrushlessMotor uses recording-level resampling with its temporal confound disclosed. The complete
machine-readable policy is `configs/evaluation/protocol_b_few_shot_adaptation_v3.json`, which carries the Phase-2.1 metric, threshold, aggregation and uncertainty sections forward unchanged.
