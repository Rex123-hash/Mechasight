# Phase 2 — leakage-safe preprocessing and data pipeline

## STATUS

**COMPLETE — awaiting independent review.** No anomaly detector was trained; Phase 3 and UI work
were not started.

## SUMMARY

Implemented separate official-reproduction and MechaSight preprocessing paths, froze a
parent-recording-disjoint split, added native multi-rate timestamp alignment, conditional regular
grids for frequency features, startup/warm-up transformations, leakage-safe normalisation,
classical features, ragged streaming caches, traceability manifests, and 27 new Phase-2 tests.

## UPSTREAM BASELINE COMMIT PINNED

Repository `https://github.com/augustif/IMAD-DS_baseline`, commit
`592bcbbed58893dc26bbab3240c5d7a3105a1a7f`, retrieved 2026-08-22. Inspected
`preprocess.ipynb`, `baseline.ipynb`, `model.py`, and `utilities.py`. Exact file hashes, DOI
`10.5281/zenodo.12665499`, null Zenodo version field, and verified archive MD5s are in
`upstream_baseline_provenance.json`.

## SAMPLING-RATE DOCUMENTATION UPDATE

The documented `INTERNAL_FREQ_FINE` equation gives ~+40 for RoboticArm and ~+47 for
BrushlessMotor. This closely reproduces the released timestamp rates, but the archived release
lacks acquisition-time register values, so the mechanism is strongly consistent rather than
proven. The nominal 6.7 kHz ODR is not called wrong.

## FINAL PROTOCOL-B SPLIT DESIGN

RoboticArm anomalies are test-only. Normal parents are stratified by domain/operating condition
with seeded SHA-256 ordering (1995), approximately 80/10/10 subject to ≥1 validation and ≥2 test
parents per cell. Regime S puts all target normals in test. BrushlessMotor keeps the official test
set and carves grouped 90/10 train/validation from released training normals.

| Machine | Regime | Split | Recordings | Segments |
|---|---:|---|---:|---:|
| RoboticArm | S | train / validation / test | 44 / 6 / 51 | 1528 / 186 / 589 |
| RoboticArm | T | train / validation / test | 59 / 9 / 33 | 1610 / 212 / 481 |
| BrushlessMotor | S | train / validation / test | 1137 / 126 / 330 | 1137 / 126 / 330 |
| BrushlessMotor | T | train / validation / test | 1150 / 131 / 312 | 1150 / 131 / 312 |

RoboticArm Regime-T target cells retain two normal test parents each: weight25 14 segments,
weight30 13, weight35 8. Candidate statistics for seeds 1995/20240821/20260821 were frozen before
adoption. Segment balance was not optimised because the physical recording is the independent unit.
Full domain, condition, environment, class, split, recording, and segment counts are in
`protocol_b_split_statistics.csv`.

## LEAKAGE ASSERTION RESULTS

All integration assertions passed: zero parents cross splits; zero segment/window siblings cross
splits; all anomalies are test-only; Regime-S target normals are unseen; seed reproduction is exact;
and every RoboticArm normal test condition has at least two parent recordings. See
`phase2_validation.json`.

## PREPROCESSING / WINDOWING / ALIGNMENT / RESAMPLING

Protocol A retains the upstream 100 ms, 7063, right-inclusive crop and pad/truncate path. Protocol B
uses configurable length/stride and half-open physical windows. Alignment uses timestamp
intersection after startup crops, never raw index. Native arrays remain native; linear regular-grid
interpolation is invoked only for frequency features at modality-specific measured rates. No global
resampling or window-size search was performed.

## STARTUP / WARM-UP / NORMALISATION

MechaSight drops one initial BM accelerometer sample and zero RA samples; Protocol A drops neither.
Both retain configurable 35 ms gyro warm-up. `std_window`, `none`, and train-only fitted channel
z-scoring are implemented with fit provenance and non-train fit rejection.

## FEATURES AND TEMPORAL REPRESENTATIONS

Implemented ten focused time features and dominant frequency, spectral energy, centroid, spread,
and configurable band energy. Native modality arrays use HDF5 ragged value/timestamp stores with
offsets and lengths; no equal-length cross-modality tensor or fusion design is imposed.

## CACHE / STORAGE / MANIFEST

Cache identity is the canonical configuration hash plus protocol/regime/machine/split path. HDF5
creation is exclusive and incompatible headers fail. The JSONL manifest traces every prediction
unit to machine, modalities, all raw files, segment, parent, physical bounds, protocol, regime,
split, conditions, label, sampling transformations, and configuration id.

## TESTS

`python -m pytest -q` → **104 passed** (77 existing + 27 Phase-2 tests). Coverage includes every
required leakage, alignment, window, irregular-sampling, resampling, startup, warm-up,
normalisation-isolation, synthetic-feature, cache-invalidation, manifest, and official-tie case.

## ACTUAL GENERATED DATA COUNTS

Four one-segment smoke caches were built (not the full corpus): MechaSight-T RoboticArm 74 windows,
MechaSight-T BrushlessMotor 51, Protocol-A RoboticArm 74, and Protocol-A BrushlessMotor 51 —
**4 segments / 250 windows total**. The MechaSight BM build logged 1 accelerometer sample removed;
all four logged gyro warm-up removal. Full
100-ms Protocol-B metadata implies 251,665 windows per regime, but those arrays were not materialised.

## MEMORY / DISK ESTIMATES

Streaming keeps one segment/window resident; complete-corpus RAM is unnecessary. Smoke builds imply
~54–57 kB/window including cache, manifest, and features: approximately **13.2 GiB per complete
Protocol-B regime**, of which ~11.7 GiB is HDF5. Regimes/protocols are separate and therefore add.

## ASSUMPTIONS / CHALLENGES / OPEN PHASE-3 QUESTIONS

- The 100 ms experimental smoke value is not optimal; window length/stride must be selected using
  validation only, with a small declared search.
- Parent independence takes precedence over matching official segment counts; long recordings are
  intentionally not given extra split-selection weight.
- BM interpolation cannot recover information or remove its class/session confound.
- Choose between amplitude-preserving/train-zscore and per-window standardisation on validation;
  decide feature bands from mechanics and sampling limits before test evaluation.
- Quantify Protocol-A strict-tie AUC versus a standard-library AUROC only after Phase 3 begins.
- `_corr` remains unresolved and inert.

## FILES CREATED / MODIFIED

Core code is under `src/preprocessing/`, `src/features/extraction.py`, and `src/evaluation/auc.py`;
entry points are `scripts/build_phase2_splits.py`, `scripts/preprocess_imadds.py`, and
`scripts/validate_phase2.py`; named configs are under `configs/preprocessing/`; tests are
`tests/test_phase2_*.py`; split/provenance/validation artifacts are in this directory. README,
requirements, evaluation protocol, source verification, research definition, audit summary, and
assumptions documentation were updated.
