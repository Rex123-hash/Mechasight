# Multi-Sensor Fusion Development — Data Foundation (Phase 5.2)

<!-- Legacy roadmap name: Phase 5.2 -->

**Status:** infrastructure only. No model fit, no L2 calibration or fusion score, no L3 logic, no
protected read, no test look. The authoritative design is unchanged:
`docs/phase5_fusion_design_v2.md` and `configs/phase5/fusion_design_v2.json` (SHA-256 pinned in
`src/fusion/assembly.py`).

## 1. Sealed test-score tables (Task A)

| Location | Sanctioned reader | Status |
|---|---|---|
| `reports/phase4_protected/staging/scores/<id>.csv` | `ScoreStaging.read_scores` | **guarded**: refuses outside an `authorized_protected_read` scope before the file is opened |
| `reports/phase3_2/experiments/<id>/scores.parquet` | `ResultStore.read_scores` (new; previously no reader existed) | **guarded**; inside a scope it also verifies the bundle-manifest hash and the declared test-cohort hash |
| `reports/phase3_4_results/scores/<id>.csv` | `focal_toggle_execution.load_persisted_scores` | **not guarded** — see below |

The contract lives in `src/preprocessing/protected_access.py`: `SEALED_SCORE_TABLES` and
`require_sealed_score_read`, built on the Phase-5.0 scope mechanism.

**Authorised paths are preserved.** The analyse-once step of `scripts/run_phase4_protected.py`
now runs its whole analysis, not only the label read, inside its existing scope. A future Phase-5
protected script reads the sealed tables only inside its own scope, after the exposure is
recorded (design v2 §12).

**Why Phase 3.4 is unguarded.** Its reader belongs to a closed analysis.
`test_four_fold_analysis_function_is_unchanged_from_64694e6` freezes that function's source, so
adding a guard would reopen closed Phase 3 without a verified defect. Phase 5 does not use these
tables. They are covered by the tripwire below. (The suite's own Phase-3.4 reproduction tests
read them, as they did before.)

**What the guard does and does not do.** It covers the repository's reader APIs.
`tests/test_phase5_sealed_score_guard.py` additionally freezes every `src/` and `scripts/` file
that references a sealed score location or reader, and requires that `src/fusion/` references
none. So new development code cannot add a reader unreviewed. It cannot stop an arbitrary program
or an interactive session from opening the files.

## 2. Fusion data assembly (Task B) — `src/fusion/assembly.py`

Every entry point refuses any role other than `training` / `source_validation` before a path is
built. Assembly never fills, drops, reorders or truncates records to make inputs fit; a mismatch
raises `FusionAssemblyError`.

| Function | Output | Contract checks |
|---|---|---|
| `load_design_registry()` | the 52 v2 units | design config SHA-256 and status unchanged; unique unit keys |
| `assemble_classical_fusion(feature_root, machine, regime, role)` | `X` (segments × 98, float64), columns, modality of each column, segment table, cohort hash | existing hash/row/cohort checks of `load_feature_artifact`; exactly 98 unique columns in the **frozen globally sorted Phase-3.1 order** (acc 42, gyro 42, mic 14 — the order the reused Phase-3.2 all-channel detectors were fitted on); blocks equal `modality_feature_columns`; machine/regime/split; allowed data pools; normal-only; finite |
| `assemble_spectral_fusion(cache_root, machine, regime, role)` | `X` (windows × 2918/2936, float32), blocks, shared window provenance, cohort hash | dims from the frozen preprocessing config = cache attributes = design (RoboticArm 800/1059/1059, BrushlessMotor 800/1068/1068); concatenation order **mic, acc, gyro**; each modality's provenance frame (window id, index, segment, parent, domain, pool, split, label) must equal mic's row for row; window count, unique window ids, contiguous segments indexed 0..k−1 with one parent and domain; segment cohort hash |
| `assemble_unit_validation_scores(unit, pretest_root)` | one row per source-validation segment: `segment_id, parent_recording, domain, mic, acc, gyro` | each frozen bundle verified by `load_frozen_bundle` and must belong to the unit and modality; direction `higher_is_more_anomalous`; normal source-validation only; unique, finite; validation cohort hash equals bundle provenance; identical segment/parent/domain across modalities; joined on segment id, sorted |

**Allowed data pools (frozen split v3):**
- training S: `source_training`;
- training T: `source_training` + `target_adaptation` (RoboticArm 27, BrushlessMotor 18);
- source validation: `source_validation`.

**Real-data verification (development artifacts only, all eight machine × regime × role
combinations):**

| Machine | Regime | Role | Classical | Spectral |
|---|---|---|---|---|
| RoboticArm | S | training | 1528 × 98 | 113,072 × 2918 |
| RoboticArm | S | source_validation | 186 × 98 | 13,764 × 2918 |
| RoboticArm | T | training | 1555 × 98 | 115,070 × 2918 |
| RoboticArm | T | source_validation | 186 × 98 | 13,764 × 2918 |
| BrushlessMotor | S | training | 1137 × 98 | 57,987 × 2936 |
| BrushlessMotor | S | source_validation | 126 × 98 | 6,426 × 2936 |
| BrushlessMotor | T | training | 1155 × 98 | 58,905 × 2936 |
| BrushlessMotor | T | source_validation | 126 × 98 | 6,426 × 2936 |

- Classical and spectral segment sets are equal in every combination.
- The classical and spectral cohort hashes equal the Phase-4 bundles' training/validation cohort
  hashes.
- All 52 L2 units assemble (RoboticArm 186, BrushlessMotor 126 segments each).

## 3. Future L3 integration contract (no L3 logic implemented)

Positions are relative to the existing pipeline in `src/preprocessing/signals.py` and
`pipeline.py`. The common-window algorithm is **not** changed.

| Step | Existing code | Integrity check that must sit here (5.4) |
|---|---|---|
| 1 raw loading | `read_sensor_parquet` (raises on non-increasing timestamps) | an absent/unreadable stream is a D failure; the loader's timestamp exception must be caught and turned into a G3 verdict, not left as an unhandled error |
| 2 timestamps | — | **G3** per modality (finite, strictly increasing, coverage ≥ 0.90 of nominal) **before** any transform; a failure → `ABSTAIN_ALIGNMENT_FAILURE` (v1) |
| 3 startup transforms | `apply_startup_transformations` (value-independent) | none |
| 4 common windows | `common_coverage`, `window_bounds`, `iter_aligned_windows` (all three modalities, timestamps only) | none; unchanged, so value failures leave the grid identical |
| 5 raw values | — | **G1/G2** per modality on raw samples inside the common coverage, **before** step 6. The failed modality is dropped here: `resample_regular` (`np.interp`), normalisation and feature extraction would otherwise propagate NaN into its features, and the existing cache writers refuse non-finite vectors for the whole window |
| 6 interpolation / features / spectra | `resample_regular`, `_normalize`, feature and `log_power_spectrum_input` per modality | surviving modalities only, with the frozen per-modality functions; must not go through the all-modality cache builders |
| 7 modality scoring | frozen Phase-4 detectors | surviving modalities only; then calibration, survivor-subset threshold and integrity flag (5.3/5.4) |

Preserved limits (design v2): value failures (non-finite, flatline) may fall back to surviving
sensors; invalid timestamps and absent sensors abstain. The 1,992-run corruption suite belongs to
5.4.

## 4. Accounting

Ledger 196 → 196. Zero test looks, zero protected score tables, zero authorised batches, zero
model fits, zero protected analyses.
