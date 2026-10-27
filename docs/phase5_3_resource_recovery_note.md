# Multi-Sensor Fusion Development — Phase 5.3 Resource Recovery Note

<!-- Legacy roadmap name: Phase 5.3 (continuation) -->

**Status:** Phase 5.3 remains **PARTIAL**. 14 of 24 L1 fits and 208 of 208 L2 calibrations are
complete. The 10 RoboticArm dense-AE fits are still **BLOCKED_RESOURCE** and were not attempted.
Documentation only: no fit, no calibration, no protected read, ledger 196.

## 1. Recovery state (verified at HEAD `92ce279`)

- **Code identity:** code is byte-identical between `c679896` (where the 14 fits ran) and `92ce279`
  (`git diff c679896 92ce279 -- src scripts tests configs requirements.txt` is empty). Phase-5
  source hash `f1c01c96…`, identical to the one recorded in the L2 manifest.
- **Completed fits:** all 14 re-verified read-only. Every bundle file matches its manifest, every
  model directory passes `ModelStore.verify`, and validation re-scoring reproduces exactly
  (maximum difference 0.0). Run log: 14 STARTED, 14 COMPLETED, 0 FAILED, 9 BLOCKED_RESOURCE
  (pre-flight refusals that wrote nothing).
- **Thread policy:** `TORCH_THREADS = "4"` in `scripts/run_phase5_development.py`; `l1-fit-one`
  refuses any other `MECHASIGHT_TORCH_THREADS`.

## 2. Resource go/no-go

| Measure | Value |
|---|---|
| Physical RAM total | 15.3 GB |
| Physical RAM available | **1.68 GB** (89 % used) |
| Required for one RoboticArm AE fit | 3.25 GB measured peak + 1.00 GB margin = **4.25 GB** |
| Decision | **BLOCKED_RESOURCE** — no fit started, no retry loop, threshold unchanged |

The pagefile is not counted as available memory.

## 3. The ten outstanding experiment identities

An L1 experiment id is a content hash of the scientific identity, **including the code commit SHA
at fit time**, so the ten ids are fixed by the commit the fits run from. The derivation below was
validated by reproducing all 10 committed BrushlessMotor AE ids exactly.

| Unit | Id if fitted at `c679896` | Id if fitted at `92ce279` |
|---|---|---|
| RoboticArm S AE seed 0 | `p3-51457fd637258a1743fc33bc` | `p3-7f056119571af0747a5c026d` |
| RoboticArm S AE seed 1 | `p3-5a4264e9fd3c327c135398e9` | `p3-6fa236f3804bfed0cc888151` |
| RoboticArm S AE seed 2 | `p3-53302afb97d1920fdcb4186d` | `p3-872e9ab9391234348f4862c7` |
| RoboticArm S AE seed 3 | `p3-5ba80778ab7807b4473fb273` | `p3-b7f37c5b091cd4777d908162` |
| RoboticArm S AE seed 4 | `p3-cd6e62df559d5623e1077400` | `p3-7cbfedc132b215e7f0f55054` |
| RoboticArm T AE seed 0 | `p3-c110d68d9e0e9329f12f8b6b` | `p3-c793edb1fbbabba0c3f2284c` |
| RoboticArm T AE seed 1 | `p3-20077731b5fef52788a93132` | `p3-1d85f6890c686a3c9d532782` |
| RoboticArm T AE seed 2 | `p3-9a9bf14d7e9c9575745c7ec7` | `p3-51acca019634df3e45af6a8f` |
| RoboticArm T AE seed 3 | `p3-f0b254805b1cc20f6edd27de` | `p3-d28321a63da2350f86d9aed1` |
| RoboticArm T AE seed 4 | `p3-ba711add9a68d70cc3530c98` | `p3-6f714822be41b90840790554` |

**Open decision:** ids are fixed by the commit the fits run from. This note's own commit moves HEAD past `92ce279`, so fitting from the latest HEAD would give a **third** set of ids, matching neither column. Recommended: run the ten fits from a clean checkout of `c679896`, where the 14 completed fits ran, so all 24 share one code SHA. The left-hand column then applies. Code is identical at every commit since `c679896`, so this is a provenance choice, not a scientific one.

## 4. Round-trip float-verification files

`tests/test_phase5_l2.py::test_round_trip_parsing_reconstructs_every_persisted_phase4_validation_score_exactly`
reads `reports/phase4_pretest/experiments/<id>/validation_scores.csv` and the sibling
`threshold.json` for all 156 Phase-4 **pre-test bundles**.

- Every row is `cohort_role = source_validation`, `anomaly_label = normal`, `domain = source`.
- There is zero overlap with the sealed staging tables
  (`reports/phase4_protected/staging/scores/`).
- These are development artifacts, not protected test scores. No protected table was read and no
  protected calculation occurred.

## 5. Accounting of every touch of sealed protected test-score files during Phase 5

| When | What | Kind | New protected calculation? |
|---|---|---|---|
| Phase-5 entry review (before 5.0) | hashed all 156 Phase-4 staging score tables against the score manifest; printed the **first 3 rows of one** sealed Phase-4 table (`segment_id, parent_recording, domain, score, window_count`; no labels) to learn the schema | content read of 3 rows; integrity hashing | no |
| Phase-5 entry review | read the published Phase-4 aggregate analysis CSVs (`family_bootstrap_metrics.csv`, `healthy_target_false_positives_individual.csv`) to identify baselines | already-published aggregate results | no (the design states they are context only) |
| every full-suite run, 5.0–5.3 | `test_phase34_impl1_correction::test_48_and_60_parent_computations_reproduce_independently_from_persisted_scores` parses the 10 persisted Phase-3.4 score tables (with labels) and recomputes the committed corrected analysis | established historical replay (`af646c5`) | no (equality with committed JSON) |
| integrity checks, 5.0–5.3 | SHA-256 over the bytes of sealed files (`reports/phase4_protected`, `reports/phase3_2/experiments/*/scores.parquet`, `reports/phase3_4_results`) by the `protected_state` test fixture, the Phase-3.2.1 artifact-hash test and the before/after snapshot script | byte hashing, no parsing | no |

Correction: the Phase-5.1 and 5.2 reports said no protected score was read. That is true of the
*design and implementation work*, but not of the entry review (3 rows printed) or of the full-suite
runs (the Phase-3.4 replay). No new protected inference, derived test analysis or ledger entry
occurred at any point.

## 6. Scope of the replacement integrity baseline

The Phase-5.3 before-snapshot was lost when temporary files were cleared during the long trial.
It was replaced by:

| Class | Artifacts | Method | Strength |
|---|---|---|---|
| Tracked Git artifacts | ledger, all `reports/phase*` protected and historical outputs, `configs/` | `git status` / `git diff` against HEAD `25ad17d` (unchanged since the phase began), then the new 2,232-entry baseline | byte-exact via Git object hashes |
| Hash-manifested model files | 156 Phase-4 models in `data/interim/phase4/models` (and the 14 Phase-5 models) | `ModelStore.verify` against the manifest hashes recorded in the committed bundles | byte-exact |
| Ignored caches | `data/interim/phase4/spectral` (including the four common-evaluation caches), `data/interim/phase32_common_test` | modification time earlier than the phase start, then size + mtime in the new baseline | detects writes; not bit-rot, and does not content-hash the protected caches (deliberately, to avoid reading them) |
