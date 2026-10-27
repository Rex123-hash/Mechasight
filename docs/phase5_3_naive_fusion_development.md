# Multi-Sensor Fusion Development — Matched Naive Fusion (Phase 5.3)

<!-- Legacy roadmap name: Phase 5.3 -->

**Status:** development only. 24 new L1 fits, 28 reused L1 references, 208 L2 source-validation
calibrations, statistical procedures tested on synthetic data. No protected input, sealed test
score or test label read; no protected inference; no test look; ledger 196 → 196. Design:
`docs/phase5_fusion_design_v2.md`, `configs/phase5/fusion_design_v2.json` (unchanged).

Development numbers below (validation loss, q90 thresholds) are fit diagnostics on normal
source-validation data. They are not detection results and rank nothing.

## 1. Entry check — Phase-3.4 historical replay (corrects the Phase-5.2 wording)

Exactly **one** test reads real persisted Phase-3.4 scores:
`tests/test_phase34_impl1_correction.py::test_48_and_60_parent_computations_reproduce_independently_from_persisted_scores`.

- **Path:** `fx.analyse_corrected([2,3,4,5])` and `([1..5])` → `load_persisted_scores` →
  `pd.read_csv` of the 10 hash-verified tables in `reports/phase3_4_results/scores/`. The tables
  carry RoboticArm common-test segment scores **and their labels** (schema
  `reports/phase3_4c_freeze/score_schema.json`).
- **Runs** on every full-suite run (no skip marker), including every 5.0–5.2 validation run.
- **Origin:** added at `af646c5` (2026-09-12), the IMPL-1 correction, as the authoritative check
  that the committed corrected analyses reproduce from persisted artifacts.
- **Effect:** `analyse_corrected` is pure. It performs no inference and no write, and it appends
  no ledger row. The test only asserts equality with the committed analysis JSONs.
- **Class:** existing authorised historical replay of a closed analysis. It is not a new test
  look, not new inference and not a new scientific result.
- **Other Phase-3.4 tests that reach the reader** (`test_phase34_inc1_recovery::test_salvage…`,
  `test_phase34_score_exposure_gate::test_H…`) redirect `RESULTS_ROOT` to `tmp_path` and read
  synthetic fixtures only.
- **Phase-5 reach:** none. `src/fusion` may reference no sealed location (tripwire), and Phase-5
  code never imports the Phase-3.4 module.

**Corrected wording for Phase 5.2:** "No *new* protected read was performed. The full-suite runs
replayed one established Phase-3.4 reproduction test, which reads the 10 persisted Phase-3.4
score tables to recompute an already-published analysis." The earlier absolute "no protected
sealed scores were read" was inaccurate.

## 2. Status: PARTIAL — 14 of 24 L1 fits; 10 RoboticArm autoencoder fits blocked by host memory

| Work item | State |
|---|---|
| 208 L2 calibrations and fused q90 thresholds | **complete**, persisted, reproduce exactly |
| 4 L1 Mahalanobis fits | **complete**, reproduce exactly |
| 10 L1 dense-AE fits, BrushlessMotor | **complete**, reproduce exactly |
| 10 L1 dense-AE fits, RoboticArm | **not run — resource NO-GO** (section 4) |
| 28 reused Phase-3.2 L1 detectors | referenced, not refitted (section 6) |
| Statistical procedures | implemented; synthetic tests only |

### Completed L1 fits (development diagnostics on normal source validation; not detection results)

| Unit | Experiment id | Epochs (best) | Best val. window MSE | q90 | Reproduced |
|---|---|---:|---:|---:|:---:|
| RoboticArm|S|mahalanobis_shrinkage|0 | `p3-d6b20c84b9b48ce9915258aa` | — | — | 220.83 | exact |
| RoboticArm|T|mahalanobis_shrinkage|0 | `p3-c3ff10cf8f348a5c6e6104c1` | — | — | 212.818 | exact |
| BrushlessMotor|S|mahalanobis_shrinkage|0 | `p3-a1a176477b5a4abe33912d45` | — | — | 115.296 | exact |
| BrushlessMotor|S|dense_autoencoder|0 | `p3-2716c4ea337f275c128e16e4` | 28 (18) | 0.8068 | 0.816259 | exact |
| BrushlessMotor|S|dense_autoencoder|1 | `p3-c7104d98baeb6b312d168b2b` | 31 (21) | 0.8070 | 0.817393 | exact |
| BrushlessMotor|S|dense_autoencoder|2 | `p3-45a029d01ee6d3e165f82e65` | 30 (20) | 0.8068 | 0.816436 | exact |
| BrushlessMotor|S|dense_autoencoder|3 | `p3-04b940acc27c7439e1d32b24` | 28 (18) | 0.8070 | 0.816842 | exact |
| BrushlessMotor|S|dense_autoencoder|4 | `p3-4dfba9a4541ea9996153c604` | 27 (17) | 0.8068 | 0.816246 | exact |
| BrushlessMotor|T|mahalanobis_shrinkage|0 | `p3-739ec5209aa8b00089c479bd` | — | — | 109.525 | exact |
| BrushlessMotor|T|dense_autoencoder|0 | `p3-06ea507b93e119054b5c3ba4` | 30 (20) | 0.7985 | 0.806411 | exact |
| BrushlessMotor|T|dense_autoencoder|1 | `p3-0abdd30c4b8457111796fd1b` | 31 (21) | 0.7983 | 0.807336 | exact |
| BrushlessMotor|T|dense_autoencoder|2 | `p3-e0bd283a234f13ae699535ca` | 31 (21) | 0.7983 | 0.807002 | exact |
| BrushlessMotor|T|dense_autoencoder|3 | `p3-0f1210655594b046269dad9d` | 31 (21) | 0.7985 | 0.808735 | exact |
| BrushlessMotor|T|dense_autoencoder|4 | `p3-cc67d2432b24bfb11aa66f4d` | 28 (18) | 0.7980 | 0.807226 | exact |

All 14 bundles are bound to code `c6798967`, with torch threads 4 for every autoencoder fit, and standardisers bit-identical to Phase 4. The run log records 14 STARTED / 14 COMPLETED / 0 FAILED, plus 9 `BLOCKED_RESOURCE` gate refusals (BrushlessMotor T seeds 2–4 waited for memory; nothing was written for a refused attempt). The 10 RoboticArm AE units were never started.

## 3. What was implemented

- **`src/fusion/l1.py`** — the frozen Phase-4 single-modality recipe with the modality input
  replaced by all three. It reuses the Phase-4 cohort contracts, standardiser, model classes,
  evaluation gate, q90 rule, model store and result store unchanged; `src/experiment/phase4.py` is
  not modified. Hyperparameters equal the Phase-4 payload except the modality label and stage
  fields (tested).
  - **Mahalanobis:** the 98 Phase-3.1 features in their frozen sorted order, train-only feature
    z-score, LedoitWolf.
  - **Dense AE:** the [mic | acc | gyro] concatenation (2918 / 2936), the same architecture,
    optimiser, loss, schedule, early stopping and best-validation checkpoint. Its standardiser is
    fitted per modality block exactly as Phase 4 fitted it, and is **verified bit-identical to the
    stored Phase-4 standardisers before any artifact is written**. The fused training matrix is
    standardised in place with the same arithmetic as `standardize_float32` (tested bitwise) to
    avoid a second full-size copy.
  - Every bundle records the experiment id, code SHA, Phase-5 source hash, design-config hash,
    training/validation cohort and contract hashes, initial-state hash (AE), checkpoint history,
    validation scores, q90 threshold, model-manifest hash and `completion_status: COMPLETE`.
    Bundles are written atomically and exclusively, after the model, so a model directory without
    a bundle is an **incomplete fit**; the driver refuses to continue past one.
- **`src/fusion/l2.py`** — design v2 §3.3 exactly: robust z from source-validation median and IQR,
  mean over the subset in fixed mic/acc/gyro order, own q90 `higher` threshold per subset.
  Non-finite scores and IQR = 0 are refused, never repaired.
- **`src/fusion/statistics.py`** — healthy-target FPR and recall at a frozen threshold on the
  shared class-stratified parent-cluster draws with R8 seed averaging. It returns the frozen
  `BootstrapDistribution`, so paired contrasts reuse the Phase-3.2/Phase-4
  `paired_distribution_difference` unchanged. It also implements the v2 metric, operating-axis and
  joint categories.
- **`scripts/run_phase5_development.py`** — one fresh process per fit, clean tracked tree required,
  append-only run log, torch threads pinned to the Phase-4 value, pre-flight memory gate, ledger
  checked unchanged before and after every command.

## 4. RoboticArm memory profile and go/no-go

Both trials ran in a scratch directory with uncommitted code and 8 torch threads, so **neither is
an official fit**; both are excluded from every count.

| Trial (scratch) | Training windows | Epochs (best) | s / epoch | Wall | Peak working set | Minimum system available |
|---|---:|---:|---:|---:|---:|---:|
| BrushlessMotor S seed 0 | 57,987 | 28 (18) | 3.6 | 122 s | 1.81 GB | 0.21 GB |
| RoboticArm S seed 0 | 113,072 | 117 (107) | 33.3 | 3,924 s | 3.25 GB | **0.00 GB** |

- **Host:** 15.3 GB RAM, of which about 13 GB is held by desktop applications (browsers and other
  desktop applications); 2–3.6 GB was free during the work. The pagefile grew from
  1.5 to 2.2 GB during the RoboticArm trial.
- **Correctness of the RoboticArm path:** the trial completed (exit 0), stopped early, its
  standardiser equals the Phase-4 standardisers bit for bit, and its validation scores reproduce
  exactly (maximum difference 0.0).
- **Why NO-GO:** physical memory was exhausted (0.00 GB available). Per-epoch time was 9.2× the
  BrushlessMotor rate for 1.95× the data at the same thread count — consistent with
  memory-pressure degradation. The peak exceeds what is reliably free on this host, so ten more
  65-minute-plus fits would run with no memory margin. A free pagefile is not taken as evidence of
  safety.
- **Where the peak comes from (estimate):** the fused training matrix (1.32 GB float32); the
  frozen `DenseAutoencoder.score_windows`, which materialises the full validation reconstruction
  plus two float64 copies (about 1 GB) every epoch; and Python/torch overhead.
- **Gate enforced in code:** a fit starts only if available memory ≥ measured peak + 1.0 GB
  (RoboticArm AE 4.25 GB, BrushlessMotor AE 2.81 GB, Mahalanobis 1.5 GB).
- **Compatible alternatives, none applied** (each needs a decision; none changes data, seeds,
  epochs, capacity or dimensions):
  1. Free memory on this host (close desktop applications — the user's decision), then run the 10
     fits under the existing gate.
  2. Run them on a machine with ≥ 8 GB free RAM, same code commit and thread setting.
  3. A change-controlled, numerically identical memory reduction: chunked per-window validation
     scoring (row-independent arithmetic) and chunked direct HDF5 reads into the fused matrix.
     Both must be proven bitwise identical to the current path before use.

## 5. L2 — 208 calibrations

- **208 records** = 52 units × 4 subsets (mic+acc+gyro 52, acc+gyro 52, mic+gyro 52, mic+acc 52); ids unique; each maps to its v2 unit and component Phase-4 experiment ids.
- **Persisted** in `reports/phase5_development/l2/` at code `c6798967a34d0cf430782496b1a6139c66de1116`: `l2_calibration_records.json` (`c3778e9e42750ee2…`) and `l2_fused_validation_scores.csv` (32,448 rows, `19c82d9f83ec850a…`).
- **Reproduction:** `l2-verify` recomputes all 208 records and every fused score from the hash-verified Phase-4 validation scores. Both match exactly.
- **No zero IQR:** the smallest IQR is 0.01056, so U3 is resolved. Every threshold is its own subset's q90 `higher`, and the four thresholds of every unit are distinct (no inheritance).
- **In-sample flag rate:** each threshold flags 9.5–9.7 % of its own source-validation segments. This is descriptive, a property of in-sample q90 `higher`, and not a result.

## 6. Reused historical L1 (no refit)

The 28 Phase-3.2 all-channel classical detectors (statistical reference 4, Isolation Forest 20,
OCSVM 4) are the L1 arm for those families: `MATCHED_BY_RECORDED_IDENTITY` (design v2 §9.2).
Their configs and Phase-3.1 validation bundles are referenced by id from the v2 registry and
asserted to exist and match by unit (`tests/test_phase5_fusion_assembly.py`). Nothing is refitted
or rescored, and their sealed test scores are not opened.

## 7. Implementation clarifications (prospective, recorded before use)

- **IQR interpolation.** v2 defines robust z with IQR_v but names no quantile method. The
  calibration uses q75 − q25 with **NumPy `linear`** — NumPy's default and the (n−1)p
  interpolation of `numpy.percentile`. Chosen as the conventional IQR, fixed before any
  calibration was computed, and not selected by results. The tested behaviour is that
  [0, 1, 2, 3, 10, 11] gives q25 = 1.25 and q75 = 8.25 under `linear`, versus 2 and 10 under
  `higher`. The q90 threshold keeps its separate frozen `higher` method. Any alternative would be a
  design change.
- **Exact float reading.** pandas' default CSV float parser is not round-trip exact (it differs by
  up to 4.4e-16 on Phase-4 validation files). Persisted scores are therefore read with
  `float_precision="round_trip"`. Tested over all 156 Phase-4 validation files: every parsed value
  equals Python's correctly rounded `float(text)`, and each recomputed q90 equals the recorded
  Phase-4 threshold exactly. This is a read-path change only; no historical artifact is modified.
- **Torch threads.** All 60 Phase-4 AE fits recorded 4 intra-op threads (set through
  `MECHASIGHT_TORCH_THREADS`), while the library default is 8. Official Phase-5 AE fits pin 4, and
  `l1-fit-one` refuses any other value.

## 8. Statistical procedures (synthetic verification only)

Tests in `tests/test_phase5_statistics.py` cover:
- parent-cluster draws with unequal cluster sizes;
- exact point estimates, and identity weights reproducing them;
- R8 seed averaging;
- determinism and paired identity (and refusal on mismatched draws);
- a shifted arm detected with the right sign;
- invalid replicates when no healthy-target parent is drawn (counted, never imputed);
- reuse of the frozen ranking bootstrap;
- all v2 metric, operating and joint categories — including that a small non-significant point
  estimate with a wide interval is INDETERMINATE, not EQUIVALENT.

No real protected metric was computed.

## 9. Accounting

Ledger 196 → 196. Zero test looks, zero protected inference, zero protected score tables, zero
protected analyses. All new artifacts are validation-only: `test_evaluated: false` in every bundle
and the L2 manifest.
