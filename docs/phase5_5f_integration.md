# Phase 5.5F — Evaluation Integration on Synthetic and Permitted Development Inputs

**Status: UNREVIEWED CANDIDATE — NOT AUTHORIZED FOR REAL PROTECTED EVALUATION.**

Starting baseline: `b9eaa2f` (Phase 5.5E, GREEN in the independent targeted verification).

Not done in this stage:
- model training;
- any protected recording, cache, label or sealed score opened;
- protected inference or metrics;
- real authorization;
- Phase 5.6 or 5.7.

The real ledger stays byte-identical at 196 rows. The three components below plug into the
existing 5.5E executor; there is no second execution framework. Every read still goes
through executor-issued capabilities, the registry-derived historical policy, the
exposure-before-read checks, the OS admission coordinator and the analyse-once claim.

## A. Trusted label-free window index (`src/experiment/phase5_window_index.py`)

**Problem.** The Phase-4 spectral caches identify their rows only via `provenance_json`, one
serialized JSON value with `anomaly_label` for every window. Aligning scoring inputs through it
would materialise outcome labels.

**Schema `phase5-label-free-window-index-v1`.** Columns:
- `machine`, `regime`;
- `segment_id`, `parent_recording`, `domain`;
- `window_index`, `window_start_s`, `window_end_s`;
- `row` (the window's row in the spectral representation).

The index is hash-bound by a manifest recording:
- schema and columns;
- row and segment counts;
- cohort identity;
- content hash and file hash;
- representation (source cache, its hash, per-modality input dimensions).

Any column containing `label`, `anomaly`, `fault`, `outcome`, … is refused.

**Construction, label-free by design.**
- Segment, parent and domain come from an explicit column projection of the frozen split
  manifest (`usecols`, with no outcome column parsed).
- Window bounds come from the raw sensor **timestamp column only**, chosen from the Parquet
  schema. They pass through the frozen value-independent startup transforms, the frozen
  `common_coverage` and the frozen `window_bounds`, in the pipeline's iteration order
  (machine, parent, segment).
- No cropping, window length, stride or ordering is re-invented.

**Evidence.**
- **Exact row order:** for all four development source-validation cells (40,380 windows), the
  index equals the cache's own row order exactly: segment, window index, parent and domain.
  The cache provenance was read only for this permitted verification.
- **Exact row contents:** the fixture copies are byte-identical to the cache rows for every
  modality.
- **Deterministic:** building twice, from reordered inputs, gives identical results.
- **Rejected by `validate_index`:**
  - another cohort;
  - altered content;
  - missing or duplicated windows;
  - another cell;
  - an injected label column;
  - misordered windows, even with a recomputed hash.
- **No label reads:** a trap shows the scoring inputs are built without ever requesting
  `provenance_json` and with label-free CSV projections.
- **Refusals:** non-monotonic timestamps and non-development cohorts.

**Disabled.** `build_protected_window_index` always refuses. The future authorized operation it
stands for needs both of the following:
1. an authorized, ledger-recorded read of the protected common-evaluation raw *timestamp*
   columns;
2. a pre-registered label-free resolver from frozen segment identity to raw file paths. The
   release-metadata catalog loader also parses outcome columns.

## B. Frozen-model inference adapter (`src/experiment/phase5_inference.py`)

`FrozenL1Adapter` is the executor's predictor for the 24 planned `new_L1_inference` units.

**Identity.** The detector must equal its frozen-registry record. The model directory must be
the registry's. `ModelStore.load` re-hashes every model and standardizer file against the
registry's manifest hash and checks the stored model id.

**Mahalanobis.**
- The input columns must be exactly identity, `window_count` and the frozen 98 features in
  registry order.
- Score: `model.score(standardizer.transform(X))`.

**Dense autoencoder.**
- Inputs are float32 spectra, concatenated mic|acc|gyro with the frozen block widths.
- They are standardized with the frozen L1 routine (`_standardize_in_place`).
- `model.score_windows` gives window scores, and the frozen Protocol-B median
  (`aggregate_window_scores_by_segment`) gives segment scores.

**Never.** The adapter never fits or updates a standardizer, never changes parameters and never
recalibrates thresholds. Input row and window order is preserved, and every call is recorded
for coverage accounting.

**Evidence.**
- **Real models:** all **24** real frozen L1 models (4 Mahalanobis, 20 dense AE) score the 24
  permitted development cohort segments through the executor. Their scores equal the frozen
  bundles' recorded validation scores **exactly** (max |Δ| = 0), and window counts match.
- **Refusals:**
  - an unplanned identity;
  - an altered detector record;
  - a bad model-manifest hash;
  - a classical layout missing a feature or carrying an extra (poison) column;
  - wrong spectral widths or dtype.
- **No hidden state change:** after scoring, the in-memory models and standardizers equal
  freshly loaded copies, model files re-verify, and registry thresholds are unchanged.

## C. Append-only JSONL ledger adapter (`src/experiment/phase5_ledger.py`)

`JsonlLedger` writes the project's established format only:
- `TestLookRecord` rows, appended with the existing `TestLookLedger.append_many_atomic`;
- one `batch_authorization.json` written **before** each append, giving batch id, start/end
  counts, look count and code SHA;
- `batch=<id>` in every row's reason;
- history checked with the **unmodified** `verify_ledger_history`, which runs before every read
  as part of the exposure check.

It runs on a temporary copy of the closed 196-row history; the real ledger path is refused as a
backend. The 5.5B–5.5E JSON ledger remains the default backend for synthetic fixtures.

**Evidence** (each with the real verifier):

| Case | Result |
|---|---|
| Clean run | 196 → **428**, one authorized batch |
| Authorization written, crash before the append | ledger stays 196; recovery appends 232 under original identities → 428 |
| Crash after exposure | recovery appends 232 `-r1` looks → 660, two authorized batches |
| Crash after the first read | → 660, two authorized batches |
| Duplicate or unauthorized scoring | refused |
| Exposure before every read | the verifier sees 428 rows at every projected read |
| Torn (interrupted) ledger append | fails closed with zero reads: verifier reports "partial append", recovery refuses, human incident required |
| Concurrent attempt | refused |

Identities are always globally unique.

## End-to-end integration

The chain is:

> label-free index → frozen-model adapter → executor → L2 derivation → JSONL ledger →
> completion gate → analyse-once

It runs on permitted development inputs, 12 source-validation segments per machine:
- **L1:** all 24 new L1 units use real models (0 stubbed).
- **L2:** all 208 L2 tables are derived from the real Phase-4 single-modality validation scores
  and equal the persisted L2 fused validation scores exactly (2,496 rows).
- **Reused L1:** the 28 reused L1 comparators (real Phase-3.2 validation scores) are readable
  only in analysis.
- **Thresholds:** frozen and unchanged.
- **Ledger:** 196 → 428, verified.
- **Analysis:** analyse-once (140 contrasts, 420 intervals) runs only after completion.

**Labels are SYNTHETIC**, because development validation contains no faults. All development
segments are source-domain, so healthy-target FPR is undefined there. Statistics are engineering
evidence only, not fault-detection performance. The original 232-unit synthetic-stub
demonstration is also re-run and recorded separately as stub coverage.

## Remaining prerequisites for real protected evaluation (not done here)

1. An authorized construction of the common-evaluation label-free window index (protected raw
   timestamps) with a label-free raw-path resolver.
2. Common-evaluation inputs in the frozen layouts:
   - classical features via projected columns;
   - spectral numeric rows located by that index, without reading `provenance_json`.
3. An executor read path for real protected files: a protected-input adapter replacing the
   temp-only `Sandbox`, under the existing capability checks.
4. The real ledger backend: append to the real ledger only under a human batch authorization
   committed in `reports/`.
5. Independent review of this candidate and human authorization.
