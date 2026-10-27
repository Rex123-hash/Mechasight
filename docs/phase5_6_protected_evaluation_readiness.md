# Phase 5.6A — Protected-Evaluation Readiness Plan

**Status: PLAN ONLY.** Not done in this stage:
- authorization;
- protected access;
- real index construction;
- real inference;
- real ledger change.

The baseline is the accepted Phase-5.5 synthetic/development integration commit
`5da28f82b94cdafafe33ac235dcea08d16e31692` (5.5G, GREEN in targeted independent
verification). Its executor source identity is
`fa021550496b1dfb75fa5ae60f965dd10200e6882a91d59fabdc8036098022f5`.

Everything below is a design for future, separately authorized work. **Verdict: READY FOR
INTEGRATION IMPLEMENTATION. Not ready for protected evaluation** (section N).

---

## 0. Frozen contract this plan must not change

| Item | Frozen value / source |
|---|---|
| Scientific registry | 52 units (`reports/phase5_freeze_candidate/scientific_registry.json`), design V2 `2e33299d…` |
| Planned protected looks | **232** = 24 new L1 + 208 derived L2 (52 all-three + 156 pairs); `proposed_evaluation_manifest.json`, proposed batch `p5-candidate-batch-17e7da9e2a1a4de7abe6e2f3` |
| Ledger | real `reports/experiment_results/test_look_ledger.jsonl`: 196 rows, SHA-256 `f7558461…`; uninterrupted completion 196 → **428** |
| Reused without a new look | 156 Phase-4 single-modality score tables (feed L2) and 28 Phase-3.2 all-channel L1 tables (analysis-only comparators) |
| Protected cohorts | common evaluation, Protocol B v3: RoboticArm **466** segments (cohort hash `23c5b7e3…`), BrushlessMotor **312** (`e1b09c05…`), identical in Regimes S and T |
| Approved protected inputs | common-evaluation **spectral caches** and common-test **feature artifacts** (24 L1-new units only); sealed Phase-4 score tables (L2); labels in analyse-once only. **Raw common-test recordings: not read** (frozen accounting) |
| Thresholds / statistics | source-validation q90 `higher`; L2 robust-z with `linear` IQR; strict `score > threshold`; parent-aware class-stratified paired bootstrap, 10,000 draws, seed 1995, R8 seed averaging; frozen margins (AUROC/AP 0.02, hFPR 0.10, recall 0.05); nominal multiplicity with yardstick |

---

## A. Protected-data access architecture

The resolver chain is: scientific registry → frozen protected cohort → segment identity →
approved protected artifacts → label-free window index → exact model representations.

| Step | Trusted source | Mechanism |
|---|---|---|
| 1. Units | `scientific_registry.json` via `FrozenPackage.load()` (git-anchored at `d7ea8ac`) | the existing `verify_memory` before every permit |
| 2. Cohort identity | the frozen split manifest `reports/phase2_2/protocol_b_few_shot_adaptation_v3.csv` (raw SHA pinned in the registry), **label-free projection only** (`usecols` = machine, regime, split, data_pool, segment_id, parent_recording, file_domain) | the ordered segment-id hash must equal the frozen cohort hash in `proposed_evaluation_manifest.json`. A mismatch STOPs before any protected file is opened |
| 3. Machine/regime/segment/parent/domain | the projection above | every protected row is matched to it; nothing comes from a caller |
| 4. Artifact binding | a new SHA-pinned **protected-input manifest** (pattern of 5.5G `development_sources.json`) | per machine × regime it records: the spectral-cache and feature-artifact paths; their committed Phase-4 metadata JSON hashes; the content hashes those metadata files declare (`cache_sha256`, `artifact_sha256`); the stat identities from `integrity_inventory.json`; and the sealed Phase-4 score-table paths and hashes. It is built **without opening protected contents** (paths come from the frozen layout `spectral_cache_path(...,"common_evaluation")` / `common_test_artifact_paths`; hashes from committed metadata and score manifests) |
| 5. Substitution prevention | the same as `_validate_source`: lexical path equality with the pinned binding before any resolve; no links or junctions; stat identity before open; post-exposure content hash against the declared hash | |
| 6. Label isolation | feature artifact: pyarrow **column projection** of exactly identity + `window_count` + the frozen 98 features (never `read_parquet(path)` then drop, as the Phase-4 loader did); spectral cache: numeric datasets `mic`/`acc`/`gyro` only. `provenance_json` is never deserialized | |
| 7. Window index | see section D: derived from the feature artifact's `window_count`, no raw timestamps | |
| 8. Layouts | classical: identity + `window_count` + registry `data_provenance.feature_columns` (98, frozen order). Spectral: float32, mic\|acc\|gyro with registry `spectral_blocks` widths (RoboticArm 800/1059/1059 = 2918, BrushlessMotor 800/1068/1068 = 2936) | |
| 9. Alignment | index `row` ↔ cache row; all three modalities share the row; every score table must align with the cohort on (segment, parent, domain) via the existing `_aligned` | |
| 10. Integrity | content hashes of the cache and artifact recomputed **after exposure is recorded** and compared with the declared hashes; mismatch = TECHNICAL failure, STOP | |
| 11. Invalid input | any missing file, hash, cohort or row-count mismatch, non-finite value or alignment failure is a TECHNICAL failure (frozen error semantics: no protected integrity gate, no "sensor degradation" outcome). The unit stays unpersisted and recovery follows the existing audited path | |
| 12. Determinism | the index is a pure function of the projected feature columns; persisted with the index manifest (schema, cohort identity, content hash, source hashes) inside the batch staging area | |

**Synthetic/development-implementable:** everything above, including the resolver, manifest
builder, projections and validators, exercised on synthetic sandboxes and the development
source-validation artifacts.

**Requires future authorized protected access:**
- the post-exposure content hashes;
- the projected reads of the common-test feature artifacts;
- the numeric cache reads;
- the sealed score reads.

## D. Label-free protected window index

**Finding (development evidence, this phase).** The Phase-4 spectral caches store windows in
the pipeline's iteration order: machine, recording parent, segment, then window. So the
index is fully determined by each segment's `window_count`. On all four development
source-validation cells (40,380 windows), expanding the Phase-3.1 feature artifact's
`window_count` in (parent, segment) order equals the raw-timestamp index exactly in
segment, window index and parent. That timestamp index itself equals the cache order
exactly (5.5F).

**Proposed protected method (recommended):** schema `phase5-label-free-window-index-v2-count-derived`.
1. Take segment, parent and domain from the label-free projection of the frozen split manifest,
   checked against the frozen cohort hash.
2. Take `window_count` per segment from a column projection (`segment_id`, `parent_recording`,
   `domain`, `window_count`) of the approved common-test feature artifact. It must equal the
   cohort identity.
3. Expand in (machine, parent, segment) order to rows `0..N-1`, with `window_index` 0..k-1.
4. Cross-check without labels:
   - N equals the cache attribute `window_count` and the dataset row counts;
   - the segment set's ordered hash equals the cache metadata `segment_cohort_hash` and the
     artifact metadata `cohort_hash`;
   - per-modality widths equal the frozen blocks.

   Any disagreement is a TECHNICAL STOP.
5. Persist the index plus its hash-bound manifest in the staging area. Window start/end times
   are not derivable without timestamps, so they are omitted (a v2 schema, not v1).

This keeps the frozen accounting intact: no raw recording, no `provenance_json`, and only
artifacts already listed as protected inputs.

**Rejected alternatives:**
- raw protected timestamps (contradicts "raw not read", needs an accounting amendment);
- parsing `provenance_json` and dropping labels (forbidden pattern);
- rebuilding caches (re-reads raw data and bypasses the pinned caches).

`build_protected_window_index` stays disabled until implemented and authorized as above.

## E. Exposure accounting and look budget

Under the existing ledger protocol, a look is **one ledger row = one new detector evaluated
against protected outcomes**. Reads are not looks; they are protected exposures that must be
durably authorized before they happen.

| Operation | Look? | Exposure record before the read |
|---|---|---|
| Index construction (projected `window_count` columns, cache attributes and metadata) | no new look | requires the batch's 232 ledger rows + reservation + exposure record + an `INDEX_CONSTRUCTION_STARTED` attempt-log event. **Human decision H2.** |
| L1 feature/spectrum reads and frozen-model scoring | yes, 24 rows | the same batch exposure |
| L2 derivation: reads of 156 sealed Phase-4 tables | yes, 208 rows (one per derived table) | the same batch exposure; reads via `require_sealed_score_read` plus projection |
| Reused 28 Phase-3.2 + 156 Phase-4 tables in analysis | no new look (frozen) | analysis authorization + analyse-once claim |
| Labels | no look | analysis authorization + durable claim, after the completion gate |
| Failed read or interruption | no extra row by itself | the already-recorded exposure stands; the unit stays pending |
| Retry of an exposed, unpersisted unit | **+1 row** under `<id>-r<k>` | a new recovery batch authorization + `batch_authorization.json` + its own append (5.5D/E machinery) |
| Retry of a never-exposed unit | no extra row | it keeps its original id |

**Budget.**
- The initial authorization is 232 looks (196 → 428).
- Retries are **not** pre-authorized. Each recovery batch needs its own human authorization.
- Proposed hard cap: 428 + 232 = 660. Beyond that, a mandatory incident review comes before
  any further look.

**Ordering (reverses the Phase-4 precedent, which materialised inputs before the append).** Per
batch, in order:
1. reservation;
2. `batch_authorization.json`;
3. atomic append of all rows (`append_many_atomic`);
4. exposure record;
5. the `EXPOSURE_RECORDED` event;
6. only then any protected open: index, features, cache, sealed scores.

The unmodified `verify_ledger_history` runs before each read (5.5F behaviour).

## F. Real-ledger authorization design

The existing `JsonlLedger` backend is pointed at the real path only by a new, separately
reviewed **real-ledger mode**. That mode:
- requires the committed human batch authorization record under `reports/phase5_protected/`;
- requires `verify_ledger_history` at 196 (closed Phase-3/4 segments byte-identical) before the
  append;
- requires the batch id, counts (196 → 428), `experiment_count` 232 and the scoring code SHA to
  equal the authorization;
- writes rows with the frozen `TestLookRecord` schema and reason
  `phase5 fusion protected look; batch=<id>; original=<eid>; policy=…; score_all_then_analyse_once; no metric during scoring`.

After the batch closes it is registered as a new entry in `CLOSED_SEGMENTS` in the same commit
that records its result (existing convention). The real ledger is never a test target; tests
use temporary copies (5.5F).

## C/F. Human authorization contract (proposal, not an authorization)

**Scoring authorization** (token proposal: `I_AUTHORIZE_THE_232_LOOK_PHASE5_FUSION_BATCH`).

Recorded in `reports/phase5_protected/pre_scoring_identity_record.json` before any protected
open, with these checks:

| # | Required equality / condition |
|---|---|
| 1 | reviewed executable commit SHA = the independently reviewed implementation commit (to be named by the reviewer); clean tree; `verify_executable_candidate(<SHA>)` passes |
| 2 | scientific registry identity `489c6fd6…` (registry SHA in the proposed manifest) and design V2 `2e33299d…` |
| 3 | cohort hashes RoboticArm `23c5b7e3…`, BrushlessMotor `e1b09c05…`; row counts 466 / 312 |
| 4 | the 232 unit ids equal `proposed_evaluation_manifest.json.identity.planned_ids` |
| 5 | protected input scope: the pinned protected-input manifest (section A4), with nothing outside it |
| 6 | allowed reads: projected feature-artifact columns; numeric cache datasets; cache attributes/metadata; 156 sealed Phase-4 score tables (label-free columns). Labels, Phase-3.2 tables and `provenance_json` are forbidden in scoring |
| 7 | initial looks 232 (196 → 428); retries need separate authorization; cap 660 |
| 8 | index construction is inside the scoring authorization, after the append (decision H2) |
| 9 | outputs under `reports/phase5_protected/` (staging, attempt log, completion gate); models and caches read-only |
| 10 | expected results: 232 tables, each hash-recorded; score manifest; completion gate GREEN; ledger 428 |
| 11 | stop conditions: any identity/hash/cohort mismatch, ledger verification failure, partial ledger append, interrupted analysis, live-owner conflict, or retry cap reached → STOP and file an incident report; no automatic continuation |

**Analysis authorization** is a separate token (proposal:
`I_AUTHORIZE_THE_PHASE5_FUSION_ANALYSE_ONCE`). It is valid only after completion gate GREEN.
It permits:
- projected label columns of the common-test feature artifacts (identity + `anomaly_label`
  only);
- the 28 Phase-3.2 and 156 Phase-4 comparator tables;
- exactly one analyse-once.

An interrupted analysis is a human incident; it is never retried automatically.

## H. Execution runbook (future; not executed)

| Stage | Preconditions | Allowed inputs | Outputs | Failure → action | New human authorization? |
|---|---|---|---|---|---|
| **0 Preflight** | reviewed SHA checked out, clean tree; `FrozenPackage.load()`; `verify_executable_candidate`; `verify_ledger_history` = 196 and byte hash; protected-input manifest pin; stat identities; free RAM ≥ measured AE peak + 1 GB; torch threads 4 | metadata, stat, committed manifests only | preflight record | any mismatch → STOP, no access | — |
| **1 Authorization** | stage 0 GREEN | the human token in chat | `pre_scoring_identity_record.json`; batch id bound to scoring HEAD | token or identity mismatch → STOP | **yes (scoring token)** |
| **2 Exposure + label-free index** | stage 1 | after reservation, `batch_authorization.json`, the 232-row append and the exposure record: projected `window_count` columns, cache attributes/metadata, post-exposure content hashes | index + manifest; `INDEX_CONSTRUCTED` event | hash/count/cohort mismatch → TECHNICAL STOP (exposure stands; recovery under new authorization) | no |
| **3 Frozen L1 inference (24)** | index valid | projected 98 features; numeric cache rows | 24 tables with `frozen_L1_inference` lineage | technical failure → unit pending, attempt log; stop the batch | no (recovery needs yes) |
| **4 L2 derivation (208)** | stage 3 complete or concurrent | 156 sealed Phase-4 tables (label-free projection) and frozen calibrations | 208 tables with `derived_L2_scoring` lineage, component hashes | alignment/IQR/finite failure → TECHNICAL STOP | no |
| **5 Completion gate** | all 232 persisted | staging, ledger, attempt log | `completion_gate.json` (ids exactly once, hashes, cohort/row counts 466/312, ledger 428, single batch, finite label-free scores, provenance lineage, metrics not computed) | any check red → no analysis; incident | no |
| **6 Analyse-once** | gate GREEN | projected labels; 28 + 156 comparator tables; the 232 tables | metrics, 140 primary contrasts / 420 intervals, S1–S6, yardsticks | interruption → human incident; never retried | **yes (analysis token)** |
| **7 Publication** | analysis complete | analysis outputs | results with intervals, lineage, hashes, limitations, ledger closed-segment registration, reproducibility record | — | review before publication |

Recovery and retry use only the 5.5D/E machinery:
- coordinator lock;
- admission ids;
- quarantine;
- `-r<k>` identities;
- a new authorization per recovery batch.

## I. Remaining implementation gaps

| Gap | Reuse | Smallest missing piece | Files | Synthetic/dev tests | Protected access needed? | Done when |
|---|---|---|---|---|---|---|
| **A. Protected label-free resolver + count-derived index** | 5.5G `development_sources.json` pattern, `_validate_source`, `validate_index`, `label_free_cohort` | protected-input manifest builder (metadata/stat only); `build_count_derived_index` (v2 schema); cache/artifact cross-checks; `build_protected_window_index` wired to require a protected capability | `phase5_window_index.py`, new `reports/phase5_protected_inputs.json` (pinned) | count-derived equals timestamp index on all dev cells (already shown); mismatch/hash/cohort/order rejection; no `provenance_json` or label column touched; refusal without a capability | **no** to build and test; yes to run | dev equality + adversarial suite green; zero protected opens in tests |
| **B. Protected-input adapter in the executor** | `ProjectedInputs` (dev schema), capability verifier, `FrozenL1Adapter` | a `PROTECTED_SCHEMA` reader that resolves paths from the pinned manifest instead of a temp `Sandbox`; post-exposure content-hash check; sealed-score projection through `require_sealed_score_read` | `phase5_inputs.py`, `phase5_execution.py` | the full 5.5F integration re-run through the new schema on a **development mirror** (dev artifacts registered as if protected); forged capability / caller path / pre-exposure read refusals | no | the integration reproduces 24/208 exactly through the protected code path on dev data |
| **C. Real-ledger batch mode** | `JsonlLedger`, `verify_ledger_history`, Phase-4 authorization schema | real-path mode gated by the committed authorization record; token check; closed-segment registration helper | `phase5_ledger.py`, new `scripts/run_phase5_protected.py` | on a temporary copy: 196 → 428, retries, torn append, missing or mismatched authorization, wrong counts, a second batch without authorization | no | all refusals green; the real path is refused without a record |
| **D. Real-cohort representation mapping** | registry feature/block layouts; Phase-4 metadata | none beyond A/B: the protected artifacts are the same frozen representation as development | — | covered by A/B | — | — |
| **E. Final identity + dry run** | `verify_executable_candidate`, driver | a `--protected-dry-run` that executes stages 0–1 checks and the full chain on the development mirror, stopping before any protected open | driver, docs | dry-run evidence bound into a new executable candidate | no | dry run green; candidate verified at the reviewed SHA |

## J. Synthetic-only tests still required

All of the following must pass on synthetic or development data:
- the count-derived index (equality, ordering, count mismatch, cohort-hash mismatch, cache
  attribute disagreement, missing segment, forged `window_count`);
- protected-manifest pin tamper, path or link substitution, and missing files;
- projection traps proving labels and `provenance_json` are never requested;
- the post-exposure hash-mismatch STOP;
- the protected-schema reader refusing every read before exposure, and with forged
  capabilities;
- the real-ledger mode refusing without an authorization record and on count or SHA
  mismatch, plus all retry/torn-append cases on a copy;
- the end-to-end development mirror reproducing 24 L1 + 208 L2 exactly with
  `frozen_L1_inference` / `derived_L2_scoring` lineage;
- a completion gate requiring 466/312 rows, adapted to the mirror's cohort sizes;
- analyse-once separation.

## K. Scientific metric and comparison contract (unchanged)

- **P1:** L2(all three) − each single modality.
- **P2:** L1 − L2(all three).
- **P3:** each L2 pair − L2(all three).

Each of P1–P3 is computed per machine × regime × family. That gives 140 primary contrasts × 3
metrics (AUROC, healthy-target FPR, recall) = 420 nominal intervals.

Secondary comparisons:
- **S1:** L1 − singles.
- **S2:** pair − members.
- **S3:** fusion − the post-hoc best single modality, labelled post-hoc.
- **S4:** source/target slices of every primary contrast, using masked shared weights
  (accepted secondary clarification).
- **S5:** AP and pAUROC.
- **S6:** T − S.

Statistics:
- parent-aware class-stratified paired bootstrap, 10,000 draws, seed 1995, R8 seed
  averaging;
- strict `score > threshold` at source-validation q90;
- frozen margins;
- fewer than 9,000 valid replicates → INDETERMINATE;
- EQUIVALENT vs INDETERMINATE per the frozen joint categories;
- family-dependence reported, never averaged away;
- nominal multiplicity with the null-exclusion yardstick.

No method, threshold or margin changes.

## L. Known limitations (carried into every claim)

- **Claim scope:**
  - Phase-4 aggregate protected results were seen before this design, so Phase 5 is
    prospective in implementation but not a pristine confirmatory test.
  - Results are per machine and per regime.
- **BrushlessMotor:** results sit under the acquisition-time confound
  `brushless_motor_acquisition_time_class_confound_v1`, so a fusion gain may reflect session
  discrimination.
- **RoboticArm:** healthy-target FPR rests on 20 healthy target segments from 6 parents
  (LOW-PRECISION; U5 valid-replicate counts are reported by analyse-once).
- **L3:** it is not evaluated on protected data. The pair tables measure clean subset
  capability; they speak to post-failure capability only for value failures under B1–B5. B5
  is untestable.
- **No physical sensor-fault evidence** exists in the dataset; synthetic controls cannot show
  that genuine faults are never suppressed.
- **Stat-only integrity:** protected-input integrity before exposure rests on stat identities
  and committed declared hashes. Content hashes are verified only after exposure.
- **DIFFERENT ≠ FAULTY:** domain-shifted healthy behaviour is reported through healthy-target
  FPR and slices, never as faults.

## M. Explicit human decisions required before any protected read

- **H1 — Index method.** Approve the count-derived label-free index (recommended: no raw
  recording read, consistent with the frozen "raw not read" accounting) instead of a
  raw-timestamp index (which would need an accounting amendment).
- **H2 — Index exposure accounting.** Confirm that index construction is covered by the
  232-look scoring batch and runs only after the batch's ledger append and exposure record,
  with no separate ledger rows. The ledger records evaluated units, and
  `verify_ledger_history` requires `experiment_count` = rows.
- **H3 — Post-exposure content hashing.** Approve hashing the protected cache and artifact
  bytes inside the authorized scope (byte hashing, no deserialization) against the declared
  Phase-4 hashes.
- **H4 — Retry policy.** Each recovery batch needs a new authorization; hard cap 660 looks.
- **H5 — Analysis label source.** Labels come from a column projection of the common-test
  feature artifacts (as in Phase 4, but projected), under a separate analysis token.
- **H6 — Reviewer and commit.** Name the independent reviewer and the reviewed implementation
  SHA that the authorization will bind.
- **H7 — Inversion of the Phase-4 precedent.** Acknowledge that the protected-input
  materialisation order is reversed from Phase 4 (exposure first, then reads).

## N. Readiness verdict

**READY FOR INTEGRATION IMPLEMENTATION.** The design closes every data-access, accounting and
authorization question except the human decisions H1–H7. The remaining code (gaps A, B, C, E)
can be built and fully tested on synthetic and development data.

**NOT READY FOR PROTECTED EVALUATION.** That is blocked on:
1. implementing gaps A–C and E;
2. an independent review of that implementation;
3. decisions H1–H7;
4. explicit human scoring authorization.
