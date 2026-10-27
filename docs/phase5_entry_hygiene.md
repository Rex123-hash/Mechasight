# Multi-Sensor Fusion Development — Entry Hygiene (Phase 5.0)

<!-- Legacy roadmap name: Phase 5.0 -->

**Status:** repository hygiene only. No fusion design, feature fusion, score fusion,
reliability weighting, model fit, protected inference, protected analysis or test look.
**Branch:** `phase5-multisensor-fusion`, created from the verified Single-Modality
Benchmarking HEAD `af819f941dffd59652bab427050e91ab3658f048`. The
`phase4-single-modality-benchmark` branch is unchanged.

## 1. Why this stage exists

The Phase-5 entry review found the repository scientifically intact but not yet a
trustworthy base for a new protected stage:

1. **The suite was red at HEAD** (12 failed, 520 passed). Every failure was a
   pre-Phase-4 snapshot ("the ledger has 28 / 40 rows", "no staging manifest exists")
   that the legitimate 156-look Phase-4 append made false. The ledger itself was
   verified append-only: the 40-row Phase-3 ledger is its exact byte prefix.
2. **Protected inputs are now materialised.** Phase 4 built the common-evaluation
   spectral caches (`data/interim/phase4/spectral/*/*/common_evaluation.spectral.h5`);
   Phase 3.2 built the common-test feature artifacts
   (`data/interim/phase32_common_test/*/*/common_test.parquet`). Building them was
   refused outside a protected stage, but *reading* an existing one was not.
3. **One recorded hash no longer matches its file** (section 5).

## 2. Ledger history (H1)

`src/experiment/ledger_history.py` replaces per-stage row counts with one verifier,
`verify_ledger_history`. It proves the ledger is

- **closed history** — four stages, each a contiguous atomic append pinned by row
  range, byte hash, protocol, code SHA, timestamp and committed authorization record:

  | Stage | Rows | Protocol | Authorization record |
  |---|---|---|---|
  | Phase 3.2 first common test | 0–27 | `mechasight_experimental_protocol` | `reports/phase3_2/evaluation_state.json` |
  | Phase 3.3 Protocol-A reproduction | 28–29 | `official_baseline_reproduction` | `reports/phase3_3_protocol_a/protocol_a_ledger_authorization.json` |
  | Phase 3.4 focal-parent sibling toggle | 30–39 | `roboticarm_focal_parent_sibling_toggle_v1` | `reports/phase3_4_results/test_look_authorization.json` |
  | Single-Modality Benchmarking | 40–195 | `mechasight_experimental_protocol` | `reports/phase4_protected/batch_authorization.json` |

  Whole-prefix identities: 40 rows `b0ecbc7a…853020d3e` (equals the Step-D record and
  the `576b8a1` blob); 196 rows `f7558461…fce64de5f`.
- **plus authorised atomic appends** — every later row must carry `batch=<id>` in its
  `reason`, and exactly one committed `batch_authorization.json` under `reports/` must
  name that id with matching `ledger_start_count`, `ledger_end_count`,
  `experiment_count` and `code_git_sha`. This is the convention Phase 4 already used, so
  a properly authorised Phase-5 batch verifies **without editing the verifier**.

It also refuses partial writes, malformed rows and any experiment looked at twice. When
a stage closes, it is registered in `CLOSED_SEGMENTS` in the commit that records its
result.

**Phase-5 requirement:** a Phase-5 protected batch must write `batch=<id>` into every
ledger row and commit a `batch_authorization.json` with the five fields above, or the
suite goes red — which is the intended behaviour for an unauthorised look.

Each of the eleven failing ledger tests now asserts its historical claim with explicit
scope (e.g. "the 28 Phase-3.2 rows are rows 0–27, identified by batch and evaluator
SHA", not "there are 28 Protocol-B rows") plus the global verifier. Closed Phase-3
protocols are pinned globally (Protocol-A looks = 2; focal-toggle looks = 10;
superseded overlap-V1 looks = 0), which also closes the Phase-3.3.1 audit gap that no
test pinned the Protocol-A count. New adversarial tests:
`tests/test_phase5_ledger_history.py`.

## 3. Harness guard (H2)

`test_guard_refuses_real_protected_staging_root` asserted that no real staging manifest
existed — true until score-all ran. The guard itself always refused. The test now
snapshots every protected artifact, attempts the refused open, and requires the
snapshot to be identical: no ledger row, score table, manifest, claim, bundle or input
cache was created, removed or rewritten. The snapshot (`snapshot_protected_state` and
the `protected_state` fixture in `tests/conftest.py`) content-hashes committed
protected records and identifies the git-ignored input caches by size and mtime; its
sensitivity to edits, additions and removals is itself tested.

## 4. Read-side protected-data guard (H3)

`src/preprocessing/protected_access.py` is the single place that decides whether code
may read common-evaluation (test-role) contents. Outside
`authorized_protected_read(authority=…, purpose=…)` every repository reader raises
`ProtectedReadRefused`:

| Reader | Refuses when | Refuses before |
|---|---|---|
| `pipeline.iter_processed_windows` (raw recordings → windows) | `split == "test"` | the call returns (eager, not at first `next()`) |
| `spectral_cache.load_spectral_cache` | file is `common_evaluation.spectral.h5`, or `expected_split="test"` | the file is opened |
| | stored split attribute is `test` (renamed copy) | any dataset is read |
| `spectral_cache.load_spectral_cache_metadata` | `role="common_evaluation"` | the metadata/cache is read |
| `spectral_cache.build_protected_spectral_cache` | always | any input is touched |
| `cache.read_window` (Phase-2 ragged caches) | stored split attribute is `test` | any window is read |
| `phase32.prepare_common_test_artifact`, `phase32.load_common_test_artifact` | always | any input is touched |
| `phase4.load_label_free_test_features`, `phase4.load_label_free_test_windows` | always | any file is read |

**Authorized versus development reads.** A scope is opened only by a protected stage,
after its own authorization checks, and names its authority (batch id or the stage's
authorization identity) and purpose. Scopes do not nest, need a non-empty authority and
purpose, and close on exceptions. The two existing protected stages keep working:
`scripts/run_phase4_protected.py` opens a scope for score-all (authority: the authorised
batch id) and analyse-once (authority: `batch_authorization.json`), and
`scripts/run_phase32_first_test.py` opens one for its common-test materialisation. The
only change to those scripts is the wrapper; both already refuse to re-run before
reading any test input (Phase 4: the staging area exists; Phase 3.2: the ledger is not
empty).

**Tripwires.** Two repository tests freeze (a) every direct `h5py.File` / parquet read in
`src/` and `scripts/` and (b) every place that opens an authorized scope. New Phase-5
code that reads HDF5/parquet directly, or grants itself a scope, turns the suite red
until reviewed.

**Residual limit.** The guard stops accidental development reads through repository
code; it is not a security boundary against deliberate bypass (opening the files by
hand), and it does not cover other protocols' own test artifacts (Protocol-A windowed
test caches under `data/interim/protocol_a/`, read only inside their own protected
stages). Both are visible to the tripwire if new code reaches for them.

Tests: `tests/test_phase5_protected_read_guard.py` (refusal with the cache present,
refusal before open, renamed-copy refusal before any dataset read, real materialised
caches refused with protected state unchanged, train/validation reads unaffected,
authorized scope admits then closes). With `require_protected_read` disabled, 9 of
these 19 tests fail.

## 5. Score-manifest hash semantics

`reports/phase4_protected/completion_gate.json` and
`reports/phase4_protected/analysis/analysis_manifest.json` record
`score_manifest_sha256 = d0874952a5685ea8eb474b6fa0095e8bf1bf9d6ace4e71e58ef5477633ca5504`.
The current `reports/phase4_protected/staging/score_manifest.json` hashes to
`f96f78d22477e5eaae855ac83202f6713e29a0b54e810637c9c471b01effcf5a`.

- **What the recorded hash identifies:** the sealed score manifest exactly as score-all
  wrote it at `d8ccf55`, before analysis — the score set the gate verified and the
  analysis consumed.
- **Why the current file differs:** analyse-once (`94f74e4`) flipped one status field,
  `metrics_computed: false → true`. No other key changed.
- **Confirmed unchanged:** all 156 `tables` entries (experiment id, file, cohort hash,
  table SHA-256) are identical across the two revisions, and every one of the 156
  sealed score tables still hashes to its manifest entry.
- **Not a result defect.** No historical hash or Phase-4 artifact is modified.

**Phase-5 convention (recommended for the 5.1 freeze):** record both
`pre_analysis_score_manifest_sha256` and `post_analysis_score_manifest_sha256`, **and**
a `score_payload_sha256` — the canonical-JSON hash of the manifest with mutable status
fields (`metrics_computed`) removed. The payload hash is the identity verifiers compare;
the file hashes record the two lifecycle states.

## 6. Reviewer constraints carried into Phase 5.1 (not implemented here)

- **C1 — Multi-objective interpretation.** Do not freeze "fusion wins only if ranking and
  operating metrics both improve." Freeze an explicit multi-objective reading before any
  protected look: a ranking gain with an operating loss is a trade-off, and a large
  false-alarm reduction with preserved ranking can be a valuable result.
- **C2 — Directional neutrality.** RoboticArm accelerometer is already strong, so headroom
  is limited; the Phase-5 hypothesis stays directionally neutral (no prior that fusion
  will hurt).
- **C3 — Frozen per-sample inference versus transduction.** A weight
  `w_m(x) = f(x_m; frozen training/validation reference)`, computed per segment with
  every parameter and reference distribution frozen beforehand and no statistic
  estimated from the protected cohort, is frozen per-sample inference. It becomes
  transductive only if protected-cohort information calibrates, normalises, adapts, fits
  or otherwise changes the rule. 5.1 states this boundary and prefers a
  cohort-independent R2 design if workable.
- **C4 — Claim boundary.** Phase-4 aggregate protected results have been observed, so the
  benchmark has informed development. Phase 5 can have a frozen implementation,
  controlled test-look accounting and no post-hoc tuning from Phase-5 outputs, but it is
  not an untouched confirmatory test of the general fusion hypothesis.
- **C5 — Comparators.** Compare fusion against the individual mic, acc and gyro baselines,
  not only against the post-hoc family-matched best modality.
- **C6 — Derived protected analyses.** Applying a new score-fusion rule to the sealed
  Phase-4 common-evaluation score tables is a new protected analysis even without new
  inference; 5.1 defines its authorization, ledger/test-look semantics, artifact
  persistence, failure/retry rules and analysis provenance.
- **C7 — Exact experiment count.** The frozen design states the exact experiment matrix
  (machines × regimes × families × fusion arms × seeds × any degradation arms), never an
  approximation.
- **C8 — L1 family selection.** If feature-level fusion is not run for every family, the
  choice rests on architecture, applicability or a pre-declared engineering constraint,
  never on favourable Phase-4 protected performance.
