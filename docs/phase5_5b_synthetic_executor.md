# Phase 5.5B — synthetic-only executor candidate

**EXECUTABLE FREEZE CANDIDATE — NOT AUTHORIZED FOR PROTECTED EVALUATION.**

Starting checkpoint: `d7ea8ac3022acd755a847b77118375fb719ef6e3`, branch
`phase5-multisensor-fusion`. This implementation exercises the entire frozen
232-unit plan using synthetic temporary files, scores and labels. No model is
trained or loaded for inference. Synthetic outputs are software verification
evidence and must never be reported as MechaSight performance.

## Immutable identities

`FrozenPackage.load` verifies the original candidate manifest against its exact
Git object at the starting checkpoint and an embedded SHA-256 anchor. Each of
the five original package documents must still match that trusted manifest.
It verifies model/artifact hashes, experiment-key identity, exact model and
standardizer file sets, calibration/threshold identities, cohort contracts,
split hashes, unit membership, all sensor subsets and L3 references. Before
each attempt it repeats artifact verification and compares nested dictionaries
against the trusted committed records, rejecting post-load mutation as well as
newly self-hashed forgeries. Existing 14-CRLF/10-LF provenance is retained.
All 114 frozen source/config Git-blob identities are also checked against the
original inventory. Immutable Git object bytes may be cached; current model and
data files are always rehashed. Fixture permits bind the new executor source
identity, and a source change invalidates an outstanding permit.

The original Phase-5.5 JSON package is not rewritten. Its support-source hashes
identify the old Git objects. The two additive test/tripwire amendments are
recorded in the new candidate instead of falsifying the old manifest. Current
scientific/builder code and original artifacts continue to match their pins.

## Execution boundaries

`phase5_execution.py` owns authorization, immutable plan loading, exposure,
exclusive attempts, atomic outputs, recovery, completion and analyse-once.
`phase5_inputs.py` owns path containment and projected reading.
`phase5_analysis.py` connects completed synthetic tables to the unchanged
Phase-3/4 ranking/bootstrap machinery and Phase-5 interpretation rules.
`phase5_fixture.py` creates the synthetic fixture recipe and fake predictor inputs.

A `Sandbox` must resolve inside a dedicated OS temporary directory, outside
the repository. Absolute/escaping input names and symlinks/junctions escaping
that directory are refused. An explicit `FixturePermit` is required for each
scoring, recovery or analysis action and binds the sandbox, fixture and package.
The fixture marker alone is not authorization. A new fixture permit cannot
authorize any real dataset path. `execute_real_protected` always refuses.
No production authorization token, production ledger writer or protected-read
scope is introduced or activated.

## Label isolation and serialization constraint

Scoring requests an explicit fixed projection of Parquet columns. It never
loads the full table and then drops outcome fields. Fixture files deliberately
contain recognizable fake labels, and tests trap column requests before the
reader returns a frame. Historical score projections similarly exclude labels.
The 28 reused L1 tables are available only in the separate analysis phase.

Spectral access reads only numeric mic/acc/gyro datasets. It never opens the
historical `provenance_json` dataset, which embeds labels in a single serialized
JSON value. A separately hash-bound, label-free window index is required.
Synthetic fixtures provide that index; this task does not manufacture or inspect
a real protected index. Future review must establish the trusted real index and
its row alignment before enabling a production adapter. Loading label-bearing
JSON and dropping fields would not satisfy this interface.

Integrity verification streams serialized file bytes to SHA-256; it does not
deserialize their outcome values. No outcome-bearing Python frame or HDF5 label
dataset is supplied to scoring. Label access requires an analysis capability,
an immutable analysis claim and a verified complete batch.

## Exposure, persistence and recovery

The synthetic ledger begins with 196 clearly fake history entries. Before any
fixture reader runs, the executor fsyncs an authorization/reservation record,
atomically publishes all planned exposure rows, publishes an exposure record
binding the ledger prefix and reservation hash, and durably logs
`EXPOSURE_RECORDED`. Read capabilities verify those records. File/hash failures
after reservation consume the synthetic exposure and are technical failures.
Failures before the ledger append consume no look and allow no input read.

Exclusive file creation prevents concurrent attempts. A failed attempt retains
its lock and state; an ordinary rerun cannot grant itself retry privileges.
Results use exclusive partial files, fsync and atomic publication, followed by
hash-bound state and append-only persisted-event records. Completion refuses
missing/duplicate persisted log events and malformed or partial attempt logs,
as well as
partial, orphan, extra, missing, misaligned, duplicate, labeled or non-finite
tables. Each result records its original unit and attempt identity; derived L2
outputs additionally record calibration identity, frozen threshold/rule and the
hashes of the exact historical component files consumed.

Explicit fixture recovery retains verified persisted outputs and assigns
`<original-id>-r<k>` identities to previously exposed failed units. Recovery
requires another fixture permit, batch identity and audit mapping; it appends
the additional exposures before rereading inputs. Orphan/partial publications
are refused for separate incident resolution rather than silently rescored.
Hard-exit recovery also checks the lock owner's PID and process creation time
using process metadata, and refuses a live owner. A killed owner can be resolved
with a new explicit fixture recovery permit. If no original exposure occurred,
the first exposure keeps the original IDs; if it occurred, retries use new IDs.
Already logged complete outputs can be finalized without rereading inputs.
The Windows implementation fsyncs file contents before atomic replacement;
directory-fsync support remains platform dependent. This is not a transactional
database or a guarantee against every storage-controller/power-loss failure.

## Analysis contract

The 24 L1 operations use a versioned synthetic predictor double with the frozen
identities and classical/spectral interfaces. The 208 L2 operations apply the
actual frozen robust-z parameters and each subset's own source-validation
threshold to fake historical scores. There is no protected L3 execution.

All 232 tables must verify before labels are accessible. Analysis claims are
exclusive and retained even if analysis fails; no automatic analysis retry is
allowed. After success, output integrity is checked against its recorded hash.

Analysis uses 10,000 shared parent-aware class-stratified draws, seed 1995,
per-replicate R8 seed averaging, percentile 95% intervals, strict threshold
comparison, frozen margins, invalid-replicate counts and existing joint-category
rules. It computes all 140 primary contrasts / 420 primary intervals and the
declared secondary comparisons, including AP/pAUROC, domain slices, member
modalities, explicitly post-hoc best modality and T-S contrasts. Every interval
is nominal, with the frozen null-exclusion yardstick. Source-slice healthy-target
FPR is undefined and reported as such, never imputed.

Missing scientific scores prevent completion. Abstention accounting preserves
all decisions in synthetic interface tests; it does not turn technical failures
into sensor abstentions or add protected L3 looks.

## Verification and handoff

Focused tests cover the 232-unit matrix, immutable identity tampering, altered
thresholds/models/contracts, projection traps, exposure ordering, input/output
corruption, duplicate/concurrent attempts, injected crashes, explicit recovery,
atomic partial publication, alignment, statistical wiring and analyse-once.
The executable-candidate manifest records source identities, retained temporary
fixture hashes and synthetic evidence. The real ledger and scientific artifacts
are checked separately and remain unchanged.

Run focused checks with `C:/Python314/python.exe -B -m pytest -p no:cacheprovider`
and disabled bytecode/cache plugins. Run the explicitly synthetic demonstration
with `C:/Python314/python.exe -B scripts/run_phase5_fixture_executor.py --synthetic-demo --test-report <successful-temp-junit.xml>`.
No full historical suite, model fits or official corruption suite is rerun.

Real execution remains disabled. Independent review must assess the executor,
fixture evidence, trusted real window-index preparation, production inference
integration, platform durability limits and human authorization contract before
any Phase-5.6 decision. The 232 scientific identities and V2 definitions are
preserved. No physical robustness or real model-performance claim follows from
the synthetic exercise.

## Recorded final checks

The focused run passed **122 tests** in **211.07 seconds**: 41 executor tests,
29 registry-candidate tests, 42 synthetic statistics tests, six L3 decision/
fallback cases and four static reader/scope tripwires. No test that opens an
actual protected scope, snapshots protected contents, replays sealed results,
fits models or reruns the official corruption suite was selected. This is not
a full-suite claim. The JUnit report is included in the new candidate.

The separately recorded synthetic demonstration completed **24 L1 + 208 L2 =
232 units**, with its separate ledger progressing **196 to 428**. It exercised
10,000 draws at seed 1995, 140 primary contrasts / 420 primary intervals, and
the frozen secondary comparisons. All temporary inputs, outputs, reservation,
exposure, attempt and analysis files have recorded hashes in
`reports/phase5_executable_freeze_candidate/synthetic_evidence.json`.

The task's initial **715-file raw-byte snapshot** (original scientific artifacts,
models, original candidate JSON and real ledger) matches after implementation.
The executable loader additionally verifies **708 original canonical hashes**
and **114 frozen code/config Git objects**. All **14,977 ignored data/report stat
identities** still match the original inventory. Protected input contents are
not content-hashed or opened; stat agreement is not a cryptographic payload
audit. The real ledger remains byte-identical at 196 rows, SHA-256
`f75584611c18ff387bee8c06f1fcc999c3046def876ba13dea1c744fce64de5f`.

## Phase 5.5C corrections (supersede the affected statements above)

The independent 5.5B audit found five issues, all corrected in Phase 5.5C. Details are in
`docs/phase5_5c_executor_hardening.md`.

- **Forgeable read permission (N1).** `Access` carried a caller-supplied verifier.
  Capabilities are now issued, matched and revoked by the owning executor.
- **Unpinned executor identity (N4).** `verify_executable_candidate` now binds executor sources
  and evidence to an externally reviewed commit.
- **Undefined control-write, recovery and analysis interruption procedures (N3).**
- **Degenerate synthetic statistics.** The v1 fixture separated its classes perfectly, so all
  140 AUROC primary contrasts were equivalent and 393 of 420 primary intervals had zero width.
  That verified plumbing only.
- **Provenance and retry statements.**
  - The evidence above was generated before its commit.
  - "196 to 428" holds only for an uninterrupted run.
  - The S4 slice bootstrap is an implementation clarification flagged for freeze review.
