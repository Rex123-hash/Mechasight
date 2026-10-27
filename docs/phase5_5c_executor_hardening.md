# Phase 5.5C — Executor Security Corrections and Final Synthetic Hardening

**Status: CANDIDATE — NOT AUTHORIZED FOR REAL PROTECTED EVALUATION.**

Synthetic fixtures only. Not done in this stage:
- model training;
- real protected input access, protected inference or analysis;
- authorization of any of the 232 proposed looks.

The real ledger stays at 196 rows. The scientific package (52 units, 24 L1 fits, 208 L2
calibrations, 232 proposed evaluation units, V2 design and registry) is unchanged.
This stage corrects the findings of the independent Phase-5.5B executor audit (N1, N3, N4, N5).

## 1. N1 — forgeable read permission (MAJOR, closed)

**Defect.** In 5.5B, `Access` was a public dataclass that carried its own `check` callback,
and the sandbox nonce was a public attribute. Any caller could therefore build
`Access(sandbox, sandbox.nonce, batch, "analysis", lambda a: None)` and read labels or inputs
with no reservation, exposure record or analysis claim.

**Correction.**
- `Access` (`phase5_inputs.py`) is now only a *reference*:
  executor identity, sandbox identity, batch, phase and a random 256-bit token. It holds no
  verifier, and the sandbox nonce is gone.
- `ProjectedInputs` is built with exactly one verifier, `Executor._verify_capability`, bound at
  construction and stored privately. Readers never consult anything the caller supplies.
- The verifier requires the access to be the very object the executor issued: an exact match
  in that executor's capability table on token, batch, phase, executor and sandbox. It then
  enforces the phase rule: labels and reused-L1 tables need an `analysis` capability.
- It then re-checks the durable state on every read:
  - the reservation record, intact against the exposure hash;
  - ledger rows for exactly the reserved attempt identities;
  - an unchanged ledger prefix;
  - the `EXPOSURE_RECORDED` event;
  - for analysis, additionally the claim file, the `ANALYSIS_CLAIMED` log event and an
    unchanged completion-state hash.
- Capabilities are issued only after these checks pass. They are revoked when their attempt
  ends, whether it succeeds or fails.
- Missing authorization or exposure files are refused, not raised as `FileNotFoundError`.
- Permit actions stay separate: `score`, `analysis`, `recovery` and `incident`. Every permit
  now has a unique batch identity.
- A static tripwire (`test_only_the_executor_constructs_read_capabilities_and_projected_readers`)
  allows `Access(`, `ProjectedInputs(` and `_capabilities` only in `phase5_execution.py`.

**In-process limit.** Python objects are not cryptographically unforgeable. Code running in the
same process can still reach the executor's private table. What is closed is the
application-level bypass: no constructed, copied, altered, stale or foreign capability is
accepted, and the approved read paths cannot skip the durable-state checks.

**Adversarial tests** (`tests/test_phase5_executor.py::test_N1_*`), all refused:
- an `Access` with an always-succeeding callback, which now raises `TypeError`;
- a constructed token;
- a copied token with an escalated phase, a changed batch or a changed executor (forged or
  copied nonce);
- a score capability used for labels;
- analysis before completion;
- recovery, analysis or incident permits used for scoring and vice versa;
- a revoked capability from a finished attempt (stale permit);
- a capability of another executor or sandbox;
- issuing without reservation;
- reading after the exposure record is removed;
- reading after the ledger prefix is truncated.

## 2. N4 — executor identity and manifest validation (closed for candidate verification)

`verify_executable_candidate(reviewed_commit)` (`phase5_execution.py`) takes the commit from
**outside** the code, from the independent-review or human-authorization record. It reads
`reports/phase5_executable_freeze_candidate/manifest.json` from that commit's Git objects and
requires:
1. the working manifest is byte-equal (LF) to the reviewed one;
2. the exact key structure, the candidate status, `real_execution_enabled=false` and
   `authorization_issued=false`;
3. the package trust anchors (`TRUSTED_COMMIT`, `TRUSTED_MANIFEST`) equal the reviewed
   candidate's;
4. every candidate source (executor, driver, tests, docs) and evidence file has the reviewed
   commit's content and the working content, both equal to the manifest digest;
5. `executor_source_identity` equals the recomputed identity, the running code's identity and
   the identity recorded in the synthetic evidence;
6. `FrozenPackage.load()` still verifies the original package, including the 708 artifact
   hashes and the 114 frozen source/config blobs, against `d7ea8ac`.

**Why it is not self-referential.** Editing a source or a trust constant changes its hash, and
the new hash passes only once a reviewer names a commit containing it. A self-consistent
working manifest for modified code fails against the reviewed commit's manifest.

**Tests**, all refused (`test_N4_*`):
- modified executor source;
- an altered trust anchor;
- a forged manifest;
- a restructured manifest;
- a non-SHA commit;
- modified evidence;
- internally consistent but unreviewed metadata, in both directions.

A simulated reviewed candidate verifies.

**How the reviewed commit becomes trusted.** This stage commits the candidate with
`review_status: UNREVIEWED`. An independent reviewer audits that commit and records its SHA.
Any future real authorization must pass `verify_executable_candidate(<that SHA>)` and record
it, and the SHA is never read from the repository itself. No such approval exists yet. The
driver's `--verify-candidate <commit>` mode runs the check read-only.

## 3. N3 — crash and recovery handling

- **Control-file fault injection.** Every control write (state, reservation, ledger, exposure,
  recovery, analysis claim and output) now has a `control_partial_write` fault hook. Result
  tables keep their 5.5B `partial_write` hook.
- **Interrupted writes.** An unpublished `*.partial`, or a published table missing from
  `state.json`, blocks every action:
  - scoring;
  - recovery;
  - completion;
  - analysis.
- **`resolve_interrupted_writes` (incident permit).** It runs only if the lock owner recorded
  its own failure or is provably dead.
  - It *moves* every such file under `incidents/<batch>/` after recording each hash and the
    published target's hash, and logs `INTERRUPTED_WRITES_QUARANTINED`. Nothing is deleted or
    overwritten.
  - A quarantined table's unit stays pending. Its look is already in the ledger, so the next
    recovery gives it a new `-r<k>` identity.
  - A lock left by an attempt that died before its first state write (no state, no
    reservation, no exposure rows) is quarantined too.
- **Lock semantics.**
  - A failing owner's handler marks its retained lock `FAILED_HANDLER_RAN`.
  - An unmarked lock, as left by a hard exit, can be taken over only after the PID and creation
    time show the owner is dead.
  - Recovery replaces the lock atomically, so it never disappears; the old owner record is
    archived under `recoveries/`.
- **Failure handler.** If the handler's own state write hits a leftover `.partial`, it no
  longer masks the original failure.
- **Recovery.** Every recovery is a distinct permit and batch, logged as `RECOVERY_AUTHORIZED`
  before the takeover. Recovery never overwrites a retained result (a changed one is refused)
  and never reuses an exposed identity: `-r1`, `-r2`, … are taken from the ledger. A
  recovery interrupted after its own exposures recovers again with `-r2` identities (tested:
  ledger 428 + 231 + 231).
- **Analysis.**
  - The claim is written and then logged as `ANALYSIS_CLAIMED` (fsync) *before* any label access
    (tested with a label-reader trap).
  - An interrupted analysis is **never repeated automatically**. `analyse_once` refuses while
    the claim exists. Even if the claim file is deleted, the append-only log still records the
    claim and `analyse_once` refuses.
  - The incident action refuses analysis files, and recovery refuses a scored batch.
  - `diagnose()` reports `INTERRUPTED_ANALYSIS`: a human incident decision is required, because
    labels may already have been read.
- **`diagnose()`** is a read-only classification giving the only permitted next action:
  - `NOT_STARTED`;
  - `STALE_LOCK_BEFORE_STATE`;
  - `INTERRUPTED_WRITE`;
  - `RECOVERABLE`;
  - `SCORED`;
  - `INTERRUPTED_ANALYSIS`;
  - `CORRUPT_AUDIT_LOG`, which always needs a human.

## 4. Retry and exposure accounting

| Situation | Ledger |
|---|---|
| Uninterrupted 232-unit evaluation | 196 → **428** |
| Recovery, each unit already exposed but unpersisted | **+1 look** under a new `-r<k>` identity |
| Interruption before the ledger publication | no look; original identities are kept |

Every additional exposure follows the same rule as the first: reservation, ledger rows,
exposure record and `EXPOSURE_RECORDED`, all before any read. Each one is authorized by its own
recovery permit (unique batch) and logged (`RECOVERY_AUTHORIZED`, mapping, retained set).

Tested ledgers after one crash range from 428 (interrupted before exposure) to 660, and 890
after an interrupted recovery. **A real evaluation must not assume 428**: each retry is an
explicit, separately authorized and counted exposure.

## 5. Statistics: degenerate 5.5B evidence and the non-degenerate checks

The 5.5B fixture separated its fake classes perfectly:
- all **140** AUROC primary contrasts were trivially `EQUIVALENT`;
- **393 of 420** primary intervals had zero width.

That verified orchestration and counting, not statistical validity.

Fixture v2 (`phase5-fixture-v2-overlapping`):
- fake class scores overlap;
- each fake historical detector gets a deterministic, identity-specific jitter, so arms rank
  differently.

`tests/test_phase5_synthetic_statistics.py` (overlapping classes, 16 parents × 3 segments, two
averaged seeds) recomputes bootstrap replicates **independently** by row expansion with
scikit-learn. It confirms:
- parent-aware, class-stratified draws (one multiplicity per parent, class sizes preserved);
- seed-averaged AUROC replicates and point estimate (to within 1e-12);
- nonzero-width paired contrasts, and an exactly zero self-contrast;
- healthy-target FPR replicates against a manual weighted count, with invalid replicates when
  no healthy target unit is drawn;
- the S4 slice construction.

These are synthetic and make no performance claim.

**Recorded 5.5C demonstration** (fixture v2, all 232 units, 10,000 draws, seed 1995). It gives
non-degenerate evidence for AUROC only:

| Metric | Nonzero-width primary intervals | Categories |
|---|---|---|
| AUROC | 126 of 140 | 14 EQUIVALENT, 2 IMPROVEMENT, 124 INDETERMINATE |
| Healthy-target FPR | 0 of 140 | — |
| Recall | 55 of 140 | — |
| All three primary metrics | 181 of 420 (239 zero width) | — |

The threshold metrics stay largely degenerate in the end-to-end demonstration:
- the frozen source-validation thresholds are real Phase-4/L2 values on real score scales;
- the fake scores are not on those scales, so the strict rule `score > threshold` puts almost
  every fake segment on the same side;
- the frozen thresholds are deliberately not altered for a fixture.

The non-degenerate threshold-metric bootstrap is therefore evidenced by the independent unit
test above, not by the demonstration.

## 6. S4 slice bootstrap — implementation clarification (flagged for freeze review)

**Method.** S4 intervals mask the shared `(replicates × segments)` weight matrix to the slice's
columns:
- each replicate is the full-cohort, parent-aware, class-stratified joint resample;
- the slice metric is computed on that resample's slice members;
- paired arms share the same masked draws;
- a replicate with no qualifying units in the slice is invalid and counted;
- source-slice healthy-target FPR is undefined and reported as such.

**What V2 freezes.** V2 freezes S4 only as "source-slice and target-slice versions of every
primary contrast", plus the shared draws for all intervals. Phase 4 reported domain slices as
point estimates only.

**Consistency.**
- The method uses the frozen draws unchanged and is consistent with V2's shared paired-draw
  rule.
- It does not touch P1–P3, which use the complete cohort.
- The test above shows each masked-weight replicate equals the slice metric of the
  corresponding joint resample.

**Not frozen.** The method itself is not written in V2. It is a secondary-analysis
implementation choice, **flagged for independent freeze review** and not silently added to the
protocol.

## 7. Synthetic evidence provenance

The 5.5B evidence was generated from an uncommitted tree (`builder_head = d7ea8ac`), although
its source hashes matched the later commit `f10f029`.

The 5.5C evidence records:
- `generated_from_head`;
- `tracked_tree_clean_at_generation`, which is false: generated before the 5.5C commit;
- the exact `executor_source_identity` of the code that produced it.

The manifest binds the same identity. A source identity is **not** an independent review; see
section 2.

## 8. Still unfinished (not authorized here)

- a trusted label-free real window index (the existing caches embed labels in
  `provenance_json`);
- the actual Phase-5 real inference adapter;
- integration with the append-only JSONL ledger and `verify_ledger_history` batch
  authorization;
- the real human batch authorization;
- independent review of this candidate and of the real protected-access procedures.

`execute_real_protected` still always refuses.
