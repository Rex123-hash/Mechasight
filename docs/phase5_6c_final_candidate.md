# Phase 5.6C — protected-style adapter, JSONL batch interface, development mirror, final candidate

Status: **IMPLEMENTED; NOT INDEPENDENTLY REVIEWED; NOT AUTHORIZED FOR REAL PROTECTED
EVALUATION.** No model was retrained and no threshold, split, comparison plan or design
changed. No protected recording, cache, feature artifact, label or sealed score was opened.
The real ledger is unchanged at 196 rows (SHA-256 `f7558461…de5f`).

Starting checkpoint: `6411322` (Phase 5.6B).

## 1. Bounded review of Phase 5.6B

Scope: `phase5_input_manifest.py`, `phase5_count_index.py`, `verify_phase5_count_index.py`
and their evidence.

| Check | Result |
|---|---|
| Candidate bytes vs commit `6411322` | `verify_phase5_count_index.py --verify-candidate` passes |
| Focused and guard tests | 123 passed (re-run in this phase) |
| Trusted manifests | rebuilt from Git-pinned registry, split, configs and inventory at `d7ea8ac`; caller copies are never authority |
| Count-derived alignment | 40,380 windows over 4 cells equal the timestamp-derived index (5.6B evidence) |
| Label isolation | count and classical projections contain identities, `window_count` and 98 features only; spectral reads touch modality datasets only |
| Development-only access | the protected scope refuses before any stat, hash or read |
| Frozen inference | 24 real models, 3,744 segment scores, max diff 0.0 |

No material defect was demonstrated, so 5.6B was not changed.

Consequence noted: 5.6B already changed `tests/test_phase5_protected_read_guard.py`, and 5.6C
changes two executor sources. The 5.5 executable candidate and the 5.6B component candidate
therefore no longer verify against the working tree. Both are retained as historical
evidence and are superseded by `reports/phase5_final_candidate/`.

## 2. Protected-style input adapter (gap B)

`src/experiment/phase5_inputs.py` adds two input schemas:
- `MIRROR_SCHEMA` (`phase5-protected-style-dev-mirror-v1`);
- `PROTECTED_SCHEMA` (`phase5-protected-common-evaluation-v1`). It is recognised only so that
  it can be refused, and it is not in `INPUT_SCHEMAS`, so an executor cannot even be built on it.

Under the mirror schema, the existing `ProjectedInputs.features` / `windows` work as follows:

1. **Executor-owned policy first.** `__authorize` → `Executor._verify_capability`, which means
   an issued, live capability plus the durable reservation, ledger and exposure records.
   Nothing else happens before this.
2. **Trusted identity.** The reader rebuilds `development_manifest(package, machine, regime)`
   from Git-pinned metadata. The fixture holds only that manifest's identity, and a mismatch
   refuses. The trusted cohort must equal the executor cohort.
3. **Strict path binding.** Every path, stat and content hash comes from the trusted manifest
   (`_checked_path`), never from the fixture or the caller. Content hashes are checked after
   exposure.
4. **Projected reads.**
   - Classical inputs use the trusted 98-feature layout.
   - Spectral inputs come from `read_spectral(values=True)`, which reads only the
     `mic` / `acc` / `gyro` datasets. `provenance_json` is never deserialized.
   - Windows come from `build_count_derived_index` → `inference_index`, the label-free ordinal
     alignment with no physical timestamps.
5. **No label access in scoring.** Labels remain behind the separate analysis capability.

The executor, `FrozenL1Adapter`, the L2 derivation and the analysis are unchanged.

## 3. JSONL batch interface (gap C, temporary ledger only)

`JsonlLedger.verify()` now also runs `validate_batches`. For every appended run it checks:
- the authorization record has exactly the synthetic schema, the counts, the look count and the
  configured code SHA;
- the record is bound to the executor's durable reservation by SHA-256 and by start/end;
- the run's identities are exactly the reserved ones;
- every look is a planned unit or a well-formed `-r<k>` retry of one, so the 28 reused
  comparators never consume a look;
- the first batch is exactly 24 L1 + 208 L2 looks;
- batch ids are never reused;
- the ledger stays within the proposed (unauthorized, H4) total cap of 660 rows. This is
  detection only; prevention is the pre-mutation `precheck` in §3a.

### 3a. Correction r1: look budget checked before any mutation

**Defect (found in independent review).** The budget was only checked in `validate_batches`,
which runs after an append. A synthetic ledger at 660 rows accepted a 232-look recovery batch:
- the batch authorization was published;
- the ledger grew to 892 rows;
- reads were then refused, but the excess looks had already been recorded.

There was also a semantic error. The proposal's cap is a **total ledger length**: 428 + 232 =
660, which is 464 additional looks above the 196 baseline. The code had treated 660 as
*additional* looks, which is 196 looks too loose.

**Fix.**
- `look_budget(package)` derives the budget only from the frozen proposal's accounting
  (baseline 196, first batch 232 ending at 428), giving a total cap of 660 and 464 additional
  looks. A fixture `base_rows` that differs from 196 is refused, and no caller argument can
  change the budget.
- `JsonlLedger.precheck(rows, attempt_ids)` enforces
  `existing_total + pending - 196 <= 464`, plus unique pending identities, no reuse of ledger
  identities, and planned or `-r<k>` retry identities only.
- `Executor._reserve` calls the precheck **before** the reservation, batch authorization,
  ledger append, exposure record or any read.
- `JsonlLedger.append` repeats it on a fresh, verified read under the executor's exclusive
  attempt lock, and refuses if the ledger moved since the reservation. A stale precheck
  therefore cannot publish anything.
- The 5.5B synthetic JSON test ledger keeps its identity check only. Its historical tests model
  repeated recoveries beyond 660 and do not use the protected ledger format.

**Result.**
- The 660 + 232 counterexample is refused with the message
  `660 existing + 232 pending - 196 baseline = 696 > 464`.
- The ledger bytes, authorizations, reservations, exposures and reads are all unchanged.
- The state is `REFUSED_PRE_EXPOSURE`.

The evidence is `reports/phase5_final_candidate/cap_precheck_evidence.json`. The tests are
`tests/test_phase5_mirror_adapter.py::test_C_*`:
- the trusted budget, and that the baseline cannot be moved;
- 660 + 232 refused before mutation, with no reader invoked;
- the append re-check;
- an exactly-on-budget batch passing and one more look failing;
- duplicate and malformed identities;
- a pre-append crash consuming no budget while exposed retries do;
- concurrent recoveries not jointly exceeding the budget;
- a ledger moved after reservation publishing no authorization.

`append_protected_batch` is the real-ledger batch mode, and it is **disabled**: it always
refuses. The real ledger path is still refused as a backend.

Tests are in `tests/test_phase5_mirror_adapter.py` (16 tests):
- 196 → 428, with a composition check and no comparator looks;
- an altered authorization (scope, extra key, reservation, code SHA) is refused;
- a reservation change after exposure is refused;
- a crash after exposure leads to recovery with unique `-r1` ids (196 → 428 → 660), and a
  reused batch or exceeding the cap is refused;
- a crash before the append leaves no look and recovers to 428;
- a scoring capability cannot read labels, and analysis is separate and runs once;
- the real ledger is never a backend.

## 4. Development-mirror end-to-end run

`scripts/run_phase5_mirror.py --write-evidence` → `reports/phase5_6c_mirror/evidence.json`.

The chain: trusted manifests → label-free count index → frozen L1 (real models) → L2 → temporary
JSONL ledger 196 → 428 → completion gate → analyse-once on SYNTHETIC labels.

**Real-model results.** These are development source-validation data, not protected performance:

| Quantity | Value |
|---|---|
| Units | 232 (24 `frozen_L1_inference` + 208 `derived_L2_scoring`) |
| Mirror cohorts | RoboticArm 186, BrushlessMotor 126 segments |
| L1 | 24 models (20 autoencoder + 4 Mahalanobis); 3,744 segment scores |
| L1 max abs diff vs frozen bundle validation scores | 0.0; window counts equal |
| L2 | 208 units, 32,448 rows |
| L2 max abs diff vs persisted L2 fused validation scores | 0.0 |
| Refitting | none; thresholds unchanged |

**Access audit.**
- Every trusted Parquet read and HDF5 open happened at ledger row 428, i.e. after exposure.
- No label or poison column was requested.
- The HDF5 datasets requested were exactly `acc`, `gyro` and `mic`.
- There were zero label reads in the score phase.
- Forged and unexposed capabilities are refused before any read. This is tested in
  `tests/test_phase5_mirror_adapter.py`; only the executor constructs capabilities.
- The disabled entry points were refused:
  - `execute_real_protected`;
  - `read_protected_inputs`;
  - `build_protected_count_index`;
  - `append_protected_batch`;
  - a protected manifest read;
  - a protected-schema executor.

**Synthetic results.** These are wiring evidence only. Development validation has no faults,
so the labels are SYNTHETIC parent-homogeneous fakes. Analyse-once produced 140 primary
contrasts and 420 intervals over 10,000 replicates, and a second analysis was refused. These
numbers are **not** model performance.

## 5. Final executable freeze candidate

`scripts/verify_phase5_final_candidate.py` writes and verifies `reports/phase5_final_candidate/`:
- `manifest.json`, which binds:
  - the source hashes (executor, adapters, cohort and index builders, ledger interface,
    drivers, tests, docs);
  - the executor identity;
  - the frozen scientific package (`7f3ae36e…` at `d7ea8ac`);
  - the registry content, design V2, evaluation-plan identity, model/standardizer manifests,
    thresholds, calibrations, cohort contracts and protected cohort hashes;
  - the starting checkpoint and the real ledger identity.
- `mirror_evidence.json`, `count_index_evidence.json`, `focused_tests.xml`, `regression_tests.xml`;
- `authorization_checklist.json`, in which H1–H7 are UNRESOLVED, no tokens are issued and
  0 looks are authorized.

`--verify-candidate <SHA>` checks the working bytes against the named commit. **This is a
self-consistency check, not an independent review or approval.**

## 6. Recorded checks and integrity report

| Check | Result |
|---|---|
| Focused tests (mirror adapter, count index, read guard), after correction r1 | 147 passed, 0 failed |
| Targeted regressions after r1 (executor; integration corrections; integration except the 4 frozen-adapter `test_B_`) | 158 passed, 0 failed |
| Before r1 (commit `49069ee`): full Phase-5 regressions incl. synthetic statistics and freeze candidate | 196 passed, 0 failed |
| Development-mirror model results | retained from `49069ee`; the score path is unchanged by r1 |
| Real ledger | 196 rows; SHA-256 `f75584611c18…de5f` unchanged |
| Frozen package | `FrozenPackage.load()` + `verify_memory()` pass; identity `7f3ae36e…` |
| L2 verify (`run_phase5_development.py l2-verify`) | 208 records, 32,448 fused rows reproduced |
| Protected snapshot (stat-only vs `d7ea8ac` inventory; no protected bytes read) | 32 entries; 0 changed / missing / new; tracked protected reports have no git changes |
| `MechaSight_First_Defence/` | untouched (newest file 2026-09-14) |

The Phase-3.4 replay test and the conftest content snapshot (which hashes protected score
payloads) were deliberately not run.

## 7. What still requires a human

- An independent review of this candidate at a named commit.
- Decisions H1–H7.
- Resolving the protected artifact bindings from approved metadata references.
- A committed batch authorization record (232 looks, 196 → 428).
- Human-issued scoring and analysis tokens.
- Then a reviewed change that activates the disabled entry points.

Until all of that happens, protected inference, protected outcome analysis and any real-ledger
write are **not executed**.

## Limitations (unchanged)

- Acquisition-time confounding on BrushlessMotor.
- Small healthy-target cohorts: RoboticArm has 20 segments from 6 parents.
- L3 evidence is synthetic corruption only.
- Phase-4 aggregate protected results were seen before this design.
