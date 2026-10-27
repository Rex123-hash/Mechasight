# Phase 5.5D — Three Confirmed Executor Security Defects Closed

**Status: CANDIDATE — NOT AUTHORIZED FOR REAL PROTECTED EVALUATION.**

Synthetic fixtures only. Not done in this stage:
- model training;
- real protected input access, protected inference or analysis;
- any real analyse-once execution.

The real ledger stays at 196 rows. The scientific package (52 units, 24 L1, 208 L2, 232
planned units, V2, registry, L3 results) is unchanged.

This stage corrects the three defects the independent reviewer demonstrated against Phase 5.5C (verdict RED). It
keeps every 5.5C protection:
- executor-issued capabilities;
- the phase split;
- durable exposure checks;
- revocation;
- the analyse-once claim;
- interrupted-write quarantine;
- recovery identities;
- candidate identity verification.

## N1 — public reader verifier bypass (closed)

**Defect.** `ProjectedInputs(sandbox, manifest, verifier)` accepted any callable. A separately
built reader with a no-op verifier returned 12 synthetic label rows for an invented batch with
no exposure record.

**Correction** (`phase5_inputs.py`). The reader takes no verifier.
- It is constructed with its owner, which must be an exact `Executor` instance; subclasses and
  callables are refused at construction.
- At **every read**, before any path is resolved, opened or hashed, it requires that the owner
  is still that exact `Executor`, that `owner.inputs is self`, and that the reader serves the
  owner's sandbox and fixture.
- It then calls `Executor._verify_capability` through the class, so an instance-level override
  is never consulted.

The verifier is unchanged from 5.5C, plus one addition: analysis reads also require the
completed scoring gate (state `SCORED`, all 232 tables, matching batch), together with the
durable claim and the `ANALYSIS_CLAIMED` event. The verifier checks:
- the issued token;
- executor and sandbox ownership;
- batch;
- phase;
- revocation;
- the reservation;
- ledger rows;
- the ledger prefix;
- the exposure record and event.

**The reviewer's counterexample, reproduced** (`test_D_N1_reviewer_counterexample…`):
- the no-op-verifier reader is refused at construction;
- a lenient `Executor` subclass is refused;
- a stray reader bound to a real executor is refused at read time;
- the invented batch is refused as unissued, and still refused when a token is issued for it,
  because there are no durable records.

**0 label rows returned, 0 parquet opens or hashes, 0 reader events.** A corrupted reservation
also invalidates existing access.

**In-process limit.** As before, this closes the application-level bypass. It is not a defence
against arbitrary code in the same Python process, which could rewrite private attributes or
monkeypatch the class.

## N2 — caller-controlled historical comparator class (closed)

**Defect.** `historical(access, eid, reused_l1=False)` let the caller decide whether a table
was analysis-only. Omitting the flag read an analysis-only L1 comparator under a scoring
capability.

**Correction.** The flag is removed. `historical_policy(registry)` derives the class from frozen
registry membership:
- the 156 Phase-4 single-modality tables (units' L0) are readable while scoring, for L2
  derivation;
- the 28 reused historical L1 comparators are analysis-only.

The policy is computed from the verified registry at executor construction and re-checked
against the registry at every permit. `Executor._historical_artifact(eid)` refuses three kinds
of request, all **before any file is opened or hashed**:
- an unregistered or non-string identity;
- an identity whose fixture path is not the identity-derived `historical/<eid>.parquet`, which
  defeats filename aliasing;
- an analysis-only comparator under a scoring capability.

**Tests** (`test_D_N2_*`):
- a scoring-stage read of an L1 comparator is refused with the flag omitted and with keyword
  arguments;
- passing `reused_l1=True`, `reused_l1=False` or a positional extra raises `TypeError`;
- an unregistered, traversal-style, `None` or case-altered identity, and an aliased path, are
  refused;
- none of these touches a file;
- a permitted L0 read under scoring returns its 12 rows;
- under a valid analysis capability every reused-L1 read is logged in phase `analysis`, and none
  in `score`.

## N3 — concurrent recovery takeover race (closed)

**Defect.** Two recoveries could both pass the stale-owner check and both replace the lock.
`os.replace` is atomic, but it confers no exclusivity.

**Correction** (`phase5_execution.py`).
- **Admission.** `recover()` first creates `recovery.admission` with O_EXCL, which is atomic on
  NTFS. Losers are refused at once.
- **Checks under admission.** Every check runs under admission: state, recovery authority, lock
  presence, the owner's failure mark or provable death (PID plus creation time), retained-result
  hashes and pending identities.
- **Takeover.** The archive, the `RECOVERY_AUTHORIZED` event and the takeover (an atomic lock
  replacement) all happen *while admission is held*.
- **Release.** Admission is released only after ownership is established. A later recovery then
  meets a live owner and is refused.
- **Live owners.** A live owner's lock is never overwritten.
- **Hard exit inside admission.** This leaves `recovery.admission` behind, so every recovery
  refuses (fail closed). `diagnose()` reports `INTERRUPTED_RECOVERY_ADMISSION`. If its owner is
  provably dead, the audited `resolve_interrupted_writes` (incident permit) quarantines it with
  its hash; otherwise a human resolves it. Quarantining only re-opens admission, so every check
  is re-run.

**Tests** (`test_D_N3*`):

| Case | Result |
|---|---|
| A — the reviewer's deterministic interleaving | B is refused inside A's admission window and never reaches the owner check; one `RECOVERY_AUTHORIZED`; ledger 428 + 231 |
| B — two real OS processes (`multiprocessing` spawn workers in `tests/_recovery_race_worker.py`, the suite's established race pattern; raw process spawns stay forbidden in tests), released by a cross-process barrier, winner holds admission 4 s | exactly one accepted, one refused, one authorization event, ledger 659 |
| C — original owner live | refused, lock untouched, no admission left |
| D — dead owner | recovers (also covered by the 5.5C hard-crash test) |
| E — interrupted before takeover | no ledger change; retry with a new unique batch succeeds |
| F — interrupted right after takeover | lock marked failed by its new owner; scoring and analysis stay refused; retry succeeds with unique identities |
| E, hard exit — stale admission | refused while its owner is live, quarantined once dead, then recovery succeeds |
| G — recovery attempted while a recovered owner is live, and after a completed recovery | both refused |

## Provenance

The executor source identity changes with these fixes. The 5.5D candidate
(`reports/phase5_executable_freeze_candidate/`) regenerates the synthetic evidence and manifest
before the commit. It binds this document as a candidate source and keeps
`review_status: UNREVIEWED`. The original scientific freeze candidate (`d7ea8ac` anchor) is
untouched.

**Which commit to review.** A future independent reviewer must verify the commit that contains
this document and the regenerated candidate. It is reported at commit time and checked with
`scripts/run_phase5_fixture_executor.py --verify-candidate <that SHA>`. No approval is claimed.

## Still out of scope

- a real label-free window index;
- a real inference adapter;
- a real ledger writer;
- human authorization;
- protected evaluation;
- Phases 5.6 and 5.7.
