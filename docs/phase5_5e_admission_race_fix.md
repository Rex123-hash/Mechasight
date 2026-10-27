# Phase 5.5E — Recovery-Admission Cleanup Race Closed

**Status: CANDIDATE — NOT AUTHORIZED FOR REAL PROTECTED EVALUATION.**

Synthetic fixtures only. Not done in this stage:
- model training;
- real protected input access, inference or analysis.

The real ledger stays at 196 rows. The scientific package is unchanged. The verified 5.5D
fixes N1 (reader binding) and N2 (registry-derived comparator class) are untouched.

## Root cause

In 5.5D, three operations used `recovery.admission` *by pathname*, outside any common lock:
- stale-admission cleanup checked the admission's owner;
- the incident resolver then moved whatever file was at that path;
- admission release deleted the path unconditionally.

The reviewer's counterexample exploits the reused pathname:
1. Cleaners A and B both judge the same dead admission stale.
2. A quarantines it.
3. Recovery R creates a *new* admission at the same path.
4. B, acting on its earlier observation, quarantines R's admission.
5. A second recovery, S, is then admitted while R is still active.

O_EXCL creation and atomic rename cannot fix this, because the check and the act are separated.

## Correction (`src/experiment/phase5_execution.py`)

**One stable cross-process coordinator.** Every admission transition runs while an OS
byte-range lock is held on `admission.coordinator`:
- admission creation;
- the recovery owner check and takeover;
- stale-admission validation and quarantine (the whole incident action);
- admission release.

The coordinator file:
- is created on first use and never moved, replaced or deleted, so its identity cannot suffer
  pathname reuse;
- is locked with `msvcrt.locking` on Windows/NTFS, or `flock` elsewhere, on an open handle;
- is released by the OS when the holder exits or crashes, so it cannot go stale;
- is non-blocking: a contender is refused immediately, with no sleep or poll.

**Exact admission instances.** Each admission records a unique `admission_id`.
- The cleaner pins `(admission_id, sha256)` of the instance it judged dead.
- Immediately before the move it re-validates that the same instance is still there and its
  owner is still dead. After the move it checks the moved file's hash; on a mismatch the file
  is restored and nothing is quarantined.
- Release removes the admission only if the recorded `admission_id` is still the releasing
  recovery's own. Otherwise it refuses, leaves the file untouched and marks its lock failed.

**Crash behaviour.**
- A crash inside a coordinated section frees the coordinator (OS) and leaves the admission file
  as durable evidence. Every later recovery then refuses: "a recovery admission is present".
- `diagnose()` reports `INTERRUPTED_RECOVERY_ADMISSION`.
- The existing audited `resolve_interrupted_writes` (incident permit) quarantines the
  admission, but only if its owner is provably dead (PID plus creation time). Otherwise a human
  resolves it.
- An incident record written before a crash is kept; the next resolution writes its own.

## Regression evidence (`tests/test_phase5_executor.py::test_E_*`, synthetic only)

| Case | Result |
|---|---|
| The reviewer's ordering: A validates → B tries → A quarantines → R admits → B resumes → S tries | B is refused twice ("coordinator is held"); R keeps its admission (same `admission_id`) and completes; S is refused; afterwards B finds nothing to resolve. One quarantined file, hash equal to the dead instance; one `RECOVERY_AUTHORIZED`; ledger 428 + 231 |
| A — two cleaner **processes** (spawn workers) on one dead admission | exactly one accepted, one refused; quarantined once; one `INTERRUPTED_WRITES_QUARANTINED` |
| B — cleanup vs a fresh admission (in process, and cleaner process vs recovery process) | the recovery is refused while cleanup holds the coordinator, then succeeds |
| C — live admission owner | refused; file byte-identical |
| D — dead admission, authorized | quarantined (reviewer test, A, E, F) |
| E/H — cleaner process killed (`os._exit`) after recording the incident, before moving | coordinator freed by the OS; admission unmoved; record kept; diagnosed; second resolution quarantines it; recovery succeeds |
| F — recovery process killed inside admission | a genuinely dead owner; recovery refused until audited quarantine; then recovers |
| G — admission replaced between validation and move; replaced under a live recovery | nothing moved; replacement untouched; recovery refuses, marks its lock failed, exposes nothing |
| I — retries | unique `-r1` identities (231); ledger identities unique |
| J — completed results | retained result unchanged |

The 5.5D tests (deterministic interleaving, two-process recovery race, live owner,
interruptions before and after takeover, stale admission) and every N1 and N2 regression
continue to pass.

## Limitations

- The coordinator relies on the OS honouring byte-range locks on a local NTFS volume. Network
  filesystems with weak lock semantics are out of scope.
- The coordinator is held only for short sections. A live recovery that hangs inside one keeps
  others refused, which is fail-safe, until it exits.
- In-process guarantees remain application-level, not cryptographic.

## Provenance

This change alters the executor source identity. The candidate in
`reports/phase5_executable_freeze_candidate/` is regenerated before the commit from starting
HEAD `c534c85`, binds this document, and stays UNREVIEWED. A future reviewer verifies the
commit that contains it with
`scripts/run_phase5_fixture_executor.py --verify-candidate <SHA>`.
