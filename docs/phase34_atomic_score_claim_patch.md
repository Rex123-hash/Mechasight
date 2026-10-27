# Phase 3.4F.2 — Atomic Protected-Score Claim (finding F3)

**Preserved unamended:** COMMIT A `107579fba1f0222519d2acd9f1c78bda78fb65e9`, patch F1/F2
`dcb57d17c135f30a0db18c93aa77eff7c45aadff`.
**Scientific config identity:** `p34d-exec-e5470b025229f59a06f4c772` — unchanged.
**Config document SHA-256 (LF-canonical):** `70b1830e5cf2608dd97c20dda87d32f7a41205873246a74bb99626d31edea965` — unchanged (no config edit in this patch).

> No model trained. No real test inference. No real test score. No checkpoint. Ledger 30 → 30.
> Protected-test exposure 0 → 0. Every scientific object of V3 is untouched.

---

## The race F1 could not close

F1 made the exposure history durable and append-only, which is correct for *sequential* retries.
But two invocations of the scorer could interleave:

```
A: history empty ✓        B: history empty ✓
A: append SCORE_STARTED    B: append SCORE_STARTED
A: inference               B: inference          ← both consume the same look
```

`O_APPEND` makes each *append* atomic. It does not make *"check-empty then claim"* atomic. That
needed an operating-system create-if-absent primitive.

## The claim

`reports/phase3_4_results/score_claims/<experiment_id>.claim`, acquired by

```python
os.open(path, O_CREAT | O_EXCL | O_WRONLY, 0o444)
```

then written and `fsync`ed **through the same descriptor** before it is closed. **The create is the
check.** Of any number of concurrent callers, exactly one can succeed for a given path; the others
get `FileExistsError` and refuse. There is no separate "exists?" step for a second process to
interleave with, and on failure nothing is written.

The claim binds `experiment_id`, `fold`, `arm`, `seed`, `checkpoint_sha256`, `evaluator_git_sha`,
`config_identity`, `training_membership_sha256`, `test_cohort_hash`, the ledger authorisation
timestamp it consumes, and its own timestamp. `SCORE_STARTED` and `SCORE_COMPLETED` now carry the
claim timestamp and checkpoint hash, so claim and history are mutually identifiable.

### Frozen sequence

```
VALIDATE AUTHORISATION  (record, checkpoint hash, ledger entry naming that hash,
                         no claim, no history, no artifact)
      ↓
ATOMIC CLAIM            (O_CREAT|O_EXCL — exactly one owner)
      ↓
SCORE_STARTED           (durable append, before the first protected forward)
      ↓
protected inference
      ↓
atomic score persistence + hash
      ↓
SCORE_COMPLETED         (durable append)
```

### Conservative crash semantics

**Once the claim is acquired the look is consumed**, regardless of where a crash occurs — before
`SCORE_STARTED`, before the HDF5 open, before the first forward, mid-inference, or before the atomic
rename. Every later invocation refuses. The claim is created read-only, is never deleted by any
scorer command, and there is no stale-lock recovery. A claim without `SCORE_COMPLETED` is an
**INCOMPLETE** experiment. A claim whose `checkpoint_sha256` differs from the frozen checkpoint is
a **HARD FAILURE**: claims are never replaced.

---

## The concurrency test — two real processes

`tests/test_phase34_score_claim_race.py::test_two_processes_race_the_claim_and_exactly_one_scores`
spawns **two independent OS processes** (`multiprocessing` `spawn`, no shared interpreter state)
that both target the same experiment from **no prior exposure**, both pass every pre-check, then
meet at a cross-process `Barrier` bound to a test seam **immediately before the `O_EXCL` attempt**,
and race it. Inference is a fake that logs each forward durably; the real cohort is never opened.

Proven, and stable across five repeated runs:

- outcomes are exactly `["REFUSED", "SCORED"]`; the refusal names the held claim;
- **exactly one** protected forward occurred, and the claim was on disk when it did;
- exactly one claim file, no `.tmp` fragment;
- history is exactly `[SCORE_STARTED, SCORE_COMPLETED]` — one STARTED, not two;
- nothing reset.

## Tamper / reset tests

| Case | Result |
|---|---|
| claim exists, no `SCORE_STARTED` | refuses as **INCOMPLETE**, zero forwards |
| claim exists + `SCORE_STARTED` | refuses, zero forwards |
| claim exists + `SCORE_COMPLETED` | refuses, forward count stays 1 |
| claim names a different checkpoint | **HARD FAILURE**, claim left untouched |
| three repeated invocations after a completed look | claim and history byte-identical afterwards |
| no scorer command unlinks a claim | asserted against the module source |
| crash **after** claim, **before** `SCORE_STARTED` | claim persists, history empty, next call refuses as INCOMPLETE |
| `O_EXCL` primitive in isolation | second `acquire` raises; first writer's content survives |

---

## Unchanged

`initialization_sha256 = d889e90224988ac27bceba086f695ac5d7f8f34ca1e2ad1debc16c6f61384d55`;
`initial_state_sha256 = 01d68f698cdf32628604536d282e218106ac48f51fb14d2a74bf667c2c980048`;
memberships (all 10 hashes reproduce), experiment matrix, seed 1995, environment
`p34c-env-b7a5de2af4c1d9614e49`, architecture, preprocessing, estimand, bootstrap, execution order,
config identity and the config-document provenance mechanism. The existing F1/F2 tests pass
unchanged in behaviour — two `match=` strings were updated because the refusal now names the held
claim.

## Next step — not executed

Independent review of this patch. If it passes and no RED issue appears, the intended next
authorisation is the frozen 10-model training run (step B). Not started here.
