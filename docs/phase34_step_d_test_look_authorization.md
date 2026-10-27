# Phase 3.4H — Step D: Ten Protected Test Looks Authorised

**Stage:** D of the audited sequence A–H. **Ledger 30 → 40. Nothing scored.**
**Config identity:** `p34d-exec-e5470b025229f59a06f4c772` · **Evaluator SHA bound in every entry:** `c10fdfcc890f97a595c132d87dc8b4bb04d7161e`

> **INCIDENT P34-STEP-D-INC-1 (see §4).** After the authorisation, one protected-score claim
> (`p34d-92fd40fc93c8c8d2e819`, fold 1 PRESENT) was consumed by the *test harness*, not by the
> scorer being misused and not by a protected forward. Zero protected-test inference ran. Under
> the frozen F1 policy the look is INCOMPLETE and the claim is retained. Not resolved unilaterally.

---

## 1. The atomic authorisation

`run_focal_toggle_experiment.py authorize --evaluator-sha c10fdfc…` built ten records under
`build_authorisations`, which refused to proceed unless all ten checkpoints were frozen (hash on
disk = COMMIT B record), the ledger held exactly the 30 historical entries, and no Phase-3.4 entry
existed. `TestLookLedger.append_many_atomic(records, expected_existing_count=30)` then appended
all ten in one durable write at `2026-09-11T19:49:04.480058Z`.

| # | Experiment | Fold | Arm | Checkpoint SHA-256 named in `reason` |
|--:|---|--:|---|---|
| 1 | `p34d-92fd40fc93c8c8d2e819` | 1 | PRESENT | `85386f2df404…` |
| 2 | `p34d-e56529a2f56f98fd4a7a` | 1 | ABSENT | `4b4d848b74d8…` |
| 3 | `p34d-fd6e3bfd4c091e155d99` | 2 | ABSENT | `1a964dedde00…` |
| 4 | `p34d-a4e8316f62fa982ec52d` | 2 | PRESENT | `2c8029ad1a7b…` |
| 5 | `p34d-88b93ddbacbc5829f326` | 3 | PRESENT | `c6b30ee064a0…` |
| 6 | `p34d-2af84983abd170013b73` | 3 | ABSENT | `0c0a46d234d2…` |
| 7 | `p34d-652bc1e3f9d29a51a460` | 4 | ABSENT | `826035dccd5b…` |
| 8 | `p34d-406b59f08e154b284159` | 4 | PRESENT | `d756eac00a03…` |
| 9 | `p34d-66656cf0901b180cf037` | 5 | PRESENT | `7a2b9911e264…` |
| 10 | `p34d-ed8467570acf122c36cd` | 5 | ABSENT | `5a49306aae72…` |

Every entry: `protocol = roboticarm_focal_parent_sibling_toggle_v1`, `regime = T`,
`model_config_identity = p34d-exec-e5470b025229f59a06f4c772`, `test_cohort_hash = 83cd1195…`,
`git_sha = c10fdfc…`, `metrics_requested = [focal_parent_concordance_delta, focal_auroc_delta,
per_segment_scores_persisted]`, and a `reason` binding fold, arm, seed 1995, the membership hash
and the exact checkpoint hash. No entry carries a numeric performance value.

## 2. Ledger verification

- 40 entries = 28 Protocol-B + 2 Protocol-A + 10 Phase-3.4; the first 30 are byte-identical to the
  pre-step-D ledger (SHA-256 before `5cef2c4f…`, after recorded in the step-D manifest).
- The 28-entry Protocol-B prefix hash `db4bde80…` is unchanged.
- 10 unique experiment ids; 10 unique checkpoint hashes; each equals the COMMIT B record and the
  file on disk; each equals the id/hash pair in `test_look_authorization.json`.

## 3. Stage-transition test updates — the exact change

Nine tests pinned the pre-step-D invariant (ledger 30, no Phase-3.4 protocol). Each now asserts
the post-step-D invariant with a stage note: ledger 40; exactly ten Phase-3.4 entries, one per
frozen experiment, each naming its checkpoint hash, config identity, cohort hash and evaluator
SHA; superseded V1 protocol value still absent; historical prefix byte-stable. The COMMIT B
manifest is now tested as a historical record against Git content at `c10fdfc` (its own test file
is listed in it), and the current state is covered by `step_d_authorization_manifest.json`.
No exposure guard was weakened.

## 4. Incident P34-STEP-D-INC-1 — full disclosure

**What happened.** While running the full suite to find stale ledger=30 tests,
`tests/test_phase34_commit_a.py::test_score_refuses_before_any_claim_or_exposure` (the COMMIT B
version) called `fx.score_experiment(fold-1 PRESENT)` against the **real** results root with
`_run_protected_inference` replaced by a sentinel that raises. Before step D that call refused at
the ledger gate. After step D the ledger gate legitimately passed, so the scorer — behaving exactly
as frozen — acquired the real O_EXCL claim and appended `SCORE_STARTED` durably, then called the
sentinel, which raised `AssertionError("forward ran")` **before any data was read**.

**Evidence.** `score_claims/p34d-92fd40fc93c8c8d2e819.claim` (`CLAIMED`, `2026-09-11T19:50:25.721473Z`,
81 s after the authorisation); `score_attempts/p34d-92fd40fc93c8c8d2e819.jsonl` with history
`[SCORE_STARTED]` and no `SCORE_COMPLETED`; no `scores/` directory; no score artifact anywhere.
A subsequent invocation refuses: *"the protected-score claim is already held (exposure history:
['SCORE_STARTED']). The look is consumed; without SCORE_COMPLETED the experiment is INCOMPLETE and
must not be re-scored."*

**Protected-test exposure.** Zero forwards. The sentinel replaced the only function that touches
test data; the attempt record proves the state machine stopped at `SCORE_STARTED`.

**Policy applied.** Frozen F1/F3: the claim and the attempt record are **retained** (never deleted),
no stale-lock recovery is implemented, no re-score is attempted. The exposure test now pins this
exact state so the claim cannot silently vanish, no other claim can appear before step E, and the
consumed look can never reach `SCORE_COMPLETED`.

**Harness fix (the cause).** The defective test is confined to `tmp_path` (a copy of the real
training record + a 30-entry ledger). `tests/conftest.py` adds an autouse guard that refuses any
claim, exposure-record write or ledger append inside the real repository paths. The scorer is
unchanged.

**Status.** INCOMPLETE for fold-1 PRESENT pending independent review. Options are for the reviewer,
not this stage: (a) accept INCOMPLETE and analyse folds 2–5 only (4 folds, pre-registered
estimand unchanged, reduced N); (b) a versioned, independently reviewed decision that recognises
the exposure record as proof of zero forwards and authorises a single scoring of the retained
claim under a new explicit rule. Neither has been taken.

## 5. Next step — not executed

Independent review of this commit and of the incident. Step E (score each authorised checkpoint
exactly once) is **not** authorised by this document.
