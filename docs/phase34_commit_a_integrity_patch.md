# Phase 3.4F.1 — COMMIT A Integrity Patch (F1 exposure gate, F2 full-state hash)

**Historical COMMIT A:** `107579fba1f0222519d2acd9f1c78bda78fb65e9` — preserved, not amended.
**Scientific config identity:** `p34d-exec-e5470b025229f59a06f4c772` — unchanged.
**Complete config document SHA-256 (LF-canonical):**
`70b1830e5cf2608dd97c20dda87d32f7a41205873246a74bb99626d31edea965`

> No model trained. No real test inference. No real test score. No checkpoint. Ledger 30 → 30.
> This patch hardens execution and provenance enforcement only; every scientific object of V3 is
> untouched and re-verified by test.

---

## F1 — the test-exposure crash gap, closed

**The gap.** COMMIT A's scorer checked checkpoint hash → ledger authorisation → no existing artifact,
then ran inference and persisted atomically. A crash *after* inference began but *before* the atomic
rename would leave no artifact, so a naive retry would run protected-test inference a second time —
violating *no re-score after exposure*.

**The fix: a durable, append-only exposure history per experiment** at
`reports/phase3_4_results/score_attempts/<experiment_id>.jsonl`.

```
AUTHORIZED → SCORE_STARTED → artifact persisted + hashed → SCORE_COMPLETED
```

- **`SCORE_STARTED`** is appended with `O_APPEND` + `fsync` — durable on disk — **before the first
  protected forward**. It binds experiment id, fold, arm, seed, checkpoint SHA-256, evaluator git SHA,
  config identity, membership hash, test-cohort hash, the ledger authorisation it consumed, and a
  timestamp.
- **`SCORE_COMPLETED`** is appended only after the 464-row artifact is atomically persisted and hashed;
  it binds the checkpoint SHA-256, artifact path, artifact SHA-256, and the row count (expected 464).
- **Any exposure record of any kind → refuse.** The scorer does not infer whether a prior process
  "probably" reached a forward. `SCORE_STARTED` means the look is consumed. A `SCORE_STARTED` without
  `SCORE_COMPLETED` is an **INCOMPLETE** experiment and is never silently re-scored.
- Refusal writes nothing and deletes nothing; no command erases exposure history.
- Aggregates are still never computed inside the scoring step.

The single function that touches protected test data, `_run_protected_inference`, is isolated so the
state machine around it can be tested with a fake that never opens the real cohort.

**Tests (`tests/test_phase34_score_exposure_gate.py`), all against a fake:**

| | Test | Result |
|---|---|---|
| A | cannot begin without ledger authorisation; no forward, no record | pass |
| B | cannot begin if the authorised hash differs; nor if the checkpoint on disk was altered | pass |
| C | `SCORE_STARTED` is on disk when the first forward runs (asserted *inside* the fake), then completes to `[STARTED, COMPLETED]` | pass |
| D | crash after `SCORE_STARTED`, no file → second invocation **refuses**, forward count stays 1 | pass |
| E | crash after partial `.tmp` write, no finalised artifact → second invocation **refuses** with INCOMPLETE | pass |
| F | existing `SCORE_COMPLETED` → second invocation **refuses**; history is byte-identical after refusal | pass |
| G | `official_single_auc` and `parent_concordance_delta` are never called during scoring; no aggregate key in the return | pass |
| H | analysis refuses on a missing artifact and on a hash-mismatched artifact | pass |

---

## F2 — full initial model state

`initialization_sha256` (trainable parameters only) is kept and **unchanged**:
`d889e90224988ac27bceba086f695ac5d7f8f34ca1e2ad1debc16c6f61384d55`.

Added **`initial_state_sha256`** over the **complete** `state_dict()` immediately after construction
and before training — 46 entries: 28 trainable parameters plus 18 buffers (`running_mean`,
`running_var`, `num_batches_tracked` for each of the six BatchNorm layers). Canonical ordering is
`state_dict()` insertion order; per entry the digest absorbs key, dtype string, shape tuple and raw
tensor bytes in the tensor's own dtype. No pickle or serialisation framing enters the hash.

**Pre-registered expected value under the frozen seed, architecture and environment:**
`01d68f698cdf32628604536d282e218106ac48f51fb14d2a74bf667c2c980048`

Verified: stable across constructions; sensitive to a buffer change the parameter hash is blind to;
different under a different seed. **Equality arose naturally** — initialisation was not altered.

**Paired assertions, enforced by the COMMIT B `freeze` stage, which refuses otherwise:**
`PRESENT.initialization_sha256 == ABSENT.initialization_sha256` and
`PRESENT.initial_state_sha256 == ABSENT.initial_state_sha256` per fold, **and** every model equals
both pre-registered values. Tests inject an unequal fold and an off-registration value and prove both
refusals fire.

---

## Config identity — why it is unchanged

`config_identity = "p34d-exec-" + sha256(json.dumps(payload, sort_keys=True))[:24]` where the payload
is exactly: `parent` design config · `family` · `seed` · the ten `planned_experiment_ids` · `env`
id · donor `slots` per fold · `bootstrap` `[fold-stratified, 10000, 27182]`. **Recomputed from these
fields in this patch and it reproduces `p34d-exec-e5470b025229f59a06f4c772`** (test-enforced).

Excluded — and documented in `config_identity_definition.excluded_fields` — are reporting text,
enforcement policy and provenance instrumentation: `claim_boundary`, `initialization_policy`,
`execution_machinery`, `provenance_hash_policy`, `config_identity_definition`, and the like. This is
safe because the payload pins everything that determines *what is trained and scored*; changing an
excluded field cannot alter a training set, a checkpoint, a score or the primary statistic. The
complete document is identified separately by `config_document_sha256` above, so the exact text
remains attributable even though it is outside the scientific identity.

---

## A6 — re-verification, precise terms

For every COMMIT A manifest artifact, `canonical_sha256(working tree)` was compared against the
**SHA-256 of the raw content bytes stored in Git at commit `107579f` for that path** — not the Git
object ID. **11/11 agree**, and each also equals the SHA-256 of the LF-normalised Git content, because
Git stores these files LF under the `.gitattributes` policy. The seven historical manifests are
untouched (test-enforced). The COMMIT A manifest is now itself a historical record: a test asserts
each of its values equals the Git-stored content hash *at commit 107579f*; the current patch state is
covered by `reports/phase3_4_commit_a/patch_f1_f2_manifest.json`, which also carries
`config_document_sha256`.

---

## Unchanged

Memberships (all 10 hashes reproduce), focal cohort, donor sets, backgrounds, PRESENT/ABSENT
construction, seed 1995, environment `p34c-env-b7a5de2af4c1d9614e49`, architecture, optimiser, loss,
validation procedure, preprocessing, execution order `1P 1A 2A 2P 3P 3A 4A 4P 5P 5A`, estimand,
bootstrap, ILP design, claim boundary, test cohort, checkpoint-selection semantics. Ledger **30 → 30**;
protected-test exposure **0 → 0**.

## Next step — not executed

Independent review of this patch. Training remains unauthorised until then.
