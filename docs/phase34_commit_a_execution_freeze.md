# Phase 3.4 — COMMIT A: Execution and Provenance Infrastructure Freeze

**Stage:** A of the audited sequence A–H. **This commit trains nothing and scores nothing.**
**Execution config:** `p34d-exec-e5470b025229f59a06f4c772` (identity unchanged; additive fields only)
**Environment:** `p34c-env-b7a5de2af4c1d9614e49` — CPU, torch 2.14.0+cpu, 8 threads, CUDA unavailable.
**Starting HEAD:** `4e44ee63521da936db473bd6e1d5d61eaf372360`

> No model trained. No inference. No test score. No checkpoint. Ledger 30 → 30. V3 memberships,
> focal cohort, donor sets, backgrounds, estimand, bootstrap, seed, environment and experiment
> identities are untouched — every one is re-verified by test.

---

## 1. What COMMIT A adds

| Component | Location |
|---|---|
| Execution library | `src/experiment/focal_toggle_execution.py` |
| Staged CLI | `scripts/run_focal_toggle_experiment.py` — `plan · train · freeze · authorize · score · analyze` |
| Canonical (LF) hashing + init hash | `src/experiment/provenance.py` |
| Line-ending policy | `.gitattributes` |
| Tests | `tests/test_phase34_commit_a.py` |

Only `plan` was executed for this commit. It confirmed all ten materialised memberships reproduce
their frozen hashes and that the next experiment in the frozen order is `p34d-92fd40fc93c8c8d2e819`
(fold 1 PRESENT).

---

## 2. Training mirrors Protocol A exactly

`train_experiment` is `scripts/run_protocol_a_reproduction.py::stage_train` with two additions and
nothing else: the windowed training set is masked to the frozen arm membership, and the
initialisation hash is captured after model construction and before the first optimiser step. Same
`torch.manual_seed(1995)`, same window-level stratified validation split (`random_state=1995`), same
per-window standardisation, same loaders, same architecture, Adam / 1e-4 / MSE / batch 1024 /
patience 3 / up to 1000 epochs, and the same reproduced upstream final-epoch aliasing.

**Materialisation reproduces the audited set equations:** `B* = official − all(F) − all(D)`,
`PRESENT = B* + focal`, `ABSENT = B* + donor`. All ten membership hashes match
`planned_experiments_v3.csv` (raw SHA-256 over newline-joined sorted ids).

---

## 3. A4 — initialisation hash is now mandatory

SHA-256 over the concatenated initial trainable parameters (named-parameter order, float32 bytes,
name-prefixed), taken **before the first optimiser step**, persisted in every training record.

Verified pre-execution that construction under seed 1995 is deterministic regardless of arm
membership size and regardless of `DataLoader` construction (which consumes no torch RNG). Since
architecture and seed are identical for all ten models, **every model must reproduce**

`d889e90224988ac27bceba086f695ac5d7f8f34ca1e2ad1debc16c6f61384d55`

— pre-registered in the config as `initialization_policy.expected_initialization_sha256`. Paired
PRESENT/ABSENT equality within each fold is therefore implied, and the COMMIT B `freeze` stage
**refuses to freeze** if any fold pair differs. A test injects a mismatched fold and confirms the
refusal fires.

---

## 4. A6 — manifest hash portability, forward fix only

**Cause.** `core.autocrlf=true` with no `.gitattributes` made most text artifacts CRLF in the
working tree after git touched them, so 52 of 73 historical manifest entries hash the CRLF form
while git stores LF.

**Fix, forward only.**

1. `.gitattributes` pins `*.json *.jsonl *.csv *.md *.py *.txt *.yaml *.yml` to `text eol=lf`;
   `*.h5 *.pth *.png *.pdf *.docx` are `binary`.
2. `canonical_sha256()` hashes CRLF→LF-normalised bytes for text artifacts (raw bytes for binary),
   so the value equals the git blob hash on every platform. `hash_form()` classifies a recorded
   hash as `LF`, `CRLF` or `NEITHER`.
3. The COMMIT A manifest and every future manifest use canonical hashes; a test asserts each entry
   reproduces as `LF`.
4. A test asserts that every historical manifest's recorded hashes are byte-identical to the
   values documented in the Phase 3.4E cross-reference.

**Historical manifests from Phase 3.3 through 3.4D are not rewritten.** They remain evidence of
their phases in the form they were recorded; their hash forms are documented in
`reports/phase3_4d_audit/manifest_hash_form_check.csv`. **This is a forward provenance correction,
not retroactive rewriting.**

---

## 5. Gates that enforce the sequence

| Stage | Refuses unless |
|---|---|
| `train` | the experiment is **next in the frozen order**; device and torch match the frozen environment; the materialised membership hash matches the plan. Raises on non-finite loss (frozen policy: STOP). |
| `freeze` (COMMIT B) | all ten training records exist and **every fold pair shares one initialisation hash**; checkpoints on disk match their recorded hashes. Contains no performance. |
| `authorize` (step D) | all ten checkpoints are frozen; the ledger holds **exactly the 30 historical entries**; no Phase-3.4 entry exists. Each entry names its checkpoint hash in `reason`. Appended atomically. |
| `score` (E/F) | the checkpoint on disk matches the frozen hash; a ledger entry names **that exact hash**; no score artifact exists (one look, once). Writes 464 rows atomically and hashes them **before returning**; computes no aggregate. |
| `analyze` (G) | all **4,640** rows exist and every artifact hash-verifies. Computes the frozen primary and secondary statistics from persisted scores only. |

Frozen order: `1P, 1A, 2A, 2P, 3P, 3A, 4A, 4P, 5P, 5A`.

---

## 6. Checkpoint / training-record schema (frozen)

`phase3.4-checkpoint-record-v1`: `experiment_id · execution_position · fold · arm · seed ·
config_identity · execution_environment_id · training_membership_sha256 · training_segments ·
training_windows · validation_windows · architecture_identity · training_configuration ·
initialization_sha256 · checkpoint_path · checkpoint_sha256 · epochs_run · stopped_early ·
best_validation_epoch_zero_indexed · best_validation_loss ·
final_epoch_actually_scored_one_indexed · training_losses · valid_losses · seconds · torch_version ·
git_head_at_training · restart_count · started_utc · finished_utc`.

---

## 7. Score persistence (binding, unchanged)

464 official test segments × 10 models = **4,640 rows**; per row: experiment id, fold, arm, seed,
segment and parent identity, focal status and fold, labels, domain metadata, `total_loss` plus six
per-sensor score columns, checkpoint SHA, evaluator SHA, config identity, test-cohort hash,
membership hash, environment id. Written atomically, hashed, and durable **before** any aggregate.
`analyze` reproduces every reported statistic from these files with no inference.

---

## 8. Ledger

**30 → 30.** COMMIT A freezes the authorisation *machinery* only. The ten entries are written at
step D — after COMMIT B, before the first inference — each naming the exact checkpoint hash it
authorises. No adaptive authorisation is possible: `authorize` refuses unless all ten are frozen.

---

## 9. What remains true

The experiment is a **post-test controlled sensitivity analysis**. Residual parent gap 0–3 (ABSENT
larger) · interval conditional on the ten fitted models · donor matching is metadata-based · 9
sibling-bearing normals outside the focal cohort · 11 of 60 parents contribute one segment ·
76–85% of official training volume. Donor-slot infeasibility for folds 1, 2 and 4 is
**solver-certified under the frozen ILP formulation**; the 69,653-partition search is **search
evidence**.

---

## 10. Next step — not executed

Independent review of COMMIT A. Then, with explicit authorisation, step B:
`python -u scripts/run_focal_toggle_experiment.py train` ten times in the frozen order.
