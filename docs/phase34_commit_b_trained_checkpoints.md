# Phase 3.4G — Step B / COMMIT B: Ten Frozen V3 Checkpoints

**Stage:** B–C of the audited sequence A–H. **Training complete; nothing scored.**
**Config identity:** `p34d-exec-e5470b025229f59a06f4c772` · **Environment:** `p34c-env-b7a5de2af4c1d9614e49`
**Training ran at HEAD:** `e93886c1fa40575b8e966dd22946b2203a01b491` (recorded in every training record)

> No protected-test inference. No protected-test score. No test-look authorisation. Ledger 30 → 30.
> Score claims / attempts / artifacts: 0 → 0. No Phase-3.4 performance was inspected.

---

## 1. Training matrix — 10 / 10, in the frozen order

Executed `1P 1A 2A 2P 3P 3A 4A 4P 5P 5A`, one model at a time, each gated by the `train` stage
(next-in-order, frozen device/torch, membership hash re-materialised and matched). Sequence ran
2026-09-11 10:10:33Z → 17:42:24Z; **7.44 h** of training, **0 restarts**, **0 failures**, every loss
finite, every run stopped by the frozen patience-3 rule between epochs 30 and 39.

| Pos | Experiment | Fold | Arm | Epochs | Best-val (1-idx) | Scored epoch | Min | Checkpoint SHA-256 |
|---:|---|---:|---|---:|---:|---:|---:|---|
| 1 | `p34d-92fd40fc93c8c8d2e819` | 1 | PRESENT | 39 | 36 | 39 | 49.2 | `85386f2df404…` |
| 2 | `p34d-e56529a2f56f98fd4a7a` | 1 | ABSENT | 35 | 32 | 35 | 42.8 | `4b4d848b74d8…` |
| 3 | `p34d-fd6e3bfd4c091e155d99` | 2 | ABSENT | 34 | 31 | 34 | 40.1 | `1a964dedde00…` |
| 4 | `p34d-a4e8316f62fa982ec52d` | 2 | PRESENT | 37 | 34 | 37 | 43.8 | `2c8029ad1a7b…` |
| 5 | `p34d-88b93ddbacbc5829f326` | 3 | PRESENT | 30 | 27 | 30 | 39.0 | `c6b30ee064a0…` |
| 6 | `p34d-2af84983abd170013b73` | 3 | ABSENT | 30 | 27 | 30 | 39.9 | `0c0a46d234d2…` |
| 7 | `p34d-652bc1e3f9d29a51a460` | 4 | ABSENT | 39 | 36 | 39 | 51.8 | `826035dccd5b…` |
| 8 | `p34d-406b59f08e154b284159` | 4 | PRESENT | 38 | 35 | 38 | 47.9 | `d756eac00a03…` |
| 9 | `p34d-66656cf0901b180cf037` | 5 | PRESENT | 34 | 31 | 34 | 46.2 | `7a2b9911e264…` |
| 10 | `p34d-ed8467570acf122c36cd` | 5 | ABSENT | 38 | 35 | 38 | 46.4 | `5a49306aae72…` |

"Scored epoch" is the final epoch, per the reproduced upstream best-checkpoint aliasing. Full SHA-256
values, paths and byte sizes are in `reports/phase3_4_results/commit_b_training_summary.json` and
`commit_b_manifest.json`.

---

## 2. Initialisation provenance — every model, both hashes, pre-registered values

All ten training records carry

- `initialization_sha256 = d889e90224988ac27bceba086f695ac5d7f8f34ca1e2ad1debc16c6f61384d55`
- `initial_state_sha256 = 01d68f698cdf32628604536d282e218106ac48f51fb14d2a74bf667c2c980048`

exactly as pre-registered before any model existed. Paired equality therefore holds in **5 / 5**
folds for both hashes, and the ten **final** checkpoint hashes are all distinct — the arms started
from byte-identical weights and buffers and diverged only through their training memberships.
Nothing was adjusted to obtain this.

---

## 3. Freeze stage — passed

`run_focal_toggle_experiment.py freeze` (schema `phase3.4-commit-b-freeze-v2`) verified: all ten
records exist; every checkpoint on disk matches its recorded hash; paired initialisation and
initial-state hashes equal in every fold; every model equals both pre-registered values;
`contains_test_performance: false`. Independently re-verified: record ids equal the frozen matrix;
fold/arm/seed match; all ten membership hashes re-materialise; config identity and environment id in
every record; live environment equals the frozen record (torch 2.14.0+cpu, CUDA unavailable, 8
threads); ledger 30 with prefix hash unchanged; zero score claims, attempts, artifacts and analyses.

---

## 4. Stage-transition test update — the exact change

Three tests asserted the **pre-training** condition *"no Phase-3.4 checkpoint exists."* That was
correct until step B was authorised and is no longer a valid current invariant. It was **not** solved
by deleting the checks. The current post-training invariant, now enforced by
`test_only_the_ten_frozen_checkpoints_exist_and_exposure_is_zero`, is:

- exactly the ten frozen checkpoints exist on disk — no extra, none missing;
- each maps to exactly one frozen experiment; path and SHA-256 match its training record;
- all ten experiment ids match the frozen matrix; fold/arm/seed agree; membership hashes reproduce;
- every `initialization_sha256` and `initial_state_sha256` equals the pre-registered value;
- **and still:** zero score claims, zero `SCORE_STARTED`/`SCORE_COMPLETED`, zero score artifacts, zero
  aggregate result artifacts, zero authorisation records; ledger 30.

Two further gate tests had also become stale for a subtler reason: `authorize` and `score` now pass
their *first* gate (checkpoints exist) and refuse at a *later* one. They were rewritten to prove
each gate individually against temporary fixtures, plus one check against the **real** post-training
state that `score` refuses at the ledger gate **before** any claim — with a sentinel that would fail
the test if a forward ever ran. That check left exposure at zero.

The three earlier "no checkpoint" tests carry a stage note. Historical manifests (3.4C, 3.4D, 3.4E,
COMMIT A, F1/F2, F3) are untouched; the F3 manifest is now tested as a historical record against
Git-stored content at `e93886c`, and the current state is covered by `commit_b_manifest.json`.

---

## 5. What COMMIT B contains

Ten training records · `commit_b_checkpoint_freeze.json` · `commit_b_environment.json` ·
`commit_b_training_summary.json` · `commit_b_manifest.json` (canonical hashes; every checkpoint by
path, SHA-256, fold, arm, seed, both initial hashes and membership hash) · the stage-transition test
updates · this document. The `.pth` files (1.6 GB) remain git-ignored by frozen policy; their exact
paths and SHA-256 values are recorded durably above.

**No test performance.** The training records contain training/validation loss curves only.

---

## 6. Next step — not executed

Independent COMMIT B review. Then, with explicit authorisation, step D: append all ten test-look
entries atomically, each naming its frozen checkpoint hash. Only after that may step E score each
checkpoint exactly once.
