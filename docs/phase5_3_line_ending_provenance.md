# Multi-Sensor Fusion Development — L1 Input-Hash Serialisation (Phase 5.3)

<!-- Legacy roadmap name: Phase 5.3 -->

**Decision (independent review, option B):** the L1 bundles carry two documented byte
serialisations of the same committed input files. This is **not** two different scientific
experiments. Each bundle keeps the raw hashes it actually computed; no hash is rewritten.

## Cause

Experiment ids hash `preprocessing_config_hash` and `split_manifest_hash`, which are raw-byte
`file_sha256` values of the files the fit read.

- The main checkout's working copies of these files predate the repository's `eol=lf` policy and
  use **CRLF**.
- A fresh checkout of the fitting commit (`c6798967a34d0cf430782496b1a6139c66de1116`, in a
  detached worktree) writes them with **LF**.

Same git blob, different bytes on disk, different raw hashes, different experiment id.

| Input | Raw SHA-256, CRLF (main checkout, Phase 4, first 14 fits) | Raw SHA-256, LF (fitting worktree) | Git blob |
|---|---|---|---|
| `configs/preprocessing/mechasight_few_shot_regime_s.json` | `c892619a1208007a642a9b94e2c0cf6b1abb891608b9d2ad59e2f6da2ab491b9` | `bb934f9d4d970e3c283fdc038202fd5e86e2fe251d9cf81873babcf467572406` | `0f5094d017b6…` |
| `configs/preprocessing/mechasight_few_shot_regime_t.json` | `950df3c8a0412ae838277cfffe596dfb89bb8ef31d54246d6488075ad30cd25c` | `56dac950944f52686785db6267d2b2c52b081c9cb44b7b4750a03cdeb946ddb9` | `b911a42d788b…` |
| `reports/phase2_2/protocol_b_few_shot_adaptation_v3.csv` | `68e5406d63e96cf5ee418ae58808f5c3e7790889ba1b8497420642d1f0d8409e` | `dedddce2d24abf24d9d498a247fcc7517d1002a143849baf664511db4decc47e` | `f3eb4f331ac1…` |

## Equivalence evidence

For each of the three files:
- the CRLF bytes with CRLF → LF are exactly the LF bytes;
- the git blob is identical at `c679896` and `3a0949d`;
- the parsed contents are identical (both JSON configs; the split manifest's 7,792 rows × 17
  columns compared as text);
- `canonical_sha256` of the CRLF file equals the raw LF hash.

The runner parses these files and never uses their bytes; every other tracked input is verified
with LF-normalised hashes.

Empirically, the worktree fit of RoboticArm S seed 0 reproduces the best validation MSE
(0.7790112328888785) and q90 threshold (0.7871912901537733) of the earlier main-checkout trial of
the same unit exactly.

## Verification

`src/fusion/identity.py`, called by `l1-verify`, checks every bundle against the git objects at
**its own** code commit, not the working tree:
- the experiment id recomputes from the stored config;
- the recorded Phase-5 source hash equals the source hash of that commit;
- each recorded raw input hash is the committed content in LF or CRLF form; anything else is
  `NEITHER`, which fails verification.

Completed fits are identified by scientific unit (machine, regime, family, seed); a second bundle
for a unit is refused. The v2 registry stores no L1 experiment ids for the 24 new fits, so no
predicted id is relied on.

## The ten RoboticArm autoencoder fits

They run from the `c679896` worktree, and their bundles carry the LF form. The ids predicted in
`docs/phase5_3_resource_recovery_note.md` assumed CRLF bytes for every input. For these ten units
they are therefore superseded by the ids actually recorded in the bundles; that note is kept
unchanged as a historical record.
