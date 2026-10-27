# Phase 3.3 — Protocol-A Official Baseline Reproduction: Result

**Config identity:** `p33-protocol-a-87fdd8c09d374556a8ff6ad0`
**Protocol:** `official_baseline_reproduction`, Regime T
**Upstream:** `augustif/IMAD-DS_baseline` @ `592bcbbed58893dc26bbab3240c5d7a3105a1a7f`
**Evaluator locked at:** `4ce1faf5894578f330226d80cd194dfb44ff0cba` (committed before any test score existed)
**Pre-test freeze:** `872f8e503d6b249f5cde31d8a351ef633830448c`

> **Purpose.** This is a **historical comparability anchor**: evidence that this project can
> reproduce the published IMAD-DS baseline on the released split. It is **not** the MechaSight
> deployment protocol and **not** the scientifically preferred evaluation.
> **Protocol B remains the primary leakage-controlled benchmark.**

---

## 1. Headline result

Reproduced against published reference values, using the tolerance frozen in
[§7 of the pre-test freeze](protocol_a_pretest_freeze.md) **before any own score was computed**.

| Machine | Slice | Published | Reproduced | Difference | Classification |
|---|---|---:|---:|---:|---|
| RoboticArm | Overall (S+T) | 91.62 | **91.51** | −0.11 | NEAR_EXACT |
| RoboticArm | Source | 93.28 | **93.14** | −0.14 | NEAR_EXACT |
| RoboticArm | Target | 90.48 | **90.81** | +0.33 | NEAR_EXACT |
| BrushlessMotor | Overall (S+T) | 58.95 | **58.98** | +0.03 | NEAR_EXACT |
| BrushlessMotor | Source | 73.63 | **73.73** | +0.10 | NEAR_EXACT |
| BrushlessMotor | Target | 55.59 | **55.70** | +0.11 | NEAR_EXACT |

**Worst absolute error across all six comparisons: 0.33 AUC points.**
Every comparison falls in the **NEAR_EXACT** band (≤ 0.5). No comparison reaches CLOSE, and none
approaches the MATERIAL_DEVIATION boundary of 2.0.

**Verdict: the published IMAD-DS official baseline reproduces.** The differences are of the size
expected from a nondeterministic training run with patience-3 early stopping (see §5), not from a
methodological disagreement.

*Full per-sensor AUCs — the paper's F-/S- rows for microphone, accelerometer and gyroscope — are in
[`protocol_a_primary_auc.csv`](protocol_a_primary_auc.csv) (42 rows: 2 machines × 7 score columns ×
3 slices).*

### The published values were references, never targets

No parameter was chosen, adjusted, or re-run to move a reproduced number toward a published one.
Each machine was trained exactly once, from the configuration frozen and committed at `872f8e5`,
and scored exactly once. There was no second attempt to compare against.

---

## 2. Training outcome

| | RoboticArm | BrushlessMotor |
|---|---:|---:|
| Train windows | 122,477 | 58,797 |
| Validation windows | 13,609 | 6,534 |
| Input dimension | 5,836 | 5,836 |
| Parameters | 40,789,724 | 40,789,724 |
| Epochs run | 36 | 26 |
| Stopped early (patience 3) | yes | yes |
| Best validation epoch | 33 | 23 |
| Best validation loss | 0.588653 | 0.815143 |
| Wall clock | 3 h 44 m | 22 m 55 s |
| Device | cpu | cpu |
| torch | 2.14.0+cpu | 2.14.0+cpu |

Neither run reached the upstream `num_epochs = 1000`; both stopped early, as upstream intends.

**Checkpoint semantics (upstream defect, reproduced).** Upstream stores `best_model_state =
self.state_dict()`, which is a **reference** to the live tensors, not a copy. The "restored best
model" is therefore the **final** epoch's weights, not the best-validation epoch's. This
reproduction preserves that behaviour deliberately: the scored models are epoch 36 (RoboticArm) and
epoch 26 (BrushlessMotor), **not** epochs 33 and 23. Correcting the bug would have changed the
result and would no longer have been a reproduction.

| Checkpoint | SHA-256 |
|---|---|
| RoboticArm | `67062daf7134d5dbc59fc120570e5c2aba46b39446ab6b553dd6e16a87553964` |
| BrushlessMotor | `70f1da7d0ea837f339790d9ff88da7615f3fc818c0dd5c50736d69fd65351248` |

---

## 3. Test-look discipline

The ledger entry was written **before** the evaluator was allowed to compute a score, and the
evaluator refuses to run without it.

| | |
|---|---|
| Ledger before | 28 entries (all Protocol B) |
| Protocol-A entries added | 2 |
| Ledger after | 30 entries |
| Protocol value | `official_baseline_reproduction` |
| Evaluator git SHA recorded | `4ce1faf5894578f330226d80cd194dfb44ff0cba` |
| Authorisation timestamp | 2026-09-08T11:29:26Z |

| Machine | Experiment id | Official test segments | Test cohort hash |
|---|---|---:|---|
| RoboticArm | `p33a-1cdb02c0d81c21288e58` | 464 | `83cd1195…6a7a8efa` |
| BrushlessMotor | `p33a-65b53c5046de90d1f1a5` | 312 | `1322f463…8ca04f32` |

The 28 Protocol-B entries were neither edited nor deleted; the authorisation script refuses to run
unless exactly 28 are present and no Protocol-A entry already exists. Because Protocol-A and
Protocol-B looks carry different `protocol` values, they can never be confused in later analysis.

**One test look per machine. Both were used. No re-scoring occurred.**

---

## 4. Discrepancies found, and how each was resolved

Seven items are recorded in
[`protocol_a_reproduction_discrepancy_log.csv`](protocol_a_reproduction_discrepancy_log.csv). The
material ones:

**4.1 — The recorded architecture was incomplete.** `docs/research_definition.md` §5.9 describes the
official autoencoder without normalisation layers. Upstream `model.py` inserts **`BatchNorm1d` in
every hidden encoder and decoder block**. The reproduction uses the actual upstream architecture.
*Upstream code wins; the discrepancy is reported rather than silently absorbed.* Had this
reproduction trusted the repository's own record instead of the pinned source, the result would have
been wrong and the cause invisible.

**4.2 — The training schedule was not recorded anywhere in this repository.** `num_epochs = 1000`,
`patience = 3`, shuffle on the training loader only, and the stratified window-level validation
split at `random_state = 1995` were recovered by reading `baseline.ipynb` and `model.py` at the
pinned commit. They could not have been inferred; nothing was guessed.

**4.3 — Best-model aliasing.** Described in §2. Reproduced, not corrected.

**4.4 — Provenance hash form.** The four SHA-256 values in
`reports/phase2/upstream_baseline_provenance.json` do not match the raw upstream bytes; they
reproduce exactly from the **CRLF-converted** bytes, and the git blob SHA-1s match the pinned tree
byte for byte. This is a **Windows line-ending recording defect, not a content discrepancy**. The
historical Phase-2 record was left unedited; the Protocol-A config now records raw SHA-256, CRLF
SHA-256 and git blob SHA-1 for each file. No result is affected.

**4.5 — Device.** Upstream `PARAMS` sets `device = "cuda"`; the upstream notebook comment explicitly
instructs adapting it per machine ("either 'mps' or 'cuda' if GPU available, otherwise 'cpu'"). CPU
was used. The choice is authorised by upstream and is recorded with every result.

**4.6 — Upstream is not deterministic.** It seeds torch and the sklearn split but never disables
cuDNN nondeterminism and never calls `use_deterministic_algorithms`. **No seed was secretly imposed
beyond upstream's own `seed = 1995`.** This reproduction runs in faithful mode and inherits
upstream's nondeterminism; no separate "deterministic mode" was run.

**4.7 — Buffered HDF5 writes.** Upstream resizes the dataset once per window; this implementation
writes in buffered blocks. Throughput only — the stored arrays are identical. Non-behavioural.

---

## 5. Why the residual differences are the expected size

Upstream stops on `patience = 3` without deterministic kernels. A single epoch's difference changes
the final weights, and because of the aliasing in §2 the **final** epoch is what gets scored. A
fraction of an AUC point of run-to-run variation is therefore expected **from the authors' own code
on their own machine**, which is precisely why bitwise equality was rejected as a criterion and the
0.5-point NEAR_EXACT band was frozen in advance. Observing a worst-case 0.33 is consistent with
that reasoning rather than a post-hoc accommodation of it.

---

## 6. Scope limit — no leakage-magnitude claim is made here

The RoboticArm official split contains substantial parent-recording overlap: **101 parent
recordings, 62 of them crossing the official train/test boundary; 212 of 232 normal test segments
have a sibling from the same parent recording in training** (0 of 232 anomaly segments do). See
[`protocol_a_split_overlap_context.csv`](protocol_a_split_overlap_context.csv).

**This overlap was retained deliberately.** Making Protocol A parent-disjoint would defeat its
purpose, which is to reproduce the released evaluation as published.

Recording overlap of this kind **creates a credible mechanism for optimistic bias; the magnitude is
to be measured.** This experiment does not measure it.

**The difference between a Protocol-A number and a Protocol-B number is NOT a leakage estimate, and
is not reported as one anywhere in this phase.** The two protocols differ simultaneously in split,
preprocessing, representation and model family. Their difference confounds all four and isolates
none. Quantifying the effect of parent-recording overlap requires a controlled experiment that holds
model and preprocessing constant and varies **only** the grouping. That experiment is not performed
here, and no arithmetic between the two protocols appears in any artifact of this phase.

---

## 7. BrushlessMotor interpretation limit

The BrushlessMotor result (58.98 overall) is a **published/reproduced benchmark discrimination
figure on the released cohort**. Under limitation
`brushless_motor_acquisition_time_class_confound_v1` it must **not** be described as pure
mechanical-fault sensitivity: normal and anomalous BrushlessMotor material differ in acquisition
time, so an acquisition-condition confound cannot be separated from mechanical fault content on this
cohort. This applies to the reproduced number exactly as it applies to the published one.

---

## 8. Protocol-B integrity

| Check | Result |
|---|---|
| Protocol-B ledger entries | 28, unchanged |
| Protocol-B split manifest `protocol_b_few_shot_adaptation_v3.csv` | hash unchanged |
| Phase-3.2 artifacts re-hashed | 155/155 unchanged |
| Protocol-B models, thresholds or decisions altered | **no** |

Nothing in Phase 3.3 retrained, re-thresholded, re-scored or re-interpreted any Protocol-B result.
Protocol A was added alongside Protocol B, never in place of it.

---

## 9. What this phase does and does not establish

**Establishes.** The published IMAD-DS official baseline reproduces on the released split, to within
0.33 AUC points at worst, from an independent implementation built against the pinned upstream
source. The project's data path, windowing, feature construction, model and scoring are therefore
consistent with the published work, which makes MechaSight's own results comparable to the
literature rather than unmoored from it. Three material parameters absent from this repository's own
records were recovered from the pinned source and are now documented.

**Does not establish.** Nothing about the magnitude of leakage (§6). Nothing about mechanical-fault
sensitivity on BrushlessMotor (§7). Nothing that changes, supports or undermines any Protocol-B
conclusion. Protocol A is a comparability anchor, not a competing benchmark, and the preferred
evaluation for MechaSight deployment claims remains the parent-disjoint Protocol B.

---

## 10. Artifacts

| File | Contents |
|---|---|
| `protocol_a_primary_auc.csv` | 42 rows — all AUCs, all score columns, all slices |
| `protocol_a_published_comparison.csv` | 6 published-vs-reproduced comparisons with classifications |
| `protocol_a_reproduction_discrepancy_log.csv` | 7 discrepancies, each with investigation and resolution |
| `protocol_a_reproduction_report.json` | machine-readable form of this report |
| `protocol_a_ledger_authorization.json` | pre-scoring test-look authorisation record |
| `evaluate_RoboticArm.json`, `evaluate_BrushlessMotor.json` | raw evaluator output |
| `training/*_training.json` | per-machine training records including loss curves |
| `artifact_manifest.json` | SHA-256 of every file above, plus checkpoint hashes |
| `protocol_a_pretest_freeze.md` / `.json` | the pre-test freeze this result was produced under |
