# Phase 3.3 — Protocol-A Pre-Test Reproduction Freeze

**Config identity:** `p33-protocol-a-87fdd8c09d374556a8ff6ad0`
**Config:** `configs/phase3/protocol_a_official_reproduction_v1.json`
**Upstream:** `augustif/IMAD-DS_baseline` @ `592bcbbed58893dc26bbab3240c5d7a3105a1a7f`
**Protocol:** `official_baseline_reproduction`, Regime T

> **Purpose.** Reproduce the published IMAD-DS official baseline faithfully, as a
> **historical comparability anchor**. This is not the MechaSight deployment protocol, and it is
> not the scientifically preferred evaluation. **Protocol B remains the primary
> leakage-controlled benchmark.**

**At freeze time no official-test score exists.** The official test *windows* have been built —
that is preprocessing — but no model has been applied to them, no metric has been computed, and the
evaluator refuses to run without a Protocol-A ledger entry.

---

## 1. Upstream code identity — verified, with a correction

The pinned commit was re-fetched and checked against the GitHub tree at that commit.

| File | git blob SHA-1 (pinned tree) | matches download |
|---|---|---|
| `preprocess.ipynb` | `a85a7c27b0d979cc11c8b9c979ec0fc248013be0` | yes |
| `baseline.ipynb` | `52615b6089d09db6456b99e5010d745bd7608b4f` | yes |
| `model.py` | `7f756a9232d958dc6a9a8ae7c7a61564e9cae101` | yes |
| `utilities.py` | `24efb2366bbd186219477251f76324b7d97dbbe0` | yes |

Commit `592bcbbe…` exists, dated 2026-04-28T20:28:35Z, message "updated readme.md", tree
`471d3bce…`. **All four files match the pinned tree byte for byte.**

### Discrepancy found and resolved

The four SHA-256 values recorded in `reports/phase2/upstream_baseline_provenance.json` do **not**
match the raw upstream bytes. They are reproduced **exactly** by the **CRLF-converted** form of the
same bytes.

**Conclusion:** the Phase-2 provenance hashes were computed on Windows after line-ending
conversion. The content is identical; only the recorded hash form was non-portable, so those hashes
could never be reproduced on a POSIX system or against the canonical Git object. This is a
provenance-recording defect, **not** a content discrepancy, and it does not affect any Phase-2
finding. The Protocol-A config now records the raw SHA-256, the CRLF SHA-256 and the git blob SHA-1
for each file. The historical Phase-2 record is left unedited.

*(This is the same class of issue the recovery document already flagged for Phase-3.1 source
hashing: "Hash construction uses Windows-native relative path strings; the identity is not a
portable all-dependency environment hash.")*

**Licence note.** The pinned upstream tree contains **no LICENSE file**. Upstream source is
therefore **not vendored** into this repository. Only its verified identity and behaviour are
recorded, and the reproduction is independently implemented in `src/reproduction/protocol_a.py`.

---

## 2. Parameters recovered that the repository did not hold

Reading the pinned source recovered several training parameters that were **not** recorded anywhere
in this repository. They could not have been inferred safely.

| Parameter | Value | Previously in repo? |
|---|---|---|
| `num_epochs` | **1000** | **No** |
| `patience` | **3** (early stopping on validation loss) | **No** |
| shuffle (train / valid / test) | **True / False / False** | No |
| validation split | sklearn `train_test_split`, **window level**, stratified by `anomaly_label + domain_shift_op + domain_shift_env`, `random_state=1995` | partially (window-level was known) |
| optimizer | `torch.optim.Adam`, defaults apart from `lr` | Adam only |
| **BatchNorm1d in every hidden block** | **present** | **No — omitted from the recorded architecture** |
| checkpoint selection | see §4 | "best model = lowest validation loss" (incorrect in effect) |

> The only epoch figure previously in the repository is **50**, and
> `docs/source_verification.md` correctly records that it belongs to a *different* framework
> described in the upstream README whose files do not exist in the repository. It is **not** the
> notebook that produced the published results, and it was not used.

### Correction to the recorded architecture

`docs/research_definition.md` §5.9 records:

> `FC(ΣL_s·C_s → 2048) → FC(2048 → 2048) → FC(2048 → 2048) → FC(2048 → 16)` (ReLU), mirrored decoder

The actual upstream architecture inserts **`BatchNorm1d`** in every hidden block:

```
encoder:  [Linear(5836,2048) → BatchNorm1d(2048) → ReLU]
          [Linear(2048,2048) → BatchNorm1d(2048) → ReLU]
          [Linear(2048,2048) → BatchNorm1d(2048) → ReLU]
          Linear(2048,16)                      # no BatchNorm, no activation
decoder:  [Linear(16,2048)   → BatchNorm1d(2048) → ReLU]
          [Linear(2048,2048) → BatchNorm1d(2048) → ReLU]
          [Linear(2048,2048) → BatchNorm1d(2048) → ReLU]
          Linear(2048,5836)                    # no BatchNorm, no activation
```

This is a **material omission** in the earlier summary — BatchNorm changes training dynamics and
makes inference depend on running statistics. The reproduction uses the actual upstream
architecture. The historical document is **not** rewritten; the correction is recorded here.

---

## 3. Frozen configuration

| Group | Value |
|---|---|
| Windows | 100 ms, non-overlapping, timestamp-based |
| Rates (fixed, **not** measured) | mic 16000, acc 7063, gyro 7063 Hz |
| Window lengths | `int(fs × 0.1)` → 1600 / 706 / 706 |
| Channels | 1 / 3 / 3 → concatenated input dim **5836** |
| Segment start | `max(gyro_first + 35 ms, max(all first timestamps))` |
| Segment end | `min(all last timestamps)`; count `(end − start) // 100 ms` |
| Slicing | pandas label-based, **right-inclusive** |
| Length handling | zero-pad if short, truncate if long |
| Normalisation | `std_window`, per window per channel, ε in the denominator, applied **independently** to train / valid / test |
| Architecture | FC autoencoder, `layer_dims [2048, 2048, 2048, 16]`, BatchNorm1d + ReLU in hidden blocks, ≈40.8 M parameters |
| Optimizer | Adam, lr 1e-4 |
| Loss | MSE per sample over dim 1, then batch mean |
| Batch | 1024 |
| Epochs | up to 1000, early stop at patience 3 |
| Validation | 10 %, window level, stratified, `random_state 1995` |
| Seed | 1995 (`torch.manual_seed`, `torch.cuda.manual_seed`, sklearn `random_state`) |
| Score | per-window reconstruction MSE, higher = more anomalous |
| Segment score | **median** of the segment's window scores |
| AUC | upstream `calculate_single_auc`: pair counts 1 only when `normal < anomaly`; **exact ties score zero** |
| Slices | all (S+T), source, target |

### Deliberately **not** inherited from Protocol B

measured machine-specific rates · BM first-accelerometer-sample correction · half-open windowing ·
native-rate processing · the 98 engineered features · feature-median aggregation · train-only
feature z-scoring · Isolation Forest · One-Class SVM · the Protocol-B common evaluation cohort ·
parent-disjoint splits · the q90 threshold policy.

---

## 4. Upstream defects reproduced, not corrected

| Defect | Effect | Reproduced |
|---|---|---|
| `best_model_state = self.state_dict()` stores **live tensor references**, not a copy, so the "best" state keeps mutating; the final `load_state_dict` restores the **last** epoch's weights | the evaluated model is the final epoch, not the best-validation epoch | **yes** |
| right-inclusive datetime slicing | adjacent windows can share a boundary sample | yes |
| window-level train/validation split | windows from one recording land on both sides | yes |
| fixed 7063 Hz applied to both machines | wrong for BrushlessMotor (measured 7135.7 Hz) | yes |
| bare `try/except` around segment reads | unreadable segments silently skipped | behaviour preserved; every skip is recorded rather than only printed |

Correcting any of these would change the result and defeat the purpose of the reproduction.

### Intentional non-behavioural departures

* HDF5 rows are written in buffered blocks of 512 rather than one `resize` per window. The stored
  arrays are identical; only write throughput differs.
* Skipped segments are recorded in JSON as well as printed.

---

## 5. Official split — overlap retained deliberately

| | RoboticArm | BrushlessMotor |
|---|---:|---:|
| Parent recordings | **101** | 1593 |
| Official train / test segments | 1839 / 464 | 1281 / 312 |
| Train / test parents | 76 / 87 | 1281 / 312 |
| **Parents crossing the official split** | **62** | **0** |
| Test segments from crossing parents | 212 | 0 |
| **Normal test segments with a training sibling** | **212 / 232** | 0 / 156 |
| **Anomaly test segments with a normal-training sibling** | **0 / 232** | 0 / 156 |

The RoboticArm asymmetry is reproduced exactly as previously audited. Because training is
normal-only, normal test segments can have near-twins in training while anomalous ones cannot.

> **Permitted language:** recording overlap creates a credible mechanism for optimistic bias; the
> magnitude is to be measured.
> **Not permitted:** "the official AUC is inflated."

Protocol A **must** preserve this overlap — making it parent-disjoint would destroy the historical
reproduction it exists to provide.

---

## 6. Preprocessing result — exact window-count fidelity

| Machine | Split | Segments | Windows | Expected | Skipped |
|---|---|---:|---:|---:|---:|
| RoboticArm | train | 1839 | **136 086** | 1839 × 74 | 0 |
| RoboticArm | test | 464 | **34 336** | 464 × 74 | 0 |
| BrushlessMotor | train | 1281 | **65 331** | 1281 × 51 | 0 |
| BrushlessMotor | test | 312 | **15 912** | 312 × 51 | 0 |

Every count matches exactly, and **no segment was skipped**, so the upstream bare-except path never
triggered. Train counts confirm Regime T: RA 1839 = 1812 source + 27 target normals; BM 1281 =
1263 + 18.

Usable span per segment after the 35 ms gyro crop: RA 7.4649 s → 74 windows; BM 5.1473 s → 51
windows. Both match the Phase-1 audit figures.

---

## 7. Reproduction tolerance — frozen before any own result

| Class | Absolute error (AUC points) |
|---|---|
| EXACT | 0.00 |
| **NEAR-EXACT** | **≤ 0.5** |
| CLOSE | > 0.5 and ≤ 2.0 |
| MATERIAL DEVIATION | > 2.0 |

**Justification, fixed in advance.** Upstream seeds `torch` and the sklearn split but does **not**
set `cudnn.deterministic` or `use_deterministic_algorithms`, so its own GPU runs are seeded but not
bitwise reproducible. Early stopping at patience 3 means a single epoch's difference changes the
final weights outright. Run-to-run variation of a fraction of an AUC point is therefore expected
**from the authors' own code**, and bitwise equality is not a reasonable criterion. 0.5 points is
adopted as the near-exact band and 2.0 as the material-deviation boundary.

This tolerance was written into the config and this report **before any MechaSight Protocol-A test
score was computed.**

### Determinism stance

Upstream is *seeded but not deterministic*. Faithful reproduction preserves that. An optional
deterministic mode (`use_deterministic_algorithms`) is **not** used, because imposing determinism
that upstream does not have would not be the same experiment.

Device is an environment parameter: the upstream notebook comments explicitly instruct
*"TO BE ADAPTED TO YOUR MACHINE: either 'mps' or 'cuda' if GPU available, otherwise 'cpu'"*, so the
device choice is authorised by upstream itself and is recorded with the result.

---

## 8. Published reference values — comparison targets, never tuning targets

| Machine | Overall (S+T) | Source | Target |
|---|---:|---:|---:|
| RoboticArm | 91.62 | 93.28 | 90.48 |
| BrushlessMotor | 58.95 | 73.63 | 55.59 |

Status: `REFERENCE_ONLY_NOT_A_TUNING_TARGET`. These are the authors' own published numbers. No
Protocol-A choice may be changed to move towards them; any correction must be justified as
restoring upstream fidelity, never as improving agreement.

---

## 9. Scope limit — no leakage-magnitude claim

Protocol A alone does **not** isolate the effect of parent-recording overlap. Protocol A and
Protocol B differ simultaneously in split, preprocessing, model family, representation and
aggregation. Therefore:

> **"Protocol A AUC − Protocol B AUC = leakage inflation" is not a valid inference and will not be
> made.**

A controlled experiment holding model and preprocessing constant while varying only the grouping
mechanism would be required. That experiment is **not** performed in this phase.

---

## 10. BrushlessMotor confound

The acquisition-time/class confound (`brushless_motor_acquisition_time_class_confound_v1`) applies
to Protocol A exactly as it does to Protocol B: all released normal recordings precede all released
anomalous recordings with no overlap, so fault state and acquisition session are not independently
identifiable.

**Permitted:** "published/reproduced benchmark discrimination on the released BrushlessMotor
cohort." **Forbidden:** any description of a BrushlessMotor AUC as pure mechanical-fault
sensitivity.

---

## 11. Protocol-B integrity at freeze

| Check | State |
|---|---|
| Protocol-B ledger entries | 28, unchanged |
| Protocol-A ledger entries | 0 at freeze |
| Phase-3.2 artifact hashes | 155 / 155 verified |
| Frozen v3 split manifest | unchanged |
| Protocol-B thresholds / models | untouched |

Protocol-A ledger entries carry `protocol = "official_baseline_reproduction"`, which the existing
ledger schema already supports as a field, so the 28 Protocol-B records remain distinguishable and
intact without any schema change.
