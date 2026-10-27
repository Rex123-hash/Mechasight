# Phase 3.3.1 — Protocol-A Audit Closure

**Role:** conservative scientific-maintenance engineer.
**Input:** Phase-3.3 independent audit, verdict **YELLOW**, results **ACCEPTED**.
**Layer type:** additive. No historical Phase-3.3 artifact was rewritten.

## Two separate judgements

| | |
|---|---|
| **Scientific result status** | **ACCEPTED** — unchanged by this phase |
| **Auditability status** | **YELLOW** — one MODERATE finding cannot be closed |

These are deliberately reported apart. The reproduction is sound; what is missing is the ability to
*re-derive* its headline numbers from artifacts. Not every accepted experiment is GREEN, and forcing
GREEN here would misrepresent the record.

**Nothing in this phase ran a model.** No inference, no re-scoring, no retraining, no seed/device/
checkpoint/preprocessing/cohort change, no ledger mutation, no leakage experiment.

---

## 1. The score-artifact recovery search — the decisive question

The audit's one MODERATE finding (F1) is that per-segment scores were never persisted. Before
accepting that as permanent, I searched read-only for an authentic artifact from the original
scoring run. Recovering one would have been allowed; **generating one would not.**

| Location | Method | Result |
|---|---|---|
| `reports/` | grep for the seven score column names | 4 matches, **all aggregate-only** |
| `data/interim/protocol_a/` | file listing + HDF5 dataset enumeration | 6 files; HDF5 holds only the 3 sensor + 5 label arrays written at preprocessing |
| `tmp/` | full listing | synopsis rendering assets only |
| temporary working / task-output directories | grep for score columns and segment-id patterns | source code only; the 3 CSVs there are Phase-3.2 **Protocol-B** audit artifacts |
| Git history, all refs | `--diff-filter=D` for deleted score artifacts | never committed, never removed |
| Stash + dangling objects | `git fsck --lost-found`, inspected every blob | all Phase-2 era source files |
| Checkpoint-adjacent | listing beside both `.pth` files | nothing but the HDF5 inputs |

**Root cause.** `stage_evaluate` builds the grouped per-segment score table in memory and returns
only the aggregate AUC dictionary. The table is never written.

**Outcome: no authentic original score artifact exists.** Nothing was fabricated and no placeholder
CSV was created.

Recorded permanently, in these words:

> Per-segment Protocol-A scores were not persisted during the original locked test run and cannot be
> reconstructed without re-inference.

### F1 final status: **OPEN_ACCEPTED_LIMITATION**

Not RESOLVED — that would require the original values, which do not exist.
Not RED — nothing suggests the recorded aggregates are wrong. The estimator was independently
validated against upstream's brute-force nested loop; all 42 CSV rows are exactly consistent with
the evaluator output; both cohort hashes recompute; the evaluator provably loaded the trained
checkpoints. **What is missing is re-derivability, not correctness.**

A future deliberate re-inference against the frozen checkpoint could produce a reproducibility
artifact — but it would be a **new test access**, requiring separate authorisation and versioning.
It was not done here, and its output would be new values, not recovered ones.

---

## 2. Reporting corrections

Three claims were corrected by supersession in
[`protocol_a_reporting_corrections.md`](../reports/phase3_3_1/protocol_a_reporting_corrections.md).
The historical report is untouched; the new wording governs future citation.

| Finding | Was | Now |
|---|---|---|
| F3 | "expected from a nondeterministic training run" | "consistent with plausible variation arising from the different execution environment and the upstream training procedure, including patience-based early stopping and final-epoch checkpoint aliasing" |
| F4 | "…windowing, **feature construction**, model and scoring…" | "dataset handling, **input construction/preprocessing**, official baseline architecture and scoring" — and explicitly: this does **not** validate the Phase-3.1 98-feature representation |
| F5 | "makes MechaSight's own results comparable to the literature" | "establishes a historical comparability anchor for Protocol-A-style results" — and explicitly: it does **not** establish numerical comparability between Protocol-B results and the published baseline |

The F3 correction matters most: the reproduction ran on **CPU**, so the cuDNN nondeterminism
originally cited does not apply to it at all. The honest attribution is the execution-environment
difference interacting with patience-based stopping and the reproduced checkpoint aliasing.

**No reported number changed.**

---

## 3. Test-look count invariant (F2 — RESOLVED)

Added `tests/test_phase331_audit_closure.py`, which separates two things the old coverage conflated:

**Permanent historical assertion** — Phase 3.3 took exactly two Protocol-A looks, named by
experiment id (`p33a-1cdb02c0d81c21288e58` → RoboticArm, `p33a-65b53c5046de90d1f1a5` →
BrushlessMotor), one per machine, no duplicate ids, both carrying config identity
`p33-protocol-a-87fdd8c09d374556a8ff6ad0` and regime `T`. This constrains *those* looks, so a future
explicitly authorised protocol may append its own without this test needing to be relaxed.

**Current-state snapshot** — 30 entries = 28 Protocol-B + 2 Protocol-A, with a docstring stating
plainly that a future authorised protocol is expected to update this count *in the same commit that
authorises those looks*. It is a snapshot, not a permanent ceiling.

Plus: the original 28 entries remain a byte-identical prefix, and the Phase-3.3 result artifacts are
re-hashed against their own manifest to prove this correction layer changed none of them.

---

## 4. Config provenance (F6 — RESOLVED)

The Protocol-A config governs every Phase-3.3 result but was absent from the Phase-3.3 artifact
manifest. The historical manifest was **not** edited. The supplement records:

| | |
|---|---|
| Path | `configs/phase3/protocol_a_official_reproduction_v1.json` |
| SHA-256 | `4bc61b56af6dc42528edb704baf32e96f83be4f35e1afc94a4faa1610375e928` |
| Identity | `p33-protocol-a-87fdd8c09d374556a8ff6ad0` |
| Committed at | `872f8e5` (pre-test freeze) |
| Unchanged since freeze | yes — empty diff to HEAD |

Git already provided historical immutability for this file. The supplement closes the
**artifact-manifest asymmetry**, putting the config on the same footing as every other artifact.

---

## 5. Checkpoint preservation (F7 — DOCUMENTED AS REPRODUCIBILITY RISK)

| | RoboticArm | BrushlessMotor |
|---|---|---|
| Path | `data/interim/protocol_a/RoboticArm/protocol_a_baseline_seed1995.pth` | `data/interim/protocol_a/BrushlessMotor/protocol_a_baseline_seed1995.pth` |
| Exists | **yes** | **yes** |
| Size | 163,273,117 B (163.3 MB / 155.7 MiB) | 163,273,117 B (163.3 MB / 155.7 MiB) |
| Recorded SHA-256 | `67062daf…553964` | `70f1da7d…351248` |
| Current SHA-256 | `67062daf…553964` | `70f1da7d…351248` |
| Matches record | **yes** | **yes** |
| Tracked by Git | no | no |

**Storage status: LOCAL-ONLY. No durable backup has been verified, and none is invented here.**

**Classification: reproducibility risk.** If these files are lost, the Phase-3.3 result cannot be
re-scored without retraining — and retraining would not reproduce the same weights, because of the
execution-environment difference, patience-based early stopping, and the final-epoch aliasing.

The files were **not** committed: ~326.5 MB (311.4 MiB) combined is not appropriate to add to the repository
automatically. Recommended mitigation is archiving both to durable storage outside the working tree
and recording the destination in a later supplement.

---

## 6. Epoch indexing (F8 — RESOLVED)

Verified from the actual training JSON; the historical JSON is unchanged.

| Machine | JSON `best_epoch` (0-indexed) | Human report (1-indexed) | Epochs run | **Epoch actually scored** |
|---|---:|---:|---:|---:|
| RoboticArm | 32 | **33** | 36 | **36** |
| BrushlessMotor | 22 | **23** | 26 | **26** |

The third column is the best-*validation* epoch. The last column is what was actually scored —
they differ because the reproduced upstream aliasing means the final epoch's weights are the ones
restored. Both facts are correct simultaneously, which is exactly why the indexing ambiguity was
worth removing.

---

## 7. Float-identity test correction (F9 — RESOLVED)

In `tests/test_phase321_reporting_correction.py`:

```python
# before
assert corrected.precision is not 0.0  # noqa: F632 - identity check is deliberate
# after
assert not (isinstance(corrected.precision, float) and corrected.precision == 0.0)
```

The test is named *"undefined precision is never silently replaced by zero"*, and `precision` is
`float | None`. The old form compared **object identity** between unrelated float objects, which
CPython does not guarantee in either direction.

I demonstrated the weakness concretely rather than asserting it. On a non-interned float — e.g. one
parsed from a stored artifact via `json.loads("0.0")` — the old expression evaluates to `True`, so
the assertion **passes and the substitution bug slips through**. The replacement evaluates to
`False` and fails, catching it. This is a genuine strengthening, not a cosmetic edit.

---

## 8. Integrity verification

| Check | Result |
|---|---|
| Phase-3.3 artifacts re-hashed against their own manifest | **15/15 unchanged** |
| Test-look ledger | 30 → **30**, byte-identical, not touched |
| Original 28-entry prefix SHA-256 | `db4bde80…0b0db0` — unchanged |
| Protocol-B / Phase-3.2 artifacts | 155/155 unchanged |
| Protocol-B v3 split manifest | `68e5406d…0d8409e` unchanged |
| First-test reporting policy | `a1f695d3…36b7da2c` unchanged |
| Model inference run | **no** |
| Test score regenerated | **no** |

---

## 9. Remaining open limitations

1. **F1 — score re-derivability (OPEN_ACCEPTED_LIMITATION).** The six headline AUCs cannot be
   independently re-derived from artifacts. Closing it requires a separately authorised
   re-inference, which would be a new test access.
2. **F7 — checkpoints are LOCAL-ONLY (reproducibility risk).** Present and intact today; no durable
   backup verified.
3. **The magnitude of the RoboticArm overlap effect remains unmeasured** — deliberately, and
   unchanged by this phase.

---

## 10. Next scientific gate — not executed

**Controlled RoboticArm leakage-magnitude experiment.** It must hold model, preprocessing and
representation constant and vary **only** the parent grouping; anything less repeats the confound it
exists to resolve.

Not designed, not implemented, not started here.
