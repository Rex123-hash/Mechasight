# Phase 3.3 — Independent Reproduction Audit

**Role:** independent skeptical ML reproduction auditor.
**Scope:** inspection and recomputation from already-persisted Protocol-A outputs only.
**Audited HEAD:** `8add93c3e448d39028934c0e38f0b00c79dcb8da` (branch `master`)

**Not done:** no retraining, no re-scoring, no model inference, no seed/device change, no
preprocessing change, no cohort change, no tolerance change, no ledger modification, no leakage
experiment, no Protocol-B run, no new model.

## Verdict: **YELLOW**

The reproduction itself is technically sound and provenance-safe. I tried to falsify it on twelve
independent fronts and could not: every structural, architectural, configuration, split, cohort and
overlap claim reproduced exactly, and there is no test-integrity violation anywhere in the phase.

Two things prevent GREEN, and neither is "the numbers look wrong":

1. **The headline metrics are not independently re-derivable from the artifacts.** Per-segment
   anomaly scores were never persisted, so §10 of this audit — recompute the six AUCs from saved
   output — *could not be executed at all* without re-running inference, which the audit forbids.
   An audit that cannot re-derive the primary result is not a complete audit.
2. **Three claims in the report narrative overstate what was established.**

Phase 3.3 is **accepted**. The results stand as recorded and nothing needs re-running. The
corrections below are documentation-only.

---

## 1. Repository and provenance

Ancestry is linear and correctly ordered:

`bd61137` (start) → `872f8e5` (freeze) → `4ce1faf` (evaluator locked) → `8add93c` (results)

| Claim | Result |
|---|---|
| All four commits exist, linear ancestry | verified |
| Freeze commit contains **no** performance artifact | verified — config, preprocessing records and code only |
| Working tree clean except the three excluded paths | verified (`reports/supervisor/`, `reports/synopsis/`, `tmp/`) |
| Artifact manifest: 15 declared files + 2 checkpoints | 17/17 hashes match, 0 missing, 0 undeclared |

**The ordering constraint that matters most holds.** The evaluator was committed before the ledger
was authorised, which was before either score existed:

| Event | Local time |
|---|---|
| Evaluator committed (`4ce1faf`) | 16:30:06 |
| Tolerance frozen (`872f8e5`, config) | 16:02:38 |
| Ledger authorisation written | 16:59:26 |
| BrushlessMotor scored | 17:00:15 |
| RoboticArm scored | 20:31:31 |

No score was produced before its evaluator was committed or before its test look was recorded.

Only **two** checkpoints exist anywhere in the repository, one per machine. There are no abandoned
runs, no second scoring attempt, and no undeclared result artifact.

---

## 2. Test-look integrity

| Check | Result |
|---|---|
| Original 28 entries byte-identical | **yes** — prefix SHA-256 `db4bde80…0b0db0`, the same constant as before |
| Exactly two Protocol-A entries appended | yes, one per machine |
| Ledger modified only by appending | yes — the results commit diff is `2 insertions, 0 deletions` |
| Ledger reset or rewritten at any point in git history | **no** — the 28-line prefix hash is constant across every version |
| Test cohort hashes | recomputed from the official test HDF5; both match the ledger exactly |
| Authorisation precedes scoring, per machine | yes (see §1) |

---

## 3. Upstream identity — verified against the live source

I did not take the recorded hashes on trust. I re-hashed the four upstream files three ways and
queried the **GitHub trees API at the pinned commit**:

| File | blob SHA-1 | size | matches live tree |
|---|---|---:|---|
| `preprocess.ipynb` | `a85a7c27…` | 24818 | yes |
| `baseline.ipynb` | `52615b60…` | 7805 | yes |
| `model.py` | `7f756a92…` | 13027 | yes |
| `utilities.py` | `24efb236…` | 17511 | yes |

**The CRLF discrepancy is provenance-only.** The Phase-2 SHA-256 values reproduce *exactly* from
the CRLF-converted bytes of each file, while the raw LF bytes match the pinned tree byte for byte.
The content was never in question; only the recorded hash form was non-portable. No vendoring or
licence issue: no upstream source is vendored, and the pinned tree contains no LICENSE.

---

## 4. Architecture — derived from source, not assumed

Read out of `build_encoder`/`build_decoder` in the pinned `model.py`:

```
encoder: Linear(5836,2048) BN ReLU  Linear(2048,2048) BN ReLU  Linear(2048,2048) BN ReLU  Linear(2048,16)
decoder: Linear(16,2048)   BN ReLU  Linear(2048,2048) BN ReLU  Linear(2048,2048) BN ReLU  Linear(2048,5836)
```

The bottleneck and the output layer carry no BatchNorm and no ReLU — the loop applies them only to
`layer_dims[:-1]`. This matches the claimed structure exactly.

Input dimension derives as `1600·1 + 706·3 + 706·3 = 5836` (upstream truncates `int(fs · 0.1)`, so
1600 and 706, not 1601/707).

**Parameter count, computed two independent ways:**

| Method | Result |
|---|---:|
| Hand arithmetic (per-layer `in·out + out`, plus `3 × 2 × 2048` BatchNorm affine per half) | 40,789,724 |
| `torch` enumeration of `requires_grad` parameters | 40,789,724 |
| Claimed | 40,789,724 |

Exact agreement. BatchNorm `running_mean`/`running_var` are buffers and correctly excluded.

---

## 5. Training configuration

Every parameter verified against `baseline.ipynb` at the pinned commit: Adam, lr `1e-4`,
`utilities.MSE`, batch 1024, `num_epochs` 1000, `patience` 3, `std_window`, `valid_size` 0.1, seed
1995, train loader `shuffle=True`, validation and test `shuffle=False`, and a window-level
stratified `train_test_split` at `random_state=1995`. Aggregation is `'median'`.

**CPU is upstream-authorised.** The notebook carries the comment *"TO BE ADAPTED TO YOUR MACHINE:
either 'mps' or 'cuda' if GPU available, otherwise 'cpu'"* immediately above `'device': 'cuda'`.

That said, the distinction must be stated plainly:

> CPU is an **upstream-compatible device substitution**. It is **not** a bitwise-equivalent
> execution environment. A different device, backend and torch version produce different
> floating-point reduction orders and therefore different weights. Nothing in this phase claims
> otherwise.

`std_window` normalisation is strictly per-window and per-channel, with **no** cross-split
statistics — which also confirms that standardising train and validation separately (the memory
refactor in `4ce1faf`) is exactly equivalent to upstream's combined call.

---

## 6. Best-checkpoint aliasing — the claim is correct

Upstream writes `best_model_state = self.state_dict()` with no `deepcopy`.

I verified the PyTorch semantics empirically rather than asserting them. On a toy model: `state_dict()`
returns **live tensor references**; a subsequent `optimizer.step()` mutates the tensors the saved
dict points at; BatchNorm `running_mean` is mutated too. After later epochs, the "best" state is
element-for-element equal to the current state.

Therefore `load_state_dict(best_model_state)` restores the **final** epoch, not the best-validation
epoch, and the reproduction correctly performs no restore at all.

**The models actually scored are RoboticArm epoch 36 and BrushlessMotor epoch 26** — not the
best-validation epochs 33 and 23. The interpretation is correct; this is **not** a MAJOR finding.

One honest limitation: this is established by code path plus verified PyTorch semantics. The
best-validation weights were never persisted, so it cannot be confirmed by comparing artifacts.

Early-stopping arithmetic checks out on both machines: exactly three non-improving epochs after the
minimum, then the break — consistent with upstream's increment-then-compare-then-check ordering.

---

## 7. Preprocessing and split

All counts reproduce from the persisted records and the windowed HDF5 files:

| | RA train | RA test | BM train | BM test |
|---|---:|---:|---:|---:|
| Segments | 1839 | 464 | 1281 | 312 |
| Windows | 136,086 | 34,336 | 65,331 | 15,912 |
| Skipped | 0 | 0 | 0 | 0 |

Sensor array shapes confirm 16000/7063/7063 Hz at 100 ms with the gyroscope warm-up applied to the
segment start rather than the window length. No BrushlessMotor first-sample correction, no
Protocol-B features, no Protocol-B split, no q90 thresholding appears anywhere in the Protocol-A path.

**Train/validation split independently reproduced** by re-running `train_test_split` on the labels
alone: 122,477/13,609 and 58,797/6,534 — exact.

It is **window-level**, as upstream intends: 1838 of 1839 RoboticArm segments appear in *both*
training and validation. This is faithful, but it means validation loss is optimistic and early
stopping is not clean model selection. Its only consequence for the reported metric is which epoch
training stopped at.

---

## 8. Metric audit — and the gap

**What I could verify:**

- All **42 rows** of `protocol_a_primary_auc.csv` are exactly consistent with `evaluate_*.json` at
  text level (7 score columns × 3 slices × 2 machines, no duplicates, no omissions).
- All **six** published comparisons recompute exactly, including differences and classifications.
- The **AUC estimator is correct**: it matches upstream's brute-force nested loop exactly on
  tie-laden data, and differs from `sklearn` by precisely half the tie fraction — the signature of
  correct strict-tie semantics. On tie-free continuous data it equals `sklearn` to machine precision.
- The evaluator provably loaded the trained checkpoints (evaluate and training SHA-256 agree, and
  both re-hash correctly on disk).
- The masked-sensor scoring path is faithful: upstream's kept block is byte-identical to the
  unmasked input, so the two formulations coincide.

**What I could not verify (F1, MODERATE).** `stage_evaluate` builds the per-segment score table in
memory and returns only aggregate AUCs. No per-window or per-segment score is persisted anywhere.
So the six headline numbers **cannot be re-derived from any committed artifact** without re-running
inference. I verified the estimator and every consistency relation around the numbers, but not the
numbers themselves.

*(A 1-ULP difference appears if the CSV is read with `pandas`; the CSV **text** is exact and
`float(text) == json` holds for all 42 rows. This is a parser artifact, not a data defect.)*

---

## 9. Tolerance freeze

| | |
|---|---|
| Tolerance committed | `872f8e5`, 16:02:38 |
| First score | 17:00:15 — **58 minutes later** |
| Config changed since the freeze | no (empty diff to HEAD) |
| Published reference values in the frozen config | identical to those used in the comparison |

`EXACT 0 | NEAR_EXACT ≤ 0.5 | CLOSE ≤ 2.0 | MATERIAL > 2.0`. Worst observed absolute error **0.33**.
No post-hoc reinterpretation: the band was fixed before any own result existed and was not touched.

---

## 10. RoboticArm overlap — recomputed independently

Recovered parent recordings by stripping the `_<index>` suffix from segment ids and recomputed
everything from the HDF5 files:

| | claimed | recomputed |
|---|---:|---:|
| Parent recordings | 101 | **101** |
| Parents crossing the official split | 62 | **62** |
| Test segments from crossing parents | 212 | **212** |
| Normal test segments with a training sibling | 212/232 | **212/232** |
| Anomaly test segments with a normal training sibling | 0/232 | **0/232** |
| BrushlessMotor crossing parents | 0 | **0** |

The overlap is **asymmetric — it touches the normal class only**. That is exactly the structure
that would produce optimistic bias, which makes the hedged claim ("a credible mechanism; the
magnitude is to be measured") well-founded without licensing any magnitude.

**No `Protocol A − Protocol B` arithmetic exists in any artifact.** I scanned every Phase-3.3 file:
the only matches for inflation-style language are the prohibitions themselves.

Every BrushlessMotor segment is its own parent (1593 parents for 1593 segments), so its zero
crossing count is structurally guaranteed rather than incidental.

---

## 11. BrushlessMotor confound

Preserved everywhere it needs to be: in the report, in the frozen config, and on **all 42 rows** of
`protocol_a_primary_auc.csv` via `scientific_limitation_id`. The 58.98 figure is described as
benchmark discrimination on the released cohort, never as mechanical-fault sensitivity.

---

## 12. Ledger test modification — **CORRECT SEMANTIC GENERALIZATION**

This deserved the most scrutiny, because "an audit-relevant test was loosened after it failed" is
exactly what a retroactive weakening looks like from the outside. It is not one, for four reasons.

**1. The cryptographic guarantee on the original evidence is unchanged.** The new test asserts the
same constant, `db4bde80…0b0db0`, over the same 28 entries. Nothing about their immutability was
relaxed.

**2. Append-only was the pre-existing design, not a Phase-3.3 invention.** `TestLookLedger` exposes
only `append`, `append_many_atomic` and `read` — no delete, no rewrite, no update — and it has been
that way since `4c0be89` (Phase 3.0), long before this phase. The old whole-file hash was an
over-tight *proxy* for immutability: it would have forbidden **any** legitimate future test look by
**any** phase. The invariant, correctly stated, is "the original entries never change", and that is
what the new test asserts.

**3. It cannot be fooled.** I ran the new test against six adversarial mutations:

| Tampering | Result |
|---|---|
| One byte edited inside an original entry | **rejected** |
| Two original entries reordered | **rejected** |
| An original entry deleted | **rejected** |
| Original deleted, then a copy re-appended at the end | **rejected** |
| Prefix tampered **and** a valid Protocol-A entry appended | **rejected** |
| A Protocol-A entry moved into the first 28 | **rejected** |
| A legitimate future append | accepted (intended) |

**4. It is strictly stronger in one dimension.** The new test adds two assertions the old one
lacked: that all 28 Phase-3.2 entries lie *within* the frozen prefix, and that the protocol values
partition cleanly. The old test could not distinguish "Phase-3.2 evidence was edited" from "a later
phase recorded a look"; the new one can.

**Residual gap (F2, MINOR):** nothing pins the *number* of Protocol-A looks. A third undisclosed
Protocol-A entry would not fail the suite. There are exactly two today — I verified — but the
discipline should be enforced by a test, not by inspection.

---

## 13. Protocol-B integrity

| Check | Result |
|---|---|
| Phase-3.2 artifacts | **155/155** unchanged |
| Phase-3.2.1 correction artifacts | present and unchanged |
| Protocol-B ledger entries | 28, byte-identical prefix |
| Protocol-B v3 split manifest | `68e5406d…0d8409e`, unchanged |
| First-test reporting policy | `a1f695d3…36b7da2c`, unchanged |
| Protocol-B models, thresholds or scores altered | **no** |

Phase 3.3's entire footprint outside its own namespace is two files: the ledger (append-only) and
the Phase-3.2.1 test (§12). No Protocol-B result was touched.

---

## 14. Findings

**MAJOR: none.**

| ID | Severity | Finding |
|---|---|---|
| F1 | **MODERATE** | Per-segment scores never persisted; headline AUCs not independently re-derivable from artifacts |
| F2 | MINOR | No test pins the number of Protocol-A test looks |
| F3 | MINOR | "expected from a nondeterministic training run" rests on a single run, on CPU |
| F4 | MINOR | "feature construction" describes a pipeline Protocol A does not use |
| F5 | MINOR | "comparable to the literature" generalises a Protocol-A result to Protocol-B work |
| F6 | MINOR | Protocol-A config not hash-declared in the artifact manifest |
| F7 | MINOR | Checkpoints hashed but not committed; re-verifiable only on this machine |
| F8 | MINOR | `best_epoch` 0-indexed in JSON, 1-indexed in the report |
| F9 | MINOR | Pre-existing `assert … is not 0.0` float identity check in the modified test file |

### Wording corrections required

**F3 — nondeterminism.** Adopt option **B**:

> ~~"The differences are of the size expected from a nondeterministic training run."~~
> **"…consistent with plausible run-to-run variation under the upstream nondeterministic
> implementation."**

Only one run per machine was performed, so no variance was measured and "expected" claims a
distribution that was never estimated. There is a second, sharper problem: **the reproduction ran
on CPU, where the cited cuDNN nondeterminism does not apply.** The dominant driver of the residual
gap is a *different execution environment* — device, backend, torch version — interacting with
patience-3 early stopping and the final-epoch aliasing. That is a stronger and more defensible
attribution than "nondeterminism", and it should replace it.

**F4 — feature construction.** Protocol A uses per-window standardised **raw multi-sensor windows**,
not the Phase-3.1 98-feature representation. Saying "feature construction… is consistent with the
published work" implies the engineered feature pipeline was validated. It was not. Replace with
**"input construction / preprocessing"**.

**F5 — comparability.** "…which makes MechaSight's own results comparable to the literature" is too
strong. Protocol B differs in preprocessing, representation, model family and split; none of it is
exercised here. Narrow to: *establishes a comparability anchor for Protocol-A-style numbers, and
validates this project's dataset handling and official-split evaluation.*

---

## 15. Claims audit

| Claim | Verdict |
|---|---|
| The official baseline reproduces | **SUPPORTED** |
| Differences are expected from nondeterminism | **TOO STRONG** → reword (F3) |
| MechaSight results are comparable to the literature | **TOO STRONG** → narrow (F5) |
| Recording overlap is a credible optimistic-bias mechanism | **SUPPORTED** |
| Protocol A − Protocol B is not a leakage estimate | **SUPPORTED** |
| CPU use is upstream-authorised | **SUPPORTED WITH CAVEAT** — substitution, not equivalence |
| No unresolved discrepancy | **SUPPORTED WITH CAVEAT** — true of the seven logged items; the score-persistence gap is open |
| BM 58.98 is benchmark discrimination, not fault sensitivity | **SUPPORTED** |
| Aliasing means the final epoch was scored | **SUPPORTED** |
| Data path/windowing/feature construction/model/scoring consistent with published work | **TOO STRONG** → reword (F4) |

---

## 16. Next gate — recommended, not designed

The **controlled RoboticArm leakage-magnitude experiment** is the right next gate, and this audit
endorses it. The overlap is now quantified and shown to be asymmetric on the normal class, which
makes the bias mechanism concrete while leaving its magnitude the largest unresolved scientific
question in the project.

It must hold model, preprocessing and representation constant and vary **only** the parent grouping.
Anything less repeats the confound it exists to resolve.

**Not designed or implemented here**, per instruction.

---

## 17. Recommended follow-up (documentation-only, no re-run)

1. Persist the per-segment score table alongside `evaluate_*.json` (F1). This writes
   already-computed values; it requires no re-scoring and no new test look.
2. Apply the three wording corrections (F3, F4, F5).
3. Add a test pinning exactly one Protocol-A look per machine (F2).
4. Add the config SHA-256 to the artifact manifest (F6) and note the checkpoints' local-only status (F7).

None of these change a single reported number.
