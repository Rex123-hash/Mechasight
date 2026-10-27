# Phase 3.4E — Independent Final Release Audit (RoboticArm Focal-Parent Sibling Toggle V3)

**Role:** independent skeptical ML experiment-release auditor.
**Audited HEAD:** `de2d19673d1c25499d136256d38b4d0195adf5f3`
**V3 execution config:** `p34d-exec-e5470b025229f59a06f4c772`

## Verdict: **YELLOW — EXECUTION AUTHORIZED WITH MANDATORY LIMITATIONS**

**No RED-level scientific or provenance defect exists.** Every structural, membership, balance,
feasibility, estimator and provenance claim in the release record reproduced **exactly** under
independent recomputation.

Two **MODERATE** defects were found. One was corrected: a *mandatory claim qualifier* still described
the superseded V2 parent imbalance — in the wrong direction (§9, A1). One is recorded for forward
repair: every phase manifest from 3.3 through 3.4D hashes CRLF working copies rather than git blobs,
so **52 of 73** manifest hashes are non-portable — content verified intact for every entry, and the
ledger unaffected (§9, A6).

YELLOW reflects genuine residual limitations, **not** the post-test framing on its own.

> No model trained. No inference. No test score. No checkpoint. No ledger change. No membership
> change. No redesign.

---

## 1. Repository and ledger

Linear ancestry confirmed: `6770750` (3.4A) → `0f388b2` (3.4B) → `ee05157` (3.4C) → `de2d196` (3.4D).
Working tree clean except the three permanently-excluded paths.

| | |
|---|---|
| Ledger entries | **30** (28 Protocol-B + 2 Protocol-A) |
| Ledger SHA-256 | `5cef2c4f…f55f297fd` |
| Protocol-B prefix SHA-256 | `db4bde80…0b0db0` — unchanged |
| Phase-3.4 test looks | **0** |
| Phase-3.4 checkpoints / score artifacts | **0 / 0** (only the two Phase-3.3 checkpoints exist) |

---

## 2. Membership reconstruction and paired-arm purity — **PASS**

Each fold was rebuilt from official training metadata and compared to the frozen memberships. All
set equations hold exactly:

`B* = official − all segments of F − all segments of D` · `PRESENT = B* + focal` ·
`ABSENT = B* + donor`

| Fold | \|F\| | \|D\| | \|B*\| | PRESENT | ABSENT | seg Δ | win Δ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12 | 13 | 1033 | 1405 | 1405 | 0 | 0 |
| 2 | 12 | 15 | 1036 | 1410 | 1410 | 0 | 0 |
| 3 | 12 | 12 | 1224 | 1507 | 1507 | 0 | 0 |
| 4 | 12 | 13 | 1067 | 1430 | 1430 | 0 | 0 |
| 5 | 12 | 12 | 1336 | 1564 | 1564 | 0 | 0 |

Every purity check passes in every fold: `F ∩ D = ∅`; the focal contribution is **all** of `F`'s
segments; `B*` excludes both parent sets; no focal segment reaches ABSENT; no donor segment reaches
PRESENT; focal parents are wholly absent from ABSENT and donor parents wholly absent from PRESENT; no
duplicate identities; every identity belongs to official training. **Any failure here would have been
RED. There were none.**

---

## 3. Focal cohort — exact

62 crossing parents · **2** excluded (`20240222_11_14_08_dataLog`, `20240405_13_09_52_dataLog`, each
holding 2 of only 3 training segments in a target stratum) · **60** eligible · 12 per fold, each
appearing exactly once · **203** focal normal test segments · **9** excluded sibling-bearing normals ·
**232** fixed anomaly reference. The focal parent set is **identical to V2**, so the primary estimand
is untouched.

---

## 4. Balance — every claim reproduced

| Fold | Donors | PRESENT parents | ABSENT parents | Gap (P−A) | Strata mismatches | Target P/A | Volume |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 13 | 63 | 64 | −1 | 0 | 22 / 22 | 76.4% |
| 2 | 15 | 61 | 64 | −3 | 0 | 21 / 21 | 76.7% |
| 3 | 12 | 64 | 64 | 0 | 0 | 19 / 19 | 81.9% |
| 4 | 13 | 63 | 64 | −1 | 0 | 23 / 23 | 77.8% |
| 5 | 12 | 64 | 64 | 0 | 0 | 22 / 22 | 85.0% |

**Sign convention: PRESENT − ABSENT. A negative value means ABSENT holds MORE parents.** ABSENT is
the larger arm in folds 1, 2 and 4; the arms are equal in folds 3 and 5.

All **25** strata equal in every fold, plus every marginal (`split_label`, `domain_shift_op`,
`domain_shift_env`). Target-domain contributions exactly equal — **no directional target deficit**.
Segment and window differences are identically zero (every segment is 74 windows).

**Kish ESS gaps** — V2 `7.1839, 4.5736, 5.8496, 7.3464, 6.2496` (mean 6.2406) → V3 `0.3104, 1.9781,
0.6049, 0.7064, 0.8075` (mean 0.8815). **Reduction 85.9%** — the claimed "~86%" is numerically
correct.

**Clarification frozen:** "zero metadata cost" means **zero paired-arm metadata imbalance**, *not*
identity with the full official training distribution — both arms train on 76.4–85.0% of the official
1,839 segments.

---

## 5. ILP verification — infeasibility is a **proof**

Formulation audited independently: one binary variable per candidate donor parent; pool = 76 − 12
focal = **64**; slot constraint an **equality** `Σx_p == S`; stratum constraints
`Σ cap[p,s]·x_p ≥ need_s`; focal parents removed from the pool before building the model; zero
objective for pure feasibility.

Reproduced feasibility by slot count:

| Fold | 12 | 13 | 14 | 15 | Minimum |
|---:|:--:|:--:|:--:|:--:|---:|
| 1 | ✗ | ✓ | ✓ | ✓ | **13** |
| 2 | ✗ | ✗ | ✗ | ✓ | **15** |
| 3 | ✓ | ✓ | ✓ | ✓ | **12** |
| 4 | ✗ | ✓ | ✓ | ✓ | **13** |
| 5 | ✓ | ✓ | ✓ | ✓ | **12** |

Exactly matches the release record.

**It is a proof, not a timeout.** scipy status **2** with HiGHS `model_status = Infeasible`; status 1
(iteration/time limit) was never returned. For **fold 2 the LP relaxation is also infeasible** — an
independently checkable certificate, since with no fractional solution there can be no integer one.
Folds 1 and 4 have feasible relaxations, so their infeasibility rests on integrality and is
established by branch-and-bound exhaustion over a bounded 64-variable binary problem.

---

## 6. Alternative-partition claim — **SEARCH EVIDENCE, not proof**

69,653 partitions were tested against a space of **2.750 × 10³⁶** distinct 5×12 partitions —
a fraction of **2.5 × 10⁻³²**.

> Future reports must **not** describe the absence of an all-feasible partition as proven. Only the
> per-fold ILP infeasibility at 12 slots is proven.

---

## 7. Primary estimand and bootstrap — verified against an independent implementation

Compared the frozen module against a reference written from the specification with explicit loops and
no shared code, on a deliberately **unbalanced** synthetic cohort (60 parents, 1–8 segments each):

- point estimate matches to **1e-12**; every parent delta matches;
- **parent-equal weighting** confirmed, and demonstrably different from segment weighting
  (0.090962 vs 0.093850) — no segment-level pseudo-replication;
- the segment-weighted mean of `C_i` equals the strict pairwise focal AUROC exactly;
- invariance under monotone within-arm transforms (`×1000`, `+7`, `exp`): `max |ΔC_i| = 0`;
- exact ties score zero, matching upstream.

**Bootstrap:** fold-stratified, 12 parents resampled within each of the 5 folds, 10,000 replicates,
seed 27182, percentile 95. Fold membership preserved; sibling segments move with their parent; folds
are never treated as N=5 inferential evidence; the interval is flagged conditional on the ten fitted
models. Module and reference intervals agree to three decimals — exact equality is not expected, as
the two implementations consume the random stream in different order.

**The primary estimand did not change from Phase 3.4C.**

---

## 8. Estimator supersession — **VALID VERSIONED SUPERSESSION**

| Check | Result |
|---|---|
| A. old hash recorded in the release record | yes |
| B. recoverable from `ee05157` | yes |
| C. matches the Phase-3.4C historical manifest | **yes** (`080cf513…`) |
| D. current matches Phase-3.4D provenance | **yes** |
| E. change limited to the mandated bootstrap finalisation | yes |
| F. primary estimand silently changed | **no** |
| Phase-3.4C historical manifest edited | **no** |

The diff touches only the docstring, a `stratify_by_fold` parameter, the result flag and the
bootstrap block. No line computing `C_i`, `delta_i`, the within-parent MEAN or the point estimate
changed. The Phase-3.4C manifest reporting 11/12 is correct and expected.

---

## 9. Findings

| ID | Severity | Status | Finding |
|---|---|---|---|
| **A1** | **MODERATE** | **CORRECTED** | mandatory claim qualifier described the superseded V2 imbalance, in the wrong direction |
| A2 | MINOR | recorded | `config_identity` does not cover `claim_boundary`, so A1 was undetectable by identity comparison |
| A3 | MINOR | no action | score-persistence requirement is referenced by pointer, not inlined — but is fully frozen and hash-verified |
| A4 | MINOR | recommended | no initialization hash is required, so identical paired initialisation is policy, not artifact |
| A5 | MINOR | clarified | ledger-authorisation timing was under-specified operationally |
| **A6** | **MODERATE** | **recorded — fix forward** | 52 of 73 manifest hashes across seven phase manifests are CRLF working-copy hashes, not git-blob hashes; content intact, ledger clean |

### A1 — corrected

The frozen qualifier read *"the ABSENT arm contains **12 fewer** physical parents by construction."*
Under V3 the gap is **0 to 3 in the opposite direction** (ABSENT larger). Wrong in both magnitude and
direction, it would have attached a false limitation to every reported result. The config already
contradicted itself: `residual_confounds` and `paired_arm_audit` carried the correct V3 figures.

Corrected to: *"a residual parent gap of 0 to 3 remains, with the ABSENT arm holding MORE parents than
PRESENT (fold gaps PRESENT minus ABSENT = −1, −3, 0, −1, 0); exact equality is proven unreachable at
12 donor slots for folds 1, 2 and 4."* The superseded text and reason are recorded in
`claim_boundary.audit_correction`. **`config_identity` is unchanged**, and no membership, estimand,
seed, environment or ledger content was touched.

Not RED: a documentation defect in an unexecuted config, correctable before execution.

### A6 — manifest hash portability (systemic, content intact)

Every artifact manifest from Phase 3.3 through 3.4D records SHA-256 over **working-tree bytes**. On
this Windows checkout (`core.autocrlf=true`, no `.gitattributes`), text files become CRLF once git
touches them, so most manifest hashes are of the CRLF form while git stores LF. Cross-referencing all
**73** entries against their committed blobs: **21 LF, 52 CRLF, 0 neither** — every entry matches its
blob *modulo line endings*, so there is **no content drift**. Full table in
`manifest_hash_form_check.csv`.

This is the same defect class the Phase-3.3 audit flagged in the Phase-2 upstream provenance record —
a **recurrence**, which is why it is MODERATE rather than MINOR despite intact content.

**The ledger is clean:** LF in both working tree and blob, and the recorded 28-entry prefix hash
matches the blob form exactly. The most important provenance object is portable.

Historical manifests were **not** edited. Required forward fix: a `.gitattributes` pinning text
artifacts to LF; every future manifest hashing LF-normalised bytes so it equals the git blob hash; a
test enforcing that. **This audit's own manifest hashes LF-normalised bytes**, setting the precedent.

A side effect worth stating: my initial config-supersession check reported that the blob recovered
from `de2d196` did *not* match the 3.4D manifest. That was A6, not a content discrepancy — the
manifest held the CRLF form. The record is corrected accordingly.

### A4 — recommended addition to execution requirements

Persist an **initialization hash** per experiment (SHA-256 over the concatenated initial parameter
tensors, taken immediately after model construction and before the first optimizer step) and assert
equality within each fold pair. Cheap, needs no training to specify, and converts the paired-
initialisation claim from policy into something checkable.

### A5 — ledger timing clarified

Documents agree on the binding requirement (**all 10 authorisations exist durably before the first
test inference**) but did not name the operational point. COMMIT A lists "ledger-authorisation
machinery" — the tooling, not the entries.

**Clarified: authorisation is written and committed as its own step after COMMIT B and before the
first inference.** Writing it after the checkpoint freeze lets each entry name the exact checkpoint
hash it authorises — strictly more provenance than authorising before training. Scientific
requirement unchanged.

---

## 10. Recommended execution sequence

1. **COMMIT A** — implementation, evaluator, memberships, environment freeze, experiment identities,
   ledger-authorisation machinery.
2. **TRAIN** all 10 models in the frozen order `1P, 1A, 2A, 2P, 3P, 3A, 4A, 4P, 5P, 5A`. No test
   inference.
3. **COMMIT B** — all checkpoint hashes, training records, initialization hashes. No performance.
4. **AUTHORISE** — atomically write and commit all 10 test-look authorisations, each naming the
   checkpoint hash it authorises, **before** the first inference.
5. **SCORE** the 10 frozen checkpoints exactly once.
6. **PERSIST** and hash the score artifact atomically **before** any aggregate.
7. **COMPUTE** the frozen metrics only from the persisted scores.
8. **COMMIT** results.

Hard rule retained: no model may be scored whose checkpoint hash is not already frozen in COMMIT B.

---

## 11. Policies verified

**Environment** `p34c-env-b7a5de2af4c1d9614e49` — live torch 2.14.0+cpu, `cuda_available False`,
threads 8, matching the frozen record exactly. **No CUDA was installed later.** Mid-phase switching
prohibited.

**Planned matrix** — exactly 10 rows = 5 folds × 2 arms × 1 seed; 10 unique experiment ids and 10
unique membership hashes; one PRESENT and one ABSENT per fold; all seed 1995; all
`PLANNED_NOT_EXECUTED`.

**Checkpoint policy** — all 11 required metadata fields present (plus the A4 recommendation); 10
checkpoints retained through independent audit.

**Score persistence** — binding; 464 × 10 = **4,640** rows; every required column present; atomic
hashed write **before** aggregates; schema committed and hash-matching.

**Failure policy** — no ambiguity: mechanical restart only before test exposure with the same
identity; no re-score after exposure; NaN stops the phase; environment change stops the phase;
unexpected early stopping is not a failure; partial completion reported as incomplete.

**Claim boundary** — all 11 required limitations present after the A1 correction.

**Config supersession** — the V3 config changed under A1. Its Phase-3.4D manifest entry is now
historical (like the estimator's Phase-3.4C entry); `config_identity` is unchanged; the change is
confined to `claim_boundary`. Recorded in `provenance_supersession_check.json`.

---

## 12. Remaining limitations (must accompany every result)

Residual parent gap 0–3, ABSENT larger · interval conditional on the ten fitted models, estimating no
training-model, seed or population uncertainty · donor material differs in signal content, matching is
metadata-only · 9 sibling-bearing normal segments outside the focal cohort · 11 of 60 focal parents
contribute a single normal test segment · training volume 76–85% of official.

---

## 13. Execution authorised

**YES**, subject to the mandatory limitations above and the A4 recommendation. Next step is execution
authorisation and COMMIT A — **not performed here**.
