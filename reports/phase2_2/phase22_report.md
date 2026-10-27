# Phase 2.2 — target-adaptation budget correction

## STATUS

**COMPLETE — INDEPENDENTLY ACCEPTED / GREEN (recorded 2026-09-01).** Protocol correction only. No
model was trained, no Phase-3 model experiment started, and no UI was built. Smoke-level
preprocessing validation only; the complete corpus was not built.

## WHY PARENT CLOSURE DOES NOT IMPLY TRAINING ON ALL SIBLINGS

Parent-recording independence is a constraint on the **evaluation** cohort. Its content is: *a
physical recording that contributes training information must not also contribute test segments,
because sibling excerpts of one continuous acquisition share the same machine state, mounting, and
background mix.*

That constraint has exactly one forced consequence — the 96 RoboticArm sibling segments of the 18
adaptation parents must leave the common test set. It says nothing about where they go next.
Blocking a segment from evaluation and feeding it to a model are independent decisions:

- **Blocking** removes an optimistic bias from the *measurement*.
- **Training** changes the *treatment* being measured.

Phase 2.1 performed the first and, as a side effect, also performed the second: it made the whole
parent closure the Regime-T training pool, so RoboticArm would have adapted on 123 target segments.
IMAD-DS deliberately releases only 27 RoboticArm target-normal training segments. Adapting on 123
answers a different question — *what if we had 4.6× the released few-shot budget?* — while being
reported as the released-budget result. A segment used by nothing leaks nothing, so withholding the
96 costs no validity; consuming them costs the experiment its meaning.

Phase 2.2 therefore keeps the block and drops the training, and records four quantities that must
never be collapsed into one another.

## FINAL ROBOTICARM ROLE COUNTS

| Primary role | Split label | Parents | Segments |
|---|---|---:|---:|
| `source_training` | `train` | 44 | 1528 |
| `source_validation` | `validation` | 6 | 186 |
| `target_adaptation` | `train` (T) / `reserved_adaptation` (S) | 18 | **27** |
| `adaptation_sibling_blocked` | `blocked_sibling` | 18 (the same 18) | **96** |
| `common_evaluation` | `test` | 33 | 466 |

Adaptation budget, stated in the four required terms:

| Quantity | Value |
|---|---:|
| `target_adaptation_observed` | 27 |
| `target_adaptation_parent_groups` | 18 |
| `target_adaptation_blocked_siblings` | 96 |
| `target_adaptation_parent_closure` | 123 |

The 18 blocked-sibling parents are **the same 18 physical recordings** as the adaptation parents,
not additional ones; `blocked_parents_subset_of_adaptation_parents` is asserted true.

## FINAL BRUSHLESSMOTOR ROLE COUNTS

| Primary role | Split label | Parents | Segments |
|---|---|---:|---:|
| `source_training` | `train` | 1137 | 1137 |
| `source_validation` | `validation` | 126 | 126 |
| `target_adaptation` | `train` (T) / `reserved_adaptation` (S) | 18 | **18** |
| `adaptation_sibling_blocked` | `blocked_sibling` | 0 | **0** |
| `common_evaluation` | `test` | 312 | 312 |

| Quantity | Value |
|---|---:|
| `target_adaptation_observed` | 18 |
| `target_adaptation_parent_groups` | 18 |
| `target_adaptation_blocked_siblings` | 0 |
| `target_adaptation_parent_closure` | 18 |

One segment per recording, so closure equals allocation and nothing is blocked.

## REGIME-S TRAIN COUNT

| Machine | Parents | Segments | Composition |
|---|---:|---:|---|
| RoboticArm | 44 | **1528** | source normals only |
| BrushlessMotor | 1137 | **1137** | source normals only |

Regime S additionally *reserves* the released allocation unused (RA 18 parents/27 segments; BM
18/18) and blocks the siblings (RA 18 parents/96 segments).

## REGIME-T TRAIN COUNT

| Machine | Parents | Segments | Composition |
|---|---:|---:|---|
| RoboticArm | 62 | **1555** | 1528 source + **27** released target |
| BrushlessMotor | 1155 | **1155** | 1137 source + **18** released target |

Verified from the frozen manifest, not from arithmetic: RA `T` train rows = 1555 = source_training
(1528) + 27, and the assertion `ra_t_train != ra_source_train + 123` is executed and passes.

## TARGET-ADAPTATION CONSUMED COUNT

RoboticArm **27**, BrushlessMotor **18**. Asserted to be exactly the set of released target-normal
segments (`file_domain == target`, `file_split == train`, `anomaly_label == normal`) — not a
superset, not a subset — and every consumed segment is asserted to belong to one of the 18
adaptation parents.

## BLOCKED-SIBLING COUNT

RoboticArm **96** segments across the 18 adaptation parents; BrushlessMotor **0**.

## PROOF BLOCKED SIBLINGS APPEAR NOWHERE ELSE

Machine-readable: `blocked_sibling_isolation_proof.csv`, `phase22_validation.json`.

| Machine | Blocked | in S/train | in S/validation | in S/test | in T/train | in T/validation | in T/test |
|---|---:|---:|---:|---:|---:|---:|---:|
| RoboticArm | 96 | 0 | 0 | 0 | 0 | 0 | 0 |
| BrushlessMotor | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Four independent layers enforce this:

1. **Construction** — blocked siblings receive `split = blocked_sibling`, a label outside
   `{train, validation, test}`, so no split-selecting consumer can reach them.
2. **Assertion** — `assert_few_shot_adaptation_cohorts` fails if any blocked segment id appears in
   a consuming split or in any other primary role, and if any blocked segment's parent is not an
   adaptation parent.
3. **Pipeline guard** — `iter_processed_windows` raises if asked to materialise a non-consuming
   split, so a blocked sibling cannot enter a cache, feature file or manifest even by direct
   request. Exercised as a test and in validation.
4. **Smoke evidence** — no smoke manifest contains a blocked segment id, asserted over all six
   builds.

Their sole recorded purpose is enforcing parent independence; the evaluation policy records
`blocked_sibling_use` with every consumer flag `false`.

## PROOF S/T COMMON TEST REMAINS IDENTICAL

| Machine | S test segments | T test segments | S/T symmetric difference | v2→v3 symmetric difference |
|---|---:|---:|---:|---:|
| RoboticArm | 466 | 466 | **0** | **0** |
| BrushlessMotor | 312 | 312 | **0** | **0** |

`TEST_SEGMENT_IDS_S == TEST_SEGMENT_IDS_T` and `TEST_PARENT_RECORDINGS_S ==
TEST_PARENT_RECORDINGS_T` hold, and `assert_common_test_unchanged` additionally proves the v3
cohort is bit-for-bit the Phase-2.1 v2 cohort for both regimes and both id fields. No blocked
sibling was moved into test; the correction changed training/adaptation role labels only.

## MANIFEST / CONFIG VERSION CHANGES

| Artifact | Phase 2.1 | Phase 2.2 |
|---|---|---|
| Split version | `protocol_b_common_test_v2` | `protocol_b_few_shot_adaptation_v3` |
| Preprocessing version | `phase2.1-v2` | `phase2.2-v3` |
| Regime-S config id | `phase2.1-v2-8688f5639d9fa7c1` | `phase2.2-v3-dcef3e9c758c24ae` |
| Regime-T config id | `phase2.1-v2-f497d27caee67b4a` | `phase2.2-v3-0a0285c9afd6ad6d` |
| Evaluation policy | `protocol_b_common_test_v2.json` | `protocol_b_few_shot_adaptation_v3.json` |

Both new config ids are asserted distinct from all four historical ids (v1 S/T and v2 S/T), so a
Phase-2.1 training cache cannot be mistaken for the corrected primary Regime-T definition:
`validate_cache_identity` rejects it on the HDF5 header, and `merge_protocol_b_assignments` rejects
a v2 split manifest under a v3 config. New manifest role vocabulary: `source_training`,
`source_validation`, `target_adaptation`, `adaptation_sibling_blocked`, `common_evaluation`, with
exactly one primary role per segment.

**Historical artifacts intact.** `reports/phase2/`, `reports/phase2_1/`, both older config pairs and
`configs/evaluation/protocol_b_common_test_v2.json` are unmodified — verified by `git status` and by
a test asserting the v2 freeze still records 123 RoboticArm adaptation segments. The previous
interpretation is superseded in writing, not erased.

## FILES MODIFIED

New:

- `src/preprocessing/splits.py` additions — `few_shot_adaptation_assignments`,
  `adaptation_budget_summary`, `assert_few_shot_adaptation_cohorts`,
  `assert_common_test_unchanged`, `CONSUMING_SPLITS` / `NON_CONSUMING_SPLITS` / `PROTOCOL_ROLES`
- `configs/preprocessing/mechasight_few_shot_regime_s.json`, `..._t.json`
- `configs/evaluation/protocol_b_few_shot_adaptation_v3.json`
- `scripts/build_phase22_splits.py`, `scripts/validate_phase22.py`
- `tests/test_phase22_adaptation_budget.py`
- `reports/phase2_2/` — split freeze, counts, adaptation budget, role counts, isolation proof,
  design json, validation json, this report

Modified:

- `src/preprocessing/splits.py` — `assert_no_parent_leakage` gains an `ignore_splits` keyword
  (default unchanged/strict, so every Phase-2/2.1 call site behaves exactly as before)
- `src/preprocessing/pipeline.py` — refuses to materialise non-consuming splits
- `scripts/preprocess_imadds.py` — optional `--data-pool` role filter for role-targeted builds
- `docs/evaluation_protocol.md` §3, §4.4, §7, §12; `docs/preprocessing_phase2.md`;
  `docs/leakage_risks.md`; `README.md`

## TESTS ADDED / UPDATED

Nineteen new tests in `tests/test_phase22_adaptation_budget.py`: budget-quantity separation (both
machines), Regime-T consuming released-not-closure, blocked siblings absent from every consuming
split, consumed set equal to the released allocation, one-role-per-segment, common test identical to
v2, three tamper-rejection tests (sibling→train, sibling→test, full parent-closure adaptation),
deterministic rebuild, leakage-check semantics for non-consuming splits, pipeline refusal, config
pairing/invalidation, v3 rejecting v2 manifests, evaluation-policy content, and Phase-2.1 artifact
preservation. No existing test was weakened or deleted.

## TOTAL TESTS PASSING

**141 passed** (122 before Phase 2.2; +19). Phase-2 and Phase-2.1 validation scripts re-run and
still report `PASS`.

## SMOKE VALIDATION RESULTS

Six one-segment builds under `data/interim/phase22_smoke_release/`, **6 segments / 375 windows**:
common-evaluation S and T for both machines (4 builds) plus a `target_adaptation` role-filtered
Regime-T build per machine (2 builds). All assertions pass:

- cache identity matches the v3 config id in every build;
- S/T common-test physical windows are identical field-by-field (segment, parent, index, start/end,
  domain, condition, background, label, raw file references);
- both adaptation builds contain only released target-normal segments;
- no smoke artifact contains a blocked sibling;
- the pipeline refuses `split = blocked_sibling`.

`phase22_validation.json` records **24 passing assertions**. The complete corpus was not built.

## INDEPENDENT DISAGREEMENTS / CHALLENGES

1. **The leakage assertion had to be refined, and this deserves scrutiny.** `assert_no_parent_leakage`
   previously meant "no parent spans two split labels". Under v3 an adaptation parent legitimately
   holds `train` and `blocked_sibling` rows, so the check now runs over consuming splits only. I
   believe this is a strengthening of precision rather than a weakening of protection — a segment
   consumed by nothing cannot transport information — but it is the one frozen protection whose
   *implementation* changed, so it is flagged rather than buried. The strict default is retained and
   still exercised by a test; only v3 call sites opt in.
2. **RoboticArm Regime T now adapts on 27 segments across 18 parents.** That is the released budget
   and the correct primary experiment, but it is a very small adaptation set, and a null S-vs-T
   result will be weak evidence of "adaptation does not help" rather than strong evidence. The
   documented (unauthorised) 0/27/123 budget study is the natural way to separate "adaptation does
   not help" from "27 examples are not enough"; Phase 3 should not silently reach for the 123 to
   make a result appear.
3. **96 RoboticArm segments are now used by nothing at all.** This is deliberate and correct, but it
   should be stated plainly in any write-up: the protocol discards usable normal data to protect the
   evaluation cohort, and a reader comparing raw dataset sizes will not otherwise see it.
4. **RA target-only evaluation stays thin** (20 normal vs 116 anomaly segments) and RA validation
   still has only six independent parents. Both were disclosed in Phase 2.1 and are unchanged here.
5. **The BrushlessMotor class/session confound is unchanged.** Phase 2.2 touches nothing on BM
   except making the "0 blocked siblings" fact explicit rather than implicit.

No frozen Phase-2.1 policy was silently modified. The common evaluation cohort, source-validation
cohort, AUROC/AUPRC primaries, pAUROC@FPR≤0.10, the 90th-percentile source-validation threshold
policy, Protocol-A aggregation, Protocol-B median segment aggregation, the cluster-bootstrap design,
the no-test-tuning rule and the no-anomaly-training-data rule are carried into the v3 evaluation
policy byte-identically, and that identity is asserted in both the validation script and a test.

## OPEN QUESTIONS FOR PHASE 3

1. Freeze each paired model/feature configuration before its first test evaluation, and record the
   adaptation budget (`27` / `18`) in the experiment config so a result can never be reported
   without its budget.
2. Decide how a 27-segment adaptation set is actually consumed — fine-tuning, prototype/centroid
   estimation, calibration-only — and freeze that choice before results; it matters more at this
   budget than it would at 123.
3. Keep window, feature, model, aggregation and metric implementations identical across S/T.
4. Decide the validation-only window/normalisation comparison without touching common test.
5. Implement the frozen parent-cluster bootstrap before reporting any model comparison.
6. Decide whether the *T-parent-complete* budget arm is worth running at all, and if so, obtain
   explicit authorisation and report it as a separate budget condition.
