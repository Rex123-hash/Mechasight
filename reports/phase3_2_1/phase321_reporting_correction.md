# Phase 3.2.1 — Versioned Reporting Correction

**Reporting correction id:** `p321-correction-75e1479967f120aff5047360`
**Policy:** `configs/phase3/phase321_reporting_correction_policy_v1.json`
(SHA-256 `aef18074a0555b219c2907cd46182f27809c39b048997f045f691a22a8414ba6`)
**Parent policy (unmodified):** `p311-reporting-5e76ab7dabd5689c7b0ced68`
**Scientific results HEAD:** `a576c9df8000dd1c0a5ad6fd73845f97facf98b2`
**Test batch:** `p32-test-batch-8f7d48324fd10df97ae29100`

> **This is not an experiment.** No model was trained, re-fit, re-scored, re-tuned or
> re-thresholded. No split, adaptation membership, feature, window, seed or hyperparameter
> changed. Every corrected value is arithmetic over confusion counts and cohort metadata that
> were already persisted and hashed at Phase 3.2. **The test-look ledger is unchanged at 28
> entries and no new test look occurred.**

The original Phase-3.2 record in `reports/phase3_2/` is untouched and remains the citable
evidence. This layer supersedes its **presentation**, not its measurements.

---

## 1. Why this correction exists

The independent audit (`docs/phase32_independent_audit.md`) returned **YELLOW**. Every score,
threshold, confusion count, ranking metric, bootstrap interval and paired contrast reproduced
exactly, and no scientific defect was found. The verdict was driven entirely by reporting
conventions and stale prose.

The decisive observation: **both reporting defects pushed in the optimistic direction.** A null
was published where a zero belonged, and an Average Precision looked strong only because its slice
was 85 % anomalous. Correcting them makes the headline result *worse*, not better — which is
precisely why the correction was necessary.

---

## 2. F1 convention

**Corrected rule**

```
F1 = 2*TP / (2*TP + FP + FN)      when 2*TP + FP + FN > 0
     undefined                    otherwise
```

This is identical to the harmonic mean of precision and recall wherever that form is itself
defined — verified across the 18 Phase-3.2 rows where both apply, to a maximum absolute difference
of **1.1 × 10⁻¹⁶**. It is a strict continuous extension, not a different metric.

The Phase-3.2 implementation returned `null` whenever `TP = 0`, through two distinct routes:

| Route | Condition | Rows |
|---|---|---:|
| A | `TP = 0`, `FP = 0` → precision is a genuine `0/0`, so the harmonic form short-circuits | 9 |
| B | `TP = 0`, `FP > 0` → precision `= 0.0` is defined, recall `= 0.0`, so `precision + recall == 0` | 1 |

**All 10 affected rows are RoboticArm Isolation Forest — all five seeds in both regimes.** In every
one the denominator is `2·0 + FP + 232 ≥ 232 > 0`, so the corrected F1 is **exactly 0.000**.

| experiment_id | regime | seed | TN | FP | FN | TP | original F1 | corrected F1 |
|---|---|---:|---:|---:|---:|---:|---|---:|
| `p3-b7df07b4435147ce84abb2fb` | S | 0 | 234 | 0 | 232 | 0 | null | **0.000** |
| `p3-d523942f6e3b110247ccd97a` | S | 1 | 234 | 0 | 232 | 0 | null | **0.000** |
| `p3-d2e84d3ccdbb25446ed3b5c2` | S | 2 | 234 | 0 | 232 | 0 | null | **0.000** |
| `p3-61da2e22429837caf51172a3` | S | 3 | 234 | 0 | 232 | 0 | null | **0.000** |
| `p3-4bad66197b1b5b39bc0e1ce8` | S | 4 | 234 | 0 | 232 | 0 | null | **0.000** |
| `p3-f2e82658337725d643461f5a` | T | 0 | 232 | **2** | 232 | 0 | null | **0.000** |
| `p3-4df2732b72997882e300c7ee` | T | 1 | 234 | 0 | 232 | 0 | null | **0.000** |
| `p3-e0cb98b1d50e1b760ad00c33` | T | 2 | 234 | 0 | 232 | 0 | null | **0.000** |
| `p3-efee4a25a307b9556f1ef9e1` | T | 3 | 234 | 0 | 232 | 0 | null | **0.000** |
| `p3-b2c1eb73ec1f836d95401be0` | T | 4 | 234 | 0 | 232 | 0 | null | **0.000** |

The two RoboticArm Isolation-Forest **family** rows are corrected from null to **0.000** as well.
All 18 previously-defined F1 values are carried through unchanged (max difference < 1e-12).

**Why the count identity is preferred for reporting.** A blank reads as "not applicable" or
"missing". The true statement is: *this detector flagged none of the 232 anomalies present, at its
own frozen operating point.* That is a real, important and unfavourable result, and it must appear
as a number.

Both values are published side by side: `original_phase32_f1` and `phase321_count_identity_f1`.

---

## 3. Precision convention

Precision remains `TP / (TP + FP)`. When `TP + FP == 0` it is reported as
**`undefined - no positive predictions`** and is **never** replaced by `0`.

That distinction is scientifically load-bearing:

- **Precision** is genuinely `0/0` — no positive prediction was made, so no precision was
  measurable. Asserting `0.0` would claim a measurement that does not exist.
- **F1** is *not* undefined in the same rows, because anomalies exist in its denominator. It is
  legitimately `0.0`.

Nine individual rows carry undefined precision. The tenth (RoboticArm / T / seed 0) has `FP = 2`,
so its precision is a genuine, defined `0.0`.

**Family aggregation.** No imputation is invented. If any seed precision is undefined the family
precision is reported as undefined, together with the count of seeds for which it *was* defined:

| Machine | Regime | Family precision | Status |
|---|---|---|---|
| RoboticArm | S | undefined | `undefined - no positive predictions; precision defined for only 0 of 5 seeds` |
| RoboticArm | T | undefined | `undefined - no positive predictions; precision defined for only 1 of 5 seeds` |

Family **F1** remains the arithmetic mean of the five corrected seed F1 values. Scores are never
averaged; no seed is selected or dropped.

---

## 4. Average-Precision prevalence context

Every reported AP now carries the anomaly prevalence of the **identical** evaluated slice, plus a
descriptive `ap_minus_prevalence = AP − prevalence`.

> This is **context only**. It is deliberately *not* called a chance-adjusted Average Precision,
> and it is not a new primary metric. AUROC is retained alongside as prevalence-insensitive
> ranking context.

**Slice class composition**

| Machine | Slice | normal | anomaly | total | prevalence |
|---|---|---:|---:|---:|---:|
| RoboticArm | source | 214 | 116 | 330 | 0.3515 |
| RoboticArm | **target** | **20** | **116** | **136** | **0.8529** |
| BrushlessMotor | source | 78 | 78 | 156 | 0.5000 |
| BrushlessMotor | target | 78 | 78 | 156 | 0.5000 |

Because the leakage-safe protocol correctly withheld the adaptation parents, the RoboticArm target
slice retains only 20 healthy segments. A constant-score classifier scores **AP ≈ 0.853** there.

**Four family-level slice AP values are at or below their own prevalence:**

| Machine | Regime | Model | Slice | AP | prevalence | AP − prevalence |
|---|---|---|---|---:|---:|---:|
| RoboticArm | S | statistical_reference | target | 0.8213 | 0.8529 | **−0.0316** |
| RoboticArm | S | one_class_svm | target | 0.8256 | 0.8529 | **−0.0273** |
| RoboticArm | T | statistical_reference | target | 0.8247 | 0.8529 | **−0.0283** |
| BrushlessMotor | T | isolation_forest | target | 0.3973 | 0.5000 | **−0.1027** |

Worked example of the trap this closes: the statistical reference records source AP 0.6486 and
target AP 0.8213. Read naively, it improves under domain shift. With prevalence attached it is
**+0.297 above** its source floor and **−0.032 below** its target floor — the opposite conclusion.
An AP near 0.82 in the RoboticArm target slice must not be described as strong merely because 0.82
is numerically high.

---

## 5. RoboticArm Isolation-Forest threshold-transfer annotation

RoboticArm Isolation Forest has **strong ranking performance** (AUROC 0.914 in S, 0.904 in T) and a
**frozen q90 operating point that does not transfer**.

Calibration versus test score scale (regime S, seed 0):

| Distribution | median | q90 | max |
|---|---:|---:|---:|
| source-validation normals (calibration) | 0.4692 | **0.5769 = threshold** | 0.5996 |
| common-test normals | 0.4290 | — | 0.5636 |
| common-test anomalies | 0.4967 | — | 0.5454 |

Within the test cohort anomalies rank clearly above normals, giving AUROC 0.912. But the entire
test distribution lies below the calibration q90, so nothing crosses the threshold. This holds for
9 of the 10 RoboticArm Isolation-Forest configurations; in the tenth, 2 segments cross.

The mechanism is a compressed Isolation-Forest score range combined with a q90 estimated from only
**6 physical parent recordings**.

> **Required reading of these rows.** No positive alarms were generated at the frozen operating
> point; therefore zero false positives coexist with zero fault detections.

**FPR 0.0000 here must not be described as excellent specificity.** It is the same absence of
alarms that produces recall 0.000. Affected rows carry
`produced_no_positive_predictions = true` and the annotation text in every corrected artifact.

The threshold is **unchanged**, and no replacement threshold is proposed in any primary table. Any
threshold redesign would now be post-test exploratory work and is out of scope.

---

## 6. Cluster and sample-size disclosure

Threshold-dependent rates and narrow slices now carry their independent-observation counts.

| Cohort | segments | independent physical recordings |
|---|---:|---:|
| RoboticArm common evaluation | 466 | **33** |
| RoboticArm source validation | 186 | **6** |
| RoboticArm healthy target | 20 | **6** |
| RoboticArm target adaptation | 27 consumed (96 siblings blocked) | 18 |
| BrushlessMotor common evaluation | 312 | 312 (one segment per recording) |
| BrushlessMotor healthy target | 78 | 78 |

> **The 466 RoboticArm common-test segments are not 466 independent physical observations.** They
> are consecutive excerpts of 33 continuous recordings, with a median of 8 and a maximum of 48
> segments per recording. This is why the RoboticArm confidence intervals are wide; that width is
> honest and must not be narrowed by re-describing the unit of analysis.

---

## 7. Healthy-target false positives

Values are **unchanged** from Phase 3.2. No new confidence-interval methodology is introduced — an
interval for this rate was not part of the frozen Phase-3.1.1 policy, and adding one now would be
new methodology rather than a correction. What is added is evidence size and an explicit warning.

| Machine | Regime | Model | healthy-target N | parents | FP | FPR |
|---|---|---|---:|---:|---:|---:|
| RoboticArm | S | statistical_reference | 20 | 6 | 1 | 0.050 |
| RoboticArm | S | isolation_forest | 20 | 6 | 0.0 | 0.000 † |
| RoboticArm | S | one_class_svm | 20 | 6 | 12 | **0.600** |
| RoboticArm | T | statistical_reference | 20 | 6 | 1 | 0.050 |
| RoboticArm | T | isolation_forest | 20 | 6 | 0.4 | 0.020 † |
| RoboticArm | T | one_class_svm | 20 | 6 | 6 | **0.300** |
| BrushlessMotor | S | statistical_reference | 78 | 78 | 63 | **0.808** |
| BrushlessMotor | S | isolation_forest | 78 | 78 | 76.6 | **0.982** |
| BrushlessMotor | S | one_class_svm | 78 | 78 | 78 | **1.000** |
| BrushlessMotor | T | statistical_reference | 78 | 78 | 62 | **0.795** |
| BrushlessMotor | T | isolation_forest | 78 | 78 | 70.8 | **0.908** |
| BrushlessMotor | T | one_class_svm | 78 | 78 | 70 | **0.897** |

† Threshold-transfer artefact, not specificity — see §5.

**RoboticArm One-Class SVM, observed direction:** S flagged **12 of 20** target-normal segments;
T flagged **6 of 20**. Those 20 segments come from only **six physical parent recordings**, and no
interval is reported for this rate. Treat the direction as informative and **the magnitude as
imprecise**.

**BrushlessMotor is the decisive unfavourable result.** At the frozen operating point every model
flags 79–100 % of *healthy* target-domain recordings; One-Class SVM in Regime S flags all 78. A
respectable ranking metric and unacceptable alarm behaviour coexist without contradiction, because
AUROC never evaluates the threshold.

---

## 8. BrushlessMotor confound

Every BrushlessMotor row in this layer carries
`brushless_motor_acquisition_time_class_confound_v1`.

All released normal recordings precede all released anomalous recordings with no overlap, so fault
state and acquisition session/time are perfectly confounded and **not independently identifiable**.

**Permitted:** "benchmark anomaly-discrimination performance on the released BrushlessMotor
cohort."

**Not permitted:** "pure mechanical-fault detection performance."

Specifically, neither the source-slice AUROC of **0.977** (S, One-Class SVM) nor the complete-cohort
AUROC of **0.8648** (T, One-Class SVM) may be presented as evidence of physical fault sensitivity. A
model keyed to slow session drift or any monotone time-correlated nuisance would score identically.

---

## 9. Representation wording

The actual Phase-3 classical representation is:

> raw sensor streams → 100 ms windows → 98 engineered features per window → **median of each
> feature** across the segment → one 98-dimensional segment vector → train-only feature z-score →
> classical model → one segment anomaly score.

It is **not** *window anomaly score → median score* for the Phase-3.1 / Phase-3.2 classical models.

`configs/evaluation/protocol_b_few_shot_adaptation_v3.json` carries a generic
`"median of window anomaly scores"` aggregation field. That field **predates** the Phase-3.1
representation freeze and is **deliberately left unchanged** so historical artifacts remain exactly
reproducible. For the Phase-3 classical baseline path it is superseded by
`configs/phase3/preprocessing_primary_v1.json` and by the `feature_representation` field recorded
in each of the 28 experiment configs. Historical configs are not edited for cosmetic currency.

---

## 10. Corrected artifacts

| File | Contents |
|---|---|
| `primary_baselines_corrected.csv` | the 12 family results with corrected F1/precision, prevalence context, cluster counts and threshold annotations |
| `threshold_metrics_corrected_individual.csv` | all 28 configurations, corrected conventions, original values retained |
| `isolation_forest_family_threshold_metrics_corrected.csv` | per-seed and family corrected F1/precision with the defined-seed count |
| `domain_slice_metrics_with_prevalence.csv` | all 80 slice rows (family + individual) with class counts, prevalence and parent counts |
| `healthy_target_false_positives_context.csv` | unchanged rates plus segment/parent counts and the uncertainty warning |
| `correction_manifest.json` | correction identity, policy hash, SHA-256 of all 16 source Phase-3.2 files and the frozen split manifest, and of every corrected file |
| `supervisor_document_refresh_checklist.md` | items for a separate supervisor-document refresh phase |
| `phase321_reporting_correction.json` | machine-readable summary of this correction |

Every corrected row carries `experiment_id`, `source_phase32_artifact`, `test_cohort_hash`,
`threshold`, `threshold_source_experiment_id` and `reporting_correction_id`. **No new experiment id
was minted**, because no model experiment occurred.

---

## 11. Integrity verification

| Check | Result |
|---|---|
| Phase-3.2 artifact manifest, all 155 hashes | **unchanged, 0 mismatched** |
| Test-look ledger entries | **28 → 28** |
| Test-look ledger SHA-256 | `db4bde80…0b0db0`, byte-identical |
| Frozen parent reporting policy SHA-256 | `a1f695d3…6b7da2c`, unchanged |
| Frozen v3 split manifest SHA-256 | `68e5406d…f0d8409e`, unchanged |
| 28 Phase-3.1 experiment configurations | unchanged |
| New model result bundles | **none** |
| New test look | **none** |
| Model scored during this phase | **none** |
| Full test suite | **291 passed** |

The correction code imports no estimator, scorer or pipeline module.
`assert_no_model_dependency()` is called before any artifact is written and fails if
`src.models.classical`, `src.experiment.phase31`, `src.experiment.phase32`, `sklearn.ensemble` or
`sklearn.svm` is present in `sys.modules` — a structural guarantee, exercised by a test rather than
merely asserted in prose.

---

## 12. What this correction does **not** do

- It does not change any anomaly score, fitted model, threshold, split membership, adaptation
  membership, feature, window, seed or hyperparameter.
- It does not re-rank the model families or revisit any comparison.
- It does not introduce a new uncertainty methodology.
- It does not propose a replacement operating point.
- It does not overwrite, edit or delete anything in `reports/phase3_2/`.
- It does not touch the test-look ledger.

The scientific state of the project is exactly what it was at
`a576c9df8000dd1c0a5ad6fd73845f97facf98b2`.
