# Phase 3.2 — Independent Scientific Audit and Post-Test Reporting-Correction Gate

**Audit id:** `p32-independent-audit-v1` · **Date:** 8 September 2026
**Role:** independent adversarial audit. The objective was to *falsify* the Phase-3.2 conclusions,
not to defend them.
**Audited HEAD:** `a576c9df8000dd1c0a5ad6fd73845f97facf98b2` (branch `master`)
**Evaluator commit:** `99f6055156c314bbf26a847713d26792979fd050`
**Test batch:** `p32-test-batch-8f7d48324fd10df97ae29100`

**No model was trained, re-fit, re-scored, tuned or re-thresholded during this audit.** Metrics
were recomputed from the already-persisted score bundles only. The test-look ledger was read and
hash-checked; it was not written.

---

## 1. Verdict

> ### YELLOW
>
> The Phase-3.2 numbers are **sound and may be retained as the primary baseline evidence**. Every
> score, threshold, confusion count, ranking metric, bootstrap interval and paired contrast
> reproduces exactly from persisted evidence, and every cohort and leakage invariant holds.
>
> **No scientific defect was found.** YELLOW is assigned solely because two reporting conventions
> and a set of stale project documents materially misrepresent the results to a reader. Both are
> correctable from existing stored quantities without rescoring anything.

The single most important sentence in this audit: **the pipeline is correct, and the headline
result is worse than the tables currently make it look.** Both reporting defects push in the
optimistic direction — a null where a zero belongs, and an Average Precision that looks high only
because the slice is 85 % anomalous.

---

## 2. Pre-audit repository check

| Check | Expected | Observed | |
|---|---|---|---|
| Branch | `master` | `master` | PASS |
| HEAD | `a576c9d…` | `a576c9df8000dd1c0a5ad6fd73845f97facf98b2` | PASS |
| Evaluator commit | `99f6055…` | `99f6055156c314bbf26a847713d26792979fd050` | PASS |
| Test batch | `p32-test-batch-8f7d4832…` | single batch, all 28 entries | PASS |
| Ledger entries | 28 | 28 | PASS |
| Tracked worktree | clean | clean (`git diff HEAD` empty) | PASS |
| Commits after results | none | none | PASS |
| Test suite | passing | **264 passed**, 0 failed, 0 skipped | PASS |
| Phase-3.2 artifact hashes | match manifest | **155 / 155 verified, 0 mismatched, 0 missing** | PASS |
| Recovery documents | present | `docs/PROJECT_STATE_RECOVERY.md`, `reports/recovery/…json` | PASS |

Untracked at audit start: `docs/PROJECT_STATE_RECOVERY.md`, `reports/recovery/`,
`reports/supervisor/`, `reports/synopsis/`, `tmp/`. None is a scientific artifact.

The repository does **not** materially differ from the recovered checkpoint.

---

## 3. Data-flow audit

Traced end to end for **RoboticArm S Isolation Forest**, **RoboticArm T One-Class SVM** and
**BrushlessMotor T One-Class SVM**, then re-verified programmatically for all 28.

```
frozen v3 split manifest (blob f3eb4f33…, byte-identical across all 7 commits since Phase 2.2)
  -> authorized training rows only  (split == "train"; blocked siblings rejected by name;
                                     normal-only asserted; complete frozen cohort required)
  -> 98-feature segment artifact    (row counts equal the frozen role counts exactly)
  -> training-only population z-score (FittedChannelStandardizer.fit raises unless split=="train")
  -> model fit                      (labels structurally excluded, see below)
  -> source-validation scores       (reproduced exactly: validation_exact_match == True for all 28)
  -> frozen q90 threshold           (reproduces to 2.2e-16)
  -> common-test feature artifact   (cohort hash recomputed == stored for all 28)
  -> anomaly score                  (finite; >= 283 distinct values per cohort)
  -> metrics                        (reproduce to 0.0e+00)
  -> class-stratified cluster bootstrap (reproduces to 1.1e-16)
  -> report row
```

### Leakage paths — all closed

| Path that must not exist | Evidence | Result |
|---|---|---|
| Anomaly labels reach training | `FORBIDDEN_MODEL_FEATURES` blocks `anomaly_label`, `is_anomaly`, `label`, `target`, `y` plus every provenance/condition field; `_normal_mask(...).all()` asserted | closed |
| Validation/test statistics enter normalisation | scaler fitted only inside `fit_training_standardizer`, which validates the cohort first and raises for any `split != "train"` | closed |
| Common test alters hyperparameters | all 7 model configs frozen at `78b5363` and byte-stable through HEAD; results commit `a576c9d` touched **no** file under `src/`, `configs/`, `scripts/` or `tests/` | closed |
| Blocked siblings enter a consumed cohort | 96 blocked ids ∩ (train ∪ validation ∪ test) = **0** in all four machine×regime cells | closed |
| Target adaptation enters Regime S | RA/S and BM/S `target_adaptation` rows carry split `reserved_adaptation`; S train roles = `{source_training}` only | closed |
| S/T evaluation cohorts differ | symmetric difference **0** for both machines; scored test == frozen `common_evaluation` role exactly | closed |
| Test parents overlap fitted parents | test ∩ train = 0, test ∩ validation = 0, train ∩ validation = 0 (all four cells) | closed |
| Scaler reuse across S/T or machines | 4 distinct `normalizer_identity` values across the 4 cells | closed |

Training-set sizes are exactly as frozen: RA S 1528 / RA T 1555 (= 1528 + 27), BM S 1137 / BM T
1155 (= 1137 + 18). Source validation is identical across S and T within each machine
(RA 186 segments / 6 parents; BM 126) and is all-source, all-normal.

---

## 4. Representation audit

**Confirmed** the primary Phase-3.1/3.2 classical representation is:

> raw sensor data → 100 ms non-overlapping windows → 98 engineered features per window →
> **median of each feature** over the segment's windows → one 98-dimensional segment vector →
> feature-wise population z-score fitted on authorised training data only → classical model →
> **one** segment anomaly score.

It is **not** *window model scores → median score*. Evidence: `aggregate_windows_by_segment` in
`src/experiment/phase31.py` computes `np.median(matrix, axis=0)` over a
(windows × features) matrix; `src/experiment/aggregation.py` (the median-window-*score*
implementation) is imported only by `tests/test_phase30_evaluation.py` and by no Phase-3.1/3.2
code path.

98 features = 7 numeric channels (1 mic + 3 accelerometer axes + 3 gyroscope axes) × 14 statistics.
The configurable band-energy helper is **not** in the primary representation, as documented.
Artifact row counts match the frozen roles exactly.

### Stale wording found (not edited)

| Location | Stale text | Reality |
|---|---|---|
| `configs/evaluation/protocol_b_few_shot_adaptation_v3.json` → `score_aggregation.protocol_b_primary` | `"median of window anomaly scores"` | Phase-3.1 primary is median of **features** |
| `configs/evaluation/protocol_b_common_test_v2.json` (same key) | same | superseded |
| `docs/evaluation_protocol.md:340` | "frozen to the **median window anomaly score**" | superseded by the Phase-3.1 freeze |
| `docs/leakage_risks.md:176` | "median of window scores" | superseded |
| experiment `config.json` → `segment_aggregation_rule: "median"` | ambiguous — reads as median of scores | the adjacent `feature_representation` field is correct and unambiguous |

This is a naming/annotation inconsistency, not a computational one: the 28 executed configs record
the correct `feature_representation` string, and the code does what Phase 3.1 froze.

---

## 5. Model-implementation audit

| Family | Verified implementation | Score transform |
|---|---|---|
| Statistical reference | `sqrt(mean(z**2))` — RMS distance from the training mean in standardised feature space; no covariance is fitted (it is not Mahalanobis); deterministic | already higher-is-more-anomalous |
| Isolation Forest | `sklearn.ensemble.IsolationForest`, `n_estimators=200`, `max_samples="auto"`, `contamination="auto"`, `max_features=1.0`, `bootstrap=False`, `n_jobs=1`, `random_state ∈ {0,1,2,3,4}` | `-score_samples` |
| One-Class SVM | `sklearn.svm.OneClassSVM`, `kernel="rbf"`, `gamma="scale"`, `nu=0.05`, `tol=0.001`, `shrinking=True`, `cache_size=512`, `max_iter=-1` | `-decision_function` |

Because the z-score uses the training mean, the training mean maps to the origin, so
`sqrt(mean(z²))` is exactly the stated RMS distance from the training mean.

- Score direction consistent (`higher_is_more_anomalous`) in every config and bundle.
- All scores finite; ≥ 283 distinct values per cohort (no degenerate constant-score model).
- One-Class SVM is correctly given **no** `random_state` (sklearn's OCSVM has none); its
  determinism comes from fixed input, and each seed carries its own frozen threshold.
- Each of the five Isolation-Forest seeds has its own separately frozen q90 threshold — verified
  numerically distinct (RA/S: 0.5769, 0.5819, 0.5913, 0.5823, 0.5775).
- No cross-machine or cross-regime model or scaler reuse.

---

## 6. Threshold audit

For all 28 configurations:

```
threshold = numpy.quantile(source_validation_normal_segment_scores, 0.90, method="higher")
decision  = anomaly iff score > threshold
```

reproduces to a maximum absolute difference of **2.2 × 10⁻¹⁶**. Verified additionally:

- the calibration rows are **source domain only** and **normal only** for all 28;
- the calibration segment ids equal the ids recorded in `threshold_provenance.source_segment_ids`;
- calibration sizes are 186 (RA) and 126 (BM), i.e. the frozen `source_validation` role;
- no test score participates;
- S and T use the identical validation *identities* within a machine, while their thresholds
  legitimately differ because the fitted model and scaler differ.

### Why AUROC 0.91 can coexist with recall 0.000 — RoboticArm Isolation Forest

AUROC is invariant to any monotone transform of the scores; the threshold is an **absolute cut**.
The two can decouple completely, and here they do.

RoboticArm / Regime S / seed 0:

| Distribution | median | q90 | max |
|---|---:|---:|---:|
| source-validation normals (calibration) | 0.4692 | **0.5769 = threshold** | 0.5996 |
| common-test normals | 0.4290 | — | 0.5636 |
| common-test anomalies | 0.4967 | — | 0.5454 |

Within the test cohort the anomalies rank clearly above the normals — hence AUROC 0.912. But the
**entire** test distribution (max 0.5636) lies below the calibration q90 (0.5769). Nothing is
flagged: TP = 0, FP = 0, recall = 0.000, FPR = 0.000. This holds for **9 of the 10** RA Isolation-
Forest configurations; in the tenth (T, seed 0) the threshold sits fractionally inside the test
range and 2 segments cross it.

Mechanism: Isolation-Forest `score_samples` has a compressed dynamic range (here ≈ 0.40–0.60), and
the calibration set is only 186 segments from **6** physical parent recordings, so its upper decile
is set by whichever of those six recordings the forest finds most isolated. A q90 estimated from
6 clustered recordings is not a transferable absolute operating point for a 33-parent test cohort.

**Two consequences that must be stated wherever these rows appear:**

1. FPR = 0.0000 for RA Isolation Forest is **not specificity skill.** It is the same artefact as
   recall = 0.000: nothing at all was flagged. Reporting it next to a genuine FPR invites the
   reader to credit the model with excellent false-alarm behaviour that it did not demonstrate.
2. The finding is real and worth reporting: **Isolation Forest ranks RoboticArm anomalies well but
   its q90 operating point does not transfer from the validation cohort to the test cohort.**

The threshold was not, and must not be, changed.

---

## 7. Metric-implementation audit

Definitions inspected in `src/experiment/metrics.py` and independently re-derived without importing
that module (so a fault inside it could not hide itself).

| Metric | Implementation | Independent verdict |
|---|---|---|
| AUROC | `sklearn.roc_auc_score` | correct |
| Average Precision | `sklearn.average_precision_score` | correct; it is the step-wise AP, not trapezoidal AUC-PR |
| pAUROC @ maxFPR 0.10 | `sklearn.roc_auc_score(..., max_fpr=0.10)` | correct, and it **is McClish-standardised**, `0.5·(1 + (A − A_min)/(A_max − A_min))`, not raw partial area. The code comment says so, the reporting freeze says so, and the bootstrap re-implementation reproduces sklearn to 1.1e-16. A standardised pAUROC of 0.50 means chance, not zero area. |
| TN / FP / FN / TP | counts under `score > threshold` | correct for all 28 |
| FPR | `FP/(FP+TN)`, `None` if no normals | correct |
| Recall | `TP/(TP+FN)`, `None` if no positives | correct |
| Precision | `TP/(TP+FP)`, `None` if no positive predictions | defensible (genuine 0/0) — see §8 |
| F1 | harmonic mean; `None` if precision undefined **or** precision+recall == 0 | **materially misleading — see §8** |

Recomputation over all 28 configurations, from the stored scores and frozen thresholds:

| Quantity | max abs difference |
|---|---|
| AUROC | **0.0e+00** |
| Average Precision | **0.0e+00** |
| standardized pAUROC@0.10 | **0.0e+00** |
| TN / FP / FN / TP | **exact match, all 28** |
| threshold | 2.2e-16 |
| test cohort hash | identical, all 28 |
| domain-slice metrics | 0.0e+00 |
| healthy-target counts | exact match |

**PASS — no mismatch at full precision.**

---

## 8. The F1 / undefined-metric finding

### What happens

`protocol_b_threshold_metrics` returns `f1 = None` in two distinct situations, both of which occur
in the real results:

| Mechanism | Condition | Rows affected |
|---|---|---|
| A | `TP = 0` and `FP = 0` → precision is genuinely `0/0` → F1 short-circuits to `None` | **9** |
| B | `TP = 0` and `FP > 0` → precision `= 0.0` is defined, recall `= 0.0`, so `precision + recall == 0` → F1 `None` | **1** (RA / T / seed 0) |

All 10 are RoboticArm Isolation Forest — all five seeds in both regimes.

### Why it is misleading

Under the standard count identity

$$F_1 = \frac{2TP}{2TP + FP + FN}$$

every affected row has denominator `2·0 + FP + 232 ≥ 232 > 0`, so **F1 is mathematically
well-defined and equals exactly 0.0 in all ten cases.** The identity is not a different metric: over
the 18 rows where both definitions apply they agree to **1.1 × 10⁻¹⁶**. The count form is a strict
continuous extension of the harmonic form.

Returning `None` is internally consistent with the harmonic-mean definition, so the stored value is
**not technically wrong** — it is convention-dependent. But the consequence is that the two
RoboticArm Isolation-Forest rows of the primary table publish a blank (CSV `NaN`, report
`UNDEFINED`) where the correct and highly informative value is **0.000: this detector caught none
of the 232 anomalies at its frozen operating point.** A reader skimming Table A sees a missing
value, not a failure. That is an optimistic distortion of the single most important negative result
in Phase 3.2.

Precision is different: for the 9 rows with `TP = FP = 0` it is a genuine `0/0` and reporting it as
undefined is defensible and arguably more honest than scikit-learn's `0.0`-with-warning convention.
It should be *labelled* ("undefined — no positive predictions"), not silently blanked.

### Family-level aggregation

The frozen policy says the Isolation-Forest family metric is the arithmetic mean of the five seed
metrics. For RA / T precision the seeds are `[0.0, None, None, None, None]` and the mean propagates
to `None`. So "mean of five seed metrics" is silently unreachable whenever any seed metric is
undefined. This is conservative and defensible, but undocumented. Under the count identity the
family F1 is unambiguous: mean of five zeros = **0.0**.

### Decision

> **C — VERSIONED REPORTING CORRECTION NEEDED.**

Two facts make this safe and clean:

1. The frozen Phase-3.1.1 reporting policy **does not specify any undefined-metric convention** —
   it merely lists `precision` and `f1` as threshold-dependent metrics. A correction therefore does
   not amend a frozen decision; it fills a gap the freeze left open.
2. **No test pins the current null behaviour**, so the change cannot silently invalidate a
   pre-registered expectation.

`reports/phase3_2_audit/undefined_metric_correction_preview.csv` and
`family_undefined_metric_preview.csv` already contain the corrected values, derived **purely from
the stored confusion counts**. Generating the correction requires no model, no scoring, no
threshold and no cohort access — only arithmetic on integers already published.

---

## 9. Isolation-Forest five-seed audit

| Frozen requirement | Implementation | Verdict |
|---|---|---|
| Metrics computed per seed 0–4 | `weighted_bootstrap_metric` applied per seed | PASS |
| Family metric = arithmetic mean of the 5 seed metrics | `np.mean(seed_values, axis=0)`; family point = `np.mean([...])` | PASS (max diff 1.1e-16) |
| Report SD, min, max and all five seed values | present in `isolation_forest_seed_summaries.csv` and the report appendix | PASS |
| No best-seed selection | none in code | PASS |
| No averaged-score ensemble | scores are never averaged before metric computation | PASS |
| Within a replicate: metric per seed, **then** mean, **then** contrast | `values = mean(seed_values, axis=0)` precedes `paired_distribution_difference` | PASS |
| Same resampled units for every seed/model/regime | one weight matrix per machine, reused everywhere | PASS |
| No unfavourable seeds dropped | all five present in every summary | PASS |

Fractional confusion counts in family rows (e.g. FP = 0.4, TN = 233.6) are correctly annotated
"mean across seeds; fractional values are not a literal confusion matrix".

---

## 10. Bootstrap audit

Phase 3.2 does **not** use the row-duplication `src/experiment/bootstrap.py`; it uses a vectorised
weighted bootstrap in `phase32.py`. That substitution is the highest-risk hidden-bug surface in the
whole pipeline, so it was tested adversarially rather than read.

**Equivalence test — 1050 randomised cases** covering heavy ties, extreme class imbalance
(20 vs 116, mirroring the RA target slice), near-separable scores, zero-weight rows and the fully
degenerate all-scores-equal case:

| Metric | max abs diff vs **literal row duplication** | vs sklearn `sample_weight` |
|---|---|---|
| AUROC | 2.2e-16 | 2.2e-16 |
| Average Precision | 8.9e-16 | 8.9e-16 |
| standardized pAUROC@0.10 | 1.1e-16 | 1.1e-16 |

**PASS.** The weighted implementation is the duplicate-row bootstrap to floating-point precision.

Design conformance:

- 10 000 replicates, seed 1995, 95 % percentile interval, class-stratified — confirmed.
- **One** weight matrix per machine, generated from the Regime-S statistical-reference frame and
  reused across every regime, family, seed and metric. `paired_distribution_difference` refuses to
  combine distributions with different `draw_identity`, so unpaired draws cannot leak in.
- **RoboticArm:** cluster = physical parent recording. Verified that every sibling segment of a
  drawn parent receives the *same* multiplicity in *every* replicate, and that the number of drawn
  parents per class is constant per replicate. No child-level independent resampling occurs.
- **BrushlessMotor:** the code *asserts* one segment per recording before proceeding; verified
  312 test segments = 312 recordings. The cluster bootstrap therefore legitimately degenerates to a
  recording-level bootstrap. This handles clustering — it does **not** address the acquisition-time
  confound, which is a bias, not a variance component.
- 0 invalid replicates anywhere.

**All 36 CIs, 18 S-vs-T deltas and 36 model-family deltas reproduce to ≤ 1.1 × 10⁻¹⁶.**

### What the intervals do and do not cover

They are **conditional on the observed evaluation cohort and the fixed fitted models and seeds**.
They cover sampling variability of the evaluation clusters. They do **not** cover:

- model-training variability (models were fit once; the fit is not bootstrapped);
- seed uncertainty as a population (the five seeds are averaged, not resampled);
- transfer to a new machine, installation or site;
- the BrushlessMotor acquisition-time confound, which no resampling scheme can remove.

---

## 11. Primary result reproduction — the 12 family results

All twelve verified; ranking metrics and confusion counts reproduce exactly, CIs to 1e-16.

**RoboticArm** (test: 234 normal / 232 anomaly; 33 parent recordings)

| Regime | Model | AUROC [95 % CI] | AP [95 % CI] | pAUROC [95 % CI] | FPR | Recall | Precision | F1 (as published → corrected) |
|---|---|---|---|---|---:|---:|---:|---|
| S | statistical | 0.8522 [0.652, 0.953] | 0.8091 [0.617, 0.942] | 0.6224 [0.493, 0.847] | 0.0043 | 0.0431 | 0.9091 | 0.0823 |
| S | IF (mean-5) | 0.9137 [0.774, 0.976] | 0.8654 [0.718, 0.968] | 0.7126 [0.511, 0.905] | 0.0000 | 0.0000 | undefined | **null → 0.000** |
| S | OCSVM | 0.8985 [0.783, 0.959] | 0.8614 [0.723, 0.955] | 0.7019 [0.495, 0.880] | 0.0556 | 0.5560 | 0.9085 | 0.6898 |
| T | statistical | 0.8457 [0.642, 0.951] | 0.8038 [0.607, 0.942] | 0.6185 [0.493, 0.844] | 0.0043 | 0.0431 | 0.9091 | 0.0823 |
| T | IF (mean-5) | 0.9038 [0.756, 0.971] | 0.8528 [0.699, 0.963] | 0.7001 [0.500, 0.894] | 0.0017 | 0.0000 | undefined | **null → 0.000** |
| T | OCSVM | 0.8932 [0.788, 0.953] | 0.8933 [0.767, 0.957] | 0.7383 [0.588, 0.866] | 0.0256 | 0.4440 | 0.9450 | 0.6041 |

**BrushlessMotor** (test: 156 normal / 156 anomaly; 312 recordings) — all values inherit the
acquisition-time/class confound.

| Regime | Model | AUROC [95 % CI] | AP [95 % CI] | pAUROC [95 % CI] | FPR | Recall | Precision | F1 |
|---|---|---|---|---|---:|---:|---:|---:|
| S | statistical | 0.5988 [0.534, 0.661] | 0.5843 [0.530, 0.647] | 0.5085 [0.491, 0.546] | 0.4423 | 0.6474 | 0.5941 | 0.6196 |
| S | IF (mean-5) | 0.6549 [0.592, 0.715] | 0.6340 [0.582, 0.692] | 0.5458 [0.522, 0.576] | 0.5256 | 0.8423 | 0.6157 | 0.7113 |
| S | OCSVM | 0.7054 [0.643, 0.764] | 0.6099 [0.555, 0.679] | 0.5010 [0.486, 0.536] | 0.5192 | 0.9551 | 0.6478 | 0.7720 |
| T | statistical | 0.5933 [0.528, 0.656] | 0.6120 [0.557, 0.670] | 0.5436 [0.517, 0.579] | 0.4359 | 0.5705 | 0.5669 | 0.5687 |
| T | IF (mean-5) | 0.5961 [0.527, 0.664] | 0.5089 [0.471, 0.564] | 0.4812 [0.476, 0.493] | 0.4885 | 0.7808 | 0.6151 | 0.6878 |
| T | OCSVM | 0.8648 [0.821, 0.905] | 0.8362 [0.784, 0.887] | 0.5969 [0.562, 0.694] | 0.4679 | 0.9487 | 0.6697 | 0.7851 |

No system is ranked on a single metric. RoboticArm intervals are wide (the statistical reference
spans 0.65–0.95) because the effective sample size is 33 clusters, not 466 segments — that width is
honest, not a defect.

---

## 12. Domain-slice audit

All slice metrics reproduce exactly. Class counts:

| Machine | Slice | normal | anomaly | AP chance baseline |
|---|---|---:|---:|---:|
| RoboticArm | source | 214 (8 parents) | 116 (13 parents) | 0.3515 |
| RoboticArm | target | **20 (6 parents)** | 116 (6 parents) | **0.8529** |
| BrushlessMotor | source | 78 | 78 | 0.5000 |
| BrushlessMotor | target | 78 | 78 | 0.5000 |

**Why AUROC degrades in the target domain.** Regime-S models never saw target-domain normals; a
healthy machine at load 25/30/35 is unfamiliar, so healthy target segments drift up the score
ranking and interleave with anomalies. RoboticArm source→target: statistical 0.811 → 0.568,
IF 0.908 → 0.675, OCSVM 0.878 → 0.579. This is the "DIFFERENT ≠ FAULTY" failure appearing directly
in the ranking metric. BrushlessMotor is worse: IF/T target AUROC is **0.331**, i.e. materially
*below* chance — in the target domain that model ranks healthy segments as more anomalous than
faulty ones.

### Average Precision cannot be compared across these slices — and currently is not caveated

The RoboticArm target slice is **85.3 % anomalous** (116 of 136), because the leakage-safe protocol
correctly withheld the adaptation parents, leaving only 20 healthy target segments. A
constant-score classifier scores AP = 0.853 there. Consequently:

**Four published slice AP values are *below* their own prevalence baseline:**

| Machine | Regime | Model | Slice | AP | baseline | AP − baseline |
|---|---|---|---|---:|---:|---:|
| RoboticArm | S | statistical | target | 0.8213 | 0.8529 | **−0.0316** |
| RoboticArm | S | OCSVM | target | 0.8256 | 0.8529 | **−0.0273** |
| RoboticArm | T | statistical | target | 0.8247 | 0.8529 | **−0.0283** |
| BrushlessMotor | T | IF | target | 0.3973 | 0.5000 | **−0.1027** |

A reader comparing the statistical reference's "source AP 0.649 vs target AP 0.821" would conclude
it performs *better* under domain shift. The truth is the reverse: +0.297 above chance in source,
−0.032 **below** chance in target. This is the most dangerous interpretive trap in the result set.
The published tables carry no prevalence baseline column.
`reports/phase3_2_audit/domain_slice_prevalence_context.csv` supplies one.

Slices are correctly treated as **descriptive**: they are not separately bootstrapped and no
confirmatory claim rests on them. That restraint should be preserved.

---

## 13. Healthy-target false-positive audit

This is the measurement the project exists to make: does the system alarm on a *healthy* machine
merely because its operating condition changed?

| Machine | Regime | Model | healthy-target N | FP | FPR |
|---|---|---|---:|---:|---:|
| RoboticArm | S | statistical | 20 | 1 | 0.050 |
| RoboticArm | S | IF (mean-5) | 20 | 0.0 | 0.000 † |
| RoboticArm | S | OCSVM | 20 | 12 | **0.600** |
| RoboticArm | T | statistical | 20 | 1 | 0.050 |
| RoboticArm | T | IF (mean-5) | 20 | 0.4 | 0.020 † |
| RoboticArm | T | OCSVM | 20 | 6 | **0.300** |
| BrushlessMotor | S | statistical | 78 | 63 | **0.808** |
| BrushlessMotor | S | IF (mean-5) | 78 | 76.6 | **0.982** |
| BrushlessMotor | S | OCSVM | 78 | 78 | **1.000** |
| BrushlessMotor | T | statistical | 78 | 62 | **0.795** |
| BrushlessMotor | T | IF (mean-5) | 78 | 70.8 | **0.908** |
| BrushlessMotor | T | OCSVM | 78 | 70 | **0.897** |

† These two zeros are the threshold-transfer artefact of §6, **not** specificity. Nothing was
flagged anywhere, so no healthy segment could be flagged either. They must never be presented as
the best false-alarm performance in the table.

**BrushlessMotor is the decisive negative result.** At the frozen operating point every model
flags 79–100 % of *healthy* target-domain recordings. One-Class SVM in Regime S flags **every
single one** (78/78). A plant would disable such a system on its first shift. This is
"DIFFERENT ≠ FAULTY" failing outright: BM One-Class SVM S has a respectable source-slice AUROC of
0.977, and is simultaneously useless at its own operating point in the target domain. High ranking
quality and unacceptable alarm behaviour coexist without contradiction, because AUROC never
evaluates the threshold.

**RoboticArm S→T (OCSVM) 60 % → 30 % must be quoted with care.** The 20 healthy-target segments
come from only **6 parent recordings**, unevenly distributed (1, 2, 2, 2, 6, 7). The improvement is
at least spread across 4 of those 6 parents rather than driven by one, which is reassuring, but
with ~6 independent units and **no confidence interval reported for healthy-target FPR**, the
magnitude is not resolvable. Report it as a direction, not a rate.

None of these threshold-dependent numbers is a ranking-performance statement.

---

## 14. S-vs-T adaptation audit

S and T are confirmed to use the **identical** evaluation cohort, so the deltas isolate training
information. Paired contrasts use the same draws (enforced by `draw_identity`).

| Machine | Model | ΔAUROC [95 % CI] | ΔAP [95 % CI] | ΔpAUROC [95 % CI] |
|---|---|---|---|---|
| RoboticArm | statistical | −0.0064 [−0.0143, 0.0004] | −0.0053 [−0.0170, 0.0031] | −0.0039 [−0.0154, 0.0112] |
| RoboticArm | IF | −0.0099 [−0.0226, **−0.0025**] | −0.0126 [−0.0258, **−0.0031**] | −0.0124 [−0.0253, 0.0004] |
| RoboticArm | OCSVM | −0.0053 [−0.0252, 0.0321] | +0.0319 [−0.0217, 0.0972] | +0.0364 [−0.0388, 0.1674] |
| BrushlessMotor | statistical | −0.0055 [−0.0151, 0.0042] | +0.0277 [**0.0081**, 0.0485] | +0.0351 [**0.0155**, 0.0540] |
| BrushlessMotor | IF | −0.0588 [−0.0813, **−0.0382**] | −0.1251 [−0.1534, **−0.0905**] | −0.0646 [−0.0911, **−0.0428**] |
| BrushlessMotor | OCSVM | **+0.1594** [0.1259, 0.1964] | **+0.2262** [0.1937, 0.2534] | +0.0959 [0.0704, 0.1626] |

Intervals excluding zero: RA Isolation Forest is **slightly worse** with adaptation (AUROC, AP);
BM Isolation Forest is **clearly worse**; BM One-Class SVM is **clearly better**; BM statistical is
better on AP/pAUROC only.

**The permitted interpretation is exactly:** *the effect of allowing the frozen released
target-normal adaptation allocation, under this benchmark, these preprocessing choices and these
model configurations.* It is **not** evidence about domain adaptation in general, and the sign is
not even consistent across model families on the same machine — which by itself forbids a general
claim.

Budget context that must accompany every delta:

- **RoboticArm:** only **27** adaptation segments from **18** parent recordings, with **96**
  same-parent siblings blocked from every use. 27 segments against 1528 source segments is ~1.7 %
  of the training set; a near-zero or slightly negative effect is unsurprising and is not evidence
  that adaptation is harmful in principle.
- **BrushlessMotor:** only **18** adaptation segments, and the machine carries the acquisition-time
  confound, so even the large positive OCSVM delta cannot be attributed cleanly to learning healthy
  target behaviour rather than to session structure.

---

## 15. Model-family comparison audit

All 36 paired contrasts reproduce exactly and use identical resampled units.

Point-estimate ranking and paired-CI evidence are kept apart:

- **RoboticArm S:** statistical − IF is −0.0616 [−0.1365, −0.0166] on AUROC — interval excludes
  zero, so IF outranks the statistical reference on AUROC. But statistical − OCSVM
  (−0.0463 [−0.1842, 0.0336]) and IF − OCSVM (0.0152 [−0.0583, 0.0726]) both include zero: the
  data do **not** separate OCSVM from either. And IF's ranking advantage coexists with zero recall.
- **BrushlessMotor T:** OCSVM beats both others decisively (statistical − OCSVM AUROC
  −0.2715 [−0.3175, −0.2280]; IF − OCSVM −0.2687 [−0.3277, −0.2119]).
- **BrushlessMotor S:** OCSVM > IF > statistical on AUROC with intervals excluding zero.

No superiority is inferred from non-overlapping marginal intervals anywhere in the audited outputs;
the report uses the paired contrasts, as frozen. That discipline is correct and should be kept.

---

## 16. BrushlessMotor confound — special section

**The constraint.** All 1437 normal BrushlessMotor recordings were acquired between 10:20:21 and
16:40:50; all 156 anomalous recordings between 16:48:07 and 18:22:40 on the same day, with a
7.3-minute gap and zero overlap. Fault status and acquisition session/time are therefore
**perfectly collinear and not independently identifiable.** No split, no resampling scheme and no
normal-only drift diagnostic can separate them, because there exists no anomalous recording from
the earlier window and no normal recording from the later one.

**What may be said**

- "benchmark anomaly-discrimination performance on the BrushlessMotor cohort";
- "separates the released anomalous recordings from the released normal recordings";
- all threshold and healthy-target numbers, with the confound attached.

**What may not be said**

- "detects mechanical faults in the brushless motor";
- "pure mechanical-fault detection performance";
- any attribution of the BM One-Class SVM S source-slice AUROC of 0.977 to fault physics. A model
  keyed to slow session drift, thermal state or any monotone time-correlated nuisance would score
  identically. **A high number here is not evidence of fault sensitivity.**
- any claim that the +0.159 AUROC S→T adaptation gain reflects learning healthy target behaviour.

**Audit of the existing Phase-3.2 wording:** the report already states the limitation explicitly
("these results are not pure isolated mechanical-fault evidence") and every BM row carries
`brushless_motor_acquisition_time_class_confound_v1`. **No overclaiming was found in the
Phase-3.2 report itself.** The risk is downstream: the confound must survive into slides, the
synopsis and any abstract, where a 0.977 or a 0.865 is exactly the kind of number that gets quoted
alone.

---

## 17. RoboticArm independence — special section

| Quantity | Value |
|---|---:|
| Physical parent recordings in the release | **101** |
| → source_training | 44 parents / 1528 segments |
| → source_validation | **6** parents / 186 segments |
| → target_adaptation | 18 parents / 27 segments |
| → adaptation_sibling_blocked | the **same 18** parents / 96 segments |
| → common_evaluation | **33** parents / 466 segments |

44 + 6 + 18 + 33 = 101: an exact partition, with the blocked siblings sharing the adaptation
parents rather than adding new ones.

**Many segments, few independent recordings.** The test cohort has 466 segments but only **33**
independent physical recordings — 14 normal and 19 anomalous. Segments per test parent: min 1,
median 8, max 48, SD 14.3. Treating 466 segments as independent would overstate precision by
roughly the average cluster size.

Cluster-bootstrap inference is therefore **appropriate and necessary**, and it is implemented
correctly (§10). The visible consequence is the wide RoboticArm intervals — the statistical
reference's AUROC CI spans 0.652–0.953. That width is the honest price of 33 clusters and should
not be narrowed by re-describing the unit of analysis.

Two consequences to carry forward:

1. The **source-validation cohort is 6 parents**. It must never be described as the 18 adaptation
   parents. Its smallness is the direct cause of the non-transferable Isolation-Forest threshold
   in §6.
2. The **healthy-target slice is 6 parents / 20 segments** — the thinnest evidence in the study,
   and unfortunately the most decision-relevant.

---

## 18. Test-look ledger and first-test integrity

| Check | Result |
|---|---|
| Entry count | **28** |
| Distinct `git_sha` | 1 — `99f6055…`, the evaluator commit |
| Distinct batch id | 1 — `p32-test-batch-8f7d48324fd10df97ae29100` |
| Distinct timestamps | 1 — all `2026-09-03T09:06:51.648233+00:00` (one atomic write) |
| Distinct cohort hashes | 2 — one per machine; RA `23c5b7e3…`, BM `e1b09c05…` |
| Cohort hashes recomputed from scored ids | identical for all 28 |
| Coverage | 7 configs × 2 regimes × 2 machines = 28 |
| Duplicate experiment ids | 0 |
| Result bundles without a ledger entry | **0** |
| Ledger entries without a result bundle | **0** |
| Ledger state | `ledger_start_count: 0` → `ledger_end_count: 28`, written before the first score |
| Ledger file history | created by the results commit only; 0 entries existed at `99f6055` |
| Post-test scientific modification | **none** — `a576c9d` touched no `src/`, `configs/`, `scripts/` or `tests/` file |
| Frozen config blob stability | all `configs/**` byte-identical from `78b5363` → HEAD |
| Split manifest blob stability | `f3eb4f33…` identical across all 7 commits since Phase 2.2 |
| Commits after the results commit | none |

Every configuration in the primary results is pre-registered; no post-test configuration appears.
The 28 entries are one first-test batch, not 28 rounds of adaptive selection.

**Ledger SHA-256 before and after this audit:**
`db4bde80099a870ae90686fa463400f02f93434fc11dfce6d8c20f2eac0b0db0` — **unchanged.**

---

## 19. Sensor terminology

The hardware-precise description of IMAD-DS:

| Level | Count | Detail |
|---|---:|---|
| Physical sensor devices | **2** | IMP23ABSU analogue MEMS microphone; ISM330DHCX 6-axis IMU |
| Sensing modalities | **3** | microphone/audio, accelerometer/vibration, gyroscope/angular motion |
| Numeric signal channels | **7** | 1 mic + 3 accelerometer axes + 3 gyroscope axes |

The accelerometer and gyroscope are **not separate devices** — both are functional blocks inside the
single ISM330DHCX package, which is why they share a time base and an internal-frequency
adjustment.

**Preferred wording where hardware precision matters:** *"three sensing modalities from two physical
sensor devices (seven numeric channels)"*, rather than "three physical sensors".

Ambiguous locations found (not edited): `README.md:64`; `docs/source_verification.md:47`;
`docs/assumptions_and_open_questions.md:143`. Occurrences at `docs/research_definition.md:77, 281,
901, 993` mostly describe the *authors'* baseline in the authors' own terms and are acceptable in
context if the two-device fact is stated once nearby.

---

## 20. Audio / dataset-representation

Verified directly against the extracted release:

| Claim | Evidence | Verdict |
|---|---|---|
| Microphone data are not distributed as MP3/MP4 | file-extension census of the extracted release: **11 688 `.parquet` + 12 `.csv`, and nothing else**; zero `.mp3/.mp4/.wav/.flac/.aac/.m4a/.ogg` | CONFIRMED |
| Signal streams are stored in Parquet | as above | CONFIRMED |
| Microphone is a timestamped numerical waveform channel | Arrow schema `Time: double`, `MIC [Waveform]: double`; 120 000 rows for a 7.499937 s RoboticArm segment = 16 kHz | CONFIRMED |
| ADC happened during original physical recording | acquisition on STEVAL-STWINBX1 before publication | CONFIRMED (from provenance, not re-measured) |
| MechaSight does not perform ADC | the pipeline's first input is already a numeric Parquet column | CONFIRMED |
| Reading Parquet is decoding, not ADC | deserialising stored float64 samples | CONFIRMED |

**Supported wording for supervisor/viva:**

> "The three levels are distinct. **Physical sensor acquisition** — transduction, conditioning and
> analogue-to-digital conversion — happened once, on the STMicroelectronics STEVAL-STWINBX1 node,
> when IMAD-DS was recorded. **Digital storage format** — the published release contains only
> Parquet columnar files and CSV metadata; there is no MP3, MP4 or WAV anywhere in it, and the
> microphone stream is a two-column table of timestamps and float64 amplitude samples.
> **ML preprocessing** — MechaSight reads those stored numbers, which is deserialisation, not
> analogue-to-digital conversion. MechaSight performs no ADC and never sees an analogue signal or
> an audio file."

---

## 21. Issues by severity

| # | Issue | Severity | Affects | Correctable without rescoring |
|---|---|---|---|---|
| 1 | F1 published as null/UNDEFINED for 10 RoboticArm Isolation-Forest rows (and both family rows) where the count identity gives a well-defined **0.0** | **MAJOR** | threshold metrics, interpretation | Yes — from stored confusion counts |
| 2 | Stale project documents state "no anomaly-detection model has been trained" after the test has been opened (`README.md`, `docs/research_definition.md`, `docs/evaluation_protocol.md`) | **MAJOR** | documentation only | Yes — text only |
| 3 | Domain-slice AP published with no prevalence baseline; 4 values are below their own chance level while looking high (RA target baseline 0.853) | **MAJOR** | interpretation | Yes — arithmetic on published counts |
| 4 | RA Isolation-Forest FPR 0.0000 presentable as specificity when it is the same "nothing flagged" artefact as recall 0.000 | **MODERATE** | interpretation | Yes — annotation |
| 5 | Healthy-target FPR reported without any interval; RA rests on 20 segments from 6 parents | **MODERATE** | uncertainty, interpretation | Partly — the caveat is text; an interval would need a new (permissible) computation on stored scores |
| 6 | Live evaluation configs and `docs/evaluation_protocol.md` still say Protocol-B primary aggregation is "median of window anomaly scores"; Phase 3.1 uses median of **features** | **MODERATE** | documentation only | Yes — text/annotation |
| 7 | Family "arithmetic mean of five seed metrics" silently becomes undefined if any seed metric is undefined (RA/T precision) | **MINOR** | threshold metrics, documentation | Yes |
| 8 | Sensor wording "three sensors" where two physical devices provide three modalities | **MINOR** | documentation only | Yes |
| 9 | Weight-matrix ↔ score-frame column alignment relies on a consistent `sort_values("segment_id")` convention rather than an explicit assertion (holds in fact — S/T cohorts verified identical) | **MINOR** | robustness of future code | N/A — hardening only |
| 10 | Duplicated `## Domain-slice ranking metrics` heading in `phase32_first_test_report.md`; healthy-target table shows fractional counts without the seed-mean note carried in the CSV | **COSMETIC** | documentation only | Yes |

**No CRITICAL issue was found. No issue affects model scores, fitted models, thresholds,
common-test membership, experiment identities or the scientific configuration.**

---

## 22. Is a reporting correction required?

**Yes — a versioned Phase-3.2.1 reporting correction (option C).** It is required for issues 1, 2
and 3 and should absorb 4, 6, 7, 8 and 10.

### Permitted scope of a Phase-3.2.1 correction gate

**Allowed**

1. Emit a new, additionally-versioned reporting layer (e.g. `reports/phase3_2_1/`) that restates
   the existing results with:
   - F1 computed as `2TP/(2TP+FP+FN)` from the **already stored** confusion counts, reported
     alongside the original harmonic-mean value, with both conventions named;
   - precision explicitly labelled "undefined — no positive predictions" where `TP = FP = 0`;
   - an anomaly-prevalence baseline column beside every Average Precision value;
   - an explicit annotation on RA Isolation-Forest rows that FPR 0.0000 and recall 0.000 share one
     cause (no segment exceeded the threshold), with the §6 diagnostic;
   - the cluster counts behind every threshold-dependent rate (e.g. "20 segments / 6 parents").
2. Add an explicit undefined-metric convention to a **new** reporting-policy version, leaving
   `p311-reporting-5e76ab7dabd5689c7b0ced68` untouched as a historical record.
3. Add tests pinning the corrected convention.
4. Synchronise stale documentation (§23) and the sensor/aggregation wording.
5. Optionally compute a cluster-bootstrap interval for healthy-target FPR from the **existing**
   stored scores and frozen thresholds — no refit, no new configuration, no new ledger entry.

**Forbidden**

- Any change to scores, fitted models, thresholds, split membership, experiment identities,
  hyperparameters, seeds, preprocessing, features, windowing or the adaptation budget.
- Overwriting or deleting any Phase-3.2 artifact, or editing `reports/phase3_2/`.
- Any modification to the test-look ledger.
- Any new model, regime, fusion, deep model or corruption experiment.

The original Phase-3.2 record must remain intact and citable; the correction supersedes its
*presentation*, not its evidence.

---

## 23. Stale documentation

| Location | Stale statement | Classification |
|---|---|---|
| `README.md:9` | status "Phase 3.0.1 … — Phase 2.2 accepted / GREEN" | **CRITICAL TO CORRECT BEFORE SUPERVISOR** |
| `README.md:17` | "No anomaly-detection model has been trained." | **CRITICAL TO CORRECT BEFORE SUPERVISOR** |
| `README.md:126` | roadmap shows "→ Phase 3.1 classical baselines" as pending | **CRITICAL TO CORRECT BEFORE SUPERVISOR** |
| `docs/research_definition.md:3-7` | "no models have been trained"; "MechaSight has produced no model results of its own" | **CRITICAL TO CORRECT BEFORE SUPERVISOR** |
| `docs/evaluation_protocol.md:3-6` | "Phase 3.0 implements the safety/evaluation harness; no model has been trained." | **CRITICAL TO CORRECT BEFORE SUPERVISOR** |
| `reports/supervisor/*.pdf/.docx` (untracked, 23 Aug) | states Phase 3 not started / no model trained | **CRITICAL TO CORRECT BEFORE SUPERVISOR** — do not present this build |
| `reports/synopsis/*.pdf/.docx` (untracked, 3 Sep) | predates the recorded results; must be re-checked line by line | **CRITICAL TO CORRECT BEFORE SUPERVISOR** |
| `configs/evaluation/protocol_b_few_shot_adaptation_v3.json` `score_aggregation.protocol_b_primary` | "median of window anomaly scores" | **CORRECT BEFORE FINAL REPORT** |
| `docs/evaluation_protocol.md:340` | "frozen to the median window anomaly score" | **CORRECT BEFORE FINAL REPORT** |
| `docs/preprocessing_phase2.md:3-4` | "No anomaly-detection model is present or trained." | **CORRECT BEFORE FINAL REPORT** |
| `docs/leakage_risks.md:7` | "Nothing here has been implemented yet — Phase 0 produces the register" | **CORRECT BEFORE FINAL REPORT** |
| `docs/leakage_risks.md:176` | "median of window scores" as the primary rule | **CORRECT BEFORE FINAL REPORT** |
| `README.md:64`, `docs/source_verification.md:47`, `docs/assumptions_and_open_questions.md:143` | "three sensors" | **CORRECT BEFORE FINAL REPORT** |
| `configs/phase3/first_test_reporting_policy_v1.json` status `PRETEST_FROZEN_NOT_YET_EXECUTED` | historical freeze record | **HISTORICAL — LEAVE AS-IS** |
| `reports/phase2/`, `reports/phase2_1/`, `reports/phase3_0/`, `reports/phase3_1/` | phase-scoped statements true at their time | **HISTORICAL — LEAVE AS-IS** |
| `docs/assumptions_and_open_questions.md` Phase-0 framing | superseded items already marked | **HISTORICAL — LEAVE AS-IS** |

Nothing in this list was edited by this audit.

---

## 24. Recommended next phase — not executed

**Recommendation: A — reporting correction and documentation synchronisation (Phase 3.2.1), before
any new experiment.** Rationale: the results are sound; the way they are currently published
overstates them in three places, and two of those would mislead a supervisor immediately. Fixing
presentation is cheap, opens no new test look, and is the only work whose scope is fully
pre-determined by evidence already in hand.

### Pre-test-planned vs post-test-inspired

| Candidate | Status | Note |
|---|---|---|
| **A. Reporting correction / doc sync** | pre-test-planned (arithmetic on published counts) | recommended next |
| **B. Protocol-A official baseline reproduction** | **pre-test-planned** — declared since Phase 1.5 as the comparability anchor and the leakage-bias test | strongest candidate for the next *experimental* phase; it answers the pre-registered question of how much the non-recording-disjoint official split matters |
| **C. True single-modality benchmark** | **pre-test-planned** — Objective 3 requires genuinely retrained single-modality models, not masked inputs | needed before any fusion claim; note it opens a new test look and needs its own pre-registration |
| **D. Reconstruction autoencoder** | pre-test-planned (Phase 4 in the roadmap) | |
| **E. Temporal model** | pre-test-planned (Phase 5) | |
| **F. Fusion** | pre-test-planned (Phase 6) but **must not** be re-scoped using Phase-3.2 outcomes | the reliability estimator design must not be chosen because OCSVM happened to win on BM |
| **G. Robustness / corruption** | pre-test-planned (Phase 7) | |
| Changing the threshold policy because IF got zero recall | **POST-TEST-INSPIRED — forbidden as primary** | may only be run as an explicitly-labelled exploratory sensitivity analysis, never as a primary result, and never retro-fitted to Phase 3.2 |
| Dropping Isolation Forest, or preferring OCSVM, on the basis of these results | **POST-TEST-INSPIRED** | the ladder was pre-registered; families are not removed because of a first-test outcome |

**Nothing was executed.** No new model, regime, fusion, corruption or deep-model work was started.

---

## 25. Unresolved questions

1. **Is the q90 policy fit for purpose given a 6-parent RoboticArm calibration cohort?** The audit
   establishes that it does not transfer for Isolation Forest. Whether the remedy is a different
   calibration population, a rank-based operating point, or accepting the finding as a reported
   negative result is a *pre-registration* decision for a future phase — it cannot be settled now
   without becoming post-test tuning.
2. **How much of the BrushlessMotor signal is fault and how much is session?** Not identifiable
   within IMAD-DS. Only new acquisition with interleaved normal/anomalous sessions could resolve it.
3. **Does the RoboticArm official-split leakage actually inflate AUC, and by how much?** Still
   unmeasured; this is precisely what candidate B would answer.
4. **Why does adaptation help BM One-Class SVM (+0.159 AUROC) but hurt BM Isolation Forest
   (−0.059)?** Unexplained; with 18 adaptation segments and the confound, it may not be explainable
   from this dataset.
5. **Is a 27-segment adaptation budget large enough for any conclusion about adaptation?** The RA
   deltas are within ±0.013 on every metric — arguably the honest reading is "no resolvable effect
   at this budget" rather than "adaptation slightly hurts".
6. **Should healthy-target FPR carry a cluster-bootstrap interval?** The audit recommends yes; it
   is computable from stored scores, but it is a reporting-policy addition and should be versioned.

---

## 26. Audit artifacts

Created by this audit (nothing else was written):

| Path | Contents |
|---|---|
| `docs/phase32_independent_audit.md` | this document |
| `reports/phase3_2_audit/phase32_independent_audit.json` | machine-readable findings and verdict |
| `reports/phase3_2_audit/metric_reproduction_check.csv` | stored vs independently recomputed metrics, all 28 |
| `reports/phase3_2_audit/undefined_metric_correction_preview.csv` | per-row null-F1 diagnosis and count-identity value |
| `reports/phase3_2_audit/family_undefined_metric_preview.csv` | family-level equivalent |
| `reports/phase3_2_audit/domain_slice_prevalence_context.csv` | AP with prevalence baselines |
| `reports/phase3_2_audit/threshold_transfer_diagnostic.csv` | calibration vs test score-scale diagnostic |
| `reports/phase3_2_audit/cluster_independence_summary.csv` | segments vs independent recordings per slice |

No Phase-3.2 result was overwritten. No model output was created. The ledger is unchanged.
