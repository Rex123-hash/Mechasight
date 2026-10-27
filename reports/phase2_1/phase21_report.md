# Phase 2.1 — matched evaluation cohort and metric freeze

## STATUS

**COMPLETE — awaiting independent review.** This is a protocol correction only. No anomaly detector,
feature selector, threshold optimiser, or UI was trained or built.

## ROOT CAUSE OF UNEQUAL S/T TEST SETS

Phase-2 v1 treated every target normal unavailable to Regime S as test data. Regime T instead split
those records between train/validation, so S/T changed both training information and evaluation
population. The historical v1 split and evidence remain under `reports/phase2/`, explicitly
superseded for future S-vs-T comparisons.

## CORRECTED COMMON-TEST DESIGN

Three normal pools are now explicit: `source_training`, `target_adaptation`, and
`common_evaluation`, plus anomaly test records. Regime S reserves the adaptation pool; Regime T may
fit on it. Source train, source validation, common-test segments, common-test parents and anomaly
test segments are identical. Frozen split: `protocol_b_common_test_v2.csv`.

Reason: **Regime S and Regime T must be compared on an identical held-out evaluation cohort so that
performance differences are attributable to target-normal adaptation rather than test-set
composition.**

## TARGET ADAPTATION POOL

| Machine | Released target-normal examples | Released parents | Leakage-safe parents | Leakage-safe segments | Added siblings |
|---|---:|---:|---:|---:|---:|
| RoboticArm | 27 | 18 | 18 | 123 | 96 |
| BrushlessMotor | 18 | 18 | 18 | 18 | 0 |

RoboticArm parents are indivisible: the 96 sibling excerpts released as target-normal test data must
join their 27 released-train siblings in adaptation and cannot enter common test.

## FINAL REGIME-S SPLIT COUNTS

| Machine | Train rec/seg | Validation rec/seg | Common test rec/seg | Reserved adaptation rec/seg |
|---|---:|---:|---:|---:|
| RoboticArm | 44 / 1528 | 6 / 186 | 33 / 466 | 18 / 123 |
| BrushlessMotor | 1137 / 1137 | 126 / 126 | 312 / 312 | 18 / 18 |

## FINAL REGIME-T SPLIT COUNTS

| Machine | Train rec/seg | Validation rec/seg | Common test rec/seg |
|---|---:|---:|---:|
| RoboticArm | 62 / 1651 | 6 / 186 | 33 / 466 |
| BrushlessMotor | 1155 / 1155 | 126 / 126 | 312 / 312 |

## PROOF THAT S/T TEST IDS ARE IDENTICAL

| Machine | S/T test segments | Segment symmetric difference | S/T test parents | Parent symmetric difference | Anomaly difference |
|---|---:|---:|---:|---:|---:|
| RoboticArm | 466 / 466 | 0 | 33 / 33 | 0 | 0 |
| BrushlessMotor | 312 / 312 | 0 | 312 / 312 | 0 | 0 |

Machine-readable proof: `matched_test_identity_proof.csv` and `phase21_validation.json`.

## VALIDATION POLICY

Both regimes use identical fixed source-normal validation parents: RA 6 parents/186 segments; BM
126 parents/segments. Target adaptation is train-only in T and reserved in S. Validation alone may
control early stopping and model/window/feature choices. Grouped CV was considered but rejected to
avoid multiplying every model fit and changing calibration populations; the small RA parent count
is disclosed as a limitation. Test data are forbidden for every selection or fitting decision.

## THRESHOLD CALIBRATION POLICY

Primary threshold-free metrics are AUROC and AUPRC; pAUROC@0.1 is secondary. F1, sensitivity,
specificity and FPR are threshold-dependent. The threshold is the empirical q90 of source-validation
normal **segment** scores using quantile method `higher`; anomaly is strict `score > threshold`.
No anomaly label or test score is used.

## WINDOW-TO-SEGMENT AGGREGATION POLICY

Protocol A retains the pinned upstream median reconstruction error. Protocol B primary aggregation
is frozen to median window anomaly score. Each segment score retains all contributing window ids.
Test-set aggregation search is forbidden.

## STATISTICAL UNCERTAINTY / CLUSTERING POLICY

RA uses 10,000 seed-1995 parent-recording cluster-bootstrap replicates, stratified by domain ×
anomaly status × operating condition. Selected parents contribute every child segment. Paired model
comparisons share parent draws and report percentile intervals on metric differences. BM uses the
same design at recording/segment level, while retaining its fault/session confound disclosure.
Windows are never inferential or bootstrap units.

## CACHE INVALIDATION CHANGES

New split/config versions are `protocol_b_common_test_v2` / `phase2.1-v2`. Config ids are
`phase2.1-v2-8688f5639d9fa7c1` (S) and `phase2.1-v2-f497d27caee67b4a` (T), distinct from v1. The CLI
validates split-version compatibility, so v2 cannot consume v1 split manifests or caches. Historical
files were not overwritten.

## TESTS AND SMOKE BUILDS

Eighteen Phase-2.1 tests were added, covering real ID equality, adaptation expansion,
deterministic rebuilding, config pairing, stale-split rejection, experiment invariants, threshold
quantile, median provenance and sibling-preserving cluster draws. Total suite: **122 passing**.

Four one-segment common-test smoke builds produced identical physical windows across S/T: RA 74+74
and BM 51+51, for **4 segments / 250 windows**. No full corpus was regenerated.

## INDEPENDENT DISAGREEMENTS / CHALLENGES

- Parent-safe adaptation reduces RA target common test to 20 normal versus 116 anomaly segments;
  target-only uncertainty and AUPRC prevalence must be reported plainly.
- RA validation has only six independent parents despite 186 segments. The fixed paired policy is
  simpler than grouped CV but model-selection instability remains possible.
- The BM class/session confound is unchanged.
- AUPRC is sensitive to the designed evaluation prevalence and is not a plant-prevalence estimate.

## OPEN QUESTIONS FOR PHASE 3

- Freeze each paired model/feature configuration before its first test evaluation.
- Keep window, feature, model, aggregation and metric implementations identical across S/T.
- Decide the small validation-only window/normalisation comparison without accessing common test.
- Implement the frozen parent-cluster bootstrap before reporting model comparisons.

## FILES MODIFIED

New v2 configs and `configs/evaluation/protocol_b_common_test_v2.json`; split construction and
validation scripts; matched-cohort and paired-evaluation assertions; threshold/aggregation helpers;
Phase-2.1 tests and reports; README, evaluation protocol, preprocessing design and leakage register.
