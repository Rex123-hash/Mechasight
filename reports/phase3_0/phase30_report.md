# Phase 3.0 — experiment safety, provenance, and evaluation preflight

## STATUS

**CORRECTED BY PHASE 3.0.1 — awaiting independent audit.** Phase 2.2 is recorded as independently
accepted / GREEN. The original Phase 3.0 bootstrap stratification and ambiguous checkpoint wording
were superseded by the narrow correction documented in `phase301_correction_report.md`.

**NO REAL ANOMALY MODEL WAS TRAINED IN PHASE 3.0.**

**NO REAL TEST PERFORMANCE WAS INSPECTED IN PHASE 3.0.**

Real-data activity was limited to reading the accepted
`protocol_b_few_shot_adaptation_v3.csv` identities/counts. All estimator, score, threshold, metric,
bootstrap, persistence, ledger and evaluation-gate tests use synthetic fixtures and pytest temporary
directories.

## IMPLEMENTED

- Label-free `EstimatorTrainingData` and `TrainingCohortContract`, bound to protocol, regime, machine,
  v3 split/preprocessing identity, preprocessing config id and exact authorized segment ids.
- Public train-fitted normalization path that validates actual manifest membership before numerical
  fitting. It rejects validation/test/anomaly/blocked/wrong-provenance rows.
- Structured `ScoreProvenance` / `ScoreBatch`, frozen `higher_is_more_anomalous` direction and
  source-normal-validation-only q90 (`higher`) threshold selection.
- Protocol-B scikit-learn metrics: AUROC, average precision and standardized partial AUROC at
  `max_fpr=0.10`, plus confusion counts, FPR, recall, precision and F1 at a validation-derived
  threshold.
- Strict median window-to-segment aggregation retaining unique contributing window IDs/count.
- Complete metric and paired-difference percentile bootstrap engines using class-stratified cluster
  resampling after machine/domain reporting slices, invalid-replicate accounting and shared paired draws.
- Strict experiment schema `phase3.0.1-experiment-v1`, deterministic scientific identity,
  machine-specific limitation metadata and exact model/bootstrap seed policies.
- Exclusive atomic result bundles, append-only JSONL test-look ledger and a validation/test
  `EvaluationGate` that logs test access before computing requested metrics.
- Protocol-B preprocessing-candidate and baseline-ladder scaffolds. No candidate was selected and no
  model family was implemented or fitted.

## FROZEN POLICY

- `MODEL_SEEDS = [0, 1, 2, 3, 4]`; `BOOTSTRAP_SEED = 1995`;
  bootstrap replicates = 10,000; confidence = 95%; percentile intervals.
- Regime S fitting = source-normal training only.
- Regime T fitting = the same source normals plus released target normals: RA +27, BM +18.
- RA 96 same-parent siblings remain blocked; 123 is parent closure, never the primary fit budget.
- Target adaptation may enter training-fitted normalization if selected, but never validation,
  threshold selection or test-driven hyperparameter selection.
- Score direction = higher means more anomalous. Scores are not probabilities.
- Protocol-B segment aggregation = median.
- Threshold = source-normal validation q90, NumPy quantile method `higher`, decision `score > threshold`.
- MechaSight's AUPRC field is operationally average precision (`average_precision`).
- pAUROC is scikit-learn's **standardized** partial AUC at FPR ≤ 0.10, not raw geometric area.
- Protocol A is untouched. Test performance may never select Protocol-B preprocessing.

## STILL OPEN

- Final Protocol-B window length, stride/overlap and normalization choice.
- Exact Phase-3.1 statistical baseline and its hyperparameters.
- How the released target-normal examples enter each later model family beyond the frozen rule that
  they are training data.
- Whether an embedding-distance comparator can be supported by primary literature.
- Whether any scientifically compatible tree-based normal-only objective should be admitted.

## ARCHITECTURE

`src/experiment/` contains constants/provenance, the training data contract, score/threshold
provenance, aggregation, metrics, bootstrap inference, experiment config, result store, ledger and
evaluation gate. Mathematical helpers remain callable, but the official Phase-3 pathway binds them
to validated cohort and experiment identities.

The canonical preprocessing manifest retains labels for audit/evaluation. Only explicitly selected
feature columns enter estimator `X`; no anomaly target exists on the estimator-facing object.

## EXPERIMENT / RESULT SCHEMAS

- Experiment: `phase3.0.1-experiment-v1`
- Metric policy: `phase3.0-protocol-b-metrics-v1`
- Threshold: `phase3.0-source-validation-q90-v1`
- Results: `phase3.0.1-result-v1`
- Test-look ledger: `phase3.0-test-look-v1`
- Adaptation: `phase2.2-released-target-normals-v1`

Experiment IDs are `p3-` plus the first 24 hexadecimal characters of the canonical scientific
configuration SHA-256. Run timestamp and dependency-location metadata are retained but do not rename
an otherwise identical scientific configuration. Scientifically relevant field changes do.

Result bundles are designed as `reports/experiment_results/<experiment_id>/` containing
`config.json`, `provenance.json`, `metrics.json`, `scores.parquet` and a file-hash manifest. Phase 3.0
wrote bundles only under pytest temporary directories. An existing experiment directory is never
overwritten.

## TESTS

Command:

```text
python -m pytest -q -p no:cacheprovider
```

Original Phase 3.0 result: **210 passed, 0 failed, 0 skipped, 0 warnings in 13.97 s**. The current
post-correction result is recorded in `phase301_correction_report.md`.

## DEPENDENCIES TESTED

- Python 3.14.4
- NumPy 2.4.4
- pandas 3.0.2
- PyArrow 24.0.0
- h5py 3.16.0
- pytest 9.0.3
- scikit-learn 1.9.0 (new project pin)
- SciPy 1.18.0
- joblib 1.5.3

`pip check` reports pre-existing missing/incompatible optional Google ADK environment dependencies;
none is imported by MechaSight and the complete MechaSight suite passes. No dependency was changed
to address that unrelated global-environment state.

## LIMITATIONS PRESERVED

- BrushlessMotor has an acquisition-time/class confound: normals occur earlier and anomalies later,
  so mechanical-fault effects cannot be separated completely from acquisition-session/drift effects.
  Early-vs-late normal analysis is diagnostic only and cannot deconfound fault from session.
- RoboticArm model selection has six independent validation parent recordings. Its 186 excerpts are
  explicitly not treated as 186 independent units; every future RA config/result bundle carries
  that warning.
- Separately, RoboticArm target adaptation has 27 released segments from 18 parents. Its 96 blocked
  siblings share those same 18 parents, producing a 123-segment closure that is not the fit budget.
- Python permits bypassing public APIs. The official pathway provides strong validation, persistence
  and auditing guardrails, not a security sandbox.

## FILES CREATED / MODIFIED

Created: `src/experiment/` (12 modules including package initialization), three `tests/test_phase30_*.py` files,
`configs/phase3/preprocessing_candidates_v1.json`, `configs/phase3/baseline_ladder_v1.json`,
`docs/experiment_safety_phase3.md`, and this report.

Modified: `README.md`, `docs/research_definition.md`, `docs/evaluation_protocol.md`,
`docs/leakage_risks.md`, `src/preprocessing/pipeline.py`,
`reports/phase2_2/phase22_report.md`, and `requirements.txt`.
