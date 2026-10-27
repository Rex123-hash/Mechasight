# Phase 3.1 Pre-Test Classical-Baseline Freeze

## Scope and lock

Phase 3.1 authorizes real fitting on the frozen Protocol-B normal training cohorts and scoring on the frozen source-normal validation cohorts only. It does not authorize common-test feature materialization, scoring, metrics, bootstrap inference, model ranking, or model selection. Protocol A is unchanged and is not run here.

The primary preprocessing and model choices below were registered before any common-test evaluation. They were selected for methodological continuity with the accepted Phase-2 pipeline, numerical transparency, and reasonable classical-model computation—not because of validation or test performance. The source-normal validation cohort has no anomaly class and cannot establish anomaly-detection superiority.

## Primary pre-registered preprocessing

- Protocol: `mechasight_experimental_protocol`, regimes S and T kept explicit.
- Startup handling: 0.035 s gyroscope warm-up removal; one initial BrushlessMotor accelerometer sample removed; no RoboticArm accelerometer sample removed.
- Windows: 0.1 s, 0.1 s stride, half-open and non-overlapping.
- Window features: the existing Phase-2 time- and frequency-domain engineered features, with no signal-level normalization.
- Estimator sample: one vector per frozen segment, formed by the median of each feature over all valid windows in that segment.
- Estimator normalization: feature-wise population z-score using means and standard deviations fitted only on the complete authorized training cohort for that machine and regime, with epsilon (10^{-5}). Regime T therefore includes only its legitimate target-normal adaptation allocation. No fitted normalizer is shared silently between S and T.
- Score unit: one score per segment, so no second score-aggregation step is applied.
- Threshold: the frozen higher-is-more-anomalous q90 rule, fitted only on the complete source-normal validation score cohort.

The segment median is a conservative robust summary and reduces the highly correlated window stream to the frozen experimental unit. It also makes the kernel baseline computationally practical. This is not an optimality claim. Alternative window lengths, strides, window-level fitting, aggregators, and training-only scalers remain sensitivity configurations and were not run in Phase 3.1.

## Model definitions

### Statistical reference

For training feature (j), fit the training-only mean \(\mu_j\) and population standard deviation \(\sigma_j\). For segment vector \(x\), define \(z_j=(x_j-\mu_j)/(\sigma_j+10^{-5})\) and

\[
s(x)=\sqrt{\frac{1}{d}\sum_{j=1}^{d}z_j^2}.
\]

The method is deterministic. Larger distance from the standardized training center means more anomalous. It assumes that equally weighted standardized marginal departures are informative; it ignores feature covariance and may underrepresent structured or multimodal normal behavior.

### Isolation Forest

Implementation: `sklearn.ensemble.IsolationForest`. Frozen parameters are `n_estimators=200`, `max_samples="auto"`, `contamination="auto"`, `max_features=1.0`, `bootstrap=false`, and `n_jobs=1`, with seeds 0–4. The official MechaSight score is `-score_samples(x)`, converting scikit-learn's larger-more-normal convention into larger-more-anomalous. Two hundred trees is a computationally moderate variance-reduction choice; it is intentionally not described as optimized.

### One-Class SVM

Implementation: `sklearn.svm.OneClassSVM`. Frozen parameters are RBF kernel, `gamma="scale"`, `nu=0.05`, `degree=3`, `coef0=0`, `tol=0.001`, `shrinking=true`, `cache_size=512`, and `max_iter=-1`. It is deterministic and therefore has one canonical seed record, 0, rather than five artificial repeats. The official MechaSight score is `-decision_function(x)`. The value `nu=0.05` is a materially judgment-based conservative allowance for training-boundary violations, not a tuned optimum. Kernel behavior may be sensitive to scaling, dimensionality, and the geometry of normal data.

No score is a calibrated probability or confidence value, and magnitudes from different model families are not comparable.

## Frozen cohorts and limitations

RoboticArm S trains on 1,528 source-normal segments. RoboticArm T trains on those 1,528 plus 27 authorized target-normal adaptation segments from 18 physical parents; 96 same-parent siblings remain blocked. Source-normal validation contains 186 correlated segment excerpts from exactly six physical parent recordings.

BrushlessMotor S trains on 1,137 source-normal segments. BrushlessMotor T trains on those 1,137 plus 18 authorized target-normal adaptation segments. Source-normal validation contains 126 segments. Every BrushlessMotor result retains limitation ID `brushless_motor_acquisition_time_class_confound_v1`: fault/class and acquisition session/time cannot be fully separated in the released cohort, so later results cannot be interpreted as pure mechanical-fault evidence.

## Authorized first-test configurations

The later first common-test evaluation is proposed for exactly the 28 immutable experiment configurations produced by the Cartesian product of two machines, two regimes, and these seed policies: statistical reference seed record 0; Isolation Forest seeds 0, 1, 2, 3, and 4; One-Class SVM seed record 0. Their deterministic experiment IDs, exact q90 thresholds, environment, cohort hashes, and validation-only artifacts are recorded in `reports/phase3_1/phase31_pretest_freeze_report.json`.

After a first test look, any preprocessing, feature, model, hyperparameter, seed-policy, threshold-policy, aggregation, data-membership, or code-identity change must receive a new experiment ID and be labelled post-test or sensitivity work.

### Code-identity clarification added in Phase 3.1.1

The immutable bundle field named `code_git_sha` contains `3bf046dd8edb29d99f1d246b33f88674a5b53a2d`. That value is the accepted base commit on which Phase 3.1 was implemented; it is not the final commit containing the newly added Phase-3.1 implementation. The per-bundle `phase31_source_hash` value `93c33580054e9e45f228e385ba726bb43ad0dfb3531f83bb9f3c9c5fb003e229` is the content identity for the six explicitly covered Phase-3.1 source/configuration paths. Final commit `78b536304d94d50afcc293348c5fa0c34e729f62` reproduces that snapshot exactly and is the authoritative complete-tree generation commit mapping. The original bundles remain immutable; the Phase-3.1.1 code-identity audit records this semantic correction before any test access.

## Excluded activity

Phase 3.1 does not implement or run XGBoost, LightGBM, autoencoders, CNNs, TCNs, fusion, corruption experiments, final selection, Protocol-A reproduction, or the real 10,000-replicate common-test bootstrap. Validation summaries are descriptive checks of finite output, score direction, stability, reproducibility, and practicality only; they do not rank the models.
