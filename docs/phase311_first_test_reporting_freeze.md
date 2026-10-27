# Phase 3.1.1 First Common-Test Reporting Freeze

## Status

This is a pre-test reporting-policy freeze. No common-test feature, score, label-derived result, performance metric, test-look entry, or real bootstrap replicate was created. The 28 Phase-3.1 experiment configurations, fitted preprocessing, hyperparameters, seeds, and q90 thresholds remain unchanged.

## Endpoints and slices

For each machine independently, each regime, and each model family, the primary ranking endpoints are AUROC and Average Precision. The secondary ranking endpoint is the McClish-standardized partial AUROC for FPR at most 0.10. Threshold-dependent reporting uses each experiment's frozen source-validation q90 and contains TN, FP, FN, TP, FPR, recall, precision, and F1.

The primary level is the complete authorized common-test cohort for one machine, reported separately for S and T. Source-domain and target-domain slices receive the same ranking metrics when mathematically defined. Healthy-target reporting contains its sample count, FP count, and FPR. Any later operating-condition breakdown is secondary/descriptive and cannot replace the primary result. No test-manifest metadata was inspected in this phase, so class-dependent slice metrics are prospectively conditional on being mathematically defined.

## Seed reporting

Isolation Forest is one stochastic model family with five pre-registered replications, seeds 0–4. Its primary family value is the arithmetic mean of all five seed-level values. Every individual value, population standard deviation, minimum, and maximum is also reported. No valid seed may be removed and no best seed may be selected. Each seed remains bound to its own already-frozen q90 threshold. Statistical reference and One-Class SVM remain deterministic single-run models with canonical seed record 0.

## Paired inference

The frozen production design remains a 10,000-replicate, seed-1995, 95% percentile class-stratified cluster bootstrap conditional on the observed evaluation cohort. RoboticArm resamples physical parent recordings; BrushlessMotor resamples its audited one-recording/one-segment units.

Within an Isolation-Forest bootstrap replicate, one sampled-unit draw is shared across all five seeds. The metric is computed separately for each seed and then averaged; that replicate-level family average enters a paired difference. The same sampled evaluation units and multiplicities are used across compared models.

For each machine and model family, adaptation effect is `metric(T) - metric(S)` on the identical common-test cohort with paired inference. It is not a causal estimate beyond this benchmark. Within each machine and regime, the pre-registered family comparisons are Statistical–Isolation Forest, Statistical–One-Class SVM, and Isolation Forest–One-Class SVM. Machines are never pooled into a universal winner.

Interpretation emphasizes effect sizes, paired differences, confidence intervals, and consistency. No p-values or unregistered significance procedure are added.

## Change control

Policy identity: `p311-reporting-5e76ab7dabd5689c7b0ced68`.

After the first test look, a policy-content change requires a new schema version and explicit post-test documentation. Primary endpoints cannot be changed retrospectively.
