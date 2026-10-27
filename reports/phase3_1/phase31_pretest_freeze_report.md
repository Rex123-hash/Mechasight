# Phase 3.1 Pre-Test Freeze Report

Generated: 2026-09-03T04:27:53.306527+00:00

Status: validation-only pre-test baseline fitting complete.

Common-test evaluation: **NO**. No test cohort was materialized or scored.

Test-look ledger entries before/after: 0/0.

Accepted starting HEAD: `3bf046dd8edb29d99f1d246b33f88674a5b53a2d`.

Phase-3.1 source snapshot hash: `93c33580054e9e45f228e385ba726bb43ad0dfb3531f83bb9f3c9c5fb003e229`.

| Machine | Regime | Model | Seed | Train segments | Validation segments | Features | q90 threshold |
|---|---:|---|---:|---:|---:|---:|---:|
| RoboticArm | S | statistical_reference | 0 | 1528 | 186 | 98 | 1.99152114187 |
| RoboticArm | S | isolation_forest | 0 | 1528 | 186 | 98 | 0.576885511269 |
| RoboticArm | S | isolation_forest | 1 | 1528 | 186 | 98 | 0.581910978505 |
| RoboticArm | S | isolation_forest | 2 | 1528 | 186 | 98 | 0.591337331443 |
| RoboticArm | S | isolation_forest | 3 | 1528 | 186 | 98 | 0.582332763263 |
| RoboticArm | S | isolation_forest | 4 | 1528 | 186 | 98 | 0.577503625371 |
| RoboticArm | S | one_class_svm | 0 | 1528 | 186 | 98 | 1.3073284147 |
| RoboticArm | T | statistical_reference | 0 | 1555 | 186 | 98 | 1.97871659428 |
| RoboticArm | T | isolation_forest | 0 | 1555 | 186 | 98 | 0.566220791596 |
| RoboticArm | T | isolation_forest | 1 | 1555 | 186 | 98 | 0.570903800979 |
| RoboticArm | T | isolation_forest | 2 | 1555 | 186 | 98 | 0.592390878481 |
| RoboticArm | T | isolation_forest | 3 | 1555 | 186 | 98 | 0.573334839435 |
| RoboticArm | T | isolation_forest | 4 | 1555 | 186 | 98 | 0.578646931039 |
| RoboticArm | T | one_class_svm | 0 | 1555 | 186 | 98 | 1.06959646968 |
| BrushlessMotor | S | statistical_reference | 0 | 1137 | 126 | 98 | 1.43211649884 |
| BrushlessMotor | S | isolation_forest | 0 | 1137 | 126 | 98 | 0.51848493283 |
| BrushlessMotor | S | isolation_forest | 1 | 1137 | 126 | 98 | 0.517708659275 |
| BrushlessMotor | S | isolation_forest | 2 | 1137 | 126 | 98 | 0.515113961281 |
| BrushlessMotor | S | isolation_forest | 3 | 1137 | 126 | 98 | 0.52094150509 |
| BrushlessMotor | S | isolation_forest | 4 | 1137 | 126 | 98 | 0.513659009999 |
| BrushlessMotor | S | one_class_svm | 0 | 1137 | 126 | 98 | -0.259314009905 |
| BrushlessMotor | T | statistical_reference | 0 | 1155 | 126 | 98 | 1.43135773872 |
| BrushlessMotor | T | isolation_forest | 0 | 1155 | 126 | 98 | 0.514301107586 |
| BrushlessMotor | T | isolation_forest | 1 | 1155 | 126 | 98 | 0.508394101096 |
| BrushlessMotor | T | isolation_forest | 2 | 1155 | 126 | 98 | 0.513855408798 |
| BrushlessMotor | T | isolation_forest | 3 | 1155 | 126 | 98 | 0.515106957244 |
| BrushlessMotor | T | isolation_forest | 4 | 1155 | 126 | 98 | 0.511817683369 |
| BrushlessMotor | T | one_class_svm | 0 | 1155 | 126 | 98 | -0.326718696208 |

Thresholds are descriptive source-normal q90 calibration values, not test results and not a model ranking.

The BrushlessMotor acquisition-time/class confound remains explicitly identified as `brushless_motor_acquisition_time_class_confound_v1`.

The RoboticArm source-validation cohort contains 186 correlated segments from six physical parent recordings; the Regime-T adaptation allocation remains 27 released segments from 18 parents, with 96 siblings blocked.

Protocol A, deep models, fusion, corruptions, bootstrap inference, test metrics, and final model selection were not run.
