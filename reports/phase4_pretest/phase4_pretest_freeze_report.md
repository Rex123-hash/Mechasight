# Single-Modality Model Benchmarking — Pre-Test Freeze Report

Generated: 2026-09-13T14:26:21.834712+00:00

Status: validation-only pre-test fitting of all 156 frozen configurations complete.

Common-test evaluation: **NO**. No test cohort was materialised, scored or metric-evaluated.

Test-look ledger entries before/after: 40/40 (expected start for the protected batch: 40).

Code HEAD at fit time: `85df48d819c25679b596286adda9df21720c93e9`. Source snapshot hash: `7296665253064b97b8b67c3b88714a29dc3b3df79d0b96a439f5eefc909fbc49`.

Proposed protected batch identity (not authorised): `p4-test-batch-f3fa02a57fc7adc6650e7522`.

| Machine | Regime | Modality | Family | Seed | Features/bins | q90 threshold | Epochs | Best epoch | Fit s |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| BrushlessMotor | S | acc | dense_autoencoder | 0 | 1068 | 0.806849 | 32 | 22 | 75.6 |
| BrushlessMotor | S | acc | dense_autoencoder | 1 | 1068 | 0.805989 | 33 | 23 | 80.2 |
| BrushlessMotor | S | acc | dense_autoencoder | 2 | 1068 | 0.806001 | 31 | 21 | 72.8 |
| BrushlessMotor | S | acc | dense_autoencoder | 3 | 1068 | 0.804696 | 28 | 18 | 65.4 |
| BrushlessMotor | S | acc | dense_autoencoder | 4 | 1068 | 0.806605 | 30 | 20 | 71.7 |
| BrushlessMotor | S | acc | isolation_forest | 0 | 42 | 0.514412 |  |  | 0.2 |
| BrushlessMotor | S | acc | isolation_forest | 1 | 42 | 0.516205 |  |  | 0.2 |
| BrushlessMotor | S | acc | isolation_forest | 2 | 42 | 0.51464 |  |  | 0.2 |
| BrushlessMotor | S | acc | isolation_forest | 3 | 42 | 0.518423 |  |  | 0.2 |
| BrushlessMotor | S | acc | isolation_forest | 4 | 42 | 0.516981 |  |  | 0.2 |
| BrushlessMotor | S | acc | mahalanobis_shrinkage | 0 | 42 | 45.0326 |  |  | 0.0 |
| BrushlessMotor | S | acc | one_class_svm | 0 | 42 | -0.436023 |  |  | 0.0 |
| BrushlessMotor | S | acc | statistical_reference | 0 | 42 | 1.52565 |  |  | 0.0 |
| BrushlessMotor | S | gyro | dense_autoencoder | 0 | 1068 | 0.864728 | 48 | 38 | 119.3 |
| BrushlessMotor | S | gyro | dense_autoencoder | 1 | 1068 | 0.867158 | 50 | 40 | 128.7 |
| BrushlessMotor | S | gyro | dense_autoencoder | 2 | 1068 | 0.862592 | 52 | 42 | 220.2 |
| BrushlessMotor | S | gyro | dense_autoencoder | 3 | 1068 | 0.864072 | 62 | 52 | 193.9 |
| BrushlessMotor | S | gyro | dense_autoencoder | 4 | 1068 | 0.866989 | 54 | 44 | 162.0 |
| BrushlessMotor | S | gyro | isolation_forest | 0 | 42 | 0.526999 |  |  | 0.2 |
| BrushlessMotor | S | gyro | isolation_forest | 1 | 42 | 0.531112 |  |  | 0.2 |
| BrushlessMotor | S | gyro | isolation_forest | 2 | 42 | 0.530177 |  |  | 0.2 |
| BrushlessMotor | S | gyro | isolation_forest | 3 | 42 | 0.532625 |  |  | 0.2 |
| BrushlessMotor | S | gyro | isolation_forest | 4 | 42 | 0.518884 |  |  | 0.2 |
| BrushlessMotor | S | gyro | mahalanobis_shrinkage | 0 | 42 | 52.4865 |  |  | 0.0 |
| BrushlessMotor | S | gyro | one_class_svm | 0 | 42 | -0.298388 |  |  | 0.0 |
| BrushlessMotor | S | gyro | statistical_reference | 0 | 42 | 1.3819 |  |  | 0.0 |
| BrushlessMotor | S | mic | dense_autoencoder | 0 | 800 | 0.698105 | 45 | 35 | 88.3 |
| BrushlessMotor | S | mic | dense_autoencoder | 1 | 800 | 0.697885 | 42 | 32 | 80.5 |
| BrushlessMotor | S | mic | dense_autoencoder | 2 | 800 | 0.699609 | 42 | 32 | 82.4 |
| BrushlessMotor | S | mic | dense_autoencoder | 3 | 800 | 0.698396 | 39 | 29 | 78.1 |
| BrushlessMotor | S | mic | dense_autoencoder | 4 | 800 | 0.69739 | 44 | 34 | 84.5 |
| BrushlessMotor | S | mic | isolation_forest | 0 | 14 | 0.527686 |  |  | 0.2 |
| BrushlessMotor | S | mic | isolation_forest | 1 | 14 | 0.53766 |  |  | 0.2 |
| BrushlessMotor | S | mic | isolation_forest | 2 | 14 | 0.529683 |  |  | 0.2 |
| BrushlessMotor | S | mic | isolation_forest | 3 | 14 | 0.549706 |  |  | 0.2 |
| BrushlessMotor | S | mic | isolation_forest | 4 | 14 | 0.52939 |  |  | 0.2 |
| BrushlessMotor | S | mic | mahalanobis_shrinkage | 0 | 14 | 19.8525 |  |  | 0.0 |
| BrushlessMotor | S | mic | one_class_svm | 0 | 14 | -0.417326 |  |  | 0.0 |
| BrushlessMotor | S | mic | statistical_reference | 0 | 14 | 1.46669 |  |  | 0.0 |
| BrushlessMotor | T | acc | dense_autoencoder | 0 | 1068 | 0.795936 | 32 | 22 | 117.7 |
| BrushlessMotor | T | acc | dense_autoencoder | 1 | 1068 | 0.795797 | 29 | 19 | 81.6 |
| BrushlessMotor | T | acc | dense_autoencoder | 2 | 1068 | 0.79436 | 35 | 25 | 94.2 |
| BrushlessMotor | T | acc | dense_autoencoder | 3 | 1068 | 0.796388 | 30 | 20 | 89.8 |
| BrushlessMotor | T | acc | dense_autoencoder | 4 | 1068 | 0.795053 | 38 | 28 | 111.9 |
| BrushlessMotor | T | acc | isolation_forest | 0 | 42 | 0.509986 |  |  | 0.2 |
| BrushlessMotor | T | acc | isolation_forest | 1 | 42 | 0.510787 |  |  | 0.2 |
| BrushlessMotor | T | acc | isolation_forest | 2 | 42 | 0.51189 |  |  | 0.2 |
| BrushlessMotor | T | acc | isolation_forest | 3 | 42 | 0.511035 |  |  | 0.2 |
| BrushlessMotor | T | acc | isolation_forest | 4 | 42 | 0.510656 |  |  | 0.2 |
| BrushlessMotor | T | acc | mahalanobis_shrinkage | 0 | 42 | 43.3291 |  |  | 0.0 |
| BrushlessMotor | T | acc | one_class_svm | 0 | 42 | -0.438996 |  |  | 0.0 |
| BrushlessMotor | T | acc | statistical_reference | 0 | 42 | 1.52465 |  |  | 0.0 |
| BrushlessMotor | T | gyro | dense_autoencoder | 0 | 1068 | 0.861314 | 50 | 40 | 145.9 |
| BrushlessMotor | T | gyro | dense_autoencoder | 1 | 1068 | 0.860149 | 55 | 45 | 167.7 |
| BrushlessMotor | T | gyro | dense_autoencoder | 2 | 1068 | 0.859927 | 54 | 44 | 158.2 |
| BrushlessMotor | T | gyro | dense_autoencoder | 3 | 1068 | 0.862265 | 60 | 50 | 166.7 |
| BrushlessMotor | T | gyro | dense_autoencoder | 4 | 1068 | 0.8626 | 50 | 40 | 151.0 |
| BrushlessMotor | T | gyro | isolation_forest | 0 | 42 | 0.518678 |  |  | 0.2 |
| BrushlessMotor | T | gyro | isolation_forest | 1 | 42 | 0.521344 |  |  | 0.2 |
| BrushlessMotor | T | gyro | isolation_forest | 2 | 42 | 0.522502 |  |  | 0.2 |
| BrushlessMotor | T | gyro | isolation_forest | 3 | 42 | 0.523041 |  |  | 0.2 |
| BrushlessMotor | T | gyro | isolation_forest | 4 | 42 | 0.526673 |  |  | 0.2 |
| BrushlessMotor | T | gyro | mahalanobis_shrinkage | 0 | 42 | 48.6521 |  |  | 0.0 |
| BrushlessMotor | T | gyro | one_class_svm | 0 | 42 | -0.36996 |  |  | 0.0 |
| BrushlessMotor | T | gyro | statistical_reference | 0 | 42 | 1.38685 |  |  | 0.0 |
| BrushlessMotor | T | mic | dense_autoencoder | 0 | 800 | 0.685742 | 38 | 28 | 81.1 |
| BrushlessMotor | T | mic | dense_autoencoder | 1 | 800 | 0.686075 | 42 | 32 | 91.3 |
| BrushlessMotor | T | mic | dense_autoencoder | 2 | 800 | 0.685838 | 33 | 23 | 78.8 |
| BrushlessMotor | T | mic | dense_autoencoder | 3 | 800 | 0.685843 | 38 | 28 | 109.9 |
| BrushlessMotor | T | mic | dense_autoencoder | 4 | 800 | 0.686087 | 45 | 35 | 110.5 |
| BrushlessMotor | T | mic | isolation_forest | 0 | 14 | 0.522348 |  |  | 0.2 |
| BrushlessMotor | T | mic | isolation_forest | 1 | 14 | 0.506844 |  |  | 0.2 |
| BrushlessMotor | T | mic | isolation_forest | 2 | 14 | 0.516522 |  |  | 0.2 |
| BrushlessMotor | T | mic | isolation_forest | 3 | 14 | 0.530134 |  |  | 0.2 |
| BrushlessMotor | T | mic | isolation_forest | 4 | 14 | 0.52577 |  |  | 0.2 |
| BrushlessMotor | T | mic | mahalanobis_shrinkage | 0 | 14 | 19.9106 |  |  | 0.0 |
| BrushlessMotor | T | mic | one_class_svm | 0 | 14 | -0.466744 |  |  | 0.0 |
| BrushlessMotor | T | mic | statistical_reference | 0 | 14 | 1.46689 |  |  | 0.0 |
| RoboticArm | S | acc | dense_autoencoder | 0 | 1059 | 0.824348 | 47 | 37 | 191.8 |
| RoboticArm | S | acc | dense_autoencoder | 1 | 1059 | 0.825346 | 60 | 50 | 236.3 |
| RoboticArm | S | acc | dense_autoencoder | 2 | 1059 | 0.825505 | 49 | 39 | 193.1 |
| RoboticArm | S | acc | dense_autoencoder | 3 | 1059 | 0.824419 | 57 | 47 | 226.1 |
| RoboticArm | S | acc | dense_autoencoder | 4 | 1059 | 0.825263 | 45 | 35 | 176.4 |
| RoboticArm | S | acc | isolation_forest | 0 | 42 | 0.628881 |  |  | 0.2 |
| RoboticArm | S | acc | isolation_forest | 1 | 42 | 0.62869 |  |  | 0.2 |
| RoboticArm | S | acc | isolation_forest | 2 | 42 | 0.63351 |  |  | 0.2 |
| RoboticArm | S | acc | isolation_forest | 3 | 42 | 0.623519 |  |  | 0.2 |
| RoboticArm | S | acc | isolation_forest | 4 | 42 | 0.622199 |  |  | 0.2 |
| RoboticArm | S | acc | mahalanobis_shrinkage | 0 | 42 | 101.216 |  |  | 0.0 |
| RoboticArm | S | acc | one_class_svm | 0 | 42 | 1.84546 |  |  | 0.0 |
| RoboticArm | S | acc | statistical_reference | 0 | 42 | 2.16084 |  |  | 0.0 |
| RoboticArm | S | gyro | dense_autoencoder | 0 | 1059 | 0.96457 | 30 | 20 | 117.3 |
| RoboticArm | S | gyro | dense_autoencoder | 1 | 1059 | 0.964466 | 25 | 15 | 96.6 |
| RoboticArm | S | gyro | dense_autoencoder | 2 | 1059 | 0.962481 | 74 | 64 | 288.5 |
| RoboticArm | S | gyro | dense_autoencoder | 3 | 1059 | 0.964445 | 22 | 12 | 85.0 |
| RoboticArm | S | gyro | dense_autoencoder | 4 | 1059 | 0.964389 | 21 | 11 | 81.7 |
| RoboticArm | S | gyro | isolation_forest | 0 | 42 | 0.560649 |  |  | 0.2 |
| RoboticArm | S | gyro | isolation_forest | 1 | 42 | 0.572194 |  |  | 0.2 |
| RoboticArm | S | gyro | isolation_forest | 2 | 42 | 0.572936 |  |  | 0.2 |
| RoboticArm | S | gyro | isolation_forest | 3 | 42 | 0.551186 |  |  | 0.2 |
| RoboticArm | S | gyro | isolation_forest | 4 | 42 | 0.566231 |  |  | 0.2 |
| RoboticArm | S | gyro | mahalanobis_shrinkage | 0 | 42 | 131.203 |  |  | 0.0 |
| RoboticArm | S | gyro | one_class_svm | 0 | 42 | 0.984095 |  |  | 0.0 |
| RoboticArm | S | gyro | statistical_reference | 0 | 42 | 2.08515 |  |  | 0.0 |
| RoboticArm | S | mic | dense_autoencoder | 0 | 800 | 0.455854 | 150 | 140 | 575.8 |
| RoboticArm | S | mic | dense_autoencoder | 1 | 800 | 0.456056 | 195 | 185 | 1020.8 |
| RoboticArm | S | mic | dense_autoencoder | 2 | 800 | 0.456114 | 145 | 135 | 711.4 |
| RoboticArm | S | mic | dense_autoencoder | 3 | 800 | 0.456362 | 198 | 188 | 871.7 |
| RoboticArm | S | mic | dense_autoencoder | 4 | 800 | 0.455276 | 200 | 198 | 680.3 |
| RoboticArm | S | mic | isolation_forest | 0 | 14 | 0.485153 |  |  | 0.2 |
| RoboticArm | S | mic | isolation_forest | 1 | 14 | 0.495034 |  |  | 0.2 |
| RoboticArm | S | mic | isolation_forest | 2 | 14 | 0.494239 |  |  | 0.2 |
| RoboticArm | S | mic | isolation_forest | 3 | 14 | 0.497534 |  |  | 0.2 |
| RoboticArm | S | mic | isolation_forest | 4 | 14 | 0.493162 |  |  | 0.2 |
| RoboticArm | S | mic | mahalanobis_shrinkage | 0 | 14 | 19.6597 |  |  | 0.0 |
| RoboticArm | S | mic | one_class_svm | 0 | 14 | -0.38738 |  |  | 0.0 |
| RoboticArm | S | mic | statistical_reference | 0 | 14 | 1.35822 |  |  | 0.0 |
| RoboticArm | T | acc | dense_autoencoder | 0 | 1059 | 0.82456 | 66 | 56 | 264.1 |
| RoboticArm | T | acc | dense_autoencoder | 1 | 1059 | 0.824379 | 47 | 37 | 188.0 |
| RoboticArm | T | acc | dense_autoencoder | 2 | 1059 | 0.824725 | 49 | 39 | 196.9 |
| RoboticArm | T | acc | dense_autoencoder | 3 | 1059 | 0.82401 | 51 | 41 | 204.1 |
| RoboticArm | T | acc | dense_autoencoder | 4 | 1059 | 0.824848 | 59 | 49 | 232.1 |
| RoboticArm | T | acc | isolation_forest | 0 | 42 | 0.62571 |  |  | 0.2 |
| RoboticArm | T | acc | isolation_forest | 1 | 42 | 0.621228 |  |  | 0.2 |
| RoboticArm | T | acc | isolation_forest | 2 | 42 | 0.619209 |  |  | 0.2 |
| RoboticArm | T | acc | isolation_forest | 3 | 42 | 0.611758 |  |  | 0.2 |
| RoboticArm | T | acc | isolation_forest | 4 | 42 | 0.612819 |  |  | 0.2 |
| RoboticArm | T | acc | mahalanobis_shrinkage | 0 | 42 | 90.626 |  |  | 0.0 |
| RoboticArm | T | acc | one_class_svm | 0 | 42 | 1.53717 |  |  | 0.0 |
| RoboticArm | T | acc | statistical_reference | 0 | 42 | 2.13743 |  |  | 0.0 |
| RoboticArm | T | gyro | dense_autoencoder | 0 | 1059 | 0.963179 | 96 | 86 | 378.0 |
| RoboticArm | T | gyro | dense_autoencoder | 1 | 1059 | 0.963147 | 102 | 92 | 355.7 |
| RoboticArm | T | gyro | dense_autoencoder | 2 | 1059 | 0.963327 | 23 | 13 | 76.1 |
| RoboticArm | T | gyro | dense_autoencoder | 3 | 1059 | 0.964494 | 22 | 12 | 73.4 |
| RoboticArm | T | gyro | dense_autoencoder | 4 | 1059 | 0.964039 | 27 | 17 | 88.8 |
| RoboticArm | T | gyro | isolation_forest | 0 | 42 | 0.557863 |  |  | 0.2 |
| RoboticArm | T | gyro | isolation_forest | 1 | 42 | 0.569154 |  |  | 0.2 |
| RoboticArm | T | gyro | isolation_forest | 2 | 42 | 0.569379 |  |  | 0.2 |
| RoboticArm | T | gyro | isolation_forest | 3 | 42 | 0.5463 |  |  | 0.2 |
| RoboticArm | T | gyro | isolation_forest | 4 | 42 | 0.570374 |  |  | 0.2 |
| RoboticArm | T | gyro | mahalanobis_shrinkage | 0 | 42 | 131.106 |  |  | 0.0 |
| RoboticArm | T | gyro | one_class_svm | 0 | 42 | 0.844677 |  |  | 0.0 |
| RoboticArm | T | gyro | statistical_reference | 0 | 42 | 2.07949 |  |  | 0.0 |
| RoboticArm | T | mic | dense_autoencoder | 0 | 800 | 0.456217 | 105 | 95 | 606.2 |
| RoboticArm | T | mic | dense_autoencoder | 1 | 800 | 0.455659 | 106 | 96 | 527.4 |
| RoboticArm | T | mic | dense_autoencoder | 2 | 800 | 0.455606 | 200 | 199 | 977.3 |
| RoboticArm | T | mic | dense_autoencoder | 3 | 800 | 0.45517 | 200 | 196 | 770.0 |
| RoboticArm | T | mic | dense_autoencoder | 4 | 800 | 0.454636 | 153 | 143 | 533.6 |
| RoboticArm | T | mic | isolation_forest | 0 | 14 | 0.482222 |  |  | 0.2 |
| RoboticArm | T | mic | isolation_forest | 1 | 14 | 0.491843 |  |  | 0.2 |
| RoboticArm | T | mic | isolation_forest | 2 | 14 | 0.483154 |  |  | 0.2 |
| RoboticArm | T | mic | isolation_forest | 3 | 14 | 0.482892 |  |  | 0.2 |
| RoboticArm | T | mic | isolation_forest | 4 | 14 | 0.488568 |  |  | 0.2 |
| RoboticArm | T | mic | mahalanobis_shrinkage | 0 | 14 | 17.6214 |  |  | 0.0 |
| RoboticArm | T | mic | one_class_svm | 0 | 14 | -0.426968 |  |  | 0.0 |
| RoboticArm | T | mic | statistical_reference | 0 | 14 | 1.35464 |  |  | 0.0 |

Thresholds are descriptive source-normal q90 calibration values, not test results and not a model ranking.

BrushlessMotor carries `brushless_motor_acquisition_time_class_confound_v1`; RoboticArm carries the six-parent validation and eighteen-parent adaptation limitations.
