# Phase 3.2 First Locked Common-Test Evaluation

Test evaluator commit: `99f6055156c314bbf26a847713d26792979fd050`. Reporting policy: `p311-reporting-5e76ab7dabd5689c7b0ced68`.

The 28 pre-registered configurations reproduced exactly before the test was opened. The ledger received exactly 28 entries atomically before the first model score.

## Table A — RoboticArm primary baselines

| Regime | Model | AUROC (95% CI) | AP (95% CI) | pAUROC@0.10 (95% CI) | FPR | Recall | Precision | F1 |
|---|---|---|---|---|---:|---:|---:|---:|
| S | statistical_reference | 0.8522 [0.6523, 0.9527] | 0.8091 [0.6171, 0.9425] | 0.6224 [0.4932, 0.8472] | 0.0043 | 0.0431 | 0.9091 | 0.0823 |
| S | isolation_forest | 0.9137 [0.7736, 0.9759] | 0.8654 [0.7181, 0.9682] | 0.7126 [0.5108, 0.9052] | 0.0000 | 0.0000 | UNDEFINED | UNDEFINED |
| S | one_class_svm | 0.8985 [0.7831, 0.9589] | 0.8614 [0.7229, 0.9554] | 0.7019 [0.4950, 0.8798] | 0.0556 | 0.5560 | 0.9085 | 0.6898 |
| T | statistical_reference | 0.8457 [0.6419, 0.9508] | 0.8038 [0.6066, 0.9419] | 0.6185 [0.4930, 0.8440] | 0.0043 | 0.0431 | 0.9091 | 0.0823 |
| T | isolation_forest | 0.9038 [0.7558, 0.9713] | 0.8528 [0.6988, 0.9631] | 0.7001 [0.5000, 0.8935] | 0.0017 | 0.0000 | UNDEFINED | UNDEFINED |
| T | one_class_svm | 0.8932 [0.7876, 0.9531] | 0.8933 [0.7666, 0.9574] | 0.7383 [0.5875, 0.8663] | 0.0256 | 0.4440 | 0.9450 | 0.6041 |

## Table B — BrushlessMotor primary baselines

| Regime | Model | AUROC (95% CI) | AP (95% CI) | pAUROC@0.10 (95% CI) | FPR | Recall | Precision | F1 |
|---|---|---|---|---|---:|---:|---:|---:|
| S | statistical_reference | 0.5988 [0.5337, 0.6614] | 0.5843 [0.5302, 0.6469] | 0.5085 [0.4906, 0.5457] | 0.4423 | 0.6474 | 0.5941 | 0.6196 |
| S | isolation_forest | 0.6549 [0.5917, 0.7149] | 0.6340 [0.5819, 0.6920] | 0.5458 [0.5224, 0.5756] | 0.5256 | 0.8423 | 0.6157 | 0.7113 |
| S | one_class_svm | 0.7054 [0.6429, 0.7642] | 0.6099 [0.5554, 0.6785] | 0.5010 [0.4863, 0.5358] | 0.5192 | 0.9551 | 0.6478 | 0.7720 |
| T | statistical_reference | 0.5933 [0.5278, 0.6558] | 0.6120 [0.5569, 0.6704] | 0.5436 [0.5174, 0.5787] | 0.4359 | 0.5705 | 0.5669 | 0.5687 |
| T | isolation_forest | 0.5961 [0.5267, 0.6637] | 0.5089 [0.4709, 0.5638] | 0.4812 [0.4755, 0.4929] | 0.4885 | 0.7808 | 0.6151 | 0.6878 |
| T | one_class_svm | 0.8648 [0.8208, 0.9046] | 0.8362 [0.7840, 0.8867] | 0.5969 [0.5624, 0.6938] | 0.4679 | 0.9487 | 0.6697 | 0.7851 |

## Domain-slice ranking metrics

| machine | regime | model_family | domain | sample_count | status | auroc | average_precision | standardized_pauroc_fpr_0_10 |
|---|---|---|---|---|---|---|---|---|
| RoboticArm | S | statistical_reference | source | 330 | defined | 0.8107 | 0.6486 | 0.5610 |
| RoboticArm | S | statistical_reference | target | 136 | defined | 0.5677 | 0.8213 | 0.4737 |
| RoboticArm | S | isolation_forest | source | 330 | defined | 0.9078 | 0.8254 | 0.7050 |
| RoboticArm | S | isolation_forest | target | 136 | defined | 0.6747 | 0.8711 | 0.4805 |
| RoboticArm | S | one_class_svm | source | 330 | defined | 0.8781 | 0.8213 | 0.7406 |
| RoboticArm | S | one_class_svm | target | 136 | defined | 0.5793 | 0.8256 | 0.4737 |
| RoboticArm | T | statistical_reference | source | 330 | defined | 0.8021 | 0.6398 | 0.5580 |
| RoboticArm | T | statistical_reference | target | 136 | defined | 0.5776 | 0.8247 | 0.4737 |
| RoboticArm | T | isolation_forest | source | 330 | defined | 0.8926 | 0.8058 | 0.6920 |
| RoboticArm | T | isolation_forest | target | 136 | defined | 0.6451 | 0.8572 | 0.4760 |
| RoboticArm | T | one_class_svm | source | 330 | defined | 0.8481 | 0.7729 | 0.6949 |
| RoboticArm | T | one_class_svm | target | 136 | defined | 0.7905 | 0.9519 | 0.6030 |
| BrushlessMotor | S | statistical_reference | source | 156 | defined | 0.7240 | 0.7540 | 0.6194 |
| BrushlessMotor | S | statistical_reference | target | 156 | defined | 0.4752 | 0.5388 | 0.5360 |
| BrushlessMotor | S | isolation_forest | source | 156 | defined | 0.9237 | 0.9228 | 0.7877 |
| BrushlessMotor | S | isolation_forest | target | 156 | defined | 0.4591 | 0.5067 | 0.5137 |
| BrushlessMotor | S | one_class_svm | source | 156 | defined | 0.9775 | 0.9820 | 0.9472 |
| BrushlessMotor | S | one_class_svm | target | 156 | defined | 0.5828 | 0.5322 | 0.5000 |
| BrushlessMotor | T | statistical_reference | source | 156 | defined | 0.7130 | 0.7420 | 0.6169 |
| BrushlessMotor | T | statistical_reference | target | 156 | defined | 0.4599 | 0.5585 | 0.5647 |
| BrushlessMotor | T | isolation_forest | source | 156 | defined | 0.8991 | 0.8939 | 0.7230 |
| BrushlessMotor | T | isolation_forest | target | 156 | defined | 0.3310 | 0.3973 | 0.4806 |
| BrushlessMotor | T | one_class_svm | source | 156 | defined | 0.9814 | 0.9841 | 0.9462 |
| BrushlessMotor | T | one_class_svm | target | 156 | defined | 0.8034 | 0.7649 | 0.5867 |

## Table D — Healthy-target false positives

| machine | regime | model_family | healthy_target_sample_count | false_positive | false_positive_rate |
|---|---|---|---|---|---|
| RoboticArm | S | statistical_reference | 20.0000 | 1.0000 | 0.0500 |
| RoboticArm | S | isolation_forest | 20.0000 | 0.0000 | 0.0000 |
| RoboticArm | S | one_class_svm | 20.0000 | 12.0000 | 0.6000 |
| RoboticArm | T | statistical_reference | 20.0000 | 1.0000 | 0.0500 |
| RoboticArm | T | isolation_forest | 20.0000 | 0.4000 | 0.0200 |
| RoboticArm | T | one_class_svm | 20.0000 | 6.0000 | 0.3000 |
| BrushlessMotor | S | statistical_reference | 78.0000 | 63.0000 | 0.8077 |
| BrushlessMotor | S | isolation_forest | 78.0000 | 76.6000 | 0.9821 |
| BrushlessMotor | S | one_class_svm | 78.0000 | 78.0000 | 1.0000 |
| BrushlessMotor | T | statistical_reference | 78.0000 | 62.0000 | 0.7949 |
| BrushlessMotor | T | isolation_forest | 78.0000 | 70.8000 | 0.9077 |
| BrushlessMotor | T | one_class_svm | 78.0000 | 70.0000 | 0.8974 |

## Table C — S vs T deltas (T minus S)

| Machine | Model | ΔAUROC (95% CI) | ΔAP (95% CI) | ΔpAUROC@0.10 (95% CI) |
|---|---|---|---|---|
| RoboticArm | statistical_reference | -0.0064 [-0.0143, 0.0004] | -0.0053 [-0.0170, 0.0031] | -0.0039 [-0.0154, 0.0112] |
| RoboticArm | isolation_forest | -0.0099 [-0.0226, -0.0025] | -0.0126 [-0.0258, -0.0031] | -0.0124 [-0.0253, 0.0004] |
| RoboticArm | one_class_svm | -0.0053 [-0.0252, 0.0321] | 0.0319 [-0.0217, 0.0972] | 0.0364 [-0.0388, 0.1674] |
| BrushlessMotor | statistical_reference | -0.0055 [-0.0151, 0.0042] | 0.0277 [0.0081, 0.0485] | 0.0351 [0.0155, 0.0540] |
| BrushlessMotor | isolation_forest | -0.0588 [-0.0813, -0.0382] | -0.1251 [-0.1534, -0.0905] | -0.0646 [-0.0911, -0.0428] |
| BrushlessMotor | one_class_svm | 0.1594 [0.1259, 0.1964] | 0.2262 [0.1937, 0.2534] | 0.0959 [0.0704, 0.1626] |

## Table E — Paired model-family differences (first minus second)

| Machine | Regime | Comparison | ΔAUROC (95% CI) | ΔAP (95% CI) | ΔpAUROC@0.10 (95% CI) |
|---|---|---|---|---|---|
| RoboticArm | S | statistical_reference_minus_isolation_forest | -0.0616 [-0.1365, -0.0166] | -0.0563 [-0.1352, 0.0021] | -0.0901 [-0.1452, -0.0025] |
| RoboticArm | S | statistical_reference_minus_one_class_svm | -0.0463 [-0.1842, 0.0336] | -0.0523 [-0.1708, 0.0258] | -0.0795 [-0.1720, 0.0361] |
| RoboticArm | S | isolation_forest_minus_one_class_svm | 0.0152 [-0.0583, 0.0726] | 0.0040 [-0.0637, 0.0717] | 0.0107 [-0.0653, 0.1085] |
| RoboticArm | T | statistical_reference_minus_isolation_forest | -0.0581 [-0.1318, -0.0144] | -0.0490 [-0.1303, 0.0087] | -0.0816 [-0.1400, 0.0033] |
| RoboticArm | T | statistical_reference_minus_one_class_svm | -0.0475 [-0.1866, 0.0360] | -0.0895 [-0.2021, 0.0176] | -0.1198 [-0.1827, 0.0363] |
| RoboticArm | T | isolation_forest_minus_one_class_svm | 0.0106 [-0.0709, 0.0704] | -0.0405 [-0.1155, 0.0484] | -0.0382 [-0.1429, 0.0886] |
| BrushlessMotor | S | statistical_reference_minus_isolation_forest | -0.0561 [-0.0850, -0.0267] | -0.0497 [-0.0932, -0.0028] | -0.0373 [-0.0687, 0.0036] |
| BrushlessMotor | S | statistical_reference_minus_one_class_svm | -0.1067 [-0.1328, -0.0827] | -0.0256 [-0.0529, -0.0005] | 0.0075 [-0.0030, 0.0220] |
| BrushlessMotor | S | isolation_forest_minus_one_class_svm | -0.0505 [-0.0820, -0.0191] | 0.0241 [-0.0309, 0.0714] | 0.0448 [0.0049, 0.0763] |
| BrushlessMotor | T | statistical_reference_minus_isolation_forest | -0.0028 [-0.0401, 0.0369] | 0.1031 [0.0519, 0.1461] | 0.0624 [0.0328, 0.0981] |
| BrushlessMotor | T | statistical_reference_minus_one_class_svm | -0.2715 [-0.3175, -0.2280] | -0.2242 [-0.2634, -0.1877] | -0.0533 [-0.1291, -0.0312] |
| BrushlessMotor | T | isolation_forest_minus_one_class_svm | -0.2687 [-0.3277, -0.2119] | -0.3273 [-0.3766, -0.2664] | -0.1157 [-0.2114, -0.0787] |

## Isolation-Forest individual-seed appendix

| machine | regime | metric | seed_0 | seed_1 | seed_2 | seed_3 | seed_4 | mean | population_std | minimum | maximum |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RoboticArm | S | auroc | 0.9116 | 0.9239 | 0.9096 | 0.9167 | 0.9069 | 0.9137 | 0.0060 | 0.9069 | 0.9239 |
| RoboticArm | S | average_precision | 0.8633 | 0.8893 | 0.8566 | 0.8648 | 0.8531 | 0.8654 | 0.0127 | 0.8531 | 0.8893 |
| RoboticArm | S | standardized_pauroc_fpr_0_10 | 0.7178 | 0.7338 | 0.7089 | 0.7061 | 0.6962 | 0.7126 | 0.0127 | 0.6962 | 0.7338 |
| RoboticArm | S | true_negative | 234.0000 | 234.0000 | 234.0000 | 234.0000 | 234.0000 | 234.0000 | 0.0000 | 234.0000 | 234.0000 |
| RoboticArm | S | false_positive | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| RoboticArm | S | false_negative | 232.0000 | 232.0000 | 232.0000 | 232.0000 | 232.0000 | 232.0000 | 0.0000 | 232.0000 | 232.0000 |
| RoboticArm | S | true_positive | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| RoboticArm | S | false_positive_rate | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| RoboticArm | S | recall | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| RoboticArm | S | precision | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED |
| RoboticArm | S | f1 | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED |
| RoboticArm | T | auroc | 0.9048 | 0.9061 | 0.8985 | 0.9037 | 0.9059 | 0.9038 | 0.0028 | 0.8985 | 0.9061 |
| RoboticArm | T | average_precision | 0.8610 | 0.8587 | 0.8460 | 0.8496 | 0.8488 | 0.8528 | 0.0059 | 0.8460 | 0.8610 |
| RoboticArm | T | standardized_pauroc_fpr_0_10 | 0.7087 | 0.7066 | 0.6859 | 0.6910 | 0.7084 | 0.7001 | 0.0097 | 0.6859 | 0.7087 |
| RoboticArm | T | true_negative | 232.0000 | 234.0000 | 234.0000 | 234.0000 | 234.0000 | 233.6000 | 0.8000 | 232.0000 | 234.0000 |
| RoboticArm | T | false_positive | 2.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.4000 | 0.8000 | 0.0000 | 2.0000 |
| RoboticArm | T | false_negative | 232.0000 | 232.0000 | 232.0000 | 232.0000 | 232.0000 | 232.0000 | 0.0000 | 232.0000 | 232.0000 |
| RoboticArm | T | true_positive | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| RoboticArm | T | false_positive_rate | 0.0085 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0017 | 0.0034 | 0.0000 | 0.0085 |
| RoboticArm | T | recall | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| RoboticArm | T | precision | 0.0000 | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED |
| RoboticArm | T | f1 | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED |
| BrushlessMotor | S | auroc | 0.6254 | 0.6620 | 0.6865 | 0.6450 | 0.6556 | 0.6549 | 0.0201 | 0.6254 | 0.6865 |
| BrushlessMotor | S | average_precision | 0.5934 | 0.6497 | 0.6870 | 0.6126 | 0.6275 | 0.6340 | 0.0323 | 0.5934 | 0.6870 |
| BrushlessMotor | S | standardized_pauroc_fpr_0_10 | 0.5209 | 0.5613 | 0.5749 | 0.5294 | 0.5426 | 0.5458 | 0.0199 | 0.5209 | 0.5749 |
| BrushlessMotor | S | true_negative | 75.0000 | 74.0000 | 74.0000 | 76.0000 | 71.0000 | 74.0000 | 1.6733 | 71.0000 | 76.0000 |
| BrushlessMotor | S | false_positive | 81.0000 | 82.0000 | 82.0000 | 80.0000 | 85.0000 | 82.0000 | 1.6733 | 80.0000 | 85.0000 |
| BrushlessMotor | S | false_negative | 27.0000 | 23.0000 | 27.0000 | 27.0000 | 19.0000 | 24.6000 | 3.2000 | 19.0000 | 27.0000 |
| BrushlessMotor | S | true_positive | 129.0000 | 133.0000 | 129.0000 | 129.0000 | 137.0000 | 131.4000 | 3.2000 | 129.0000 | 137.0000 |
| BrushlessMotor | S | false_positive_rate | 0.5192 | 0.5256 | 0.5256 | 0.5128 | 0.5449 | 0.5256 | 0.0107 | 0.5128 | 0.5449 |
| BrushlessMotor | S | recall | 0.8269 | 0.8526 | 0.8269 | 0.8269 | 0.8782 | 0.8423 | 0.0205 | 0.8269 | 0.8782 |
| BrushlessMotor | S | precision | 0.6143 | 0.6186 | 0.6114 | 0.6172 | 0.6171 | 0.6157 | 0.0026 | 0.6114 | 0.6186 |
| BrushlessMotor | S | f1 | 0.7049 | 0.7170 | 0.7030 | 0.7068 | 0.7249 | 0.7113 | 0.0083 | 0.7030 | 0.7249 |
| BrushlessMotor | T | auroc | 0.5912 | 0.6034 | 0.6072 | 0.5857 | 0.5930 | 0.5961 | 0.0080 | 0.5857 | 0.6072 |
| BrushlessMotor | T | average_precision | 0.4999 | 0.5082 | 0.5206 | 0.5078 | 0.5080 | 0.5089 | 0.0066 | 0.4999 | 0.5206 |
| BrushlessMotor | T | standardized_pauroc_fpr_0_10 | 0.4761 | 0.4791 | 0.4838 | 0.4873 | 0.4796 | 0.4812 | 0.0039 | 0.4761 | 0.4873 |
| BrushlessMotor | T | true_negative | 84.0000 | 76.0000 | 84.0000 | 79.0000 | 76.0000 | 79.8000 | 3.6000 | 76.0000 | 84.0000 |
| BrushlessMotor | T | false_positive | 72.0000 | 80.0000 | 72.0000 | 77.0000 | 80.0000 | 76.2000 | 3.6000 | 72.0000 | 80.0000 |
| BrushlessMotor | T | false_negative | 42.0000 | 26.0000 | 30.0000 | 39.0000 | 34.0000 | 34.2000 | 5.8103 | 26.0000 | 42.0000 |
| BrushlessMotor | T | true_positive | 114.0000 | 130.0000 | 126.0000 | 117.0000 | 122.0000 | 121.8000 | 5.8103 | 114.0000 | 130.0000 |
| BrushlessMotor | T | false_positive_rate | 0.4615 | 0.5128 | 0.4615 | 0.4936 | 0.5128 | 0.4885 | 0.0231 | 0.4615 | 0.5128 |
| BrushlessMotor | T | recall | 0.7308 | 0.8333 | 0.8077 | 0.7500 | 0.7821 | 0.7808 | 0.0372 | 0.7308 | 0.8333 |
| BrushlessMotor | T | precision | 0.6129 | 0.6190 | 0.6364 | 0.6031 | 0.6040 | 0.6151 | 0.0122 | 0.6031 | 0.6364 |
| BrushlessMotor | T | f1 | 0.6667 | 0.7104 | 0.7119 | 0.6686 | 0.6816 | 0.6878 | 0.0197 | 0.6667 | 0.7119 |

Isolation-Forest rows are arithmetic five-seed family summaries. Full individual-seed, domain, healthy-target, bootstrap, S-vs-T, and paired-comparison tables accompany this report.

No pooled cross-machine result or universal winner is reported.

BrushlessMotor limitation `brushless_motor_acquisition_time_class_confound_v1`: fault/class and acquisition session/time cannot be fully separated, so these results are not pure isolated mechanical-fault evidence.

RoboticArm uncertainty clusters by physical parent recording. Source-normal validation was 186 segments from six parents; Regime-T adaptation was 27 released segments from 18 parents, with 96 siblings blocked.
