"""Frozen Phase-3.0 policy constants."""

MODEL_SEEDS = [0, 1, 2, 3, 4]
BOOTSTRAP_SEED = 1995
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_POLICY_VERSION = "phase3.0.1-class-stratified-cluster-v1"
BOOTSTRAP_INFERENCE_SCOPE = "conditional_on_observed_evaluation_cohort"
PRIMARY_BOOTSTRAP_STRATA = ("is_anomaly",)
PRIMARY_REPORTING_SLICES = ("machine", "domain")
SCORE_DIRECTION = "higher_is_more_anomalous"
EXPERIMENT_SCHEMA_VERSION = "phase3.0.1-experiment-v1"
METRIC_POLICY_VERSION = "phase3.0-protocol-b-metrics-v1"
THRESHOLD_POLICY_VERSION = "phase3.0-source-validation-q90-v1"
RESULT_SCHEMA_VERSION = "phase3.0.1-result-v1"
TEST_LOOK_SCHEMA_VERSION = "phase3.0-test-look-v1"
ADAPTATION_POLICY_VERSION = "phase2.2-released-target-normals-v1"

ROBOTIC_ARM_ADAPTATION_SEGMENTS = 27
BRUSHLESS_MOTOR_ADAPTATION_SEGMENTS = 18
ROBOTIC_ARM_VALIDATION_PARENTS = 6
ROBOTIC_ARM_VALIDATION_SEGMENTS = 186
BRUSHLESS_MOTOR_VALIDATION_SEGMENTS = 126
ROBOTIC_ARM_ADAPTATION_PARENTS = 18
ROBOTIC_ARM_BLOCKED_SIBLING_SEGMENTS = 96
ROBOTIC_ARM_ADAPTATION_CLOSURE_SEGMENTS = 123

PHASE31_EVALUATION_STAGE = "validation_only_pretest"
PHASE31_RESULT_SCHEMA_VERSION = "phase3.1-validation-only-result-v1"
PHASE31_PREPROCESSING_POLICY_VERSION = "phase3.1-primary-preprocessing-v1"

BRUSHLESS_MOTOR_ACQUISITION_SESSION_CONFOUND_ID = (
    "brushless_motor_acquisition_time_class_confound_v1"
)
ROBOTIC_ARM_VALIDATION_PARENT_LIMITATION_ID = (
    "robotic_arm_source_validation_six_parents_v1"
)
ROBOTIC_ARM_ADAPTATION_PARENT_LIMITATION_ID = (
    "robotic_arm_released_adaptation_eighteen_parents_v1"
)

ROBOTIC_ARM_VALIDATION_PARENT_IDS = (
    "20240221_12_23_16",
    "20240221_13_46_42",
    "20240221_15_10_26",
    "20240405_14_54_41",
    "20240418_10_41_53",
    "20240418_14_53_33",
)
ROBOTIC_ARM_ADAPTATION_PARENT_IDS = (
    "20240221_15_27_06",
    "20240222_10_16_25",
    "20240222_10_41_10",
    "20240222_10_57_39",
    "20240222_11_05_54",
    "20240222_11_14_08",
    "20240405_11_49_50",
    "20240405_12_15_00",
    "20240405_12_23_24",
    "20240405_12_31_47",
    "20240405_12_53_04",
    "20240405_13_09_52",
    "20240405_13_18_15",
    "20240405_14_27_24",
    "20240416_13_27_14",
    "20240416_13_35_37",
    "20240416_16_45_47",
    "20240416_16_54_10",
)

BRUSHLESS_MOTOR_LIMITATION = (
    "BrushlessMotor has an acquisition-time/class confound in the released cohort: normal "
    "recordings occur earlier and anomalous recordings later, so mechanical-fault effects cannot "
    "be completely separated from acquisition-session/drift effects. Early-vs-late normal "
    "analysis can diagnose normal drift but cannot deconfound fault from session."
)
ROBOTIC_ARM_VALIDATION_LIMITATION = (
    "Model selection uses six independent RoboticArm validation parent recordings; the 186 "
    "segment excerpts are correlated children, not 186 independent units."
)
ROBOTIC_ARM_ADAPTATION_LIMITATION = (
    "The released RoboticArm Regime-T adaptation budget contains 27 target-normal segments from "
    "18 independent physical parent recordings; its 96 blocked siblings come from the same 18 "
    "parents, giving a 123-segment parent closure that is not the training budget."
)
