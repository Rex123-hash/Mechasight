# Phase 3.0.1 — scientific consistency correction

## Status

**COMPLETE — awaiting independent audit.** Starting accepted implementation commit:
`4c0be89203e228fb8093b0e387646c52c3f0cfe5`.

**NO REAL ANOMALY MODEL WAS TRAINED.**

**NO REAL TEST PERFORMANCE WAS INSPECTED.**

The accepted Phase 2.2 membership artifact was read for identity/count verification only and was not
modified. All bootstrap metric tests use synthetic fixtures.

## Verified RoboticArm cohort distinction

Direct recomputation from `reports/phase2_2/protocol_b_few_shot_adaptation_v3.csv`, restricted to
the Regime-T copy of each role, gives:

- Source-normal validation: 186 segments from 6 physical parents.
- Released target-normal adaptation: 27 segments from 18 physical parents.
- Blocked adaptation siblings: 96 segments from the same 18 adaptation parents.
- Adaptation parent closure: 123 segments from 18 parents.
- Validation and adaptation parent sets are disjoint.
- Released-adaptation and blocked-sibling segment sets are disjoint.
- Blocked-sibling parent IDs equal the released-adaptation parent IDs exactly.

Validation parent IDs:

```text
20240221_12_23_16
20240221_13_46_42
20240221_15_10_26
20240405_14_54_41
20240418_10_41_53
20240418_14_53_33
```

Released-adaptation and blocked-sibling parent IDs:

```text
20240221_15_27_06
20240222_10_16_25
20240222_10_41_10
20240222_10_57_39
20240222_11_05_54
20240222_11_14_08
20240405_11_49_50
20240405_12_15_00
20240405_12_23_24
20240405_12_31_47
20240405_12_53_04
20240405_13_09_52
20240405_13_18_15
20240405_14_27_24
20240416_13_27_14
20240416_13_35_37
20240416_16_45_47
20240416_16_54_10
```

The validation limitation governs model/preprocessing selection. The adaptation limitation governs
the independent physical parents represented by the released few-shot training budget. They are
encoded under different machine-readable IDs and are never interchangeable.

## BrushlessMotor limitation

BrushlessMotor carries limitation ID
`brushless_motor_acquisition_time_class_confound_v1`. Normal recordings occur earlier in the
acquisition timeline and anomalous recordings later. Mechanical-fault effects therefore cannot be
completely separated from acquisition-session/drift effects using this released cohort alone. An
early-vs-late normal analysis can diagnose normal drift but cannot deconfound fault from session.
Preprocessing does not solve this limitation.

## Frozen primary bootstrap

Policy ID: `phase3.0.1-class-stratified-cluster-v1`.

The primary interval is a **class-stratified cluster bootstrap conditional on the observed
evaluation cohort**:

- Report/evaluate separate machine and source/target domain slices before bootstrap.
- Within a slice, sample normal clusters with replacement from the normal pool and anomalous
  clusters with replacement from the anomalous pool.
- Preserve the observed number of clusters in each class.
- RoboticArm unit: physical parent recording; retain every child segment, including multiplicity
  when a parent is drawn more than once.
- BrushlessMotor unit: released recording/segment under the audited one-segment-per-recording
  structure.
- Paired comparison: use identical sampled clusters and multiplicities for both models.
- Replicates: 10,000; seed: 1995; confidence: 95%; interval: percentile.
- Operating condition is not an automatic primary stratum.
- Invalid-replicate detection remains a defensive guard.
- This is conditional uncertainty for the observed evaluation cohort, not a population-wide
  uncertainty estimate.

No secondary/sensitivity bootstrap is authorized or implemented in Phase 3.0.1.

## Provenance correction

Future experiment configs and result bundles carry:

- Separate RoboticArm validation/adaptation limitation IDs and exact parent identities/counts.
- The BrushlessMotor acquisition-time/class-confound limitation ID and explicit wording.
- Bootstrap method, class-only strata, reporting slices, cluster unit, inference scope, replicate
  count, seed, confidence level, percentile method, and paired-draw policy.

The corrected experiment schema is `phase3.0.1-experiment-v1` and corrected result schema is
`phase3.0.1-result-v1`. Experiment identity changes when the corrected bootstrap configuration
changes.

## Tests

Command:

```text
python -m pytest -q -p no:cacheprovider
```

Result: **222 passed, 0 failed, 0 skipped, 0 warnings in 13.88 s**.

The prior 210 tests remain passing. Twelve new regression tests verify exact cohort identities,
non-interchangeable 6-vs-18 parent counts, blocked-parent closure, machine-readable limitations,
class-cluster counts, complete sibling retention and duplication, condition-independent primary
draws, pre-bootstrap machine/domain slicing, shared paired draws, frozen production defaults,
conditional interpretation, and BrushlessMotor one-to-one resampling.

## Scope confirmation

- Real anomaly detector training: **NO**.
- Real test metric computation or inspection: **NO**.
- Phase 2.2 membership change: **NO**.
- Supervisor DOCX/PDF touched: **NO**.
- Phase 3.1 work: **NO**.
