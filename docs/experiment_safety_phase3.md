# Phase 3 experiment safety and evaluation contract

**Status:** Phase 3.0.1 scientific-consistency correction implemented; no real anomaly model trained
and no real test performance inspected. Phase 2.2 is independently accepted / GREEN and remains
the frozen data foundation.

## Official data boundaries

The canonical preprocessing manifest remains an audit/provenance object and may retain
`anomaly_label`. It is not passed directly to an estimator. `src/experiment/data.py` binds a
preprocessing config to the accepted v3 split manifest, validates the complete authorized cohort,
and returns an `EstimatorTrainingData` object whose `X` contains only explicitly named feature
columns. It has no label or target field.

Protocol-B fitting semantics are frozen:

- Regime S: source-domain normal `source_training` rows only.
- Regime T: the same source normals plus the released `target_adaptation` allocation—27
  RoboticArm segments or 18 BrushlessMotor segments.
- The 96 RoboticArm `adaptation_sibling_blocked` segments are consumed by nothing.
- Target adaptation remains training data. It is not validation data and cannot calibrate the
  threshold.

The public train-fitted normalization entry point performs this cohort validation before calling
the low-level numerical standardizer. A caller-provided `split="train"` string is not sufficient.

## Scores and thresholds

Every `ScoreBatch` carries experiment, machine, protocol, regime, model, seed, preprocessing,
split, segment, parent, cohort-role, domain, score-direction, data and config provenance.
The only permitted direction is `higher_is_more_anomalous`; a raw score is never called a
probability.

Protocol-B threshold selection is the empirical 90th percentile (`higher`) of a pure
`source_validation`, source-domain, normal-labelled score batch. Test, target-adaptation,
source-training, anomalous, mixed-role and stale-experiment batches are rejected before the
mathematical quantile helper is called.

## Metrics

Protocol-B metrics use scikit-learn 1.9.0:

- `auroc`: standard binary ROC AUC.
- `average_precision`: average precision (AP). MechaSight uses this operational definition for its
  reported AUPRC; the field is named explicitly to avoid mathematical ambiguity.
- `standardized_pauroc_fpr_0_10`: scikit-learn's standardized partial ROC AUC with `max_fpr=0.10`.
  It is not the raw geometric partial area and the two are not interchangeable.
- Confusion counts, FPR, recall, precision and F1 use only the frozen validation-selected threshold
  and the strict decision `score > threshold`.

Window scores aggregate to a segment score by median and retain unique contributing window IDs and
their count. Mixed segments, mixed parents, duplicate/missing window IDs and non-finite scores are
rejected.

## Seeds and inference

Model seeds are exactly `[0, 1, 2, 3, 4]`. The bootstrap/split seed remains `1995`; these concepts
are not interchangeable. Deterministic models may return identical results across seeds.

The primary inference engine implements a **class-stratified cluster bootstrap conditional on the
observed evaluation cohort**, with 10,000 requested replicates, seed 1995 and 95% percentile
intervals. Evaluation is sliced by machine and source/target domain before resampling. Within each
slice, normal clusters are sampled with replacement from the normal pool and anomalous clusters
from the anomalous pool, preserving the observed number of clusters in each class. Operating
condition is not an automatic primary stratum.

For RoboticArm, the resampling unit is the physical parent recording and every child segment of a
sampled parent is retained; duplicate parent draws duplicate the complete child set. For
BrushlessMotor, the unit is the released recording/segment under its audited one-segment-per-
recording structure. Paired methods share the exact parent/recording draws and multiplicities.
Invalid one-class replicates remain detected and reported defensively. The interval is conditional
on the observed evaluation cohort and must not be described as population-wide uncertainty. No
secondary bootstrap is currently authorized or implemented.

## Experiment and result provenance

`ExperimentConfig` records the complete scientific configuration, code/data identities and
environment provenance. Its deterministic `p3-<hash>` identity excludes run timestamp but changes
when a scientifically relevant setting changes.

`ResultStore` writes complete bundles through a temporary sibling directory and promotes them by
rename. Existing experiment directories are never overwritten. Phase 3.0 creates bundles only in
test temporary directories, never under `reports/experiment_results/`.

Every future real test evaluation must pass through `EvaluationGate.evaluate_test`, which appends a
durable JSONL test-look record before metric computation. Validation evaluation does not write a
test-look record. This is an engineering guardrail for the official pathway, not a claim that raw
Python functions cannot be called directly.

## Preprocessing selection remains open

Protocol A is unchanged. Protocol B window length, stride/overlap and normalization are still
research decisions. Candidate settings must be declared and selected using source validation only.
**Test performance must never decide which preprocessing configuration becomes primary.** The
current 100 ms / 100 ms / `none` setting is a Phase-2 smoke reference, not an optimality claim.

## Mandatory limitations

- BrushlessMotor has an acquisition-time/class confound in the released cohort. Normal recordings
  occur earlier and anomalous recordings later, so mechanical-fault effects cannot be completely
  separated from acquisition-session/drift effects using this cohort alone. An early-vs-late
  normal analysis may diagnose normal drift but cannot deconfound fault from session; preprocessing
  does not solve the confound. Every BM result bundle carries the machine-readable limitation ID
  `brushless_motor_acquisition_time_class_confound_v1`.
- RoboticArm model/preprocessing selection has 186 source-normal validation segments from six
  independent physical parents. This validation limitation is distinct from target adaptation.
- RoboticArm Regime-T target adaptation contains 27 released target-normal segments from 18
  independent physical parents. The 96 blocked siblings are from those same 18 parents, yielding a
  123-segment parent closure that is not the training budget. Every RA result bundle carries separate
  validation and adaptation limitation IDs so the 6-parent and 18-parent cohorts cannot be conflated.
