# Phase 5.4 incident record — protected raw recording inspection

## Incident and evidence

On 25 October 2026, an ad-hoc Python format inspection selected the first raw-catalog row
before restricting it to the development cohort. The Phase 5.4 report identifies that row as
BrushlessMotor `20240423_17_20_26_segment_0`, a common-evaluation segment assigned to `test`
in both regimes. Its microphone, accelerometer and gyroscope Parquet files were loaded with
`pd.read_parquet` to inspect the raw schema and timestamp representation for the L3 work.

This was a read of the three complete raw sensor files into process memory, including their
signal values. It was not merely a filesystem-metadata query. The reported console output
comprised sensor file paths, data types, row counts/shapes and initial/final timestamps;
signal-value arrays, anomaly labels, prediction scores and performance metrics were not
reported as printed or used.

Evidence available for this record is the original disclosure in
[the Phase 5.4 report, section 6](phase5_4_sensor_integrity_fusion.md#6-protected-access-accounting-for-phase-54),
the closeout request describing the operation, the implementation and frozen-design Git history,
and the recorded development-run manifests. These establish a disclosed raw-data exposure and
the earlier design definitions. The console-output description is retrospective: this record
does not independently authenticate a timestamped command transcript, reconstruct the process's
complete activity, or establish that nothing beyond the reported information was observed.
The protected recording was not reopened to prepare this record.

## Timeline and decisions already fixed

Times below use Asia/Calcutta (UTC+05:30). Commit dates establish recorded chronology, not
experimental authorization.

| Event | Recorded time and evidence |
|---|---|
| V2 design definitions committed | 15 October 2026, 00:03:00; `86706dc1a05dc6a009f8eea0b0b9f7cbc3ac6a25` |
| L1/L2 development completion recorded | 25 October 2026, 19:50:35; `27d4bebb2c3ed64623d69d749a8c2c6d131fb1c9` |
| Ad-hoc raw inspection | Reported on 25 October 2026; exact time and its ordering relative to the implementation commit are not established by the available disclosure |
| L3 implementation committed | 25 October 2026, 20:30:46; `77d9cc5299bbf6c5abe01f9131ed2afda6e80271` |
| Official development falsification run | 25 October 2026, 20:34:03–20:46:02; `falsification_manifest.json` records the UTC timestamps and implementation commit |
| Original incident disclosure and development results committed | 25 October 2026, 20:52:07; `6a98b9e7116222aebf3234897e2dffd3df116cb6` |

The [V2 configuration](../configs/phase5/fusion_design_v2.json) committed on 15 October already
specified all four relevant settings:

- Own-stream timestamp coverage of at least **90%** of nominal duration.
- Equal-value runs qualifying at **20 ms**.
- A flatline trigger at **50%** of a channel's coverage samples.
- Nominal durations of **7.5 s** for RoboticArm and **5.2 s** for BrushlessMotor.

Both the design document and configuration at the starting closeout HEAD are identical to
that commit after LF normalization. The configuration SHA-256 is
`2e33299d3c0347ebdc13392d149b248b8ec175f2bea5b2049e5bb2fa996a759a`.
These definitions predate the inspection by ten calendar days. The historical design's
`PROPOSED — NOT AUTHORIZED FOR EXPERIMENTAL EXECUTION` status is preserved; committing its
definitions did not itself authorize access. The inspection cannot be the source of those
four settings. This chronology does not prove the absence of every possible influence on
subsequent implementation choices.

## Policy deviation and scientific implications

Development access was restricted to training and source-validation recordings and synthetic
copies. V2 also excluded raw common-evaluation reads from the planned protected batch.
Selecting an unfiltered catalog row and calling pandas directly bypassed that development
selection policy. The cohort was therefore not untouched: one protected segment's three raw
streams were exposed outside the approved workflow.

The documented operation was inspection, not model fitting or protected inference. The report
records no labels, detector scores or outcome metrics used by that inspection. There is no
evidence in this record that a model was trained on protected data, that this inspection
produced a protected fusion result, or that it changed the frozen gate settings. Conversely,
the retrospective disclosure alone cannot prove every action of the original process.

The official clean scan and falsification suite use development-selected rows. Their outputs
remain development evidence, rather than protected-cohort results. This incident is separate
from the already-disclosed historical score inspections and Phase 3.4 replay tests. It records
an additional raw-data exposure, without implying a new model-training leakage incident.

The segment remains in the frozen evaluation cohort. No recording is removed, no membership,
model, calibration, official result or design setting is changed, and no ledger entry is added
or retrospectively presented as authorization for this inspection. The ledger remains at 196
rows. Its unchanged state does not erase the raw-data exposure.

## Corrective disclosure and prevention

This incident record expands the original disclosure and corrects its characterization as
metadata-only access. The Phase 5.4 report now identifies the actual access boundary:
`development_segment_rows` selects permitted catalog rows for the development driver, while
`load_raw_stream`, `run_gate` and `run_segment` accept explicit paths without independently
checking split membership. They are not a universal barrier against protected raw reads.

Development format checks and runner calls should obtain paths from the filtered development
rows, with the intended segment's eligibility checked before raw-file access. An arbitrary
catalog row or a manually supplied raw path is not evidence of permitted membership. The
existing cohort-selection test exercises exclusion of test segments; it does not certify a
split check inside every lower-level reader. These are procedural requirements and an accurate
description of the existing checks, not a newly implemented runtime safeguard.

The bounded closeout adds no raw-data enforcement mechanism. Any future protected workflow
needs its own reviewed authorization and access boundary. Independent review should retain
this exposure in the study's provenance and assess its stated limits alongside the existing
claim boundaries.
