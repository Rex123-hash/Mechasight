# Multi-Sensor Fusion Development — Sensor-Integrity-Aware Fusion (L3) and Development Falsification (Phase 5.4)

<!-- Legacy roadmap name: Phase 5.4 -->

**Scope.** Development only. Inputs are raw training and source-validation recordings plus synthetic
corrupted copies of them. Not done: protected evaluation, Phase 5.5, new protected inference, new
protected metrics, and ledger appends. Not changed: the 24 L1 fits, the 208 L2 calibrations, and
design V1/V2.

This scope describes the official development runs. A separate out-of-policy raw inspection
is disclosed in section 6 and the linked incident record.

## 1. What was implemented

| Component | File | Frozen source |
|---|---|---|
| Gates G1, G2, G3; type-D absence | `src/fusion/l3.py` (`load_raw_stream`, `g3_timing`, `flatline_profile`, `g1_g2_values`, `run_gate`) | `integrity_gate` (`phase5-integrity-gate-v2`) |
| Surviving-modality processing | `l3.process_modalities` | same calls, in the same order, as `pipeline._iter_processed_windows` and the spectral-cache builder |
| Frozen Phase-4 scoring | `l3.DetectorBank` (78 detectors per machine, hash-verified via `ModelStore.load`) | `phase4.score_one_experiment` / `verify_bundle_reproduces_validation` |
| Fallback decision + flag | `l3.decide_unit`, `l3.run_segment` | `arms.L3_sensor_integrity_fallback.decision_policy` |
| Cohort guard | `l3.development_segment_rows` | split manifest v3: Regime-T split must be train/validation, and no regime may say test |
| Clean scan, scenarios, corruption, assertions | `src/fusion/l3_falsification.py` | `development_falsification` |
| Driver | `scripts/run_phase5_l3.py` (`clean-scan`, `falsify`) | — |

### Order of operations (as frozen)

1. Load each modality from its raw file. A missing, unreadable or ill-formed file is type D.
2. Apply G3 to the **stored** row order of each stream. The frozen `read_sensor_parquet` sorts by
   timestamp: it would silently repair a swapped pair (C1), and it accepts a NaN timestamp at the
   tail (C3). G3 therefore inspects the file before the frozen reader runs. Any C or D failure
   gives `ABSTAIN_ALIGNMENT_FAILURE`, with every unit row retained and flagged.
3. Run the frozen value-independent startup transforms: the BrushlessMotor initial accelerometer
   drop and the 0.035 s gyro warm-up.
4. Compute the frozen `common_coverage` over all three streams (timestamps only).
5. Apply G1 and G2 to each modality's samples in [start, end).
6. Process **surviving modalities only**. The shared window grid comes from the frozen
   `iter_aligned_windows`, which slices but never reads values. Each survivor then gets the frozen
   `_normalize('none')` → time and frequency features → segment median, plus the
   `log_power_spectrum_input` → float32 path. An invalid modality's values are never processed.
7. Decide per unit:

   | Surviving modalities | Score | Threshold |
   |---|---|---|
   | 3 | L2(all three) | its q90 |
   | 2 | L2 robust-z mean of the pair | the pair's q90 |
   | 1 | that modality's **raw** Phase-4 score | its own raw Phase-4 q90 (never robust-z against a raw threshold) |
   | 0 | `ABSTAIN_SENSOR_INVALID` | — |

   The decision is anomaly when score > threshold, as in the frozen metrics.

`integrity_flag` is true whenever any modality is not `VALID`, independent of the decision. A gated
segment is never reported as a plain healthy decision.

**Check-placement feasibility.** No frozen preprocessing function was changed. The S and T
preprocessing configs differ only in `regime` (asserted in code), so one processing pass serves the
detectors of both regimes.

## 2. Implementation clarifications

These were fixed before the official runs and are recorded in `falsification_manifest.json`.
None is a tuned value.

| Item | Clarification |
|---|---|
| G2 duration comparison | `duration >= 0.020 - 1e-9` s. The tolerance absorbs float64 representation error only: 0.030 − 0.010 = 0.0199999… The shortest sample period is 62.5 µs. |
| G3 stored order | G3 runs on the file's own row order before the frozen sorting reader. See section 5, Finding B. |
| Coverage for G1/G2 | Half-open [start, end), as the design states. |
| One processing pass | The S and T preprocessing configs differ only in `regime`; asserted by `l3.load_regime_configs`. |
| AE segment score | Median of float64 window scores. Equals `aggregate_window_scores_by_segment` (E0). |
| Segment selection | Lexicographic sort of segment ids, then indices `round(k (n−1)/11)`. n is 186 (RoboticArm) and 126 (BrushlessMotor). No index falls on a .5 tie. |
| Corruption constants | `l3_falsification.CORRUPTION_PARAMETERS`. Partial regions are centred in the coverage. Dead and dropout value is 0.0. B5 plateau is the signed clean-training extreme. N4 uses the pooled clean-training SD. Noise power is the segment variance of the channel. N8 adds +10 SD every 0.25 s. N9 is a 113 Hz tone at 1 SD. Noise seeds are derived from sha256 of the run. The S1–S4 full scale is ±15.988832 g (acc) and the clean-training max abs value (gyro, mic). |
| "Clean training" | Every segment whose Regime-T split is `train`: Regime-S source training plus the 27 / 18 released target normals. |

## 3. Clean scan (U1 acceptance)

`reports/phase5_development/l3/clean_scan/`, produced at commit `77d9cc5` from a clean tracked tree.

| Machine | Training segments | Source-validation segments | Segment-channels |
|---|---|---|---|
| RoboticArm | 1,555 | 186 | 12,187 |
| BrushlessMotor | 1,155 | 126 | 8,967 |

Results:
- **0 G1, G2 or G3 triggers in 21,154 segment-channels**, so the clean scan is accepted.
- The maximum flatline fraction is **0.0 for every machine × sensor × channel**. No sensor is
  LOW-MARGIN (threshold > 0.25).
- Longest equal-value run anywhere, in seconds:

  | Machine | acc | gyro | mic |
  |---|---|---|---|
  | RoboticArm | 0.00085 | 0.00411 | 0.0 |
  | BrushlessMotor | 0.00028 | 0.00084 | 0.0 |

  The microphones never repeat a stored value. Every run is at least 4.9× shorter than the 20 ms
  trigger.
- Shortest own-stream duration: RoboticArm 7.4999 s against the 6.75 s requirement; BrushlessMotor
  5.1989 s against 4.68 s.

## 4. End-to-end falsification suite

`reports/phase5_development/l3/falsification/`, produced at commit `77d9cc5` on an official clean tree.

The suite ran 12 segments × 83 scenarios × 2 machines = **1,992 runs, all of them**, in 714 s. It
produced 51,792 unit decisions (26 units per run) and 108,576 frozen-detector scores. The corrupted
copies (2.0 GB, git-ignored) are under `data/interim/phase5/l3_falsification/official/corrupted`;
every file's sha256 is in `runs.csv`.

**Scenario diversity.** `A1|mic` and `A2|mic` apply equivalent corruption: A1 sets every
microphone channel to NaN inside the coverage, while A2 sets the first channel to NaN, and the
microphone has only one channel. Consequently, 24 runs (one A2 microphone run for each of
12 segments on two machines) duplicate the corresponding A1 microphone corruption. The saved
`runs.csv` records identical corrupted-file hashes for all 24 pairs. The full 1,992-run matrix
was executed, but these are not 1,992 distinct corruptions. This is a test-diversity limitation;
the frozen registry, official counts and saved results are retained unchanged.

| Criterion | Result | Evidence |
|---|---|---|
| E0 clean reproduction | **PASS** | 1,872 detector-segment comparisons (24 segments × 78); max abs diff 8.9e-16 (≤ 1e-9) |
| E1 gate verdicts | **PASS** | 1,704 runs with an expectation; 0 mismatches; every scenario code 100 % |
| E2 surviving-modality equality (B2) | **PASS** | 648 runs (A1–A3, B1–B5, M1), 31,824 comparisons; max abs diff **0.0** |
| E3 subset threshold (B3) | **PASS** | 26,832 all-three, 13,104 pair and 1,872 single decisions; 0 violations; the single-survivor raw decision equals the robust-z decision in 1,872 / 1,872 |
| E4 abstention | **PASS** | 384 runs (C1–C4, D1, M2); all 9,984 unit rows retained, flagged, with no score and no decision |
| E5 identity on clean / N | **PASS** | 17,472 unit decisions equal an independently computed L2(all three); clean vs persisted L2 max abs diff 4.4e-16 |
| E6 exceptions | **PASS** | 0 |

A closeout regression test, `test_l3_pair_fallback_matches_l2_pair_scores`, separately checks
all three surviving pairs with synthetic component scores and the persisted development-only
L2 calibrations. It compares the L3 fallback with L2 pair fusion and checks the pair threshold,
calibration identity and decision. This focused check does not rerun the official suite or
change its recorded E0–E6 results.

Bridging conditions:
- B1 holds (E1 on A/B).
- B2 holds (E2).
- B3 holds (E3).
- B4 holds: C and D abstain.
- **B5 is an assumption:** the surviving physical sensors are unaffected by the other failure.
  Each corruption alters only its target file, so the suite cannot test B5.

### Descriptive behaviour (not pass/fail)

Unit alarm rates by scenario (anomaly = score > threshold), on normal source-validation segments:

| Scenario | Unit alarm rate | Reading |
|---|---|---|
| Clean | 0.093 | ≈ the q90 operating point |
| A1 / B1 (one modality failed, gated → pair) | 0.087 | the fallback keeps the validation operating point |
| M1 (two failed → raw single Phase-4) | 0.076 | the same |
| B4 (40 % flatline, below trigger, valid) | 0.673 | partial flatlines are not gated; the detectors see them as anomalous |
| N1–N3 gain ×0.5 / ×2 / ×3 | 0.647 / 0.716 / 0.789 | undetected by the gate by design; the detectors alarm |
| N4 offset +3 SD | 0.732 | |
| N5–N7 noise at 20 / 10 / 0 dB | 0.223 / 0.527 / 0.755 | |
| N8 impulses / N9 tone | 0.534 / 0.568 | |

The negative controls pass E1 (the gate keeps them valid), and the L3 decision equals L2 (E5).
L3 does not try to separate sensor degradation from machine faults. These perturbations surface
as anomaly alarms, not as integrity flags.

**Saturation, S1–S4** (gain ×1.5 / 2 / 3 / 5, then clip at full scale): **0 of 288 runs gated.**
The maximum flatline fraction is 0.0, because clipped excursions never stay at the rail for 20 ms
on these normal segments. The known suppression envelope (≥ 50 % of a channel at the rail) was not
reached at any tested gain. The saturated modality stays in the fusion, and the unit alarm rate
rises with gain:
- acc: 0.76 → 0.84;
- gyro: 0.79 → 0.86;
- mic: 0.44 → 0.87.

**Ungated demonstration** (`ungated_demonstration.json`; in-sample on the validation cohort that
set the thresholds, so illustrative only):
- Each of the 52 units was paired with each modality replaced by its dead-input score, giving 156
  cases.
- The dead-input score is identical across the 12 segments for every detector (range 0.0).
- Ungated L2(all three) FPR:
  - **≥ 0.5 in 118 of 156 cases** (alarm flood);
  - **below the 0.097 operating point in 9 of 156 cases**, all RoboticArm dense-AE units with a
    dead **microphone**. There the dead input's robust z is as low as −4.18, and the ungated FPR
    falls to 0.000 in 6 of them. That is silent suppression: a dead sensor would hide alarms.
- L3 replaces the dead modality by the surviving pair, with an FPR of 0.095–0.097 in all 156
  cases, and raises the integrity flag.

## 5. Independent-audit findings (external reviewer, YELLOW, checkpoint `27d4beb`)

**Finding A (major): incomplete scientific-unit binding in the L1 completion verifier.**
- *Root cause.* `completed_l1_bundles` keyed bundles by their own `provenance.fusion_unit`. Then
  `bundle_identity` checked only internal consistency: the id recomputes, the source hash matches
  the commit, and the input-hash form is LF or CRLF. Nothing compared the provenance labels with
  the stored config (`model_seed`, `machine`, …), the frozen recipe or the model identity. A
  bundle with swapped seed labels was therefore counted as the other unit.
- *Correction.* `src/fusion/identity.py::unit_binding`, required by `l1-verify`. The status is
  now `FAILED_VERIFICATION` unless every check holds; completion is never inferred from the
  bundle count. It checks, for the registry unit the bundle is counted for:
  - the labels in provenance, config and score provenance;
  - one experiment id across all five id fields, recomputing from the config;
  - stored hyperparameters equal to `l1._hyperparameters` (the frozen recipe; for the AE,
    `input_dim` from the design's spectral widths);
  - representation fields (feature representation and version, normalisation, preprocessing
    config id, window, stride, split version, modality);
  - frozen training and validation cohort-contract hashes for that machine and regime. A fully
    self-consistent relabel of a deterministic Mahalanobis bundle as the other machine is caught
    only here;
  - model identity (`model_id` the recipe gives that family, seed and input size; manifest
    family);
  - model files against the manifest.

  It is a verifier only: no bundle, raw hash or id algorithm was changed.
- *Result.* 24 / 24 bundles bound, 14 CRLF + 10 LF forms preserved; `l1-verify` COMPLETE at
  `77d9cc5` (max abs diff 0.0). Adversarial tests in `tests/test_phase5_identity.py` reject:
  - A: swapped seeds (labels only, and a fully self-consistent forgery);
  - B: machine and regime relabels, self-consistent, both families;
  - C: a family relabel;
  - D: five recipe-parameter edits with ids recomputed, and a representation edit.

  They accept: E, all 24 real bundles; F, both line-ending forms.

**Finding B (minor): timestamp integrity after sorting.**
- *Root cause.* `read_sensor_parquet` sorts by timestamp and then checks monotonicity. A swapped
  pair or a reversed block therefore looks monotonic afterwards. `SensorSignal` also admits
  NaN-timestamp padding at the tail.
- *Correction.* G3 (`l3.load_raw_stream`, `l3.g3_timing`) reads the file's stored order and
  rejects:
  - non-finite timestamps;
  - non-increasing or duplicate timestamps;
  - own-stream coverage below 0.90 × nominal;

  all before the frozen reader runs. Malformed files are type D and abstain without an exception:
  - no or two timestamp columns;
  - non-numeric timestamps;
  - no samples;
  - no channels;
  - truncated bytes.

  NaN tail padding is the Protocol-A window representation produced by `_official_shape`. It
  never occurs in a raw IMAD-DS file (dataset audit; clean scan 0 G3 triggers), so in a raw file
  it is a G3 failure. The frozen reader and all closed Phase-3/4 preprocessing are unchanged, and
  clean processing reproduces Phase 4 (E0).
- *Tests* in `tests/test_phase5_l3.py`, each a G3 failure:
  - adjacent swap;
  - duplicate;
  - NaN;
  - 50 % truncation;
  - a reversed 500-sample block (which the frozen reader would silently sort away);
  - a NaN tail (which the frozen reader would accept).

  Also tested: five malformed formats, a truncated file, a one-sample stream, and segment-level
  abstention with every modality accounted for.

## 6. Protected-access accounting for Phase 5.4

| What | Kind | New protected calculation? |
|---|---|---|
| **Accidental raw inspection** on 25 October 2026: an unfiltered catalog row selected protected BrushlessMotor segment `20240423_17_20_26_segment_0` (reported as `test` in both regimes). All three raw sensor files were loaded with `pd.read_parquet`; the reported output was paths, dtypes, row counts/shapes and initial/final timestamps. | complete raw streams, including signal values, entered process memory; signal arrays, anomaly labels, prediction scores and performance metrics were not reported as printed or used | no protected inference or outcome analysis is reported. The gate settings and nominal durations were already committed in V2 on 15 October; this does not establish that the protected cohort was untouched. See the [formal incident record](phase5_4_incident_protected_raw_read.md). |
| Full-suite runs | the established Phase-3.4 replay test parses the 10 persisted Phase-3.4 score tables (historical) | no |
| Integrity snapshots | byte hashing of sealed files (2,232 entries), equal before and after | no |

**Access-check boundary.** The development driver's `_development_rows` calls
`development_segment_rows` before the clean scan or falsification suite receives raw paths.
That function filters the catalog to segments assigned train/validation in Regime T and not
test in either regime. It enforces selection for that caller; it does not authenticate paths
passed independently to lower-level functions. `load_raw_stream`, `run_gate` and `run_segment`
accept explicit paths and do not independently check cohort membership. Direct calls or ad-hoc
pandas reads can therefore bypass development selection. The L3 runner is not a universal
protected-segment refusal mechanism.

The official clean scan and falsification suite select development recordings through this
guard; synthetic unit tests also supply their own temporary paths. Their recorded outputs
remain development-only. The separate accidental raw inspection above is an access-policy
deviation, not evidence that the cohort remained unexposed. No new protected inference,
protected metrics, fusion tables or ledger appends are reported. No Phase 5.5 step was performed.
The ledger stays at 196.

## 7. Limitations

- Synthetic corruptions are negative controls for gate behaviour, not physical fault evidence.
  B5 is untested.
- These failures are not detected by design and surface only as anomaly alarms:
  - partial flatlines below 0.50 (B4);
  - gain, offset, noise, impulses and tones;
  - saturation that does not hold the rail for 20 ms.
- A genuine fault violent enough to hold a channel at the rail for ≥ 50 % of a segment would be
  gated, suppressing that modality's evidence. The integrity flag is still raised.
- One dead axis gates the whole modality.
- C and D failures abstain (v1); they are not represented by the pair tables.
- The ungated demonstration is in-sample and descriptive.
- The clean scan and falsification outputs use development data and are not protected-cohort
  results. The separate raw inspection is disclosed in section 6.
