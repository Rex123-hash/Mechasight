# Phase 5.5 pre-test freeze candidate

**CANDIDATE — NOT YET AUTHORIZED FOR PROTECTED EVALUATION.**

Prepared on branch `phase5-multisensor-fusion` from
`e69e85e4616c8e8ded1f0a04bc0f50a0ff16e825`. This is a governance and
metadata candidate. It is **not execution-ready**, does not authorize an exposure,
and does not declare the independent audit complete. No fitting, tuning,
protected inference, protected score analysis or L3 protected gate scan occurs.

## Closeout integration check

The check of `e69e85e` and its Phase 5.4 predecessors is an integration review
of the implementer's previous work, not independent approval. The closeout adds
the incident note, precise guard/duplicate-scenario wording and the parameterized
pair-score/threshold regression test; it does not change V2, models or official
scientific artifacts. The 90% coverage rule appears in the V2 freeze at
`86706dc1a05dc6a009f8eea0b0b9f7cbc3ac6a25` (15 October), before the 25 October
inspection. The documented implementation checks stored timestamp order before
the historical reader can sort it. The development driver selects permitted
cohorts; the low-level L3 path-based loader is not itself a cohort authorization
boundary.

The protected raw-input incident loaded the three sensor files for the
BrushlessMotor segment `20240423_17_20_26_segment_0` into memory. Reported output
was paths, schemas, counts and timestamp ranges. Full historical console output
and the exact access time are unavailable; no stronger absence-of-exposure claim
is made. The incident does not authorize reuse, exclusion, retuning or a claim
of an untouched confirmatory cohort. See
`docs/phase5_4_incident_protected_raw_read.md` for the evidence boundary.

The official 1,992 scenarios and E0–E6 are retained. A1/mic and A2/mic produce
the same corruption on the single microphone channel in 24 segment pairs;
the scenario count is not a count of unique perturbations. Existing successful
assertions are pinned rather than rerunning the suite.

## Package and artifact resolution

`reports/phase5_freeze_candidate/candidate_manifest.json` hashes the five package
documents, builder, review documentation and guard tests. Its Git commit anchors the candidate. Rebuild with
`C:/Python314/python.exe -B scripts/prepare_phase5_freeze_candidate.py` from the
repository root. The builder writes only the candidate directory and refuses
tracked scientific changes relative to the base. Stat inventories are specific
to this checkout; a different checkout may have different mtimes.

| Artifact | Contents |
|---|---|
| `scientific_registry.json` | All 52 machine/regime/family/seed units, 208 detector references, 208 calibration records and 52 complete L3 fallback maps |
| `proposed_evaluation_manifest.json` | The exact 232 proposed IDs, deterministic proposed batch ID, common-cohort hashes, accounting, exposure ordering and recovery policy |
| `scientific_contract.json` | Verbatim V2 recipes, cohorts, metrics, interpretation, contrasts, statistics, claim limits and protected protocol |
| `integrity_inventory.json` | Complete base Git-tree blob inventory, ignored data/report stat inventory, development/model content hashes and closed ledger segments |
| `validation_report.json` | Checks actually performed, environment, unchanged ledger count and unresolved authorization gates |

L0 is the 156 frozen Phase-4 modality detectors and their own raw thresholds.
L1 is 24 actual development-fit IDs plus 28 reused Phase-3.2 all-channel IDs.
The ten RoboticArm AE IDs come from the completed bundles, not the superseded
CRLF predictions. Historical classical model objects were not serialized;
their score tables are reused and no model is reconstructed or refitted.

For every detector the registry records its bundle path/hash, fit commit,
hyperparameters, recorded environment, preprocessing/split hashes, all four
training/validation cohort/contract hashes, source-validation threshold, and
persisted model/standardizer hashes where models exist. Threshold segment lists
remain in the original hash-pinned bundle; the registry includes their count
and cohort hash. Shared recipe model IDs are not unique fitted artifacts:
the experiment ID, scientific unit, store path and file hashes disambiguate them.

Each of the 52 L2 units has all-three, acc+gyro, mic+gyro and mic+acc calibrations.
The registry embeds the original median/Q25/Q75/IQR values and each subset's own
q90 threshold. L3 all-valid maps to all-three; one value-invalid modality maps to
its surviving pair; two map to the original surviving Phase-4 raw detector and
threshold. All-invalid abstains; any timestamp/alignment/absent-stream failure
abstains. An integrity flag is emitted independently of the anomaly decision.
L3 is not run on protected recordings under this design.

## Proposed exposure and statistical contract

24 new L1 inferences + 208 derived L2 tables = **232 proposed new looks**.
Only a future authorized batch could append those rows, taking 196 to 428.
The 156 historical single-modality and 28 reused all-channel tables are
184 comparator reanalyses, adding zero looks. L3 adds no protected execution.
No row is appended by this package. Planned IDs denote proposed evaluation
identities, not completed protected results. A proposed batch ID is not a token.
The actual scoring HEAD must be bound separately without changing the approved
unit/cohort set.

The machine-readable contract copies V2 rather than restating new rules.
Primary axes remain full-cohort AUROC, healthy-target FPR and full-cohort recall.
AUPRC is the existing average-precision implementation, with prevalence context;
secondary outputs include standardized pAUROC at FPR <= 0.10 and the frozen
domain slices, precision and count-identity F1. Every operating decision uses its
source-validation q90 (`higher`) threshold. L2 IQR quantiles use `linear`;
non-finite scores or zero IQR are technical defects, not imputed values.

Paired class-stratified parent-cluster bootstrap uses 10,000 shared draws,
seed 1995 and percentile 95% intervals. R8 averages per-seed metrics inside
each replicate; it does not select a best seed or pool seed scores. Invalid
healthy-target replicates are counted, not filled; fewer than 9,000 valid
replicates yields the frozen inconclusive/indeterminate interpretation.
Practical margins remain AUROC/AP 0.02, healthy-target FPR 0.10 and recall 0.05.
Equivalence requires the whole interval strictly inside the relevant margin.
The exact ranking/operating/joint category rules are preserved in the copied
contract and existing statistics implementation. The 140 primary contrasts
produce 420 nominal intervals with no multiplicity adjustment. Secondary and
post-hoc comparisons retain their declared status. No fusion advantage is claimed.

## Verification scope and evidence limits

Fresh checks recompute experiment IDs, verify development bundle and model-file
hashes without deserializing models, bind all 24 new L1 units to their labels,
frozen recipe and Phase-4 pre-test cohort contracts, verify source hashes at the
recorded fit commit, and reconstruct every L2 calibration/validation row.
The existing unit-aware verifier's 24 successful binding records and
LF/CRLF provenance classifications are pinned unchanged.

The full `unit_binding` and `bundle_identity` functions are **not rerun** here:
they read the full split-assignment CSV, including protected labels. Metadata
cross-checking against Phase-4 pre-test contracts is explicitly a different
check, not an assertion of fresh label-derived verification. Prior validation
score reproduction and standardizer equivalence are recorded evidence; no model
inference is repeated in this task.

All base tracked files are inventoried through Git object names and sizes;
Git name-only integrity comparison detects changes without displaying scientific
payloads. Every ignored data/report file is inventoried with size and mtime,
including legacy caches and raw files. These stats do **not** establish current
content hashes. Protected cache sidecars and score payloads are not opened.
The 196 ledger rows and their four closed append segments are checked by the
existing history verifier using authorization metadata only. Previously recorded
historical table hashes remain evidence, not a fresh byte-level table audit.

No new scope is enabled. Existing protections are accident guards, not an OS
security boundary. In particular arbitrary raw-file access and legacy readers
cannot be described as universally blocked. Prior aggregate protected results,
historical score inspections disclosed in the entry/audit records, and the
Phase 5.4 raw-input incident remain part of the exposure history. The candidate
does not reset that history or certify an untouched common-test cohort.

## Authorization blockers and safe handoff

1. Independent read-only review has not occurred. The implementer cannot approve
   their own candidate. No human protected-evaluation token has been issued.
2. Frozen V2 section 13 calls for a protected executor tested on fixtures and a
   reviewed scope entry. That executor is absent. This candidate deliberately
   leaves the scope inventory unchanged. An explicitly scoped implementation,
   fixture tests, refreshed freeze and independent audit are required **before
   Phase 5.6 authorization**, preserving exposure-before-read and score-all-before-
   analysis ordering. The metadata candidate does not waive that requirement.
3. Future authorized integrity checks must verify protected input/table contents
   after exposure is recorded, before inference/derivation. A mismatch consumes
   the recorded looks and is a technical failure under V2. Persisted units are
   retained; retries need new authorization, batch and recovery IDs.
4. The independent reviewer must assess the recorded full-verifier evidence,
   metadata binding, LF/CRLF identity history, known access incidents and residual
   guard limits without replaying historical protected outputs.

Scientific limits remain: prior benchmark exposure; BrushlessMotor acquisition-
time/class confounding; RoboticArm healthy-target 20 segments from six parents;
capacity asymmetry; in-sample validation calibration; surviving-physical-sensor
independence (B5) as an assumption; and synthetic corruptions as software evidence,
not demonstrated physical-world robustness. Partial flatlines, gain/drift/noise
and some timing errors lie outside the gate; saturation may suppress genuine
fault evidence while still raising the integrity flag.

The safe next step is an independent read-only audit of this candidate and its
listed blockers. Phase 5.6/5.7 remain unstarted. `MechaSight_First_Defence/` is
outside this task and must remain unstaged.

## Focused validation evidence

The final focused run passed **104 tests**, with **32 deselected**, in 4.43 seconds.
The first run found an ordering defect in the new builder: JSON round-tripping
changed L2 dictionary iteration order. Sorting planned units by experiment ID
fixed it before the candidate was committed; no scientific artifact changed.

Coverage includes adverse registry mutations (unit/family/seed, components,
subsets, calibration policy/cohorts, fallback maps, abstentions, non-finite
thresholds and missing identities), ledger collision/start refusal, package
hashes, exact V2 contract preservation and a metadata rebuild with protected
input/payload reads replaced by fail-before-open sentinels.
New L1 binding checks additionally refuse changed config seeds, recipes, cohort
contracts, model identities and score-provenance experiment IDs. The existing
synthetic statistics tests, L3 fallback/threshold regressions and guarded-reader
tests also pass. The new governance module adds two metadata-manifest references
to the sealed-location tripwire; it introduces no payload reader or scope.

The run used Python 3.14.4, `PYTHONDONTWRITEBYTECODE=1`,
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` and `pytest -p no:cacheprovider -q`, selecting
the candidate/statistics suites and the permitted synthetic/static checks in
`test_phase5_l3.py`, `test_phase5_sealed_score_guard.py` and
`test_phase5_protected_read_guard.py`. Real protected-state snapshot tests and
real model/raw-data reproduction tests were deselected. No historical score
replay, fitting or full corruption-suite rerun was used to obtain this count.

Exact PowerShell invocation (from repository root):

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
C:/Python314/python.exe -B -m pytest -p no:cacheprovider -q tests/test_phase5_freeze_candidate.py tests/test_phase5_statistics.py tests/test_phase5_l3.py tests/test_phase5_sealed_score_guard.py tests/test_phase5_protected_read_guard.py -k 'not real_sealed and not real_materialised and not clean_runner and not dead_modality and not development_rows and not segment_selection and not catalogue and not flatline_run and not irregular_timestamps and not nan_breaks and not g1_and_g2 and not g3_reads and not larger_reversed and not nan_timestamp_tail and not malformed_streams and not truncated_file and not any_c_or_d and not absent_or_unreadable and not partial_corruptions and not value_corruptions and not timestamp_corruptions_and_absence and not noise_is_deterministic'
```

All 208 calibration records and 32,448 fused source-validation rows reproduce
exactly. A separate before/after comparison confirms **236 original raw-byte
file hashes unchanged** (Phase-5 development artifacts/models, V1/V2 design,
V2 document and ledger) and **32 cache-file stat identities unchanged**.
The inventory records all **2,195 base tracked files** and **14,977 ignored
data/report files**. The ledger remains 196 and the frozen design's canonical
SHA-256 remains `2e33299d3c0347ebdc13392d149b248b8ec175f2bea5b2049e5bb2fa996a759a`.
