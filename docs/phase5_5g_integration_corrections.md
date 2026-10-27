# Phase 5.5G — Two integration corrections

**UNREVIEWED — NOT AUTHORIZED FOR REAL PROTECTED EVALUATION.**

Starting commit: `4016dbb1c8ea07d71051c7f77957b1c774ac128d`.
This implementation corrects the two Phase 5.5F findings only. It does not approve
itself, train models, construct a protected index or enable protected execution.

## Development source boundary

`development_sources.json` binds 624 machine/regime/segment records (312 distinct
source-validation segments) to their original recording parents, domain, three
canonical raw paths, preprocessing configurations and original inventory stat
identities. The manifest is pinned by an explicit SHA-256 in the index code and
included in the executable candidate inventory. This is a development allowlist,
not a second protected authorization mechanism.

The source records were built from the unchanged frozen v3 split manifest at the
starting commit, projected to identity/split/path-resolution fields. Sensor paths
were resolved from identity and sensor-filename projections of only the necessary
release metadata CSVs. No outcome column was deserialized. Every selected segment
is source-validation and absent from the protected test split in both regimes;
every bound raw path matched the original freeze inventory's size and mtime.
Released file-split names are not treated as experimental permissions: some
permitted RoboticArm source-validation recordings reside in the released `test`
folder, but none belongs to the frozen protected experimental cohort.

Before any sensor schema, content hash or timestamp read, the generic builder
preflights **all** sources against the pinned manifest. Direct `segment_windows`
and `_timestamps` calls enforce the same membership, parent/domain, modalities,
configuration and canonical-path rules. The fixed manifest, frozen split and
configuration files are rechecked on entry. Changed/missing trust metadata fails
closed. Caller claims and substituted catalogs cannot grant access.

Traversal, unexpected external paths, symlinks and Windows junctions are refused.
Allowed paths retain their original stat identity. This is the existing local
filesystem/application trust model, not a sandbox against hostile filesystem
mutation during an OS read. No raw recording is content-hashed by this builder.
Timestamp transformations, common coverage, bounds and row ordering are unchanged.
The default development fixture now resolves paths from this label-free manifest
instead of loading the full release catalog. The bounded allowlist deliberately
supports the established source-validation reproduction only, not training or
protected index construction. The protected constructor still always refuses.

## Computation provenance

The executor recognizes the actual callable: the exact synthetic predictor or an
exact `FrozenL1Adapter` bound to its frozen package, with a compatible input schema.
Other callbacks cannot claim a real-adapter identity by supplying a string. Real
inference invokes the adapter through its class. Each output records:

* `frozen_L1_inference`: adapter version/source hash, frozen model manifest,
  experiment identity, family, seed and original fitting-code identity;
* `synthetic_L1_stub`: explicit `SYNTHETIC_STUB_V1`, unit identity and source hash;
* `derived_L2_scoring`: frozen robust-z/mean implementation, calibration identity
  and hash, subset, registered component IDs, frozen bundle hashes and exact
  component fixture-table hashes. Component origin distinguishes real development
  validation scores from fabricated synthetic fixture scores.

Completion recomputes expected computation lineage and checks both `computation`
and `predictor`, as well as the existing scientific provenance and file hashes.
Missing/forged identities fail even if the output and state hashes are recomputed.
Historical L1 comparators remain existing analysis-only inputs; their frozen
registry/bundle identities and capability checks are unchanged. They are not
relabeled as newly inferred models or appended as new looks.

## Verification and evidence

`test_phase5_integration_corrections.py` repeats the original unregistered-target
counterexample at all three index boundaries and exercises identity, path,
modality, representation, forged-claim, link/junction and trust-manifest failures.
Rejection tests assert zero timestamp reads, Parquet schema opens and raw-source
hashes. A late bad source is rejected before an earlier valid source is read.
Genuine development window construction and timestamp projections still work.

Existing real-model reproduction tests additionally check both real L1 families
and all L2 provenance. Synthetic tests cover all 24 stub outputs, 208 derived
outputs, forged lineage and deceptive callbacks. Candidate evidence separately
records real-model development reproduction, the stub demonstration, provenance
coverage, original-counterexample refusals, and focused/regression JUnit reports.
The historical protected-score replay is excluded; no official L3 experiment or
model-training job is run. Statistical methods and numerical scoring are unchanged.

## Remaining prerequisites

Real evaluation still requires a separately reviewed protected-input/path-resolver
interface, explicit human authorization and durable pre-access ledger accounting
for protected timestamp/index construction, frozen common-evaluation
representations, a real-ledger batch integration, final executable freeze and
human approval. None is implemented or authorized by this correction.

## Recorded final checks

The focused JUnit combines two disjoint passing partitions: 76 integration and
adversarial tests plus 242 executor/security/statistics/recovery tests, **318 total**
with no failures, errors or skips. The broader compatibility suite has **97 passes**;
one test whose snapshot fixture hashes protected score payloads was deliberately
deselected. Historical Phase-3.4 protected-score replay and real-data fitting tests
were not selected. JUnit reports retain their actual run timestamps and test cases.

The final demonstration reproduces all 24 real frozen L1 models and all 208 L2
tables (2,496 fusion rows) exactly: both maximum absolute differences are 0.0.
All 232 numerical output tables also equal the Phase 5.5F outputs separately in
the real-model development fixture and the stub-only synthetic fixture. Provenance
now records 24 real L1 + 208 derived L2 outputs in the former, and 24 explicit stub
L1 + 208 synthetic-component L2 outputs in the latter. The original counterexample
is refused at all three entry points with zero sensor opens/reads/hashes. The
temporary JSONL history verifies at 428; the real ledger remains byte-identical at
196. The 708 frozen artifact hashes and 14,977 original ignored-file stat identities
verify unchanged. These are implementation checks, not independent approval.
