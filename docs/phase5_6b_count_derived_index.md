# Phase 5.6B — Trusted input references and count-derived ordinal index

**UNREVIEWED — NOT AUTHORIZED FOR REAL PROTECTED EVALUATION.**

Starting checkpoint: `fc22821de81954c46ff503b244fd8b086564603b`.
Accepted executor baseline: `5da28f82b94cdafafe33ac235dcea08d16e31692`.

## Interface and trust anchor

`phase5_input_manifest.py` implements `phase5-trusted-input-manifest-v1`.
A manifest is a reference, not permission. Every read reconstructs the expected
reference independently and checks its content identity. The scientific registry,
proposed units, split manifest, original integrity inventory and two preprocessing
configs are compared with Git objects at the existing scientific freeze commit
`d7ea8ac3022acd755a847b77118375fb719ef6e3`. The metadata helpers accept only those
six fixed paths. Parsed dictionaries are fresh; caller changes cannot mutate a
cached authority. The split CSV uses an identity-only column projection.

The manifest binds the registry/design, machine/regime, ordered cohort identities,
registered scientific units and new-L1 experiment IDs, preprocessing identity,
98-feature classical layout, mic|acc|gyro order, float32 spectral widths, artifact
paths, original stat identities and expected binary content hashes.

Development paths are fixed source-validation layouts; expected hashes come from
the existing frozen L1 registry, not caller-supplied metadata. Paths are compared
before content hashing. Traversal, external substitutions, symlinks and junctions
are refused. Dedicated synthetic fixtures use three fixed fake segment identities
and fixed temporary artifact names, with poisoned labels/provenance. They cannot
register repository or external protected paths. This preserves the existing
local-filesystem/application trust model; it is not an OS sandbox against hostile
concurrent filesystem mutation.

The protected template contains the frozen common-evaluation cohort identity and
label-free segment/parent/domain metadata (466 RoboticArm / 312 BrushlessMotor).
Protected artifact paths, hashes and stat bindings remain **null and unresolved**.
This component does not ingest protected artifact metadata or inspect protected
files to populate them. A future metadata-binding/activation task must establish
them independently; no values are invented. No protected index is constructed.

## Count-derived algorithm

`phase5_count_index.py` implements
`phase5-label-free-window-index-v2-count-derived`.

1. Read only segment, parent, domain and `window_count` from the approved feature
   artifact, using Parquet projection at read time.
2. Require exactly the frozen (machine, parent, segment) segment order and cohort.
   No silent reordering, duplicate/missing segment repair or count coercion occurs.
3. Require strictly positive integer counts. Booleans, nulls, strings, floats,
   fractions, negatives, zero and ordinal overflow are refused.
4. Check that the sum equals all three numeric dataset row counts and any declared
   spectral count; enforce frozen widths, dtype and representation identities.
5. Expand each segment into zero-based `window_index` rows, with global `row`,
   cumulative `segment_row_offset`, repeated expected `window_count`, cohort hash
   and feature-artifact identity.
6. Bind descriptor/content/count hashes. Validation reconstructs from the approved
   artifacts, so self-rehashed offset, row, count or identity tampering is refused.
   The lower draft-expansion helper reads no artifacts and grants no authority.

**No timestamps, window start/end times or physical sample boundaries are inferred.**
Counts establish ordinal alignment only. The current frozen autoencoder adapter
requires identities and ordinal grouping, so no timing field is missing for this
consumer. Any future consumer requiring physical times must obtain an independently
authorized source for them; this index cannot supply those values.

## Label isolation and protected boundary

Feature reads request exactly the count projection or frozen classical projection.
Spectral access requests only numeric mic/acc/gyro datasets and named non-outcome
attributes. `provenance_json` is never requested or deserialized. Binary hashing of
approved development/synthetic artifacts does not deserialize their poison fields.

All protected read/index interfaces always refuse in this phase, including supplied
flags, fabricated capabilities, missing/wrong batches and missing durable exposure.
The generic builder and every artifact-reader helper reject protected scope or
forged development classification before artifact hashing/opening. No credentials,
protected scopes, real-ledger mode or fallback raw reader are introduced. The 5.5G
development source manifest and its timestamp-reader protections are unchanged;
its raw reader is used only as a permitted-development validation reference.

## Development evidence and compatibility

The validation compares all four development cells with the established timestamp
index on machine, regime, segment, parent, domain, window ordinal and representation
row; it independently checks cumulative offsets and segment window counts. Coverage
is RoboticArm S/T 13,764 windows each and BrushlessMotor S/T 6,426 each: **40,380**.
This proves the demonstrated ordinal correspondence, not timestamp reconstruction
or alignment on protected contents that have never been inspected here.

The count index projects directly to the unchanged `FrozenL1Adapter` input schema.
All 24 real frozen L1 models are rescored on the complete permitted source-validation
cohorts and compared with the genuine persisted validation scores, including segment
sets and window counts. No stub, fitting, recalibration, threshold change, outcome
metric or protected analysis is involved. Discrepancies stop validation rather than
changing preprocessing. Machine-readable evidence records each model and cell.

## Additive candidate and remaining work

`reports/phase5_6b_candidate/` holds the new component source bindings, evidence and
actual focused/regression JUnit records. `verify_phase5_count_index.py` checks those
bytes against an externally supplied final Git SHA, verifies that accepted executor
sources still match the 5.5 baseline and rechecks the immutable scientific package.
The old 5.5 candidate/evidence is retained as historical evidence; it is not
regenerated or reapproved. The read-guard test allowlist has one bounded new reader
entry. This component candidate is not a complete protected-execution freeze.

Remaining: resolve protected artifact bindings from approved references; integrate
the protected input adapter and count-index event into the reviewed executor after
human authorization and durable exposure; implement/review the separate real-ledger
batch mode and final dry run/freeze. **H1–H7 remain unresolved.** In particular, this
implementation approves neither the protected count-index method nor protected
reads, 232 looks, retry cap 660, label access or ledger writes. No Phase 5.7 work is
started. The proposed uninterrupted 196 → 428 progression remains a proposal.

## Recorded checks

Final focused suite: **103 passed**. Safe regressions: **140 passed**, plus **11**
selected executor capability/comparator/recovery regressions (including native
process contention/crash cases). Both regression partitions are preserved in the
combined 151-test JUnit. There are no failures, errors or skips in these reports.
The historical protected-score replay, real-data fitting tests and official L3
experiment were not selected; one compatibility test whose snapshot hashes
protected score payloads was explicitly deselected.

The four cells agree exactly on all 40,380 ordinal window identities and offsets.
All 24 models reproduce their persisted complete source-validation scores exactly
(maximum absolute difference 0.0; 3,744 segment scores). The poisoned fixtures pass
column/dataset request traps; nine explicit unauthorized entry-point probes record
zero Parquet reads, HDF5 opens and artifact content hashes. These are implementation
checks and development evidence, not independent approval or protected performance.
