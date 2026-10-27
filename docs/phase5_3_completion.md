# Multi-Sensor Fusion Development — Phase 5.3 Completion Record

<!-- Legacy roadmap name: Phase 5.3 -->

**Status: COMPLETE.** 24 of 24 L1 development fits and 208 of 208 L2 calibrations are verified.
Development only: no protected read, no protected inference, no test look; ledger 196.

- **Fitting code:** every fit is bound to `c6798967a34d0cf430782496b1a6139c66de1116` (Phase-5 source
  hash `f1c01c96…`) and ran with 4 torch threads, one fresh process per fit, behind the memory gate
  (measured peak + 1 GB). The RoboticArm autoencoder fits ran from a detached worktree at that commit;
  its four junctions were removed one by one and every target verified intact before the worktree was
  removed without `--force`.
- **Official `l1-verify`:** exit 0. All 24 model manifests verify; validation scores and thresholds
  reproduce exactly; every experiment id recomputes from its stored config; every source hash matches
  its commit; every fused standardiser is bit-identical to Phase 4; no duplicate unit or id.
- **Input-hash serialisation** (`docs/phase5_3_line_ending_provenance.md`): 14 CRLF, 10 LF, same
  committed content, raw hashes kept as computed. The ids predicted for the ten RoboticArm autoencoder
  fits in `docs/phase5_3_resource_recovery_note.md` assumed CRLF and are superseded by the recorded
  ids below.
- **L2:** `l2-verify` reproduces all 208 calibration records and 32,448 fused validation scores exactly.
- **Historical protected-file disclosures** (`docs/phase5_3_resource_recovery_note.md` §5) stand
  unchanged.
- **Suite:** 698 passed, 0 failed.

| Unit | Experiment id | Input-hash form | q90 | Reproduced |
|---|---|---|---:|:---:|
| RoboticArm|S|mahalanobis_shrinkage|0 | `p3-d6b20c84b9b48ce9915258aa` | CRLF | 220.83 | exact |
| RoboticArm|S|dense_autoencoder|0 | `p3-684cab6c96c667b54e1b4f3a` | LF | 0.787191 | exact |
| RoboticArm|S|dense_autoencoder|1 | `p3-b92a4d84588d57d680e72949` | LF | 0.788608 | exact |
| RoboticArm|S|dense_autoencoder|2 | `p3-2e565557cfa9e411ea66ac24` | LF | 0.787702 | exact |
| RoboticArm|S|dense_autoencoder|3 | `p3-d95d13c5eac87b81b6b00142` | LF | 0.788388 | exact |
| RoboticArm|S|dense_autoencoder|4 | `p3-a9790d804bc4fa232341b364` | LF | 0.788154 | exact |
| RoboticArm|T|mahalanobis_shrinkage|0 | `p3-c3ff10cf8f348a5c6e6104c1` | CRLF | 212.818 | exact |
| RoboticArm|T|dense_autoencoder|0 | `p3-0b4eab16d7dc7f32887145d3` | LF | 0.788089 | exact |
| RoboticArm|T|dense_autoencoder|1 | `p3-3673cbd9302b3f784b04a6a4` | LF | 0.787406 | exact |
| RoboticArm|T|dense_autoencoder|2 | `p3-374e6c8b9768d4a542412576` | LF | 0.787346 | exact |
| RoboticArm|T|dense_autoencoder|3 | `p3-fb2ff00838b9ac616550c0cd` | LF | 0.787568 | exact |
| RoboticArm|T|dense_autoencoder|4 | `p3-88b539eb6d84111eaad0f648` | LF | 0.787266 | exact |
| BrushlessMotor|S|mahalanobis_shrinkage|0 | `p3-a1a176477b5a4abe33912d45` | CRLF | 115.296 | exact |
| BrushlessMotor|S|dense_autoencoder|0 | `p3-2716c4ea337f275c128e16e4` | CRLF | 0.816259 | exact |
| BrushlessMotor|S|dense_autoencoder|1 | `p3-c7104d98baeb6b312d168b2b` | CRLF | 0.817393 | exact |
| BrushlessMotor|S|dense_autoencoder|2 | `p3-45a029d01ee6d3e165f82e65` | CRLF | 0.816436 | exact |
| BrushlessMotor|S|dense_autoencoder|3 | `p3-04b940acc27c7439e1d32b24` | CRLF | 0.816842 | exact |
| BrushlessMotor|S|dense_autoencoder|4 | `p3-4dfba9a4541ea9996153c604` | CRLF | 0.816246 | exact |
| BrushlessMotor|T|mahalanobis_shrinkage|0 | `p3-739ec5209aa8b00089c479bd` | CRLF | 109.525 | exact |
| BrushlessMotor|T|dense_autoencoder|0 | `p3-06ea507b93e119054b5c3ba4` | CRLF | 0.806411 | exact |
| BrushlessMotor|T|dense_autoencoder|1 | `p3-0abdd30c4b8457111796fd1b` | CRLF | 0.807336 | exact |
| BrushlessMotor|T|dense_autoencoder|2 | `p3-e0bd283a234f13ae699535ca` | CRLF | 0.807002 | exact |
| BrushlessMotor|T|dense_autoencoder|3 | `p3-0f1210655594b046269dad9d` | CRLF | 0.808735 | exact |
| BrushlessMotor|T|dense_autoencoder|4 | `p3-cc67d2432b24bfb11aa66f4d` | CRLF | 0.807226 | exact |

q90 values are source-validation calibration thresholds — development diagnostics, not detection
results.
