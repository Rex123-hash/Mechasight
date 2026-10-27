# Phase 3.1.1 Experiment Code-Identity Audit

Status: **PASS with explicit Git-field semantic clarification**.

All 28 immutable Phase-3.1 bundles contain the same source snapshot hash, `93c33580054e9e45f228e385ba726bb43ad0dfb3531f83bb9f3c9c5fb003e229`, in both model hyperparameters and score-provenance configuration. Their identity mismatch count is zero.

The snapshot covers exactly:

1. `src/models/classical.py`
2. `src/experiment/phase31.py`
3. `src/experiment/data.py`
4. `src/experiment/pretest.py`
5. `configs/phase3/preprocessing_primary_v1.json`
6. `configs/phase3/baseline_ladder_v1.json`

The six file contents at accepted Phase-3.1 commit `78b536304d94d50afcc293348c5fa0c34e729f62` reproduce the snapshot exactly.

The bundle field named `code_git_sha` contains `3bf046dd8edb29d99f1d246b33f88674a5b53a2d`. This is the accepted base commit, not the final generation commit. Its corrected semantic interpretation is `accepted_base_git_sha`. The authoritative complete generation-tree identity is commit `78b536304d94d50afcc293348c5fa0c34e729f62`, combined with the source snapshot above. This audit and the amended Phase-3.1 documentation establish that interpretation before any common-test access without mutating or regenerating the frozen bundles.

The six-file snapshot does not independently cover every imported dependency. Dependencies unchanged during implementation are anchored by the accepted base SHA; the final Phase-3.1 commit supplies the complete repository-tree identity.

No real experiment was rerun. No test feature, score, metric, or test-look record was created.
