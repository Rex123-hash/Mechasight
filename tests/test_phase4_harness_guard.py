"""The suite-wide guard must refuse the real Single-Modality Benchmarking roots."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.experiment.phase4 import BatchAuthorization, ScoreStaging, ModelStore
from src.experiment.pretest import PretestResultStore

REAL_PROTECTED_ROOT = Path("reports/phase4_protected")
REAL_PRETEST_ROOT = Path("reports/phase4_pretest/experiments")


def _authorization() -> BatchAuthorization:
    return BatchAuthorization(
        batch_id="p4-test-batch-guard", experiment_ids=frozenset({"x"}),
        experiment_test_cohort_hashes={"x": "h"}, ledger_start_count=40, ledger_end_count=196,
    )


def test_guard_refuses_real_protected_staging_root(protected_state):
    """A refused open must leave every protected artifact byte-identical.

    Stage note: until the protected score-all ran (d8ccf55) this asserted that
    no real staging manifest existed.  The legitimate 156-table staging area now
    exists, so the property tested is the one that always mattered: the refused
    operation creates, removes or rewrites nothing - no ledger row, score table,
    manifest or input cache.
    """
    staging = ScoreStaging(REAL_PROTECTED_ROOT / "staging")
    manifest_before = staging.manifest_path.read_bytes() if staging.manifest_path.exists() else None
    with pytest.raises(AssertionError, match="TEST HARNESS GUARD"):
        staging.open(_authorization())
    protected_state()
    after = staging.manifest_path.read_bytes() if staging.manifest_path.exists() else None
    assert after == manifest_before


def test_guard_refuses_real_pretest_bundle_root():
    store = PretestResultStore(REAL_PRETEST_ROOT)
    with pytest.raises(AssertionError, match="TEST HARNESS GUARD"):
        store.write_bundle(config=None, provenance={}, summary={}, threshold=None, validation_scores=None)


def test_guard_allows_tmp_roots(tmp_path):
    staging = ScoreStaging(tmp_path / "staging")
    staging.open(_authorization())
    assert staging.manifest_path.exists()
