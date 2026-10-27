"""Suite-wide guard: no test may consume a real protected-test look.

Added after the step-D incident (2026-09-11T19:50:25Z): a gate test pointed
``score_experiment`` at the real ``reports/phase3_4_results`` root, expecting
the ledger gate to refuse. Once the ten looks were authorised that gate passed
and the real single-use claim for ``p34d-92fd40fc93c8c8d2e819`` was created
before the test's inference sentinel fired. The scorer behaved exactly as
frozen; the test harness did not. This fixture makes the harness structurally
unable to repeat that: every in-process claim, exposure-record write, and
ledger append is checked against the real repository paths and refused before
any file is created. The scorer's own guards are not modified.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

REAL_RESULTS_ROOT = Path("reports/phase3_4_results").resolve()
REAL_LEDGER = Path("reports/experiment_results/test_look_ledger.jsonl").resolve()
# Single-Modality Model Benchmarking (Phase 4) real roots: pre-test bundles
# and the protected score-all / analyse-once staging area.
REAL_PHASE4_PRETEST_ROOT = Path("reports/phase4_pretest").resolve()
REAL_PHASE4_PROTECTED_ROOT = Path("reports/phase4_protected").resolve()
# Multi-Sensor Fusion development roots (Phase 5.3) and the frozen Phase-4 model
# store: tests must never write into them.
REAL_PHASE5_DEV_ROOT = Path("reports/phase5_development").resolve()
REAL_PHASE5_MODEL_ROOT = Path("data/interim/phase5").resolve()
REAL_PHASE4_MODEL_ROOT = Path("data/interim/phase4/models").resolve()
REAL_WRITE_ROOTS = (REAL_PHASE4_PRETEST_ROOT, REAL_PHASE4_PROTECTED_ROOT, REAL_PHASE5_DEV_ROOT,
                    REAL_PHASE5_MODEL_ROOT, REAL_PHASE4_MODEL_ROOT)
# Materialised protected inputs (git-ignored).  Reads of their contents are
# refused outside an authorized scope by src.preprocessing.protected_access.
REAL_PHASE4_SPECTRAL_ROOT = Path("data/interim/phase4/spectral").resolve()
REAL_COMMON_TEST_FEATURE_ROOT = Path("data/interim/phase32_common_test").resolve()


def _inside(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(root)
        return True
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def _never_touch_the_real_protected_state(monkeypatch):
    from src.experiment import focal_toggle_execution as fx
    from src.experiment import ledger as tl

    real_acquire = fx.acquire_score_claim
    real_append_durable = fx._append_durable
    real_append_many = tl.TestLookLedger.append_many_atomic
    real_append = tl.TestLookLedger.append

    def guarded_acquire(exp, claim):
        target = fx.score_claim_path(exp)
        if _inside(target, REAL_RESULTS_ROOT):
            raise AssertionError(
                f"TEST HARNESS GUARD: refusing to create a real protected-score claim at "
                f"{target}. Tests must redirect fx.RESULTS_ROOT to tmp_path.")
        return real_acquire(exp, claim)

    def guarded_append_durable(path, record):
        if _inside(path, REAL_RESULTS_ROOT):
            raise AssertionError(
                f"TEST HARNESS GUARD: refusing to append a real exposure record at {path}.")
        return real_append_durable(path, record)

    def guarded_append_many(self, *args, **kwargs):
        if Path(self.path).resolve() == REAL_LEDGER:
            raise AssertionError("TEST HARNESS GUARD: refusing to append to the real ledger.")
        return real_append_many(self, *args, **kwargs)

    def guarded_append(self, record):
        if Path(self.path).resolve() == REAL_LEDGER:
            raise AssertionError("TEST HARNESS GUARD: refusing to append to the real ledger.")
        return real_append(self, record)

    from src.experiment import phase4 as p4
    from src.experiment import pretest as pt

    real_staging_open = p4.ScoreStaging.open
    real_model_write = p4.ModelStore.write
    real_bundle_write = pt.PretestResultStore.write_bundle

    def guarded_staging_open(self, authorization):
        if _inside(self.root, REAL_PHASE4_PROTECTED_ROOT):
            raise AssertionError(
                f"TEST HARNESS GUARD: refusing to open real protected staging at {self.root}.")
        return real_staging_open(self, authorization)

    def guarded_model_write(self, **kwargs):
        if any(_inside(self.root, root) for root in REAL_WRITE_ROOTS):
            raise AssertionError(
                f"TEST HARNESS GUARD: refusing to write a real model artifact under {self.root}.")
        return real_model_write(self, **kwargs)

    def guarded_bundle_write(self, **kwargs):
        if any(_inside(self.root, root) for root in REAL_WRITE_ROOTS):
            raise AssertionError(
                f"TEST HARNESS GUARD: refusing to write a real pre-test bundle under {self.root}.")
        return real_bundle_write(self, **kwargs)

    monkeypatch.setattr(fx, "acquire_score_claim", guarded_acquire)
    monkeypatch.setattr(fx, "_append_durable", guarded_append_durable)
    monkeypatch.setattr(tl.TestLookLedger, "append_many_atomic", guarded_append_many)
    monkeypatch.setattr(tl.TestLookLedger, "append", guarded_append)
    monkeypatch.setattr(p4.ScoreStaging, "open", guarded_staging_open)
    monkeypatch.setattr(p4.ModelStore, "write", guarded_model_write)
    monkeypatch.setattr(pt.PretestResultStore, "write_bundle", guarded_bundle_write)

    from src.fusion import l2 as fusion_l2
    real_l2_write = fusion_l2.write_calibration_artifacts

    def guarded_l2_write(out_dir, *args, **kwargs):
        if any(_inside(Path(out_dir), root) for root in REAL_WRITE_ROOTS):
            raise AssertionError(f"TEST HARNESS GUARD: refusing to write real L2 artifacts under {out_dir}.")
        return real_l2_write(out_dir, *args, **kwargs)

    monkeypatch.setattr(fusion_l2, "write_calibration_artifacts", guarded_l2_write)
    yield


def snapshot_protected_state() -> dict[str, str]:
    """Identity of every protected artifact a refused operation must leave untouched.

    Committed protected records are content-hashed.  The large git-ignored input
    caches are identified by size and modification time, which any write
    changes, so the snapshot stays cheap.  The file set itself is part of the
    identity: a newly created score, manifest or cache is a mutation.
    """

    state = {"ledger": sha256(REAL_LEDGER.read_bytes()).hexdigest()}
    for root in (REAL_RESULTS_ROOT, REAL_PHASE4_PRETEST_ROOT, REAL_PHASE4_PROTECTED_ROOT):
        for path in sorted(root.rglob("*")):
            if path.is_file():
                state[path.as_posix()] = sha256(path.read_bytes()).hexdigest()
    for root in (REAL_PHASE4_SPECTRAL_ROOT, REAL_COMMON_TEST_FEATURE_ROOT):
        for path in sorted(root.rglob("*")) if root.exists() else ():
            if path.is_file():
                stat = path.stat()
                state[path.as_posix()] = f"size={stat.st_size};mtime_ns={stat.st_mtime_ns}"
    return state


@pytest.fixture
def protected_state():
    """Snapshot before the test; call the returned function to compare after."""

    before = snapshot_protected_state()

    def assert_unchanged() -> None:
        after = snapshot_protected_state()
        added = sorted(set(after) - set(before))
        removed = sorted(set(before) - set(after))
        changed = sorted(k for k in set(before) & set(after) if before[k] != after[k])
        assert not (added or removed or changed), (
            f"protected state mutated: added={added[:5]} removed={removed[:5]} changed={changed[:5]}")

    return assert_unchanged
