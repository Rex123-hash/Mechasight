"""Multi-Sensor Fusion entry: sealed historical test-score tables cannot be read in development.

The Phase-5.0 guard covered protected *inputs*.  Sealed protected *outputs* -
the Phase-4 staged score tables and the Phase-3.2 result-store score tables -
were still readable by ordinary code.  Their sanctioned readers now refuse
outside an ``authorized_protected_read`` scope before any file is opened.

Scope of this protection (stated, not overstated): it covers the repository's
reader APIs, and a tripwire freezes every source file that references a sealed
score location, so new development code cannot add a reader unreviewed.  It
cannot stop an arbitrary program or interactive session from opening the files.
The Phase-3.4 reader is a closed analysis whose source is frozen at 64694e6 by a
standing test; it is left unguarded and covered by the tripwire only.

No real sealed score content is opened here: real-file tests replace every
reader with a sentinel that fails if called.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import pandas as pd
import pytest

from src.experiment.phase4 import SCORE_COLUMNS, ScoreStaging
from src.experiment.provenance import file_sha256, ordered_id_hash
from src.experiment.store import ResultStore
from src.preprocessing.protected_access import (
    SEALED_SCORE_TABLES,
    ProtectedReadRefused,
    active_protected_read,
    authorized_protected_read,
)
from src.preprocessing.spectral_cache import load_spectral_cache

REAL_PHASE4_STAGING = Path("reports/phase4_protected/staging")
REAL_PHASE32_STORE = Path("reports/phase3_2/experiments")


def _sentinel(name):
    def refuse(*_args, **_kwargs):
        raise AssertionError(f"{name} was called: sealed score content would have been opened")
    return refuse


@pytest.fixture
def no_content_reads(monkeypatch):
    """Every way the guarded readers could open table content now fails loudly."""

    monkeypatch.setattr(pd, "read_csv", _sentinel("pandas.read_csv"))
    monkeypatch.setattr(pd, "read_parquet", _sentinel("pandas.read_parquet"))
    from src.experiment import store
    monkeypatch.setattr(store, "file_sha256", _sentinel("store.file_sha256"))


def _synthetic_staging(root: Path) -> tuple[ScoreStaging, str]:
    staging = ScoreStaging(root)
    staging.scores.mkdir(parents=True)
    table = pd.DataFrame({"segment_id": ["s1", "s2"], "parent_recording": ["p", "p"],
                          "domain": ["source", "target"], "score": [0.1, 0.9], "window_count": [3, 3]})
    table.loc[:, list(SCORE_COLUMNS)].to_csv(staging.scores / "e1.csv", index=False)
    return staging, "e1"


def _synthetic_bundle(root: Path, experiment_id: str = "p32-synthetic") -> Path:
    bundle = root / experiment_id
    bundle.mkdir(parents=True)
    scores = pd.DataFrame({"segment_id": ["s1", "s2"], "parent_recording": ["p", "q"], "score": [0.2, 0.7]})
    scores.to_parquet(bundle / "scores.parquet", index=False)
    (bundle / "provenance.json").write_text(
        json.dumps({"test_cohort_hash": ordered_id_hash(("s1", "s2"))}), encoding="utf-8")
    (bundle / "bundle_manifest.json").write_text(json.dumps({
        "experiment_id": experiment_id, "files": {"scores.parquet": file_sha256(bundle / "scores.parquet")},
    }), encoding="utf-8")
    return bundle


# ----------------------------------------------- unauthorised reads refuse
def test_unauthorised_phase4_score_table_read_refuses_before_opening(tmp_path, no_content_reads):
    staging, experiment_id = _synthetic_staging(tmp_path / "staging")
    assert (staging.scores / f"{experiment_id}.csv").exists() and active_protected_read() is None
    with pytest.raises(ProtectedReadRefused, match="sealed test-score table e1.csv"):
        staging.read_scores(experiment_id)


def test_unauthorised_phase32_score_table_read_refuses_before_opening(tmp_path, no_content_reads):
    _synthetic_bundle(tmp_path)
    assert (tmp_path / "p32-synthetic" / "scores.parquet").exists()
    with pytest.raises(ProtectedReadRefused, match="ResultStore.read_scores"):
        ResultStore(tmp_path).read_scores("p32-synthetic")


@pytest.mark.skipif(not (REAL_PHASE4_STAGING / "scores").is_dir() or not REAL_PHASE32_STORE.is_dir(),
                    reason="sealed historical score tables not present")
def test_real_sealed_tables_are_refused_without_being_opened_and_nothing_changes(
    no_content_reads, protected_state
):
    phase4_ids = sorted(json.loads((REAL_PHASE4_STAGING / "score_manifest.json").read_text(encoding="utf-8"))["tables"])
    phase32_ids = sorted(p.name for p in REAL_PHASE32_STORE.iterdir() if (p / "scores.parquet").is_file())
    assert len(phase4_ids) == 156 and len(phase32_ids) == 28
    staging = ScoreStaging(REAL_PHASE4_STAGING)
    for experiment_id in phase4_ids:
        assert (staging.scores / f"{experiment_id}.csv").is_file()        # existence does not bypass
        with pytest.raises(ProtectedReadRefused):
            staging.read_scores(experiment_id)
    store = ResultStore(REAL_PHASE32_STORE)
    for experiment_id in phase32_ids:
        with pytest.raises(ProtectedReadRefused):
            store.read_scores(experiment_id)
    protected_state()


# ------------------------------------------------- authorised path is preserved
def test_authorised_scope_reads_synthetic_phase4_and_phase32_tables(tmp_path):
    staging, experiment_id = _synthetic_staging(tmp_path / "staging")
    _synthetic_bundle(tmp_path / "store")
    with authorized_protected_read(authority="p5-test-batch-fixture", purpose="unit test fixture"):
        assert list(staging.read_scores(experiment_id).columns) == list(SCORE_COLUMNS)
        assert list(ResultStore(tmp_path / "store").read_scores("p32-synthetic")["segment_id"]) == ["s1", "s2"]
    assert active_protected_read() is None
    with pytest.raises(ProtectedReadRefused):
        staging.read_scores(experiment_id)


def test_authorised_phase32_read_still_verifies_hash_and_cohort(tmp_path):
    bundle = _synthetic_bundle(tmp_path)
    store = ResultStore(tmp_path)
    with authorized_protected_read(authority="p5-test-batch-fixture", purpose="unit test fixture"):
        pd.DataFrame({"segment_id": ["s1", "s3"], "parent_recording": ["p", "q"], "score": [0.2, 0.7]}) \
            .to_parquet(bundle / "scores.parquet", index=False)
        with pytest.raises(ValueError, match="hash mismatch"):
            store.read_scores("p32-synthetic")
        manifest = json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8"))
        manifest["files"]["scores.parquet"] = file_sha256(bundle / "scores.parquet")
        (bundle / "bundle_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="cohort mismatch"):
            store.read_scores("p32-synthetic")


# ------------------------------------- legitimate development reads stay open
def test_source_validation_scores_remain_readable_outside_any_scope():
    from src.fusion.assembly import assemble_unit_validation_scores, load_design_registry

    unit = load_design_registry()[0]
    scores = assemble_unit_validation_scores(unit, "reports/phase4_pretest/experiments").scores
    assert active_protected_read() is None and len(scores) in (126, 186)


def test_phase50_input_guard_is_still_operational(tmp_path, monkeypatch):
    monkeypatch.setattr("src.preprocessing.spectral_cache.h5py.File", _sentinel("h5py.File"))
    with pytest.raises(ProtectedReadRefused):
        load_spectral_cache(tmp_path / "common_evaluation.spectral.h5", modality="acc")


# ------------------------------------------------------------- contract/tripwire
def test_sealed_score_contract_names_every_known_location():
    assert set(SEALED_SCORE_TABLES) == {"phase3_2_result_store", "phase3_4_focal_toggle", "phase4_score_staging"}
    assert SEALED_SCORE_TABLES["phase3_2_result_store"][2] == "guarded"
    assert SEALED_SCORE_TABLES["phase4_score_staging"][2] == "guarded"
    assert SEALED_SCORE_TABLES["phase3_4_focal_toggle"][2].startswith("closed-stage reader frozen")


SEALED_TOKENS = re.compile(
    r"scores\.parquet|phase4_protected|phase3_2/experiments|phase3_4_results|"
    r"ScoreStaging|read_scores|load_persisted_scores|\bResultStore\b"   # not PretestResultStore (validation-only)
)
# Every src/ and scripts/ file that references a sealed score location or reader
# at the Multi-Sensor Fusion data-foundation stage.  A new reference (for
# example in fusion development code) turns this red until reviewed.
SEALED_REFERENCES = {
    "scripts/run_phase32_first_test.py": 2,
    "scripts/run_phase4_protected.py": 4,
    "src/experiment/focal_toggle_execution.py": 7,
    # Phase 5.5 governance: JSON identity manifests only; no score payload reader
    # and no protected scope. The metadata-only boundary is fixture-tested.
    "src/experiment/fusion_freeze.py": 2,
    "src/experiment/ledger_history.py": 2,
    "src/experiment/phase4.py": 5,
    "src/experiment/phase4_analysis.py": 2,
    "src/experiment/store.py": 9,
    "src/preprocessing/protected_access.py": 9,
}


def test_no_new_source_file_references_a_sealed_score_location():
    found = {}
    for root in (Path("src"), Path("scripts")):
        for path in sorted(root.rglob("*.py")):
            count = len(SEALED_TOKENS.findall(path.read_text(encoding="utf-8")))
            if count:
                found[path.as_posix()] = count
    assert found == SEALED_REFERENCES


def test_fusion_development_code_never_touches_sealed_or_protected_paths():
    for path in Path("src/fusion").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not SEALED_TOKENS.search(text), path
        assert "authorized_protected_read" not in text and "common_evaluation" not in text, path
