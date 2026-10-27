"""Multi-Sensor Fusion entry hygiene: the test-look ledger is append-only history.

``verify_ledger_history`` replaces the per-stage row-count snapshots that went
red when Single-Modality Benchmarking legitimately appended 156 looks.  These
tests pin the real ledger and show, on temporary copies, that a future
authorised batch verifies while edits, reorders, deletions, partial writes,
unauthorised appends and repeated looks do not.  No real ledger is written.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from src.experiment.ledger_history import (
    CLOSED_SEGMENTS,
    LEDGER_PATH,
    PHASE3_CLOSE_PREFIX_SHA256,
    PHASE3_CLOSE_ROWS,
    PHASE4_CLOSE_PREFIX_SHA256,
    PHASE4_CLOSE_ROWS,
    LedgerHistoryError,
    prefix_sha256,
    verify_ledger_history,
)


# ------------------------------------------------------------- real ledger
def test_real_ledger_is_four_closed_authorised_atomic_segments():
    history = verify_ledger_history()
    closed = [(s.stage, s.start, s.end, s.protocol, s.batch_id) for s in history.segments if s.closed]
    assert closed == [
        ("phase3_2_first_common_test", 0, 28, "mechasight_experimental_protocol",
         "p32-test-batch-8f7d48324fd10df97ae29100"),
        ("phase3_3_protocol_a_reproduction", 28, 30, "official_baseline_reproduction", None),
        ("phase3_4_focal_parent_sibling_toggle", 30, 40, "roboticarm_focal_parent_sibling_toggle_v1", None),
        ("phase4_single_modality_benchmark", 40, 196, "mechasight_experimental_protocol",
         "p4-test-batch-0884109383d3695de7a07121"),
    ]
    assert prefix_sha256(history.lines, PHASE3_CLOSE_ROWS) == PHASE3_CLOSE_PREFIX_SHA256
    assert prefix_sha256(history.lines, PHASE4_CLOSE_ROWS) == PHASE4_CLOSE_PREFIX_SHA256


def test_phase4_segment_matches_its_batch_authorization_and_sealed_score_manifest():
    history = verify_ledger_history()
    rows = history.rows_of("phase4_single_modality_benchmark")
    auth = json.loads(Path("reports/phase4_protected/batch_authorization.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        Path("reports/phase4_protected/staging/score_manifest.json").read_text(encoding="utf-8"))
    assert (auth["ledger_start_count"], auth["ledger_end_count"], auth["experiment_count"]) == (40, 196, 156)
    assert {r["git_sha"] for r in rows} == {auth["code_git_sha"]}
    assert {r["experiment_id"] for r in rows} == set(manifest["tables"])
    assert manifest["batch_id"] == auth["batch_id"]


def test_closed_segment_registry_is_contiguous_from_the_first_row():
    assert CLOSED_SEGMENTS[0].start == 0
    for previous, current in zip(CLOSED_SEGMENTS, CLOSED_SEGMENTS[1:]):
        assert current.start == previous.end
    assert CLOSED_SEGMENTS[-1].end == PHASE4_CLOSE_ROWS


# ------------------------------------------------------ temporary copies
FUTURE_SHA = "f" * 40


@pytest.fixture
def copy(tmp_path):
    """A temporary ledger copy plus an empty authorization root."""

    ledger = tmp_path / "ledger.jsonl"
    ledger.write_bytes(Path(LEDGER_PATH).read_bytes())
    root = tmp_path / "reports"
    root.mkdir()
    return ledger, root


def _row(experiment_id: str, batch_id: str | None, *, sha: str = FUTURE_SHA,
         timestamp: str = "2026-11-01T00:00:00+00:00") -> dict:
    reason = "multi-sensor fusion test look"
    if batch_id is not None:
        reason += f"; batch={batch_id}; policy=p5-test"
    return {
        "experiment_id": experiment_id, "git_sha": sha, "machine": "RoboticArm",
        "metrics_requested": ["auroc"], "model_config_identity": "p5-model-test",
        "protocol": "mechasight_experimental_protocol", "reason": reason, "regime": "T",
        "schema_version": "phase3.0-test-look-v1", "test_cohort_hash": "c" * 64,
        "timestamp_utc": timestamp,
    }


def _append(ledger: Path, rows: list[dict]) -> None:
    with ledger.open("ab") as handle:
        for row in rows:
            handle.write((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))


def _authorize(root: Path, batch_id: str, start: int, count: int, *, sha: str = FUTURE_SHA,
               where: str = "phase5_protected", **override) -> Path:
    record = {"batch_id": batch_id, "ledger_start_count": start, "ledger_end_count": start + count,
              "code_git_sha": sha, "experiment_count": count, **override}
    path = root / where / "batch_authorization.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def _rewrite(ledger: Path, transform) -> None:
    lines = ledger.read_bytes().splitlines(keepends=True)
    ledger.write_bytes(b"".join(transform(lines)))


def test_an_authorised_future_batch_verifies_without_editing_the_registry(copy):
    ledger, root = copy
    _append(ledger, [_row(f"p5-x{i}", "p5-test-batch-a") for i in range(3)])
    _authorize(root, "p5-test-batch-a", 196, 3)
    history = verify_ledger_history(ledger, authorization_root=root)
    tail = history.segments[-1]
    assert (tail.stage, tail.start, tail.end, tail.closed) == ("authorized_batch:p5-test-batch-a", 196, 199, False)
    assert prefix_sha256(history.lines, PHASE4_CLOSE_ROWS) == PHASE4_CLOSE_PREFIX_SHA256


def test_two_consecutive_authorised_batches_verify(copy):
    ledger, root = copy
    _append(ledger, [_row("p5-a0", "p5-test-batch-a"), _row("p5-a1", "p5-test-batch-a")])
    _append(ledger, [_row("p5-b0", "p5-test-batch-b", timestamp="2026-11-02T00:00:00+00:00")])
    _authorize(root, "p5-test-batch-a", 196, 2, where="a")
    _authorize(root, "p5-test-batch-b", 198, 1, where="b")
    stages = [s.stage for s in verify_ledger_history(ledger, authorization_root=root).segments]
    assert stages[-2:] == ["authorized_batch:p5-test-batch-a", "authorized_batch:p5-test-batch-b"]


def test_a_crlf_checkout_of_the_ledger_still_verifies(copy):
    ledger, root = copy
    ledger.write_bytes(ledger.read_bytes().replace(b"\n", b"\r\n"))
    assert len(verify_ledger_history(ledger, authorization_root=root).rows) == PHASE4_CLOSE_ROWS


@pytest.mark.parametrize("mutate, message", [
    (lambda lines: lines[:5] + [lines[5].replace(b'"regime":"', b'"regime":"X')] + lines[6:],
     "byte-identical"),
    (lambda lines: lines[:100] + [lines[100].replace(b"RoboticArm", b"RoboticArX")] + lines[101:],
     "byte-identical"),
    (lambda lines: [lines[1], lines[0], *lines[2:]], "byte-identical"),
    (lambda lines: lines[:-1], "deleted"),
    (lambda lines: lines[:PHASE3_CLOSE_ROWS - 1], "deleted"),
    (lambda lines: lines[:-1] + [lines[-1].rstrip(b"\n")], "partial append"),
    (lambda lines: lines + [b"not json\n"], "not JSON"),
    (lambda lines: lines + [b'{"experiment_id": "orphan"}\n'], "required look fields"),
], ids=["edit-phase32-row", "edit-phase4-row", "reorder", "delete-last", "truncate-phase3",
        "partial-write", "garbage-row", "incomplete-row"])
def test_history_rejects_edits_reorders_deletions_and_malformed_rows(copy, mutate, message):
    ledger, root = copy
    _rewrite(ledger, mutate)
    with pytest.raises(LedgerHistoryError, match=message):
        verify_ledger_history(ledger, authorization_root=root)


def test_an_append_that_names_no_batch_is_unauthorised(copy):
    ledger, root = copy
    _append(ledger, [_row("p5-x0", None)])
    with pytest.raises(LedgerHistoryError, match="unauthorised"):
        verify_ledger_history(ledger, authorization_root=root)


def test_an_append_without_its_authorization_record_is_unauthorised(copy):
    ledger, root = copy
    _append(ledger, [_row("p5-x0", "p5-test-batch-a")])
    with pytest.raises(LedgerHistoryError, match="exactly one"):
        verify_ledger_history(ledger, authorization_root=root)
    _authorize(root, "p5-test-batch-a", 196, 1, where="one")
    _authorize(root, "p5-test-batch-a", 196, 1, where="two")
    with pytest.raises(LedgerHistoryError, match="exactly one"):
        verify_ledger_history(ledger, authorization_root=root)


@pytest.mark.parametrize("override, message", [
    ({"ledger_start_count": 195}, "disagree"),
    ({"ledger_end_count": 199}, "disagree"),
    ({"experiment_count": 3}, "experiment_count"),
    ({"code_git_sha": "e" * 40}, "code_git_sha"),
])
def test_an_append_must_match_its_authorization_exactly(copy, override, message):
    ledger, root = copy
    _append(ledger, [_row("p5-x0", "p5-test-batch-a"), _row("p5-x1", "p5-test-batch-a")])
    _authorize(root, "p5-test-batch-a", 196, 2, **override)
    with pytest.raises(LedgerHistoryError, match=message):
        verify_ledger_history(ledger, authorization_root=root)


def test_a_second_look_at_an_already_looked_experiment_is_refused(copy):
    ledger, root = copy
    existing = json.loads(Path(LEDGER_PATH).read_bytes().splitlines()[50])["experiment_id"]
    _append(ledger, [_row(existing, "p5-test-batch-a")])
    _authorize(root, "p5-test-batch-a", 196, 1)
    with pytest.raises(LedgerHistoryError, match="more than once"):
        verify_ledger_history(ledger, authorization_root=root)


def test_a_closed_stage_with_a_missing_authorization_record_is_refused(copy, tmp_path):
    ledger, root = copy
    first = CLOSED_SEGMENTS[0]
    moved = type(first)(**{**first.__dict__, "authorization": str(tmp_path / "missing.json")})
    with pytest.raises(LedgerHistoryError, match="missing"):
        verify_ledger_history(ledger, authorization_root=root, closed=(moved, *CLOSED_SEGMENTS[1:]))


def test_registry_hashes_are_the_hashes_of_the_rows_they_name():
    lines = Path(LEDGER_PATH).read_bytes().replace(b"\r\n", b"\n").splitlines(keepends=True)
    for seg in CLOSED_SEGMENTS:
        assert sha256(b"".join(lines[seg.start:seg.end])).hexdigest() == seg.sha256, seg.stage
