"""Append-only history of the protected test-look ledger.

The ledger only grows.  Every authorised protected stage appended its looks in
one atomic write, so the file is a sequence of contiguous segments, one per
stage.  Closed stages are frozen here by row range, byte hash and committed
authorization record.

Anything appended after the last closed stage must be an atomic segment bound
to a committed batch authorization record, following the convention that
Single-Modality Benchmarking introduced: every row's ``reason`` carries
``batch=<id>``, and exactly one ``batch_authorization.json`` under ``reports/``
names that id together with the segment's ledger start/end counts, look count
and code SHA.  A properly authorised later stage therefore verifies without
editing this module; an unauthorised, edited, reordered or deleted row does
not.  When a stage closes, it is registered in ``CLOSED_SEGMENTS`` in the same
commit that records its result.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re


LEDGER_PATH = Path("reports/experiment_results/test_look_ledger.jsonl")
AUTHORIZATION_ROOT = Path("reports")
BATCH_AUTHORIZATION_NAME = "batch_authorization.json"

PROTOCOL_B = "mechasight_experimental_protocol"
PROTOCOL_A = "official_baseline_reproduction"
FOCAL_TOGGLE = "roboticarm_focal_parent_sibling_toggle_v1"

REQUIRED_FIELDS = (
    "experiment_id", "git_sha", "machine", "protocol", "reason", "regime",
    "test_cohort_hash", "timestamp_utc",
)
_BATCH_IN_REASON = re.compile(r"(?:^|;\s*)batch=([A-Za-z0-9._-]+)")


class LedgerHistoryError(AssertionError):
    """The ledger is not an append-only extension of its authorised history."""


@dataclass(frozen=True)
class ClosedSegment:
    stage: str
    start: int
    end: int  # exclusive
    protocol: str
    git_sha: str
    timestamp_utc: str
    sha256: str  # SHA-256 of the LF bytes of rows[start:end]
    authorization: str  # committed record naming this segment's ledger counts


CLOSED_SEGMENTS: tuple[ClosedSegment, ...] = (
    ClosedSegment(
        "phase3_2_first_common_test", 0, 28, PROTOCOL_B,
        "99f6055156c314bbf26a847713d26792979fd050", "2026-09-03T09:06:51.648233+00:00",
        "db4bde80099a870ae90686fa463400f02f93434fc11dfce6d8c20f2eac0b0db0",
        "reports/phase3_2/evaluation_state.json",
    ),
    ClosedSegment(
        "phase3_3_protocol_a_reproduction", 28, 30, PROTOCOL_A,
        "4ce1faf5894578f330226d80cd194dfb44ff0cba", "2026-09-08T11:29:26.289247+00:00",
        "0fc5bc8e35af4727d8dda90ba7e46e5e91a24859e863f78e36539ce485fbe02d",
        "reports/phase3_3_protocol_a/protocol_a_ledger_authorization.json",
    ),
    ClosedSegment(
        "phase3_4_focal_parent_sibling_toggle", 30, 40, FOCAL_TOGGLE,
        "c10fdfcc890f97a595c132d87dc8b4bb04d7161e", "2026-09-11T19:49:04.480058+00:00",
        "93a0145c9997c9bed1fe2731d2486065541a0cd7ac2c8a2f45a8a36a8bc25f76",
        "reports/phase3_4_results/test_look_authorization.json",
    ),
    ClosedSegment(
        "phase4_single_modality_benchmark", 40, 196, PROTOCOL_B,
        "0021c93135a4d31800703520d5026bbed57f9b2e", "2026-09-13T17:39:18.591092+00:00",
        "26025ae1f0b8c438ac03568dc5527fd404aae2560c6970396694d1a7cb836906",
        "reports/phase4_protected/batch_authorization.json",
    ),
)

# Whole-prefix identities at the two stage boundaries later evidence refers to.
PHASE3_CLOSE_ROWS = 40
PHASE3_CLOSE_PREFIX_SHA256 = "b0ecbc7ac99edbc14aa4e5006b9210a42f7aa1a7c2045b2b80046ec853020d3e"
PHASE4_CLOSE_ROWS = 196
PHASE4_CLOSE_PREFIX_SHA256 = "f75584611c18ff387bee8c06f1fcc999c3046def876ba13dea1c744fce64de5f"


@dataclass(frozen=True)
class LedgerSegment:
    stage: str
    start: int
    end: int
    protocol: str
    git_sha: str
    timestamp_utc: str
    batch_id: str | None
    authorization: str
    closed: bool


@dataclass(frozen=True)
class LedgerHistory:
    lines: tuple[bytes, ...]
    rows: tuple[dict, ...]
    segments: tuple[LedgerSegment, ...]

    def segment(self, stage: str) -> LedgerSegment:
        matches = [s for s in self.segments if s.stage == stage]
        if len(matches) != 1:
            raise KeyError(stage)
        return matches[0]

    def rows_of(self, stage: str) -> tuple[dict, ...]:
        s = self.segment(stage)
        return self.rows[s.start:s.end]


def ledger_lines(path: str | Path = LEDGER_PATH) -> tuple[bytes, ...]:
    """LF-normalised ledger lines, each keeping its terminating newline."""

    raw = Path(path).read_bytes().replace(b"\r\n", b"\n")
    if raw and not raw.endswith(b"\n"):
        raise LedgerHistoryError("ledger does not end with a newline: a partial append")
    return tuple(raw.splitlines(keepends=True))


def prefix_sha256(lines: tuple[bytes, ...], rows: int) -> str:
    if len(lines) < rows:
        raise LedgerHistoryError(f"ledger has {len(lines)} rows, fewer than the {rows}-row history")
    return sha256(b"".join(lines[:rows])).hexdigest()


def _authorized_counts(record: dict) -> tuple[int | None, int | None]:
    """Ledger start/end counts under either historical authorization schema."""

    start = record.get("ledger_start_count", record.get("ledger_before"))
    end = record.get("ledger_end_count", record.get("ledger_after"))
    return start, end


def _check_authorization(path: Path, *, start: int, end: int, label: str) -> dict:
    if not path.is_file():
        raise LedgerHistoryError(f"{label}: authorization record {path} is missing")
    record = json.loads(path.read_text(encoding="utf-8"))
    if _authorized_counts(record) != (start, end):
        raise LedgerHistoryError(
            f"{label}: rows {start}..{end} disagree with the ledger counts in {path} "
            f"{_authorized_counts(record)}"
        )
    return record


def _batch_of(row: dict) -> str | None:
    match = _BATCH_IN_REASON.search(str(row.get("reason", "")))
    return match.group(1) if match else None


def _batch_authorizations(root: Path) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {}
    for path in sorted(root.rglob(BATCH_AUTHORIZATION_NAME)):
        batch_id = json.loads(path.read_text(encoding="utf-8")).get("batch_id")
        if batch_id:
            found.setdefault(str(batch_id), []).append(path)
    return found


def _tail_runs(rows: tuple[dict, ...], start: int) -> list[tuple[int, int]]:
    """Maximal runs of one atomic append: same code SHA, timestamp and batch."""

    runs = []
    index = start
    while index < len(rows):
        key = (rows[index]["git_sha"], rows[index]["timestamp_utc"], _batch_of(rows[index]))
        end = index + 1
        while end < len(rows) and (
            rows[end]["git_sha"], rows[end]["timestamp_utc"], _batch_of(rows[end])
        ) == key:
            end += 1
        runs.append((index, end))
        index = end
    return runs


def verify_ledger_history(
    ledger_path: str | Path = LEDGER_PATH,
    *,
    authorization_root: str | Path = AUTHORIZATION_ROOT,
    closed: tuple[ClosedSegment, ...] = CLOSED_SEGMENTS,
) -> LedgerHistory:
    """Prove the ledger is closed history plus authorised atomic appends, or raise."""

    root = Path(authorization_root)
    lines = ledger_lines(ledger_path)
    rows = []
    for number, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise LedgerHistoryError(f"ledger row {number} is not JSON") from error
        if not isinstance(row, dict) or any(field not in row for field in REQUIRED_FIELDS):
            raise LedgerHistoryError(f"ledger row {number} lacks the required look fields")
        rows.append(row)
    rows = tuple(rows)

    segments: list[LedgerSegment] = []
    expected_start = 0
    for seg in closed:
        if seg.start != expected_start:
            raise LedgerHistoryError(f"closed segment {seg.stage} is not contiguous")
        if len(rows) < seg.end:
            raise LedgerHistoryError(
                f"{seg.stage}: ledger has {len(rows)} rows; rows {seg.start}..{seg.end} were deleted"
            )
        if sha256(b"".join(lines[seg.start:seg.end])).hexdigest() != seg.sha256:
            raise LedgerHistoryError(f"{seg.stage}: rows {seg.start}..{seg.end} are not byte-identical")
        for row in rows[seg.start:seg.end]:
            if (row["protocol"], row["git_sha"], row["timestamp_utc"]) != (
                seg.protocol, seg.git_sha, seg.timestamp_utc
            ):
                raise LedgerHistoryError(f"{seg.stage}: a row is not part of the atomic append")
        _check_authorization(Path(seg.authorization), start=seg.start, end=seg.end, label=seg.stage)
        batches = {_batch_of(row) for row in rows[seg.start:seg.end]}
        segments.append(LedgerSegment(
            seg.stage, seg.start, seg.end, seg.protocol, seg.git_sha, seg.timestamp_utc,
            batches.pop() if len(batches) == 1 else None, seg.authorization, True,
        ))
        expected_start = seg.end

    authorizations = _batch_authorizations(root)
    for start, end in _tail_runs(rows, expected_start):
        first = rows[start]
        batch_id = _batch_of(first)
        label = f"rows {start}..{end}"
        if batch_id is None:
            raise LedgerHistoryError(f"{label}: appended look names no batch=<id>; unauthorised")
        paths = authorizations.get(batch_id, [])
        if len(paths) != 1:
            raise LedgerHistoryError(
                f"{label}: batch {batch_id} needs exactly one {BATCH_AUTHORIZATION_NAME}, found {len(paths)}"
            )
        record = _check_authorization(paths[0], start=start, end=end, label=label)
        if record.get("experiment_count") != end - start:
            raise LedgerHistoryError(f"{label}: look count differs from the authorised experiment_count")
        if record.get("code_git_sha") != first["git_sha"]:
            raise LedgerHistoryError(f"{label}: code SHA differs from the authorised code_git_sha")
        segments.append(LedgerSegment(
            f"authorized_batch:{batch_id}", start, end, first["protocol"], first["git_sha"],
            first["timestamp_utc"], batch_id, paths[0].as_posix(), False,
        ))

    ids = [row["experiment_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise LedgerHistoryError("an experiment id was looked at more than once")
    return LedgerHistory(lines, rows, tuple(segments))
