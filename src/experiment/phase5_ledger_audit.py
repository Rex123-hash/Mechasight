"""Independent, read-only integrity audit of the test-look ledger. DETECTION ONLY; it prevents nothing.

It does not import or trust the ledger writer. It recomputes everything from the bytes on disk and checks
them against anchors held outside the ledger file:

1. the closed 196-row history, against the SHA-256 constant in the reviewed source;
2. the frozen ``verify_ledger_history``: closed segments, and every later run authorized by exactly one
   ``batch_authorization.json`` with matching start/end counts;
3. for every appended batch, the SHA-256 of the ledger before and after it, which its authorization
   recorded before the write. Any later edit, insertion, deletion or replacement of rows up to that
   batch's end changes these digests;
4. the Git anchor: the ledger committed at ``HEAD`` must be a byte prefix of the working ledger. The
   protocol commits the ledger and its batch records after every protected operation, so a removed or
   rewritten committed row is detected even when every local record has been rewritten consistently;
5. the total look cap, and that every appended row belongs to the protected scope.

Anyone with write access to the ledger, the batch records and the Git repository could forge all of
these anchors consistently. Detecting that needs an external copy (the Git bundle backups). The audit
reports what it checked and fails closed on the first mismatch.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
import stat

from src.experiment.ledger_history import BATCH_AUTHORIZATION_NAME, LedgerHistoryError, ledger_lines, verify_ledger_history
from src.experiment.phase5_inputs import Refused, require

_BATCH = re.compile(r"(?:^|;\s*)batch=([A-Za-z0-9._-]+)")


def _digest(lines, rows):
    return sha256(b"".join(lines[:rows])).hexdigest()


def _authorizations(root):
    found = {}
    for path in sorted(Path(root).rglob(BATCH_AUTHORIZATION_NAME)):
        record = json.loads(path.read_text(encoding="utf-8"))
        found.setdefault(str(record.get("batch_id")), []).append(record)
    return found


def audit_ledger(path, *, authorization_root, baseline_rows, baseline_sha256, max_total_rows, scope_marker,
                 committed_bytes=None, require_committed_anchor=False):
    """Recompute the ledger's integrity from its bytes and the external anchors. Raises ``Refused`` on any mismatch."""
    path = Path(path)
    try:
        lines = ledger_lines(path)
        history = verify_ledger_history(path, authorization_root=authorization_root)
    except (OSError, ValueError, LedgerHistoryError) as exc:
        raise Refused(f"ledger integrity: unreadable, partial or unauthorized history ({exc})") from exc
    require(len(lines) >= baseline_rows and _digest(lines, baseline_rows) == baseline_sha256,
            "ledger integrity: the closed 196-row history differs from its reviewed SHA-256")
    require(len(lines) <= max_total_rows, f"ledger integrity: {len(lines)} rows exceed the {max_total_rows}-row look cap")
    data = b"".join(lines)
    committed_rows = None
    if committed_bytes is None:
        require(not require_committed_anchor, "ledger integrity: the ledger is not committed in the records repository")
    else:
        committed = committed_bytes.replace(b"\r\n", b"\n")
        committed_rows = len(committed.splitlines())
        require(data.startswith(committed),
                "ledger integrity: the working ledger does not extend the committed ledger (committed rows were "
                "removed, rewritten or the file was replaced)")
    records = _authorizations(authorization_root)
    batches, i = [], baseline_rows
    while i < len(lines):
        row = json.loads(lines[i])
        match = _BATCH.search(str(row.get("reason", "")))
        require(match is not None and f"scope={scope_marker}" in str(row.get("reason", "")),
                f"ledger integrity: row {i + 1} is not a {scope_marker} batch look")
        batch, j = match.group(1), i
        while j < len(lines) and (m := _BATCH.search(str(json.loads(lines[j]).get("reason", "")))) and m.group(1) == batch:
            j += 1
        found = records.get(batch, [])
        require(len(found) == 1, f"ledger integrity: batch {batch} needs exactly one authorization record")
        record = found[0]
        require((record.get("ledger_start_count"), record.get("ledger_end_count")) == (i, j)
                and record.get("ledger_sha256_before") == _digest(lines, i)
                and record.get("ledger_sha256_after") == _digest(lines, j),
                f"ledger integrity: batch {batch} rows {i}..{j} differ from the ledger bytes its authorization recorded")
        batches.append({"batch_id": batch, "start": i, "end": j})
        i = j
    present = {b["batch_id"] for b in batches}
    aborted = sorted(b for b in records if b not in present)
    for batch in aborted:
        record = records[batch][0]
        # An authorization whose append never became durable (a crash before the swap): it must describe a
        # ledger state that the current ledger extends, and counts no look.
        start = record.get("ledger_start_count")
        require(isinstance(start, int) and start <= len(lines) and record.get("ledger_sha256_before") == _digest(lines, start),
                f"ledger integrity: authorization {batch} names a ledger history that no longer exists")
    mode = path.stat().st_mode
    return {"rows": len(lines), "ledger_sha256": sha256(data).hexdigest(), "baseline_rows": baseline_rows,
            "batches": batches, "authorizations_without_rows": aborted,
            "committed_anchor": committed_bytes is not None, "committed_rows": committed_rows, "sealed": not (mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)),
            "frozen_history_rows": len(history.rows), "detection_only": True}
