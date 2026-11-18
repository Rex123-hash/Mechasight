"""Exposure-ledger backends for the Phase-5 executor, and the single writer of the test-look ledger.

``JsonLedger``: the 5.5B-5.5E synthetic JSON-array ledger (unchanged).

``JsonlLedger`` / ``ProtectedJsonlLedger``: the project's append-only JSONL test-look ledger, in the
frozen ``TestLookLedger`` line format. ``JsonlLedger.append`` is the ONLY code that writes ledger bytes,
and it begins with every check; there is no public byte-level writer.

One append (``append`` -> ``_commit``), entirely under ``ledger_lock`` (a cross-process OS lock on
``<ledger>.lock``; the protected drivers hold it for the whole operation):

1. protected scope: the running operation's authority, re-derived from the committed records
   (``phase5_protected.require_write_authority``);
2. the ledger bytes, read once, verified (frozen ``verify_ledger_history``, batch validation including
   each batch's before/after digests) and checked unchanged after verification;
3. the expected start count and the look budget (660 total) on those fresh rows;
4. a batch authorization record naming the start/end counts and the SHA-256 of the ledger before
   and after the append (written first: the frozen verifier refuses an unauthorized run);
5. the next ledger written to a pending file, sealed like the ledger, and atomically swapped in.
   The ledger path names a complete file at every instant;
6. the result verified byte for byte. A failure before the rows are durable withdraws the
   authorization (``*.aborted-<id>.json``).

The real ledger is SEALED (read-only) at rest, so the operating system refuses every ordinary write:
the frozen ``TestLookLedger.append`` / ``append_many_atomic``, their callers, ``open(path, "a")``.
That is the only thing the seal enforces. Any same-user process can still rename, replace or delete
the file, or clear the attribute. Such changes are outside prevention; ``phase5_ledger_audit``
detects them.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import threading
import time
import uuid

from src.experiment.ledger import TestLookRecord
from src.experiment.ledger_history import (
    BATCH_AUTHORIZATION_NAME, LEDGER_PATH, LedgerHistoryError, PROTOCOL_B, ledger_lines, verify_ledger_history,
)
from src.experiment.phase5_inputs import SYNTHETIC, Refused, require

PROTECTED_SCOPE = "PROTECTED_COMMON_EVALUATION"
from src.experiment.provenance import content_sha256

_ORIGINAL = re.compile(r"(?:^|;\s*)original=([A-Za-z0-9._-]+)")
_RETRY = re.compile(r"^(?P<original>.+)-r(?P<k>[1-9][0-9]*)$")
AUTHORIZATION_KEYS = {"scope", "batch_id", "ledger_start_count", "ledger_end_count", "experiment_count",
                      "code_git_sha", "reservation_sha256", "ledger_sha256_before", "ledger_sha256_after"}


def look_budget(package):
    """Trusted look budget, derived only from the frozen evaluation proposal.

    The proposal's accounting fixes the baseline (196) and the first batch (232 looks,
    ending at 428). The readiness plan proposes a hard cap of 428 + 232 = 660 as a
    TOTAL ledger length (H4; not authorized), i.e. at most 464 looks above the
    baseline, retries included. No caller argument or fixture field can change it.
    """
    ledger = package.proposed["accounting"]["ledger"]
    baseline, first_end, initial = ledger["expected_start"], ledger["expected_end"], ledger["rows_appended"]
    require((baseline, first_end, initial) == (196, 428, 232) and baseline + initial == first_end,
            "frozen ledger accounting differs from the proposal")
    total_cap = first_end + initial
    return {"baseline_rows": baseline, "max_total_rows": total_cap, "max_additional_looks": total_cap - baseline}
_BATCH = re.compile(r"(?:^|;\s*)batch=([A-Za-z0-9._-]+)")
METRICS_REQUESTED = ("auroc", "average_precision", "standardized_pauroc_fpr_0_10", "threshold_metrics")

# ---------------------------------------------------------------- cross-process ledger lock (B4)
LEDGER_LOCK_SUFFIX = ".lock"
LEDGER_LOCK_TIMEOUT_SECONDS = 120.0
_HELD = {}                      # (lock path, thread) -> depth: re-entrant within one thread only
_HELD_GUARD = threading.Lock()


def ledger_lock_path(path):
    """The lock file of a ledger: created once beside it, never moved, replaced or deleted."""
    return Path(str(Path(path)) + LEDGER_LOCK_SUFFIX)


def _os_lock(handle):
    handle.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _os_unlock(handle):
    handle.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def ledger_lock(path, *, timeout=None):
    """Exclusive OS byte-range lock on ``<ledger>.lock``, held on an open handle.

    The OS releases it when the holder exits or crashes, so it never goes stale. A contender
    waits up to ``timeout`` seconds (bounded), then is refused before any mutation. Re-entrant
    only for the same thread; another thread or process always contends for the OS lock.
    """
    lock = ledger_lock_path(path)
    lock.parent.mkdir(parents=True, exist_ok=True)
    key = (os.path.normcase(str(lock.resolve())), threading.get_ident())
    with _HELD_GUARD:
        depth = _HELD.get(key, 0)
        if depth:
            _HELD[key] = depth + 1
    if depth:
        try:
            yield lock
        finally:
            with _HELD_GUARD:
                _HELD[key] -= 1
        return
    handle = open(lock, "a+b")
    try:
        deadline = time.monotonic() + (LEDGER_LOCK_TIMEOUT_SECONDS if timeout is None else timeout)
        while True:
            try:
                _os_lock(handle)
                break
            except OSError as exc:
                if time.monotonic() >= deadline:
                    raise Refused("the ledger lock is held by another writer; refused before any mutation") from exc
                time.sleep(0.02)
        with _HELD_GUARD:
            _HELD[key] = 1
        try:
            yield lock
        finally:
            with _HELD_GUARD:
                _HELD.pop(key, None)
            _os_unlock(handle)
    finally:
        handle.close()


def ledger_lock_held(path):
    lock = ledger_lock_path(path)
    key = (os.path.normcase(str(lock.resolve())), threading.get_ident())
    with _HELD_GUARD:
        return _HELD.get(key, 0) > 0


# ---------------------------------------------------------------- the seal and the atomic replace
PENDING_TAG = ".pending-"
SWAP_RETRY_SECONDS = 10.0
_WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


def ledger_sealed(path):
    """True when the ledger is a regular file with no write permission (Windows: read-only attribute)."""
    path = Path(path)
    return path.is_file() and not (path.stat().st_mode & _WRITE_BITS)


def seal_ledger(path):
    """Make the ledger file read-only. Content is untouched. For the real ledger this is a human step."""
    os.chmod(path, 0o444)
    require(ledger_sealed(path), "the ledger could not be sealed read-only")


def _unlink_sealed(path):
    os.chmod(path, 0o644)
    os.unlink(path)


def _pending_files(path):
    path = Path(path)
    return sorted(path.parent.glob(path.name + PENDING_TAG + "*"))


def _replace_sealed(source, target):
    """Atomically replace ``target`` by ``source`` although both are read-only.

    The ledger path names a complete, sealed file at every instant: there is no moment at which it is
    missing or writable. POSIX ``rename`` replaces atomically whatever the target's mode. Windows needs
    ``SetFileInformationByHandle(FileRenameInfoEx)`` with REPLACE_IF_EXISTS | POSIX_SEMANTICS |
    IGNORE_READONLY_ATTRIBUTE (Windows 10 1809 or later; refused, never emulated, on older systems).
    """
    if os.name != "nt":
        os.replace(source, target)
        return
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = wintypes.HANDLE
    k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
                                wintypes.DWORD, wintypes.HANDLE]
    k32.SetFileInformationByHandle.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.SetFileInformationByHandle.restype = wintypes.BOOL
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    delete_synchronize, share_all, open_existing, file_rename_info_ex = 0x00010000 | 0x00100000, 0x7, 3, 22
    handle = k32.CreateFileW(str(Path(source).resolve()), delete_synchronize, share_all, None, open_existing, 0, None)
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        name = str(Path(target).resolve())
        size = len(name) * 2
        info = ctypes.create_string_buffer(20 + size + 2)        # FILE_RENAME_INFO: Flags, RootDirectory, Length, Name
        ctypes.c_uint32.from_buffer(info, 0).value = 0x1 | 0x2 | 0x40
        ctypes.c_uint32.from_buffer(info, 16).value = size
        ctypes.memmove(ctypes.addressof(info) + 20, ctypes.create_unicode_buffer(name), size)
        if not k32.SetFileInformationByHandle(handle, file_rename_info_ex, info, len(info)):
            error = ctypes.get_last_error()
            if error in (1, 87, 124):                            # not supported by this Windows version
                raise Refused("the sealed ledger needs an atomic read-only replace (Windows 10 1809 or later)")
            raise ctypes.WinError(error)
    finally:
        k32.CloseHandle(handle)


def _replace_bounded(source, target):
    """A reader holding the ledger open (sharing violation) delays the replace; bounded, then refused."""
    deadline = time.monotonic() + SWAP_RETRY_SECONDS
    while True:
        try:
            _replace_sealed(source, target)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.02)


def _fsync_dir(directory):
    if os.name != "nt":
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _discard_pending(path):
    """Under the ledger lock: delete pending next-ledger files a crash left behind (never the ledger)."""
    for item in _pending_files(path):
        _unlink_sealed(item)


def _same_file(a, b):
    a, b = Path(a), Path(b)
    try:
        return a.resolve() == b.resolve() or (a.exists() and b.exists() and os.path.samefile(a, b))
    except OSError:
        return False


def ledger_digest(lines, rows):
    """SHA-256 of the first ``rows`` LF-normalised ledger lines."""
    return sha256(b"".join(lines[:rows])).hexdigest()


class JsonLedger:
    """5.5B-5.5E synthetic JSON-array ledger in the sandbox (``ledger.json``)."""

    kind = "json"

    def __init__(self, executor, spec):
        self.executor, self.base_rows = executor, int(spec.get("base_rows", 196))

    def rows(self):
        try:
            return self.executor.sandbox.json("ledger.json")
        except (OSError, ValueError) as exc:
            raise Refused("synthetic ledger missing or unreadable; human incident resolution required") from exc

    def check_foreign(self, rows):
        require(all(r["scope"] == SYNTHETIC for r in rows), "foreign ledger refused")

    def prefix_digest(self, rows, end):
        return content_sha256(rows[:end])

    def precheck(self, rows, attempt_ids):
        """5.5B synthetic JSON test ledger: identities only; the look budget is modelled by JsonlLedger."""
        require(not set(attempt_ids.values()) & {r["experiment_id"] for r in rows}, "duplicate evaluation attempt identity")

    def append(self, permit, pending, attempt_ids, start):
        rows = self.rows()
        rows += [{"scope": SYNTHETIC, "batch_id": permit.batch_id, "experiment_id": attempt_ids[eid],
                  "original_experiment_id": eid, "state": "EXPOSED"} for eid in pending]
        self.executor._write("ledger.json", rows, replace=True)
        return rows

    def verify(self):
        rows = self.rows()
        self.check_foreign(rows)
        require(len({r["experiment_id"] for r in rows}) == len(rows), "duplicate ledger identities")
        return rows


class JsonlLedger:
    """Established append-only JSONL test-look ledger, on a temporary synthetic copy."""

    kind = "jsonl"
    record_scope = SYNTHETIC
    reason_prefix = "scope=SYNTHETIC phase5.5F integration look"
    require_sealed = False                  # a synthetic copy may be sealed (seed_jsonl_ledger(sealed=True))

    def __init__(self, executor, spec):
        self.executor = executor
        self.path = executor.sandbox.path(spec["path"])
        self.authorizations = executor.sandbox.path(spec["authorization_root"])
        from src.experiment import phase5_protected as pp
        require(not _same_file(self.path, LEDGER_PATH) and not _same_file(self.path, pp.REAL_LEDGER),
                "the real protected ledger is never a synthetic executor backend (also not through an alias)")
        self.budget = look_budget(executor.package)
        self.base_rows = int(spec["base_rows"])
        require(self.base_rows == self.budget["baseline_rows"], "ledger baseline differs from the frozen accounting")
        self.code_git_sha = spec["code_git_sha"]

    def _raw(self):
        try:
            return ledger_lines(self.path)
        except (OSError, LedgerHistoryError) as exc:
            raise Refused(f"JSONL ledger unreadable or partially appended ({exc}); human incident resolution required") from exc

    def rows(self):
        out = []
        for line in self._raw():
            row = json.loads(line)
            reason = str(row.get("reason", ""))
            batch = _BATCH.search(reason)
            original = _ORIGINAL.search(reason)
            out.append({"experiment_id": row["experiment_id"], "batch_id": batch.group(1) if batch else None,
                        "original_experiment_id": original.group(1) if original else row["experiment_id"],
                        "scope": (SYNTHETIC if "scope=SYNTHETIC" in reason else
                                  PROTECTED_SCOPE if f"scope={PROTECTED_SCOPE}" in reason else "HISTORICAL")})
        return out

    def check_foreign(self, rows):
        require(all(r["scope"] == self.record_scope for r in rows[self.base_rows:]), "foreign rows appended to this ledger")

    def prefix_digest(self, rows, end):
        return sha256(b"".join(self._raw()[:end])).hexdigest()

    def verify(self):
        """The unmodified project history verifier over this synthetic copy."""
        try:
            verify_ledger_history(self.path, authorization_root=self.authorizations)
        except (LedgerHistoryError, OSError, ValueError) as exc:
            raise Refused(f"JSONL ledger history verification failed: {exc}") from exc
        rows = self.rows()
        self.check_foreign(rows)
        self.validate_batches(rows)
        return rows

    def precheck(self, rows, attempt_ids):
        """Look-budget and identity check, run BEFORE any reservation, authorization, append or read.

        ``rows`` is the verified current ledger (every previously exposed look, retries
        included); ``attempt_ids`` the pending batch. Enforces
        ``existing_total + pending - baseline <= max_additional_looks``.
        """
        b, ex = self.budget, self.executor
        pending = list(attempt_ids.values())
        require(len(rows) >= b["baseline_rows"], "ledger is shorter than its closed baseline")
        require(pending and len(set(pending)) == len(pending), "empty batch or duplicate pending identities")
        require(not set(pending) & {r["experiment_id"] for r in rows}, "duplicate evaluation attempt identity")
        for original, attempt in attempt_ids.items():
            retry = _RETRY.fullmatch(attempt)
            require(original in ex.operations and (attempt == original or (retry is not None and retry["original"] == original)),
                    "pending look is not a planned unit or a well-formed retry identity")
        additional = len(rows) + len(pending) - b["baseline_rows"]
        require(additional <= b["max_additional_looks"],
                f"look budget exceeded before any mutation: {len(rows)} existing + {len(pending)} pending - "
                f"{b['baseline_rows']} baseline = {additional} > {b['max_additional_looks']} additional looks "
                f"(max total {b['max_total_rows']}); incident review and new authorization required")

    def validate_batches(self, rows):
        """Phase 5.6C batch interface: every appended run is one authorized, reserved, planned batch.

        Beyond ``verify_ledger_history`` (counts, look count, code SHA): the record has
        exactly the synthetic schema and the configured code SHA; it is bound to the
        executor's durable reservation; the run's identities are that reservation's;
        every look is a planned unit or a ``-r<k>`` retry of one (reused comparators
        never consume a look); the first batch is exactly 24 L1 + 208 L2; a batch id is
        never reused; and the ledger stays within the proposed total cap (detection;
        prevention is ``precheck``).
        """
        ex = self.executor
        tail = rows[self.base_rows:]
        lines = self._raw()
        require(len(rows) <= self.budget["max_total_rows"], "ledger exceeds the proposed look cap")
        seen, i = set(), 0
        while i < len(tail):
            batch = tail[i]["batch_id"]
            j = i
            while j < len(tail) and tail[j]["batch_id"] == batch:
                j += 1
            require(batch is not None and batch not in seen, "batch identity missing or reused")
            seen.add(batch)
            start, end, run = self.base_rows + i, self.base_rows + j, tail[i:j]
            try:
                record = json.loads((self.authorizations / batch / BATCH_AUTHORIZATION_NAME).read_text(encoding="utf-8"))
                reservation = ex.sandbox.json(f"reservations/{batch}.json")
                reservation_hash = ex._reservation_hash(batch)
            except (OSError, ValueError) as exc:
                raise Refused("batch authorization or reservation missing/unreadable") from exc
            require(set(record) == AUTHORIZATION_KEYS and record["scope"] == self.record_scope and record["batch_id"] == batch
                    and (record["ledger_start_count"], record["ledger_end_count"], record["experiment_count"])
                    == (start, end, end - start) and record["code_git_sha"] == self.code_git_sha,
                    "batch authorization record mismatch")
            require(record["ledger_sha256_before"] == ledger_digest(lines, start)
                    and record["ledger_sha256_after"] == ledger_digest(lines, end),
                    "the ledger bytes differ from those its batch authorization recorded (modified or replaced)")
            require(record["reservation_sha256"] == reservation_hash and reservation["batch_id"] == batch
                    and (reservation["ledger_start"], reservation["ledger_end"]) == (start, end),
                    "batch authorization is not bound to its durable reservation")
            ids = [r["experiment_id"] for r in run]
            require(sorted(ids) == sorted(reservation["attempt_ids"].values()), "ledger run differs from reserved identities")
            operations = []
            for row in run:
                original, attempt = row["original_experiment_id"], row["experiment_id"]
                retry = _RETRY.fullmatch(attempt)
                require(original in ex.operations and reservation["attempt_ids"].get(original) == attempt
                        and (attempt == original or (retry is not None and retry["original"] == original)),
                        "look is not a planned unit or a well-formed retry identity")
                operations.append(ex.operations[original]["operation"])
            if i == 0:
                require(sorted(operations) == sorted(["new_L1_inference"] * 24 + ["derived_L2_scoring"] * 208),
                        "the scoring batch must be exactly 24 L1 + 208 L2 looks")
            i = j

    def _locked_checks(self, permit):
        """Authority checks under the lock (the protected backend re-derives the running operation's authority)."""

    def _authorization_path(self, batch_id):
        return self.authorizations / batch_id / BATCH_AUTHORIZATION_NAME

    def _withdraw_authorization(self, batch_id, start):
        """After a failed append: withdraw this operation's authorization if none of its rows is durable."""
        path = self._authorization_path(batch_id)
        try:
            appended = len(self._raw()) != start
        except Refused:
            appended = True                     # unreadable/partial ledger: keep the record for the incident
        if not appended and path.is_file():
            os.replace(path, path.with_name(f"batch_authorization.aborted-{uuid.uuid4().hex[:12]}.json"))

    def append(self, permit, pending, attempt_ids, start):
        """The single writer: every check and the write, under one cross-process lock."""
        with ledger_lock(self.path):
            return self._commit(permit, pending, attempt_ids, start)

    def _commit(self, permit, pending, attempt_ids, start):
        # The checks live in the only code that writes ledger bytes: calling this directly cannot skip them.
        from dataclasses import asdict
        from src.experiment.provenance import canonical_json
        ex = self.executor
        require(ledger_lock_held(self.path), "the ledger is written only under the ledger lock")
        self._locked_checks(permit)
        batch = {eid: attempt_ids[eid] for eid in pending}
        _discard_pending(self.path)
        sealed = ledger_sealed(self.path)
        require(sealed or not self.require_sealed,
                "the real ledger is not sealed read-only; protected writes need the sealed ledger")
        old = b"".join(self._raw())
        require(self.path.read_bytes() == old, "the ledger is not in its LF line form; human incident resolution required")
        rows = self.verify()
        require(self.path.read_bytes() == old, "the ledger changed while it was being verified; refused before any mutation")
        self.precheck(rows, batch)                                     # the 660 budget on the fresh, verified rows
        require(len(rows) == start, "ledger changed between reservation and append")
        require(not self._authorization_path(permit.batch_id).exists(),
                "a batch authorization for this batch already exists; human incident resolution required")
        stamp = datetime.now(timezone.utc).isoformat()
        records = []
        for eid in pending:
            identity = ex._look_identity(eid)
            records.append(TestLookRecord(
                timestamp_utc=stamp, experiment_id=attempt_ids[eid], git_sha=self.code_git_sha,
                machine=identity["machine"], protocol=PROTOCOL_B, regime=identity["regime"],
                model_config_identity=identity["model_config_identity"], test_cohort_hash=identity["cohort_hash"],
                reason=(f"{self.reason_prefix}; batch={permit.batch_id}; original={eid}; "
                        f"operation={identity['operation']}"),
                metrics_requested=METRICS_REQUESTED))
        data = old + "".join(canonical_json(asdict(r)) + "\n" for r in records).encode("utf-8")
        # Authorization first (the frozen verifier refuses an appended run without one); it names the exact
        # ledger bytes before and after this append, which every later verification and the audit re-check.
        self._write_authorization(permit.batch_id, {
            "scope": self.record_scope, "batch_id": permit.batch_id, "ledger_start_count": start,
            "ledger_end_count": start + len(pending), "experiment_count": len(pending), "code_git_sha": self.code_git_sha,
            "reservation_sha256": ex._reservation_hash(permit.batch_id),
            "ledger_sha256_before": sha256(old).hexdigest(), "ledger_sha256_after": sha256(data).hexdigest()})
        try:
            ex._hook("after_batch_authorization")
            pending_file = self.path.with_name(self.path.name + PENDING_TAG + uuid.uuid4().hex)
            try:
                with open(pending_file, "xb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                if sealed:
                    seal_ledger(pending_file)
                ex._hook("before_ledger_swap")
                _replace_bounded(pending_file, self.path)         # atomic; the path always names a complete ledger
            except BaseException:
                if pending_file.exists():
                    _unlink_sealed(pending_file)                 # a failed or torn pending file never becomes the ledger
                raise
            _fsync_dir(self.path.parent)
            ex._hook("after_ledger_swap")
            require(self.path.read_bytes() == data and ledger_sealed(self.path) == sealed,
                    "durable ledger append verification failed; human incident resolution required")
        except BaseException:
            self._withdraw_authorization(permit.batch_id, start)
            raise
        return self.rows()

    def _write_authorization(self, batch_id, payload):
        ex = self.executor
        ex._write(f"{self.authorizations.relative_to(ex.sandbox.root).as_posix()}/{batch_id}/{BATCH_AUTHORIZATION_NAME}", payload)


class ProtectedJsonlLedger(JsonlLedger):
    """A3/A6: the real test-look ledger, behind the reviewed, authorized activation only.

    Identical checks to JsonlLedger (unmodified verify_ledger_history, batch validation, the
    660-total precheck before any mutation and its re-check under the attempt lock). Rows carry
    ``scope=PROTECTED_COMMON_EVALUATION``; batch authorization records live in the committed
    ``reports/phase5_protected/batches/<batch>/batch_authorization.json``.
    """

    kind = "protected_jsonl"
    record_scope = None                     # set from the protected scope in __init__
    reason_prefix = None
    require_sealed = True                   # the real ledger is written only while sealed

    def __init__(self, executor, spec):
        from src.experiment import phase5_protected as pp
        require(executor.scope == pp.PROTECTED, "the real ledger is written only by an authorized protected batch")
        pp.verified_activation()                                     # fails closed; re-verified again at append
        require(ledger_sealed(pp.REAL_LEDGER), "the real ledger is not sealed read-only")
        self.executor = executor
        self.record_scope = pp.PROTECTED
        self.reason_prefix = f"scope={pp.PROTECTED} phase5.7 protected look"
        self.path = Path(pp.REAL_LEDGER)
        self.authorizations = Path(pp.BATCH_AUTH_ROOT)
        require(Path(spec["authorization_root"]) == self.authorizations, "unregistered batch-authorization root")
        self.budget = look_budget(executor.package)
        self.base_rows = int(spec["base_rows"])
        require(self.base_rows == self.budget["baseline_rows"], "ledger baseline differs from the frozen accounting")
        self.code_git_sha = spec["code_git_sha"]

    def _locked_checks(self, permit):
        # Under the ledger lock: the running operation's authority for this batch, re-derived now.
        from src.experiment import phase5_protected as pp
        pp.require_write_authority(self.executor, permit)

    def _write_authorization(self, batch_id, payload):
        from src.experiment.phase5_inputs import atomic_json
        atomic_json(self.authorizations / batch_id / BATCH_AUTHORIZATION_NAME, payload)


def append_protected_batch(*args, **kwargs):
    """The real protected ledger batch mode stays disabled (Phase 5.6C)."""
    raise Refused("real protected ledger batches are disabled: human authorization H1-H7, an independent review "
                  "and a committed batch authorization record are required")


def ledger_backend(executor):
    spec = executor.fixture.get("ledger", {"kind": "json", "base_rows": 196})
    require(spec.get("kind") in {"json", "jsonl", "protected_jsonl"}, "unknown ledger backend")
    return {"json": JsonLedger, "jsonl": JsonlLedger, "protected_jsonl": ProtectedJsonlLedger}[spec["kind"]](executor, spec)


def seed_jsonl_ledger(sandbox, *, real_ledger=LEDGER_PATH, rows=196, sealed=False):
    """Copy the closed 196-row history (LF form) into a sandbox; the real file is only read."""

    lines = ledger_lines(real_ledger)
    require(len(lines) == rows, "the real ledger is not at its closed 196-row state")
    target = sandbox.path("ledger/test_look_ledger.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"".join(lines))
    if sealed:
        seal_ledger(target)
    sandbox.path("ledger/authorizations").mkdir(parents=True, exist_ok=True)
    return {"kind": "jsonl", "path": "ledger/test_look_ledger.jsonl", "authorization_root": "ledger/authorizations",
            "base_rows": rows, "base_sha256": sha256(b"".join(lines)).hexdigest()}
