"""Phase 5.5F: exposure-ledger backends for the Phase-5 executor.

``JsonLedger`` is the 5.5B-5.5E synthetic JSON-array ledger (unchanged behaviour).

``JsonlLedger`` writes the project's established protected test-look ledger
format and nothing else:

* every exposure is a ``TestLookRecord`` appended with the existing
  ``TestLookLedger.append_many_atomic`` (one durable write, fsync, count check);
* every batch first gets exactly one ``batch_authorization.json`` naming its
  ``batch_id``, ledger start/end counts, look count and code SHA, and every row's
  ``reason`` carries ``batch=<id>`` — the convention ``verify_ledger_history``
  enforces for any append after the closed segments;
* history is checked with the *unmodified* ``verify_ledger_history`` (closed
  segments byte-identical, authorized atomic tail runs, globally unique ids).

It is only ever pointed at a temporary SYNTHETIC copy of the ledger; the real
ledger path is refused.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re

from src.experiment.ledger import TestLookLedger, TestLookRecord
from src.experiment.ledger_history import (
    BATCH_AUTHORIZATION_NAME, LEDGER_PATH, LedgerHistoryError, PROTOCOL_B, ledger_lines, verify_ledger_history,
)
from src.experiment.phase5_inputs import SYNTHETIC, Refused, require
from src.experiment.provenance import content_sha256

_ORIGINAL = re.compile(r"(?:^|;\s*)original=([A-Za-z0-9._-]+)")
_RETRY = re.compile(r"^(?P<original>.+)-r(?P<k>[1-9][0-9]*)$")
AUTHORIZATION_KEYS = {"scope", "batch_id", "ledger_start_count", "ledger_end_count", "experiment_count",
                      "code_git_sha", "reservation_sha256"}


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

    def __init__(self, executor, spec):
        self.executor = executor
        self.path = executor.sandbox.path(spec["path"])
        self.authorizations = executor.sandbox.path(spec["authorization_root"])
        require(self.path.resolve() != LEDGER_PATH.resolve(), "the real protected ledger is never an executor backend")
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
                        "scope": SYNTHETIC if "scope=SYNTHETIC" in reason else "HISTORICAL"})
        return out

    def check_foreign(self, rows):
        require(all(r["scope"] == SYNTHETIC for r in rows[self.base_rows:]), "foreign rows appended to the synthetic ledger")

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
            require(set(record) == AUTHORIZATION_KEYS and record["scope"] == SYNTHETIC and record["batch_id"] == batch
                    and (record["ledger_start_count"], record["ledger_end_count"], record["experiment_count"])
                    == (start, end, end - start) and record["code_git_sha"] == self.code_git_sha,
                    "batch authorization record mismatch")
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

    def append(self, permit, pending, attempt_ids, start):
        ex = self.executor
        end = start + len(pending)
        # Re-check on a fresh read under the executor's exclusive attempt lock, so a stale
        # precheck cannot authorize: nothing is published if the ledger moved or the budget fails.
        rows = self.verify()
        require(len(rows) == start, "ledger changed between reservation and append")
        self.precheck(rows, {eid: attempt_ids[eid] for eid in pending})
        # Authorization first: verify_ledger_history refuses any appended run without one.
        ex._write(f"{self.authorizations.relative_to(ex.sandbox.root).as_posix()}/{permit.batch_id}/{BATCH_AUTHORIZATION_NAME}",
                  {"scope": SYNTHETIC, "batch_id": permit.batch_id, "ledger_start_count": start,
                   "ledger_end_count": end, "experiment_count": len(pending), "code_git_sha": self.code_git_sha,
                   "reservation_sha256": ex._reservation_hash(permit.batch_id)})
        ex._hook("after_batch_authorization")
        stamp = datetime.now(timezone.utc).isoformat()
        records = []
        for eid in pending:
            identity = ex._look_identity(eid)
            records.append(TestLookRecord(
                timestamp_utc=stamp, experiment_id=attempt_ids[eid], git_sha=self.code_git_sha,
                machine=identity["machine"], protocol=PROTOCOL_B, regime=identity["regime"],
                model_config_identity=identity["model_config_identity"], test_cohort_hash=identity["cohort_hash"],
                reason=(f"scope=SYNTHETIC phase5.5F integration look; batch={permit.batch_id}; original={eid}; "
                        f"operation={identity['operation']}"),
                metrics_requested=METRICS_REQUESTED))
        TestLookLedger(self.path).append_many_atomic(records, expected_existing_count=start)
        return self.rows()


def append_protected_batch(*args, **kwargs):
    """The real protected ledger batch mode stays disabled (Phase 5.6C)."""
    raise Refused("real protected ledger batches are disabled: human authorization H1-H7, an independent review "
                  "and a committed batch authorization record are required")


def ledger_backend(executor):
    spec = executor.fixture.get("ledger", {"kind": "json", "base_rows": 196})
    require(spec.get("kind") in {"json", "jsonl"}, "unknown ledger backend")
    return (JsonlLedger if spec["kind"] == "jsonl" else JsonLedger)(executor, spec)


def seed_jsonl_ledger(sandbox, *, real_ledger=LEDGER_PATH, rows=196):
    """Copy the closed 196-row history (LF form) into a sandbox; the real file is only read."""

    lines = ledger_lines(real_ledger)
    require(len(lines) == rows, "the real ledger is not at its closed 196-row state")
    target = sandbox.path("ledger/test_look_ledger.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"".join(lines))
    sandbox.path("ledger/authorizations").mkdir(parents=True, exist_ok=True)
    return {"kind": "jsonl", "path": "ledger/test_look_ledger.jsonl", "authorization_root": "ledger/authorizations",
            "base_rows": rows, "base_sha256": sha256(b"".join(lines)).hexdigest()}
