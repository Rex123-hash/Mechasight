"""Append-only audit ledger for future real test-set evaluations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path

from src.experiment.constants import TEST_LOOK_SCHEMA_VERSION
from src.experiment.provenance import canonical_json


@dataclass(frozen=True)
class TestLookRecord:
    __test__ = False

    timestamp_utc: str
    experiment_id: str
    git_sha: str
    machine: str
    protocol: str
    regime: str
    model_config_identity: str
    test_cohort_hash: str
    reason: str
    metrics_requested: tuple[str, ...]
    schema_version: str = TEST_LOOK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TEST_LOOK_SCHEMA_VERSION:
            raise ValueError("unsupported test-look schema")
        try:
            datetime.fromisoformat(self.timestamp_utc.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("test-look timestamp must be ISO-8601") from error
        required = (
            self.experiment_id, self.git_sha, self.machine, self.protocol, self.regime,
            self.model_config_identity, self.test_cohort_hash, self.reason,
        )
        if any(not str(value).strip() for value in required) or not self.metrics_requested:
            raise ValueError("test-look provenance must be complete")


class TestLookLedger:
    __test__ = False

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, record: TestLookRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = (canonical_json(asdict(record)) + "\n").encode("utf-8")
        descriptor = os.open(
            self.path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_BINARY", 0),
        )
        try:
            written = os.write(descriptor, payload)
            if written != len(payload):
                raise OSError(
                    f"incomplete test-look entry write: {written} of {len(payload)} bytes"
                )
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def append_many_atomic(
        self,
        records: tuple[TestLookRecord, ...] | list[TestLookRecord],
        *,
        expected_existing_count: int,
    ) -> None:
        """Append a predeclared experiment batch in one durable write operation."""

        items = tuple(records)
        if not items:
            raise ValueError("test-look batch cannot be empty")
        observed = self.read()
        if len(observed) != expected_existing_count:
            raise RuntimeError(
                "test-look ledger count changed before batch authorization: "
                f"expected {expected_existing_count}, found {len(observed)}"
            )
        experiment_ids = [record.experiment_id for record in items]
        if len(experiment_ids) != len(set(experiment_ids)):
            raise ValueError("test-look batch contains duplicate experiment ids")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(canonical_json(asdict(record)) + "\n" for record in items).encode(
            "utf-8"
        )
        descriptor = os.open(
            self.path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_BINARY", 0),
        )
        try:
            written = os.write(descriptor, payload)
            if written != len(payload):
                raise OSError(
                    f"incomplete test-look batch write: {written} of {len(payload)} bytes"
                )
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if len(self.read()) != expected_existing_count + len(items):
            raise RuntimeError("durable test-look batch verification failed")

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
