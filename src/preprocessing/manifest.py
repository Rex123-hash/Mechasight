"""Processed-window traceability schema and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = (
    "machine", "sensor_modalities", "raw_file_references", "segment_id",
    "parent_recording", "window_id", "window_start_s", "window_end_s",
    "protocol", "regime", "split", "domain", "operating_condition",
    "background_condition", "anomaly_label", "sampling_rate_metadata",
    "preprocessing_config_id",
)


def validate_manifest(frame: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"manifest is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("processed manifest cannot be empty")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        columns = frame[list(REQUIRED_COLUMNS)].columns[
            frame[list(REQUIRED_COLUMNS)].isna().any()
        ].tolist()
        raise ValueError(f"manifest has null traceability fields: {columns}")
    if frame["window_id"].duplicated().any():
        raise ValueError("window_id must be globally unique in a manifest")
    if (frame["window_end_s"] <= frame["window_start_s"]).any():
        raise ValueError("manifest contains non-positive physical windows")
    for column in ("sensor_modalities", "raw_file_references", "sampling_rate_metadata"):
        for value in frame[column]:
            parsed = json.loads(value) if isinstance(value, str) else value
            if not parsed:
                raise ValueError(f"manifest field {column} must contain non-empty JSON")


class JsonLinesManifestWriter:
    """Append-only manifest writer suitable for large streaming builds."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = None
        self.count = 0

    def __enter__(self) -> "JsonLinesManifestWriter":
        self._handle = self.path.open("w", encoding="utf-8", newline="\n")
        return self

    def append(self, record: dict) -> None:
        if self._handle is None:
            raise RuntimeError("manifest writer is not open")
        missing = set(REQUIRED_COLUMNS) - set(record)
        if missing:
            raise ValueError(f"record lacks traceability fields: {sorted(missing)}")
        self._handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        self.count += 1

    def __exit__(self, *_exc) -> None:
        if self._handle is not None:
            self._handle.close()


def read_jsonl_manifest(path: str | Path) -> pd.DataFrame:
    with Path(path).open("r", encoding="utf-8") as handle:
        return pd.DataFrame(json.loads(line) for line in handle if line.strip())
