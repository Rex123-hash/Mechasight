"""Phase 5.5F: trusted label-free window index (versioned, hash-bound).

The Phase-4 spectral caches identify their rows only through ``provenance_json``,
a single serialized JSON value that embeds ``anomaly_label`` for every window, so
reading it to align scoring inputs would materialise outcome labels. This module
derives the window identity *without* any label-bearing input:

* segment identities, recording parents and domains come from a label-free
  projection of the frozen split manifest (only the listed identity columns are
  parsed into the frame);
* window boundaries come from the raw sensor **timestamp column only** (its name
  is taken from the Parquet schema, no value column is read), passed through the
  frozen value-independent startup transforms, the frozen shared
  ``common_coverage`` and the frozen ``window_bounds`` — the exact functions the
  preprocessing pipeline uses, so no cropping, window length, stride or ordering
  is re-invented;
* row order is the pipeline's iteration order (machine, recording parent,
  segment), so ``row`` is the window's row in the cache built from the same cohort.

``validate_index`` enforces the schema, forbids any outcome-like column, binds
the cohort identity and rejects missing, duplicated or misordered windows.

Real protected index construction is DISABLED (``build_protected_window_index``):
it would require an authorized, ledger-recorded read of the protected
common-evaluation raw timestamps and a label-free raw-path resolver.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.experiment.phase5_inputs import Refused
from src.experiment.provenance import canonical_bytes, canonical_sha256, content_sha256, ordered_id_hash
from src.preprocessing.config import PreprocessingConfig
from src.preprocessing.signals import (
    SENSOR_KEYS,
    SensorSignal,
    apply_startup_transformations,
    common_coverage,
    window_bounds,
)

INDEX_SCHEMA_VERSION = "phase5-label-free-window-index-v1"
INDEX_COLUMNS = ("machine", "regime", "segment_id", "parent_recording", "domain",
                 "window_index", "window_start_s", "window_end_s", "row")
SEGMENT_SOURCE_COLUMNS = ("machine", "segment_id", "parent_recording", "domain")
FORBIDDEN_TOKENS = ("label", "anomaly", "fault", "outcome", "is_normal", "target_class")
MANIFEST_IDENTITY_COLUMNS = ("machine", "regime", "split", "segment_id", "parent_recording", "file_domain")
TIME_COLUMN_NAMES = {"time", "timestamp", "t"}
REPOSITORY = Path(__file__).resolve().parents[2]
DEVELOPMENT_SOURCES = REPOSITORY / "reports/phase5_executable_freeze_candidate/development_sources.json"
# Pinned independently of caller inputs and the mutable executable manifest.
DEVELOPMENT_SOURCES_SHA256 = "b3c513b383a42021bf9c336dcd991eee4c7b05af2bccbf8b11c82cee4fd47757"


@lru_cache(maxsize=1)
def _decode_sources(raw: bytes):
    contract = json.loads(raw)
    _need(contract["schema"] == "phase5-trusted-development-sources-v1"
          and contract["representation"] == "phase4-log-power-spectrum-v1", "unknown development contract")
    return contract


def _development_contract():
    """Recheck pinned metadata on every entry; never hash or open a sensor here."""
    try:
        _no_alias(DEVELOPMENT_SOURCES)
        raw = canonical_bytes(DEVELOPMENT_SOURCES)
        _need(sha256(raw).hexdigest() == DEVELOPMENT_SOURCES_SHA256,
              "trusted development manifest changed")
        contract = _decode_sources(raw)
        for record in (contract["split_manifest"], *contract["configs"].values()):
            _no_alias(REPOSITORY / record["path"])
            _need(canonical_sha256(REPOSITORY / record["path"]) == record["sha256"],
                  "frozen development split/config manifest changed")
        return contract
    except (OSError, ValueError, KeyError) as exc:
        raise IndexRefused("trusted development manifest missing or malformed") from exc


def _no_alias(path):
    _need(not any(p.is_symlink() or p.is_junction() for p in (path, *path.parents))
          and path.resolve(strict=True) == path.absolute(), "development path alias/link refused")


def development_catalog() -> pd.DataFrame:
    """Pinned label-free paths; never parse the released label-bearing catalog."""
    contract = _development_contract()
    rows = [{"machine": r["machine"], "segment_id": r["segment_id"],
             **{m + "_path": str(REPOSITORY / p) for m, p in r["paths"].items()}}
            for r in contract["sources"]]
    return pd.DataFrame(rows).drop_duplicates(["machine", "segment_id"]).reset_index(drop=True)


def _validate_source(source, config, contract):
    _need(type(source) is SegmentSource, "a registered development SegmentSource is required")
    _need(type(config) is PreprocessingConfig, "frozen preprocessing config required")
    _need(config.regime in contract["configs"] and
          content_sha256(config.canonical_dict()) == contract["configs"][config.regime]["config_sha256"],
          "representation/config differs from frozen development contract")
    matches = [r for r in contract["sources"] if (r["machine"], r["regime"], r["segment_id"]) ==
               (source.machine, config.regime, source.segment_id)]
    _need(len(matches) == 1, "unregistered development source")
    record = matches[0]
    _need(record["split"] == "validation" and source.domain == record["domain"] == "source"
          and source.parent_recording == record["parent_recording"], "development identity mismatch")
    _need(set(source.paths) == set(SENSOR_KEYS), "development modalities mismatch")
    # Validate every supplied path lexically BEFORE resolving or inspecting it.
    paths = {}
    for sensor in SENSOR_KEYS:
        path = Path(source.paths[sensor])
        expected = REPOSITORY / record["paths"][sensor]
        _need(".." not in path.parts and path.absolute() == expected,
              "source path differs from trusted development binding")
        paths[sensor] = expected
    for sensor, path in paths.items():
        # Reject links/junctions even if they resolve back inside the repository.
        try:
            _no_alias(path)
            stat = path.stat()
        except OSError as exc:
            raise IndexRefused("registered development source is missing/inaccessible") from exc
        _need({"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns} == record["stat"][sensor],
              "development source stat identity changed")
    return paths


class IndexRefused(Refused):
    """The index cannot be constructed or accepted safely."""


def _need(value, message):
    if not value:
        raise IndexRefused(message)


@dataclass(frozen=True)
class SegmentSource:
    """Label-free identity and raw timestamp sources of one segment."""

    machine: str
    segment_id: str
    parent_recording: str
    domain: str
    paths: Mapping[str, str]


def _assert_label_free(columns: Iterable[str]) -> None:
    for column in columns:
        name = str(column).lower()
        if any(token in name for token in FORBIDDEN_TOKENS):
            raise IndexRefused(f"outcome-like column {column!r} is not permitted in a label-free index")


def label_free_cohort(split_manifest: str | Path, *, machine: str, regime: str, split: str) -> pd.DataFrame:
    """Identity columns of one development cohort, parsed with an explicit column projection."""

    _need(split in {"train", "validation"}, "only development cohorts may be indexed here")
    frame = pd.read_csv(split_manifest, usecols=list(MANIFEST_IDENTITY_COLUMNS))
    _assert_label_free(frame.columns)
    frame = frame[frame["machine"].eq(machine) & frame["regime"].eq(regime) & frame["split"].eq(split)]
    _need(not frame.empty, "empty development cohort")
    return frame.rename(columns={"file_domain": "domain"}).loc[:, list(SEGMENT_SOURCE_COLUMNS)].reset_index(drop=True)


def _timestamps(path: str | Path, sensor: str, *, source=None, config=None) -> SensorSignal:
    """The stored timestamp column only (value columns are never read)."""

    _need(source is not None and config is not None, "registered development source/config required before timestamp access")
    paths = _validate_source(source, config, _development_contract())
    _need(sensor in paths and Path(path).absolute() == paths[sensor], "timestamp modality/path mismatch")
    names = [n for n in pq.read_schema(paths[sensor]).names if n.lower() in TIME_COLUMN_NAMES]
    _need(len(names) == 1, f"{sensor}: expected exactly one timestamp column")
    t = pd.read_parquet(paths[sensor], columns=names)[names[0]].to_numpy(np.float64)
    if len(t) < 2 or not np.all(np.isfinite(t)) or not np.all(np.diff(t) > 0):
        raise IndexRefused(f"{sensor}: timestamps are not finite and strictly increasing; index not constructed")
    return SensorSignal(sensor, t, np.empty((len(t), 0)), (), len(t), len(t))


def segment_windows(source: SegmentSource, config: PreprocessingConfig) -> list[tuple[float, float]]:
    """Frozen window bounds of one segment from its timestamps alone."""

    _validate_source(source, config, _development_contract())
    signals = {s: _timestamps(source.paths[s], s, source=source, config=config) for s in SENSOR_KEYS}
    transformed, _ = apply_startup_transformations(signals, source.machine, config)
    start, end = common_coverage(transformed)
    bounds = window_bounds(start, end, config.window_length_s, config.stride_s)
    if not bounds:
        raise IndexRefused(f"{source.segment_id}: the common coverage holds no complete window")
    return bounds


def build_window_index(sources: Sequence[SegmentSource], config: PreprocessingConfig, *, regime: str) -> pd.DataFrame:
    """Label-free index of a whole cohort in the frozen pipeline iteration order."""

    _need(regime == config.regime, "regime differs from the preprocessing configuration")
    contract = _development_contract()
    # Whole-request preflight: an invalid final source must not expose earlier ones.
    for source in sources:
        _validate_source(source, config, contract)
    ordered = sorted(sources, key=lambda s: (s.machine, s.parent_recording, s.segment_id))
    _need(len({(s.machine, s.segment_id) for s in ordered}) == len(ordered), "duplicate segment in cohort")
    rows, row = [], 0
    for source in ordered:
        for k, (start, end) in enumerate(segment_windows(source, config)):
            rows.append((source.machine, regime, source.segment_id, source.parent_recording, source.domain,
                         k, float(start), float(end), row))
            row += 1
    return pd.DataFrame(rows, columns=list(INDEX_COLUMNS))


def cohort_identity(segment_ids: Iterable[str]) -> str:
    return ordered_id_hash(tuple(sorted(str(s) for s in segment_ids)))


def index_manifest(index: pd.DataFrame, *, representation: Mapping, file_sha256: str) -> dict:
    """The hash-bound description that accompanies an index file."""

    return {"schema": INDEX_SCHEMA_VERSION, "columns": list(INDEX_COLUMNS), "rows": int(len(index)),
            "segments": int(index["segment_id"].nunique()), "cohort_identity": cohort_identity(index["segment_id"].unique()),
            "machine": str(index["machine"].iloc[0]), "regime": str(index["regime"].iloc[0]),
            "content_sha256": content_sha256(index.to_dict(orient="list")), "file_sha256": file_sha256,
            "representation": dict(representation)}


def validate_index(index: pd.DataFrame, manifest: Mapping, *, expected_segments: Sequence[Mapping],
                   machine: str, regime: str) -> pd.DataFrame:
    """Accept an index only if it is exactly the label-free, hash-bound window index it claims to be."""

    try:
        _assert_label_free(index.columns)
        _need(manifest.get("schema") == INDEX_SCHEMA_VERSION, "unknown window-index schema")
        _need(tuple(index.columns) == INDEX_COLUMNS and list(manifest.get("columns", [])) == list(INDEX_COLUMNS),
                "window-index columns differ from the frozen schema")
        _need((manifest["machine"], manifest["regime"]) == (machine, regime)
                and set(index["machine"]) == {machine} and set(index["regime"]) == {regime}, "window index is for another cell")
        _need(len(index) == manifest["rows"] and content_sha256(index.to_dict(orient="list")) == manifest["content_sha256"],
                "window-index content differs from its manifest")
        expected = pd.DataFrame(list(expected_segments)).loc[:, ["segment_id", "parent_recording", "domain"]]
        _need(manifest["cohort_identity"] == cohort_identity(expected["segment_id"]), "window index belongs to another cohort")
        observed = index.drop_duplicates("segment_id").loc[:, ["segment_id", "parent_recording", "domain"]]
        _need(observed.sort_values("segment_id").reset_index(drop=True)
                .equals(expected.sort_values("segment_id").reset_index(drop=True)), "segment/parent/domain alignment mismatch")
        _need(not index.duplicated(["segment_id", "window_index"]).any() and index["row"].is_unique,
                "duplicate window identity")
        order = index.loc[:, ["machine", "parent_recording", "segment_id"]].apply(tuple, axis=1).tolist()
        _need(order == sorted(order), "segments are not in the frozen pipeline order")
        _need(index["row"].is_monotonic_increasing, "window rows are misordered")
        for _, group in index.groupby("segment_id", sort=False):
            _need(group["window_index"].tolist() == list(range(len(group))), "missing or misordered windows")
            _need((group["window_end_s"] > group["window_start_s"]).all()
                    and group["window_start_s"].is_monotonic_increasing, "window bounds are inconsistent")
    except (KeyError, TypeError, ValueError) as exc:
        raise IndexRefused(f"window index is malformed: {exc}") from exc
    return index


def build_protected_window_index(*args, **kwargs):
    """DISABLED in Phase 5.5F.

    Building the common-evaluation index would require, under a human batch
    authorization and a ledger-recorded exposure: (1) an authorized read of the
    protected raw *timestamp* columns of every common-evaluation segment and (2) a
    pre-registered label-free resolver from frozen segment identity to raw file
    paths (the release metadata loader also parses outcome columns). Neither is
    authorized here.
    """
    raise IndexRefused("protected window-index construction is disabled: it needs an authorized, ledger-recorded "
                       "read of protected raw timestamps and a label-free raw-path resolver")
