"""Ordinal-only, label-free window alignment. Counts cannot recover physical times."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.experiment.phase5_input_manifest import (
    COUNT_COLUMNS, MODALITIES, read_features, read_spectral, validate_manifest,
)
from src.experiment.phase5_inputs import IDENTITY_COLUMNS, Refused, require
from src.experiment.provenance import content_sha256

SCHEMA = "phase5-label-free-window-index-v2-count-derived"
INDEX_COLUMNS = ("machine", "regime", *IDENTITY_COLUMNS, "window_index", "row", "segment_row_offset",
                 "window_count", "cohort_identity", "source_artifact_identity")


@dataclass(frozen=True)
class CountIndex:
    rows: pd.DataFrame
    descriptor: dict


def _expand_count_rows(manifest, features, spectral, *, unit=None):
    """Pure draft calculation, not a reader or an authorization grant.

    Even a self-hashed draft must pass validate_count_index against the bound
    artifacts before the public inference projection accepts it.
    """
    doc = validate_manifest(manifest, unit=unit, for_read=True)
    require(isinstance(features, pd.DataFrame) and tuple(features.columns) == COUNT_COLUMNS,
            "count projection has missing/unexpected columns")
    require(not features["segment_id"].duplicated().any(), "duplicated segment")
    expected = pd.DataFrame(doc["segments"], columns=list(IDENTITY_COLUMNS))
    observed = features.loc[:, list(IDENTITY_COLUMNS)].reset_index(drop=True)
    require(observed.equals(expected), "segment cohort/order/parent/domain mismatch")
    counts = features["window_count"].tolist()
    require(all(isinstance(k, (int, np.integer)) and not isinstance(k, (bool, np.bool_)) and k > 0 for k in counts),
            "window_count must contain positive integers; zero/missing/fractional counts refused")
    total = sum(int(k) for k in counts)
    require(total <= np.iinfo(np.int64).max, "window count overflows ordinal representation")
    require(isinstance(spectral, dict) and set(spectral) == {"modalities", "declared_window_count"}, "spectral layout malformed")
    require(set(spectral["modalities"]) == set(MODALITIES), "missing/unexpected modality data")
    for m in MODALITIES:
        record = spectral["modalities"][m]
        require(record == {"rows": total, "width": doc["layouts"]["spectral"]["widths"][m], "dtype": "float32"},
                "feature/spectral row total or representation mismatch")
    declared = spectral["declared_window_count"]
    require(declared is None or (isinstance(declared, (int, np.integer)) and not isinstance(declared, (bool, np.bool_))
                                and declared == total), "declared spectral count mismatch")
    source = content_sha256(doc["artifacts"]["features"])
    repeats = np.asarray(counts, dtype=np.int64)
    offsets = np.cumsum(np.r_[0, repeats[:-1]])
    index = observed.loc[observed.index.repeat(repeats)].reset_index(drop=True)
    index.insert(0, "regime", doc["regime"])
    index.insert(0, "machine", doc["machine"])
    index["row"] = np.arange(total, dtype=np.int64)
    index["segment_row_offset"] = np.repeat(offsets, repeats)
    index["window_index"] = index["row"] - index["segment_row_offset"]
    index["window_count"] = np.repeat(repeats, repeats)
    index["cohort_identity"] = doc["cohort_identity"]
    index["source_artifact_identity"] = source
    index = index.loc[:, list(INDEX_COLUMNS)]
    descriptor = {"schema": SCHEMA, "manifest_identity": manifest.identity, "columns": list(INDEX_COLUMNS),
                  "scope": doc["scope"], "rows": total, "segments": len(expected), "cohort_identity": doc["cohort_identity"],
                  "source_artifact_identity": source, "features_sha256": doc["artifacts"]["features"]["sha256"],
                  "spectral_sha256": doc["artifacts"]["spectral"]["sha256"],
                  "counts_sha256": content_sha256(features.to_dict("list")),
                  "content_sha256": content_sha256(index.to_dict("list")),
                  "alignment": "ordinal only; no physical timestamps or sample boundaries reconstructed"}
    return CountIndex(index, descriptor)


def build_count_derived_index(manifest, *, unit=None):
    # Manifest scope and unit are checked before any artifact read or content hash.
    validate_manifest(manifest, unit=unit, for_read=True)
    features = read_features(manifest, unit=unit)
    layout, _ = read_spectral(manifest, unit=unit)
    return _expand_count_rows(manifest, features, layout, unit=unit)


def validate_count_index(index, manifest, *, unit=None):
    validate_manifest(manifest, unit=unit, for_read=True)
    require(type(index) is CountIndex, "count-derived index reference required")
    # Reconstruct from approved artifacts, not caller-supplied counts/hash claims.
    expected = build_count_derived_index(manifest, unit=unit)
    require(index.descriptor == expected.descriptor and index.rows.equals(expected.rows),
            "count index hash/offset/order/content mismatch")
    return index


def inference_index(index, manifest, *, unit=None):
    """The unchanged FrozenL1Adapter needs identities and ordinals, not times."""
    validate_count_index(index, manifest, unit=unit)
    return index.rows.loc[:, [*IDENTITY_COLUMNS, "window_index"]].copy()


def build_protected_count_index(*args, capability=None, batch_id=None, **request):
    raise Refused("protected count-derived index disabled in Phase 5.6B; human authorization, reviewed executor capability, batch and durable exposure required")
