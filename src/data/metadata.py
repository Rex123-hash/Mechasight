"""Loading and cross-checking IMAD-DS `attributes_*.csv` metadata.

Two independent sources describe each segment's split, domain and class:

1. the metadata *filename* (`attributes_anomaly_target_test.csv`), and
2. columns *inside* the file (`split_label`, `anomaly_label`, ...).

This module keeps both and compares them. Merging them would hide exactly the
kind of labelling error the audit exists to find.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.discovery import SENSOR_TOKENS, parse_metadata_name

# Columns the reference baseline reads. Absence is recorded, not raised.
EXPECTED_LABEL_COLUMNS = [
    "split_label",
    "anomaly_label",
    "domain_shift_op",
    "domain_shift_env",
]

MIC_PREFIX = f"{SENSOR_TOKENS['mic']}_"


def load_metadata_file(path: Path) -> tuple[pd.DataFrame, dict]:
    """Read one attributes CSV verbatim and describe what it contained.

    Returns (frame, info). The frame carries the file's own columns plus
    provenance columns prefixed `file_` derived from the filename.
    """
    info: dict = {"path": str(path), "name": path.name}

    try:
        # Read WITHOUT index_col: the released files carry `segment_id` as their
        # first column, and consuming it into the index would hide it from
        # `raw.columns`, silently downgrading us to a filename-derived id.
        raw = pd.read_csv(path)
        unnamed = [c for c in raw.columns if str(c).startswith("Unnamed: 0")]
        if unnamed:
            raw = raw.drop(columns=unnamed)
    except Exception as exc:
        info.update(readable=False, error=f"{type(exc).__name__}: {exc}", n_rows=0)
        return pd.DataFrame(), info

    info.update(
        readable=True,
        n_rows=int(len(raw)),
        columns=list(raw.columns),
        n_columns=int(raw.shape[1]),
    )

    frame = raw.copy()
    frame["source_csv"] = path.name

    parsed = parse_metadata_name(path)
    info["filename_parsed"] = parsed is not None
    for field in ("anomaly", "domain", "split"):
        frame[f"file_{field}"] = parsed[field] if parsed else None

    # Which sensor columns this file actually provides.
    present = {
        key: token for key, token in SENSOR_TOKENS.items() if token in frame.columns
    }
    info["sensor_columns_present"] = sorted(present)
    info["sensor_columns_missing"] = sorted(set(SENSOR_TOKENS) - set(present))
    info["label_columns_present"] = [
        c for c in EXPECTED_LABEL_COLUMNS if c in frame.columns
    ]
    info["label_columns_missing"] = [
        c for c in EXPECTED_LABEL_COLUMNS if c not in frame.columns
    ]

    # The release provides its own `segment_id` column. Prefer it; derive one
    # only if it is absent. The derived form is kept alongside as
    # `segment_id_from_mic` so the two can be compared -- they are NOT equal in
    # the real release (the column ends `_segment_N`, the filename `_DataLog_N`),
    # which is worth recording rather than papering over.
    derived = _derive_segment_id(frame)
    if "segment_id" in raw.columns:
        info["segment_id_source"] = "released_column"
        frame["segment_id_from_mic"] = derived
        frame["segment_id"] = frame["segment_id"].astype("string")
    else:
        info["segment_id_source"] = "derived_from_mic_filename"
        frame["segment_id"] = derived
        frame["segment_id_from_mic"] = derived

    info["n_null_segment_id"] = int(frame["segment_id"].isna().sum())
    info["n_unique_segment_id"] = int(frame["segment_id"].nunique(dropna=True))

    return frame, info


def _derive_segment_id(frame: pd.DataFrame) -> pd.Series:
    """Segment id derived from the microphone filename, as the baseline does.

    Falls back to the accelerometer, then the gyroscope, so a file missing the
    microphone column still gets an id rather than being dropped from the audit.
    """
    for key in ("mic", "acc", "gyro"):
        column = SENSOR_TOKENS[key]
        if column in frame.columns:
            prefix = f"{column}_"
            return (
                frame[column]
                .astype("string")
                .str.replace(r"\.parquet$", "", regex=True)
                .str.removeprefix(prefix)
            )
    return pd.Series([pd.NA] * len(frame), index=frame.index, dtype="string")


def load_machine_metadata(
    metadata_files: list[Path],
) -> tuple[pd.DataFrame, list[dict]]:
    """Load and concatenate every attributes CSV for one machine."""
    frames, infos = [], []
    for path in sorted(metadata_files):
        frame, info = load_metadata_file(path)
        infos.append(info)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame(), infos

    combined = pd.concat(frames, ignore_index=True)
    return combined, infos


def check_label_consistency(frame: pd.DataFrame) -> dict:
    """Compare filename-derived labels against the in-file label columns.

    A mismatch means the release's own two descriptions of a segment disagree,
    which would invalidate any split built on either one alone.
    """
    result: dict = {
        "n_rows": int(len(frame)),
        "checked_split_label": False,
        "checked_anomaly_label": False,
    }

    if "split_label" in frame.columns:
        result["checked_split_label"] = True
        label = frame["split_label"].astype("string").str.lower()
        # split_label looks like "Normal_Source_Test".
        for field, position in (("anomaly", 0), ("domain", 1), ("split", 2)):
            part = label.str.split("_").str[position]
            expected = frame[f"file_{field}"].astype("string").str.lower()
            comparable = part.notna() & expected.notna()
            mismatch = comparable & (part != expected)
            result[f"{field}_mismatch_count"] = int(mismatch.sum())
            result[f"{field}_comparable_count"] = int(comparable.sum())
            if mismatch.any():
                result[f"{field}_mismatch_examples"] = (
                    frame.loc[mismatch, ["segment_id", "split_label"]]
                    .head(5)
                    .to_dict("records")
                )

    if "anomaly_label" in frame.columns:
        result["checked_anomaly_label"] = True
        is_normal = frame["anomaly_label"].astype("string").str.lower().eq("normal")
        file_says_normal = frame["file_anomaly"].astype("string").str.lower().eq("normal")
        mismatch = is_normal != file_says_normal
        result["anomaly_vs_filename_mismatch_count"] = int(mismatch.sum())
        if mismatch.any():
            result["anomaly_vs_filename_examples"] = (
                frame.loc[mismatch, ["segment_id", "anomaly_label", "file_anomaly"]]
                .head(10)
                .to_dict("records")
            )

    duplicated = frame["segment_id"].duplicated(keep=False)
    result["n_duplicate_segment_ids"] = int(duplicated.sum())
    if duplicated.any():
        result["duplicate_segment_id_examples"] = sorted(
            frame.loc[duplicated, "segment_id"].dropna().unique().tolist()
        )[:10]

    return result


def resolve_sensor_paths(frame: pd.DataFrame, machine_root: Path) -> pd.DataFrame:
    """Attach an absolute path per sensor per row, and whether it exists.

    Metadata stores paths relative to the machine root's split folder (the
    reference baseline prepends `<root>/<split>/`). Several candidate locations
    are tried so that a layout difference shows up as a resolved path rather
    than as a wall of false "missing file" findings.
    """
    out = frame.copy()
    for key, column in SENSOR_TOKENS.items():
        if column not in out.columns:
            out[f"{key}_path"] = pd.NA
            out[f"{key}_exists"] = False
            continue

        paths, exists = [], []
        for value, split in zip(out[column], out["file_split"], strict=False):
            resolved = _resolve_one(value, split, machine_root)
            paths.append(str(resolved) if resolved else pd.NA)
            exists.append(resolved is not None)
        out[f"{key}_path"] = paths
        out[f"{key}_exists"] = exists
    return out


def _resolve_one(value, split, machine_root: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    rel = value if value.lower().endswith(".parquet") else f"{value}.parquet"
    candidates = [machine_root / rel]
    if isinstance(split, str):
        candidates.insert(0, machine_root / split / rel)
    candidates.append(machine_root / "train" / rel)
    candidates.append(machine_root / "test" / rel)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None
