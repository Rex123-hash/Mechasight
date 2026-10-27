"""Recording-time structure of the release (Q8, risk R9).

IMAD-DS segment ids embed the wall-clock start time of the recording, e.g.
`20240423_15_30_05_segment_0`. That makes it possible to reconstruct the order
in which the dataset was recorded and to ask a question the metadata alone
cannot answer: is the class label confounded with recording time?

It matters because a detector that separates "recorded later in the session"
from "recorded earlier" would score well on anomaly detection without detecting
anything mechanical. Session drift — thermal, mechanical settling, sensor bias,
ambient conditions — is exactly the kind of nuisance that accumulates over hours.

Metadata only; no signal content is read here.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Segment id shapes differ between machines and were discovered from the release,
# not assumed:
#   BrushlessMotor : 20240423_15_30_05_segment_0        (index always 0)
#   RoboticArm     : 20240418_14_01_48_dataLog_12       (index varies; several
#                    segments share one stamp, i.e. one parent recording)
#   RoboticArm     : 20240418_..._corr3_dataLog_7  /  ..._corr_3_dataLog_7
#                    (an undocumented "corr" marker on 85 of 2303 segments)
# The leading wall-clock stamp identifies the parent recording in every shape.
SEGMENT_ID_RE = re.compile(
    r"^(?P<stamp>\d{8}_\d{2}_\d{2}_\d{2})"
    r"(?P<extra>(?:_[A-Za-z]+_?\d*)*?)"
    r"_(?:segment|dataLog|DataLog)_(?P<index>\d+)$"
)
CORR_MARKER_RE = re.compile(r"corr", re.IGNORECASE)

STAMP_FORMAT = "%Y%m%d_%H_%M_%S"


def parse_segment_times(segment_ids: pd.Series) -> pd.DataFrame:
    """Extract recording timestamp and within-recording index from segment ids.

    Unparseable ids yield NaT rather than raising -- an id shape we did not
    anticipate is a finding to report, not a crash.
    """
    extracted = segment_ids.astype("string").str.extract(SEGMENT_ID_RE)
    return pd.DataFrame(
        {
            "recording_stamp": extracted["stamp"],
            "segment_index": pd.to_numeric(extracted["index"], errors="coerce"),
            "recording_time": pd.to_datetime(
                extracted["stamp"], format=STAMP_FORMAT, errors="coerce"
            ),
            "id_extra": extracted["extra"].fillna(""),
            "has_corr_marker": extracted["extra"].fillna("").str.contains(
                CORR_MARKER_RE, regex=True
            ),
        },
        index=segment_ids.index,
    )


def _window(times: pd.Series) -> dict:
    return {
        "n": int(times.notna().sum()),
        "first": times.min(),
        "last": times.max(),
    }


def _overlap_seconds(a: pd.Series, b: pd.Series) -> float:
    """Seconds of overlap between two groups' recording windows.

    Negative means a clean temporal separation, and the magnitude is the size of
    the gap -- which is more informative than clamping to zero.
    """
    if a.notna().sum() == 0 or b.notna().sum() == 0:
        return float("nan")
    latest_start = max(a.min(), b.min())
    earliest_end = min(a.max(), b.max())
    return float((earliest_end - latest_start).total_seconds())


def analyse_temporal_structure(frame: pd.DataFrame) -> dict:
    """Full Q8 analysis for one machine.

    Expects columns: segment_id, file_split, anomaly_label (and optionally
    file_domain).
    """
    times = parse_segment_times(frame["segment_id"])
    data = frame.join(times)

    result: dict = {
        "n_segments": int(len(data)),
        "n_unparsed_segment_ids": int(data["recording_time"].isna().sum()),
        "n_unique_recording_stamps": int(data["recording_stamp"].nunique(dropna=True)),
        "segment_index_values": sorted(
            data["segment_index"].dropna().astype(int).unique().tolist()
        ),
    }
    valid = data[data["recording_time"].notna()]
    if valid.empty:
        result["analysable"] = False
        return result
    result["analysable"] = True

    result["recording_dates"] = sorted(
        valid["recording_time"].dt.date.astype(str).unique().tolist()
    )
    result["session_first"] = str(valid["recording_time"].min())
    result["session_last"] = str(valid["recording_time"].max())
    result["session_hours"] = float(
        (valid["recording_time"].max() - valid["recording_time"].min()).total_seconds() / 3600
    )

    # More than one segment per recording would mean parent-recording leakage is
    # possible; a single index value means one segment per recording.
    result["one_segment_per_recording"] = result["segment_index_values"] == [0]
    result["n_stamps_reused_across_segments"] = int(
        len(valid) - valid["recording_stamp"].nunique()
    )
    per_recording = valid.groupby("recording_stamp").size()
    result["segments_per_recording"] = {
        "min": int(per_recording.min()),
        "median": float(per_recording.median()),
        "max": int(per_recording.max()),
        "n_recordings": int(per_recording.size),
    }
    result["n_segments_with_corr_marker"] = int(valid["has_corr_marker"].sum())
    result["id_extra_values"] = sorted(
        v for v in valid["id_extra"].unique().tolist() if v
    )[:20]

    # THE parent-recording leakage check (risk R9): do segments cut from one
    # recording land in more than one split, or in more than one class?
    grouped = valid.groupby("recording_stamp")
    splits_per_recording = grouped["file_split"].nunique()
    result["n_recordings_spanning_train_and_test"] = int((splits_per_recording > 1).sum())
    result["recordings_spanning_splits_examples"] = (
        splits_per_recording[splits_per_recording > 1].index.tolist()[:10]
    )
    labels_per_recording = grouped["anomaly_label"].nunique()
    result["n_recordings_spanning_multiple_labels"] = int((labels_per_recording > 1).sum())
    if "file_domain" in valid.columns:
        domains_per_recording = grouped["file_domain"].nunique()
        result["n_recordings_spanning_multiple_domains"] = int(
            (domains_per_recording > 1).sum()
        )
    # How many segments sit in a recording that straddles the train/test line.
    straddling = splits_per_recording[splits_per_recording > 1].index
    result["n_segments_in_straddling_recordings"] = int(
        valid["recording_stamp"].isin(straddling).sum()
    )

    is_anomaly = valid["anomaly_label"].astype("string").str.lower().ne("normal")
    normal_times = valid.loc[~is_anomaly, "recording_time"]
    anomaly_times = valid.loc[is_anomaly, "recording_time"]

    result["normal_window"] = {k: str(v) for k, v in _window(normal_times).items()}
    result["anomaly_window"] = {k: str(v) for k, v in _window(anomaly_times).items()}
    overlap = _overlap_seconds(normal_times, anomaly_times)
    result["normal_anomaly_overlap_s"] = overlap
    result["classes_temporally_disjoint"] = bool(np.isfinite(overlap) and overlap < 0)
    result["n_normal_after_first_anomaly"] = (
        int((normal_times > anomaly_times.min()).sum()) if len(anomaly_times) else 0
    )
    result["n_anomaly_before_last_normal"] = (
        int((anomaly_times < normal_times.max()).sum()) if len(normal_times) else 0
    )

    # Per anomaly type -- the types may also be blocked in time relative to
    # each other, which would confound any per-type comparison.
    per_type = {}
    for label, part in valid.groupby(valid["anomaly_label"].astype("string")):
        per_type[str(label)] = {k: str(v) for k, v in _window(part["recording_time"]).items()}
    result["window_by_anomaly_label"] = per_type

    types = [
        label for label in valid["anomaly_label"].astype("string").unique()
        if str(label).lower() != "normal"
    ]
    type_overlaps = {}
    for i, first in enumerate(types):
        for second in types[i + 1:]:
            type_overlaps[f"{first}|{second}"] = _overlap_seconds(
                valid.loc[valid["anomaly_label"] == first, "recording_time"],
                valid.loc[valid["anomaly_label"] == second, "recording_time"],
            )
    result["anomaly_type_overlap_s"] = type_overlaps

    # Train/test interleaving among the normal segments: if normal train and
    # normal test are interleaved, they share session conditions closely, which
    # makes the normal-vs-normal comparison easy and the normal-vs-anomaly
    # comparison partly a comparison of session epochs.
    normals = valid[~is_anomaly]
    if "file_split" in normals.columns and normals["file_split"].nunique() > 1:
        result["normal_train_test_overlap_s"] = _overlap_seconds(
            normals.loc[normals["file_split"] == "train", "recording_time"],
            normals.loc[normals["file_split"] == "test", "recording_time"],
        )
        ordered = normals.sort_values("recording_time")
        is_test = ordered["file_split"].eq("test")
        blocks = (is_test != is_test.shift()).cumsum()
        sizes = ordered.groupby(blocks).size()
        result["normal_train_test_alternating_blocks"] = int(sizes.size)
        result["normal_train_test_block_size_median"] = float(sizes.median())

    gaps = valid.sort_values("recording_time")["recording_time"].diff().dt.total_seconds()
    result["inter_recording_gap_s"] = {
        "min": float(gaps.min()) if gaps.notna().any() else float("nan"),
        "median": float(gaps.median()) if gaps.notna().any() else float("nan"),
        "max": float(gaps.max()) if gaps.notna().any() else float("nan"),
    }
    return result


def temporal_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-segment recording times, for the manifest and for later drift probes."""
    times = parse_segment_times(frame["segment_id"])
    out = frame[["machine", "segment_id", "file_split", "file_domain",
                 "anomaly_label"]].copy()
    out["recording_time"] = times["recording_time"]
    out["segment_index"] = times["segment_index"]
    return out.sort_values(["machine", "recording_time"])
