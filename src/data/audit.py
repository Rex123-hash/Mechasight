"""Per-segment audit of IMAD-DS sensor parquet files.

Produces one manifest row per segment plus per-channel quality records. This
module measures and reports only: it never resamples, filters, normalises or
repairs. Every anomaly found becomes a warning string on the manifest row.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.timestamps import (
    analyse_alignment,
    analyse_timestamps,
    discarded_fraction,
    rate_drift,
)

SENSOR_KEYS = ("mic", "acc", "gyro")

# Documented in the reference preprocessing notebook as the gyroscope warm-up
# period. Used here only to *quantify* what that cut would cost.
GYRO_WARMUP_S = 0.035

TIME_COLUMN_CANDIDATES = ("time", "timestamp", "t")

# A channel whose values sit at its own min or max this often is likely clipped.
SATURATION_FRACTION_WARN = 0.001

# Warning kinds that describe a *property of the release* rather than a fault in
# it. Timestamp quantisation, for instance, is how the data was stored -- it
# constrains which rate estimator is trustworthy, but no segment is damaged by
# it. Counting these as defects would report a clean dataset as entirely faulty.
INFO_WARNING_KINDS = frozenset({
    "timestamps_quantised",
    "timestamp_precision_limited",
})


def classify_warnings(warnings: list[str]) -> tuple[int, int]:
    """Split warning strings into (informational, defect) counts.

    Warning strings look like "<sensor>:<kind>[:<detail>]".
    """
    info = defect = 0
    for warning in warnings:
        parts = warning.split(":")
        kind = parts[1] if len(parts) > 1 else parts[0]
        if kind in INFO_WARNING_KINDS:
            info += 1
        else:
            defect += 1
    return info, defect


def _find_time_column(columns: list[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for candidate in TIME_COLUMN_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _channel_quality(name: str, values: np.ndarray) -> dict[str, Any]:
    """Quality record for one data channel."""
    finite_mask = np.isfinite(values)
    finite = values[finite_mask]
    n = int(values.size)

    record: dict[str, Any] = {
        "channel": name,
        "n_samples": n,
        "n_nan": int(np.count_nonzero(np.isnan(values))),
        "n_posinf": int(np.count_nonzero(np.isposinf(values))),
        "n_neginf": int(np.count_nonzero(np.isneginf(values))),
        "n_finite": int(finite.size),
    }

    if finite.size == 0:
        record.update(
            min=np.nan, max=np.nan, mean=np.nan, std=np.nan,
            n_unique=0, is_constant=True, saturation_fraction=np.nan,
        )
        return record

    vmin, vmax = float(finite.min()), float(finite.max())
    at_rail = int(np.count_nonzero(finite == vmin) + np.count_nonzero(finite == vmax))

    record.update(
        min=vmin,
        max=vmax,
        mean=float(finite.mean()),
        std=float(finite.std()),
        # Capped: exact uniqueness over 160k floats is expensive and only the
        # small-cardinality case (a stuck or quantised channel) is interesting.
        n_unique=int(np.unique(finite[: min(finite.size, 50_000)]).size),
        is_constant=bool(vmin == vmax),
        saturation_fraction=float(at_rail / finite.size),
    )
    return record


def audit_sensor_file(path: str | Path, sensor: str) -> dict[str, Any]:
    """Read one sensor parquet and measure its timing and channel quality."""
    result: dict[str, Any] = {"sensor": sensor, "path": str(path)}
    warnings: list[str] = []

    file_path = Path(path)
    if not file_path.is_file():
        result.update(readable=False, error="file not found", warnings=["missing_file"])
        return result

    try:
        size = file_path.stat().st_size
        result["file_bytes"] = size
        if size == 0:
            result.update(readable=False, error="zero-byte file",
                          warnings=["zero_byte_file"])
            return result
        frame = pd.read_parquet(file_path)
    except Exception as exc:
        result.update(readable=False, error=f"{type(exc).__name__}: {exc}",
                      warnings=["unreadable_parquet"])
        return result

    result["readable"] = True
    result["columns"] = list(map(str, frame.columns))
    result["n_rows"] = int(len(frame))

    if len(frame) == 0:
        result.update(warnings=["empty_parquet"])
        return result

    time_column = _find_time_column(result["columns"])
    result["time_column"] = time_column
    if time_column is None:
        warnings.append("no_time_column")
        result["warnings"] = warnings
        return result

    time_values = pd.to_numeric(frame[time_column], errors="coerce").to_numpy(np.float64)
    result["time_dtype"] = str(frame[time_column].dtype)
    n_bad_time = int(np.count_nonzero(~np.isfinite(time_values)))
    if n_bad_time:
        warnings.append(f"non_finite_timestamps:{n_bad_time}")

    stats = analyse_timestamps(time_values[np.isfinite(time_values)])
    result["timing"] = stats.to_dict()
    result.update(rate_drift(time_values[np.isfinite(time_values)]))

    if stats.n_nonmonotonic:
        warnings.append(f"nonmonotonic_timestamps:{stats.n_nonmonotonic}")
    if stats.n_duplicate_timestamps:
        warnings.append(f"duplicate_timestamps:{stats.n_duplicate_timestamps}")
    if stats.n_gaps:
        warnings.append(f"timestamp_gaps:{stats.n_gaps}")
    if stats.precision_limited:
        warnings.append("timestamp_precision_limited")
    if stats.quantised:
        warnings.append(f"timestamps_quantised:{stats.n_unique_dt}_distinct_dt")

    channels = [c for c in result["columns"] if c != time_column]
    result["data_channels"] = channels
    result["n_data_channels"] = len(channels)

    quality = []
    for channel in channels:
        values = pd.to_numeric(frame[channel], errors="coerce").to_numpy(np.float64)
        record = _channel_quality(str(channel), values)
        quality.append(record)
        if record["n_nan"]:
            warnings.append(f"nan:{channel}:{record['n_nan']}")
        if record["n_posinf"] or record["n_neginf"]:
            warnings.append(f"inf:{channel}")
        if record["is_constant"]:
            warnings.append(f"constant_channel:{channel}")
        if (record["saturation_fraction"] or 0) > SATURATION_FRACTION_WARN:
            warnings.append(f"possible_saturation:{channel}")
    result["channel_quality"] = quality
    result["warnings"] = warnings
    return result


def audit_segment(task: dict[str, Any]) -> dict[str, Any]:
    """Audit one segment across all its sensors.

    `task` carries the metadata row fields plus a `<sensor>_path` per sensor.
    Returns a flat manifest row plus nested per-sensor detail.
    """
    row: dict[str, Any] = {
        key: task.get(key)
        for key in (
            "machine", "segment_id", "source_csv", "file_split", "file_domain",
            "file_anomaly", "split_label", "anomaly_label",
            "domain_shift_op", "domain_shift_env",
        )
    }

    sensors: dict[str, dict] = {}
    warnings: list[str] = []
    spans: dict[str, tuple[float, float]] = {}

    for key in SENSOR_KEYS:
        path = task.get(f"{key}_path")
        if not path or (isinstance(path, float) and np.isnan(path)):
            row[f"{key}_readable"] = False
            row[f"{key}_samples"] = 0
            warnings.append(f"no_path:{key}")
            continue

        detail = audit_sensor_file(path, key)
        sensors[key] = detail
        readable = bool(detail.get("readable"))
        row[f"{key}_readable"] = readable
        row[f"{key}_samples"] = int(detail.get("n_rows", 0))
        row[f"{key}_n_channels"] = detail.get("n_data_channels")
        row[f"{key}_file_bytes"] = detail.get("file_bytes")

        timing = detail.get("timing")
        if timing:
            row[f"{key}_rate_median_dt"] = timing["rate_from_median_dt"]
            row[f"{key}_rate_span"] = timing["rate_from_span"]
            row[f"{key}_rate_mean_dt"] = timing["rate_from_mean_dt"]
            row[f"{key}_duration_s"] = timing["span_s"]
            row[f"{key}_t_start"] = timing["t_start"]
            row[f"{key}_t_end"] = timing["t_end"]
            row[f"{key}_n_gaps"] = timing["n_gaps"]
            row[f"{key}_precision_limited"] = timing["precision_limited"]
            row[f"{key}_quantised"] = timing["quantised"]
            row[f"{key}_timestamp_resolution_s"] = timing["timestamp_resolution_s"]
            row[f"{key}_n_unique_dt"] = timing["n_unique_dt"]
            row[f"{key}_quantisation_bias_pct"] = timing["quantisation_rate_bias_pct"]
            row[f"{key}_frac_dt_irregular"] = timing["frac_dt_irregular"]
            row[f"{key}_dt_p01"] = timing["dt_p01"]
            row[f"{key}_dt_p99"] = timing["dt_p99"]
            row[f"{key}_dt_std"] = timing["dt_std"]
            row[f"{key}_drift_ratio"] = detail.get("drift_rate_ratio")
            if timing["n_samples"] >= 2:
                spans[key] = (timing["t_start"], timing["t_end"])

        warnings.extend(f"{key}:{w}" for w in detail.get("warnings", []))

    row["n_sensors_readable"] = sum(
        1 for key in SENSOR_KEYS if row.get(f"{key}_readable")
    )
    row["triplet_complete"] = row["n_sensors_readable"] == len(SENSOR_KEYS)

    if len(spans) >= 2:
        alignment = analyse_alignment(spans)
        row.update(
            start_spread_s=alignment.start_spread_s,
            end_spread_s=alignment.end_spread_s,
            intersection_s=alignment.intersection_s,
            union_s=alignment.union_s,
            coverage_ratio=alignment.coverage_ratio,
        )
        # Cost of the reference pipeline's crop: sensor intersection, plus the
        # documented 35 ms gyroscope warm-up removal.
        keep_start = alignment.intersection_start
        if "gyro" in spans:
            keep_start = max(keep_start, spans["gyro"][0] + GYRO_WARMUP_S)
        keep = (keep_start, alignment.intersection_end)
        row["usable_s_after_crop"] = float(keep[1] - keep[0])
        for key, span in spans.items():
            row[f"{key}_discarded_fraction"] = discarded_fraction(span, keep)
        if row["usable_s_after_crop"] <= 0:
            warnings.append("no_usable_common_interval")
        if alignment.intersection_s <= 0:
            warnings.append("sensors_do_not_overlap")

    n_info, n_defect = classify_warnings(warnings)
    row["n_warnings"] = len(warnings)
    row["n_info_warnings"] = n_info
    row["n_defect_warnings"] = n_defect
    row["clean"] = bool(n_defect == 0 and row["triplet_complete"])
    row["warnings"] = ";".join(warnings)
    return {"row": row, "sensors": sensors}
