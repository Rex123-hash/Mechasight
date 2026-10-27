"""Reusable Phase-2 preprocessing orchestration; no model code lives here."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from src.data.discovery import find_machine_roots, scan_machine
from src.data.metadata import load_machine_metadata, resolve_sensor_paths
from src.data.temporal import parse_segment_times
from src.features.extraction import frequency_features, flatten_features, time_features
from src.preprocessing.cache import RaggedHDF5Writer
from src.preprocessing.config import OFFICIAL_PROTOCOL, PreprocessingConfig
from src.preprocessing.manifest import JsonLinesManifestWriter
from src.preprocessing.normalization import FittedChannelStandardizer, standardize_window
from src.preprocessing.protected_access import is_protected_split, require_protected_read
from src.preprocessing.splits import CONSUMING_SPLITS
from src.preprocessing.signals import (
    SENSOR_KEYS,
    SensorSignal,
    apply_startup_transformations,
    effective_rate_hz,
    iter_aligned_windows,
    read_sensor_parquet,
    resample_regular,
)


@dataclass
class PipelineCounts:
    segments_read: int = 0
    windows_written: int = 0
    accelerometer_files_affected: int = 0
    accelerometer_samples_dropped: int = 0
    gyro_files_affected: int = 0
    gyro_samples_dropped: int = 0


def load_raw_catalog(data_root: str | Path) -> pd.DataFrame:
    outputs = []
    for machine, root in sorted(find_machine_roots(Path(data_root)).items()):
        layout = scan_machine(machine, root)
        metadata, _ = load_machine_metadata(layout.metadata_files)
        metadata["machine"] = machine
        metadata = resolve_sensor_paths(metadata, root)
        parsed = parse_segment_times(metadata["segment_id"])
        metadata["parent_recording"] = parsed["recording_stamp"].astype("string")
        outputs.append(metadata)
    if not outputs:
        raise FileNotFoundError(f"no IMAD-DS machines beneath {data_root}")
    return pd.concat(outputs, ignore_index=True)


def official_assignments(catalog: pd.DataFrame, regime: str = "T") -> pd.DataFrame:
    """Released split membership; Protocol A validation remains window-level."""

    if regime != "T":
        raise ValueError("faithful official baseline reproduction is defined only for Regime T")
    output = catalog.copy()
    output["split"] = output["file_split"]
    output["regime"] = regime
    output["protocol"] = OFFICIAL_PROTOCOL
    output["data_pool"] = output["split"].map({
        "train": "released_training", "test": "released_evaluation"
    })
    return output


def merge_protocol_b_assignments(
    catalog: pd.DataFrame,
    assignments: pd.DataFrame,
    regime: str,
    *,
    split_version: str | None = None,
) -> pd.DataFrame:
    if split_version is not None:
        if "split_version" not in assignments:
            raise ValueError("split manifest lacks split_version")
        observed = set(assignments["split_version"].dropna().astype(str))
        if observed != {split_version}:
            raise ValueError(
                f"split manifest version {sorted(observed)} is incompatible with {split_version}"
            )
    selected = assignments[assignments["regime"].eq(regime)].copy()
    key = ["machine", "segment_id"]
    columns = key + ["split", "regime", "protocol", "parent_recording"]
    for optional in ("data_pool", "split_version", "preprocessing_version"):
        if optional in selected:
            columns.append(optional)
    merged = catalog.drop(columns=["parent_recording"], errors="ignore").merge(
        selected[columns], on=key, how="inner", validate="one_to_one",
    )
    if len(merged) != len(catalog):
        raise ValueError(f"split manifest covers {len(merged)} of {len(catalog)} catalog rows")
    return merged


def _load_signals(row: pd.Series) -> dict[str, SensorSignal]:
    return {
        sensor: read_sensor_parquet(row[f"{sensor}_path"], sensor)
        for sensor in SENSOR_KEYS
    }


def _normalize(
    signals: dict[str, SensorSignal],
    config: PreprocessingConfig,
    scalers: dict[str, FittedChannelStandardizer] | None,
) -> dict[str, SensorSignal]:
    output = {}
    for sensor, signal in signals.items():
        if config.normalization == "none":
            values = signal.values
        elif config.normalization == "std_window":
            values = standardize_window(signal.values)
        elif config.normalization == "train_zscore":
            if scalers is None or sensor not in scalers:
                raise ValueError("train_zscore requires a train-fitted scaler for every modality")
            values = scalers[sensor].transform(signal.values)
        else:
            raise ValueError(f"unsupported normalization {config.normalization!r}")
        output[sensor] = SensorSignal(
            sensor=signal.sensor,
            timestamps=signal.timestamps,
            values=values,
            channels=signal.channels,
            raw_sample_count=signal.raw_sample_count,
            transformed_sample_count=len(values),
            transformation=signal.transformation + f"|normalization:{config.normalization}",
        )
    return output


def _sampling_metadata(signals: dict[str, SensorSignal]) -> str:
    return json.dumps({
        sensor: {
            "effective_rate_hz": effective_rate_hz(signal),
            "original_sample_count": signal.raw_sample_count,
            "transformed_sample_count": signal.transformed_sample_count,
            "transformation": signal.transformation,
            "timestamp_preserved": True,
        }
        for sensor, signal in signals.items()
    }, sort_keys=True, separators=(",", ":"))


def _manifest_record(
    row: pd.Series,
    window_index: int,
    start_s: float,
    end_s: float,
    signals: dict[str, SensorSignal],
    config: PreprocessingConfig,
) -> dict:
    window_id = (
        f"{config.config_id}:{row['machine']}:{config.regime}:"
        f"{row['segment_id']}:{window_index:04d}"
    )
    raw_references = {sensor: str(row[f"{sensor}_path"]) for sensor in SENSOR_KEYS}
    return {
        "machine": row["machine"],
        "sensor_modalities": json.dumps(list(SENSOR_KEYS)),
        "raw_file_references": json.dumps(raw_references, sort_keys=True),
        "source_file": raw_references["mic"],
        "segment_id": row["segment_id"],
        "original_segment_id": row["segment_id"],
        "parent_recording": row["parent_recording"],
        "window_id": window_id,
        "window_index": window_index,
        "window_start_s": start_s,
        "window_end_s": end_s,
        "protocol": config.protocol,
        "regime": config.regime,
        "split": row["split"],
        "data_pool": row.get("data_pool", "unspecified"),
        "domain": row["file_domain"],
        "operating_condition": row["domain_shift_op"],
        "background_condition": row["domain_shift_env"],
        "anomaly_label": row["anomaly_label"],
        "sampling_rate_metadata": _sampling_metadata(signals),
        "preprocessing_config_id": config.config_id,
        "preprocessing_version": config.version,
        "split_version": config.split_version,
    }


def iter_processed_windows(
    catalog: pd.DataFrame,
    config: PreprocessingConfig,
    *,
    split: str,
    scalers: dict[str, FittedChannelStandardizer] | None = None,
    max_segments: int | None = None,
    counts: PipelineCounts | None = None,
) -> Iterator[tuple[dict, dict[str, SensorSignal], dict[str, float]]]:
    # Checked at call time, not at the first ``next()``, so a refused split
    # never yields a window and the refusal cannot be deferred past a scope.
    if split not in CONSUMING_SPLITS:
        raise ValueError(
            f"split {split!r} is consumed by no computation and must never be "
            "materialised into a cache, features, or a manifest"
        )
    if is_protected_split(split):
        require_protected_read(f"raw {split} windows")
    return _iter_processed_windows(
        catalog, config, split=split, scalers=scalers, max_segments=max_segments,
        counts=counts if counts is not None else PipelineCounts(),
    )


def _iter_processed_windows(
    catalog: pd.DataFrame,
    config: PreprocessingConfig,
    *,
    split: str,
    scalers: dict[str, FittedChannelStandardizer] | None,
    max_segments: int | None,
    counts: PipelineCounts,
) -> Iterator[tuple[dict, dict[str, SensorSignal], dict[str, float]]]:
    selected = catalog[
        catalog["split"].eq(split)
        & catalog["machine"].isin(config.machines)
        & catalog["regime"].eq(config.regime)
        & catalog["protocol"].eq(config.protocol)
    ].sort_values(["machine", "parent_recording", "segment_id"])
    if max_segments is not None:
        selected = selected.head(max_segments)
    for _, row in selected.iterrows():
        raw = _load_signals(row)
        transformed, log = apply_startup_transformations(raw, row["machine"], config)
        counts.segments_read += 1
        if log["accelerometer_samples_dropped"]:
            counts.accelerometer_files_affected += 1
            counts.accelerometer_samples_dropped += log["accelerometer_samples_dropped"]
        if log["gyro_samples_dropped"]:
            counts.gyro_files_affected += 1
            counts.gyro_samples_dropped += log["gyro_samples_dropped"]
        for window in iter_aligned_windows(transformed, config):
            normalized = _normalize(window.signals, config, scalers)
            record = _manifest_record(
                row, window.window_index, window.start_s, window.end_s, normalized, config
            )
            features = {}
            for sensor, signal in normalized.items():
                if "time" in config.features:
                    features.update(flatten_features(
                        sensor, signal.channels, time_features(signal.values)
                    ))
                if "frequency" in config.features:
                    target = config.resampling.target_rates_hz[row["machine"]][sensor]
                    regular = resample_regular(signal, target, method=config.resampling.method)
                    features.update(flatten_features(
                        sensor, regular.channels, frequency_features(regular.values, target)
                    ))
            counts.windows_written += 1
            yield record, normalized, features


def build_cache(
    catalog: pd.DataFrame,
    config: PreprocessingConfig,
    *,
    machine: str,
    split: str,
    cache_path: str | Path,
    manifest_path: str | Path,
    feature_path: str | Path | None = None,
    scalers: dict[str, FittedChannelStandardizer] | None = None,
    max_segments: int | None = None,
) -> PipelineCounts:
    counts = PipelineCounts()
    machine_catalog = catalog[catalog["machine"].eq(machine)]
    feature_handle = None
    if feature_path is not None:
        path = Path(feature_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        feature_handle = path.open("w", encoding="utf-8", newline="\n")
    try:
        with RaggedHDF5Writer(
            cache_path, config, machine=machine, split=split
        ) as cache, JsonLinesManifestWriter(manifest_path) as manifest:
            for record, signals, features in iter_processed_windows(
                machine_catalog, config, split=split, scalers=scalers,
                max_segments=max_segments, counts=counts,
            ):
                cache.append(record["window_id"], signals)
                manifest.append(record)
                if feature_handle is not None:
                    trace = {
                        key: record[key]
                        for key in (
                            "window_id", "machine", "protocol", "regime", "split",
                            "data_pool", "domain", "parent_recording", "segment_id", "original_segment_id",
                            "operating_condition", "background_condition", "anomaly_label",
                            "preprocessing_config_id", "preprocessing_version", "split_version",
                        )
                    }
                    feature_handle.write(json.dumps(
                        {**trace, **features},
                        sort_keys=True, separators=(",", ":"),
                    ) + "\n")
    finally:
        if feature_handle is not None:
            feature_handle.close()
    return counts
