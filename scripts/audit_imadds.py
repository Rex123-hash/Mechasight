"""MechaSight Phase 1 — reproducible IMAD-DS dataset auditor.

Answers the Phase 0 open questions against the real release and writes
machine-readable outputs to reports/dataset_audit/.

Usage:
    python scripts/audit_imadds.py --data-root data/raw/extracted
    python scripts/audit_imadds.py --data-root data/raw/extracted --limit 20
    python scripts/audit_imadds.py --data-root data/raw/extracted --machine BrushlessMotor

Phase 1 scope: measures only. No resampling, filtering, normalisation, windowing
or feature extraction happens anywhere in this pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.audit import SENSOR_KEYS, audit_segment  # noqa: E402
from src.data.balance import analyse_balance, condition_class_table  # noqa: E402
from src.data.discovery import find_machine_roots, scan_machine, summarise_layout  # noqa: E402
from src.data.temporal import analyse_temporal_structure, temporal_table  # noqa: E402
from src.data.metadata import (  # noqa: E402
    check_label_consistency,
    load_machine_metadata,
    resolve_sensor_paths,
)

# Published Table 2 counts, quoted for comparison only. The audit reports what is
# actually present; a mismatch is a finding, never something to reconcile away.
PUBLISHED_COUNTS = {
    "RoboticArm": {
        ("normal", "source", "train"): 1812, ("normal", "target", "train"): 27,
        ("normal", "source", "test"): 116, ("anomaly", "source", "test"): 116,
        ("normal", "target", "test"): 116, ("anomaly", "target", "test"): 116,
    },
    "BrushlessMotor": {
        ("normal", "source", "train"): 1263, ("normal", "target", "train"): 18,
        ("normal", "source", "test"): 78, ("anomaly", "source", "test"): 78,
        ("normal", "target", "test"): 78, ("anomaly", "target", "test"): 78,
    },
}


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return str(obj)


def rate_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    """Q1 — sampling-rate distributions per machine, sensor and estimator."""
    rows = []
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    for machine, part in manifest.groupby("machine"):
        for sensor in SENSOR_KEYS:
            for estimator in ("rate_median_dt", "rate_span", "rate_mean_dt"):
                column = f"{sensor}_{estimator}"
                if column not in part.columns:
                    continue
                values = pd.to_numeric(part[column], errors="coerce").dropna()
                values = values[np.isfinite(values)]
                if values.empty:
                    continue
                entry = {
                    "machine": machine, "sensor": sensor, "estimator": estimator,
                    "n_segments": int(values.size),
                    "mean": float(values.mean()), "median": float(values.median()),
                    "std": float(values.std()), "min": float(values.min()),
                    "max": float(values.max()),
                }
                for p in percentiles:
                    entry[f"p{p:02d}"] = float(np.percentile(values, p))
                entry["range_pct_of_median"] = (
                    float(100 * (values.max() - values.min()) / values.median())
                    if values.median() else float("nan")
                )
                rows.append(entry)

            flag = f"{sensor}_precision_limited"
            if flag in part.columns and rows:
                fraction = float(pd.Series(part[flag]).fillna(False).mean())
                for row in rows[-3:]:
                    if row["machine"] == machine and row["sensor"] == sensor:
                        row["fraction_precision_limited"] = fraction
    return pd.DataFrame(rows)


def duration_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    """Q2 — segment-duration distributions, sliced by the design factors."""
    rows = []
    groupings = [
        ("machine", ["machine"]),
        ("machine+split", ["machine", "file_split"]),
        ("machine+split+domain", ["machine", "file_split", "file_domain"]),
        ("machine+split+domain+class", ["machine", "file_split", "file_domain", "file_anomaly"]),
    ]
    for sensor in SENSOR_KEYS:
        column = f"{sensor}_duration_s"
        if column not in manifest.columns:
            continue
        for label, keys in groupings:
            for group, part in manifest.groupby(keys, dropna=False):
                values = pd.to_numeric(part[column], errors="coerce").dropna()
                values = values[np.isfinite(values)]
                if values.empty:
                    continue
                group = group if isinstance(group, tuple) else (group,)
                rows.append({
                    "grouping": label,
                    **dict(zip(keys, map(str, group), strict=False)),
                    "sensor": sensor,
                    "n_segments": int(values.size),
                    "min_s": float(values.min()),
                    "p05_s": float(np.percentile(values, 5)),
                    "median_s": float(values.median()),
                    "mean_s": float(values.mean()),
                    "p95_s": float(np.percentile(values, 95)),
                    "max_s": float(values.max()),
                    "std_s": float(values.std()),
                    "n_distinct_rounded_100ms": int(values.round(1).nunique()),
                })
    return pd.DataFrame(rows)


def alignment_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    """Q6 — sensor synchronisation statistics."""
    columns = [
        "start_spread_s", "end_spread_s", "intersection_s", "union_s",
        "coverage_ratio", "usable_s_after_crop",
        *[f"{s}_discarded_fraction" for s in SENSOR_KEYS],
    ]
    rows = []
    for machine, part in manifest.groupby("machine"):
        for column in columns:
            if column not in part.columns:
                continue
            values = pd.to_numeric(part[column], errors="coerce").dropna()
            values = values[np.isfinite(values)]
            if values.empty:
                continue
            rows.append({
                "machine": machine, "metric": column, "n_segments": int(values.size),
                "min": float(values.min()),
                "p05": float(np.percentile(values, 5)),
                "median": float(values.median()),
                "mean": float(values.mean()),
                "p95": float(np.percentile(values, 95)),
                "max": float(values.max()),
                "std": float(values.std()),
            })
    return pd.DataFrame(rows)


def count_comparison(machine: str, metadata: pd.DataFrame,
                     manifest: pd.DataFrame) -> pd.DataFrame:
    """Q5 — published vs metadata vs actually-readable counts, per cell."""
    published = PUBLISHED_COUNTS.get(machine, {})
    keys = set(published)
    if not metadata.empty:
        keys |= set(
            metadata.groupby(["file_anomaly", "file_domain", "file_split"]).groups
        )

    rows = []
    for key in sorted(keys):
        anomaly, domain, split = key
        meta_mask = (
            (metadata["file_anomaly"] == anomaly)
            & (metadata["file_domain"] == domain)
            & (metadata["file_split"] == split)
        ) if not metadata.empty else pd.Series(dtype=bool)
        n_meta = int(meta_mask.sum()) if len(meta_mask) else 0

        if not manifest.empty:
            man_mask = (
                (manifest["file_anomaly"] == anomaly)
                & (manifest["file_domain"] == domain)
                & (manifest["file_split"] == split)
            )
            n_triplets = int(man_mask.sum())
            n_complete = int((man_mask & manifest["triplet_complete"]).sum())
            n_clean = int(
                (man_mask & manifest["triplet_complete"]
                 & (manifest["n_defect_warnings"] == 0)).sum()
            )
        else:
            n_triplets = n_complete = n_clean = 0

        n_pub = published.get(key)
        rows.append({
            "machine": machine, "anomaly": anomaly, "domain": domain, "split": split,
            "published_paper": n_pub,
            "metadata_rows": n_meta,
            "sensor_triplets_present": n_triplets,
            "triplets_fully_readable": n_complete,
            "triplets_readable_no_defects": n_clean,
            "metadata_vs_published": None if n_pub is None else n_meta - n_pub,
            "readable_vs_published": None if n_pub is None else n_complete - n_pub,
        })
    return pd.DataFrame(rows)


def label_condition_counts(metadata: pd.DataFrame) -> pd.DataFrame:
    """Q3/Q4 — label vocabulary crossed with conditions, splits and domains."""
    keys = [
        c for c in ("machine", "file_split", "file_domain", "anomaly_label",
                    "domain_shift_op", "domain_shift_env")
        if c in metadata.columns
    ]
    if not keys:
        return pd.DataFrame()
    return (
        metadata.groupby(keys, dropna=False).size()
        .reset_index(name="n_segments")
        .sort_values(keys)
    )


def audit_machine(machine: str, root: Path, jobs: int, limit: int | None,
                  permutations: int) -> dict:
    print(f"\n=== {machine} ===  root={root}")

    layout = scan_machine(machine, root)
    summary = summarise_layout(layout)
    print(f"  {summary['n_metadata_csv']} metadata CSV, "
          f"{summary['n_parquet']} parquet, "
          f"{summary['n_other_files']} other files")
    if summary["n_zero_byte_files"]:
        print(f"  WARNING: {summary['n_zero_byte_files']} zero-byte files")
    if summary["n_parquet_unclassified"]:
        print(f"  WARNING: {summary['n_parquet_unclassified']} unclassified parquet")

    metadata, file_infos = load_machine_metadata(layout.metadata_files)
    if metadata.empty:
        print("  ERROR: no metadata rows loaded")
        return {"machine": machine, "layout": summary, "metadata_files": file_infos,
                "manifest": pd.DataFrame(), "metadata": metadata,
                "counts": pd.DataFrame(), "channels": pd.DataFrame(),
                "temporal": {}, "temporal_table": pd.DataFrame()}

    metadata["machine"] = machine
    consistency = check_label_consistency(metadata)
    metadata = resolve_sensor_paths(metadata, root)
    print(f"  {len(metadata)} metadata rows; "
          f"{consistency['n_duplicate_segment_ids']} duplicate segment ids")

    for key in SENSOR_KEYS:
        missing = int((~metadata[f"{key}_exists"]).sum())
        if missing:
            print(f"  WARNING: {missing} rows with no {key} file on disk")

    # Parquet files present on disk that no metadata row references.
    referenced = set()
    for key in SENSOR_KEYS:
        referenced |= {
            Path(p).resolve() for p in metadata[f"{key}_path"].dropna().tolist()
        }
    on_disk = {p.resolve() for p in layout.parquet_files}
    orphan_files = sorted(on_disk - referenced)
    summary["n_orphan_parquet_files"] = len(orphan_files)
    summary["orphan_parquet_examples"] = [str(p) for p in orphan_files[:20]]
    if orphan_files:
        print(f"  WARNING: {len(orphan_files)} parquet files with no metadata row")

    tasks = metadata.to_dict("records")
    if limit:
        tasks = tasks[:limit]
    print(f"  auditing {len(tasks)} segments with {jobs} worker(s) ...", flush=True)

    rows, sensor_details = [], {}
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(audit_segment, t): t for t in tasks}
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                rows.append(result["row"])
                sensor_details[result["row"]["segment_id"]] = result["sensors"]
                if i % 200 == 0:
                    print(f"    {i}/{len(tasks)}", flush=True)
    else:
        for i, task in enumerate(tasks, 1):
            result = audit_segment(task)
            rows.append(result["row"])
            sensor_details[result["row"]["segment_id"]] = result["sensors"]
            if i % 200 == 0:
                print(f"    {i}/{len(tasks)}", flush=True)

    manifest = pd.DataFrame(rows)
    print(f"  complete triplets: {int(manifest['triplet_complete'].sum())}/{len(manifest)}")
    print(f"  segments with defect warnings: {int((manifest['n_defect_warnings'] > 0).sum())}")
    print(f"  segments with informational flags only: "
          f"{int(((manifest['n_defect_warnings'] == 0) & (manifest['n_info_warnings'] > 0)).sum())}")

    # Q4 balance, on test metadata only (training data is normal-only).
    balance = {}
    if "anomaly_label" in metadata.columns:
        test = metadata[metadata["file_split"] == "test"].copy()
        if not test.empty:
            test["is_anomaly"] = ~test["anomaly_label"].astype("string").str.lower().eq("normal")
            balance = analyse_balance(test, n_permutations=permutations)
            balance["machine"] = machine

    # Q8 -- recording-time structure and class/time confound.
    temporal = analyse_temporal_structure(metadata)
    temporal["machine"] = machine
    if temporal.get("classes_temporally_disjoint"):
        print(f"  FINDING: normal and anomalous recordings are temporally DISJOINT "
              f"(gap {abs(temporal['normal_anomaly_overlap_s']) / 60:.1f} min)")

    # Channel schema actually observed, for Q10.
    schema: dict[str, dict] = {}
    for detail in sensor_details.values():
        for key, info in detail.items():
            if not info.get("readable"):
                continue
            entry = schema.setdefault(key, {"column_sets": {}, "time_dtypes": {}})
            signature = "|".join(info.get("columns", []))
            entry["column_sets"][signature] = entry["column_sets"].get(signature, 0) + 1
            dtype = str(info.get("time_dtype"))
            entry["time_dtypes"][dtype] = entry["time_dtypes"].get(dtype, 0) + 1

    channel_rows = []
    for detail in sensor_details.values():
        for key, info in detail.items():
            for record in info.get("channel_quality", []):
                channel_rows.append({"machine": machine, "sensor": key, **record})
    channels = pd.DataFrame(channel_rows)

    return {
        "machine": machine,
        "layout": summary,
        "metadata_files": file_infos,
        "label_consistency": consistency,
        "metadata": metadata,
        "manifest": manifest,
        "balance": balance,
        "schema": schema,
        "channels": channels,
        "counts": count_comparison(machine, metadata, manifest),
        "temporal": temporal,
        "temporal_table": temporal_table(metadata),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/raw/extracted", type=Path)
    parser.add_argument("--out", default="reports/dataset_audit", type=Path)
    parser.add_argument("--machine", default=None, help="audit only this machine")
    parser.add_argument("--limit", type=int, default=None,
                        help="audit only the first N segments per machine (smoke test)")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--permutations", type=int, default=2000,
                        help="permutations for the mutual-information null")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    roots = find_machine_roots(args.data_root)
    if args.machine:
        roots = {k: v for k, v in roots.items() if k == args.machine}
    if not roots:
        print(f"ERROR: no machine directories under {args.data_root}")
        return 1
    print(f"machines discovered: {list(roots)}")

    results = [
        audit_machine(machine, root, args.jobs, args.limit, args.permutations)
        for machine, root in sorted(roots.items())
    ]

    manifests = [r["manifest"] for r in results if not r["manifest"].empty]
    manifest = pd.concat(manifests, ignore_index=True) if manifests else pd.DataFrame()
    metas = [r["metadata"] for r in results if not r["metadata"].empty]
    metadata = pd.concat(metas, ignore_index=True) if metas else pd.DataFrame()

    if not manifest.empty:
        manifest.to_csv(args.out / "segment_manifest.csv", index=False)
        # Quality report carries real defects; informational flags (e.g. timestamp
        # quantisation) are on every row of the manifest and are not defects.
        quality = (
            manifest[manifest["n_defect_warnings"] > 0]
            if "n_defect_warnings" in manifest else pd.DataFrame()
        )
        quality.to_csv(args.out / "quality_report.csv", index=False)
        rate_summary(manifest).to_csv(args.out / "sampling_rate_summary.csv", index=False)
        duration_summary(manifest).to_csv(args.out / "duration_summary.csv", index=False)
        alignment_summary(manifest).to_csv(
            args.out / "timestamp_alignment_summary.csv", index=False)

    if not metadata.empty:
        label_condition_counts(metadata).to_csv(
            args.out / "label_condition_counts.csv", index=False)

    counts = [r["counts"] for r in results if not r["counts"].empty]
    if counts:
        pd.concat(counts, ignore_index=True).to_csv(
            args.out / "count_comparison.csv", index=False)

    temporal_tables = [r["temporal_table"] for r in results if not r["temporal_table"].empty]
    if temporal_tables:
        pd.concat(temporal_tables, ignore_index=True).to_csv(
            args.out / "recording_times.csv", index=False)

    channels = [r["channels"] for r in results if not r["channels"].empty]
    if channels:
        pd.concat(channels, ignore_index=True).to_csv(
            args.out / "channel_quality.csv", index=False)

    # Condition x class tables for the shortcut audit.
    tables = []
    for result in results:
        meta = result["metadata"]
        if meta.empty or "anomaly_label" not in meta.columns:
            continue
        test = meta[meta["file_split"] == "test"].copy()
        if test.empty:
            continue
        test["is_anomaly"] = ~test["anomaly_label"].astype("string").str.lower().eq("normal")
        for column in ("domain_shift_op", "domain_shift_env"):
            if column in test.columns:
                tables.append(
                    condition_class_table(test, column, extra_group=["machine", "file_domain"])
                )
    if tables:
        pd.concat(tables, ignore_index=True).to_csv(
            args.out / "condition_class_balance.csv", index=False)

    findings = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(args.data_root),
        "limit_applied": args.limit,
        "machines": [
            {
                "machine": r["machine"],
                "layout": r["layout"],
                "metadata_files": r["metadata_files"],
                "label_consistency": r.get("label_consistency"),
                "balance": r.get("balance"),
                "temporal_structure": r.get("temporal"),
                "observed_schema": r.get("schema"),
            }
            for r in results
        ],
    }
    (args.out / "audit_findings.json").write_text(
        json.dumps(findings, indent=2, default=_json_default), encoding="utf-8")

    print(f"\noutputs written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
