"""Phase 1.5 — full-dataset investigation of two unresolved Phase 1 findings.

1. ACCELEROMETER FIRST-SAMPLE ARTEFACT. Phase 1 saw it in 37 of 40 sampled
   BrushlessMotor files. Before any "drop the first sample" rule can be
   justified, its prevalence must be measured across the whole release and
   checked for association with split, domain, class and condition.

   IMPORTANT: label associations are computed as a DIAGNOSTIC only. If the
   artefact were associated with the anomaly label, a label-driven cleaning rule
   would be unavailable at deployment and would leak. The rule this evidence can
   support must depend only on information present at inference time.

2. THE `_corr` MARKER on 85 RoboticArm segment ids, undocumented in the paper,
   the Zenodo description and the authors' released code. Tested for association
   with every available metadata field and with timing characteristics.

Measurement only. No preprocessing decision is executed here.

Usage:
    python scripts/investigate_artifacts.py --data-root data/raw/extracted
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.discovery import find_machine_roots, scan_machine  # noqa: E402
from src.data.metadata import load_machine_metadata, resolve_sensor_paths  # noqa: E402
from src.data.temporal import parse_segment_times  # noqa: E402

ACC_CHANNELS = ["A_x [g]", "A_y [g]", "A_z [g]"]
# Full-scale rail observed in Phase 1. Compared with a tolerance rather than
# exactly, so a differently-rounded rail value would still be caught.
RAIL_ABS_G = 15.988832
RAIL_TOL = 1e-4


def scan_accelerometer(path: str, n_head: int = 16) -> dict:
    """Locate full-scale rail samples in one accelerometer file."""
    frame = pd.read_parquet(path, columns=["Time"] + ACC_CHANNELS)
    values = frame[ACC_CHANNELS].to_numpy(np.float64)
    railed = np.abs(np.abs(values) - RAIL_ABS_G) < RAIL_TOL
    rows_railed = railed.any(axis=1)
    indices = np.flatnonzero(rows_railed)

    return {
        "n_rows": int(values.shape[0]),
        "n_railed_rows": int(indices.size),
        "first_row_railed": bool(rows_railed[0]) if values.size else False,
        "n_railed_axes_in_first_row": int(railed[0].sum()) if values.size else 0,
        "railed_only_in_first_row": bool(indices.size == 1 and indices[0] == 0),
        "max_railed_index": int(indices.max()) if indices.size else -1,
        "n_railed_in_first_n": int(np.count_nonzero(indices < n_head)),
        "first_row_values": values[0].tolist() if values.size else [],
    }


def investigate(machine: str, root: Path, limit: int | None) -> dict:
    layout = scan_machine(machine, root)
    metadata, _ = load_machine_metadata(layout.metadata_files)
    metadata["machine"] = machine
    metadata = resolve_sensor_paths(metadata, root)
    metadata = metadata.join(parse_segment_times(metadata["segment_id"]))

    records = metadata.to_dict("records")
    if limit:
        records = records[:limit]

    print(f"  {machine}: scanning {len(records)} accelerometer files ...", flush=True)
    rows = []
    for i, record in enumerate(records, 1):
        path = record.get("acc_path")
        if not isinstance(path, str):
            continue
        result = scan_accelerometer(path)
        rows.append({
            "machine": machine,
            "segment_id": record["segment_id"],
            "file_split": record["file_split"],
            "file_domain": record["file_domain"],
            "anomaly_label": record["anomaly_label"],
            "domain_shift_op": record["domain_shift_op"],
            "domain_shift_env": record["domain_shift_env"],
            "recording_stamp": record["recording_stamp"],
            "has_corr_marker": bool(record["has_corr_marker"]),
            **{k: v for k, v in result.items() if k != "first_row_values"},
            "first_row_max_abs": max(abs(v) for v in result["first_row_values"])
            if result["first_row_values"] else np.nan,
        })
        if i % 500 == 0:
            print(f"    {i}/{len(records)}", flush=True)
    return {"machine": machine, "frame": pd.DataFrame(rows), "metadata": metadata}


def _assoc(frame: pd.DataFrame, flag: str, field: str) -> dict:
    """Chi-square association between a boolean flag and a categorical field."""
    table = pd.crosstab(frame[field], frame[flag])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {"field": field, "testable": False,
                "note": "flag or field has a single value"}
    chi2, p_value, _, _ = stats.chi2_contingency(table)
    return {
        "field": field,
        "testable": True,
        "chi2": float(chi2),
        "p_value": float(p_value),
        "rate_by_value": {
            str(k): float(v) for k, v in frame.groupby(field)[flag].mean().items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/raw/extracted", type=Path)
    parser.add_argument("--out", default="reports/dataset_audit", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    roots = find_machine_roots(args.data_root)
    if not roots:
        print(f"ERROR: no machines under {args.data_root}")
        return 1

    findings: dict = {"accelerometer_artefact": {}, "corr_marker": {}}
    frames = []

    for machine, root in sorted(roots.items()):
        result = investigate(machine, root, args.limit)
        frame = result["frame"]
        if frame.empty:
            continue
        frames.append(frame)

        entry: dict = {
            "n_segments_scanned": int(len(frame)),
            "n_with_any_railed_row": int((frame["n_railed_rows"] > 0).sum()),
            "fraction_with_any_railed_row": float((frame["n_railed_rows"] > 0).mean()),
            "n_first_row_railed": int(frame["first_row_railed"].sum()),
            "fraction_first_row_railed": float(frame["first_row_railed"].mean()),
            "n_railed_only_in_first_row": int(frame["railed_only_in_first_row"].sum()),
            "railed_rows_per_file": {
                "min": int(frame["n_railed_rows"].min()),
                "median": float(frame["n_railed_rows"].median()),
                "max": int(frame["n_railed_rows"].max()),
            },
            "max_railed_row_index_over_all_files": int(frame["max_railed_index"].max()),
            "n_files_with_railed_row_beyond_first": int(
                (frame["max_railed_index"] > 0).sum()
            ),
            "n_railed_axes_in_first_row_counts": {
                str(k): int(v)
                for k, v in frame["n_railed_axes_in_first_row"].value_counts().items()
            },
        }
        # Diagnostic only -- see module docstring.
        entry["association_with_metadata"] = {
            field: _assoc(frame, "first_row_railed", field)
            for field in ("file_split", "file_domain", "anomaly_label",
                          "domain_shift_op", "domain_shift_env")
        }
        findings["accelerometer_artefact"][machine] = entry

        print(f"\n  --- {machine}: accelerometer rail artefact ---")
        print(f"    first row railed: {entry['n_first_row_railed']}/{entry['n_segments_scanned']}"
              f" ({100 * entry['fraction_first_row_railed']:.1f}%)")
        print(f"    any railed row  : {entry['n_with_any_railed_row']}"
              f" ({100 * entry['fraction_with_any_railed_row']:.1f}%)")
        print(f"    railed ONLY in first row: {entry['n_railed_only_in_first_row']}")
        print(f"    highest railed row index anywhere: {entry['max_railed_row_index_over_all_files']}")

        # ---- corr marker ----
        if frame["has_corr_marker"].any():
            corr_entry: dict = {
                "n_with_marker": int(frame["has_corr_marker"].sum()),
                "n_total": int(len(frame)),
                "association": {
                    field: _assoc(frame, "has_corr_marker", field)
                    for field in ("file_split", "file_domain", "anomaly_label",
                                  "domain_shift_op", "domain_shift_env")
                },
            }
            marked = frame[frame["has_corr_marker"]]
            corr_entry["n_parent_recordings_with_marker"] = int(
                marked["recording_stamp"].nunique()
            )
            per_recording = frame.groupby("recording_stamp")["has_corr_marker"]
            mixed = per_recording.nunique()
            corr_entry["n_recordings_mixing_marked_and_unmarked"] = int((mixed > 1).sum())
            corr_entry["n_recordings_entirely_marked"] = int(
                (per_recording.mean() == 1.0).sum()
            )
            corr_entry["marked_recording_stamps"] = sorted(
                marked["recording_stamp"].unique().tolist()
            )
            findings["corr_marker"][machine] = corr_entry

            print(f"\n  --- {machine}: _corr marker ---")
            print(f"    {corr_entry['n_with_marker']}/{corr_entry['n_total']} segments,"
                  f" in {corr_entry['n_parent_recordings_with_marker']} parent recordings")
            print(f"    recordings mixing marked+unmarked: "
                  f"{corr_entry['n_recordings_mixing_marked_and_unmarked']}")
            for field, assoc in corr_entry["association"].items():
                if assoc.get("testable"):
                    print(f"    {field:18s} chi2 p={assoc['p_value']:.4f}")

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(args.out / "accelerometer_artefact_scan.csv", index=False)

    (args.out / "artefact_investigation.json").write_text(
        json.dumps(findings, indent=2, default=str), encoding="utf-8")
    print(f"\noutputs written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
