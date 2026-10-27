"""Q7 — does the released microphone data already contain the simulated noise?

The paper describes mixing factory background noise into the machine audio at
SNR -4 dB via a simulated room. The Zenodo release ships only parquet + csv, with
no noise-source or impulse-response files, so the question is whether the mixing
was applied before release or is left to the user.

Decisive test, using metadata that exists independently of the signal: each
segment carries a `domain_shift_env` label naming which background recording
applies. Hold the machine's operating condition fixed and vary only that label.

  - If the release is pre-mixed, segments with different env labels must differ
    systematically in microphone statistics.
  - If the release is anechoic and mixing is left to the user, the env label is
    inert metadata and microphone statistics cannot depend on it.

The same comparison is run on the accelerometer and gyroscope, which tests the
paper's separate claim that environmental noise was applied to audio only.

SCOPE: reads TRAINING NORMAL segments only. No test-set signal content is read,
so nothing here can inform feature design in a leaky way (Phase 1 rule 7).
This script measures; it makes no preprocessing decision.

Usage:
    python scripts/audio_noise_probe.py --data-root data/raw/extracted
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.discovery import find_machine_roots, scan_machine  # noqa: E402
from src.data.metadata import load_machine_metadata, resolve_sensor_paths  # noqa: E402

SENSOR_CHANNELS = {
    "mic": ["MIC [Waveform]"],
    "acc": ["A_x [g]", "A_y [g]", "A_z [g]"],
    "gyro": ["G_x [mdps]", "G_y [mdps]", "G_z [mdps]"],
}
# Sampling rates are derived per file from its own timestamps rather than being
# assumed: the Phase 1 audit found the two machines run at different effective
# rates (RoboticArm IMU ~7063 Hz, BrushlessMotor IMU ~7136 Hz).


def segment_features(path: str, sensor: str) -> dict:
    """Coarse summary statistics for one sensor file."""
    frame = pd.read_parquet(path)
    channels = [c for c in SENSOR_CHANNELS[sensor] if c in frame.columns]
    if not channels:
        return {}

    time = frame["Time"].to_numpy(np.float64)
    rate = (len(time) - 1) / (time[-1] - time[0]) if len(time) > 1 else np.nan

    values = frame[channels].to_numpy(np.float64)
    # Drop the first sample: the accelerometer's first row is railed at full
    # scale in most files (a startup artefact found during the Phase 1 audit),
    # and one railed sample would dominate an RMS.
    values = values[1:]
    magnitude = np.sqrt(np.mean(values**2, axis=0)).mean()

    out = {"rms": float(magnitude), "rate_hz": float(rate)}

    freqs, psd = signal.welch(values[:, 0], fs=rate, nperseg=2048)
    total = float(np.trapezoid(psd, freqs))
    out["log_total_power"] = float(np.log10(total + 1e-30))
    for lo, hi in ((20, 200), (200, 1000), (1000, 3000)):
        band = (freqs >= lo) & (freqs < hi)
        if band.any():
            power = float(np.trapezoid(psd[band], freqs[band]))
            out[f"band_{lo}_{hi}_frac"] = power / total if total > 0 else np.nan
    return out


def probe_machine(machine: str, root: Path, per_group: int, seed: int) -> dict:
    layout = scan_machine(machine, root)
    metadata, _ = load_machine_metadata(layout.metadata_files)
    metadata["machine"] = machine
    metadata = resolve_sensor_paths(metadata, root)

    train_normal = metadata[
        (metadata["file_split"] == "train")
        & (metadata["anomaly_label"].astype("string").str.lower() == "normal")
    ].copy()
    if train_normal.empty:
        return {"machine": machine, "error": "no training normal segments"}

    rng = np.random.default_rng(seed)
    picked = []
    for _, group in train_normal.groupby(["domain_shift_op", "domain_shift_env"]):
        picked.append(
            group.sample(min(len(group), per_group),
                         random_state=int(rng.integers(1 << 31)))
        )
    sampled = pd.concat(picked, ignore_index=True)
    print(f"  {machine}: {len(sampled)} training-normal segments sampled from "
          f"{sampled.groupby(['domain_shift_op', 'domain_shift_env']).ngroups} "
          f"(speed x background) cells")

    rows = []
    for i, record in enumerate(sampled.to_dict("records"), 1):
        row = {
            "machine": machine,
            "segment_id": record["segment_id"],
            "domain_shift_op": record["domain_shift_op"],
            "domain_shift_env": record["domain_shift_env"],
        }
        for sensor in SENSOR_CHANNELS:
            path = record.get(f"{sensor}_path")
            if isinstance(path, str):
                for key, value in segment_features(path, sensor).items():
                    row[f"{sensor}_{key}"] = value
        rows.append(row)
        if i % 50 == 0:
            print(f"    {i}/{len(sampled)}", flush=True)

    features = pd.DataFrame(rows)

    # Within each speed, does the background label explain variation in each
    # sensor's statistics? Kruskal-Wallis is used because it assumes neither
    # normality nor equal variance across the background groups.
    results: dict = {"machine": machine, "n_segments": int(len(features)),
                     "tests": {}}
    for sensor in SENSOR_CHANNELS:
        column = f"{sensor}_rms"
        if column not in features.columns:
            continue
        per_speed = []
        for speed, part in features.groupby("domain_shift_op"):
            groups = [
                g[column].dropna().to_numpy()
                for _, g in part.groupby("domain_shift_env")
                if g[column].notna().sum() >= 3
            ]
            if len(groups) < 2:
                continue
            try:
                statistic, p_value = stats.kruskal(*groups)
            except ValueError:
                continue
            pooled = np.concatenate(groups)
            between = float(np.var([g.mean() for g in groups]))
            within = float(np.mean([g.var() for g in groups]))
            per_speed.append({
                "speed": str(speed),
                "n_groups": len(groups),
                "n": int(pooled.size),
                "kruskal_H": float(statistic),
                "p_value": float(p_value),
                "between_group_var": between,
                "within_group_var": within,
                "var_ratio": float(between / within) if within > 0 else float("nan"),
                "group_means": {
                    str(k): float(g[column].mean())
                    for k, g in part.groupby("domain_shift_env")
                },
            })
        if per_speed:
            p_values = [entry["p_value"] for entry in per_speed]
            results["tests"][sensor] = {
                "per_speed": per_speed,
                "n_speeds_tested": len(per_speed),
                "n_speeds_p_below_0.05": int(sum(p < 0.05 for p in p_values)),
                "median_p": float(np.median(p_values)),
                "median_var_ratio": float(
                    np.median([entry["var_ratio"] for entry in per_speed])
                ),
            }
    return {"summary": results, "features": features}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/raw/extracted", type=Path)
    parser.add_argument("--out", default="reports/dataset_audit", type=Path)
    parser.add_argument("--per-group", type=int, default=12,
                        help="training-normal segments sampled per (speed, background)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    roots = find_machine_roots(args.data_root)
    if not roots:
        print(f"ERROR: no machines under {args.data_root}")
        return 1

    summaries, frames = [], []
    for machine, root in sorted(roots.items()):
        result = probe_machine(machine, root, args.per_group, args.seed)
        if "error" in result:
            print(f"  {machine}: {result['error']}")
            continue
        summaries.append(result["summary"])
        frames.append(result["features"])

        print(f"\n  --- {machine}: does background label explain sensor statistics? ---")
        for sensor, entry in result["summary"]["tests"].items():
            print(f"    {sensor:5s}: significant in "
                  f"{entry['n_speeds_p_below_0.05']}/{entry['n_speeds_tested']} speeds"
                  f" | median p={entry['median_p']:.2e}"
                  f" | median between/within var={entry['median_var_ratio']:.3f}")

    if frames:
        pd.concat(frames, ignore_index=True).to_csv(
            args.out / "audio_noise_probe_features.csv", index=False)
    (args.out / "audio_noise_probe.json").write_text(
        json.dumps(summaries, indent=2, default=str), encoding="utf-8")
    print(f"\noutputs written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
