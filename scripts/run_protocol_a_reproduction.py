"""Run the Protocol-A official-baseline reproduction.

Stages, run separately so that the pre-test freeze can be committed before any
official-test score exists:

    python scripts/run_protocol_a_reproduction.py preprocess --machine RoboticArm
    python scripts/run_protocol_a_reproduction.py train      --machine RoboticArm
    python scripts/run_protocol_a_reproduction.py evaluate   --machine RoboticArm

``evaluate`` is the only stage that touches official-test scores, and it refuses
to run unless a Protocol-A test-look ledger entry already exists for the
configuration.

Faithful to the upstream notebooks at commit
592bcbbed58893dc26bbab3240c5d7a3105a1a7f. The only intentional departures are
performance-level, never behavioural: HDF5 rows are written in buffered blocks
rather than one resize per window, and each skipped segment is recorded instead
of only printed. Both leave the stored arrays identical.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reproduction.protocol_a import (  # noqa: E402
    LABEL_NAMES,
    OFFICIAL_PARAMS,
    OFFICIAL_SENSOR_SPEC,
    SENSORS,
    TrainingOutcome,
    build_official_autoencoder,
    group_by_segment_id_median,
    official_segment_bounds,
    official_single_auc,
    official_window,
    official_window_length,
    read_official_sensor_frame,
    sensor_slices,
    standardize_window,
)
from src.experiment.provenance import content_sha256, file_sha256  # noqa: E402

CONFIG_PATH = Path("configs/phase3/protocol_a_official_reproduction_v1.json")
DATA_ROOT = Path("data/raw/extracted")
WORK_ROOT = Path("data/interim/protocol_a")
OUT_ROOT = Path("reports/phase3_3_protocol_a")
LEDGER = Path("reports/experiment_results/test_look_ledger.jsonl")

TRAIN_CSVS = ("attributes_normal_source_train.csv", "attributes_normal_target_train.csv")
TEST_CSVS = ("attributes_normal_source_test.csv", "attributes_anomaly_source_test.csv",
             "attributes_normal_target_test.csv", "attributes_anomaly_target_test.csv")

BUFFER_WINDOWS = 512


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def machine_root(machine: str) -> Path:
    return DATA_ROOT / machine / machine


def read_metadata(machine: str, split: str) -> pd.DataFrame:
    root = machine_root(machine) / split
    names = TRAIN_CSVS if split == "train" else TEST_CSVS
    frames = [pd.read_csv(root / n, index_col=0) for n in names]
    meta = pd.concat(frames, axis=0).reset_index(drop=True)
    meta["segment_id"] = meta["imp23absu_mic"].apply(
        lambda x: x.replace("imp23absu_mic_", "").replace(".parquet", ""))
    for sensor in SENSORS:
        meta[sensor] = meta[sensor].apply(lambda p, r=root: str(r / p))
    return meta


def windowed_path(machine: str, split: str) -> Path:
    return WORK_ROOT / machine / f"{split}_dataset_window_0.100s.h5"


# ------------------------------------------------------------- preprocess
def stage_preprocess(machine: str) -> dict:
    summary = {}
    for split in ("train", "test"):
        path = windowed_path(machine, split)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            with h5py.File(path, "r") as f:
                summary[split] = {"windows": int(f[SENSORS[0]].shape[0]),
                                  "skipped": [], "reused_existing": True}
            print(f"  {machine}/{split}: existing artifact reused "
                  f"({summary[split]['windows']} windows)")
            continue

        meta = read_metadata(machine, split)
        started = perf_counter()
        skipped: list[dict] = []
        buffers = {s: [] for s in SENSORS}
        label_buffer: list[list[str]] = []
        written = 0

        with h5py.File(path, "w") as h5:
            datasets = {}
            for sensor in SENSORS:
                datasets[sensor] = h5.create_dataset(
                    sensor,
                    shape=(0, OFFICIAL_SENSOR_SPEC[sensor]["number_of_channel"],
                           official_window_length(sensor)),
                    maxshape=(None, OFFICIAL_SENSOR_SPEC[sensor]["number_of_channel"],
                              official_window_length(sensor)),
                    chunks=True)
            for label in LABEL_NAMES:
                datasets[label] = h5.create_dataset(
                    label, shape=(0, 1), maxshape=(None, 1), chunks=True,
                    dtype=h5py.string_dtype(encoding="utf-8"))

            def flush() -> None:
                nonlocal written
                n = len(label_buffer)
                if n == 0:
                    return
                for sensor in SENSORS:
                    d = datasets[sensor]
                    d.resize(written + n, axis=0)
                    d[written:written + n] = np.stack(buffers[sensor])
                    buffers[sensor].clear()
                arr = np.array(label_buffer, dtype=object)
                for i, label in enumerate(LABEL_NAMES):
                    d = datasets[label]
                    d.resize(written + n, axis=0)
                    d[written:written + n] = arr[:, i].reshape(n, 1)
                label_buffer.clear()
                written += n

            for index in range(len(meta)):
                try:
                    frames = {s: read_official_sensor_frame(meta[s][index]) for s in SENSORS}
                    start, _end, n_windows = official_segment_bounds(frames)
                    labels = [str(meta[name][index]) for name in LABEL_NAMES]
                    for w in range(n_windows):
                        w_start = start + w * pd.to_timedelta("100ms")
                        for sensor in SENSORS:
                            buffers[sensor].append(
                                official_window(frames[sensor], w_start, sensor))
                        label_buffer.append(labels)
                        if len(label_buffer) >= BUFFER_WINDOWS:
                            flush()
                except Exception as exc:                       # upstream: print & continue
                    skipped.append({"file_index": index, "error": f"{type(exc).__name__}: {exc}"})
                    print(f"could not read file index {index} {exc}")
                if index % 200 == 0:
                    print(f"  {machine}/{split}: {index}/{len(meta)} segments, "
                          f"{written + len(label_buffer)} windows", end="\r")
            flush()

            for sensor in SENSORS:
                assert datasets[sensor].shape[0] == written
        elapsed = perf_counter() - started
        summary[split] = {
            "segments": int(len(meta)), "windows": int(written),
            "skipped": skipped, "seconds": round(elapsed, 1),
            "sha256": file_sha256(path), "bytes": path.stat().st_size,
            "reused_existing": False,
        }
        print(f"\n  {machine}/{split}: {written} windows from {len(meta)} segments "
              f"in {elapsed:.0f}s ({len(skipped)} skipped)")
    return summary


# ---------------------------------------------------------------- loading
def load_windowed(machine: str, split: str):
    with h5py.File(windowed_path(machine, split), "r") as f:
        X = [f[s][:] for s in SENSORS]
        Y = pd.DataFrame(
            [[v.decode("utf-8") for v in f[name][:].flatten()] for name in LABEL_NAMES]).T
        Y.columns = list(LABEL_NAMES)
    return X, Y


# ------------------------------------------------------------------ train
def stage_train(machine: str, device: str) -> dict:
    import torch
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Dataset

    params = dict(OFFICIAL_PARAMS)
    torch.manual_seed(params["seed"])
    if device == "cuda":
        torch.cuda.manual_seed(params["seed"])

    # Labels first, so the split indices exist before any window array is held.
    path = windowed_path(machine, "train")
    with h5py.File(path, "r") as f:
        Y_train_raw = pd.DataFrame(
            [[v.decode("utf-8") for v in f[name][:].flatten()] for name in LABEL_NAMES]).T
        Y_train_raw.columns = list(LABEL_NAMES)
    Y_train_raw["combined_label"] = (Y_train_raw["anomaly_label"]
                                     + Y_train_raw["domain_shift_op"]
                                     + Y_train_raw["domain_shift_env"])

    train_idx, valid_idx, _, _ = train_test_split(
        range(len(Y_train_raw)), Y_train_raw,
        stratify=Y_train_raw["combined_label"],
        test_size=params["valid_size"], random_state=params["seed"])
    train_idx = np.asarray(train_idx)
    valid_idx = np.asarray(valid_idx)

    # Stream one sensor at a time: identical arrays, roughly half the peak memory
    # of loading every sensor and then fancy-indexing.
    X_train, X_valid = [], []
    with h5py.File(path, "r") as f:
        for sensor in SENSORS:
            raw = f[sensor][:]
            X_train.append(standardize_window(raw[train_idx]))
            X_valid.append(standardize_window(raw[valid_idx]))
            del raw

    num_channels = [x.shape[1] for x in X_train]
    window_lengths = [x.shape[2] for x in X_train]

    class _DS(Dataset):
        def __init__(self, X): self.X = X
        def __len__(self): return len(self.X[0])
        def __getitem__(self, i): return [x[i] for x in self.X]

    to_t = lambda xs: [torch.from_numpy(np.ascontiguousarray(x)).to(device) for x in xs]
    train_loader = DataLoader(_DS(to_t(X_train)), batch_size=params["batch_size"], shuffle=True)
    valid_loader = DataLoader(_DS(to_t(X_valid)), batch_size=params["batch_size"], shuffle=False)

    model = build_official_autoencoder(window_lengths, num_channels, params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

    def mse(a, b):
        return torch.mean((a - b) ** 2, dim=1)

    def evaluate(loader) -> float:
        model.eval()
        total, batches = 0.0, 0
        with torch.no_grad():
            for xb in loader:
                xb = torch.concat([x.flatten(1) for x in xb], axis=1)
                _, est = model(xb)
                total += torch.mean(mse(xb, est)).item()
                batches += 1
        return total / batches

    training_losses, valid_losses = [], []
    best_valid, best_epoch, patience_counter = float("inf"), -1, 0
    stopped_early = False
    started = perf_counter()

    for epoch in range(params["num_epochs"]):
        model.train()
        epoch_loss, batches = 0.0, 0
        for xb in train_loader:
            xb = torch.concat([x.flatten(1) for x in xb], axis=1)
            optimizer.zero_grad()
            _, est = model(xb)
            loss = torch.mean(mse(xb, est))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1
        training_losses.append(epoch_loss / batches)
        v = evaluate(valid_loader)
        valid_losses.append(v)
        print(f"  epoch {epoch + 1:3d} | train {training_losses[-1]:.6f} | valid {v:.6f}")

        # Upstream ordering: increment first, then compare, then check patience.
        patience_counter += 1
        if v < best_valid:
            best_valid, best_epoch, patience_counter = v, epoch, 0
        if patience_counter == params["patience"]:
            stopped_early = True
            print(f"  early stop: no improvement for {params['patience']} epochs")
            break
    # Upstream restores best_model_state, which aliases the live tensors, so the
    # weights in memory are already the final epoch's. Reproduced by doing nothing.

    outcome = TrainingOutcome(
        epochs_run=len(training_losses), stopped_early=stopped_early,
        best_valid_loss=float(best_valid), best_epoch=int(best_epoch),
        training_losses=tuple(map(float, training_losses)),
        valid_losses=tuple(map(float, valid_losses)),
        extra={"seconds": round(perf_counter() - started, 1), "device": device,
               "train_windows": int(len(train_idx)), "valid_windows": int(len(valid_idx)),
               "input_dim": int(model.input_dim),
               "parameters": int(sum(p.numel() for p in model.parameters()))})

    ckpt_dir = WORK_ROOT / machine
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / f"protocol_a_baseline_seed{params['seed']}.pth"
    torch.save(model.state_dict(), ckpt)

    (OUT_ROOT / "training").mkdir(parents=True, exist_ok=True)
    record = {"machine": machine, "checkpoint": str(ckpt).replace("\\", "/"),
              "checkpoint_sha256": file_sha256(ckpt),
              "torch_version": torch.__version__, **asdict(outcome)}
    (OUT_ROOT / "training" / f"{machine}_training.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")
    print(f"  saved {ckpt}")
    return record


# --------------------------------------------------------------- evaluate
def _ledger_has_protocol_a(machine: str, config_identity: str) -> bool:
    if not LEDGER.exists():
        return False
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if (e.get("protocol") == "official_baseline_reproduction"
                and e.get("machine") == machine
                and e.get("model_config_identity") == config_identity):
            return True
    return False


def stage_evaluate(machine: str, device: str) -> dict:
    import torch

    cfg = load_config()
    identity = cfg["config_identity"]
    if not _ledger_has_protocol_a(machine, identity):
        raise SystemExit(
            f"refusing to score the official test set for {machine}: no Protocol-A "
            f"test-look ledger entry exists for {identity}")

    params = dict(OFFICIAL_PARAMS)
    X_test, Y_test = load_windowed(machine, "test")
    Y_test["combined_label"] = (Y_test["anomaly_label"] + Y_test["domain_shift_op"]
                                + Y_test["domain_shift_env"])
    X_test = [standardize_window(x) for x in X_test]
    num_channels = [x.shape[1] for x in X_test]
    window_lengths = [x.shape[2] for x in X_test]

    model = build_official_autoencoder(window_lengths, num_channels, params).to(device)
    ckpt = WORK_ROOT / machine / f"protocol_a_baseline_seed{params['seed']}.pth"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    bounds = sensor_slices(window_lengths, num_channels)
    tensors = [torch.from_numpy(np.ascontiguousarray(x)) for x in X_test]
    n = len(tensors[0])
    batch = params["batch_size"]

    total_loss, fused, masked = [], {s: [] for s in SENSORS}, {s: [] for s in SENSORS}
    with torch.no_grad():
        for i in range(0, n, batch):
            xb = torch.concat([t[i:i + batch].flatten(1) for t in tensors], axis=1).to(device)
            _, est = model(xb)
            total_loss.append(torch.mean((xb - est) ** 2, dim=1).cpu().numpy())
            for (lo, hi), sensor in zip(bounds, SENSORS, strict=True):
                fused[sensor].append(
                    torch.mean((xb[:, lo:hi] - est[:, lo:hi]) ** 2, dim=1).cpu().numpy())
            # upstream get_masked_batch: zero every other sensor block, re-run
            for (lo, hi), sensor in zip(bounds, SENSORS, strict=True):
                masked_in = torch.zeros_like(xb)
                masked_in[:, lo:hi] = xb[:, lo:hi]
                _, masked_est = model(masked_in)
                masked[sensor].append(
                    torch.mean((xb[:, lo:hi] - masked_est[:, lo:hi]) ** 2, dim=1).cpu().numpy())

    scores = pd.DataFrame({"total_loss": np.concatenate(total_loss)})
    for sensor in SENSORS:
        scores[f"f_{sensor}"] = np.concatenate(fused[sensor])
        scores[f"s_{sensor}"] = np.concatenate(masked[sensor])
    combined = pd.concat([Y_test, scores], axis=1)
    grouped = group_by_segment_id_median(combined, scores.columns)

    results = {col: official_single_auc(grouped, col) for col in scores.columns}
    return {"machine": machine, "config_identity": identity,
            "segments_scored": int(len(grouped)), "windows_scored": int(n),
            "auc": results,
            "score_columns": list(scores.columns),
            "checkpoint_sha256": file_sha256(ckpt)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preprocess", "train", "evaluate"))
    parser.add_argument("--machine", required=True,
                        choices=("RoboticArm", "BrushlessMotor"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device
    if device is None and args.stage in ("train", "evaluate"):
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.stage == "preprocess":
        out = stage_preprocess(args.machine)
        (OUT_ROOT / f"preprocess_{args.machine}.json").write_text(
            json.dumps({"machine": args.machine,
                        "generated_utc": datetime.now(timezone.utc).isoformat(),
                        "splits": out}, indent=2), encoding="utf-8")
    elif args.stage == "train":
        out = stage_train(args.machine, device)
    else:
        out = stage_evaluate(args.machine, device)
        (OUT_ROOT / f"evaluate_{args.machine}.json").write_text(
            json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("training_losses", "valid_losses")}
                     if isinstance(out, dict) else out, indent=2, default=str)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
