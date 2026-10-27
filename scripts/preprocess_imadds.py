"""Build a configuration-bound streaming cache and traceability manifest.

Examples (Phase-2 smoke build; remove --max-segments for the complete split):

    python scripts/preprocess_imadds.py --config configs/preprocessing/mechasight_regime_t.json \
      --machine RoboticArm --split train --max-segments 1
    python scripts/preprocess_imadds.py --config configs/preprocessing/official_baseline_reproduction.json \
      --machine RoboticArm --split train --max-segments 1
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.cache import cache_path  # noqa: E402
from src.preprocessing.config import OFFICIAL_PROTOCOL, load_config  # noqa: E402
from src.preprocessing.pipeline import (  # noqa: E402
    build_cache,
    load_raw_catalog,
    merge_protocol_b_assignments,
    official_assignments,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/extracted"))
    parser.add_argument(
        "--split-manifest", type=Path,
        default=Path("reports/phase2_1/protocol_b_common_test_v2.csv"),
    )
    parser.add_argument("--out", type=Path, default=Path("data/interim/phase2"))
    parser.add_argument("--machine", choices=("RoboticArm", "BrushlessMotor"), required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument(
        "--data-pool", default=None,
        help="restrict the build to one primary protocol role, e.g. target_adaptation",
    )
    parser.add_argument("--max-segments", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    catalog = load_raw_catalog(args.data_root)
    if config.protocol == OFFICIAL_PROTOCOL:
        catalog = official_assignments(catalog, config.regime)
    else:
        assignments = pd.read_csv(args.split_manifest)
        catalog = merge_protocol_b_assignments(
            catalog, assignments, config.regime, split_version=config.split_version
        )

    if args.data_pool is not None:
        if "data_pool" not in catalog:
            raise ValueError("split manifest carries no data_pool role column")
        catalog = catalog[catalog["data_pool"].eq(args.data_pool)]
        if catalog.empty:
            raise ValueError(f"no segments carry the role {args.data_pool!r}")

    h5_path = cache_path(args.out / "cache", config, machine=args.machine, split=args.split)
    base = h5_path.with_suffix("")
    manifest_path = base.parent / f"{base.name}__manifest.jsonl"
    feature_path = base.parent / f"{base.name}__features.jsonl"
    stats_path = base.parent / f"{base.name}__build_stats.json"
    conflicts = [path for path in (h5_path, manifest_path, feature_path, stats_path) if path.exists()]
    if conflicts:
        raise FileExistsError(
            "refusing to overwrite an existing cache build: " + ", ".join(map(str, conflicts))
        )

    counts = build_cache(
        catalog, config, machine=args.machine, split=args.split,
        cache_path=h5_path, manifest_path=manifest_path, feature_path=feature_path,
        max_segments=args.max_segments,
    )
    stats = {
        **asdict(counts),
        "machine": args.machine,
        "split": args.split,
        "data_pool_filter": args.data_pool,
        "protocol": config.protocol,
        "regime": config.regime,
        "preprocessing_config_id": config.config_id,
        "max_segments": args.max_segments,
        "cache_path": str(h5_path),
        "manifest_path": str(manifest_path),
        "feature_path": str(feature_path),
        "cache_bytes": h5_path.stat().st_size,
        "manifest_bytes": manifest_path.stat().st_size,
        "feature_bytes": feature_path.stat().st_size,
    }
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
