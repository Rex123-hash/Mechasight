"""Build the Single-Modality Benchmarking training/validation spectral caches.

Training and source-validation roles only.  The common-evaluation cache is
built exclusively inside the protected score-all stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.spectral_cache import build_pretest_spectral_cache  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/extracted"))
    parser.add_argument(
        "--split-manifest", type=Path,
        default=Path("reports/phase2_2/protocol_b_few_shot_adaptation_v3.csv"),
    )
    parser.add_argument("--out", type=Path, default=Path("data/interim/phase4/spectral"))
    parser.add_argument("--machines", nargs="+", default=["RoboticArm", "BrushlessMotor"])
    parser.add_argument("--regimes", nargs="+", default=["S", "T"])
    args = parser.parse_args()

    outputs = []
    for regime in args.regimes:
        config = Path(f"configs/preprocessing/mechasight_few_shot_regime_{regime.lower()}.json")
        for machine in args.machines:
            for role in ("training", "source_validation"):
                started = perf_counter()
                metadata = build_pretest_spectral_cache(
                    data_root=args.data_root,
                    split_manifest_path=args.split_manifest,
                    preprocessing_config_path=config,
                    cache_root=args.out,
                    machine=machine,
                    role=role,
                )
                metadata["build_seconds"] = round(perf_counter() - started, 1)
                outputs.append(metadata)
                print(json.dumps(metadata, sort_keys=True), flush=True)
    print(json.dumps({"caches_built": len(outputs), "test_evaluated": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
