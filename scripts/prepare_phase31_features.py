"""Prepare only Phase-3.1 training and source-normal validation features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment.phase31 import prepare_feature_artifact  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/extracted"))
    parser.add_argument(
        "--split-manifest", type=Path,
        default=Path("reports/phase2_2/protocol_b_few_shot_adaptation_v3.csv"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/interim/phase31_primary")
    )
    args = parser.parse_args()

    outputs = []
    for regime in ("S", "T"):
        config = Path(f"configs/preprocessing/mechasight_few_shot_regime_{regime.lower()}.json")
        for machine in ("RoboticArm", "BrushlessMotor"):
            for role in ("training", "source_validation"):
                metadata = prepare_feature_artifact(
                    data_root=args.data_root,
                    split_manifest_path=args.split_manifest,
                    preprocessing_config_path=config,
                    artifact_root=args.out,
                    machine=machine,
                    role=role,
                )
                outputs.append(metadata)
                print(json.dumps(metadata, sort_keys=True), flush=True)
    print(json.dumps({"artifacts_built": len(outputs), "test_evaluated": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
