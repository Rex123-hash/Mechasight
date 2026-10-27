"""Reproduction check: every frozen bundle re-scores its source validation exactly.

Validation-only.  No common-test data is touched.  Also run by the protected
score-all stage before the ledger is opened.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment.phase4 import (  # noqa: E402
    ModelStore,
    load_frozen_bundles,
    verify_bundle_reproduces_validation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretest-root", type=Path, default=Path("reports/phase4_pretest"))
    parser.add_argument("--model-root", type=Path, default=Path("data/interim/phase4/models"))
    parser.add_argument("--feature-root", type=Path, default=Path("data/interim/phase31_primary"))
    parser.add_argument("--spectral-root", type=Path, default=Path("data/interim/phase4/spectral"))
    parser.add_argument(
        "--split-manifest", type=Path,
        default=Path("reports/phase2_2/protocol_b_few_shot_adaptation_v3.csv"),
    )
    args = parser.parse_args()
    bundles = load_frozen_bundles(args.pretest_root / "experiments")
    store = ModelStore(args.model_root)
    results = []
    for experiment_id, bundle in sorted(bundles.items()):
        config = Path(f"configs/preprocessing/mechasight_few_shot_regime_{bundle.config.regime.lower()}.json")
        results.append(verify_bundle_reproduces_validation(
            bundle, model_store=store, feature_root=args.feature_root,
            spectral_cache_root=args.spectral_root, split_manifest_path=args.split_manifest,
            preprocessing_config_path=config,
        ))
    summary = {
        "bundles_verified": len(results),
        "max_abs_score_difference": max(r["max_abs_score_difference"] for r in results) if results else None,
        "families": sorted({r["family"] for r in results}),
        "test_evaluated": False,
    }
    (args.pretest_root / "bundle_reproduction_check.json").write_text(
        json.dumps({"summary": summary, "results": results}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
