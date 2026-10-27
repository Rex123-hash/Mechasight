"""Protected evaluation for Single-Modality Model Benchmarking.

Two explicit sub-commands, each requiring a written authorisation token:

    score-all     verify all 156 frozen bundles and model artifacts, build the
                  common-evaluation spectral cache, append the 156 looks to the
                  ledger atomically, then score every bundle into a sealed
                  staging area.  No anomaly label is read; no metric exists.
    analyse-once  refuse unless exactly 156 hashed score tables verify; then
                  compute every metric and contrast in one pass.  Final.

This script is committed for review.  It is NOT to be executed until the
protected batch is explicitly authorised.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment.ledger import TestLookLedger  # noqa: E402
from src.experiment.phase32 import (  # noqa: E402
    CommonTestCohortContract,
    common_test_artifact_paths,
)
from src.experiment.phase4 import (  # noqa: E402
    FAMILIES,
    LEDGER_EXPECTED_START,
    MACHINES,
    MODALITIES,
    REGIMES,
    ModelStore,
    ScoreStaging,
    authorize_protected_batch,
    expected_seeds,
    load_frozen_bundles,
    load_label_free_test_features,
    load_label_free_test_windows,
    plan_protected_batch,
    score_all,
    verify_bundle_reproduces_validation,
)
from src.experiment.phase4_analysis import LABEL_COLUMNS, analyse_once  # noqa: E402
from src.experiment.provenance import canonical_json, file_sha256  # noqa: E402
from src.preprocessing.config import load_config  # noqa: E402
from src.preprocessing.protected_access import authorized_protected_read  # noqa: E402
from src.preprocessing.spectral_cache import (  # noqa: E402
    build_protected_spectral_cache,
    spectral_cache_path,
)


LEDGER_PATH = Path("reports/experiment_results/test_look_ledger.jsonl")
AUTHORIZATION_TOKEN = "I_AUTHORIZE_THE_156_LOOK_PROTECTED_BATCH"


def _head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def _require_token(args: argparse.Namespace) -> None:
    if args.authorization != AUTHORIZATION_TOKEN:
        raise SystemExit("refusing: protected evaluation requires the explicit authorisation token")
    if subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip():
        raise SystemExit("refusing: working tree must be clean so the code SHA is the evaluated code")


def _contracts(split_manifest: Path) -> dict[tuple[str, str], CommonTestCohortContract]:
    assignments = pd.read_csv(split_manifest)
    contracts = {}
    for regime in REGIMES:
        config = load_config(f"configs/preprocessing/mechasight_few_shot_regime_{regime.lower()}.json")
        for machine in MACHINES:
            contracts[(machine, regime)] = CommonTestCohortContract.from_frozen_assignments(
                assignments, config, machine=machine
            )
    return contracts


def score_all_command(args: argparse.Namespace) -> int:
    _require_token(args)
    head = _head()
    proposed = json.loads((args.pretest_root / "proposed_protected_batch.json").read_text(encoding="utf-8"))
    bundles = load_frozen_bundles(args.pretest_root / "experiments")
    model_store = ModelStore(args.model_root)
    for experiment_id, bundle in bundles.items():
        model_store.verify(experiment_id, expected_manifest_sha256=bundle.model_manifest_sha256)
        verify_bundle_reproduces_validation(
            bundle, model_store=model_store, feature_root=args.feature_root,
            spectral_cache_root=args.spectral_root, split_manifest_path=args.split_manifest,
            preprocessing_config_path=Path(
                f"configs/preprocessing/mechasight_few_shot_regime_{bundle.config.regime.lower()}.json"
            ),
        )
    contracts = _contracts(args.split_manifest)
    planned = plan_protected_batch(
        bundles,
        test_cohort_hashes_by_machine=proposed["test_cohort_hashes_by_machine"],
        code_git_sha=head,
    )
    # The proposed identity was computed at the fit HEAD; the authorised identity
    # binds the scoring HEAD.  Experiment ids and cohort hashes must be identical.
    if list(planned.experiment_ids) != proposed["experiment_ids"]:
        raise SystemExit("refusing: frozen bundles differ from the proposed 156 experiment ids")
    if proposed.get("status") != "PROPOSED_NOT_AUTHORIZED":
        raise SystemExit("refusing: proposed batch file is not in the expected state")
    staging = ScoreStaging(args.out / "staging")
    if staging.manifest_path.exists():
        raise SystemExit("refusing: a staging area already exists; recovery needs a new versioned identity")

    # Materialise label-free test inputs BEFORE the ledger append; nothing here
    # reads an anomaly label into the scoring path.
    test_features = {}
    test_windows = {}
    with authorized_protected_read(
        authority=planned.batch_id,
        purpose="score-all: label-free common-test inputs",
    ):
        for regime in REGIMES:
            config_path = Path(f"configs/preprocessing/mechasight_few_shot_regime_{regime.lower()}.json")
            for machine in MACHINES:
                contract = contracts[(machine, regime)]
                test_features[(machine, regime)] = load_label_free_test_features(
                    args.common_test_root, machine=machine, regime=regime, contract=contract
                )
                cache = spectral_cache_path(args.spectral_root, machine=machine, regime=regime, role="common_evaluation")
                if not cache.exists():
                    build_protected_spectral_cache(
                        data_root=args.data_root, split_manifest_path=args.split_manifest,
                        preprocessing_config_path=config_path, cache_root=args.spectral_root,
                        machine=machine,
                    )
                test_windows[(machine, regime)] = load_label_free_test_windows(
                    args.spectral_root, machine=machine, regime=regime, contract=contract
                )
                for loaded in (test_features[(machine, regime)], test_windows[(machine, regime)]):
                    if loaded.cohort_hash != planned.test_cohort_hashes_by_machine[machine]:
                        raise SystemExit("refusing: materialised test cohort differs from the frozen cohort hash")

    ledger = TestLookLedger(LEDGER_PATH)
    authorization = authorize_protected_batch(
        ledger=ledger, planned=planned, bundles=bundles, expected_ledger_start=LEDGER_EXPECTED_START
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "batch_authorization.json").write_text(
        canonical_json({
            "batch_id": authorization.batch_id,
            "ledger_start_count": authorization.ledger_start_count,
            "ledger_end_count": authorization.ledger_end_count,
            "code_git_sha": head,
            "experiment_count": len(authorization.experiment_ids),
        }) + "\n", encoding="utf-8",
    )
    staging.open(authorization)
    manifest = score_all(
        bundles=bundles, authorization=authorization, model_store=model_store,
        test_features=test_features, test_windows=test_windows, staging=staging,
        progress=lambda item: print(canonical_json(item), flush=True),
    )
    print(canonical_json({
        "status": "scored_all_no_metrics", "batch_id": manifest["batch_id"],
        "tables": len(manifest["tables"]), "ledger_count": len(ledger.read()),
    }))
    return 0


def analyse_once_command(args: argparse.Namespace) -> int:
    _require_token(args)
    bundles = load_frozen_bundles(args.pretest_root / "experiments")
    staging = ScoreStaging(args.out / "staging")
    labels = {}
    authority = json.loads((args.out / "batch_authorization.json").read_text(encoding="utf-8"))["batch_id"]
    with authorized_protected_read(authority=authority, purpose="analyse-once: common-test labels and sealed scores"):
        for machine in MACHINES:
            artifact, metadata_path = common_test_artifact_paths(args.common_test_root, machine=machine, regime="S")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if file_sha256(artifact) != metadata["artifact_sha256"]:
                raise SystemExit("refusing: frozen common-test label artifact hash mismatch")
            labels[machine] = pd.read_parquet(artifact).loc[:, list(LABEL_COLUMNS)]
        result = analyse_once(
            staging=staging, bundles=bundles, labels_by_machine=labels, out_dir=args.out / "analysis",
            families=FAMILIES, seeds_for=expected_seeds, modalities=MODALITIES,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", default="")
    parser.add_argument("--pretest-root", type=Path, default=Path("reports/phase4_pretest"))
    parser.add_argument("--out", type=Path, default=Path("reports/phase4_protected"))
    parser.add_argument("--model-root", type=Path, default=Path("data/interim/phase4/models"))
    parser.add_argument("--feature-root", type=Path, default=Path("data/interim/phase31_primary"))
    parser.add_argument("--spectral-root", type=Path, default=Path("data/interim/phase4/spectral"))
    parser.add_argument("--common-test-root", type=Path, default=Path("data/interim/phase32_common_test"))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/extracted"))
    parser.add_argument(
        "--split-manifest", type=Path,
        default=Path("reports/phase2_2/protocol_b_few_shot_adaptation_v3.csv"),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("score-all").set_defaults(func=score_all_command)
    sub.add_parser("analyse-once").set_defaults(func=analyse_once_command)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
