"""Fail-closed persistence for Phase-3.1 validation-only experiment bundles."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import pandas as pd

from src.experiment.config import ExperimentConfig
from src.experiment.constants import (
    PHASE31_EVALUATION_STAGE,
    PHASE31_RESULT_SCHEMA_VERSION,
    SCORE_DIRECTION,
)
from src.experiment.provenance import canonical_json, file_sha256, ordered_id_hash
from src.experiment.scores import ThresholdSelection


class PretestResultStore:
    """Write one immutable, explicitly non-test bundle per experiment."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def write_bundle(
        self,
        *,
        config: ExperimentConfig,
        provenance: Mapping[str, Any],
        summary: Mapping[str, Any],
        threshold: ThresholdSelection,
        validation_scores: pd.DataFrame,
    ) -> Path:
        required = {
            "evaluation_stage", "test_evaluated", "score_direction",
            "training_cohort_hash", "validation_cohort_hash",
        }
        missing = required - set(provenance)
        if missing:
            raise ValueError(f"pre-test provenance missing fields: {sorted(missing)}")
        if provenance["evaluation_stage"] != PHASE31_EVALUATION_STAGE:
            raise ValueError("Phase-3.1 results must be validation_only_pretest")
        if provenance["test_evaluated"] is not False:
            raise ValueError("Phase-3.1 result store refuses test-evaluated data")
        if provenance["score_direction"] != SCORE_DIRECTION:
            raise ValueError("pre-test result score direction is incompatible")
        forbidden_summary_keys = {
            "auroc", "average_precision", "auprc", "pauroc", "f1",
            "test_auroc", "test_average_precision", "test_scores",
        }
        if forbidden_summary_keys & {str(key).lower() for key in summary}:
            raise ValueError("Phase-3.1 summary cannot contain test/performance metrics")
        required_scores = {
            "segment_id", "parent_recording", "cohort_role", "domain",
            "anomaly_label", "score",
        }
        if not required_scores <= set(validation_scores.columns):
            raise ValueError("validation score table lacks required provenance columns")
        if validation_scores.empty or validation_scores["segment_id"].duplicated().any():
            raise ValueError("validation score table must contain unique segment results")
        if set(validation_scores["cohort_role"].astype(str)) != {"source_validation"}:
            raise ValueError("pre-test storage accepts source validation scores only")
        if set(validation_scores["domain"].astype(str)) != {"source"}:
            raise ValueError("pre-test storage accepts source-domain scores only")
        if not validation_scores["anomaly_label"].astype(str).str.lower().eq("normal").all():
            raise ValueError("pre-test storage accepts normal validation scores only")
        observed_hash = ordered_id_hash(tuple(validation_scores["segment_id"].astype(str)))
        if observed_hash != provenance["validation_cohort_hash"]:
            raise ValueError("validation identities do not match provenance")
        if threshold.source_cohort_hash != observed_hash:
            raise ValueError("threshold does not belong to the stored validation cohort")

        self.root.mkdir(parents=True, exist_ok=True)
        final = self.root / config.experiment_id
        if final.exists():
            raise FileExistsError(f"pre-test experiment already exists: {final}")
        stage = Path(tempfile.mkdtemp(prefix=f".{config.experiment_id}.", dir=self.root))
        try:
            config_path = stage / "config.json"
            provenance_path = stage / "provenance.json"
            summary_path = stage / "validation_summary.json"
            threshold_path = stage / "threshold.json"
            scores_path = stage / "validation_scores.csv"
            config_path.write_text(canonical_json(config.to_dict()) + "\n", encoding="utf-8")
            provenance_payload = {
                **dict(provenance),
                "result_schema_version": PHASE31_RESULT_SCHEMA_VERSION,
                "experiment_id": config.experiment_id,
                "scientific_limitations": list(config.scientific_limitations),
                "scientific_limitation_ids": list(config.scientific_limitation_ids),
                "cohort_provenance": config.cohort_provenance,
            }
            provenance_path.write_text(
                canonical_json(provenance_payload) + "\n", encoding="utf-8"
            )
            summary_path.write_text(
                canonical_json({
                    "result_schema_version": PHASE31_RESULT_SCHEMA_VERSION,
                    **dict(summary),
                }) + "\n",
                encoding="utf-8",
            )
            threshold_path.write_text(
                canonical_json(asdict(threshold)) + "\n", encoding="utf-8"
            )
            validation_scores.to_csv(scores_path, index=False, lineterminator="\n")
            payload_files = (
                config_path, provenance_path, summary_path, threshold_path, scores_path,
            )
            (stage / "bundle_manifest.json").write_text(
                canonical_json({
                    "result_schema_version": PHASE31_RESULT_SCHEMA_VERSION,
                    "experiment_id": config.experiment_id,
                    "files": {path.name: file_sha256(path) for path in payload_files},
                }) + "\n",
                encoding="utf-8",
            )
            if final.exists():
                raise FileExistsError(f"pre-test experiment already exists: {final}")
            stage.rename(final)
            return final
        except Exception:
            if stage.exists():
                shutil.rmtree(stage)
            raise
